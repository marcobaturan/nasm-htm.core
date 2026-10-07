; ==============================================================================
; PROJECT: nasm-htm.core
; MODULE:  src/sp_core.asm
; AUTHOR:  Antigravity AI Team
; LICENSE: MIT / Apache-2.0
; ------------------------------------------------------------------------------
; DESCRIPTION:
; Ultra-high-performance x86_64 NASM assembly core for HTM (Hierarchical
; Temporal Memory) Spatial Pooler overlap calculation.
;
; HARDWARE ACCELERATION:
; Optimized for AVX2 (256-bit SIMD YMM registers), SSSE3 (PSHUFB), and x86_64
; scalar POPCNT. Specifically engineered for processors such as AMD Ryzen 4000
; (Zen 2 architecture) and Intel Haswell/Skylake CPUs that support AVX2 but
; lack hardware AVX-512 VPOPCNTDQ (vpopcntd) vector instructions.
;
; ALGORITHM DETAILS:
; Computes column overlap scores: overlap(c) = popcount(input_sdr AND syn_mask_c)
; Uses a 256-bit vector nibble lookup popcount algorithm (PSHUFB SIMD shuffle):
;   1. Vector Bitwise AND: YMM0 = input_sdr[i..i+31] AND syn_mask_c[i..i+31]
;   2. Split bytes into low (0..3) and high (4..7) 4-bit nibbles.
;   3. Parallel nibble popcount lookup via VPSHUFB using 16-entry LUT.
;   4. Sum nibbles per byte via VPADDB (max count per byte = 8 <= 255).
;   5. Horizontal byte accumulation into 64-bit quadwords via VPSADBW against zero.
;   6. Quadword vector accumulation via VPADDQ across 32-byte chunks.
;   7. Final horizontal reduction across 4 quadwords into 32-bit overlap result.
;   8. Tail bytes (< 32 bytes remaining) are processed with scalar POPCNT.
; ==============================================================================

global calculate_overlap

section .rodata
align 32
; ------------------------------------------------------------------------------
; Popcount Nibble Lookup Table (LUT)
; Maps 4-bit values (0x0..0xF) to their number of set bits (0..4).
; Duplicated across two 16-byte lanes to fill a full 256-bit YMM register.
; Index: 0 1 2 3 4 5 6 7 8 9 A B C D E F
; Count: 0 1 1 2 1 2 2 3 1 2 2 3 2 3 3 4
; ------------------------------------------------------------------------------
popcount_lut:
    db 0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4
    db 0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4

align 32
; 32-byte mask isolating lower 4 bits (0x0F) of each byte
mask_0f:
    times 32 db 0x0F

section .text

; ------------------------------------------------------------------------------
; C FUNCTION SIGNATURE (System V AMD64 ABI):
; void calculate_overlap(
;     const uint8_t* input_sdr,    ; RDI -> Pointer to 32-byte aligned input bit array
;     const uint8_t* syn_mask,     ; RSI -> Pointer to connected synapse matrix [num_columns * sdr_bytes]
;     uint32_t* overlaps,          ; RDX -> Pointer to output overlap array [num_columns]
;     uint64_t num_columns,        ; RCX -> Total number of columns (M)
;     uint64_t sdr_bytes           ; R8  -> Length of SDR bitmask in bytes (N / 8)
; )
; ------------------------------------------------------------------------------
calculate_overlap:
    ; Standard x86_64 Stack Frame Setup
    push rbp
    mov rbp, rsp
    
    ; Preserve Callee-Saved Registers per System V AMD64 ABI
    push rbx
    push r12
    push r13
    push r14
    push r15

    ; Save parameter registers into preserved registers for outer column loop
    mov r12, rdi            ; R12 = input_sdr pointer
    mov r13, rsi            ; R13 = syn_mask pointer base
    mov r14, rdx            ; R14 = overlaps output pointer base
    mov r15, rcx            ; R15 = num_columns total counter

    ; Pre-load SIMD constants into dedicated YMM registers
    vmovdqa ymm6, [rel popcount_lut]  ; YMM6 = 256-bit nibble popcount LUT
    vmovdqa ymm7, [rel mask_0f]       ; YMM7 = 256-bit 0x0F nibble bitmask

    xor rax, rax            ; RAX = column index counter (c = 0)

.col_loop:
    cmp rax, r15
    jge .done               ; If c >= num_columns, exit function

    ; Calculate memory address for current column synapse mask:
    ; RSI = syn_mask_base + (c * sdr_bytes)
    mov rsi, rax
    imul rsi, r8
    add rsi, r13

    ; Clear 256-bit SIMD quadword accumulator (YMM5 = [0, 0, 0, 0])
    vpxor ymm5, ymm5, ymm5

    xor rbx, rbx            ; RBX = byte offset within SDR (i = 0)

.simd_loop:
    mov r10, r8
    sub r10, rbx
    cmp r10, 32
    jl .scalar_tail         ; If remaining bytes < 32, jump to tail handler

    ; Load 32 bytes (256 bits) from input SDR and column synapse bitmask
    vmovdqu ymm0, [r12 + rbx]     ; YMM0 = 32 input bytes
    vmovdqu ymm1, [rsi + rbx]     ; YMM1 = 32 synapse mask bytes
    vpand ymm0, ymm0, ymm1        ; YMM0 = Active connected synapses (Bitwise AND)

    ; --- PSHUFB SIMD Vector Popcount Process ---
    
    ; Step A: Extract Low 4-bit Nibbles
    vpand ymm1, ymm0, ymm7        ; YMM1 = YMM0 & 0x0F
    vpshufb ymm1, ymm6, ymm1      ; YMM1 = Parallel lookup of low nibble popcounts

    ; Step B: Extract High 4-bit Nibbles
    vpsrlw ymm2, ymm0, 4          ; Shift 16-bit words right by 4 bits
    vpand ymm2, ymm2, ymm7        ; YMM2 = (YMM0 >> 4) & 0x0F
    vpshufb ymm2, ymm6, ymm2      ; YMM2 = Parallel lookup of high nibble popcounts

    ; Step C: Combine Nibble Counts per Byte
    vpaddb ymm1, ymm1, ymm2       ; YMM1[b] = Low_Popcount[b] + High_Popcount[b]

    ; Step D: Horizontal Sum Bytes to 64-bit Quadwords via VPSADBW
    vpxor ymm3, ymm3, ymm3
    vpsadbw ymm1, ymm1, ymm3      ; Sum 8-byte blocks into 16-bit fields zero-extended to 64-bit

    ; Step E: Accumulate Quadwords into Running Total
    vpaddq ymm5, ymm5, ymm1       ; YMM5 += YMM1

    add rbx, 32
    jmp .simd_loop

.scalar_tail:
    xor r9, r9              ; R9 = scalar tail overlap accumulator
    cmp rbx, r8
    jge .store_result

.tail_loop:
    movzx r10d, byte [r12 + rbx]  ; Load tail byte from input SDR
    movzx r11d, byte [rsi + rbx]  ; Load tail byte from synapse mask
    and r10d, r11d                ; Bitwise AND
    popcnt r10d, r10d             ; Native hardware POPCNT instruction
    add r9, r10                   ; Add to tail sum

    inc rbx
    cmp rbx, r8
    jl .tail_loop

.store_result:
    ; Horizontal Reduction of 4x 64-bit Quadwords in YMM5 Register
    vextracti128 xmm4, ymm5, 1    ; Extract upper 128-bit lane into XMM4
    vpaddq xmm5, xmm5, xmm4       ; Add upper 128 bits to lower 128 bits
    vmovq r10, xmm5               ; Extract low 64-bit quadword sum
    vpextrq r11, xmm5, 1          ; Extract high 64-bit quadword sum
    add r10, r11                  ; Combined SIMD sum
    add r10, r9                   ; Add scalar tail sum

    ; Store 32-bit overlap count into output array: overlaps[c] = (uint32_t)r10
    mov dword [r14 + rax * 4], r10d

    inc rax                       ; c++
    jmp .col_loop

.done:
    vzeroupper                    ; Reset upper YMM registers to prevent AVX-SSE transition penalties
    
    ; Restore Callee-Saved Registers
    pop r15
    pop r14
    pop r13
    pop r12
    pop rbx
    pop rbp
    ret

; Mark stack as non-executable per ELF security standards
section .note.GNU-stack noexec alloc progbits
