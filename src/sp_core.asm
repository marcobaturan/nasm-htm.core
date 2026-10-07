; ==============================================================================
; PROJECT: nasm-htm.core
; MODULE:  src/sp_core.asm
; AUTHOR:  Antigravity AI Team
; LICENSE: MIT / Apache-2.0
; ------------------------------------------------------------------------------
; DESCRIPTION:
; Ultra-high-performance x86_64 NASM assembly core for HTM Spatial Pooler.
; Contains:
;   1. calculate_overlap: AVX2 PSHUFB SIMD nibble popcount overlap engine.
;   2. update_permanences: AVX2 SIMD saturation arithmetic Hebbian learning engine.
; ==============================================================================

global calculate_overlap
global update_permanences

section .rodata
align 32
popcount_lut:
    db 0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4
    db 0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4

align 32
mask_0f:
    times 32 db 0x0F

section .text

; ------------------------------------------------------------------------------
; C FUNCTION SIGNATURE 1:
; void calculate_overlap(
;     const uint8_t* input_sdr,    ; RDI -> Input SDR bit array pointer [sdr_bytes]
;     const uint8_t* syn_mask,     ; RSI -> Synapse mask matrix [num_columns * sdr_bytes]
;     uint32_t* overlaps,          ; RDX -> Output overlap array [num_columns]
;     uint64_t num_columns,        ; RCX -> Total number of columns (M)
;     uint64_t sdr_bytes           ; R8  -> Length of SDR bitmask in bytes (N / 8)
; )
; ------------------------------------------------------------------------------
calculate_overlap:
    push rbp
    mov rbp, rsp
    push rbx
    push r12
    push r13
    push r14
    push r15

    mov r12, rdi            ; R12 = input_sdr pointer
    mov r13, rsi            ; R13 = syn_mask pointer base
    mov r14, rdx            ; R14 = overlaps output pointer base
    mov r15, rcx            ; R15 = num_columns total counter

    vmovdqa ymm6, [rel popcount_lut]
    vmovdqa ymm7, [rel mask_0f]

    xor rax, rax            ; RAX = column index c = 0

.col_loop:
    cmp rax, r15
    jge .done

    mov rsi, rax
    imul rsi, r8
    add rsi, r13            ; RSI = syn_mask + (c * sdr_bytes)

    vpxor ymm5, ymm5, ymm5  ; YMM5 = accumulator
    xor rbx, rbx            ; RBX = byte offset i = 0

.simd_loop:
    mov r10, r8
    sub r10, rbx
    cmp r10, 32
    jl .scalar_tail

    vmovdqu ymm0, [r12 + rbx]     ; Input SDR 32 bytes
    vmovdqu ymm1, [rsi + rbx]     ; Synapse mask 32 bytes
    vpand ymm0, ymm0, ymm1        ; Bitwise AND

    ; PSHUFB Low nibbles
    vpand ymm1, ymm0, ymm7
    vpshufb ymm1, ymm6, ymm1

    ; PSHUFB High nibbles
    vpsrlw ymm2, ymm0, 4
    vpand ymm2, ymm2, ymm7
    vpshufb ymm2, ymm6, ymm2

    vpaddb ymm1, ymm1, ymm2

    ; Horizontal VPSADBW
    vpxor ymm3, ymm3, ymm3
    vpsadbw ymm1, ymm1, ymm3
    vpaddq ymm5, ymm5, ymm1

    add rbx, 32
    jmp .simd_loop

.scalar_tail:
    xor r9, r9
    cmp rbx, r8
    jge .store_result

.tail_loop:
    movzx r10d, byte [r12 + rbx]
    movzx r11d, byte [rsi + rbx]
    and r10d, r11d
    popcnt r10d, r10d
    add r9, r10

    inc rbx
    cmp rbx, r8
    jl .tail_loop

.store_result:
    vextracti128 xmm4, ymm5, 1
    vpaddq xmm5, xmm5, xmm4
    vmovq r10, xmm5
    vpextrq r11, xmm5, 1
    add r10, r11
    add r10, r9

    mov dword [r14 + rax * 4], r10d

    inc rax
    jmp .col_loop

.done:
    vzeroupper
    pop r15
    pop r14
    pop r13
    pop r12
    pop rbx
    pop rbp
    ret


; ------------------------------------------------------------------------------
; C FUNCTION SIGNATURE 2:
; void update_permanences(
;     uint8_t* syn_perms,          ; RDI -> Permenence matrix [num_columns * num_inputs]
;     const uint8_t* input_sdr,    ; RSI -> Dense input SDR array [num_inputs] (0 or 1)
;     const uint32_t* active_cols, ; RDX -> Indices of active winning columns [num_active_cols]
;     uint64_t num_active_cols,    ; RCX -> Number of active winning columns
;     uint64_t num_inputs,         ; R8  -> Total input size
;     uint8_t perm_inc,            ; R9  -> Permanence increment step
;     uint8_t perm_dec             ; [stack + 8] -> Permanence decrement step
; )
; ------------------------------------------------------------------------------
update_permanences:
    push rbp
    mov rbp, rsp
    push rbx
    push r12
    push r13
    push r14
    push r15

    mov r12, rdi            ; R12 = syn_perms base
    mov r13, rsi            ; R13 = input_sdr base
    mov r14, rdx            ; R14 = active_cols array base
    mov r15, rcx            ; R15 = num_active_cols counter

    movzx r10d, byte [rbp + 16] ; R10D = perm_dec (passed on stack)
    movzx r9d, r9b              ; R9D = perm_inc

    ; Broadcast perm_inc and perm_dec to 256-bit YMM registers
    vmovd xmm0, r9d
    vpbroadcastb ymm8, xmm0     ; YMM8 = [inc, inc, ..., inc]
    vmovd xmm1, r10d
    vpbroadcastb ymm9, xmm1     ; YMM9 = [dec, dec, ..., dec]

    xor rax, rax            ; Active column index k = 0

.active_col_loop:
    cmp rax, r15
    jge .perm_done

    ; Get column index c = active_cols[k]
    mov edx, dword [r14 + rax * 4]
    
    ; Compute column permanence pointer: rdi_col = syn_perms + c * num_inputs
    mov rdi, rdx
    imul rdi, r8
    add rdi, r12

    xor rbx, rbx            ; Input index i = 0

.perm_simd_loop:
    mov r11, r8
    sub r11, rbx
    cmp r11, 32
    jl .perm_scalar_tail

    ; Load 32 byte permanences and 32 byte input states
    vmovdqu ymm0, [rdi + rbx]     ; YMM0 = Permanences (0..255)
    vmovdqu ymm1, [r13 + rbx]     ; YMM1 = Inputs (0 or 1)

    ; Apply Hebbian updates:
    ; If input[i] > 0: perm[i] = min(255, perm[i] + perm_inc) (via VPADDUSB)
    ; If input[i] == 0: perm[i] = max(0, perm[i] - perm_dec) (via VPSUBUSB)
    vpaddusb ymm2, ymm0, ymm8     ; YMM2 = Saturated addition (perm + inc)
    vpsubusb ymm3, ymm0, ymm9     ; YMM3 = Saturated subtraction (perm - dec)

    ; Select updated values based on input mask (YMM1 > 0)
    vpxor ymm4, ymm4, ymm4
    vpcmpgtb ymm1, ymm1, ymm4     ; YMM1 = Mask (0xFF where input > 0, 0x00 where input == 0)
    vpblendvb ymm0, ymm3, ymm2, ymm1 ; Blend: YMM2 where input > 0, YMM3 where input == 0

    ; Store updated permanences back to memory
    vmovdqu [rdi + rbx], ymm0

    add rbx, 32
    jmp .perm_simd_loop

.perm_scalar_tail:
    cmp rbx, r8
    jge .next_active_col

.perm_tail_loop:
    movzx r11d, byte [rdi + rbx]  ; Current permanence
    movzx ebx, byte [r13 + rbx]   ; Input state (0 or 1)
    
    test ebx, ebx
    jz .dec_perm

.inc_perm:
    add r11d, r9d
    cmp r11d, 255
    jle .save_perm
    mov r11d, 255
    jmp .save_perm

.dec_perm:
    sub r11d, r10d
    jge .save_perm
    xor r11d, r11d

.save_perm:
    mov byte [rdi + rbx], r11b

    inc rbx
    cmp rbx, r8
    jl .perm_tail_loop

.next_active_col:
    inc rax
    jmp .active_col_loop

.perm_done:
    vzeroupper
    pop r15
    pop r14
    pop r13
    pop r12
    pop rbx
    pop rbp
    ret

section .note.GNU-stack noexec alloc progbits
