import os
import ctypes
import numpy as np

def align_array(arr, alignment=32):
    """
    Ensures a NumPy array is contiguous and aligned to `alignment` bytes (default 32 bytes for AVX2 SIMD).
    """
    arr = np.ascontiguousarray(arr)
    if arr.ctypes.data % alignment == 0:
        return arr
    buf = np.empty(arr.size * arr.itemsize + alignment, dtype=np.uint8)
    offset = buf.ctypes.data % alignment
    shift = 0 if offset == 0 else (alignment - offset)
    aligned_arr = buf[shift:shift + arr.nbytes].view(arr.dtype).reshape(arr.shape)
    np.copyto(aligned_arr, arr)
    return aligned_arr

class SpatialPoolerNASM:
    """
    Complete Python Spatial Pooler Engine backed by NASM x86_64 AVX2 assembly core.
    Provides:
      - AVX2 SIMD PSHUFB Overlap calculation.
      - Top-K Column Inhibition & Winner Selection.
      - Moving Average Duty Cycles & Column Boosting.
      - AVX2 SIMD Hebbian Permanence Learning (`learn=True`).
    """
    def __init__(
        self,
        input_dimensions,
        column_dimensions,
        potential_radius=1.0,
        syn_perm_connected=0.1,
        syn_perm_active_inc=5,
        syn_perm_inactive_dec=2,
        stimulus_threshold=0,
        num_active_columns_per_inh=0.02,  # 2% target sparsity
        max_boost=2.0,
        duty_cycle_period=1000
    ):
        self.input_dimensions = input_dimensions if isinstance(input_dimensions, (tuple, list)) else (input_dimensions,)
        self.column_dimensions = column_dimensions if isinstance(column_dimensions, (tuple, list)) else (column_dimensions,)
        
        self.num_inputs = int(np.prod(self.input_dimensions))
        self.num_columns = int(np.prod(self.column_dimensions))
        
        self.syn_perm_connected = int(syn_perm_connected * 255)
        self.syn_perm_active_inc = int(syn_perm_active_inc)
        self.syn_perm_inactive_dec = int(syn_perm_inactive_dec)
        self.stimulus_threshold = stimulus_threshold
        self.max_boost = float(max_boost)
        self.duty_cycle_period = float(duty_cycle_period)

        if isinstance(num_active_columns_per_inh, float) and num_active_columns_per_inh <= 1.0:
            self.k_winners = max(1, int(self.num_columns * num_active_columns_per_inh))
        else:
            self.k_winners = int(num_active_columns_per_inh)

        raw_sdr_bytes = (self.num_inputs + 7) // 8
        self.sdr_bytes = ((raw_sdr_bytes + 31) // 32) * 32

        # 1. Permanence Matrix [num_columns, num_inputs] (values 0..255)
        initial_perms = (np.random.rand(self.num_columns, self.num_inputs) * 0.2 * 255).astype(np.uint8)
        self.syn_perms = align_array(initial_perms, alignment=32)

        # 2. Derived Connected Synapse Mask Matrix [num_columns, sdr_bytes]
        self._update_syn_mask()

        # 3. Duty Cycles & Boost Factors
        self.active_duty_cycles = np.zeros(self.num_columns, dtype=np.float32)
        self.overlap_duty_cycles = np.zeros(self.num_columns, dtype=np.float32)
        self.boost_factors = np.ones(self.num_columns, dtype=np.float32)

        # Load shared C library
        lib_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src", "libspcore.so"))
        if not os.path.exists(lib_path):
            raise FileNotFoundError(f"Dynamic library not found at {lib_path}. Run 'make' in src/ directory.")
        
        self._lib = ctypes.CDLL(lib_path)
        
        # calculate_overlap signature
        self._lib.calculate_overlap.argtypes = [
            ctypes.c_void_p, # input_sdr
            ctypes.c_void_p, # syn_mask
            ctypes.c_void_p, # overlaps
            ctypes.c_uint64, # num_columns
            ctypes.c_uint64  # sdr_bytes
        ]
        self._lib.calculate_overlap.restype = None

        # update_permanences signature
        self._lib.update_permanences.argtypes = [
            ctypes.c_void_p, # syn_perms
            ctypes.c_void_p, # input_sdr
            ctypes.c_void_p, # active_cols
            ctypes.c_uint64, # num_active_cols
            ctypes.c_uint64, # num_inputs
            ctypes.c_uint8,  # perm_inc
            ctypes.c_uint8   # perm_dec
        ]
        self._lib.update_permanences.restype = None

    def _update_syn_mask(self):
        """Generates packed connected synapse bitmask from permanence values."""
        conn_bits = (self.syn_perms >= self.syn_perm_connected).astype(np.uint8)
        if conn_bits.shape[1] < self.sdr_bytes * 8:
            padded = np.zeros((self.num_columns, self.sdr_bytes * 8), dtype=np.uint8)
            padded[:, :conn_bits.shape[1]] = conn_bits
            conn_bits = padded
        self.syn_mask = align_array(np.packbits(conn_bits, axis=1), alignment=32)

    def update_duty_cycles(self, overlaps, active_array):
        """Updates moving average duty cycles and column boosting factors."""
        alpha = 1.0 / self.duty_cycle_period
        self.overlap_duty_cycles = (1.0 - alpha) * self.overlap_duty_cycles + alpha * (overlaps > 0)
        self.active_duty_cycles = (1.0 - alpha) * self.active_duty_cycles + alpha * (active_array > 0)

        max_duty = float(np.max(self.active_duty_cycles))
        if max_duty > 0:
            ratio = self.active_duty_cycles / max_duty
            self.boost_factors = np.exp((1.0 - ratio) * (self.max_boost - 1.0))
        else:
            self.boost_factors.fill(1.0)

    def compute(self, input_array, learn=False, active_array=None):
        """
        Executes full Spatial Pooler compute cycle:
          1. Overlap calculation (AVX2 NASM engine)
          2. Boost Factor Application & Top-K Column Inhibition
          3. Moving Average Duty Cycle Updating
          4. Hebbian Permanence Learning (if learn=True via AVX2 NASM engine)
        """
        if input_array.dtype != np.uint8:
            input_array = input_array.astype(np.uint8)

        if input_array.size == self.num_inputs:
            dense_input = input_array
            packed_input = np.packbits(input_array)
            if packed_input.size < self.sdr_bytes:
                padded = np.zeros(self.sdr_bytes, dtype=np.uint8)
                padded[:packed_input.size] = packed_input
                packed_input = padded
        elif input_array.size == self.sdr_bytes:
            packed_input = input_array
            dense_input = np.unpackbits(input_array)[:self.num_inputs]
        else:
            raise ValueError(f"Input array size {input_array.size} mismatch with num_inputs {self.num_inputs} or sdr_bytes {self.sdr_bytes}")

        packed_input = align_array(packed_input, alignment=32)

        # 1. STEP 1: Overlap calculation (NASM AVX2 engine)
        overlaps = np.zeros(self.num_columns, dtype=np.uint32)
        overlaps = align_array(overlaps, alignment=32)

        self._lib.calculate_overlap(
            packed_input.ctypes.data,
            self.syn_mask.ctypes.data,
            overlaps.ctypes.data,
            ctypes.c_uint64(self.num_columns),
            ctypes.c_uint64(self.sdr_bytes)
        )

        # Apply column boosting
        boosted_overlaps = overlaps.astype(np.float32) * self.boost_factors

        # 2. STEP 2: Top-K Column Inhibition
        if self.stimulus_threshold > 0:
            boosted_overlaps[overlaps < self.stimulus_threshold] = 0

        if self.k_winners < self.num_columns:
            active_col_indices = np.argpartition(boosted_overlaps, -self.k_winners)[-self.k_winners:]
            active_col_indices = active_col_indices[boosted_overlaps[active_col_indices] > 0]
        else:
            active_col_indices = np.where(boosted_overlaps > 0)[0]

        active_col_indices = align_array(active_col_indices.astype(np.uint32), alignment=32)

        if active_array is None or active_array.size != self.num_columns:
            active_array = np.zeros(self.num_columns, dtype=np.uint8)
        else:
            active_array.fill(0)

        active_array[active_col_indices] = 1

        # 3. STEP 3: Duty Cycle Updating
        if learn:
            self.update_duty_cycles(overlaps, active_array)

        # 4. STEP 4: Hebbian Learning Update (NASM AVX2 engine)
        if learn and active_col_indices.size > 0:
            dense_input_aligned = align_array(dense_input, alignment=32)
            self._lib.update_permanences(
                self.syn_perms.ctypes.data,
                dense_input_aligned.ctypes.data,
                active_col_indices.ctypes.data,
                ctypes.c_uint64(active_col_indices.size),
                ctypes.c_uint64(self.num_inputs),
                ctypes.c_uint8(self.syn_perm_active_inc),
                ctypes.c_uint8(self.syn_perm_inactive_dec)
            )
            self._update_syn_mask()

        return active_array
