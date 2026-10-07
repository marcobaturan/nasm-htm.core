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
    # Allocate aligned memory buffer
    buf = np.empty(arr.size * arr.itemsize + alignment, dtype=np.uint8)
    offset = buf.ctypes.data % alignment
    shift = 0 if offset == 0 else (alignment - offset)
    aligned_arr = buf[shift:shift + arr.nbytes].view(arr.dtype).reshape(arr.shape)
    np.copyto(aligned_arr, arr)
    return aligned_arr

class SpatialPoolerNASM:
    """
    Python wrapper for the nasm-htm.core Spatial Pooler engine written in NASM x86_64 AVX2 assembly.
    Imitates the basic interface of htm.core SpatialPooler.
    """
    def __init__(self, input_dimensions, column_dimensions, potential_radius=1.0, syn_perm_connected=0.1):
        self.input_dimensions = input_dimensions if isinstance(input_dimensions, (tuple, list)) else (input_dimensions,)
        self.column_dimensions = column_dimensions if isinstance(column_dimensions, (tuple, list)) else (column_dimensions,)
        
        self.num_inputs = int(np.prod(self.input_dimensions))
        self.num_columns = int(np.prod(self.column_dimensions))
        
        # Calculate SDR bytes (bits packed into bytes, rounded up to multiple of 32 for AVX2 safety)
        raw_sdr_bytes = (self.num_inputs + 7) // 8
        self.sdr_bytes = ((raw_sdr_bytes + 31) // 32) * 32
        
        # Initialize synapse connection matrix (num_columns, sdr_bytes * 8 bits)
        syn_mask_init = (np.random.rand(self.num_columns, self.sdr_bytes * 8) < syn_perm_connected).astype(np.uint8)
        self.syn_mask = np.packbits(syn_mask_init, axis=1)
        self.syn_mask = align_array(self.syn_mask, alignment=32)

        # Load shared C library
        lib_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src", "libspcore.so"))
        if not os.path.exists(lib_path):
            raise FileNotFoundError(f"Dynamic library not found at {lib_path}. Run 'make' in src/ directory.")
        
        self._lib = ctypes.CDLL(lib_path)
        self._lib.calculate_overlap.argtypes = [
            ctypes.c_void_p, # input_sdr
            ctypes.c_void_p, # syn_mask
            ctypes.c_void_p, # overlaps
            ctypes.c_uint64, # num_columns
            ctypes.c_uint64  # sdr_bytes
        ]
        self._lib.calculate_overlap.restype = None

    def compute(self, input_array, learn=False, overlaps=None):
        """
        Computes overlaps for input SDR array.
        
        Parameters:
            input_array: 1D NumPy array (0s and 1s of size num_inputs, or pre-packed uint8 bytes).
            learn: Boolean (reserved for future learning updates).
            overlaps: Optional uint32 NumPy array of size num_columns to receive results.
            
        Returns:
            overlaps: 1D uint32 NumPy array containing active synapse counts per column.
        """
        if input_array.dtype != np.uint8:
            input_array = input_array.astype(np.uint8)
            
        if input_array.size == self.num_inputs:
            packed_input = np.packbits(input_array)
            if packed_input.size < self.sdr_bytes:
                padded = np.zeros(self.sdr_bytes, dtype=np.uint8)
                padded[:packed_input.size] = packed_input
                packed_input = padded
        elif input_array.size == self.sdr_bytes:
            packed_input = input_array
        else:
            raise ValueError(f"Input array size {input_array.size} mismatch with num_inputs {self.num_inputs} or sdr_bytes {self.sdr_bytes}")

        packed_input = align_array(packed_input, alignment=32)
        
        if overlaps is None or overlaps.size != self.num_columns:
            overlaps = np.zeros(self.num_columns, dtype=np.uint32)
        overlaps = align_array(overlaps, alignment=32)

        self._lib.calculate_overlap(
            packed_input.ctypes.data,
            self.syn_mask.ctypes.data,
            overlaps.ctypes.data,
            ctypes.c_uint64(self.num_columns),
            ctypes.c_uint64(self.sdr_bytes)
        )

        return overlaps
