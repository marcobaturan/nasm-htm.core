"""
Example 02: Dense vs Packed Bit SDR Inputs
-------------------------------------------
Demonstrates how SpatialPoolerNASM accepts both 1D unpacked NumPy uint8 arrays
(0s and 1s) and pre-packed uint8 bitmasks for maximum memory efficiency.
"""

import os
import sys
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from bindings.sp_wrapper import SpatialPoolerNASM

def main():
    print("=== Example 02: Dense vs Packed Bit SDR Inputs ===")
    
    input_bits = 2048
    num_columns = 1024
    
    sp = SpatialPoolerNASM(input_dimensions=input_bits, column_dimensions=num_columns)
    
    # 1. Unpacked Dense Array (2048 elements of uint8)
    dense_sdr = (np.random.rand(input_bits) < 0.1).astype(np.uint8)
    print(f"1. Dense SDR (unpacked): size = {dense_sdr.nbytes} bytes")
    overlaps_dense = sp.compute(dense_sdr)
    
    # 2. Pre-packed Bit Array (256 elements of uint8)
    packed_sdr = np.packbits(dense_sdr)
    if packed_sdr.size < sp.sdr_bytes:
        padded = np.zeros(sp.sdr_bytes, dtype=np.uint8)
        padded[:packed_sdr.size] = packed_sdr
        packed_sdr = padded
        
    print(f"2. Packed SDR (bitmask):  size = {packed_sdr.nbytes} bytes")
    overlaps_packed = sp.compute(packed_sdr)
    
    # Verify outputs match 100%
    assert np.array_equal(overlaps_dense, overlaps_packed), "Error: Overlap mismatch!"
    print("\nSUCCESS: Overlaps from dense and pre-packed SDRs match 100%!")
    print(f"  First 5 Overlaps (Dense):  {overlaps_dense[:5]}")
    print(f"  First 5 Overlaps (Packed): {overlaps_packed[:5]}")

if __name__ == "__main__":
    main()
