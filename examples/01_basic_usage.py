"""
Example 01: Basic Spatial Pooler NASM Usage
-------------------------------------------
Demonstrates basic initialization of SpatialPoolerNASM and computing overlaps
over Sparse Distributed Representation (SDR) inputs.
"""

import os
import sys
import numpy as np

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from bindings.sp_wrapper import SpatialPoolerNASM

def main():
    print("=== Example 01: Basic nasm-htm.core Overlap Calculation ===")
    
    # Define dimensions: 2048 input bits, 1024 columns
    input_bits = 2048
    num_columns = 1024
    
    # Initialize SpatialPoolerNASM engine
    sp = SpatialPoolerNASM(
        input_dimensions=input_bits,
        column_dimensions=num_columns,
        syn_perm_connected=0.2  # 20% initial connected synapses
    )
    
    print(f"SpatialPoolerNASM initialized.")
    print(f"  Input Dimensions:  {input_bits} bits ({sp.sdr_bytes} bytes packed)")
    print(f"  Column Dimensions: {num_columns} columns")
    print(f"  Synapse Mask Size: {sp.syn_mask.shape}")
    
    # Create a random input SDR (5% active bits = 102 active bits out of 2048)
    input_sdr = (np.random.rand(input_bits) < 0.05).astype(np.uint8)
    print(f"Generated input SDR with {np.count_nonzero(input_sdr)} active bits.")
    
    # Compute column overlaps using AVX2 NASM core
    overlaps = sp.compute(input_sdr)
    
    print("\nOverlap Results:")
    print(f"  Overlap Array Shape: {overlaps.shape}")
    print(f"  Max Overlap Count:   {overlaps.max()}")
    print(f"  Min Overlap Count:   {overlaps.min()}")
    print(f"  Mean Overlap Count:  {overlaps.mean():.2f}")
    print(f"  First 10 Overlaps:   {overlaps[:10]}")

if __name__ == "__main__":
    main()
