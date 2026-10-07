"""
Example 01: Complete Spatial Pooler Pipeline (Inhibition & AVX2 Learning)
--------------------------------------------------------------------------
Demonstrates initialization of SpatialPoolerNASM, top-k column inhibition,
and AVX2 SIMD Hebbian permanence learning (learn=True).
"""

import os
import sys
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from bindings.sp_wrapper import SpatialPoolerNASM

def main():
    print("=== Example 01: Complete Spatial Pooler Pipeline ===")
    
    input_bits = 2048
    num_columns = 1024
    target_sparsity = 0.02  # 2% active columns (20 active columns out of 1024)
    
    # Initialize SpatialPoolerNASM engine
    sp = SpatialPoolerNASM(
        input_dimensions=input_bits,
        column_dimensions=num_columns,
        syn_perm_connected=0.1,
        num_active_columns_per_inh=target_sparsity
    )
    
    print(f"SpatialPoolerNASM initialized.")
    print(f"  Input Size:       {input_bits} bits")
    print(f"  Column Count:     {num_columns} columns")
    print(f"  Target Sparsity:  {target_sparsity * 100:.1f}% ({sp.k_winners} active columns)")
    
    # Generate a random 5% active input SDR
    input_sdr = (np.random.rand(input_bits) < 0.05).astype(np.uint8)
    print(f"Generated input SDR with {np.count_nonzero(input_sdr)} active bits.")
    
    # Cycle 1: Compute active column SDR with Hebbian Learning (learn=True)
    active_sdr1 = sp.compute(input_sdr, learn=True)
    print(f"\nCycle 1 Output:")
    print(f"  Active Columns Count: {active_sdr1.sum()}")
    print(f"  Active Column Indices: {np.where(active_sdr1 == 1)[0]}")
    
    # Cycle 2: Present same input SDR again (permanences reinforced by AVX2 learning engine)
    active_sdr2 = sp.compute(input_sdr, learn=True)
    print(f"\nCycle 2 Output (Post-Learning Reinforcement):")
    print(f"  Active Columns Count: {active_sdr2.sum()}")
    print(f"  Active Column Indices: {np.where(active_sdr2 == 1)[0]}")

if __name__ == "__main__":
    main()
