"""
Example 04: Full HTM Cognitive Arc (Encoder -> Spatial Pooler -> Temporal Memory)
---------------------------------------------------------------------------------
Demonstrates a complete end-to-end HTM Cognitive Architecture:
  1. Encoder: Converts continuous scalar values into SDRs.
  2. Spatial Pooler (NASM AVX2): Computes overlaps and active column SDRs.
  3. Temporal Memory (NASM TM): Learns temporal sequence transitions and predicts next state.
"""

import os
import sys
import datetime
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from bindings.encoders import ScalarEncoder, DateEncoder
from bindings.sp_wrapper import SpatialPoolerNASM
from bindings.tm_wrapper import TemporalMemoryNASM

def main():
    print("=========================================================================")
    print("     FULL HTM COGNITIVE ARC: ENCODER -> SPATIAL POOLER -> TEMPORAL MEMORY ")
    print("=========================================================================")

    # 1. Initialize Encoders
    input_bits = 2048
    encoder = ScalarEncoder(n=input_bits, w=41, min_val=0.0, max_val=100.0)
    print(f"[1/3] ScalarEncoder initialized: {input_bits} output bits, 41 active bits.")

    # 2. Initialize Spatial Pooler (NASM AVX2)
    num_columns = 1024
    sp = SpatialPoolerNASM(
        input_dimensions=input_bits,
        column_dimensions=num_columns,
        syn_perm_connected=0.1,
        num_active_columns_per_inh=0.02  # 2% sparsity = 20 active columns
    )
    print(f"[2/3] SpatialPoolerNASM initialized: {num_columns} columns, AVX2 engine ready.")

    # 3. Initialize Temporal Memory
    cells_per_col = 8
    tm = TemporalMemoryNASM(
        column_dimensions=num_columns,
        cells_per_column=cells_per_col,
        activation_threshold=4
    )
    print(f"[3/3] TemporalMemoryNASM initialized: {num_columns * cells_per_col} total cells.")

    # 4. Stream a Sine-Wave Sequence through the Full Cognitive Arc
    print("\n-------------------------------------------------------------------------")
    print("Streaming Sine Wave Sequence through Full Cognitive Arc...")
    print("-------------------------------------------------------------------------")

    sequence = [50.0 + 40.0 * np.sin(i * 0.2) for i in range(20)]
    
    for t, val in enumerate(sequence):
        # Step A: Encode Scalar to SDR
        input_sdr = encoder.encode(val)
        
        # Step B: Spatial Pooler Compute (AVX2 Overlap + Inhibition + AVX2 Learning)
        active_columns = sp.compute(input_sdr, learn=True)
        
        # Step C: Temporal Memory Compute (Sequence Learning & Prediction)
        active_cells, predictive_cells = tm.compute(active_columns, learn=True)
        
        print(f"Step {t+1:02d} | Val: {val:5.1f} | Active Cols: {active_columns.sum():2d} | Active Cells: {active_cells.sum():3d} | Predicted Cells: {predictive_cells.sum():3d}")

    print("\n=========================================================================")
    print("SUCCESS: Full Cognitive Arc (Encoder -> SP AVX2 -> TM Engine) Executed!")
    print("=========================================================================")

if __name__ == "__main__":
    main()
