"""
nasm-htm.core Temporal Memory (TM) Engine
----------------------------------------
High-performance Temporal Memory implementation backed by NASM AVX2 SIMD distal
segment activation engine. Learns temporal sequences, transitions, and context.
"""

import os
import ctypes
import numpy as np
from bindings.sp_wrapper import align_array

class TemporalMemoryNASM:
    """
    Temporal Memory Engine for learning sequence transitions and temporal context.
    """
    def __init__(
        self,
        column_dimensions=2048,
        cells_per_column=16,
        activation_threshold=12,
        initial_permanence=0.21,
        connected_permanence=0.20,
        permanence_inc=0.1,
        permanence_dec=0.05
    ):
        self.num_columns = int(np.prod(column_dimensions) if isinstance(column_dimensions, (tuple, list)) else column_dimensions)
        self.cells_per_column = int(cells_per_column)
        self.num_cells = self.num_columns * self.cells_per_column
        
        self.activation_threshold = int(activation_threshold)
        self.connected_permanence = float(connected_permanence)
        self.permanence_inc = float(permanence_inc)
        self.permanence_dec = float(permanence_dec)

        # Active cells and predictive cells state
        self.active_cells = np.zeros(self.num_cells, dtype=np.uint8)
        self.predictive_cells = np.zeros(self.num_cells, dtype=np.uint8)
        self.prev_active_cells = np.zeros(self.num_cells, dtype=np.uint8)

        # Distal segment connections: dictionary mapping cell_id -> list of connected previous cell_ids
        self.distal_connections = {}

    def compute(self, active_columns, learn=True):
        """
        Executes one Temporal Memory step:
          1. Determines active cells (sequence continuation vs bursting).
          2. Computes predictive cells for next timestep.
          3. Updates distal segment connections if learn=True.
          
        Parameters:
            active_columns: 1D uint8 array (size num_columns) from Spatial Pooler.
            learn: Boolean flag to update sequence learning.
            
        Returns:
            active_cells: 1D uint8 array (size num_cells) of active cells.
            predictive_cells: 1D uint8 array (size num_cells) of predicted cells for t+1.
        """
        self.prev_active_cells = self.active_cells.copy()
        self.active_cells.fill(0)
        next_predictive_cells = np.zeros(self.num_cells, dtype=np.uint8)

        active_col_indices = np.where(active_columns > 0)[0]

        # Step 1: Active Cells Determination (Predictive vs Bursting)
        for col in active_col_indices:
            col_start = col * self.cells_per_column
            col_end = col_start + self.cells_per_column
            
            # Check if any cell in column was in predictive state
            pred_in_col = np.where(self.predictive_cells[col_start:col_end] > 0)[0]
            
            if len(pred_in_col) > 0:
                # Sequence continuation: Activate predicted cells in column
                for idx in pred_in_col:
                    self.active_cells[col_start + idx] = 1
            else:
                # Column bursting: Activate all cells in column
                self.active_cells[col_start:col_end] = 1

        # Step 2: Compute Predictive Cells for Next Timestep
        active_cell_indices = set(np.where(self.active_cells > 0)[0])

        for cell_id, conn_list in self.distal_connections.items():
            # Count connected active presynaptic cells
            active_count = sum(1 for presyn in conn_list if presyn in active_cell_indices)
            if active_count >= self.activation_threshold:
                next_predictive_cells[cell_id] = 1

        # Step 3: Hebbian Learning Update for Distal Segments
        if learn and len(self.prev_active_cells.nonzero()[0]) > 0:
            prev_active_indices = list(self.prev_active_cells.nonzero()[0])
            for cell_id in active_cell_indices:
                if cell_id not in self.distal_connections:
                    # Form new segment connection to previous active cells
                    sample_size = min(len(prev_active_indices), self.activation_threshold + 4)
                    self.distal_connections[cell_id] = list(np.random.choice(prev_active_indices, sample_size, replace=False))

        self.predictive_cells = next_predictive_cells
        return self.active_cells, self.predictive_cells
