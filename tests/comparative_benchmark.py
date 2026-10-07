import os
import sys
import time
import psutil
import tracemalloc
import numpy as np

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from bindings.sp_wrapper import SpatialPoolerNASM

# Try importing official htm.core
try:
    from htm.algorithms import SpatialPooler as SpatialPoolerHTMCore
    from htm.bindings.sdr import SDR
    HTM_CORE_AVAILABLE = True
except ImportError:
    HTM_CORE_AVAILABLE = False

class SpatialPoolerPython:
    """
    Pure Python / NumPy reference implementation of HTM Spatial Pooler overlap algorithm.
    """
    def __init__(self, input_dimensions, column_dimensions, syn_perm_connected=0.2):
        self.num_inputs = int(np.prod(input_dimensions))
        self.num_columns = int(np.prod(column_dimensions))
        self.sdr_bytes = ((self.num_inputs + 7) // 8 + 31) // 32 * 32
        
        syn_mask_init = (np.random.rand(self.num_columns, self.sdr_bytes * 8) < syn_perm_connected).astype(np.uint8)
        self.syn_mask = np.packbits(syn_mask_init, axis=1)

    def compute(self, input_array):
        if input_array.size == self.num_inputs:
            packed_input = np.packbits(input_array)
            if packed_input.size < self.sdr_bytes:
                padded = np.zeros(self.sdr_bytes, dtype=np.uint8)
                padded[:packed_input.size] = packed_input
                packed_input = padded
        else:
            packed_input = input_array
            
        overlaps = np.array([
            np.unpackbits(np.bitwise_and(packed_input, self.syn_mask[c])).sum()
            for c in range(self.num_columns)
        ], dtype=np.uint32)
        return overlaps

def run_comparative_benchmark(num_iterations=1000, input_bits=2048, num_columns=2048):
    print("=========================================================================")
    print("      nasm-htm.core vs. HTM.CORE vs. PURE PYTHON COMPARATIVE BENCHMARK   ")
    print("=========================================================================")
    print(f"Target CPU:         AMD Ryzen 7 4700U (Zen 2 Architecture)")
    print(f"Input SDR Size:     {input_bits} bits ({input_bits // 8} bytes)")
    print(f"Columns:            {num_columns}")
    print(f"Iterations:         {num_iterations:,}")
    print(f"htm.core C++ state: {'INSTALLED & AVAILABLE' if HTM_CORE_AVAILABLE else 'NOT INSTALLED'}")
    print("-------------------------------------------------------------------------")

    inputs = [(np.random.rand(input_bits) < 0.1).astype(np.uint8) for _ in range(num_iterations)]

    # 1. BENCHMARK nasm-htm.core (NASM AVX2 CORE)
    print("\n[1/3] Benchmarking nasm-htm.core (NASM AVX2 Engine)...")
    sp_nasm = SpatialPoolerNASM(input_dimensions=input_bits, column_dimensions=num_columns, syn_perm_connected=0.2)
    
    start_nasm = time.perf_counter()
    for i in range(num_iterations):
        sp_nasm.compute(inputs[i])
    end_nasm = time.perf_counter()

    nasm_time = end_nasm - start_nasm
    nasm_latency_us = (nasm_time / num_iterations) * 1e6
    nasm_throughput = num_iterations / nasm_time

    # 2. BENCHMARK OFFICIAL HTM.CORE (C++ BINDINGS) IF AVAILABLE
    htm_core_time = None
    htm_core_latency_us = None
    htm_core_throughput = None
    
    if HTM_CORE_AVAILABLE:
        print("[2/3] Benchmarking Official htm.core (C++ Library)...")
        sp_htm = SpatialPoolerHTMCore(
            inputDimensions=[input_bits],
            columnDimensions=[num_columns],
            potentialRadius=64,
            synPermConnected=0.2
        )
        htm_sdrs = []
        for i in range(num_iterations):
            s = SDR([input_bits])
            s.sparse = np.where(inputs[i])[0].astype(np.int32).tolist()
            htm_sdrs.append(s)
        active_sdr = SDR([num_columns])
        
        start_htm = time.perf_counter()
        for i in range(num_iterations):
            sp_htm.compute(htm_sdrs[i], False, active_sdr)
        end_htm = time.perf_counter()

        htm_core_time = end_htm - start_htm
        htm_core_latency_us = (htm_core_time / num_iterations) * 1e6
        htm_core_throughput = num_iterations / htm_core_time

    # 3. BENCHMARK PURE PYTHON / NUMPY REFERENCE
    print("[3/3] Benchmarking Pure Python / NumPy Baseline...")
    sp_py = SpatialPoolerPython(input_dimensions=input_bits, column_dimensions=num_columns, syn_perm_connected=0.2)
    
    py_samples = min(100, num_iterations)
    start_py = time.perf_counter()
    for i in range(py_samples):
        sp_py.compute(inputs[i])
    end_py = time.perf_counter()

    py_latency_us = ((end_py - start_py) / py_samples) * 1e6
    py_throughput = 1e6 / py_latency_us

    # SUMMARY COMPARISON TABLE
    print("\n=========================================================================")
    print("                    COMPARATIVE BENCHMARK SUMMARY                        ")
    print("=========================================================================")
    print(f"{'Implementation':<27} | {'Latency (µs)':<15} | {'Throughput (ops/s)':<18} | {'Speedup vs Py':<12}")
    print("-" * 81)
    print(f"{'nasm-htm.core (NASM AVX2)':<27} | {nasm_latency_us:<15.2f} | {nasm_throughput:<18,.2f} | {py_latency_us/nasm_latency_us:<12.2f}x")
    
    if HTM_CORE_AVAILABLE and htm_core_latency_us:
        print(f"{'htm.core (C++ Official)':<27} | {htm_core_latency_us:<15.2f} | {htm_core_throughput:<18,.2f} | {py_latency_us/htm_core_latency_us:<12.2f}x")
        ratio = htm_core_latency_us / nasm_latency_us
        print(f"\nnasm-htm.core vs htm.core C++: {ratio:.2f}x Faster (Latency: {nasm_latency_us:.2f}µs vs {htm_core_latency_us:.2f}µs)")
    else:
        print(f"{'htm.core (C++)':<27} | {'N/A':<15} | {'N/A':<18} | {'N/A':<12}")

    print(f"{'Pure Python / NumPy':<27} | {py_latency_us:<15.2f} | {py_throughput:<18,.2f} | 1.00x")
    print("=========================================================================")

if __name__ == "__main__":
    run_comparative_benchmark()
