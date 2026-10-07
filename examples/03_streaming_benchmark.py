"""
Example 03: Streaming Pattern Processing Benchmark
--------------------------------------------------
Measures real-time streaming pattern throughput (ops/sec) and latency (microseconds)
over 10,000 continuous SDR pattern computations.
"""

import os
import sys
import time
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from bindings.sp_wrapper import SpatialPoolerNASM

def main():
    print("=== Example 03: High-Throughput Streaming Pattern Processing ===")
    
    num_iterations = 10000
    input_bits = 2048
    num_columns = 2048
    
    print(f"Initializing SpatialPoolerNASM ({input_bits} inputs, {num_columns} columns)...")
    sp = SpatialPoolerNASM(input_dimensions=input_bits, column_dimensions=num_columns)
    
    print(f"Pre-generating {num_iterations:,} streaming SDR patterns...")
    patterns = [(np.random.rand(input_bits) < 0.1).astype(np.uint8) for _ in range(num_iterations)]
    
    print(f"Streaming {num_iterations:,} patterns through NASM AVX2 assembly core...")
    start_time = time.perf_counter()
    
    for i in range(num_iterations):
        sp.compute(patterns[i])
        
    end_time = time.perf_counter()
    
    total_time = end_time - start_time
    latency_us = (total_time / num_iterations) * 1e6
    throughput = num_iterations / total_time
    
    print("\nPerformance Metrics:")
    print(f"  Total Streaming Time:  {total_time:.4f} seconds")
    print(f"  Latency per Pattern:   {latency_us:.2f} microseconds (µs)")
    print(f"  Streaming Throughput:  {throughput:,.2f} patterns/sec")

if __name__ == "__main__":
    main()
