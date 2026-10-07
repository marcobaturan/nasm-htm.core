import os
import sys
import time
import psutil
import tracemalloc
import numpy as np

# Ensure parent directory is in path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from bindings.sp_wrapper import SpatialPoolerNASM

def run_benchmark(num_iterations=10000, input_bits=2048, num_columns=2048):
    print("==========================================================")
    print("       TITAN-HTM Spatial Pooler NASM AVX2 Benchmark       ")
    print("==========================================================")
    print(f"Target CPU:       AMD Ryzen 7 4700U (Zen 2 Architecture)")
    print(f"SDR Input Size:   {input_bits} bits ({input_bits // 8} bytes)")
    print(f"Columns:          {num_columns}")
    print(f"Iterations:       {num_iterations:,}")
    print("----------------------------------------------------------")

    # RAM baseline measurement
    process = psutil.Process(os.getpid())
    ram_before_mb = process.memory_info().rss / (1024 * 1024)
    tracemalloc.start()

    # Initialize Spatial Pooler
    sp = SpatialPoolerNASM(input_dimensions=input_bits, column_dimensions=num_columns, syn_perm_connected=0.2)
    
    # Generate 10,000 random input SDR vectors (dense noise)
    print("Generating 10,000 random input SDR vectors...")
    inputs = [(np.random.rand(input_bits) < 0.1).astype(np.uint8) for _ in range(num_iterations)]
    
    ram_during_mb = process.memory_info().rss / (1024 * 1024)

    # Initial CPU snapshot
    psutil.cpu_percent(percpu=True)
    
    print("Executing NASM AVX2 Spatial Pooler core...")
    start_time = time.perf_counter()
    
    for i in range(num_iterations):
        sp.compute(inputs[i])

    end_time = time.perf_counter()
    
    cpu_usage_per_core = psutil.cpu_percent(percpu=True)
    ram_after_mb = process.memory_info().rss / (1024 * 1024)
    _, peak_malloc = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    total_time_sec = end_time - start_time
    latency_us_per_cycle = (total_time_sec / num_iterations) * 1e6
    throughput_ops_sec = num_iterations / total_time_sec

    # Baseline benchmark: NumPy bitwise unpack unpackbits sample
    print("Executing NumPy Baseline benchmark for comparison...")
    sample_size = min(300, num_iterations)
    numpy_start = time.perf_counter()
    for i in range(sample_size):
        packed_in = np.packbits(inputs[i])
        if packed_in.size < sp.sdr_bytes:
            padded = np.zeros(sp.sdr_bytes, dtype=np.uint8)
            padded[:packed_in.size] = packed_in
            packed_in = padded
        _ = np.array([np.unpackbits(np.bitwise_and(packed_in, sp.syn_mask[c])).sum() for c in range(sp.num_columns)], dtype=np.uint32)
    numpy_end = time.perf_counter()
    
    numpy_latency_us = ((numpy_end - numpy_start) / sample_size) * 1e6
    speedup_factor = numpy_latency_us / latency_us_per_cycle

    print("\n==========================================================")
    print("                  BENCHMARK RESULTS                       ")
    print("==========================================================")
    print(f"Total Execution Time:    {total_time_sec:.4f} seconds")
    print(f"Latency per Cycle:       {latency_us_per_cycle:.2f} microseconds (µs)")
    print(f"Throughput:              {throughput_ops_sec:,.2f} ops/sec")
    print(f"NumPy Baseline Latency:  {numpy_latency_us:.2f} microseconds (µs)")
    print(f"Speedup vs NumPy:        {speedup_factor:.2f}x faster")
    print("----------------------------------------------------------")
    print("CPU Consumption per Core:")
    for idx, cpu in enumerate(cpu_usage_per_core):
        print(f"  Core {idx}: {cpu:.1f}%")
    print("----------------------------------------------------------")
    print("Memory Usage (RAM):")
    print(f"  RAM Before Init:       {ram_before_mb:.2f} MB")
    print(f"  RAM During Iterations: {ram_during_mb:.2f} MB")
    print(f"  RAM After Completion:  {ram_after_mb:.2f} MB")
    print(f"  Tracemalloc Peak:      {peak_malloc / (1024 * 1024):.2f} MB")
    print("==========================================================")

    return {
        "total_time_sec": total_time_sec,
        "latency_us_per_cycle": latency_us_per_cycle,
        "throughput_ops_sec": throughput_ops_sec,
        "numpy_latency_us": numpy_latency_us,
        "speedup_factor": speedup_factor,
        "cpu_usage_per_core": cpu_usage_per_core,
        "ram_before_mb": ram_before_mb,
        "ram_during_mb": ram_during_mb,
        "ram_after_mb": ram_after_mb,
        "peak_malloc_mb": peak_malloc / (1024 * 1024)
    }

if __name__ == "__main__":
    run_benchmark()
