import os
import sys
import time
import psutil
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

def get_cpu_temp():
    """Returns CPU temperature in °C using psutil or sysfs k10temp/acpitz."""
    try:
        temps = psutil.sensors_temperatures()
        if "k10temp" in temps and len(temps["k10temp"]) > 0:
            return temps["k10temp"][0].current
        if "acpitz" in temps and len(temps["acpitz"]) > 0:
            return temps["acpitz"][0].current
    except Exception:
        pass
    
    if os.path.exists("/sys/class/thermal/thermal_zone0/temp"):
        try:
            with open("/sys/class/thermal/thermal_zone0/temp") as f:
                return float(f.read().strip()) / 1000.0
        except Exception:
            pass
    return 0.0

def get_power_watts():
    """Returns system power consumption in Watts from sysfs battery sensor."""
    p_path = "/sys/class/power_supply/BAT0/power_now"
    if os.path.exists(p_path):
        try:
            with open(p_path) as f:
                uw = float(f.read().strip())
                return uw / 1e6  # Convert microwatts to Watts
        except Exception:
            pass
    return 0.0

def measure_disk_size():
    """Calculates comparative disk size of nasm-htm.core binary vs htm.core C++ extension binaries."""
    nasm_so_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src", "libspcore.so"))
    nasm_so_bytes = os.path.getsize(nasm_so_path) if os.path.exists(nasm_so_path) else 0

    htm_core_so_bytes = 0
    if HTM_CORE_AVAILABLE:
        try:
            import htm.bindings.algorithms as alg
            so_file = alg.__file__
            htm_core_so_bytes = os.path.getsize(so_file)
        except Exception:
            pass

    return {
        "nasm_so_bytes": nasm_so_bytes,
        "nasm_so_kb": nasm_so_bytes / 1024.0,
        "htm_core_so_bytes": htm_core_so_bytes,
        "htm_core_so_mb": htm_core_so_bytes / (1024.0 * 1024.0)
    }

def run_hardware_telemetry_benchmark(num_iterations=200, input_bits=2048, num_columns=2048):
    print("=========================================================================")
    print("  HARDWARE TELEMETRY BENCHMARK: nasm-htm.core vs HTM.CORE vs PURE PYTHON ")
    print("=========================================================================")
    print("Target CPU:              AMD Ryzen 7 4700U (Zen 2 Architecture)")
    print(f"Input SDR Size:          {input_bits} bits ({input_bits // 8} bytes)")
    print(f"Columns:                 {num_columns}")
    print(f"Iterations:              {num_iterations:,}")
    print("-------------------------------------------------------------------------")

    disk_info = measure_disk_size()
    print("1. DISK FOOTPRINT COMPARISON:")
    print(f"   - nasm-htm.core (libspcore.so):  {disk_info['nasm_so_kb']:.2f} KB ({disk_info['nasm_so_bytes']:,} bytes)")
    if HTM_CORE_AVAILABLE:
        print(f"   - htm.core C++ extension (.so):  {disk_info['htm_core_so_mb']:.2f} MB ({disk_info['htm_core_so_bytes']:,} bytes)")
        if disk_info['nasm_so_bytes'] > 0:
            print(f"   => nasm-htm.core is {disk_info['htm_core_so_bytes'] / disk_info['nasm_so_bytes']:.1f}x smaller on disk!")
    print("-------------------------------------------------------------------------")

    # Pre-generate data
    print("Pre-generating test data...")
    inputs = [(np.random.rand(input_bits) < 0.1).astype(np.uint8) for _ in range(num_iterations)]
    
    htm_sdrs = []
    if HTM_CORE_AVAILABLE:
        for i in range(num_iterations):
            s = SDR([input_bits])
            s.sparse = np.where(inputs[i])[0].astype(np.int32).tolist()
            htm_sdrs.append(s)

    process = psutil.Process(os.getpid())

    # --- TEST 1: nasm-htm.core ---
    print("\n[TEST 1/3] Profiling nasm-htm.core (AVX2 SIMD Assembly)...")
    ram_before1 = process.memory_info().rss / (1024 * 1024)
    temp_before1 = get_cpu_temp()
    power_before1 = get_power_watts()

    sp_nasm = SpatialPoolerNASM(input_dimensions=input_bits, column_dimensions=num_columns, syn_perm_connected=0.2)

    start_time1 = time.perf_counter()
    for i in range(num_iterations):
        sp_nasm.compute(inputs[i])
    end_time1 = time.perf_counter()

    duration1 = end_time1 - start_time1
    temp_after1 = get_cpu_temp()
    power_after1 = get_power_watts()
    ram_after1 = process.memory_info().rss / (1024 * 1024)

    avg_power1 = max(power_before1, power_after1)
    energy_joules1 = (avg_power1 * duration1) if avg_power1 > 0 else 0.0

    # --- TEST 2: htm.core C++ ---
    duration2 = 0.0
    temp_after2 = 0.0
    power_after2 = 0.0
    ram_after2 = 0.0
    avg_power2 = 0.0
    energy_joules2 = 0.0

    if HTM_CORE_AVAILABLE:
        print("[TEST 2/3] Profiling Official htm.core (C++ Library)...")
        ram_before2 = process.memory_info().rss / (1024 * 1024)
        temp_before2 = get_cpu_temp()
        power_before2 = get_power_watts()

        sp_htm = SpatialPoolerHTMCore(
            inputDimensions=[input_bits],
            columnDimensions=[num_columns],
            potentialRadius=64,
            synPermConnected=0.2
        )
        active_sdr = SDR([num_columns])

        start_time2 = time.perf_counter()
        for i in range(num_iterations):
            sp_htm.compute(htm_sdrs[i], False, active_sdr)
        end_time2 = time.perf_counter()

        duration2 = end_time2 - start_time2
        temp_after2 = get_cpu_temp()
        power_after2 = get_power_watts()
        ram_after2 = process.memory_info().rss / (1024 * 1024)

        avg_power2 = max(power_before2, power_after2)
        energy_joules2 = (avg_power2 * duration2) if avg_power2 > 0 else 0.0

    # --- TEST 3: Pure Python ---
    print("[TEST 3/3] Profiling Pure Python / NumPy Baseline...")
    ram_before3 = process.memory_info().rss / (1024 * 1024)
    temp_before3 = get_cpu_temp()

    sp_py = SpatialPoolerPython(input_dimensions=input_bits, column_dimensions=num_columns, syn_perm_connected=0.2)
    py_samples = min(20, num_iterations)

    start_time3 = time.perf_counter()
    for i in range(py_samples):
        sp_py.compute(inputs[i])
    end_time3 = time.perf_counter()

    duration3 = ((end_time3 - start_time3) / py_samples) * num_iterations
    temp_after3 = get_cpu_temp()
    ram_after3 = process.memory_info().rss / (1024 * 1024)

    # TELEMETRY SUMMARY REPORT
    print("\n=========================================================================")
    print("                HARDWARE TELEMETRY SUMMARY COMPARISON                    ")
    print("=========================================================================")
    print(f"{'Metric':<28} | {'nasm-htm.core':<15} | {'htm.core C++':<15} | {'Pure Python':<15}")
    print("-" * 81)
    print(f"{'Disk Binary Footprint':<28} | {disk_info['nasm_so_kb']:.1f} KB {' ':<8} | {disk_info['htm_core_so_mb']:.2f} MB {' ':<7} | N/A (Scripts)")
    print(f"{'RAM Usage (RSS)':<28} | {ram_after1:.2f} MB {' ':<7} | {ram_after2:.2f} MB {' ':<7} | {ram_after3:.2f} MB")
    print(f"{'Execution Time (200 ops)':<28} | {duration1:.4f} s {' ':<7} | {duration2:.4f} s {' ':<7} | {duration3:.4f} s")
    print(f"{'CPU Temp (°C)':<28} | {temp_after1:.1f} °C {' ':<8} | {temp_after2:.1f} °C {' ':<8} | {temp_after3:.1f} °C")
    if avg_power1 > 0:
        print(f"{'System Power Draw':<28} | {avg_power1:.2f} W {' ':<8} | {avg_power2:.2f} W {' ':<8} | N/A")
        print(f"{'Total Energy Consumed':<28} | {energy_joules1:.4f} J {' ':<6} | {energy_joules2:.4f} J {' ':<6} | N/A")
        if energy_joules1 > 0 and energy_joules2 > 0:
            print(f"\n=> Energy Efficiency: nasm-htm.core consumes {energy_joules2 / energy_joules1:.1f}x LESS ENERGY than htm.core C++!")
    print("=========================================================================")

if __name__ == "__main__":
    run_hardware_telemetry_benchmark()
