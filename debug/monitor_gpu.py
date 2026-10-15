"""
GPU monitoring script to track utilization, memory, and identify bottlenecks.
Run this in a separate terminal while training is running.
"""
import time
import argparse
from collections import deque

try:
    import pynvml
    PYNVML_AVAILABLE = True
except ImportError:
    PYNVML_AVAILABLE = False
    print("⚠️  pynvml not installed. Install with: pip install pynvml")


def monitor_gpu(interval=1.0, duration=None):
    if not PYNVML_AVAILABLE:
        print("Cannot monitor GPU without pynvml library")
        return

    pynvml.nvmlInit()
    device_count = pynvml.nvmlDeviceGetCount()

    print(f"=== GPU Monitoring ===")
    print(f"Found {device_count} GPU(s)")
    print(f"Monitoring interval: {interval}s")
    if duration:
        print(f"Total duration: {duration}s")
    print(f"Press Ctrl+C to stop\n")

    # History for averaging
    history = {
        'util': [deque(maxlen=60) for _ in range(device_count)],
        'mem_used': [deque(maxlen=60) for _ in range(device_count)],
        'mem_total': [deque(maxlen=60) for _ in range(device_count)],
        'temp': [deque(maxlen=60) for _ in range(device_count)],
        'power': [deque(maxlen=60) for _ in range(device_count)],
    }

    start_time = time.time()
    iteration = 0

    try:
        while True:
            if duration and (time.time() - start_time) > duration:
                break

            iteration += 1
            print(f"\n=== Iteration {iteration} (t={time.time()-start_time:.1f}s) ===")

            for i in range(device_count):
                handle = pynvml.nvmlDeviceGetHandleByIndex(i)

                # GPU name
                name = pynvml.nvmlDeviceGetName(handle)
                print(f"\nGPU {i}: {name.decode('utf-8')}")

                # Utilization
                util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                gpu_util = util.gpu
                mem_util = util.memory
                history['util'][i].append(gpu_util)

                # Memory
                mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                mem_used_gb = mem_info.used / 1e9
                mem_total_gb = mem_info.total / 1e9
                mem_percent = (mem_info.used / mem_info.total) * 100
                history['mem_used'][i].append(mem_used_gb)
                history['mem_total'][i].append(mem_total_gb)

                # Temperature
                try:
                    temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
                    history['temp'][i].append(temp)
                except:
                    temp = "N/A"

                # Power
                try:
                    power = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000  # Convert to watts
                    history['power'][i].append(power)
                except:
                    power = "N/A"

                # Clock speeds
                try:
                    clock_graphics = pynvml.nvmlDeviceGetClockInfo(handle, pynvml.NVML_CLOCK_GRAPHICS)
                    clock_memory = pynvml.nvmlDeviceGetClockInfo(handle, pynvml.NVML_CLOCK_MEM)
                except:
                    clock_graphics = "N/A"
                    clock_memory = "N/A"

                print(f"  GPU Utilization: {gpu_util}%")
                print(f"  Memory Utilization: {mem_util}%")
                print(f"  Memory Used: {mem_used_gb:.2f} GB / {mem_total_gb:.2f} GB ({mem_percent:.1f}%)")
                print(f"  Temperature: {temp}°C" if temp != "N/A" else f"  Temperature: {temp}")
                print(f"  Power: {power} W" if power != "N/A" else f"  Power: {power}")
                print(f"  Clock (GPU/Mem): {clock_graphics} MHz / {clock_memory} MHz")

                # Average stats
                if len(history['util'][i]) > 0:
                    avg_util = sum(history['util'][i]) / len(history['util'][i])
                    avg_mem = sum(history['mem_used'][i]) / len(history['mem_used'][i])
                    print(f"  [Avg last 60s] GPU: {avg_util:.1f}%, Mem: {avg_mem:.2f} GB")

                    # Bottleneck detection
                    if avg_util < 70:
                        print(f"  ⚠️  WARNING: Low GPU utilization ({avg_util:.1f}%)")
                        print(f"     Possible causes:")
                        print(f"     - CPU bottleneck (data loading)")
                        print(f"     - I/O bottleneck (disk/network)")
                        print(f"     - Too small batch size")
                        print(f"     - Inefficient model implementation")
                    elif avg_util > 90:
                        print(f"  ✓ Good GPU utilization ({avg_util:.1f}%)")

            time.sleep(interval)

    except KeyboardInterrupt:
        print("\n\n=== Monitoring stopped ===")

    # Print summary
    print("\n=== Summary Statistics (last 60 samples) ===")
    for i in range(device_count):
        if len(history['util'][i]) > 0:
            avg_util = sum(history['util'][i]) / len(history['util'][i])
            avg_mem = sum(history['mem_used'][i]) / len(history['mem_used'][i])
            print(f"\nGPU {i}:")
            print(f"  Average GPU utilization: {avg_util:.1f}%")
            print(f"  Average memory used: {avg_mem:.2f} GB")

    pynvml.nvmlShutdown()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Monitor GPU utilization")
    parser.add_argument("--interval", type=float, default=1.0, help="Monitoring interval in seconds")
    parser.add_argument("--duration", type=float, default=None, help="Total monitoring duration in seconds")
    args = parser.parse_args()

    monitor_gpu(interval=args.interval, duration=args.duration)
