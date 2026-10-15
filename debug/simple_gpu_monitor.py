"""
Simple GPU monitor without external dependencies.
Uses nvidia-smi command to monitor GPU utilization.
"""
import subprocess
import time
import argparse


def monitor_gpu_simple(interval=1.0, duration=None):
    print(f"=== Simple GPU Monitor ===")
    print(f"Using nvidia-smi to monitor GPU")
    print(f"Interval: {interval}s")
    if duration:
        print(f"Duration: {duration}s")
    print(f"Press Ctrl+C to stop\n")

    start_time = time.time()
    iteration = 0

    try:
        while True:
            if duration and (time.time() - start_time) > duration:
                break

            iteration += 1
            elapsed = time.time() - start_time

            # Run nvidia-smi
            result = subprocess.run(
                ['nvidia-smi', '--query-gpu=index,name,utilization.gpu,utilization.memory,memory.used,memory.total,temperature.gpu,power.draw,clocks.gr,clocks.mem',
                 '--format=csv,noheader,nounits'],
                capture_output=True,
                text=True
            )

            if result.returncode == 0:
                lines = result.stdout.strip().split('\n')
                print(f"\n=== Iteration {iteration} (t={elapsed:.1f}s) ===")

                for line in lines:
                    parts = [p.strip() for p in line.split(',')]
                    if len(parts) >= 10:
                        idx, name, gpu_util, mem_util, mem_used, mem_total, temp, power, clock_gpu, clock_mem = parts

                        print(f"\nGPU {idx}: {name}")
                        print(f"  GPU Util: {gpu_util}%")
                        print(f"  Mem Util: {mem_util}%")
                        print(f"  Memory: {mem_used} MB / {mem_total} MB ({int(mem_used)/int(mem_total)*100:.1f}%)")
                        print(f"  Temp: {temp}°C")
                        print(f"  Power: {power} W")
                        print(f"  Clocks: {clock_gpu} MHz / {clock_mem} MHz")

                        # Simple bottleneck detection
                        gpu_util_val = int(gpu_util)
                        if gpu_util_val < 70:
                            print(f"  ⚠️  WARNING: Low GPU utilization ({gpu_util_val}%)")
                            print(f"     Possible causes:")
                            print(f"     - CPU bottleneck (data loading)")
                            print(f"     - I/O bottleneck")
                            print(f"     - Batch size too small")
                        elif gpu_util_val > 85:
                            print(f"  ✓ Good GPU utilization ({gpu_util_val}%)")
            else:
                print("Error running nvidia-smi")

            time.sleep(interval)

    except KeyboardInterrupt:
        print("\n\n=== Monitoring stopped ===")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Simple GPU monitor using nvidia-smi")
    parser.add_argument("--interval", type=float, default=1.0, help="Monitoring interval in seconds")
    parser.add_argument("--duration", type=float, default=None, help="Total monitoring duration in seconds")
    args = parser.parse_args()

    monitor_gpu_simple(interval=args.interval, duration=args.duration)
