# Debugging Tools for Training Analysis

This directory contains tools to profile and monitor your training pipeline without modifying the main training code.

## Tools Available

### 1. Model FLOPs and Timing Profiler (`profile_model.py`)
Calculates FLOPs, parameters, and timing information for the model.

**Usage:**
```bash
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
python debug/profile_model.py --batch_size 1 --num_choices 4 --seq_len_q 512 --seq_len_c 32
```

**What it measures:**
- Forward pass time per batch and per sample
- Total FLOPs (using thop library)
- Number of parameters
- GPU memory usage
- Time per GFLOP (efficiency metric)

**Install dependencies:**
```bash
pip install thop
```

### 2. Batch Size Benchmark (`benchmark_batch_size.py`) ⭐ NEW
Tests different batch sizes to find the optimal configuration for maximum throughput.

**Usage:**
```bash
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
python debug/benchmark_batch_size.py --max_batch_size 16 --seq_len_q 512 --seq_len_c 32
```

**What it measures:**
- Throughput for each batch size
- Time per sample for each batch size
- Memory usage for each batch size
- Speedup and efficiency compared to batch size 1
- **Recommended optimal batch size**

**Why this is important:**
- Your current training uses batch_size=1 which is likely too small
- This tool helps you find the sweet spot between memory and performance
- GPU has 14.56 GB but you're only using 0.54 GB - huge waste!

### 3. Simple GPU Monitor (`simple_gpu_monitor.py`) ⭐ NEW
Lightweight GPU monitor using nvidia-smi (no extra dependencies).

**Usage:**
```bash
# Run in a separate terminal while training is running
python debug/simple_gpu_monitor.py --interval 1.0

# Or run for a specific duration
python debug/simple_gpu_monitor.py --interval 0.5 --duration 300
```

**What it monitors:**
- GPU utilization (%)
- Memory usage (MB and %)
- Temperature (°C)
- Power consumption (W)
- Clock speeds (GPU and memory)
- **Bottleneck detection** - warns if GPU utilization is low (<70%)

**Advantages:**
- No extra dependencies (uses nvidia-smi)
- Works on any system with NVIDIA GPU
- Simple and reliable

### 4. Advanced GPU Monitor (`monitor_gpu.py`)
Real-time GPU monitoring with pynvml (more detailed statistics).

**Usage:**
```bash
python debug/monitor_gpu.py --interval 1.0
```

**Install dependencies:**
```bash
pip install pynvml
```

### 5. Torch Profiler (`torch_profiler.py`)
Detailed profiling of the training pipeline using PyTorch's built-in profiler.

**Usage:**
```bash
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
python debug/torch_profiler.py --profiler_steps 10 --warmup_steps 3 --output_dir ./profiler_output
```

**What it profiles:**
- Forward pass time
- Backward pass time
- Optimizer step time
- Detailed operation-level timing
- Memory usage per operation
- FLOPs per operation

## Quick Start Guide

### Step 1: Find optimal batch size ⭐ IMPORTANT
Your current training is very slow because batch_size=1. First, find the optimal batch size:

```bash
python debug/benchmark_batch_size.py --max_batch_size 16
```

This will tell you:
- Which batch size gives maximum throughput
- How much memory each batch size uses
- Expected speedup

### Step 2: Update launch scripts with optimal batch size
After finding the optimal batch size, update your launch scripts:
- Update `--micro_batch_size` to the recommended value
- Adjust `--grad_accum_steps` to maintain the same effective batch size

Example: If optimal batch_size=4 and current grad_accum=128:
```bash
--micro_batch_size 4
--grad_accum_steps 32  # (128 / 4)
```

### Step 3: Monitor GPU during training
```bash
# Terminal 1: Start training with new batch size
bash launch_multi_gpu.sh

# Terminal 2: Monitor GPU
python debug/simple_gpu_monitor.py --interval 1.0
```

This will show you:
- Real-time GPU utilization
- Memory usage
- Bottleneck warnings

### Step 4: Profile model (optional)
```bash
python debug/profile_model.py
```
This will tell you:
- How long each forward pass takes
- Total FLOPs
- Time per GFLOP (efficiency)

### Step 5: Profile training pipeline (if GPU utilization is still low)
If GPU monitor shows low utilization after optimizing batch size:
```bash
python debug/torch_profiler.py --profiler_steps 20 --warmup_steps 5
```

## Common Issues and Solutions

### Issue: GPU utilization is low (<70%)
**Possible causes:**
1. **Batch size too small** ⭐ MOST COMMON
   - **Solution**: Use `benchmark_batch_size.py` to find optimal batch size
   - Your current batch_size=1 is very inefficient

2. **CPU bottleneck** - Data loading is too slow
   - Solution: Increase `num_workers` in DataLoader (currently 2, try 4-8)
   - Solution: Reduce preprocessing in dataset
   - Solution: Use pinned memory (`pin_memory=True` - already enabled)

3. **I/O bottleneck** - Disk/network is too slow
   - Solution: Cache dataset in memory
   - Solution: Use faster storage (SSD)
   - Solution: Stream data more efficiently

4. **Communication overhead** (multi-GPU)
   - Solution: Increase batch size per GPU
   - Solution: Use gradient checkpointing

### Issue: Training is very slow
**Diagnosis steps:**
1. Run `benchmark_batch_size.py` - is batch size optimal?
2. Run `simple_gpu_monitor.py` - is GPU utilization high?
3. If GPU utilization is low, run `torch_profiler.py` to find bottleneck

### Issue: Out of memory
**Solutions:**
- Reduce micro_batch_size
- Reduce sequence lengths (max_q_len, max_c_len in data collator)
- Use gradient checkpointing
- Use a smaller model

## Performance Optimization Checklist

- [ ] Run `benchmark_batch_size.py` to find optimal batch size
- [ ] Update launch scripts with optimal batch size
- [ ] Monitor GPU with `simple_gpu_monitor.py` during training
- [ ] If GPU utilization < 70%, check:
  - [ ] Increase `num_workers` in DataLoader
  - [ ] Check if data loading is the bottleneck (use torch_profiler)
  - [ ] Consider caching dataset in memory
- [ ] If GPU utilization > 85% but still slow:
  - [ ] Profile with `torch_profiler.py` to find slow operations
  - [ ] Check if model implementation can be optimized

## Example Output Interpretation

**From benchmark_batch_size.py:**
```
Batch  1: 1558.93ms/batch, 1558.93ms/sample,  0.64 samples/s, 0.54GB allocated
Batch  2: 2100.45ms/batch, 1050.23ms/sample,  0.95 samples/s, 0.87GB allocated
Batch  4: 2800.12ms/batch,  700.03ms/sample,  1.43 samples/s, 1.45GB allocated
Batch  8: 4200.34ms/batch,  525.04ms/sample,  1.90 samples/s, 2.89GB allocated

Best throughput: batch_size=8, 1.90 samples/s
Expected speedup: 2.97x
```

This means:
- Increasing from batch_size=1 to batch_size=8 gives ~3x speedup
- Memory usage increases from 0.54GB to 2.89GB (still well under 14.56GB limit)
- You should update your launch script to use batch_size=8

## Updating requirements.txt

Add these to your requirements.txt if you want to use all tools:
```
thop>=0.1.1
pynvml>=11.4.0
```

Note:
- `simple_gpu_monitor.py` doesn't need any extra dependencies
- torch profiler is built into PyTorch

