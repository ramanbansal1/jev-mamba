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

### 2. GPU Monitor (`monitor_gpu.py`)
Real-time GPU utilization, memory, temperature, and power monitoring.

**Usage:**
```bash
# Run in a separate terminal while training is running
python debug/monitor_gpu.py --interval 1.0

# Or run for a specific duration
python debug/monitor_gpu.py --interval 0.5 --duration 300
```

**What it monitors:**
- GPU utilization (%)
- Memory usage (GB and %)
- Temperature (°C)
- Power consumption (W)
- Clock speeds (GPU and memory)
- Average statistics over last 60 samples
- **Bottleneck detection** - warns if GPU utilization is low (<70%)

**Install dependencies:**
```bash
pip install pynvml
```

**Interpreting the output:**
- If GPU utilization is consistently low (<70%), you have a bottleneck:
  - **CPU bottleneck**: Data loading or preprocessing is too slow
  - **I/O bottleneck**: Disk or network is too slow
  - **Batch size too small**: Not enough work per GPU
  - **Inefficient model**: Model implementation can be optimized

### 3. Torch Profiler (`torch_profiler.py`)
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

**Output:**
- `trace.json` - Chrome trace file (view in chrome://tracing)
- `profiler_stacks.txt` - Stack traces for performance analysis
- Console output with top operations by CUDA time

**Viewing the trace:**
1. Open Chrome browser
2. Navigate to `chrome://tracing`
3. Click "Load"
4. Select `profiler_output/trace.json`

## Quick Start Guide

### Step 1: Profile the model (calculate FLOPs and timing)
```bash
python debug/profile_model.py
```
This will tell you:
- How long each forward pass takes
- Total FLOPs
- Whether the model fits in memory

### Step 2: Monitor GPU during training
```bash
# Terminal 1: Start training
bash launch_multi_gpu.sh

# Terminal 2: Monitor GPU
python debug/monitor_gpu.py --interval 1.0
```
This will show you:
- Real-time GPU utilization
- Memory usage
- Bottleneck warnings

### Step 3: Profile training pipeline (if needed)
If GPU monitor shows low utilization, use the torch profiler to find the bottleneck:
```bash
python debug/torch_profiler.py --profiler_steps 20 --warmup_steps 5
```
Then examine the trace in Chrome to see which operations are slow.

## Common Issues and Solutions

### Issue: GPU utilization is low (<70%)
**Possible causes:**
1. **CPU bottleneck** - Data loading is too slow
   - Solution: Increase `num_workers` in DataLoader
   - Solution: Reduce preprocessing in dataset
   - Solution: Use pinned memory (`pin_memory=True`)

2. **I/O bottleneck** - Disk/network is too slow
   - Solution: Cache dataset in memory
   - Solution: Use faster storage (SSD)
   - Solution: Stream data more efficiently

3. **Batch size too small**
   - Solution: Increase micro_batch_size (if memory allows)
   - Solution: Reduce gradient accumulation steps proportionally

4. **Communication overhead** (multi-GPU)
   - Solution: Increase batch size per GPU
   - Solution: Use gradient checkpointing

### Issue: Training is very slow
**Check with profiler:**
- Which operation takes the most time?
- Is it data loading or model forward/backward?
- Are there inefficient operations in the model?

### Issue: Out of memory
**Solutions:**
- Reduce micro_batch_size
- Reduce sequence lengths (max_q_len, max_c_len in data collator)
- Use gradient checkpointing
- Use a smaller model

## Updating requirements.txt

Add these to your requirements.txt if you want to use all tools:
```
thop>=0.1.1
pynvml>=11.4.0
```

Note: torch profiler is built into PyTorch, no extra installation needed.
