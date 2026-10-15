"""
Torch Profiler script to identify bottlenecks in the training pipeline.
This creates a modified version of train.py with profiling enabled.
Run this separately from the main training to profile a few steps.
"""
import argparse
import os
import torch
from torch.utils.data import DataLoader
from torch.profiler import profile, record_function, ProfilerActivity
from transformers import AutoTokenizer, MambaForCausalLM, get_cosine_schedule_with_warmup
from datasets import load_dataset
import wandb
from tqdm import tqdm
from accelerate import Accelerator

from src.model import ForkMambaChooser
from src.dataset import ForkMambaDataset, ForkMambaDataCollator
from src.metrics import compute_step_metrics


def parse_args():
    parser = argparse.ArgumentParser(description="Profile training with torch profiler")
    parser.add_argument("--model_name", type=str, default="state-spaces/mamba-130m-hf")
    parser.add_argument("--dataset_name", type=str, default="SargeDev/jev-distill-corpus-v3")
    parser.add_argument("--micro_batch_size", type=int, default=1)
    parser.add_argument("--grad_accum_steps", type=int, default=128)
    parser.add_argument("--profiler_steps", type=int, default=10, help="Number of steps to profile")
    parser.add_argument("--warmup_steps", type=int, default=3, help="Warmup steps before profiling")
    parser.add_argument("--output_dir", type=str, default="./profiler_output", help="Output directory for profiler results")
    return parser.parse_args()


def main():
    args = parse_args()

    # Initialize Accelerator
    accelerator = Accelerator(
        mixed_precision="fp16",
        gradient_accumulation_steps=args.grad_accum_steps,
        device_placement=True
    )

    print(f"Profiling with {accelerator.num_processes} GPU(s)")

    # Tokenizer Setup
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    custom_tokens = {"additional_special_tokens": [
        "<noul/>", "<score/>", "<choice/>",
        "<answer/>", "<question>", "<question/>",
        "<state>", "<state/>"
    ]}
    tokenizer.add_special_tokens(custom_tokens)

    # Backbone Model Setup
    print("Loading model...")
    hf_model = MambaForCausalLM.from_pretrained(args.model_name, torch_dtype=torch.float32)
    hf_model.resize_token_embeddings(len(tokenizer))

    model = ForkMambaChooser(hf_model, tokenizer=tokenizer)
    model.setup_peft(r=16, alpha=32, target_modules=("in_proj", "x_proj", "dt_proj"))

    # Dataset & DataLoader
    print("Loading dataset...")
    raw_ds = load_dataset(args.dataset_name)
    train_raw = raw_ds["train"] if "train" in raw_ds else raw_ds

    train_dataset = ForkMambaDataset(train_raw)
    collator = ForkMambaDataCollator(tokenizer)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.micro_batch_size,
        shuffle=True,
        collate_fn=collator,
        pin_memory=True,
        num_workers=2
    )

    # Optimization Setup
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=3e-4, weight_decay=0.01)

    # Prepare with accelerator
    model, optimizer, train_loader = accelerator.prepare(model, optimizer, train_loader)

    model.train()
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"\n=== Profiling Configuration ===")
    print(f"Profiler steps: {args.profiler_steps}")
    print(f"Warmup steps: {args.warmup_steps}")
    print(f"Output directory: {args.output_dir}")
    print(f"\nStarting warmup...")

    # Warmup
    for step, batch in enumerate(train_loader):
        if step >= args.warmup_steps:
            break

        with accelerator.accumulate(model):
            outputs = model(**batch)
            loss = outputs["loss"]
            accelerator.backward(loss)
            accelerator.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        print(f"  Warmup step {step + 1}/{args.warmup_steps} completed")

    print("\n=== Starting Profiling ===")

    # Profiling
    with profile(
        activities=[
            ProfilerActivity.CPU,
            ProfilerActivity.CUDA,
        ],
        record_shapes=True,
        profile_memory=True,
        with_stack=True,
        with_flops=True,
        with_modules=True,
        on_trace_ready=lambda trace: trace.save(f"{args.output_dir}/trace_{trace.step_num}.json"),
        schedule=torch.profiler.schedule(wait=1, warmup=1, active=args.profiler_steps, repeat=1),
    ) as prof:

        for step, batch in enumerate(train_loader):
            if step >= args.warmup_steps + args.profiler_steps:
                break

            with accelerator.accumulate(model):
                with record_function("forward_pass"):
                    outputs = model(**batch)
                    loss = outputs["loss"]

                with record_function("backward_pass"):
                    accelerator.backward(loss)

                with record_function("optimizer_step"):
                    accelerator.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)

            prof.step()

            if accelerator.is_main_process:
                if step >= args.warmup_steps:
                    print(f"  Profiling step {step - args.warmup_steps + 1}/{args.profiler_steps}")

    # Print profiler summary
    print("\n=== Profiler Summary ===")
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=20))

    # Print top operations by CUDA time
    print("\n=== Top 20 CUDA Operations ===")
    print(prof.key_averages(group_by_input_shape=True).table(sort_by="cuda_time_total", row_limit=20))

    # Export detailed trace
    if accelerator.is_main_process:
        print(f"\n=== Exporting Trace ===")
        prof.export_chrome_trace(f"{args.output_dir}/trace.json")
        print(f"Trace saved to {args.output_dir}/trace.json")
        print(f"Open chrome://tracing in Chrome browser to view the trace")

        # Export memory summary
        print("\n=== Memory Summary ===")
        print(prof.key_averages().table(sort_by="self_cuda_memory_usage", row_limit=20))

        # Export stats
        prof.export_stacks(f"{args.output_dir}/profiler_stacks.txt", "self_cuda_time_total")
        print(f"Stack traces saved to {args.output_dir}/profiler_stacks.txt")

    print("\n=== Profiling Complete ===")
    print(f"Results saved to {args.output_dir}/")


if __name__ == "__main__":
    main()
