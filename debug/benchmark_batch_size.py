"""
Batch size benchmarking script to find optimal batch size for max performance.
Tests different batch sizes and measures throughput and memory usage.
"""
import torch
from transformers import AutoTokenizer, MambaForCausalLM
from src.model import ForkMambaChooser
import time
import argparse


def benchmark_batch_sizes():
    parser = argparse.ArgumentParser(description="Benchmark different batch sizes")
    parser.add_argument("--model_name", type=str, default="state-spaces/mamba-130m-hf")
    parser.add_argument("--num_choices", type=int, default=4)
    parser.add_argument("--seq_len_q", type=int, default=512)
    parser.add_argument("--seq_len_c", type=int, default=32)
    parser.add_argument("--max_batch_size", type=int, default=16, help="Maximum batch size to test")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load model
    print("Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    custom_tokens = {"additional_special_tokens": [
        "<noul/>", "<score/>", "<choice/>",
        "<answer/>", "<question>", "<question/>",
        "<state>", "<state/>"
    ]}
    tokenizer.add_special_tokens(custom_tokens)

    hf_model = MambaForCausalLM.from_pretrained(args.model_name, torch_dtype=torch.float32)
    hf_model.resize_token_embeddings(len(tokenizer))
    hf_model = hf_model.to(device)

    model = ForkMambaChooser(hf_model, tokenizer=tokenizer)
    model.setup_peft(r=16, alpha=32, target_modules=("in_proj", "x_proj", "dt_proj"))
    model = model.to(device)
    model.eval()

    print(f"\n=== Batch Size Benchmark ===")
    print(f"Sequence lengths: Q={args.seq_len_q}, C={args.seq_len_c}")
    print(f"Num choices: {args.num_choices}")
    print(f"Testing batch sizes from 1 to {args.max_batch_size}\n")

    results = []

    for batch_size in range(1, args.max_batch_size + 1):
        try:
            # Clear cache
            torch.cuda.empty_cache()

            # Create dummy input
            q_ids = torch.randint(0, len(tokenizer), (batch_size, args.seq_len_q), device=device)
            q_mask = torch.ones(batch_size, args.seq_len_q, device=device)
            c_ids = torch.randint(0, len(tokenizer), (batch_size, args.num_choices, args.seq_len_c), device=device)
            c_mask = torch.ones(batch_size, args.num_choices, args.seq_len_c, device=device)
            valid = torch.ones(batch_size, args.num_choices, dtype=torch.bool, device=device)

            # Warmup
            with torch.no_grad():
                for _ in range(3):
                    _ = model(q_ids=q_ids, q_mask=q_mask, c_ids=c_ids, c_mask=c_mask, valid=valid)

            # Time forward pass
            torch.cuda.synchronize()
            start_time = time.time()

            num_iterations = 20
            with torch.no_grad():
                for _ in range(num_iterations):
                    outputs = model(q_ids=q_ids, q_mask=q_mask, c_ids=c_ids, c_mask=c_mask, valid=valid)

            torch.cuda.synchronize()
            end_time = time.time()

            avg_time = (end_time - start_time) / num_iterations
            time_per_sample = avg_time / batch_size
            throughput = batch_size / avg_time

            # Memory usage
            mem_allocated = torch.cuda.memory_allocated() / 1e9
            mem_reserved = torch.cuda.memory_reserved() / 1e9

            results.append({
                'batch_size': batch_size,
                'avg_time_ms': avg_time * 1000,
                'time_per_sample_ms': time_per_sample * 1000,
                'throughput': throughput,
                'mem_allocated_gb': mem_allocated,
                'mem_reserved_gb': mem_reserved,
            })

            print(f"Batch {batch_size:2d}: {avg_time*1000:7.2f}ms/batch, {time_per_sample*1000:7.2f}ms/sample, "
                  f"{throughput:6.2f} samples/s, {mem_allocated:5.2f}GB allocated")

        except RuntimeError as e:
            if "out of memory" in str(e):
                print(f"Batch {batch_size:2d}: OOM (memory exhausted)")
                break
            else:
                raise

    # Find optimal batch size
    print(f"\n=== Analysis ===")
    if results:
        # Find batch size with best throughput
        best_throughput = max(results, key=lambda x: x['throughput'])
        print(f"Best throughput: batch_size={best_throughput['batch_size']}, "
              f"{best_throughput['throughput']:.2f} samples/s")

        # Find batch size with best time per sample
        best_time_per_sample = min(results, key=lambda x: x['time_per_sample_ms'])
        print(f"Best time per sample: batch_size={best_time_per_sample['batch_size']}, "
              f"{best_time_per_sample['time_per_sample_ms']:.2f} ms/sample")

        # Show efficiency (throughput improvement relative to batch size 1)
        baseline = results[0]
        for r in results:
            speedup = r['throughput'] / baseline['throughput']
            efficiency = speedup / r['batch_size'] * 100
            print(f"Batch {r['batch_size']:2d}: {speedup:5.2f}x speedup, {efficiency:5.1f}% efficiency")

        # Recommendations
        print(f"\n=== Recommendations ===")
        print(f"1. Current training uses batch_size=1")
        print(f"2. Recommended batch_size for max throughput: {best_throughput['batch_size']}")
        print(f"3. Memory usage at recommended batch: {best_throughput['mem_allocated_gb']:.2f} GB")
        print(f"4. Expected speedup: {best_throughput['throughput'] / baseline['throughput']:.2f}x")

        # Calculate FLOPs for optimal batch
        try:
            from thop import profile
            q_ids = torch.randint(0, len(tokenizer), (best_throughput['batch_size'], args.seq_len_q), device=device)
            q_mask = torch.ones(best_throughput['batch_size'], args.seq_len_q, device=device)
            c_ids = torch.randint(0, len(tokenizer), (best_throughput['batch_size'], args.num_choices, args.seq_len_c), device=device)
            c_mask = torch.ones(best_throughput['batch_size'], args.num_choices, args.seq_len_c, device=device)
            valid = torch.ones(best_throughput['batch_size'], args.num_choices, dtype=torch.bool, device=device)

            flop_count, params = profile(model, inputs=(q_ids, q_mask, c_ids, c_mask, valid), verbose=False)
            print(f"5. FLOPs at optimal batch: {flop_count / 1e9:.2f} GFLOPs")
            print(f"6. Achieved compute: {flop_count / 1e9 / best_throughput['avg_time_ms'] * 1000:.2f} GFLOP/s")
        except:
            pass


if __name__ == "__main__":
    benchmark_batch_sizes()
