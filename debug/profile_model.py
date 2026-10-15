"""
Model profiling script to calculate FLOPs and time per data point.
Run this separately from training to profile the model.
"""
import torch
from transformers import AutoTokenizer, MambaForCausalLM
from src.model import ForkMambaChooser
import time
import argparse


def profile_model_flops():
    parser = argparse.ArgumentParser(description="Profile model FLOPs and timing")
    parser.add_argument("--model_name", type=str, default="state-spaces/mamba-130m-hf")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_choices", type=int, default=4)
    parser.add_argument("--seq_len_q", type=int, default=512)
    parser.add_argument("--seq_len_c", type=int, default=32)
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

    # Create dummy input
    print(f"\nCreating dummy batch:")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Num choices: {args.num_choices}")
    print(f"  Question seq len: {args.seq_len_q}")
    print(f"  Choice seq len: {args.seq_len_c}")

    q_ids = torch.randint(0, len(tokenizer), (args.batch_size, args.seq_len_q), device=device)
    q_mask = torch.ones(args.batch_size, args.seq_len_q, device=device)
    c_ids = torch.randint(0, len(tokenizer), (args.batch_size, args.num_choices, args.seq_len_c), device=device)
    c_mask = torch.ones(args.batch_size, args.num_choices, args.seq_len_c, device=device)
    valid = torch.ones(args.batch_size, args.num_choices, dtype=torch.bool, device=device)

    # Warmup
    print("\nWarming up...")
    with torch.no_grad():
        for _ in range(5):
            _ = model(q_ids=q_ids, q_mask=q_mask, c_ids=c_ids, c_mask=c_mask, valid=valid)

    # Time forward pass
    print("\nTiming forward pass...")
    torch.cuda.synchronize()
    start_time = time.time()

    num_iterations = 50
    with torch.no_grad():
        for _ in range(num_iterations):
            outputs = model(q_ids=q_ids, q_mask=q_mask, c_ids=c_ids, c_mask=c_mask, valid=valid)

    torch.cuda.synchronize()
    end_time = time.time()

    avg_time = (end_time - start_time) / num_iterations
    time_per_sample = avg_time / args.batch_size

    print(f"\n=== Timing Results ===")
    print(f"Average time per batch: {avg_time*1000:.2f} ms")
    print(f"Time per sample: {time_per_sample*1000:.2f} ms")
    print(f"Throughput: {args.batch_size / avg_time:.2f} samples/sec")

    # Calculate FLOPs using thop
    try:
        from thop import profile
        print("\n=== FLOPs Calculation ===")
        flop_count, params = profile(model, inputs=(q_ids, q_mask, c_ids, c_mask, valid), verbose=False)
        print(f"Total FLOPs per forward pass: {flop_count / 1e9:.2f} GFLOPs")
        print(f"Total parameters: {params / 1e6:.2f} M")
        print(f"FLOPs per sample: {flop_count / 1e9 / args.batch_size:.2f} GFLOPs")
        print(f"Time per GFLOP: {avg_time / (flop_count / 1e9):.4f} s")
    except ImportError:
        print("\n⚠️  thop not installed. Install with: pip install thop")
        print("To calculate FLOPs manually:")
        print(f"  - Use: flops = 6 * num_params * seq_len * batch_size")
        print(f"  - Or install thop for automatic calculation")

    # Memory usage
    if torch.cuda.is_available():
        print(f"\n=== Memory Usage ===")
        print(f"GPU memory allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
        print(f"GPU memory reserved: {torch.cuda.memory_reserved() / 1e9:.2f} GB")


if __name__ == "__main__":
    profile_model_flops()
