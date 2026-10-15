"""
Detailed profiler to analyze where time is being consumed in the model.
Breaks down the forward pass into individual components.
"""
import torch
from transformers import AutoTokenizer, MambaForCausalLM
from src.model import ForkMambaChooser
import time
import argparse


def detailed_profile():
    parser = argparse.ArgumentParser(description="Detailed profiling of model components")
    parser.add_argument("--model_name", type=str, default="state-spaces/mamba-130m-hf")
    parser.add_argument("--batch_size", type=int, default=12)
    parser.add_argument("--num_choices", type=int, default=4)
    parser.add_argument("--seq_len_q", type=int, default=512)
    parser.add_argument("--seq_len_c", type=int, default=32)
    parser.add_argument("--num_iterations", type=int, default=50)
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
    print(f"\n=== Profiling Configuration ===")
    print(f"Batch size: {args.batch_size}")
    print(f"Num choices: {args.num_choices}")
    print(f"Sequence lengths: Q={args.seq_len_q}, C={args.seq_len_c}")
    print(f"Num iterations: {args.num_iterations}\n")

    q_ids = torch.randint(0, len(tokenizer), (args.batch_size, args.seq_len_q), device=device)
    q_mask = torch.ones(args.batch_size, args.seq_len_q, device=device)
    c_ids = torch.randint(0, len(tokenizer), (args.batch_size, args.num_choices, args.seq_len_c), device=device)
    c_mask = torch.ones(args.batch_size, args.num_choices, args.seq_len_c, device=device)
    valid = torch.ones(args.batch_size, args.num_choices, dtype=torch.bool, device=device)

    # Warmup
    print("Warming up...")
    with torch.no_grad():
        for _ in range(5):
            _ = model(q_ids=q_ids, q_mask=q_mask, c_ids=c_ids, c_mask=c_mask, valid=valid)

    # Profile individual components
    print("\n=== Component-wise Profiling ===")

    timings = {
        'embed_q': [],
        'run_backbone_q': [],
        'fork_states': [],
        'embed_c': [],
        'run_backbone_c': [],
        'merge_states': [],
        'embed_ans': [],
        'run_backbone_ans': [],
        'scoring': [],
        'softmax': [],
        'total': []
    }

    with torch.no_grad():
        for i in range(args.num_iterations):
            torch.cuda.synchronize()

            # Total forward pass
            start_total = time.time()

            # Embed question
            start = time.time()
            q_embeds = model._embed_ids(q_ids)
            torch.cuda.synchronize()
            timings['embed_q'].append((time.time() - start) * 1000)

            # Run backbone on question
            start = time.time()
            _, q_states = model._run_backbone(q_embeds, layer_states=None, mask=q_mask)
            torch.cuda.synchronize()
            timings['run_backbone_q'].append((time.time() - start) * 1000)

            # Fork states
            start = time.time()
            forked_states = model._fork_states(q_states, args.num_choices)
            torch.cuda.synchronize()
            timings['fork_states'].append((time.time() - start) * 1000)

            # Embed choices
            start = time.time()
            flat_c_ids = c_ids.reshape(args.batch_size * args.num_choices, args.seq_len_c)
            flat_c_mask = c_mask.reshape(args.batch_size * args.num_choices, args.seq_len_c)
            c_embeds = model._embed_ids(flat_c_ids)
            torch.cuda.synchronize()
            timings['embed_c'].append((time.time() - start) * 1000)

            # Run backbone on choices
            start = time.time()
            c_hidden, c_states = model._run_backbone(c_embeds, layer_states=forked_states, mask=flat_c_mask)
            torch.cuda.synchronize()
            timings['run_backbone_c'].append((time.time() - start) * 1000)

            # Merge states
            start = time.time()
            B, N, Lc = c_ids.shape
            merged_states = model._merge_states(c_states, valid, B, N)
            torch.cuda.synchronize()
            timings['merge_states'].append((time.time() - start) * 1000)

            # Embed answer
            start = time.time()
            ans_ids = torch.full((args.batch_size, 1), model.ans_token_id, dtype=torch.long, device=q_ids.device)
            ans_embeds = model._embed_ids(ans_ids)
            torch.cuda.synchronize()
            timings['embed_ans'].append((time.time() - start) * 1000)

            # Run backbone on answer
            start = time.time()
            ans_hidden, _ = model._run_backbone(ans_embeds, layer_states=merged_states)
            torch.cuda.synchronize()
            timings['run_backbone_ans'].append((time.time() - start) * 1000)

            # Scoring
            start = time.time()
            answer_repr = ans_hidden[:, 0]
            choice_reprs = c_hidden[torch.arange(B * N, device=c_hidden.device),
                                    (flat_c_mask.sum(dim=1).long() - 1).clamp(min=0)].view(B, N, -1)
            scores = torch.einsum("bd,de,bne->bn", answer_repr, model.W, choice_reprs)
            logits = scores * model.logit_scale.exp() / (model.hidden_dim ** 0.5)
            logits = logits.masked_fill(~valid, -1e4)
            torch.cuda.synchronize()
            timings['scoring'].append((time.time() - start) * 1000)

            # Softmax
            start = time.time()
            probs = torch.nn.functional.softmax(logits, dim=-1)
            torch.cuda.synchronize()
            timings['softmax'].append((time.time() - start) * 1000)

            torch.cuda.synchronize()
            timings['total'].append((time.time() - start_total) * 1000)

    # Calculate statistics
    print("\n=== Timing Statistics (average over {} iterations) ===".format(args.num_iterations))
    print(f"{'Component':<25} {'Time (ms)':<12} {'% of Total':<12}")
    print("-" * 50)

    avg_total = sum(timings['total']) / len(timings['total'])

    for component in ['embed_q', 'run_backbone_q', 'fork_states', 'embed_c',
                     'run_backbone_c', 'merge_states', 'embed_ans',
                     'run_backbone_ans', 'scoring', 'softmax']:
        avg_time = sum(timings[component]) / len(timings[component])
        percent = (avg_time / avg_total) * 100
        print(f"{component:<25} {avg_time:>10.2f}    {percent:>10.1f}%")

    print("-" * 50)
    print(f"{'TOTAL':<25} {avg_total:>10.2f}    {100.0:>10.1f}%")

    # Identify bottlenecks
    print(f"\n=== Bottleneck Analysis ===")
    component_times = {k: sum(v)/len(v) for k, v in timings.items() if k != 'total'}
    sorted_components = sorted(component_times.items(), key=lambda x: x[1], reverse=True)

    for i, (component, time_ms) in enumerate(sorted_components[:3]):
        percent = (time_ms / avg_total) * 100
        print(f"{i+1}. {component}: {time_ms:.2f}ms ({percent:.1f}% of total)")

    # Backbone analysis
    backbone_time = (sum(timings['run_backbone_q']) + sum(timings['run_backbone_c']) +
                     sum(timings['run_backbone_ans'])) / len(timings['total'])
    backbone_percent = (backbone_time / avg_total) * 100
    print(f"\nTotal backbone time: {backbone_time:.2f}ms ({backbone_percent:.1f}% of total)")

    # Throughput
    throughput = args.batch_size / (avg_total / 1000)
    print(f"\n=== Throughput ===")
    print(f"Throughput: {throughput:.2f} samples/sec")
    print(f"Time per sample: {avg_total / args.batch_size:.2f} ms")


if __name__ == "__main__":
    detailed_profile()
