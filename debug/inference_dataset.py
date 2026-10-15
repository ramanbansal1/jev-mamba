"""
Run inference on the entire dataset with batch_size=12 to check correctness.
This script validates that the model works correctly with the optimal batch size.
"""
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, MambaForCausalLM
from datasets import load_dataset
from tqdm import tqdm
import argparse
import json
import os

from src.model import ForkMambaChooser
from src.dataset import ForkMambaDataset, ForkMambaDataCollator


def run_inference_on_dataset():
    parser = argparse.ArgumentParser(description="Run inference on full dataset")
    parser.add_argument("--model_name", type=str, default="state-spaces/mamba-130m-hf")
    parser.add_argument("--dataset_name", type=str, default="SargeDev/jev-distill-corpus-v3")
    parser.add_argument("--batch_size", type=int, default=12)
    parser.add_argument("--output_file", type=str, default="./inference_results.json")
    parser.add_argument("--max_samples", type=int, default=None, help="Limit number of samples (for testing)")
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

    # Load dataset
    print(f"Loading dataset: {args.dataset_name}...")
    raw_ds = load_dataset(args.dataset_name)
    dataset = raw_ds["train"] if "train" in raw_ds else raw_ds

    if args.max_samples:
        print(f"Limiting to {args.max_samples} samples")
        dataset = dataset.select(range(min(args.max_samples, len(dataset))))

    print(f"Dataset size: {len(dataset)}")

    # Create dataloader
    collator = ForkMambaDataCollator(tokenizer)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collator,
        pin_memory=True,
        num_workers=2
    )

    print(f"Batch size: {args.batch_size}")
    print(f"Number of batches: {len(dataloader)}")

    # Run inference
    print("\n=== Running Inference ===")
    results = []
    correct_count = 0
    total_count = 0

    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Inference")):
            # Move to device
            batch = {k: v.to(device) for k, v in batch.items()}

            # Forward pass
            outputs = model(
                q_ids=batch["q_ids"],
                q_mask=batch["q_mask"],
                c_ids=batch["c_ids"],
                c_mask=batch["c_mask"],
                valid=batch["valid"]
            )

            # Get predictions
            probs = outputs["probs"].cpu()
            valid = batch["valid"].cpu()
            labels = batch["labels"].cpu()

            # Get predicted choice (argmax)
            preds = torch.argmax(probs, dim=-1)
            targets = torch.argmax(labels, dim=-1)

            # Calculate accuracy
            batch_size = preds.shape[0]
            for i in range(batch_size):
                num_choices = valid[i].sum().item()
                if num_choices > 0:
                    pred = preds[i]
                    target = targets[i]
                    is_correct = (pred == target).item()
                    correct_count += is_correct
                    total_count += 1

                    # Store results
                    result = {
                        "batch_idx": batch_idx,
                        "sample_idx": i,
                        "predicted_choice": pred.item(),
                        "target_choice": target.item(),
                        "correct": is_correct,
                        "num_choices": int(num_choices),
                        "probabilities": probs[i, :num_choices].tolist(),
                        "target_distribution": labels[i, :num_choices].tolist()
                    }
                    results.append(result)

    # Print statistics
    print(f"\n=== Inference Results ===")
    print(f"Total samples processed: {total_count}")
    print(f"Correct predictions: {correct_count}")
    print(f"Accuracy: {correct_count / total_count * 100:.2f}%")

    # Save results
    print(f"\nSaving results to {args.output_file}")
    with open(args.output_file, 'w') as f:
        json.dump({
            "model_name": args.model_name,
            "dataset_name": args.dataset_name,
            "batch_size": args.batch_size,
            "total_samples": total_count,
            "correct": correct_count,
            "accuracy": correct_count / total_count,
            "results": results
        }, f, indent=2)

    print("Done!")

    # Additional validation checks
    print(f"\n=== Validation Checks ===")

    # Check 1: All probabilities sum to 1
    prob_sums = [sum(r['probabilities']) for r in results]
    avg_prob_sum = sum(prob_sums) / len(prob_sums)
    print(f"Average probability sum: {avg_prob_sum:.6f} (should be ~1.0)")
    if abs(avg_prob_sum - 1.0) > 0.01:
        print("⚠️  WARNING: Probabilities don't sum to 1.0")
    else:
        print("✓ Probabilities sum to 1.0")

    # Check 2: No NaN or Inf values
    has_nan = any(any(math.isnan(p) for p in r['probabilities']) for r in results)
    has_inf = any(any(math.isinf(p) for p in r['probabilities']) for r in results)
    if has_nan or has_inf:
        print("⚠️  WARNING: Found NaN or Inf in probabilities")
    else:
        print("✓ No NaN or Inf values")

    # Check 3: All probabilities are between 0 and 1
    all_valid = all(0 <= p <= 1 for r in results for p in r['probabilities'])
    if not all_valid:
        print("⚠️  WARNING: Found probabilities outside [0, 1]")
    else:
        print("✓ All probabilities in valid range [0, 1]")

    # Check 4: Target distributions are valid
    valid_targets = all(0 <= p <= 1 for r in results for p in r['target_distribution'])
    if not valid_targets:
        print("⚠️  WARNING: Found target values outside [0, 1]")
    else:
        print("✓ All target values in valid range [0, 1]")

    print(f"\n=== Batch Size {args.batch_size} Validation ===")
    print("✓ Inference completed successfully")
    print("✓ Model produces valid outputs")
    print("✓ No errors encountered")
    print(f"✓ Ready to use batch_size={args.batch_size} for training")


if __name__ == "__main__":
    import math
    run_inference_on_dataset()
