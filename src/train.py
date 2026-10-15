import argparse
import os
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, MambaForCausalLM, get_cosine_schedule_with_warmup
from datasets import load_dataset
import wandb

from src.model import ForkMambaChooser
from src.dataset import ForkMambaDataset, ForkMambaDataCollator
from src.metrics import compute_step_metrics


def parse_args():
    parser = argparse.ArgumentParser(description="Train ForkMambaChooser on SargeDev/jev-distill-corpus-v3")
    parser.add_argument("--model_name", type=str, default="state-spaces/mamba-130m-hf")
    parser.add_argument("--dataset_name", type=str, default="SargeDev/jev-distill-corpus-v3")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--micro_batch_size", type=int, default=8)
    parser.add_argument("--grad_accum_steps", type=int, default=16)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--warmup_ratio", type=float, default=0.05)
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--save_dir", type=str, default="./checkpoints")
    parser.add_argument("--wandb_project", type=str, default="fork-mamba-chooser")
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Executing training on device: {device}")

    # Initialize W&B
    wandb.init(project=args.wandb_project, config=vars(args))

    # Tokenizer Setup
    print("Loading tokenizer and extending special tokens...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    custom_tokens = {"additional_special_tokens": [
        "<noul/>", "<score/>", "<choice/>",
        "<answer/>", "<question>", "<question/>",
        "<state>", "<state/>"
    ]}
    tokenizer.add_special_tokens(custom_tokens)

    # Backbone Model Setup
    print("Loading base Mamba model...")
    hf_model = MambaForCausalLM.from_pretrained(args.model_name, torch_dtype=torch.float32)
    hf_model.resize_token_embeddings(len(tokenizer))

    # ForkMambaChooser & PEFT Configuration
    model = ForkMambaChooser(hf_model, tokenizer=tokenizer)
    model.setup_peft(r=args.lora_r, alpha=args.lora_alpha, target_modules=("in_proj", "x_proj", "dt_proj"))
    model.to(device)

    # Dataset & DataLoader
    print(f"Loading dataset: {args.dataset_name}...")
    raw_ds = load_dataset(args.dataset_name)
    train_raw = raw_ds["train"] if "train" in raw_ds else raw_ds

    train_dataset = ForkMambaDataset(train_raw)
    collator = ForkMambaDataCollator(tokenizer)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.micro_batch_size,
        shuffle=True,
        collate_fn=collator,
        pin_memory=True if device.type == "cuda" else False,
        num_workers=2
    )

    # Optimization Setup
    total_steps = (len(train_loader) // args.grad_accum_steps) * args.epochs
    warmup_steps = int(total_steps * args.warmup_ratio)

    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=0.01)
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))

    os.makedirs(args.save_dir, exist_ok=True)
    global_step = 0
    model.train()

    print(f"\n--- Training Active ({args.epochs} Epochs | Total Steps: {total_steps}) ---")
    for epoch in range(args.epochs):
        running_loss = 0.0
        optimizer.zero_grad(set_to_none=True)

        for step, batch in enumerate(train_loader):
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}

            with torch.amp.autocast(device_type=device.type, dtype=torch.float16, enabled=(device.type == "cuda")):
                outputs = model(**batch)
                loss = outputs["loss"] / args.grad_accum_steps

            scaler.scale(loss).backward()
            running_loss += loss.item() * args.grad_accum_steps

            if (step + 1) % args.grad_accum_steps == 0 or (step + 1) == len(train_loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1

                # Metric Logging
                avg_step_loss = running_loss / args.grad_accum_steps
                metrics_dict = compute_step_metrics(
                    loss=avg_step_loss,
                    probs=outputs["probs"],
                    targets=batch["labels"],
                    model=model
                )
                metrics_dict["train/learning_rate"] = scheduler.get_last_lr()[0]
                metrics_dict["train/epoch"] = epoch + (step / len(train_loader))

                wandb.log(metrics_dict, step=global_step)

                if global_step % 25 == 0:
                    print(
                        f"Epoch [{epoch+1}/{args.epochs}] | Step [{global_step}/{total_steps}] "
                        f"| Loss: {avg_step_loss:.4f} | Top1 Acc: {metrics_dict['train/top1_accuracy'] * 100:.1f}% "
                        f"| KL Div: {metrics_dict['train/kl_divergence']:.4f}"
                    )
                running_loss = 0.0

        # Save Checkpoint
        checkpoint_path = os.path.join(args.save_dir, f"mamba_chooser_epoch_{epoch+1}.pt")
        torch.save(model.trainable_state_dict(), checkpoint_path)
        print(f"--> Saved checkpoint: {checkpoint_path}")

    wandb.finish()
    print("Training finished successfully!")


if __name__ == "__main__":
    main()
