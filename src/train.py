import argparse
import os
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, MambaForCausalLM, get_cosine_schedule_with_warmup
from datasets import load_dataset
import wandb
from tqdm import tqdm
from accelerate import Accelerator

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

    # Initialize Accelerator for multi-GPU training
    accelerator = Accelerator(
        mixed_precision="fp16",
        device_placement=True
    )

    # Initialize W&B (only on main process)
    if accelerator.is_main_process:
        wandb.init(project=args.wandb_project, config=vars(args))
        print(f"Training with {accelerator.num_processes} GPU(s)")

    # Tokenizer Setup
    if accelerator.is_main_process:
        print("Loading tokenizer and extending special tokens...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    custom_tokens = {"additional_special_tokens": [
        "<noul/>", "<score/>", "<choice/>",
        "<answer/>", "<question>", "<question/>",
        "<state>", "<state/>"
    ]}
    tokenizer.add_special_tokens(custom_tokens)

    # Backbone Model Setup
    if accelerator.is_main_process:
        print("Loading base Mamba model...")
    hf_model = MambaForCausalLM.from_pretrained(args.model_name, torch_dtype=torch.float32)
    hf_model.resize_token_embeddings(len(tokenizer))

    # ForkMambaChooser & PEFT Configuration
    model = ForkMambaChooser(hf_model, tokenizer=tokenizer)
    model.setup_peft(r=args.lora_r, alpha=args.lora_alpha, target_modules=("in_proj", "x_proj", "dt_proj"))

    # Dataset & DataLoader
    if accelerator.is_main_process:
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
        pin_memory=True,
        num_workers=2
    )

    # Optimization Setup
    total_steps = (len(train_loader) // args.grad_accum_steps) * args.epochs
    warmup_steps = int(total_steps * args.warmup_ratio)

    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=0.01)
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)

    # Prepare with accelerator for multi-GPU
    model, optimizer, train_loader, scheduler = accelerator.prepare(
        model, optimizer, train_loader, scheduler
    )

    os.makedirs(args.save_dir, exist_ok=True)
    global_step = 0
    model.train()

    if accelerator.is_main_process:
        print(f"\n--- Training Active ({args.epochs} Epochs | Total Steps: {total_steps}) ---")

    for epoch in range(args.epochs):
        running_loss = 0.0
        optimizer.zero_grad(set_to_none=True)

        # Progress bar for steps within epoch (only on main process)
        if accelerator.is_main_process:
            pbar = tqdm(
                train_loader,
                desc=f"Epoch {epoch+1}/{args.epochs}",
                leave=False
            )

        for batch in train_loader:
            with accelerator.accumulate(model, accumulation_steps=args.grad_accum_steps):
                outputs = model(**batch)
                loss = outputs["loss"]

                accelerator.backward(loss)
                running_loss += loss.item()

                accelerator.clip_grad_norm_(model.parameters(), max_norm=1.0)

                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

                # Only log metrics at the end of accumulation steps
                if accelerator.sync_gradients:
                    if accelerator.is_main_process:
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
                        metrics_dict["train/epoch"] = epoch + (global_step / total_steps)

                        wandb.log(metrics_dict, step=global_step)

                        # Update progress bar with metrics
                        pbar.set_postfix({
                            "Step": f"{global_step}/{total_steps}",
                            "Loss": f"{avg_step_loss:.4f}",
                            "Top1": f"{metrics_dict['train/top1_accuracy'] * 100:.1f}%",
                            "KL": f"{metrics_dict['train/kl_divergence']:.4f}",
                            "LR": f"{metrics_dict['train/learning_rate']:.2e}"
                        })

                    running_loss = 0.0

        if accelerator.is_main_process:
            pbar.close()

        # Save Checkpoint (only on main process)
        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            checkpoint_path = os.path.join(args.save_dir, f"mamba_chooser_epoch_{epoch+1}.pt")
            unwrapped_model = accelerator.unwrap_model(model)
            torch.save(unwrapped_model.trainable_state_dict(), checkpoint_path)
            print(f"--> Saved checkpoint: {checkpoint_path}")

    if accelerator.is_main_process:
        wandb.finish()
        print("Training finished successfully!")


if __name__ == "__main__":
    main()
