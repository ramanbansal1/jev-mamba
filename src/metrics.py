import torch
import torch.nn as nn


def compute_step_metrics(loss: float, probs: torch.Tensor, targets: torch.Tensor, model: nn.Module) -> dict:
    """
    Computes metrics for checking model learning progress.
    Returns a dictionary designed for direct `wandb.log(...)` ingestion.
    """
    with torch.no_grad():
        p = probs.detach().cpu().float().clamp(min=1e-7)
        y = targets.detach().cpu().float().clamp(min=1e-7)

        # 1. Top-1 Argmax Accuracy
        pred_argmax = torch.argmax(p, dim=-1)
        target_argmax = torch.argmax(y, dim=-1)
        top1_acc = (pred_argmax == target_argmax).float().mean().item()

        # 2. KL Divergence: sum(y * log(y / p))
        kl_div = (y * torch.log(y / p)).sum(dim=-1).mean().item()

        # 3. Prediction Entropy: -sum(p * log(p))
        pred_entropy = -(p * torch.log(p)).sum(dim=-1).mean().item()

        # 4. Target Entropy: -sum(y * log(y))
        target_entropy = -(y * torch.log(y)).sum(dim=-1).mean().item()

        # 5. Temperature Scale
        current_scale = model.logit_scale.exp().item()

    return {
        "train/loss": float(loss),
        "train/top1_accuracy": top1_acc,
        "train/kl_divergence": kl_div,
        "train/pred_entropy": pred_entropy,
        "train/target_entropy": target_entropy,
        "train/logit_scale": current_scale
    }
