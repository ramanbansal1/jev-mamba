import argparse
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, MambaForCausalLM

from src.model import ForkMambaChooser
from src.dataset import ForkMambaDataset, ForkMambaDataCollator


def parse_args():
    parser = argparse.ArgumentParser(description="Inference evaluation for ForkMambaChooser")
    parser.add_argument("--model_name", type=str, default="state-spaces/mamba-130m-hf")
    parser.add_argument("--checkpoint_path", type=str, required=True)
    parser.add_argument("--temperature", type=float, default=1.0)
    return parser.parse_args()


@torch.no_grad()
def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load Tokenizer & Model
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    custom_tokens = {"additional_special_tokens": [
        "<noul/>", "<score/>", "<choice/>",
        "<answer/>", "<question>", "<question/>",
        "<state>", "<state/>"
    ]}
    tokenizer.add_special_tokens(custom_tokens)

    hf_model = MambaForCausalLM.from_pretrained(args.model_name, torch_dtype=torch.float32)
    hf_model.resize_token_embeddings(len(tokenizer))

    model = ForkMambaChooser(hf_model, tokenizer=tokenizer)
    model.setup_peft(r=16, alpha=32)

    # Load Adapter Weights
    print(f"Loading checkpoint weights from {args.checkpoint_path}...")
    saved_weights = torch.load(args.checkpoint_path, map_location=device)
    model.load_state_dict(saved_weights, strict=False)
    model.to(device)
    model.eval()

    # Sample Inference Input
    sample_record = [{
        "state": "A statistic in the draft attributes to a study that says the opposite...",
        "question": "Rate claim reliability for this scenario on a 0-5 scale.",
        "kind": "score",
        "options": ["0", "1", "2", "3", "4", "5"],
        "target": [0.0, 0.0, 0.0, 0.5, 0.5, 0.0]
    }]

    dataset = ForkMambaDataset(sample_record)
    collator = ForkMambaDataCollator(tokenizer)
    batch = collator([dataset[0]])
    batch = {k: v.to(device) for k, v in batch.items()}

    outputs = model(**batch, temperature=args.temperature)
    probs = outputs["probs"].squeeze(0).cpu().tolist()

    print("\n--- Inference Output ---")
    for opt, prob in zip(sample_record[0]["options"], probs):
        print(f"Option '{opt}': {prob * 100:.2f}%")


if __name__ == "__main__":
    main()
