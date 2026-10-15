import torch
from torch.utils.data import Dataset


class ForkMambaDataset(Dataset):
    """Formats Hugging Face dataset items into XML-tagged prompts and target arrays."""
    def __init__(self, raw_dataset):
        self.dataset = raw_dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]

        state_str = (item.get("state") or "").strip()
        question_str = (item.get("question") or "").strip()
        kind_str = (item.get("kind") or "score").strip()

        kind_token = f"<{kind_str}/>" if f"<{kind_str}/>" in ("<score/>", "<choice/>", "<noul/>") else "<score/>"

        if state_str:
            formatted_q = f"<state> {state_str} <state/> <question> {question_str} <question/> {kind_token}"
        else:
            formatted_q = f"<question> {question_str} <question/> {kind_token}"

        formatted_choices = [f"<choice/> {str(opt)}" for opt in item["options"]]

        return {
            "question": formatted_q,
            "choices": formatted_choices,
            "target": item["target"]
        }


class ForkMambaDataCollator:
    """Dynamic padding and batch tensor creator for GPU execution."""
    def __init__(self, tokenizer, max_q_len=1024, max_c_len=64):
        self.tokenizer = tokenizer
        self.max_q_len = max_q_len
        self.max_c_len = max_c_len

    def __call__(self, batch):
        batch_size = len(batch)
        max_num_choices = max(len(x["choices"]) for x in batch)

        q_encodings = [
            self.tokenizer(x["question"], truncation=True, max_length=self.max_q_len).input_ids
            for x in batch
        ]

        c_encodings = [
            [self.tokenizer(c, truncation=True, max_length=self.max_c_len).input_ids for c in x["choices"]]
            for x in batch
        ]

        max_q_len = max(len(q) for q in q_encodings)
        max_c_len = max(len(c) for row in c_encodings for c in row)

        q_ids = torch.zeros(batch_size, max_q_len, dtype=torch.long)
        q_mask = torch.zeros(batch_size, max_q_len, dtype=torch.float32)

        c_ids = torch.zeros(batch_size, max_num_choices, max_c_len, dtype=torch.long)
        c_mask = torch.zeros(batch_size, max_num_choices, max_c_len, dtype=torch.float32)
        valid = torch.zeros(batch_size, max_num_choices, dtype=torch.bool)
        labels = torch.zeros(batch_size, max_num_choices, dtype=torch.float32)

        for i, example in enumerate(batch):
            q_len = len(q_encodings[i])
            q_ids[i, :q_len] = torch.tensor(q_encodings[i])
            q_mask[i, :q_len] = 1.0

            for j, c_tokens in enumerate(c_encodings[i]):
                c_len = len(c_tokens)
                c_ids[i, j, :c_len] = torch.tensor(c_tokens)
                c_mask[i, j, :c_len] = 1.0
                valid[i, j] = True
                labels[i, j] = example["target"][j]

        return {
            "q_ids": q_ids,
            "q_mask": q_mask,
            "c_ids": c_ids,
            "c_mask": c_mask,
            "valid": valid,
            "labels": labels
        }
