import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from peft import LoraConfig, inject_adapter_in_model


class ForkMambaChooser(nn.Module):
    """
    PEFT-Compatible State-Branching Mamba Chooser supporting custom Hugging Face
    tokenizer special tokens and temperature-scaled softmax scoring.
    """
    def __init__(self, hf_model, tokenizer=None):
        super().__init__()
        self.backbone = hf_model.backbone
        self.hidden_dim = self.backbone.config.hidden_size

        self.W = nn.Parameter(torch.eye(self.hidden_dim))
        self.logit_scale = nn.Parameter(torch.zeros(()))

        if tokenizer is not None:
            self.set_special_token_ids(tokenizer)
        else:
            self.ans_token_id = None
            self.choice_token_id = None

    def set_special_token_ids(self, tokenizer):
        self.ans_token_id = tokenizer.convert_tokens_to_ids("<answer/>")
        self.choice_token_id = tokenizer.convert_tokens_to_ids("<choice/>")

        if self.ans_token_id == tokenizer.unk_token_id or self.choice_token_id == tokenizer.unk_token_id:
            raise ValueError("Special tokens '<answer/>' or '<choice/>' not found in tokenizer vocabulary!")

    def setup_peft(
        self,
        r: int = 16,
        alpha: int = 32,
        dropout: float = 0.05,
        target_modules=("in_proj", "x_proj", "dt_proj")
    ):
        lora_config = LoraConfig(
            r=r,
            lora_alpha=alpha,
            lora_dropout=dropout,
            target_modules=list(target_modules),
            bias="none"
        )
        inject_adapter_in_model(lora_config, self.backbone)

        for name, param in self.named_parameters():
            if any(k in name for k in ("lora_", "W", "logit_scale", "embeddings")):
                param.requires_grad = True
            else:
                param.requires_grad = False

        return self

    def trainable_state_dict(self):
        trainable_keys = ("lora_", "W", "logit_scale", "embeddings")
        return {k: v.detach().cpu() for k, v in self.state_dict().items() if any(tk in k for tk in trainable_keys)}

    def _embed_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.backbone.embeddings(input_ids)

    @staticmethod
    def _fork_states(layer_states, num_choices: int):
        def dup(t): return t.repeat_interleave(num_choices, dim=0)
        return [(dup(conv), dup(ssm)) for conv, ssm in layer_states]

    @staticmethod
    def _merge_states(layer_states, valid_mask: torch.Tensor, B: int, N: int):
        w = valid_mask.float()
        cnt = w.sum(dim=1).clamp(min=1)
        def avg(t):
            t = t.view(B, N, *t.shape[1:])
            m = w.view(B, N, *([1] * (t.dim() - 2)))
            return (t * m).sum(dim=1) / cnt.view(B, *([1] * (t.dim() - 2)))
        return [(avg(conv), avg(ssm)) for conv, ssm in layer_states]

    def _step_mamba_layer(self, mixer, x, prev_state=None, mask=None):
        B, L, _ = x.shape
        K, Din, N = mixer.conv_kernel_size, mixer.intermediate_size, mixer.ssm_state_size

        u, gate = mixer.in_proj(x).transpose(1, 2).float().chunk(2, dim=1)

        if prev_state is None:
            conv_state, ssm_state = u.new_zeros(B, Din, K - 1), u.new_zeros(B, Din, N)
        else:
            conv_state, ssm_state = prev_state

        u_full = torch.cat([conv_state, u], dim=-1)
        conv_out = F.silu(F.conv1d(u_full, mixer.conv1d.weight, mixer.conv1d.bias, groups=Din)).float()

        ssm_params = mixer.x_proj(conv_out.transpose(1, 2)).float()
        dt_raw, ssm_B, ssm_C = ssm_params.split([mixer.time_step_rank, N, N], dim=-1)
        dt = F.softplus(mixer.dt_proj(dt_raw).float()).transpose(1, 2)

        if mask is not None:
            dt = dt * mask.to(dt.dtype)[:, None, :]

        A_matrix = -torch.exp(mixer.A_log.float())
        outputs = []
        for t in range(L):
            dt_t = dt[:, :, t, None]
            dA = torch.exp(dt_t * A_matrix)
            dB_u = dt_t * ssm_B[:, None, t, :] * conv_out[:, :, t, None]
            ssm_state = dA * ssm_state + dB_u
            outputs.append((ssm_state * ssm_C[:, None, t, :]).sum(-1))

        y = (torch.stack(outputs, dim=-1) + conv_out * mixer.D.float()[None, :, None]) * F.silu(gate)
        output = mixer.out_proj(y.transpose(1, 2))

        valid_lens = mask.sum(dim=1).long() if mask is not None else torch.full((B,), L, device=x.device)
        idx = valid_lens[:, None] + torch.arange(K - 1, device=x.device)
        new_conv_state = torch.gather(u_full, 2, idx[:, None, :].expand(B, Din, K - 1))

        return output, (new_conv_state, ssm_state)

    def _run_backbone(self, hidden_states, layer_states=None, mask=None):
        new_layer_states = []
        for i, block in enumerate(self.backbone.layers):
            prev_s = layer_states[i] if layer_states is not None else None
            y, new_s = self._step_mamba_layer(block.mixer, block.norm(hidden_states), prev_s, mask)
            hidden_states = hidden_states + y.to(hidden_states.dtype)
            new_layer_states.append(new_s)

        return self.backbone.norm_f(hidden_states), new_layer_states

    def forward(
        self,
        q_ids: torch.Tensor,
        q_mask: torch.Tensor,
        c_ids: torch.Tensor,
        c_mask: torch.Tensor,
        valid: torch.Tensor,
        labels: torch.Tensor = None,
        temperature: float = 1.0
    ):
        if self.ans_token_id is None:
            raise RuntimeError("Special token IDs not set! Call `model.set_special_token_ids(tokenizer)` first.")

        B, N, Lc = c_ids.shape

        q_embeds = self._embed_ids(q_ids)
        _, q_states = self._run_backbone(q_embeds, layer_states=None, mask=q_mask)

        forked_states = self._fork_states(q_states, N)
        flat_c_ids = c_ids.reshape(B * N, Lc)
        flat_c_mask = c_mask.reshape(B * N, Lc)

        c_embeds = self._embed_ids(flat_c_ids)
        c_hidden, c_states = self._run_backbone(c_embeds, layer_states=forked_states, mask=flat_c_mask)

        last_idx = (flat_c_mask.sum(dim=1).long() - 1).clamp(min=0)
        choice_reprs = c_hidden[torch.arange(B * N, device=c_hidden.device), last_idx].view(B, N, -1)

        merged_states = self._merge_states(c_states, valid, B, N)
        ans_ids = torch.full((B, 1), self.ans_token_id, dtype=torch.long, device=q_ids.device)
        ans_embeds = self._embed_ids(ans_ids)
        ans_hidden, _ = self._run_backbone(ans_embeds, layer_states=merged_states)
        answer_repr = ans_hidden[:, 0]

        scores = torch.einsum("bd,de,bne->bn", answer_repr, self.W, choice_reprs)
        logits = scores * self.logit_scale.exp() / math.sqrt(self.hidden_dim)
        logits = logits.masked_fill(~valid, -1e4)

        probs = F.softmax(logits / max(temperature, 1e-5), dim=-1)

        output = {"logits": logits, "probs": probs}
        if labels is not None:
            output["loss"] = F.cross_entropy(logits, labels)

        return output
