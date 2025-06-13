import torch
import torch.nn as nn
import torch.nn.functional as F

import math
from datasets import load_dataset
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

torch.backends.cudnn.benchmark = True


def is_training(module):
    return module.training


class RoPE(nn.Module):
    def __init__(self, dim, max_seq_len=8192, base=10000.0):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.base = base
        inv_freq = 1.0 / (
            self.base ** (torch.arange(0, self.dim, 2).float() / self.dim)
        )
        self.register_buffer("inv_freq", inv_freq)
        self.max_seq_len_cached = 0
        self.cos_cached = None
        self.sin_cached = None

    def _build_cache(self, seq_len):
        if seq_len > self.max_seq_len_cached:
            self.max_seq_len_cached = seq_len
            t = torch.arange(seq_len, device=self.inv_freq.device)
            freqs = torch.einsum("i,j->ij", t, self.inv_freq)
            emb = torch.cat((freqs, freqs), dim=-1)
            self.cos_cached = emb.cos()
            self.sin_cached = emb.sin()
        return self.cos_cached[:seq_len], self.sin_cached[:seq_len]

    def _apply_rotary_emb(self, x, cos, sin):
        x_rotated = torch.cat(
            [-x[..., self.dim // 2 :], x[..., : self.dim // 2]], dim=-1
        )
        return x * cos + x_rotated * sin

    def forward(self, q, k):
        seq_len = q.shape[1]
        cos, sin = self._build_cache(seq_len)
        q_out = self._apply_rotary_emb(q, cos, sin)
        k_out = self._apply_rotary_emb(k, cos, sin)
        return q_out, k_out


class GatedMLP(nn.Module):
    def __init__(self, d_model, d_ffn):
        super().__init__()
        self.w1 = nn.Linear(d_model, d_ffn, bias=False)
        self.w2 = nn.Linear(d_ffn, d_model, bias=False)
        self.w3 = nn.Linear(d_model, d_ffn, bias=False)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


# --- Part 2: The Core YAAD Layer (Unchanged) ---
class YAADLayer(nn.Module):
    def __init__(self, d_model, d_head, d_state, low_rank_dim=64, expansion_factor=2):
        super().__init__()
        self.d_model = d_model
        self.d_head = d_head
        self.d_state = d_state
        self.d_mlp_inner = d_state * expansion_factor

        self.q_proj = nn.Linear(d_model, d_head, bias=False)
        self.k_proj = nn.Linear(d_model, d_head, bias=False)
        self.v_proj = nn.Linear(d_model, d_head, bias=False)

        self.conv = nn.Conv1d(
            in_channels=d_head,
            out_channels=d_head,
            kernel_size=4,
            padding=3,
            groups=d_head,
        )

        self.eta_proj = nn.Sequential(
            nn.Linear(d_model, low_rank_dim, bias=False),
            nn.Linear(low_rank_dim, d_head, bias=False),
        )
        self.delta_proj = nn.Sequential(
            nn.Linear(d_model, low_rank_dim, bias=False),
            nn.Linear(low_rank_dim, d_head, bias=False),
        )
        self.alpha_proj = nn.Sequential(
            nn.Linear(d_model, low_rank_dim, bias=False),
            nn.Linear(low_rank_dim, d_head, bias=False),
        )

        self.out_proj = nn.Linear(d_head, d_model, bias=False)
        self.gate_proj = nn.Linear(d_model, d_model, bias=False)

        self.q_norm = nn.LayerNorm(d_head)
        self.k_norm = nn.LayerNorm(d_head)

        self.memory_mlp_w1 = nn.Parameter(torch.empty(self.d_mlp_inner, d_head))
        self.memory_mlp_b1 = nn.Parameter(torch.empty(self.d_mlp_inner))
        self.memory_mlp_w2 = nn.Parameter(torch.empty(d_head, self.d_mlp_inner))
        self.memory_mlp_b2 = nn.Parameter(torch.empty(d_head))
        self.reset_memory_parameters()

    def reset_memory_parameters(self):
        nn.init.kaiming_uniform_(self.memory_mlp_w1, a=math.sqrt(5))
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.memory_mlp_w1)
        bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
        nn.init.uniform_(self.memory_mlp_b1, -bound, bound)
        nn.init.kaiming_uniform_(self.memory_mlp_w2, a=math.sqrt(5))
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.memory_mlp_w2)
        bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
        nn.init.uniform_(self.memory_mlp_b2, -bound, bound)

    def _memory_forward(self, k_chunk, memory_state, return_intermediate=False):
        w1, b1, w2, b2 = memory_state
        intermediate = k_chunk @ w1.transpose(-2, -1) + b1.unsqueeze(1)
        activated = F.gelu(intermediate)
        output = activated @ w2.transpose(-2, -1) + b2.unsqueeze(1)
        if return_intermediate:
            return output, activated
        return output

    def forward(self, x, chunk_size=64):
        batch_size, seq_len, _ = x.shape

        q, k, v = self.q_proj(x), self.k_proj(x), self.v_proj(x)
        eta, delta_t, alpha = (
            F.softplus(self.eta_proj(x)),
            F.softplus(self.delta_proj(x)),
            torch.sigmoid(self.alpha_proj(x)),
        )

        q, k, v = (
            self.conv(t.transpose(1, 2)).transpose(1, 2)[:, :seq_len] for t in (q, k, v)
        )
        q, k = self.q_norm(q), self.k_norm(k)

        num_chunks = (seq_len + chunk_size - 1) // chunk_size
        outputs = []

        current_memory_state = [
            self.memory_mlp_w1.unsqueeze(0).expand(batch_size, -1, -1),
            self.memory_mlp_b1.unsqueeze(0).expand(batch_size, -1),
            self.memory_mlp_w2.unsqueeze(0).expand(batch_size, -1, -1),
            self.memory_mlp_b2.unsqueeze(0).expand(batch_size, -1),
        ]

        for i in range(num_chunks):
            start_idx, end_idx = i * chunk_size, min((i + 1) * chunk_size, seq_len)
            k_chunk, v_chunk = k[:, start_idx:end_idx], v[:, start_idx:end_idx]
            eta_chunk, delta_chunk, alpha_chunk = (
                eta[:, start_idx:end_idx],
                delta_t[:, start_idx:end_idx],
                alpha[:, start_idx:end_idx],
            )

            if is_training(self):
                memory_output, activated_intermediate = self._memory_forward(
                    k_chunk, current_memory_state, return_intermediate=True
                )
            else:
                memory_output = self._memory_forward(k_chunk, current_memory_state)
            outputs.append(memory_output)

            if is_training(self):
                with torch.no_grad():
                    error = memory_output - v_chunk
                    error_norm = torch.linalg.norm(error, dim=-1, keepdim=True)
                    grad_l2, grad_l1 = (
                        error,
                        delta_chunk * (error / (error_norm + 1e-9)),
                    )
                    grad_pred = torch.where(
                        (error_norm > delta_chunk), grad_l1, grad_l2
                    )

                    alpha_w, eta_w = (
                        alpha_chunk[:, -1, :].mean(-1).view(-1, 1, 1),
                        eta_chunk[:, -1, :].mean(-1).view(-1, 1, 1),
                    )
                    update_signal = grad_pred.mean(1, keepdim=True)
                    w1, b1, w2, b2 = current_memory_state
                    w1_next = alpha_w * w1 - eta_w * (
                        activated_intermediate.mean(1, keepdim=True).transpose(-2, -1)
                        @ update_signal
                    )
                    w2_next = alpha_w * w2 - eta_w * (
                        update_signal.transpose(-2, -1)
                        @ activated_intermediate.mean(1, keepdim=True)
                    )

                    alpha_b, eta_b = alpha_chunk[:, -1, :], eta_chunk[:, -1, :]
                    b2_update_signal, b1_update_signal = (
                        grad_pred.mean(1),
                        (grad_pred @ w2).mean(1),
                    )
                    b2_next = alpha_b * b2 - eta_b * b2_update_signal
                    b1_next = (
                        alpha_b.mean(-1, keepdim=True) * b1
                        - eta_b.mean(-1, keepdim=True) * b1_update_signal
                    )
                    current_memory_state = [w1_next, b1_next, w2_next, b2_next]

        y = torch.cat(outputs, dim=1)
        gated_y = self.out_proj(y)
        gate = torch.sigmoid(self.gate_proj(x))
        return gated_y * gate


# --- Part 3: The Full LLM Model Structure (Unchanged) ---
class YAADBlock(nn.Module):
    def __init__(self, d_model, d_head, d_state, d_ffn, low_rank_dim, expansion_factor):
        super().__init__()
        self.yaad_layer = YAADLayer(
            d_model, d_head, d_state, low_rank_dim, expansion_factor
        )
        self.mlp = GatedMLP(d_model, d_ffn)
        self.norm1 = nn.RMSNorm(d_model)
        self.norm2 = nn.RMSNorm(d_model)

    def forward(self, x):
        h = x + self.yaad_layer(self.norm1(x))
        out = h + self.mlp(self.norm2(h))
        return out


class YAADModel(nn.Module):
    def __init__(
        self,
        vocab_size,
        n_layers,
        d_model,
        d_head,
        d_state,
        d_ffn,
        low_rank_dim,
        expansion_factor,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.layers = nn.ModuleList(
            [
                YAADBlock(
                    d_model, d_head, d_state, d_ffn, low_rank_dim, expansion_factor
                )
                for _ in range(n_layers)
            ]
        )
        self.norm_out = nn.RMSNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, tokens):
        x = self.embedding(tokens)
        for layer in self.layers:
            x = layer(x)
        x = self.norm_out(x)
        logits = self.lm_head(x)
        return logits


def generate(model, tokenizer, prompt):
    device = next(model.parameters()).device
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        for _ in range(50):
            outputs = model(input_ids)
            next_token_logits = outputs[:, -1, :]
            next_token = torch.argmax(next_token_logits, dim=-1).unsqueeze(0)
            input_ids = torch.cat([input_ids, next_token], dim=1)
        output = input_ids[:, len(prompt) :]
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    return generated_text


def get_tokenizer():
    tok = AutoTokenizer.from_pretrained(
        "gpt2",
    )
    return tok


def create_dataloader(tokenizer, seq_len=256, batch_size=32):
    """Creates a dataloader for the TinyStories dataset."""
    dataset = load_dataset("roneneldan/TinyStories", split="train")

    def tokenize_and_chunk(examples):
        tokens = tokenizer(examples["text"]).input_ids
        all_token_ids = []
        for t in tokens:
            all_token_ids.extend(t + [tokenizer.eos_token_id])
        chunked_tokens = []
        for i in range(0, len(all_token_ids) - seq_len, seq_len):
            chunked_tokens.append(all_token_ids[i : i + seq_len])
        return {"input_ids": chunked_tokens}

    print("Tokenizing and chunking dataset...")
    processed_dataset = dataset.map(
        tokenize_and_chunk,
        batched=True,
        batch_size=1000,
        remove_columns=dataset.column_names,
        num_proc=4,
    )
    processed_dataset.set_format(type="torch")

    return DataLoader(processed_dataset, batch_size=batch_size, shuffle=True)


def train_tinystories():
    SEQ_LEN = 256
    BATCH_SIZE = 16
    LEARNING_RATE = 3e-4
    EPOCHS = 3

    config = {
        "n_layers": 4,
        "d_model": 256,
        "d_head": 64,
        "d_state": 128,
        "d_ffn": 256 * 4,
        "low_rank_dim": 64,
        "expansion_factor": 2,
    }

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    tokenizer = get_tokenizer()
    dataloader = create_dataloader(tokenizer, seq_len=SEQ_LEN, batch_size=BATCH_SIZE)

    model = YAADModel(vocab_size=len(tokenizer), **config).to(device)
    model_fused = torch.compile(
        model,
    )
    print(
        f"Model created with {sum(p.numel() for p in model.parameters()):,} parameters."
    )

    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
    scheduler = CosineAnnealingLR(optimizer, T_max=len(dataloader) * EPOCHS)

    for epoch in range(EPOCHS):
        model.train()
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{EPOCHS}")
        total_loss = 0.0

        for i, batch in enumerate(progress_bar):
            inputs = batch["input_ids"].to(device)
            targets = inputs[:, 1:].contiguous()
            inputs = inputs[:, :-1].contiguous()

            optimizer.zero_grad()

            logits = model_fused(inputs)

            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            if i % 100 == 0:
                print(
                    generate(
                        model,
                        tokenizer,
                        "Once upon a time in a land far, far away,",
                    )
                )
            total_loss += loss.item()
            progress_bar.set_postfix(
                {
                    "loss": f"{loss.item():.4f}",
                    "avg_loss": f"{total_loss / (i + 1):.4f}",
                }
            )

    print("Training finished.")


if __name__ == "__main__":
    train_tinystories()
