import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
from ds import create_loader, tokenizer
from model import YAADModel
from main_big import generate_text, val_tokens

torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision("high")
GENERATE_MAX_TOKENS = 50
GRADIENT_CLIP_VALUE = 1.0


def train_tinystories():
    BATCH_SIZE = 64
    LEARNING_RATE = 3e-4
    TRAINING_EPOCHS = 3

    model_configuration = {
        "number_of_layers": 4,
        "model_dimension": 256,
        "head_dimension": 64,
        "state_dimension": 128,
        "feedforward_dimension": 256 * 4,
        "low_rank_dimension": 64,
        "expansion_factor": 2,
    }

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    data_loader = create_loader(BATCH_SIZE)

    model = YAADModel(vocab_size=len(tokenizer), **model_configuration).to(
        device, dtype=torch.bfloat16
    )
    compiled_model = torch.compile(
        model, options={"triton.cudagraphs": True}, fullgraph=True
    )

    total_parameters = sum(parameter.numel() for parameter in model.parameters())
    print(f"Model created with {total_parameters / 1e6:.2f}M parameters.")

    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
    scheduler = CosineAnnealingLR(optimizer, T_max=len(data_loader) * TRAINING_EPOCHS)

    for epoch_index in range(TRAINING_EPOCHS):
        model.train()
        progress_bar = tqdm(
            data_loader, desc=f"Epoch {epoch_index + 1}/{TRAINING_EPOCHS}"
        )
        cumulative_loss = 0.0

        for batch_index, batch_data in enumerate(progress_bar):
            input_sequences = batch_data["input_ids"].to(device)
            target_sequences = input_sequences[:, 1:].contiguous()
            input_sequences = input_sequences[:, :-1].contiguous()

            optimizer.zero_grad()
            predicted_logits = compiled_model(input_sequences)
            loss = F.cross_entropy(
                predicted_logits.view(-1, predicted_logits.size(-1)),
                target_sequences.view(-1),
            )

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_CLIP_VALUE)
            optimizer.step()
            scheduler.step()
            if batch_index % 100 == 0:
                sample_generation = generate_text(
                    model, tokenizer, "Once upon a time in a land far, far away,"
                )
                input_sequences = val_tokens.to(device)
                target_sequences = input_sequences[:, 1:].contiguous()
                input_sequences = input_sequences[:, :-1].contiguous()
                val_logits = compiled_model(input_sequences)
                val_loss = F.cross_entropy(
                    val_logits.view(-1, val_logits.size(-1)),
                    target_sequences.view(-1),
                )
                print(sample_generation, f"Validation Loss: {val_loss.item():.4f}")

            cumulative_loss += loss.item()
            average_loss = cumulative_loss / (batch_index + 1)

            progress_bar.set_postfix(
                {
                    "loss": f"{loss.item():.4f}",
                    "avg_loss": f"{average_loss:.4f}",
                }
            )

    print("Training finished.")


if __name__ == "__main__":
    train_tinystories()
