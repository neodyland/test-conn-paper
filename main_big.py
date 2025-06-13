import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
from ds import create_loader, tokenizer
from model import YAADModel

torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision("high")
GENERATE_MAX_TOKENS = 100
GRADIENT_CLIP_VALUE = 1.0


def generate_text(model, tokenizer, prompt_text):
    device = next(model.parameters()).device
    input_token_ids = tokenizer.encode(prompt_text, return_tensors="pt").to(device)

    with torch.no_grad():
        for _ in range(GENERATE_MAX_TOKENS):
            model_outputs = model(input_token_ids)
            next_token_logits = model_outputs[:, -1, :]
            next_token_id = torch.argmax(next_token_logits, dim=-1).unsqueeze(0)
            input_token_ids = torch.cat([input_token_ids, next_token_id], dim=1)

        generated_tokens = input_token_ids[:, len(prompt_text) :]

    generated_text = tokenizer.decode(generated_tokens[0], skip_special_tokens=True)
    return generated_text


val_tokens = tokenizer(
    """Once upon a time, in a big forest, there lived a rhinoceros named Roxy. Roxy loved to climb. She climbed trees, rocks, and hills. One day, Roxy found an icy hill. She had never seen anything like it before. It was shiny and cold, and she wanted to climb it.
Roxy tried to climb the icy hill, but it was very slippery. She tried again and again, but she kept falling down. Roxy was sad. She wanted to climb the icy hill so much. Then, she saw a little bird named Billy. Billy saw that Roxy was sad and asked, "Why are you sad, Roxy?"
Roxy told Billy about the icy hill and how she couldn't climb it. Billy said, "I have an idea! Let's find some big leaves to put under your feet. They will help you climb the icy hill." Roxy and Billy looked for big leaves and found some. Roxy put the leaves under her feet and tried to climb the icy hill again.
This time, Roxy didn't slip. She climbed and climbed until she reached the top of the icy hill. Roxy was so happy! She and Billy played on the icy hill all day. From that day on, Roxy and Billy were the best of friends, and they climbed and played together all the time. And Roxy learned that with a little help from a friend, she could climb anything.""",
    max_length=256,
    truncation=True,
    return_tensors="pt",
    padding="max_length",
).input_ids


def train_tinystories():
    BATCH_SIZE = 64
    LEARNING_RATE = 3e-4
    TRAINING_EPOCHS = 3

    model_configuration = {
        "number_of_layers": 28,
        "model_dimension": 1024,
        "head_dimension": 128,
        "state_dimension": 256,
        "feedforward_dimension": 1024 * 3,
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
                val_loss = F.cross_entropy(
                    compiled_model(val_tokens.to(device)).view(
                        -1, compiled_model.vocab_size
                    ),
                    val_tokens[:, 1:].contiguous().view(-1).to(device),
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
