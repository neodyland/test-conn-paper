import torch
import torch.nn.functional as F
import time
from torch.optim.lr_scheduler import CosineAnnealingLR
from ds import create_loader, tokenizer
from model import YAADModel
import time
from muon import SingleDeviceMuonWithAuxAdam
import os

torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision("high")
GENERATE_MAX_TOKENS = 100
GRADIENT_CLIP_VALUE = 1.0


@torch.inference_mode()
def generate_text(model, tokenizer, prompt_text):
    device = next(model.parameters()).device
    input_token_ids = tokenizer.encode(prompt_text, return_tensors="pt").to(device)

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


@torch.inference_mode()
def calc_val_loss(device, compiled_model):
    input_sequences = val_tokens.to(device)
    target_sequences = input_sequences[:, 1:].contiguous()
    input_sequences = input_sequences[:, :-1].contiguous()
    val_logits = compiled_model(input_sequences)
    val_loss = F.cross_entropy(
        val_logits.view(-1, val_logits.size(-1)),
        target_sequences.view(-1),
    )
    return val_loss.item(), torch.exp(val_loss).item()


def train_tinystories(
    model_configuration, batch_size=64, ds_path="./data/tinystories", workers=0
):
    exp = time.strftime("%Y-%m-%d_%H-%M-%S")
    os.makedirs(f"data/exps/{exp}", exist_ok=True)
    log = open(f"data/exps/{exp}/train.log", "w")
    best_pt_path = f"data/exps/{exp}/best.pt"
    best_val_path = f"data/exps/{exp}/best_val.txt"
    best_val = float("inf")
    old_print = print

    def print(*args, **kwargs):
        log.write(" ".join(map(str, args)) + "\n")
        log.flush()
        old_print(*args, **kwargs)

    LEARNING_RATE = 3e-4
    TRAINING_EPOCHS = 3

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = YAADModel(vocab_size=len(tokenizer), **model_configuration).to(
        device, dtype=torch.bfloat16
    )
    total_parameters = sum(parameter.numel() for parameter in model.parameters())
    print(f"Model created with {total_parameters / 1e6:.2f}M parameters.")
    compiled_model = torch.compile(
        model, options={"triton.cudagraphs": True}, fullgraph=True
    )

    print(f"Using device: {device}")
    data_loader = create_loader(batch_size, ds_path, workers)

    optimizer = SingleDeviceMuonWithAuxAdam(
        [
            {
                "params": model.muon_parameters(),
                "lr": LEARNING_RATE * 5.0,
                "use_muon": True,
            },
            {
                "params": model.adam_parameters(),
                "lr": LEARNING_RATE,
                "use_muon": False,
            },
        ]
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=len(data_loader) * TRAINING_EPOCHS)
    now = time.time()
    for epoch_index in range(TRAINING_EPOCHS):
        model.train()

        for batch_index, batch_data in enumerate(data_loader):
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
                model.eval()
                sample_generation = generate_text(
                    model, tokenizer, "Once upon a time in a land far, far away,"
                )
                val_loss, ppl = calc_val_loss(device, compiled_model)
                model.train()
                print(sample_generation)
                print(
                    f"epoch={epoch_index} step={batch_index}/{len(data_loader)} time={time.time() - now:.4f}s loss={loss.item():.4f} val={val_loss:.4f} ppl={ppl:.4f}"
                )
                if batch_index % 1000 == 0 and val_loss < best_val:
                    torch.save(
                        model.state_dict(),
                        best_pt_path,
                    )
                    if os.path.exists(best_val_path):
                        os.remove(best_val_path)
                    with open(best_val_path, "w") as f:
                        f.write(
                            f"epoch={epoch_index} step={batch_index}/{len(data_loader)} time={time.time() - now:.4f}s loss={loss.item():.4f} val={val_loss:.4f} ppl={ppl:.4f}\n"
                        )
                    best_val = val_loss

    print("Training finished.")


if __name__ == "__main__":
    train_tinystories(
        model_configuration={
            "number_of_layers": 28,
            "model_dimension": 1024,
            "head_dimension": 128,
            "state_dimension": 256,
            "feedforward_dimension": 1024 * 3,
            "low_rank_dimension": 64,
            "expansion_factor": 2,
        }
    )
