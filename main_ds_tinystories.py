from ds import process_ds
from datasets import load_dataset

if __name__ == "__main__":
    process_ds(
        load_dataset("roneneldan/TinyStories", split="train"),
        sequence_length=256,
        ds_path="./data/tinystories",
    )
