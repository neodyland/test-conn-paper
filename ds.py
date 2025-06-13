from transformers import AutoTokenizer
from datasets import load_from_disk, Dataset
from torch.utils.data import DataLoader
import os
import torch
import numpy as np

threads = os.cpu_count()
tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-1.7B-Instruct")


def process_ds(dataset: Dataset, sequence_length=256, ds_path="./data/ds"):
    def tokenize_and_create_chunks(examples):
        tokenized_texts = tokenizer(examples["text"]).input_ids
        all_token_ids = []

        for token_sequence in tokenized_texts:
            all_token_ids.extend(token_sequence + [tokenizer.eos_token_id])

        chunked_sequences = []
        for start_index in range(
            0, len(all_token_ids) - sequence_length, sequence_length
        ):
            chunk = all_token_ids[start_index : start_index + sequence_length]
            chunked_sequences.append(chunk)

        return {"input_ids": chunked_sequences}

    print("Tokenizing and chunking dataset...")
    processed_dataset = dataset.map(
        tokenize_and_create_chunks,
        batched=True,
        batch_size=10000,
        remove_columns=dataset.column_names,
        num_proc=threads,
    )
    processed_dataset.save_to_disk(ds_path)


def create_loader(batch_size: int, ds_path="./data/ds"):
    dataset = load_from_disk(ds_path)

    def collate_fn(batch):
        input_ids = [item["input_ids"] for item in batch]
        return {"input_ids": torch.from_numpy(np.array(input_ids, dtype=np.int64))}

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0,
    )
