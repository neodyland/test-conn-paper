from ds import load_ds, threads

ds = load_ds()


def count_sequences(batch):
    sequences = sum(len(item) for item in batch["input_ids"])
    return {"total_sequences": [sequences]}


seqs = ds.map(
    count_sequences,
    batched=True,
    batch_size=100000,
    remove_columns=ds.column_names,
    num_proc=threads,
)
global_total_sequences = sum(seqs["total_sequences"])

print(f"Total number of sequences: {len(ds)}")
print(f"Total number of sequences in all items: {global_total_sequences}")
