import os

os.environ["HF_HOME"] = "/raid/hf_cache"
from ds import process_ds
from datasets import load_dataset

if __name__ == "__main__":
    process_ds(
        load_dataset(
            "Zyphra/Zyda-2",
            data_files={"train": "sample/100BT/dclm_crossdeduped/*.parquet"},
            split="train",
        ),
        sequence_length=256,
        ds_path="/raid/zyda2_dclm_sample/",
    )
