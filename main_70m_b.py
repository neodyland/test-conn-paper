import os

os.environ["TOKENIZERS_PARALLELISM"] = "true"
os.environ["HF_HOME"] = "/raid/hf_cache"
from main_0_5b import train_tinystories
from ds import threads


if __name__ == "__main__":
    train_tinystories(
        model_configuration={
            "number_of_layers": 4,
            "model_dimension": 256,
            "head_dimension": 64,
            "state_dimension": 128,
            "feedforward_dimension": 256 * 4,
            "low_rank_dimension": 64,
            "expansion_factor": 2,
        },
        ds_path="/raid/zyda2_dclm_sample/",
        workers=threads // 2,
        batch_size=84,
    )
