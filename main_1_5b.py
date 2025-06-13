import os

os.environ["HF_HOME"] = "/raid/hf_cache"
from main_0_5b import train_tinystories


if __name__ == "__main__":
    train_tinystories(
        model_configuration={
            "number_of_layers": 36,
            "model_dimension": 2048,
            "head_dimension": 192,
            "state_dimension": 256,
            "feedforward_dimension": 2048 * 3,
            "low_rank_dimension": 64,
            "expansion_factor": 2,
        },
        ds_path="/raid/zyda2_dclm_sample/",
    )
