from main_0_5b import train_tinystories


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
        }
    )
