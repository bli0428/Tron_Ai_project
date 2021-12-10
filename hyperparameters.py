TRAINER_PARAMETERS = {
    "num_iterations": 1000,
    "num_episodes": 100,
    "temp_threshold": 15,
    "num_games": 20,
    "update_threshold": 0.55,
    "train_history_size": 20
}

MCTS_PARAMETERS = {
    "c_puct": 1,
    "num_sim": 25
}

MODEL_PARAMETERS = {
    "num_epochs": 10,
    "batch_size": 64,
    "dropout": 0.3,
    "network_size": 512,
    "learning_rate": 0.001
}