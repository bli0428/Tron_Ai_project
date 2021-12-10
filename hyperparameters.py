TRAINER_PARAMETERS = {
    "num_iterations": 1000, # Number of iterations of training a new model and comparing it to the old model
    "num_episodes": 100, # Number of games played per iteration (More games = more examples to train on)
    "temp_threshold": 15, # Threshhold for temperature, increasing means spending longer exploring vs exploiting
    "num_games": 20, # How many games to play when comparing bots (Note: this means 20 games per map)
    "update_threshold": 0.55, # The ratio of wins to games - if the bot scores above this for all maps, we update
    "train_history_size": 20
}

MCTS_PARAMETERS = {
    "c_puct": 1,
    "num_sim": 25,
    "epsilon": 1e-8
}

MODEL_PARAMETERS = {
    "num_epochs": 10,
    "batch_size": 64,
    "dropout": 0.3,
    "network_size": 512,
    "learning_rate": 0.001
}