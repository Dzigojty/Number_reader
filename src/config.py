# Гиперпараметры и пути
CONFIG = {
    "data_path": "../data/",
    "model_path": "../models/mnist_model.keras",
    "plot_path": "../plots/training_plot.png",
    "input_shape": (784,),
    "num_classes": 10,
    "batch_size": 32,
    "epochs": 10,
    "hidden_units": [128, 64],
    "activation": "relu",
    "optimizer": "adam",
    "loss": "categorical_crossentropy",
    "metrics": ["accuracy"]
}