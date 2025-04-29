import os
import numpy as np
import matplotlib.pyplot as plt
from .config import CONFIG
from keras.datasets import mnist
from keras.utils import to_categorical


def load_data():
    # Скачиваем данные в папку data/raw/
    path = to_categorical(
        fname=CONFIG["data/raw/"],
        origin='https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz',
        cache_dir=os.path.dirname(CONFIG["data/raw/"]),
        cache_subdir=''
    )

        # Загрузка данных из локального файла
    with np.load(path) as f:
        x_train, y_train = f['x_train'], f['y_train']
        x_test, y_test = f['x_test'], f['x_test']
    
    # Предобработка
    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0
    
    # Сохранение предобработанных данных (опционально)
    np.savez_compressed(
        os.path.join(CONFIG["processed_data_path"], "mnist_processed.npz"),
        x_train=x_train, y_train=y_train,
        x_test=x_test, y_test=y_test
    )
    
    return (x_train, y_train), (x_test, y_test)


def plot_training_history(history, plot_path):
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig(plot_path)
    plt.close()