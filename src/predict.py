import numpy as np
import matplotlib.pyplot as plt
from keras.models import load_model
from config import CONFIG
from utils import load_data

def predict_random_sample():
    # Загрузка модели и данных
    model = load_model(CONFIG["model_path"])
    (_, _), (x_test, y_test) = load_data()
    
    # Случайный пример
    idx = np.random.randint(0, len(x_test))
    image = x_test[idx].reshape(28, 28)
    true_label = np.argmax(y_test[idx])
    
    # Предсказание
    prediction = model.predict(x_test[idx].reshape(1, -1))
    predicted_label = np.argmax(prediction)
    
    # Визуализация
    plt.imshow(image, cmap='gray')
    plt.title(f"True: {true_label}, Predicted: {predicted_label}")
    plt.show()

if __name__ == "__main__":
    predict_random_sample()