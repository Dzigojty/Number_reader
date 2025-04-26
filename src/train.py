import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense
from utils import load_data, plot_training_history
from config import CONFIG

def build_model():
    model = Sequential()
    model.add(Dense(CONFIG["hidden_units"][0], activation=CONFIG["activation"], input_shape=CONFIG["input_shape"]))
    model.add(Dense(CONFIG["hidden_units"][1], activation=CONFIG["activation"]))
    model.add(Dense(CONFIG["num_classes"], activation='softmax'))
    return model

def main():
    # Загрузка данных
    (x_train, y_train), (x_test, y_test) = load_data()
    
    # Построение модели
    model = build_model()
    model.compile(
        optimizer=CONFIG["optimizer"],
        loss=CONFIG["loss"],
        metrics=CONFIG["metrics"]
    )
    
    # Обучение
    history = model.fit(
        x_train, y_train,
        epochs=CONFIG["epochs"],
        batch_size=CONFIG["batch_size"],
        validation_split=0.2
    )
    
    # Сохранение модели и графиков
    model.save(CONFIG["model_path"])
    plot_training_history(history, CONFIG["plot_path"])
    
    # Оценка на тестовых данных
    test_loss, test_acc = model.evaluate(x_test, y_test)
    print(f"Test Accuracy: {test_acc:.4f}")

if __name__ == "__main__":
    main()