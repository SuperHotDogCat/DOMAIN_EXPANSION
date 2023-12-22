import pickle
from typing import List
import numpy as np
from utils import preprocessing_train_data

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import models, layers, optimizers
from sklearn.model_selection import train_test_split

# https://qiita.com/yohachi/items/434f0da356161e82c242にしたがってモデルを作りたい

model = models.Sequential(
    [
        layers.Dense(units=40, activation="relu", input_shape=(122,)),
        layers.Dense(units=20, activation="relu"),
        layers.Dense(units=2, activation="softmax"),
    ]
)

optimizer = optimizers.Adam(lr=0.001)

model.compile(
    loss="sparse_categorical_crossentropy",
    optimizer=optimizer,
    metrics=["accuracy"],
)
if __name__ == "__main__":
    data = pickle.load(open("gesture_data.bin", "rb"))
    processed_data, label = preprocessing_train_data(data)
    X_train, X_val, y_train, y_val = train_test_split(
        processed_data, label, test_size=0.2, random_state=42
    )
    history = model.fit(
        X_train, y_train, epochs=100, validation_data=(X_val, y_val)
    )
    model.save("model")
