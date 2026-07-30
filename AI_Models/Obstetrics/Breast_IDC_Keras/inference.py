"""Patch-level invasive ductal carcinoma inference."""
import os
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import keras
import numpy as np
import pandas as pd
from PIL import Image


def _build_model():
    inputs = keras.Input((50, 50, 3), name="input_1")
    x = keras.layers.Rescaling(1 / 255.0, name="rescaling")(inputs)
    for index in range(4):
        suffix = "" if index == 0 else f"_{index}"
        x = keras.layers.Conv2D(256, 3, use_bias=False, name=f"conv2d{suffix}")(x)
        x = keras.layers.BatchNormalization(name=f"batch_normalization{suffix}")(x)
        x = keras.layers.Activation("relu", name=f"activation{suffix}")(x)
        if index in (1, 2, 3):
            pool_suffix = "" if index == 1 else f"_{index - 1}"
            x = keras.layers.MaxPooling2D(name=f"max_pooling2d{pool_suffix}")(x)
    x = keras.layers.Flatten(name="flatten")(x)
    x = keras.layers.Dropout(0.5, name="dropout")(x)
    x = keras.layers.Dense(512, activation="relu", name="dense")(x)
    x = keras.layers.Dense(128, activation="relu", name="dense_1")(x)
    x = keras.layers.Dense(32, activation="relu", name="dense_2")(x)
    return keras.Model(inputs, keras.layers.Dense(1, activation="sigmoid", name="dense_3")(x))


def main(input_data, model_path: str):
    image = Image.open(input_data).convert("RGB").resize((50, 50))
    model = _build_model()
    model.load_weights(str(Path(model_path)))
    positive = float(np.asarray(model.predict(np.asarray(image, dtype=np.float32)[None], verbose=0)).reshape(-1)[0])
    pred = int(positive >= 0.5)
    return pd.DataFrame({
        "pred": [pred],
        "pred_name": ["IDC positive" if pred else "IDC negative"],
        "prob": [positive if pred else 1.0 - positive],
        "idc_positive_prob": [positive],
    })
