"""1D-CNN architecture used for ECG beat classification."""

import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import (
    Conv1D,
    Dense,
    Dropout,
    Flatten,
    Input,
    MaxPooling1D,
)


INPUT_LENGTH = 200
INPUT_CHANNELS = 1
NUMBER_OF_CLASSES = 3


def build_model(
    input_shape: tuple[int, int] = (INPUT_LENGTH, INPUT_CHANNELS),
    number_of_classes: int = NUMBER_OF_CLASSES,
) -> tf.keras.Model:
    """Build and compile the 1D-CNN used in the ECG experiments."""

    model = Sequential(
        [
            Input(shape=input_shape),

            Conv1D(
                filters=32,
                kernel_size=5,
                activation="relu",
            ),

            MaxPooling1D(pool_size=2),

            Conv1D(
                filters=64,
                kernel_size=5,
                activation="relu",
            ),

            MaxPooling1D(pool_size=2),

            Flatten(),

            Dense(
                units=128,
                activation="relu",
            ),

            Dropout(rate=0.5),

            Dense(
                units=number_of_classes,
                activation="softmax",
            ),
        ],
        name="ecg_arrhythmia_1d_cnn",
    )

    model.compile(
        loss="categorical_crossentropy",
        optimizer="adam",
        metrics=["accuracy"],
    )

    return model


if __name__ == "__main__":
    model = build_model()
    model.summary()