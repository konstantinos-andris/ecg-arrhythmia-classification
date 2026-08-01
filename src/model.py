"""1D-CNN architecture for ECG beat classification."""

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
    """Build and compile the CNN used in the ECG experiment."""

    model = Sequential(
        [
            Input(shape=input_shape, name="ecg_beat"),

            Conv1D(
                filters=32,
                kernel_size=5,
                activation="relu",
                name="conv_1",
            ),

            MaxPooling1D(
                pool_size=2,
                name="pool_1",
            ),

            Conv1D(
                filters=64,
                kernel_size=5,
                activation="relu",
                name="conv_2",
            ),

            MaxPooling1D(
                pool_size=2,
                name="pool_2",
            ),

            Flatten(name="flatten"),

            Dense(
                units=128,
                activation="relu",
                name="dense_features",
            ),

            Dropout(
                rate=0.5,
                name="dropout",
            ),

            Dense(
                units=number_of_classes,
                activation="softmax",
                name="class_probabilities",
            ),
        ],
        name="ecg_arrhythmia_1d_cnn",
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model


if __name__ == "__main__":
    cnn_model = build_model()
    cnn_model.summary()