"""Train the 1D-CNN using the preprocessed MIT-BIH training set."""

import json
from collections import Counter
from pathlib import Path

import numpy as np
import tensorflow as tf
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.utils import to_categorical

try:
    from .model import build_model
except ImportError:
    from model import build_model


PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIRECTORY = PROJECT_ROOT / "data" / "processed"
MODELS_DIRECTORY = PROJECT_ROOT / "models"
RESULTS_DIRECTORY = PROJECT_ROOT / "results"

TRAIN_DATA_FILE = PROCESSED_DIRECTORY / "train.npz"
MODEL_FILE = MODELS_DIRECTORY / "ecg_arrhythmia_1d_cnn.keras"
CLASS_NAMES_FILE = MODELS_DIRECTORY / "class_names.npy"
HISTORY_FILE = RESULTS_DIRECTORY / "training_history.json"

CLASS_NAMES = np.asarray(["A", "N", "V"])

RANDOM_STATE = 42
SMOTE_NEIGHBORS = 3
EPOCHS = 100
BATCH_SIZE = 64
VALIDATION_SIZE = 0.2


def load_training_data() -> tuple[np.ndarray, np.ndarray]:
    """Load the preprocessed training beats and labels."""

    if not TRAIN_DATA_FILE.exists():
        raise FileNotFoundError(
            "The processed training dataset was not found. "
            "Run data_preparation.py first."
        )

    dataset = np.load(TRAIN_DATA_FILE)

    features = dataset["x"].astype(np.float32)
    labels = dataset["y"].astype(str)

    if features.ndim != 2:
        raise ValueError(
            "Training features must have shape (samples, 200) "
            f"before SMOTE, but received {features.shape}."
        )

    if len(features) != len(labels):
        raise ValueError(
            "The number of training beats does not match "
            "the number of labels."
        )

    return features, labels


def split_training_validation(
    features: np.ndarray,
    labels: np.ndarray,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """
    Split the original training data before SMOTE.

    The validation set remains completely untouched by synthetic
    oversampling.
    """

    return train_test_split(
        features,
        labels,
        test_size=VALIDATION_SIZE,
        random_state=RANDOM_STATE,
        stratify=labels,
    )


def balance_training_data(
    features: np.ndarray,
    encoded_labels: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply SMOTE exclusively to the training subset."""

    print("\nTraining distribution before SMOTE:")
    print(Counter(encoded_labels))

    smote = SMOTE(
        random_state=RANDOM_STATE,
        k_neighbors=SMOTE_NEIGHBORS,
    )

    balanced_features, balanced_labels = smote.fit_resample(
        features,
        encoded_labels,
    )

    print("\nTraining distribution after SMOTE:")
    print(Counter(balanced_labels))

    return (
        balanced_features.astype(np.float32),
        balanced_labels.astype(np.int64),
    )


def save_training_history(
    history: tf.keras.callbacks.History,
) -> None:
    """Save training and validation metrics as JSON."""

    serializable_history = {
        metric: [float(value) for value in values]
        for metric, values in history.history.items()
    }

    with HISTORY_FILE.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            serializable_history,
            file,
            indent=2,
        )


def train_model() -> None:
    """Split data, apply SMOTE to training only, and train the CNN."""

    tf.keras.utils.set_random_seed(RANDOM_STATE)

    MODELS_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    RESULTS_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("--- Loading preprocessed training data ---")

    features, labels = load_training_data()

    print(f"Original feature shape: {features.shape}")
    print(f"Original class distribution: {Counter(labels)}")

    # ---------------------------------------------------------
    # 1. Split BEFORE SMOTE.
    # ---------------------------------------------------------

    (
        x_train,
        x_validation,
        y_train,
        y_validation,
    ) = split_training_validation(
        features,
        labels,
    )

    print("\n--- Train / validation split ---")

    print(f"Training samples: {len(x_train)}")
    print(f"Validation samples: {len(x_validation)}")

    print(
        "Training distribution: "
        f"{Counter(y_train)}"
    )

    print(
        "Validation distribution: "
        f"{Counter(y_validation)}"
    )

    # ---------------------------------------------------------
    # 2. Encode labels.
    # ---------------------------------------------------------

    encoder = LabelEncoder()
    encoder.fit(CLASS_NAMES)

    y_train_encoded = encoder.transform(y_train)
    y_validation_encoded = encoder.transform(y_validation)

    # ---------------------------------------------------------
    # 3. Apply SMOTE ONLY to the training subset.
    # ---------------------------------------------------------

    x_train_smote, y_train_smote = balance_training_data(
        x_train,
        y_train_encoded,
    )

    # ---------------------------------------------------------
    # 4. Reshape for Conv1D.
    # ---------------------------------------------------------

    x_train_cnn = x_train_smote.reshape(
        -1,
        x_train_smote.shape[1],
        1,
    )

    x_validation_cnn = x_validation.reshape(
        -1,
        x_validation.shape[1],
        1,
    )

    y_train_categorical = to_categorical(
        y_train_smote,
        num_classes=len(encoder.classes_),
    )

    y_validation_categorical = to_categorical(
        y_validation_encoded,
        num_classes=len(encoder.classes_),
    )

    print(f"\nCNN training shape: {x_train_cnn.shape}")
    print(
        "CNN validation shape: "
        f"{x_validation_cnn.shape}"
    )
    print(f"Class order: {list(encoder.classes_)}")

    # ---------------------------------------------------------
    # 5. Build model.
    # ---------------------------------------------------------

    model = build_model(
        input_shape=(
            x_train_cnn.shape[1],
            x_train_cnn.shape[2],
        ),
        number_of_classes=len(encoder.classes_),
    )

    model.summary()

    callbacks = [
        EarlyStopping(
            monitor="val_loss",
            patience=10,
            restore_best_weights=True,
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=5,
            min_lr=1e-6,
        ),
    ]

    # ---------------------------------------------------------
    # 6. Train using the untouched validation set.
    # ---------------------------------------------------------

    print("\n--- Training the 1D-CNN ---")

    history = model.fit(
        x_train_cnn,
        y_train_categorical,
        validation_data=(
            x_validation_cnn,
            y_validation_categorical,
        ),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=callbacks,
        verbose=1,
    )

    # ---------------------------------------------------------
    # 7. Save trained artifacts.
    # ---------------------------------------------------------

    model.save(MODEL_FILE)

    np.save(
        CLASS_NAMES_FILE,
        encoder.classes_,
    )

    save_training_history(history)

    print("\nTraining completed.")
    print(f"Model saved to: {MODEL_FILE}")
    print(f"Class names saved to: {CLASS_NAMES_FILE}")
    print(
        "Training history saved to: "
        f"{HISTORY_FILE}"
    )

    print(
        "\nImportant: patient 106 is NOT used here. "
        "It remains reserved exclusively for final evaluation."
    )


if __name__ == "__main__":
    train_model()

def load_training_data() -> tuple[np.ndarray, np.ndarray]:
    """Load the preprocessed training beats and labels."""

    if not TRAIN_DATA_FILE.exists():
        raise FileNotFoundError(
            "The processed training dataset was not found. "
            "Run data_preparation.py first."
        )

    dataset = np.load(TRAIN_DATA_FILE)

    features = dataset["x"].astype(np.float32)
    labels = dataset["y"].astype(str)

    if features.ndim != 2:
        raise ValueError(
            "Training features must have shape (samples, 200) "
            f"before SMOTE, but received {features.shape}."
        )

    if len(features) != len(labels):
        raise ValueError(
            "The number of training beats does not match "
            "the number of labels."
        )

    return features, labels


def balance_training_data(
    features: np.ndarray,
    encoded_labels: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply SMOTE exclusively to the training set."""

    print("\nClass distribution before SMOTE:")
    print(Counter(encoded_labels))

    smote = SMOTE(
        random_state=RANDOM_STATE,
        k_neighbors=SMOTE_NEIGHBORS,
    )

    balanced_features, balanced_labels = smote.fit_resample(
        features,
        encoded_labels,
    )

    print("\nClass distribution after SMOTE:")
    print(Counter(balanced_labels))

    return (
        balanced_features.astype(np.float32),
        balanced_labels,
    )


def save_training_history(history: tf.keras.callbacks.History) -> None:
    """Save training and validation metrics as JSON."""

    serializable_history = {
        metric: [float(value) for value in values]
        for metric, values in history.history.items()
    }

    with HISTORY_FILE.open("w", encoding="utf-8") as file:
        json.dump(
            serializable_history,
            file,
            indent=2,
        )


def train_model() -> None:
    """Prepare the training set, train the CNN and save the model."""

    tf.keras.utils.set_random_seed(RANDOM_STATE)

    MODELS_DIRECTORY.mkdir(parents=True, exist_ok=True)
    RESULTS_DIRECTORY.mkdir(parents=True, exist_ok=True)

    print("--- Loading preprocessed training data ---")
    x_train, y_train = load_training_data()

    print(f"Training feature shape: {x_train.shape}")
    print(f"Original labels: {Counter(y_train)}")

    encoder = LabelEncoder()
    encoder.fit(CLASS_NAMES)

    y_train_encoded = encoder.transform(y_train)

    x_train_smote, y_train_smote = balance_training_data(
        x_train,
        y_train_encoded,
    )

    # Conv1D expects:
    # (samples, time steps, channels)
    x_train_cnn = x_train_smote.reshape(
        -1,
        x_train_smote.shape[1],
        1,
    )

    y_train_categorical = to_categorical(
        y_train_smote,
        num_classes=len(encoder.classes_),
    )

    print(f"\nCNN training shape: {x_train_cnn.shape}")
    print(f"Class order: {list(encoder.classes_)}")

    model = build_model(
        input_shape=(
            x_train_cnn.shape[1],
            x_train_cnn.shape[2],
        ),
        number_of_classes=len(encoder.classes_),
    )

    model.summary()

    callbacks = [
        EarlyStopping(
            monitor="val_loss",
            patience=10,
            restore_best_weights=True,
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=5,
            min_lr=1e-6,
        ),
    ]

    print("\n--- Training the 1D-CNN ---")

    history = model.fit(
        x_train_cnn,
        y_train_categorical,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_split=VALIDATION_SPLIT,
        callbacks=callbacks,
        verbose=1,
    )

    model.save(MODEL_FILE)
    np.save(CLASS_NAMES_FILE, encoder.classes_)
    save_training_history(history)

    print("\nTraining completed.")
    print(f"Model saved to: {MODEL_FILE}")
    print(f"Class names saved to: {CLASS_NAMES_FILE}")
    print(f"Training history saved to: {HISTORY_FILE}")


if __name__ == "__main__":
    train_model()
