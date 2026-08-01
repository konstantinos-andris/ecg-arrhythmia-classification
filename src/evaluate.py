"""Evaluate the trained ECG model on the unseen MIT-BIH test patient."""

import json
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    auc,
    classification_report,
    confusion_matrix,
    roc_curve,
)
from tensorflow.keras.utils import to_categorical


PROJECT_ROOT = Path(__file__).resolve().parent.parent

PROCESSED_DIRECTORY = PROJECT_ROOT / "data" / "processed"
MODELS_DIRECTORY = PROJECT_ROOT / "models"
RESULTS_DIRECTORY = PROJECT_ROOT / "results"
FIGURES_DIRECTORY = RESULTS_DIRECTORY / "figures"

TEST_DATA_FILE = PROCESSED_DIRECTORY / "test.npz"
MODEL_FILE = MODELS_DIRECTORY / "ecg_arrhythmia_1d_cnn.keras"
CLASS_NAMES_FILE = MODELS_DIRECTORY / "class_names.npy"

METRICS_FILE = RESULTS_DIRECTORY / "evaluation_metrics.json"
REPORT_FILE = RESULTS_DIRECTORY / "classification_report.txt"
CONFUSION_MATRIX_FILE = FIGURES_DIRECTORY / "confusion_matrix.png"
ROC_CURVE_FILE = FIGURES_DIRECTORY / "roc_curves.png"


def load_test_data() -> tuple[np.ndarray, np.ndarray]:
    """Load the preprocessed unseen-patient test dataset."""

    if not TEST_DATA_FILE.exists():
        raise FileNotFoundError(
            "The processed test dataset was not found. "
            "Run data_preparation.py first."
        )

    dataset = np.load(TEST_DATA_FILE)

    features = dataset["x"].astype(np.float32)
    labels = dataset["y"].astype(str)

    if features.ndim != 2:
        raise ValueError(
            "Test features must have shape (samples, 200), "
            f"but received {features.shape}."
        )

    if len(features) != len(labels):
        raise ValueError(
            "The number of test beats does not match "
            "the number of test labels."
        )

    return features, labels


def load_class_names() -> np.ndarray:
    """Load the class order used during model training."""

    if not CLASS_NAMES_FILE.exists():
        raise FileNotFoundError(
            "The class names file was not found. "
            "Run train.py first."
        )

    return np.load(CLASS_NAMES_FILE).astype(str)


def encode_labels(
    labels: np.ndarray,
    class_names: np.ndarray,
) -> np.ndarray:
    """Convert string labels to their corresponding numeric indices."""

    class_to_index = {
        class_name: index
        for index, class_name in enumerate(class_names)
    }

    unknown_labels = sorted(set(labels) - set(class_names))

    if unknown_labels:
        raise ValueError(
            f"Unknown labels found in the test set: {unknown_labels}"
        )

    return np.asarray(
        [class_to_index[label] for label in labels],
        dtype=np.int64,
    )


def save_classification_report(
    true_labels: np.ndarray,
    predicted_labels: np.ndarray,
    class_names: np.ndarray,
) -> dict:
    """Create and save the classification report."""

    numeric_labels = list(range(len(class_names)))

    report_text = classification_report(
        true_labels,
        predicted_labels,
        labels=numeric_labels,
        target_names=class_names,
        zero_division=0,
    )

    report_dictionary = classification_report(
        true_labels,
        predicted_labels,
        labels=numeric_labels,
        target_names=class_names,
        zero_division=0,
        output_dict=True,
    )

    REPORT_FILE.write_text(
        report_text,
        encoding="utf-8",
    )

    print("\nClassification Report")
    print(report_text)

    return report_dictionary


def create_confusion_matrix(
    true_labels: np.ndarray,
    predicted_labels: np.ndarray,
    class_names: np.ndarray,
) -> np.ndarray:
    """Create and save the confusion matrix figure."""

    numeric_labels = list(range(len(class_names)))

    matrix = confusion_matrix(
        true_labels,
        predicted_labels,
        labels=numeric_labels,
    )

    figure, axis = plt.subplots(figsize=(8, 6))

    display = ConfusionMatrixDisplay(
        confusion_matrix=matrix,
        display_labels=class_names,
    )

    display.plot(
        cmap=plt.cm.Purples,
        ax=axis,
        values_format="d",
        colorbar=True,
    )

    axis.set_xlabel(
        "Predicted Label",
        fontsize=13,
    )

    axis.set_ylabel(
        "True Label",
        fontsize=13,
    )

    axis.set_title(
        "Unseen-Patient Confusion Matrix",
        fontsize=14,
    )

    figure.tight_layout()

    figure.savefig(
        CONFUSION_MATRIX_FILE,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(figure)

    return matrix


def create_roc_curves(
    true_labels_categorical: np.ndarray,
    predicted_probabilities: np.ndarray,
    class_names: np.ndarray,
) -> dict[str, float]:
    """Create one-vs-rest ROC curves for classes present in the test set."""

    figure, axis = plt.subplots(figsize=(9, 7))

    auc_scores: dict[str, float] = {}

    for class_index, class_name in enumerate(class_names):
        binary_true_labels = true_labels_categorical[:, class_index]

        positive_count = int(np.sum(binary_true_labels))
        negative_count = int(
            len(binary_true_labels) - positive_count
        )

        if positive_count == 0 or negative_count == 0:
            print(
                f"ROC skipped for class {class_name}: "
                "the test set does not contain both positive "
                "and negative examples."
            )
            continue

        false_positive_rate, true_positive_rate, _ = roc_curve(
            binary_true_labels,
            predicted_probabilities[:, class_index],
        )

        class_auc = auc(
            false_positive_rate,
            true_positive_rate,
        )

        auc_scores[class_name] = float(class_auc)

        axis.plot(
            false_positive_rate,
            true_positive_rate,
            linewidth=2,
            label=f"Class {class_name} (AUC = {class_auc:.4f})",
        )

    axis.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        linewidth=1.5,
        label="Random Guess",
    )

    axis.set_xlim(0.0, 1.0)
    axis.set_ylim(0.0, 1.05)

    axis.set_xlabel(
        "False Positive Rate",
        fontsize=13,
    )

    axis.set_ylabel(
        "True Positive Rate",
        fontsize=13,
    )

    axis.set_title(
        "One-vs-Rest ROC Curves",
        fontsize=14,
    )

    axis.grid(
        True,
        linestyle=":",
        alpha=0.6,
    )

    axis.legend(
        loc="lower right",
        fontsize=10,
    )

    figure.tight_layout()

    figure.savefig(
        ROC_CURVE_FILE,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(figure)

    return auc_scores


def evaluate_model() -> None:
    """Evaluate the trained model on the independent test patient."""

    if not MODEL_FILE.exists():
        raise FileNotFoundError(
            "The trained model was not found. "
            "Run train.py first."
        )

    RESULTS_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    FIGURES_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("--- Loading unseen-patient test data ---")

    x_test, y_test = load_test_data()
    class_names = load_class_names()

    print(f"Test feature shape: {x_test.shape}")
    print(f"Test class distribution: {Counter(y_test)}")
    print(f"Class order: {list(class_names)}")

    y_test_encoded = encode_labels(
        y_test,
        class_names,
    )

    y_test_categorical = to_categorical(
        y_test_encoded,
        num_classes=len(class_names),
    )

    x_test_cnn = x_test.reshape(
        -1,
        x_test.shape[1],
        1,
    )

    print("\n--- Loading trained model ---")

    model = tf.keras.models.load_model(MODEL_FILE)

    print("\n--- Evaluating on unseen patient 106 ---")

    test_loss, model_accuracy = model.evaluate(
        x_test_cnn,
        y_test_categorical,
        verbose=0,
    )

    predicted_probabilities = model.predict(
        x_test_cnn,
        verbose=0,
    )

    predicted_labels = np.argmax(
        predicted_probabilities,
        axis=1,
    )

    calculated_accuracy = accuracy_score(
        y_test_encoded,
        predicted_labels,
    )

    print(f"\nTest loss: {test_loss:.6f}")
    print(
        "Final unseen-patient accuracy: "
        f"{calculated_accuracy * 100:.2f}%"
    )

    report = save_classification_report(
        y_test_encoded,
        predicted_labels,
        class_names,
    )

    matrix = create_confusion_matrix(
        y_test_encoded,
        predicted_labels,
        class_names,
    )

    auc_scores = create_roc_curves(
        y_test_categorical,
        predicted_probabilities,
        class_names,
    )

    metrics = {
        "test_patient": "106",
        "test_samples": int(len(x_test)),
        "test_loss": float(test_loss),
        "keras_accuracy": float(model_accuracy),
        "accuracy": float(calculated_accuracy),
        "class_names": class_names.tolist(),
        "class_distribution": {
            str(label): int(count)
            for label, count in Counter(y_test).items()
        },
        "confusion_matrix": matrix.tolist(),
        "auc": auc_scores,
        "classification_report": report,
    }

    with METRICS_FILE.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            metrics,
            file,
            indent=2,
        )

    print("\nEvaluation completed.")
    print(f"Metrics saved to: {METRICS_FILE}")
    print(f"Report saved to: {REPORT_FILE}")
    print(f"Confusion matrix saved to: {CONFUSION_MATRIX_FILE}")
    print(f"ROC curves saved to: {ROC_CURVE_FILE}")


if __name__ == "__main__":
    evaluate_model()