"""Generate a saliency map for a ventricular ECG beat."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf


PROJECT_ROOT = Path(__file__).resolve().parent.parent

PROCESSED_DIRECTORY = PROJECT_ROOT / "data" / "processed"
MODELS_DIRECTORY = PROJECT_ROOT / "models"
RESULTS_DIRECTORY = PROJECT_ROOT / "results"
FIGURES_DIRECTORY = RESULTS_DIRECTORY / "figures"

TEST_DATA_FILE = PROCESSED_DIRECTORY / "test.npz"
MODEL_FILE = MODELS_DIRECTORY / "ecg_arrhythmia_1d_cnn.keras"
CLASS_NAMES_FILE = MODELS_DIRECTORY / "class_names.npy"

SALIENCY_FIGURE_FILE = FIGURES_DIRECTORY / "saliency_map_v.png"
SALIENCY_VALUES_FILE = RESULTS_DIRECTORY / "saliency_values_v.npz"

TARGET_CLASS = "V"
R_PEAK_POSITION = 100


def load_required_files() -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    tf.keras.Model,
]:
    """Load the test dataset, class names and trained model."""

    required_files = [
        TEST_DATA_FILE,
        MODEL_FILE,
        CLASS_NAMES_FILE,
    ]

    missing_files = [
        str(file_path)
        for file_path in required_files
        if not file_path.exists()
    ]

    if missing_files:
        missing_text = "\n".join(missing_files)

        raise FileNotFoundError(
            "The following required files were not found:\n"
            f"{missing_text}\n\n"
            "Run data_preparation.py and train.py first."
        )

    dataset = np.load(TEST_DATA_FILE)

    features = dataset["x"].astype(np.float32)
    labels = dataset["y"].astype(str)
    class_names = np.load(CLASS_NAMES_FILE).astype(str)

    if features.ndim != 2:
        raise ValueError(
            "Test features must have shape (samples, 200), "
            f"but received {features.shape}."
        )

    model = tf.keras.models.load_model(
        MODEL_FILE,
        compile=False,
    )

    return features, labels, class_names, model


def find_target_sample(
    features: np.ndarray,
    labels: np.ndarray,
    target_class: str,
) -> tuple[int, np.ndarray]:
    """Return the first test beat belonging to the target class."""

    target_indices = np.where(labels == target_class)[0]

    if len(target_indices) == 0:
        raise RuntimeError(
            f"No beat belonging to class {target_class} "
            "was found in the test dataset."
        )

    sample_index = int(target_indices[0])
    sample = features[sample_index]

    return sample_index, sample


def calculate_saliency(
    model: tf.keras.Model,
    sample: np.ndarray,
    target_class_index: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Calculate absolute input gradients for the target class."""

    sample_cnn = sample.reshape(1, -1, 1)

    ecg_tensor = tf.convert_to_tensor(
        sample_cnn,
        dtype=tf.float32,
    )

    with tf.GradientTape() as tape:
        tape.watch(ecg_tensor)

        predictions = model(
            ecg_tensor,
            training=False,
        )

        target_probability = predictions[:, target_class_index]

    gradients = tape.gradient(
        target_probability,
        ecg_tensor,
    )

    if gradients is None:
        raise RuntimeError(
            "TensorFlow could not calculate gradients "
            "for the selected ECG beat."
        )

    saliency = tf.abs(gradients).numpy().squeeze()
    prediction_values = predictions.numpy().squeeze()

    minimum = float(np.min(saliency))
    maximum = float(np.max(saliency))

    if maximum > minimum:
        saliency = (
            (saliency - minimum)
            / (maximum - minimum)
        )
    else:
        saliency = np.zeros_like(saliency)

    return saliency, prediction_values


def create_saliency_figure(
    ecg_signal: np.ndarray,
    saliency: np.ndarray,
) -> None:
    """Create and save the ECG saliency-map figure."""

    time_samples = np.arange(len(ecg_signal))

    figure, axis = plt.subplots(
        figsize=(10, 5),
    )

    axis.plot(
        time_samples,
        ecg_signal,
        color="lightgray",
        linewidth=1.5,
        zorder=1,
        label="ECG",
    )

    scatter = axis.scatter(
        time_samples,
        ecg_signal,
        c=saliency,
        cmap="jet",
        s=30,
        zorder=2,
    )

    axis.axvline(
        x=R_PEAK_POSITION,
        color="black",
        linestyle=":",
        linewidth=1.5,
        zorder=0,
        label="R-peak",
    )

    axis.set_xlabel(
        "Time Samples",
        fontsize=12,
    )

    axis.set_ylabel(
        "Z-Score",
        fontsize=12,
    )

    axis.set_title(
        "Saliency Map for a Ventricular Beat",
        fontsize=14,
    )

    axis.grid(
        True,
        linestyle=":",
        alpha=0.6,
    )

    axis.legend(
        loc="upper right",
        fontsize=10,
    )

    colorbar = figure.colorbar(
        scatter,
        ax=axis,
    )

    colorbar.set_label(
        "Normalized Saliency",
        fontsize=11,
    )

    figure.tight_layout()

    figure.savefig(
        SALIENCY_FIGURE_FILE,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(figure)


def generate_saliency_map() -> None:
    """Generate a saliency map for the first test-set V beat."""

    FIGURES_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    RESULTS_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("--- Loading model and unseen-patient data ---")

    features, labels, class_names, model = load_required_files()

    if TARGET_CLASS not in class_names:
        raise ValueError(
            f"Target class {TARGET_CLASS} is not present "
            f"in the stored class names: {list(class_names)}"
        )

    target_class_index = int(
        np.where(class_names == TARGET_CLASS)[0][0]
    )

    sample_index, sample = find_target_sample(
        features,
        labels,
        TARGET_CLASS,
    )

    saliency, prediction_values = calculate_saliency(
        model,
        sample,
        target_class_index,
    )

    predicted_class_index = int(
        np.argmax(prediction_values)
    )

    predicted_class = class_names[
        predicted_class_index
    ]

    predicted_probability = float(
        prediction_values[predicted_class_index]
    )

    target_probability = float(
        prediction_values[target_class_index]
    )

    print(f"Selected test sample index: {sample_index}")
    print(f"True class: {TARGET_CLASS}")
    print(f"Predicted class: {predicted_class}")
    print(
        "Predicted-class probability: "
        f"{predicted_probability:.4f}"
    )
    print(
        f"Class {TARGET_CLASS} probability: "
        f"{target_probability:.4f}"
    )

    create_saliency_figure(
        sample,
        saliency,
    )

    np.savez_compressed(
        SALIENCY_VALUES_FILE,
        ecg_signal=sample,
        saliency=saliency,
        sample_index=sample_index,
        true_class=TARGET_CLASS,
        predicted_class=predicted_class,
        prediction_probabilities=prediction_values,
        class_names=class_names,
    )

    print("\nSaliency-map generation completed.")
    print(f"Figure saved to: {SALIENCY_FIGURE_FILE}")
    print(f"Saliency values saved to: {SALIENCY_VALUES_FILE}")


if __name__ == "__main__":
    generate_saliency_map()