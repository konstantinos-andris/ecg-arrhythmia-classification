"""Download and preprocess ECG beats from the MIT-BIH database."""

from collections import Counter
from pathlib import Path

import numpy as np
import wfdb


TRAIN_RECORDS = ["100", "101", "102", "103", "104", "105", "115"]
TEST_RECORDS = ["106"]
ALL_RECORDS = TRAIN_RECORDS + TEST_RECORDS

VALID_SYMBOLS = {"N", "A", "V"}

WINDOW_BEFORE_R_PEAK = 100
WINDOW_AFTER_R_PEAK = 100
WINDOW_SIZE = WINDOW_BEFORE_R_PEAK + WINDOW_AFTER_R_PEAK

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIRECTORY = PROJECT_ROOT / "data" / "mit-bih"
PROCESSED_DIRECTORY = PROJECT_ROOT / "data" / "processed"


def record_is_available(record_id: str) -> bool:
    """Check whether the required local files exist for one record."""

    required_extensions = ("hea", "dat", "atr")

    return all(
        (DATA_DIRECTORY / f"{record_id}.{extension}").exists()
        for extension in required_extensions
    )


def download_dataset() -> None:
    """Download any required MIT-BIH records that are missing locally."""

    DATA_DIRECTORY.mkdir(parents=True, exist_ok=True)

    missing_records = [
        record_id
        for record_id in ALL_RECORDS
        if not record_is_available(record_id)
    ]

    if not missing_records:
        print("All required MIT-BIH records are already available.")
        return

    print(f"Downloading missing records: {missing_records}")

    wfdb.dl_database(
        db_dir="mitdb",
        dl_dir=str(DATA_DIRECTORY),
        records=missing_records,
    )

    print("Dataset download completed.")


def normalize_beat(beat: np.ndarray) -> np.ndarray:
    """Apply per-beat Z-score normalization."""

    mean_value = np.mean(beat)
    standard_deviation = np.std(beat)

    if standard_deviation <= 0:
        raise ValueError("The ECG beat has zero standard deviation.")

    return (beat - mean_value) / standard_deviation


def extract_beats(record_id: str) -> tuple[np.ndarray, np.ndarray]:
    """Extract normalized 200-sample beats from one MIT-BIH record."""

    record_path = DATA_DIRECTORY / record_id

    record = wfdb.rdrecord(str(record_path))
    annotation = wfdb.rdann(
        str(record_path),
        extension="atr",
    )

    # Use the first ECG channel, matching the original experiment.
    signal = record.p_signal[:, 0]

    beats: list[np.ndarray] = []
    labels: list[str] = []

    for position, symbol in zip(
        annotation.sample,
        annotation.symbol,
        strict=True,
    ):
        if symbol not in VALID_SYMBOLS:
            continue

        # Preserve the same boundary condition used in main_ai.py.
        if (
            position <= WINDOW_BEFORE_R_PEAK
            or position >= len(signal) - WINDOW_AFTER_R_PEAK
        ):
            continue

        start = position - WINDOW_BEFORE_R_PEAK
        end = position + WINDOW_AFTER_R_PEAK

        beat = signal[start:end]

        if len(beat) != WINDOW_SIZE:
            continue

        if np.std(beat) <= 0:
            continue

        normalized_beat = normalize_beat(beat)

        beats.append(normalized_beat)
        labels.append(symbol)

    if not beats:
        raise RuntimeError(
            f"No valid ECG beats were extracted from record {record_id}."
        )

    return np.asarray(beats), np.asarray(labels)


def load_records(
    record_ids: list[str],
) -> tuple[np.ndarray, np.ndarray]:
    """Load and combine beats from multiple patient records."""

    all_beats: list[np.ndarray] = []
    all_labels: list[np.ndarray] = []

    for record_id in record_ids:
        beats, labels = extract_beats(record_id)

        all_beats.append(beats)
        all_labels.append(labels)

        print(
            f"Record {record_id}: "
            f"{len(beats)} beats, "
            f"{dict(Counter(labels))}"
        )

    features = np.concatenate(all_beats, axis=0)
    targets = np.concatenate(all_labels, axis=0)

    # Keep features two-dimensional here:
    # (samples, 200)
    #
    # SMOTE must be applied before reshaping the data for Conv1D.
    return features, targets


def print_dataset_summary(
    features: np.ndarray,
    labels: np.ndarray,
    dataset_name: str,
) -> None:
    """Print dataset shape and class distribution."""

    print(f"\n{dataset_name} dataset:")
    print(f"Feature shape: {features.shape}")
    print(f"Class distribution: {dict(Counter(labels))}")


def prepare_dataset() -> None:
    """Create strict patient-independent training and test datasets."""

    download_dataset()
    PROCESSED_DIRECTORY.mkdir(parents=True, exist_ok=True)

    print("\nPreparing training records...")
    x_train, y_train = load_records(TRAIN_RECORDS)

    print("\nPreparing unseen test record...")
    x_test, y_test = load_records(TEST_RECORDS)

    print_dataset_summary(
        x_train,
        y_train,
        "Training",
    )

    print_dataset_summary(
        x_test,
        y_test,
        "Unseen test",
    )

    np.savez_compressed(
        PROCESSED_DIRECTORY / "train.npz",
        x=x_train,
        y=y_train,
    )

    np.savez_compressed(
        PROCESSED_DIRECTORY / "test.npz",
        x=x_test,
        y=y_test,
    )

    print("\nDataset preparation completed.")
    print(f"Files saved in: {PROCESSED_DIRECTORY}")


if __name__ == "__main__":
    prepare_dataset()
