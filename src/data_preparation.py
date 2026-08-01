"""Download and preprocess ECG beats from the MIT-BIH database."""

from pathlib import Path

import numpy as np
import wfdb


TRAIN_RECORDS = ["100", "101", "102", "103", "104", "105", "115"]
TEST_RECORDS = ["106"]
ALL_RECORDS = TRAIN_RECORDS + TEST_RECORDS

LABEL_MAPPING = {
    "N": 0,  # Normal
    "A": 1,  # Atrial / supraventricular
    "V": 2,  # Ventricular
}

WINDOW_BEFORE_R_PEAK = 100
WINDOW_AFTER_R_PEAK = 100
WINDOW_SIZE = WINDOW_BEFORE_R_PEAK + WINDOW_AFTER_R_PEAK

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIRECTORY = PROJECT_ROOT / "data" / "mit-bih"
PROCESSED_DIRECTORY = PROJECT_ROOT / "data" / "processed"


def download_dataset() -> None:
    """Download the required MIT-BIH records when they are not available."""

    DATA_DIRECTORY.mkdir(parents=True, exist_ok=True)

    first_record_header = DATA_DIRECTORY / f"{ALL_RECORDS[0]}.hea"

    if first_record_header.exists():
        print("MIT-BIH records are already available.")
        return

    print("Downloading MIT-BIH records from PhysioNet...")

    wfdb.dl_database(
        db_dir="mitdb",
        dl_dir=str(DATA_DIRECTORY),
        records=ALL_RECORDS,
    )

    print("Download completed.")


def standardize_beat(beat: np.ndarray) -> np.ndarray:
    """Apply Z-score normalization independently to one ECG beat."""

    mean = np.mean(beat)
    standard_deviation = np.std(beat)

    if standard_deviation == 0:
        raise ValueError("The ECG beat has zero standard deviation.")

    return (beat - mean) / standard_deviation


def extract_beats(record_id: str) -> tuple[np.ndarray, np.ndarray]:
    """Extract 200-sample R-peak-centered beats from one MIT-BIH record."""

    record_path = DATA_DIRECTORY / record_id

    record = wfdb.rdrecord(str(record_path))
    annotation = wfdb.rdann(str(record_path), extension="atr")

    signal = record.p_signal[:, 0]

    beats: list[np.ndarray] = []
    labels: list[int] = []

    for position, symbol in zip(
        annotation.sample,
        annotation.symbol,
        strict=True,
    ):
        if symbol not in LABEL_MAPPING:
            continue

        start = position - WINDOW_BEFORE_R_PEAK
        end = position + WINDOW_AFTER_R_PEAK

        if start < 0 or end > len(signal):
            continue

        beat = signal[start:end]

        if len(beat) != WINDOW_SIZE or np.std(beat) == 0:
            continue

        normalized_beat = standardize_beat(beat)

        beats.append(normalized_beat.astype(np.float32))
        labels.append(LABEL_MAPPING[symbol])

    if not beats:
        raise RuntimeError(f"No valid beats were extracted from record {record_id}.")

    return (
        np.asarray(beats, dtype=np.float32),
        np.asarray(labels, dtype=np.int64),
    )


def load_records(record_ids: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """Load and combine beats from multiple patient records."""

    all_beats: list[np.ndarray] = []
    all_labels: list[np.ndarray] = []

    for record_id in record_ids:
        beats, labels = extract_beats(record_id)

        all_beats.append(beats)
        all_labels.append(labels)

        print(f"Record {record_id}: extracted {len(beats)} beats.")

    features = np.concatenate(all_beats, axis=0)
    targets = np.concatenate(all_labels, axis=0)

    # Add the channel dimension required by Conv1D:
    # (samples, time steps) -> (samples, time steps, channels)
    features = np.expand_dims(features, axis=-1)

    return features, targets


def print_class_distribution(labels: np.ndarray, dataset_name: str) -> None:
    """Display the number of beats belonging to every class."""

    inverse_mapping = {
        value: key for key, value in LABEL_MAPPING.items()
    }

    unique_labels, counts = np.unique(labels, return_counts=True)

    print(f"\n{dataset_name} class distribution:")

    for label, count in zip(unique_labels, counts, strict=True):
        class_name = inverse_mapping[int(label)]
        print(f"  {class_name}: {count}")


def prepare_dataset() -> None:
    """Create strict patient-independent training and test datasets."""

    download_dataset()
    PROCESSED_DIRECTORY.mkdir(parents=True, exist_ok=True)

    print("\nPreparing training records...")
    x_train, y_train = load_records(TRAIN_RECORDS)

    print("\nPreparing unseen test record...")
    x_test, y_test = load_records(TEST_RECORDS)

    print_class_distribution(y_train, "Training")
    print_class_distribution(y_test, "Test")

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
    print(f"Training shape: {x_train.shape}")
    print(f"Test shape: {x_test.shape}")
    print(f"Files saved inside: {PROCESSED_DIRECTORY}")


if __name__ == "__main__":
    prepare_dataset()
    