# ECG Arrhythmia Classification

A deep-learning project for ECG beat classification using a lightweight 1D Convolutional Neural Network and a strict inter-patient evaluation protocol.

The project was developed as part of undergraduate research at the University of Thessaly and was accepted for presentation at IEEE CIBCB 2026.

## Overview

The goal of this project is to classify ECG beats from the MIT-BIH Arrhythmia Database while evaluating the model on a completely unseen patient.

Unlike random beat-level splitting, the inter-patient protocol separates patients between training and testing, reducing the risk of data leakage and providing a more realistic evaluation of model generalization.

## Methodology

- Dataset: MIT-BIH Arrhythmia Database
- Signal preprocessing and beat extraction
- Z-score normalization
- Class balancing with SMOTE applied only to the training set
- Lightweight 1D Convolutional Neural Network
- Strict patient-independent training and evaluation
- Classification of normal and ventricular ectopic beats

## Patient Split

### Training records

`100, 101, 102, 103, 104, 105, 115`

### Test record

`106`

The test patient is not used during model training, normalization fitting or class balancing.

## Project Structure

```text
ecg-arrhythmia-classification/
├── README.md
├── requirements.txt
├── data/
│   └── README.md
├── docs/
├── models/
├── notebooks/
├── results/
│   ├── README.md
│   └── figures/
└── src/
    ├── __init__.py
    ├── data_preparation.py
    ├── model.py
    ├── train.py
    ├── evaluate.py
    └── saliency.py
    ```
## Technologies

- Python
- TensorFlow / Keras
- NumPy
- Pandas
- Scikit-learn
- Imbalanced-learn
- WFDB
- Matplotlib

## Installation

Clone the repository:

```bash
git clone https://github.com/konstantinos-andris/ecg-arrhythmia-classification.git
cd ecg-arrhythmia-classification
```

Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install the required packages:

```bash
pip install -r requirements.txt
```

## Data

The MIT-BIH Arrhythmia Database is not included in this repository.

The dataset can be accessed through PhysioNet and downloaded using the WFDB Python package.

Place local dataset files inside the `data/` directory. Dataset files are excluded from Git through `.gitignore`.

## Results

Final evaluation metrics, confusion matrices and ROC curves will be added after the research materials have been reviewed and prepared for public release.

## Research

This project is connected to research accepted for presentation at IEEE CIBCB 2026.

Additional publication information will be added after the official proceedings become publicly available.

## Author

**Konstantinos Andris**  
Undergraduate Student in Informatics and Telecommunications  
University of Thessaly

## License

The source code and research materials are currently shared for academic and portfolio purposes. A formal license will be selected after confirming the publication and collaboration requirements.
# Dataset

This project uses the MIT-BIH Arrhythmia Database, available through PhysioNet.

## Records Used

### Training Set

- 100
- 101
- 102
- 103
- 104
- 105
- 115

### Test Set

- 106

Record 106 is kept completely separate from the training process to support strict inter-patient evaluation.

## Download

The dataset is not included in this repository.

It can be downloaded from PhysioNet using the WFDB Python package:

```python
import wfdb

records = ["100", "101", "102", "103", "104", "105", "106", "115"]

wfdb.dl_database(
    "mitdb",
    dl_dir="data/mit-bih",
    records=records
)
```

## Important

Dataset files should remain inside the `data/` directory and must not be committed to GitHub.

The repository's `.gitignore` excludes these files while preserving this documentation file.
