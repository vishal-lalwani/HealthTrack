# HealthTrack – AI Driven Healthcare Data Analysis

This project builds and trains deep learning and ML models for patient condition classification and clinical note analysis.

## Features
- Preprocesses structured and unstructured patient data.
- TF-IDF + SVD dimensionality reduction on clinical notes.
- Neural network model (Keras) and MLP pipeline (Scikit-learn).
- Training reports saved automatically.

## Project Structure
```
HealthTrack-AI/
├── README.md
├── requirements.txt
├── .gitignore
├── data/
│   ├── healthtrack_dataset.csv
│   └── healthtrack_dataset_preprocessed.csv
├── src/
│   ├── preprocess_healthtrack.py
│   ├── train_healthtrack_keras.py
│   └── train_healthtrack_sklearn.py
├── models/
└── reports/
    └── training_reports/
```

## Installation
```bash
git clone https://github.com/vishal-lalwani/HealthTrack-AI.git
cd HealthTrack-AI
pip install -r requirements.txt
