# HealthTrack – AI Driven Healthcare Data Analysis

This project builds and trains deep learning and ML models for patient condition classification and clinical note analysis.

## Features
- Preprocesses structured and unstructured patient data.
- TF-IDF + SVD dimensionality reduction on clinical notes.
- Neural network model (Keras) and MLP pipeline (Scikit-learn).
- Training reports saved automatically.

## Project Structure
HealthTrack-AI/
├── README.md                        # Project overview
├── requirements.txt                 # Dependencies
├── .gitignore                       # Ignore large/data files & cache
├── data/
│   ├── healthtrack_dataset.csv       # (dummy/sample only – not sensitive)
│   └── healthtrack_dataset_preprocessed.csv (optional, or generate via script)
├── src/
│   ├── preprocess_healthtrack.py    # Preprocessing pipeline
│   ├── train_healthtrack_keras.py   # Keras training
│   └── train_healthtrack_sklearn.py # Scikit-learn training
├── models/                          # Saved models (or empty with .gitkeep)
├── notebooks/                       # Optional: Jupyter notebooks for EDA
└── reports/
    └── training_reports/            # Saved text reports from training


## Installation
```bash
git clone https://github.com/vishal-lalwani/HealthTrack-AI.git
cd HealthTrack-AI
pip install -r requirements.txt
