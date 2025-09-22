import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler
import numpy as np
import os

df = pd.read_csv("C:/Coding/codes/HealthTrack/data/healthtrack_dataset.csv")

numeric_cols = ['age', 'systolic_bp', 'diastolic_bp', 'heart_rate',
                'temperature', 'wbc', 'rbc', 'glucose', 'creatinine']

noisy_cols = [c for c in df.columns if "noisy_feature" in c]

X_num = df[numeric_cols + noisy_cols]
y = df['condition']

vectorizer = TfidfVectorizer(max_features=200, stop_words='english')
X_text = vectorizer.fit_transform(df['clinical_note'])

svd = TruncatedSVD(n_components=20, random_state=42)
X_text_reduced = svd.fit_transform(X_text)

scaler = StandardScaler()
X_num_scaled = scaler.fit_transform(X_num)

X_final = np.hstack([X_num_scaled, X_text_reduced])

X_train, X_test, y_train, y_test = train_test_split(
    X_final, y, test_size=0.2, random_state=42, stratify=y
)

out_path = "C:/Coding/codes/HealthTrack/data/healthtrack_dataset_preprocessed.csv"
os.makedirs(os.path.dirname(out_path), exist_ok=True)

pd.DataFrame(X_final).assign(condition=y).to_csv(out_path, index=False)

print(f"Preprocessed data saved to {out_path}")
