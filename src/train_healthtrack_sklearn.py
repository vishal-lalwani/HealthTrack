import os
import pandas as pd
import joblib
import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score

DATA_PATH = "data/healthtrack_dataset_preprocessed.csv"
df = pd.read_csv(DATA_PATH)

X = df.drop(columns=["condition"]).values
y = df["condition"].astype(int).values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

clf = Pipeline([
    ("scaler", StandardScaler()),
    ("mlp", MLPClassifier(
        hidden_layer_sizes=(128, 64),
        activation="relu",
        solver="adam",
        alpha=1e-4,
        learning_rate="adaptive",
        max_iter=800,
        early_stopping=True,
        n_iter_no_change=20,
        random_state=42
    ))
])

clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
y_proba = clf.predict_proba(X_test)[:, 1]

acc = accuracy_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_proba)

print(f"Test Accuracy: {acc*100:.2f}%")
print(f"ROC-AUC: {auc:.3f}")
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))

os.makedirs("models", exist_ok=True)
joblib.dump(clf, "models/sklearn_pipeline.joblib")

with open(f"training_report_sklearn_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.txt", "w") as f:
    f.write(f"Accuracy: {acc*100:.2f}%\n")
    f.write(f"ROC-AUC: {auc:.3f}\n")
    f.write(classification_report(y_test, y_pred))
