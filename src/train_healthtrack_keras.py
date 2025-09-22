import os
import pandas as pd
import datetime
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.callbacks import EarlyStopping, ModelCheckpoint

DATA_PATH = "data/healthtrack_dataset_preprocessed.csv"
df = pd.read_csv(DATA_PATH)

print(f"Loaded dataset with {len(df)} patients and {df.shape[1]} features.")

X = df.drop(columns=["condition"]).values
y = df["condition"].values

X_trainval, X_test, y_trainval, y_test = train_test_split(
    X, y, test_size=0.15, stratify=y, random_state=42
)
X_train, X_val, y_train, y_val = train_test_split(
    X_trainval, y_trainval, test_size=0.176, stratify=y_trainval, random_state=42
)

model = Sequential([
    Dense(128, activation="relu", input_shape=(X_train.shape[1],)),
    Dropout(0.3),
    Dense(64, activation="relu"),
    Dropout(0.3),
    Dense(1, activation="sigmoid")
])
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

os.makedirs("models", exist_ok=True)
ckpt = ModelCheckpoint("models/keras_model_best.h5", monitor="val_loss", save_best_only=True)
early = EarlyStopping(monitor="val_loss", patience=6, restore_best_weights=True)

model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=50, batch_size=64,
    callbacks=[early, ckpt],
    verbose=2
)

y_pred = (model.predict(X_test) >= 0.5).astype(int)
acc = accuracy_score(y_test, y_pred)
print(f"Test Accuracy: {acc*100:.2f}%")
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))

model.save("models/keras_model.h5")

with open(f"training_report_keras_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.txt", "w") as f:
    f.write(f"Accuracy: {acc*100:.2f}%\n")
    f.write(classification_report(y_test, y_pred))
