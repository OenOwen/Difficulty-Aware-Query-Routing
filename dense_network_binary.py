import os

from sentence_transformers import SentenceTransformer
from sklearn.model_selection import train_test_split

# Suppress TensorFlow logs
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["XLA_FLAGS"] = "--xla_gpu_autotune_level=0"
os.environ["GRPC_VERBOSITY"] = "ERROR"
os.environ["GLOG_minloglevel"] = "3"

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay, f1_score
import matplotlib.pyplot as plt
import tensorflow as tf

tf.get_logger().setLevel("ERROR")
tf.autograph.set_verbosity(0)

from tensorflow.keras import layers, models, regularizers
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import torch

import pipeline_classifier.embedder as embedder

MODEL_NAME = "all-MiniLM-L6-v2" 
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 32
EPOCHS = 100
RANDOM_SEED = 42

# Load datasets
train_df = pd.read_csv("dataset/final_binary/train.csv")
test_df = pd.read_csv("dataset/final_binary/val.csv")


# Binary labels: 1 if Accuracy Fraction < 0.75 (hard), else 0
train_df['complex'] = (train_df['slm_accuracy_fraction'] < 0.75).astype(int)
test_df['complex'] = (test_df['slm_accuracy_fraction'] < 0.75).astype(int)

# Train/validation split
X_train_text = train_df["question"].tolist()
y_train = train_df["complex"].values

X_val_text, X_test_text, y_val, y_test = train_test_split(test_df["question"].tolist(), test_df["complex"].values, test_size=0.5, random_state=RANDOM_SEED, stratify=test_df["complex"].values)


# Initialize embedder
encoder = SentenceTransformer(MODEL_NAME, device=DEVICE)

# Encode datasets
X_train = encoder.encode(X_train_text)
X_val = encoder.encode(X_val_text)
X_test = encoder.encode(X_test_text)

# Build model
input_dim = X_train.shape[1]

model = models.Sequential([
    layers.Input(shape=(input_dim,)),
    layers.Dense(1024, activation="relu", kernel_regularizer=regularizers.l2(1e-4)),
    layers.Dropout(0.3),
    layers.Dense(512, activation="relu", kernel_regularizer=regularizers.l2(1e-4)),
    layers.Dropout(0.2),
    layers.Dense(64, activation="relu", kernel_regularizer=regularizers.l2(1e-4)),
    layers.Dense(1, activation="sigmoid")
])

model.summary()

# Compile for classification
model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

# Train model
history = model.fit(
    X_train,
    y_train,
    validation_data=(X_val, y_val),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=[
        EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True),
        ReduceLROnPlateau(factor=0.5, patience=4)
    ],
    verbose=1
)

# Predict
y_pred = (model.predict(X_test).flatten() >= 0.5).astype(int)

# Classification metrics
acc = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
print(f"Test Accuracy: {acc:.4f}")
print(f"Test F1 Score: {f1:.4f}")

# Plot training curves
plt.figure(figsize=(12, 4))

# Loss Curve
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Loss Curve')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

# Accuracy Curve
plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Accuracy Curve')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.tight_layout()
plt.savefig(f"training_curves_{MODEL_NAME.replace(' ', '_')}.png")
plt.close()

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Simple", "Complex"])
disp.plot(cmap="Blues")
plt.title("Confusion Matrix - Acc: {:.4f}".format(acc))
plt.savefig(f"confusion_matrix_dense_{MODEL_NAME.replace(' ', '_')}.png")
plt.close()
