import os
import warnings

from sentence_transformers import SentenceTransformer
from sklearn.model_selection import train_test_split

# Suppress TensorFlow logs
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["XLA_FLAGS"] = "--xla_gpu_autotune_level=0"
os.environ["GRPC_VERBOSITY"] = "ERROR"
os.environ["GLOG_minloglevel"] = "3"

warnings.filterwarnings("ignore")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, mean_absolute_error, r2_score
import tensorflow as tf
import torch
from tensorflow.keras import layers, models, regularizers
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

tf.get_logger().setLevel("ERROR")
tf.autograph.set_verbosity(0)

MODEL_NAME = "all-roberta-large-v1"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 32
EPOCHS = 100
RANDOM_SEED = 42

TRAIN_DF_PATH = "dataset/final_regressor/train.csv"
TEST_DF_PATH = "dataset/final_regressor/split_val.csv"
TARGET_COLUMN = "slm_accuracy_fraction"
SLM_CORRECT_COLUMN = "is_correct_qwen_7b"
ROUTING_THRESHOLD = 0.55


def route_to_slm(values, threshold):
    return (np.asarray(values) < threshold).astype(int)


# Load datasets
train_df = pd.read_csv(TRAIN_DF_PATH)
test_df = pd.read_csv(TEST_DF_PATH)

for column in [TARGET_COLUMN, "question"]:
    if column not in train_df.columns or column not in test_df.columns:
        raise ValueError(f"Column '{column}' must exist in both train and test data")

if SLM_CORRECT_COLUMN not in test_df.columns:
    raise ValueError(f"SLM correctness column '{SLM_CORRECT_COLUMN}' must exist in the test data")

# Regression target follows pipeline_regression: lower score means the SLM is more likely to be enough.
train_df["complexity_score"] = 1.0 - train_df[TARGET_COLUMN].astype(float)
test_df["complexity_score"] = 1.0 - test_df[TARGET_COLUMN].astype(float)

# Hold out part of the training set for early stopping; score routing on split_val like pipeline_regression.
train_split_df, val_df = train_test_split(
    train_df,
    test_size=0.2,
    random_state=RANDOM_SEED,
)

X_train_text = train_split_df["question"].tolist()
X_val_text = val_df["question"].tolist()
X_test_text = test_df["question"].tolist()

y_train = train_split_df["complexity_score"].values
y_val = val_df["complexity_score"].values
y_test = test_df["complexity_score"].values


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
    layers.Dense(512, activation="relu", kernel_regularizer=regularizers.l2(1e-4)),
    layers.Dropout(0.2),
    layers.Dense(64, activation="relu", kernel_regularizer=regularizers.l2(1e-4)),
    layers.Dense(1),
])

model.summary()

# Compile for regression
model.compile(
    optimizer="adam",
    loss="mse",
    metrics=["mae"],
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
        ReduceLROnPlateau(factor=0.5, patience=4),
    ],
    verbose=1,
)

# Predict
final_preds = np.clip(model.predict(X_test).flatten(), 0.0, 1.0)

# Regression metrics
mae = mean_absolute_error(y_test, final_preds)
r2 = r2_score(y_test, final_preds)
print(f"Validation MAE (dense regression): {mae:.4f}")
print(f"Validation R^2 (dense regression): {r2:.4f}")

# Plot training curves
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history["loss"], label="Train Loss")
plt.plot(history.history["val_loss"], label="Validation Loss")
plt.title("Loss Curve")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history["mae"], label="Train MAE")
plt.plot(history.history["val_mae"], label="Validation MAE")
plt.title("MAE Curve")
plt.xlabel("Epochs")
plt.ylabel("MAE")
plt.legend()

plt.tight_layout()
plt.savefig(f"training_curves_dense_regressor_{MODEL_NAME.replace(' ', '_')}.png")
plt.close()

# Routing confusion matrix based on Qwen 2.5 7B performance.
sent_to_slm = route_to_slm(final_preds, ROUTING_THRESHOLD)
slm_correct = test_df[SLM_CORRECT_COLUMN].astype(int).values

cm = confusion_matrix(slm_correct, sent_to_slm, labels=[0, 1])
tn, fp, fn, tp = cm.ravel()
print("Routing Confusion Matrix:")
print(cm)
print(f"True positives  (sent to SLM, SLM correct):   {tp}")
print(f"False positives (sent to SLM, SLM incorrect): {fp}")
print(f"False negatives (sent to LLM, SLM correct):   {fn}")
print(f"True negatives  (sent to LLM, SLM incorrect): {tn}")

routing_accuracy = np.mean(slm_correct == sent_to_slm)
print(f"Routing Accuracy at threshold {ROUTING_THRESHOLD:.2f}: {routing_accuracy:.4f}")

cm_row_pct = cm.astype(float) / cm.sum(axis=1, keepdims=True)
cm_row_pct = np.nan_to_num(cm_row_pct, nan=0.0)

fig, ax = plt.subplots(figsize=(8, 6))
im = ax.imshow(cm_row_pct, cmap="Blues", vmin=0.0, vmax=1.0)
fig.colorbar(im, ax=ax)

ax.set_xticks([0, 1], labels=["sent to LLM", "sent to SLM"])
ax.set_yticks([0, 1], labels=["SLM incorrect", "SLM correct"])
ax.set_xlabel("Route")
ax.set_ylabel("Qwen 2.5 7B Correctness")
ax.set_title(
    "Dense regressor, MAE: {:.4f}, R^2: {:.4f}, routing acc: {:.4f}, threshold: {:.2f}".format(
        mae,
        r2,
        routing_accuracy,
        ROUTING_THRESHOLD,
    )
)

for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        ax.text(
            j,
            i,
            f"{cm[i, j]}\n{cm_row_pct[i, j] * 100:.1f}%",
            ha="center",
            va="center",
            color="white" if cm_row_pct[i, j] > 0.5 else "black",
        )

plt.tight_layout()
os.makedirs("images/regression_confusion", exist_ok=True)
plt.savefig(f"images/regression_confusion/confusion_matrix_dense_{MODEL_NAME.replace(' ', '_')}.png")
plt.close()
