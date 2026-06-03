import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.ensemble import VotingClassifier

import joblib
import re
from config import CONFIG
from embedder import Embedder, augment_minority
from pipeline import build_pipeline


def downsample_df(df):
    neg_df = df[df["complex"] == 0]
    pos_df = df[df["complex"] == 1]

    target_pos_count = len(neg_df)
    if len(pos_df) <= target_pos_count:
        return df.copy()

    pos_sampled = pos_df.sample(n=target_pos_count, random_state=42)
    return pd.concat([neg_df, pos_sampled], ignore_index=True).sample(frac=1.0, random_state=42).reset_index(drop=True)


def vote_accuracy_threshold(df, threshold=0.6):
    df['complex'] = (df['slm_accuracy_fraction'] < threshold).astype(int)
    return df


def safe_filename_part(value):
    return re.sub(r"[^A-Za-z0-9._-]+", "_", str(value)).strip("_")


def main():
    # Make a string which adds the classifier names and voting strategy to the output files for clarity
    classifiers_string = "_".join(CONFIG["classifier"])
    print(classifiers_string)
    rng = CONFIG["random_seed"]

    # ---- LOAD DATA ----
    train_df = pd.read_csv(CONFIG["train_df_path"])
    test_df = pd.read_csv(CONFIG["test_df_path"])

    train_df = vote_accuracy_threshold(train_df, threshold=CONFIG["vote_accuracy_threshold"])
    test_df = vote_accuracy_threshold(test_df, threshold=CONFIG["vote_accuracy_threshold"])

    if CONFIG.get("category_specific") != "none":
        category = CONFIG["category_specific"]
        train_df = train_df[train_df["category"] == category].reset_index(drop=True)
        test_df = test_df[test_df["category"] == category].reset_index(drop=True)

    # ---- EMBEDDINGS ----
    embedder = Embedder(
        CONFIG["embedding_model"],
        custom_model_path=CONFIG.get("custom_embedding_path"),
        custom_backend=CONFIG.get("custom_embedding_backend", "auto"),
        custom_tokenizer_path=CONFIG.get("custom_tokenizer_path"),
    )

    X_train = embedder.encode(train_df["question"].tolist())
    X_test = embedder.encode(test_df["question"].tolist())

    y_train = train_df["complex"].values
    y_test = test_df["complex"].values

    # ---- AUGMENT MINORITY IF CONFIGURED ----
    if CONFIG.get("balance_strategy") == "augment":
        X_train, y_train = augment_minority(
            X_train,
            y_train,
            minority_label=CONFIG.get("minority_label", 0),
            factor=CONFIG.get("augment_factor", 2),
            noise_std=CONFIG.get("augment_noise_std", 0.01),
            random_seed=rng,
        )

    classifiers = CONFIG["classifier"]
    if not classifiers:
        raise ValueError("CONFIG['classifier'] must contain at least one classifier name")

    # ---- TRAIN MODEL ----
    if len(classifiers) == 1:
        model = build_pipeline(classifier_name=classifiers[0])
        model_label = classifiers[0]
        artifact_name = f"trained_{classifiers[0]}.joblib"
    else:
        estimators = [(clf_name, build_pipeline(classifier_name=clf_name)) for clf_name in classifiers]
        model = VotingClassifier(estimators=estimators, voting=CONFIG["voting_strategy"], verbose=1, n_jobs=-1)
        model_label = f"{CONFIG['voting_strategy'].title()} Voting Ensemble ({len(classifiers)} models)"
        artifact_name = f"trained_{CONFIG['voting_strategy']}_voting_ensemble_{classifiers_string}.joblib"

    model.fit(X_train, y_train)

    final_preds = model.predict(X_test)
    acc = accuracy_score(y_test, final_preds)
    f1 = f1_score(y_test, final_preds)
    print(f"Validation Accuracy ({model_label}):", f"{acc:.4f}")
    print(f"Validation F1 Score ({model_label}):", f"{f1:.4f}")

    # ---- CONFUSION MATRIX ----
    cm = confusion_matrix(y_test, final_preds)
    print("Confusion Matrix:")
    print(cm)

    cm_row_pct = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    cm_row_pct = np.nan_to_num(cm_row_pct, nan=0.0)

    annot = np.empty_like(cm, dtype=object)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            annot[i, j] = f"{cm[i, j]}\n{cm_row_pct[i, j] * 100:.1f}%"

    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm_row_pct,
        annot=annot,
        fmt="",
        cmap="Blues",
        vmin=0.0,
        vmax=1.0,
        xticklabels=model.classes_,
        yticklabels=model.classes_,
    )

    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title(
        f'Model: {model_label}, Encoder: {CONFIG["embedding_model"]}',
        fontsize=12,
        linespacing=1.6,
    )
    embedding_name = safe_filename_part(CONFIG["embedding_model"])
    plt.savefig(f"images/binary_confusion/confusion_matrix_{classifiers_string}_{embedding_name}.png")

    # ---- SAVE MODEL ----
    if CONFIG.get("save_model", True):
        joblib.dump(model, artifact_name)



if __name__ == "__main__":
    main()
