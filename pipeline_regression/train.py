import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.ensemble import VotingRegressor

import joblib
import re
from config import CONFIG
from embedder import Embedder
from pipeline import build_pipeline


def get_category_column(df):
    if "category" in df.columns:
        return "category"
    if "dataset_name" in df.columns:
        return "dataset_name"
    raise ValueError("category_specific is set, but neither 'category' nor 'dataset_name' exists in the data")


def safe_filename_part(value):
    return re.sub(r"[^A-Za-z0-9._-]+", "_", str(value)).strip("_")


def main():
    # Make a string which adds the regressor names to the output files for clarity.
    regressors = CONFIG["regressor"]
    regressors_string = "_".join(regressors)
    print(regressors_string)

    # ---- LOAD DATA ----
    train_df = pd.read_csv(CONFIG["train_df_path"])
    test_df = pd.read_csv(CONFIG["test_df_path"])

    if CONFIG.get("category_specific") != "none":
        category = CONFIG["category_specific"]
        train_category_column = get_category_column(train_df)
        test_category_column = get_category_column(test_df)
        train_df = train_df[train_df[train_category_column] == category].reset_index(drop=True)
        test_df = test_df[test_df[test_category_column] == category].reset_index(drop=True)

    accuracy_fraction = CONFIG.get("target_column", "slm_accuracy_fraction")
    if accuracy_fraction not in train_df.columns or accuracy_fraction not in test_df.columns:
        raise ValueError(f"Target column '{accuracy_fraction}' must exist in both train and test data")

    train_df["complexity_score"] = train_df[accuracy_fraction].apply(lambda x: round(1.0 - x, 1))
    test_df["complexity_score"] = test_df[accuracy_fraction].apply(lambda x: round(1.0 - x, 1))

    target_column = "complexity_score"

    # ---- EMBEDDINGS ----
    embedder = Embedder(
        CONFIG["embedding_model"],
        custom_model_path=CONFIG.get("custom_embedding_path"),
        custom_backend=CONFIG.get("custom_embedding_backend", "auto"),
        custom_tokenizer_path=CONFIG.get("custom_tokenizer_path"),
    )

    X_train = embedder.encode(train_df["question"].tolist())
    X_test = embedder.encode(test_df["question"].tolist())

    y_train = train_df[target_column].astype(float).values
    y_test = test_df[target_column].astype(float).values

    if not regressors:
        raise ValueError("CONFIG['regressor'] must contain at least one regressor name")

    # ---- TRAIN MODEL ----
    if len(regressors) == 1:
        model = build_pipeline(regressor_name=regressors[0])
        model_label = regressors[0]
        artifact_name = f"trained_{regressors[0]}_regressor.joblib"
    else:
        estimators = [(reg_name, build_pipeline(regressor_name=reg_name)) for reg_name in regressors]
        model = VotingRegressor(estimators=estimators, verbose=True, n_jobs=-1)
        model_label = f"Average Voting Regressor ({len(regressors)} models)"
        artifact_name = f"trained_voting_regressor_{regressors_string}.joblib"

    model.fit(X_train, y_train)

    final_preds = np.clip(model.predict(X_test), 0.0, 1.0)
    mae = mean_absolute_error(y_test, final_preds)
    r2 = r2_score(y_test, final_preds)
    print(f"Validation MAE ({model_label}):", f"{mae:.4f}")
    print(f"Validation R^2 ({model_label}):", f"{r2:.4f}")

    # ---- BOXPLOT ----
    plot_df = pd.DataFrame({"actual": y_test, "predicted": final_preds})
    sorted_labels = sorted(plot_df["actual"].unique())

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.boxplot(data=plot_df, x="actual", y="predicted", order=sorted_labels, ax=ax)
    ax.set_xlabel("Actual complexity score")
    ax.set_ylabel("Predicted complexity score")
    ax.set_title(
        f"Model: {model_label}, Encoder: {CONFIG['embedding_model']}",
        fontsize=12,
        linespacing=1.6,
    )
    embedding_name = safe_filename_part(CONFIG["embedding_model"])
    fig.savefig(f"images/regression_boxplot_no_smiles/boxplot_{regressors_string}_{embedding_name}.png")

    # ---- SAVE MODEL ----
    if CONFIG.get("save_model", False):
        joblib.dump(model, artifact_name)



if __name__ == "__main__":
    main()
