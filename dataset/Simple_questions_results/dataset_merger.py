from pathlib import Path

import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
GPT_PATH = BASE_DIR / "simple_questions_evaluation_results_gpt-5.csv"
QWEN_PATH = BASE_DIR / "simple_questions_evaluation_results_unsloth_Qwen2.5-7B-Instruct-bnb-4bit.csv"
OUT_PATH = BASE_DIR / "simple_questions_evaluation_results_merged.csv"


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize known key columns to lowercase names used for merging."""
    rename_map = {}
    if "ID" in df.columns:
        rename_map["ID"] = "id"
    if "Problem" in df.columns:
        rename_map["Problem"] = "problem"
    return df.rename(columns=rename_map)


def prefix_non_key_columns(df: pd.DataFrame, prefix: str) -> pd.DataFrame:
    """Prefix all columns except merge keys."""
    key_cols = {"id", "problem"}
    renamed = {col: f"{prefix}{col}" for col in df.columns if col not in key_cols}
    return df.rename(columns=renamed)


def main() -> None:
    gpt_df = pd.read_csv(GPT_PATH)
    qwen_df = pd.read_csv(QWEN_PATH)

    gpt_df = normalize_columns(gpt_df)
    qwen_df = normalize_columns(qwen_df)

    gpt_df = prefix_non_key_columns(gpt_df, "gpt_")
    qwen_df = prefix_non_key_columns(qwen_df, "qwen_")

    merged_df = pd.merge(gpt_df, qwen_df, on=["id", "problem"], how="inner")
    merged_df.to_csv(OUT_PATH, index=False)

    print(f"Merged file written to: {OUT_PATH}")
    print(f"Rows: {len(merged_df)} | Columns: {len(merged_df.columns)}")


if __name__ == "__main__":
    main()

