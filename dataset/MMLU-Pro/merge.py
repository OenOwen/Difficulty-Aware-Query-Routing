import pandas as pd

qwen_df = pd.read_csv("dataset/MMLU-Pro_results/MMLU-Pro_results_unsloth_Qwen2.5-7B-Instruct-bnb-4bit.csv")
gpt5_df = pd.read_csv("dataset/MMLU-Pro_results/MMLU-Pro_results_gpt-5.csv")

shared_cols = ["ID", "Problem", "Ground Truth", "Category"]
per_model_cols = ["Correct", "Latency", "Input Tokens", "Output Tokens"]

merged = qwen_df[shared_cols + per_model_cols].merge(
    gpt5_df[["ID"] + per_model_cols],
    on="ID",
    suffixes=("_qwen", "_gpt5")
)

merged.to_csv("dataset/MMLU-Pro_results/MMLU_Pro_results_merged.csv", index=False)
print(f"Merged length: {len(merged)}")
print(f"Columns: {list(merged.columns)}")
