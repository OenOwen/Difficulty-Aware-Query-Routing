import pandas as pd
import os
from sklearn.model_selection import train_test_split

# Input files
files = [
    "math_evaluation_results_unsloth_gemma-3-12b-it-bnb-4bit.csv",
    "math_evaluation_results_unsloth_granite-3.2-8b-instruct-bnb-4bit.csv",
    "math_evaluation_results_unsloth_Llama-3.1-8B-Instruct-bnb-4bit.csv",
    "math_evaluation_results_unsloth_Qwen2.5-7B-Instruct-bnb-4bit.csv"
]

# Read all CSVs
dfs = [pd.read_csv(f) for f in files]

# Start with the first dataframe as base
base_df = dfs[0][[
    "Unique ID", "Problem", "Ground Truth", "Subject", "Level"
]].copy()

# Initialize accuracy sum column
base_df["Accuracy Sum"] = 0

# Add correctness from each file
for df in dfs:
    base_df["Accuracy Sum"] += df["Correct"]

# Compute accuracy fraction
base_df["Accuracy Fraction"] = base_df["Accuracy Sum"] / 4

# Select final columns
final_df = base_df[[
    "Unique ID", "Problem", "Ground Truth", "Subject", "Level", "Accuracy Fraction"
]]

# Ensure output directory exists
os.makedirs("Final_datasets", exist_ok=True)

# --- 80/20 Split ---
train_df, test_df = train_test_split(
    final_df,
    test_size=0.2,
    random_state=42,
    stratify=final_df["Accuracy Fraction"]
)

# Save splits
full_output_path = "Final_datasets/full_final_dataset.csv"
train_output_path = "Final_datasets/train_final_dataset.csv"
test_output_path = "Final_datasets/test_final_dataset.csv"

final_df.to_csv(full_output_path, index=False)
train_df.to_csv(train_output_path, index=False)
test_df.to_csv(test_output_path, index=False)

print(f"Full dataset saved to {full_output_path}")
print(f"Train dataset saved to {train_output_path}")
print(f"Test dataset saved to {test_output_path}")
