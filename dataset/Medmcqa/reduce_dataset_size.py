import pandas as pd

# File paths
input_path = "dataset/Medmcqa/medmcqa_train.csv"
output_path = "dataset/Medmcqa/medmcqa_train_reduced.csv"

# Load dataset
df = pd.read_csv(input_path)

# Remove rows where choice_type == "multi"
df_filtered = df[df["choice_type"] != "multi"]

# Stratified sampling: keep 10% from each subject_name group
df_reduced = (
    df_filtered
    .groupby("subject_name", group_keys=False)
    .apply(lambda x: x.sample(frac=0.1, random_state=42))
)

# Save to new CSV
df_reduced.to_csv(output_path, index=False)

print(f"Original rows: {len(df)}")
print(f"After removing 'multi': {len(df_filtered)}")
print(f"Final reduced rows (stratified): {len(df_reduced)}")