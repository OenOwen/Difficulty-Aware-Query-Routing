import pandas as pd

# Load your CSV
directory = "slm_results/single_results/"
path = "single_results_unsloth_Qwen2.5-7B-Instruct-bnb-4bit.csv"
df = pd.read_csv(directory + "incorrect_accuracy/" + path)

# Recalculate accuracy
df['accuracy'] = (df['model_answer'] == df['correct_answer']).astype(float)

# Save the updated CSV
df.to_csv(directory + path, index=False)