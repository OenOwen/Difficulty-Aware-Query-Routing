import pandas as pd
import time
import requests
import json
from dotenv import load_dotenv
import os
import re
import csv
import concurrent.futures


load_dotenv()
API_KEY = os.getenv("API_KEY")
URL = os.getenv("LLM_URL")
MODEL = "gpt-5"
DATASET_PATH = "dataset/Smiles/smile-dataset-reduced.csv"

HEADERS = {
    "Content-Type": "application/json",
    "api-key": API_KEY
}

NUM_THREADS = 20
failed_answer_extraction = 0


def call_model(prompt, temperature=0.0, max_retries=3):
    payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}]
    }

    for attempt in range(max_retries):
        try:
            start_time = time.time_ns()

            response = requests.post(
                URL,
                headers=HEADERS,
                data=json.dumps(payload),
                timeout=300
            )

            response.raise_for_status()

            end_time = time.time_ns()
            time_taken = (end_time - start_time) / 1e9

            response_json = response.json()
            content = response_json["choices"][0]["message"]["content"].strip()
            input_tokens = response_json.get("usage", {}).get("prompt_tokens", 0)
            output_tokens = response_json.get("usage", {}).get("completion_tokens", 0)

            return content, time_taken, input_tokens, output_tokens

        except Exception as e:
            if attempt == max_retries - 1:
                raise e
            sleep_time = 2 ** attempt
            print(f"Retry {attempt+1}/{max_retries} after error: {e}")
            time.sleep(sleep_time)


def build_prompt(smiles, question, options):
    letters = ['A', 'B', 'C', 'D']
    formatted_options = "\n".join(
        f"{letters[i]}. {opt}" for i, opt in enumerate(options)
    )

    prompt = (
        f"You are an expert chemist and pharmacologist. "
        f"You are given a molecule represented in SMILES notation and a multiple choice question about its properties.\n\n"
        f"Molecule (SMILES): {smiles}\n\n"
        f"Analyze the molecular structure and think step by step about the relevant chemical and pharmacological properties. "
        f"Then output the answer in the format \"The answer is (X)\" at the end.\n\n"
        f"Question: {question}\n"
        f"Options:\n{formatted_options}\n"
    )

    return prompt


def extract_answer(text):
    pattern = r"answer is \(?([A-D])\)?"
    match = re.search(pattern, text)
    if match:
        return match.group(1)
    return extract_again(text)


def extract_again(text):
    match = re.search(r'.*[aA]nswer:\s*([A-D])', text)
    if match:
        return match.group(1)
    return extract_final(text)


def extract_final(text):
    pattern = r"\b[A-D]\b(?!.*\b[A-JD]\b)"
    match = re.search(pattern, text, re.DOTALL)
    if match:
        return match.group(0)
    global failed_answer_extraction
    failed_answer_extraction += 1
    return None


def save_results_to_csv(results, filename):
    with open(filename, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(["ID", "SMILES", "Problem", "Model Response", "Extracted Answer", "Ground Truth", "Correct", "Latency", "Input Tokens", "Output Tokens"])
        writer.writerows(results)


def process_problem(row):
    smiles = row["SMILES"]
    problem = row["Question"]
    options = [row["Option_A"], row["Option_B"], row["Option_C"], row["Option_D"]]
    ground_truth = row["Answer"]
    id = row["CID"]

    prompt = build_prompt(smiles, problem, options)
    try:
        model_response, time_taken, input_tokens, output_tokens = call_model(prompt)
        extracted_answer = extract_answer(model_response)
        is_correct = extracted_answer == ground_truth
    except Exception as e:
        model_response = str(e)
        extracted_answer = None
        is_correct = False
        time_taken = 0.0
        input_tokens = 0
        output_tokens = 0

    result = [
        id,
        smiles,
        problem.replace("\n", " "),
        model_response.replace("\n", " "),
        extracted_answer,
        ground_truth,
        is_correct,
        time_taken,
        input_tokens,
        output_tokens
    ]
    return result, extracted_answer, ground_truth, is_correct, time_taken, id


def main():
    df = pd.read_csv(DATASET_PATH)
    results = []
    total = 0
    correct = 0
    total_response_time = 0.0
    start_time = time.time()

    with concurrent.futures.ThreadPoolExecutor(max_workers=NUM_THREADS) as executor:
        futures = [executor.submit(process_problem, row) for _, row in df.iterrows()]

        for i, future in enumerate(concurrent.futures.as_completed(futures)):
            result, extracted_answer, ground_truth, is_correct, time_taken, id = future.result()
            results.append(result)
            total += 1
            if is_correct:
                correct += 1
            total_response_time += time_taken
            print(f"Problem {i + 1}/{len(df)} | Correct: {is_correct} | Extracted: {extracted_answer} | Ground Truth: {ground_truth} | Time: {time_taken:.2f}s | ID: {id}")

    end_time = time.time()
    total_time = end_time - start_time
    save_results_to_csv(results, f"dataset/Smiles_results/smiles_evaluation_results_{MODEL}.csv")
    print(f"\nModel: {MODEL}")
    print(f"Final Accuracy: {correct}/{total} = {correct/total:.2%}")
    print(f"Total Time: {total_time:.2f}s")
    print(f"Average Time per Problem: {total_response_time/total:.2f}s")
    print(f"Failed Extractions: {failed_answer_extraction}")


if __name__ == "__main__":
    main()
