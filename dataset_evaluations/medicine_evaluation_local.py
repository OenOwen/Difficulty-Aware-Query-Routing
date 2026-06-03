import pandas as pd
import time
import requests
import json
import re
import csv


URL = "http://localhost:8000/v1/chat/completions"
DATASET_PATH = "dataset/Medmcqa/medmcqa_train_reduced.csv"

HEADERS = {
    "Content-Type": "application/json",
}

failed_answer_extraction = 0
has_not_print_model_used = True
model_used = None

def call_model(prompt, temperature=0.0, top_p=1.0, max_retries=3):
    payload = {
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "temperature": temperature,
        "top_p": top_p
    }

    for attempt in range(max_retries):
        try:
            start_time = time.time_ns()

            response = requests.post(URL, headers=HEADERS, data=json.dumps(payload), timeout=300)

            response.raise_for_status()

            end_time = time.time_ns()
            time_taken = (end_time - start_time) / 1e9

            response_json = response.json()
            content = response_json["choices"][0]["message"]["content"].strip()
            input_tokens = response_json.get("usage", {}).get("prompt_tokens", 0)
            output_tokens = response_json.get("usage", {}).get("completion_tokens", 0)

            global has_not_print_model_used, model_used
            if has_not_print_model_used:
                model_used = response_json.get("model", "Unknown model")
                has_not_print_model_used = False

            return content, time_taken, input_tokens, output_tokens

        except Exception as e:
            if attempt == max_retries - 1:
                raise e
            sleep_time = 2 ** attempt
            print(f"Retry {attempt+1}/{max_retries} after error: {e}")
            time.sleep(sleep_time)


def build_prompt(question, options):
    letters = ['A','B','C','D',]
    formatted_options = "\n".join(
        f"{letters[i]}. {opt}" for i, opt in enumerate(options)
    )

    prompt = (
        f"The following are multiple choice questions (with answers). "
        f"Think step by step and then output the answer in the format of "
        f"\"The answer is (X)\" at the end.\n\n"
    )

    prompt += f"Question: {question}\nOptions:\n{formatted_options}\n"
    return prompt

def index_to_letter(index):
    if index is None:
        return None
    return chr(ord('A') + index)


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
    return None


def save_results_to_csv(results, filename):
    with open(filename, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(["ID", "Problem", "Model Response", "Extracted Answer", "Ground Truth", "Correct", "Latency", "Subject", "Input Tokens", "Output Tokens"])
        writer.writerows(results)


def process_problem(row):
    problem = row["question"]
    options = [row["opa"], row["opb"], row["opc"], row["opd"]]
    ground_truth = index_to_letter(row["cop"])
    subject = row["subject_name"]
    id = row["id"]

    prompt = build_prompt(problem, options)
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
        problem.replace("\n", " "),
        model_response.replace("\n", " "),
        extracted_answer,
        ground_truth,
        is_correct,
        time_taken,
        subject,
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

    for i, row in df.iterrows():
        result, extracted_answer, ground_truth, is_correct, time_taken, id = process_problem(row)
        results.append(result)
        total += 1
        if is_correct:
            correct += 1
        total_response_time += time_taken
        print(f"Problem {i + 1}/{len(df)} | Correct: {is_correct} | Extracted: {extracted_answer} | Ground Truth: {ground_truth} | Time: {time_taken:.2f}s | ID: {id}")

    end_time = time.time()
    total_time = end_time - start_time
    save_results_to_csv(results, f"dataset/Medmcqa_results/medmcqa_evaluation_results_{model_used.replace('/', '_')}.csv")
    print(f"\nModel: {model_used}")
    print(f"Final Accuracy: {correct}/{total} = {correct/total:.2%}")
    print(f"Total Time: {total_time:.2f}s")
    print(f"Average Time per Problem: {total_response_time/total:.2f}s")
    print(f"Failed Extractions: {failed_answer_extraction}")


if __name__ == "__main__":
    main()
