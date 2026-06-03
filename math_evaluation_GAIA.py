import pandas as pd
import time
import requests
import json
from dotenv import load_dotenv
import os
import re
from math_verify import parse, verify
from math_verify.parser import LatexExtractionConfig, ExprExtractionConfig
import csv
import concurrent.futures


load_dotenv()
API_KEY = os.getenv("API_KEY")
URL = os.getenv("LLM_URL")
MODEL = "gpt-5"
DATASET_PATH = "dataset/MATH/MATH_train.csv"

HEADERS = {
    "Content-Type": "application/json",
    "api-key": API_KEY
}

NUM_THREADS = 10
failed_answer_extraction = 0


def build_prompt(question):
    prompt = """
Solve the following math problem step by step.

You may write full reasoning.

At the very end, place your final answer inside a LaTeX boxed expression, like this:

\\boxed{YOUR_ANSWER}

Where YOUR_ANSWER is the final answer expressed as simply as possible (e.g., a number, fraction, or simple expression).

Problem:\n
    """
    prompt += question
    return prompt


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


def extract_boxed(text):
    """Correctly extracts content from \\boxed{} handling nested braces."""
    idx = text.find(r'\boxed{')
    if idx == -1:
        return None

    start = idx + len(r'\boxed{')
    depth = 1
    i = start
    while i < len(text) and depth > 0:
        if text[i] == '{':
            depth += 1
        elif text[i] == '}':
            depth -= 1
        i += 1

    if depth != 0:
        return None  # unmatched braces

    return text[start:i-1].strip()


def extract_answer(response_text):
    global failed_answer_extraction
    result = extract_boxed(response_text)
    if result is None:
        failed_answer_extraction += 1
        print(response_text.replace("\n", " "))
    return result


def normalise(s):
    """Normalise a LaTeX string for fallback string comparison."""
    # Replace \tfrac with \frac
    s = s.replace(r'\tfrac', r'\frac')
    s = s.replace(r'\dfrac', r'\frac')
    s = s.replace(r'\textbf', r'\text')
    s = re.sub(r'\\frac(\d)(\d)', r'\\frac{\1}{\2}', s)
    s = s.replace(r'\$', '').replace('$', '').replace('£', '').replace('€', '')

    # Remove \text{...} wrappers but keep their content
    # e.g. \text{east} -> east, \text{ degrees} -> (removed as unit)
    # First remove pure unit \text{} blocks (preceded by a number or })
    s = re.sub(r'(?<=[0-9}])\s*\\text\{[^}]*\}', '', s)
    # Then unwrap remaining \text{...}
    s = re.sub(r'\\text\{([^}]*)\}', r'\1', s)

    # Remove common trailing unit words
    s = re.sub(r'\s*(degrees?|radians?|units?|cm|km|m|ft|s|kg|g)\s*$', '', s, flags=re.IGNORECASE)

    # Remove all whitespace and lowercase
    s = s.replace(' ', '').lower()

    return s.strip()


def _verify_in_process(extracted, correct_answer):
    gold = parse(
        correct_answer,
        extraction_config=[LatexExtractionConfig(), ExprExtractionConfig()]
    )
    prediction = parse(
        extracted,
        extraction_config=[LatexExtractionConfig(), ExprExtractionConfig()]
    )
    return verify(gold, prediction)


def compare_answers(model_response, correct_answer):
    if model_response is None:
        return False

    extracted = extract_boxed(model_response)
    if extracted is None:
        return False

    # Try Math-Verify first
    try:
        with concurrent.futures.ProcessPoolExecutor(max_workers=1) as executor:
            future = executor.submit(_verify_in_process, extracted, correct_answer)
            result = future.result(timeout=300)
        if result:
            return True
    except Exception as e:
        print(f"Math-Verify error: {e}")

    # Fallback: normalised string match
    return normalise(extracted) == normalise(correct_answer)


def save_results_to_csv(results, filename):
    with open(filename, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(["Unique ID", "Problem", "Model Response", "Extracted Answer", "Ground Truth", "Correct", "Latency", "Subject", "Level", "Input Tokens", "Output Tokens"])
        writer.writerows(results)


def process_problem(row):
    problem = row["problem"]
    ground_truth = row["answer"]
    subject = row["subject"]
    level = row["level"]
    unique_id = row["unique_id"]

    prompt = build_prompt(problem)
    try:
        model_response, time_taken, input_tokens, output_tokens = call_model(prompt)
        extracted_answer = extract_answer(model_response)
        is_correct = compare_answers(model_response, ground_truth)
    except Exception as e:
        model_response = str(e)
        extracted_answer = None
        is_correct = False
        time_taken = 0.0
        input_tokens = 0
        output_tokens = 0

    result = [
        unique_id,
        problem.replace("\n", " "),
        model_response.replace("\n", " "),
        extracted_answer,
        ground_truth,
        is_correct,
        time_taken,
        subject,
        level,
        input_tokens,
        output_tokens
    ]
    return result, extracted_answer, ground_truth, is_correct, time_taken, unique_id


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
            result, extracted_answer, ground_truth, is_correct, time_taken, unique_id = future.result()
            results.append(result)
            total += 1
            if is_correct:
                correct += 1
            total_response_time += time_taken
            print(f"Problem {i + 1}/{len(df)} | Correct: {is_correct} | Extracted: {extracted_answer} | Ground Truth: {ground_truth} | Time: {time_taken:.2f}s | Unique ID: {unique_id}")

    end_time = time.time()
    total_time = end_time - start_time
    save_results_to_csv(results, f"dataset/MATH_results/math_evaluation_results_{MODEL}.csv")
    print(f"\nModel: {MODEL}")
    print(f"Final Accuracy: {correct}/{total} = {correct/total:.2%}")
    print(f"Total Time: {total_time:.2f}s")
    print(f"Average Time per Problem: {total_response_time/total:.2f}s")
    print(f"Failed Extractions: {failed_answer_extraction}")


if __name__ == "__main__":
    main()
