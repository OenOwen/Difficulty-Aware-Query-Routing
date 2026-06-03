import pandas as pd
import time
import requests
import json
from dotenv import load_dotenv
import concurrent.futures
import os
import csv
import ast
import re
import sys
import subprocess
import tempfile

load_dotenv()
API_KEY = os.getenv("API_KEY")
URL = os.getenv("LLM_URL")
MODEL = "gpt-4o-europe"
DATASET_PATH = "dataset/APPS/merged.jsonl"

HEADERS = {
    "Content-Type": "application/json",
    "api-key": API_KEY
}

NUM_THREADS = 10
EXEC_TIMEOUT = 60


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

def build_prompt_stdin(question):
    return f"""You are solving a competitive programming problem.

Write a complete Python script that reads input from stdin and prints the answer to stdout.
Return ONLY valid Python code with no markdown, no explanations, no code fences.

Problem:
{question}"""


def build_prompt_function(question, fn_name, starter_code):
    starter = f"\nStart from this skeleton:\n{starter_code}\n" if starter_code and starter_code.strip() else ""
    return f"""You are solving a programming problem.

Implement the function `{fn_name}` in Python.{starter}
The function will be called directly with its arguments and its return value will be checked.
Return ONLY valid Python code with no markdown, no explanations, no code fences.
Include the full class/function definition.
Do NOT include example usage or test calls — only the class/function definition.

Problem:
{question}"""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def extract_code(text):
    """Strip markdown code fences if present."""
    match = re.search(r"```(?:python)?\s*\n(.*?)```", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return text.strip()


def normalize_output(s):
    """Strip trailing whitespace per line and overall."""
    lines = s.strip().splitlines()
    return "\n".join(line.rstrip() for line in lines)


def _tokens_float_match(a_tok, e_tok, tol=1e-6):
    """Return True if every space-separated token pair matches as floats."""
    a_parts = a_tok.split()
    e_parts = e_tok.split()
    if len(a_parts) != len(e_parts):
        return False
    for a, e in zip(a_parts, e_parts):
        try:
            af, ef = float(a), float(e)
            if abs(af - ef) > max(tol, tol * abs(ef)):
                return False
        except ValueError:
            return False
    return True


def outputs_match(actual, expected):
    """
    Compare stdout strings with progressive fallbacks:
    1. Exact normalized match
    2. Line-by-line float tolerance (for numeric outputs)
    """
    a = normalize_output(actual)
    e = normalize_output(expected)
    if a == e:
        return True

    a_lines = a.splitlines()
    e_lines = e.splitlines()
    if len(a_lines) == len(e_lines) and a_lines and e_lines:
        if all(_tokens_float_match(al, el) for al, el in zip(a_lines, e_lines)):
            return True

    return False


def _values_match(actual, expected):
    """
    Compare function return values.
    1. Direct equality
    2. Float tolerance (scalars and nested lists)
    """
    if actual == expected:
        return True
    if isinstance(expected, float) or isinstance(actual, float):
        try:
            af, ef = float(actual), float(expected)
            return abs(af - ef) <= max(1e-6, 1e-6 * abs(ef))
        except (TypeError, ValueError):
            pass
    if isinstance(expected, list) and isinstance(actual, list) and len(actual) == len(expected):
        return all(_values_match(a, e) for a, e in zip(actual, expected))
    return False


def _run_with_timeout(fn, args=(), timeout=EXEC_TIMEOUT):
    """Run fn(*args) in a thread, return (result, error_or_None)."""
    ex = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    future = ex.submit(fn, *args)
    try:
        return future.result(timeout=timeout), None
    except concurrent.futures.TimeoutError:
        return None, "TimeoutExpired"
    except Exception as e:
        return None, e
    finally:
        ex.shutdown(wait=False)


# ---------------------------------------------------------------------------
# Evaluation: stdin / stdout
# ---------------------------------------------------------------------------

def _run_stdin(code, stdin_text, timeout=EXEC_TIMEOUT):
    """Run code as a subprocess with stdin piped. Returns (stdout, stderr)."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False, encoding="utf-8") as f:
        f.write(code)
        tmpfile = f.name
    try:
        result = subprocess.run(
            [sys.executable, tmpfile],
            input=stdin_text,
            capture_output=True,
            text=True,
            timeout=timeout
        )
        return result.stdout, result.stderr
    except subprocess.TimeoutExpired:
        return None, "TimeoutExpired"
    finally:
        os.unlink(tmpfile)


def evaluate_stdin(code, inputs, outputs):
    """
    inputs: list of str (full stdin per test case)
    outputs: list of str (expected stdout)
    Returns (passed, total).
    """
    passed = 0
    for inp, expected in zip(inputs, outputs):
        stdout, _ = _run_stdin(code, inp)
        if stdout is not None and outputs_match(stdout, expected):
            passed += 1
    return passed, len(inputs)


def evaluate_list_stdin(code, inputs, outputs):
    """
    inputs: list of list-of-str  (lines joined into stdin)
    outputs: list of list-of-str (lines joined into expected stdout)
    """
    str_inputs = ["\n".join(parts) for parts in inputs]
    str_outputs = ["\n".join(parts) if isinstance(parts, list) else parts for parts in outputs]
    return evaluate_stdin(code, str_inputs, str_outputs)


# ---------------------------------------------------------------------------
# Evaluation: function call style
# ---------------------------------------------------------------------------

def _make_exec_namespace():
    """Return a namespace pre-loaded with common imports for LeetCode-style problems."""
    ns = {}
    exec(
        "from typing import List, Optional, Dict, Tuple, Set, Any, Union, Callable, Iterator\n"
        "from collections import defaultdict, Counter, deque, OrderedDict\n"
        "from functools import lru_cache, reduce, partial, cache\n"
        "from itertools import product, permutations, combinations, chain, accumulate, groupby\n"
        "import math, heapq, bisect, string, operator, sys, re, copy\n"
        "inf = math.inf\n",
        ns
    )
    return ns


def _exec_code(code):
    """Execute code in a namespace pre-loaded with common imports."""
    namespace = _make_exec_namespace()
    exec(compile(code, "<string>", "exec"), namespace)
    return namespace


def _call_fn(namespace, fn_name, args):
    """
    Call fn_name from namespace.
    Tries Solution().fn_name first (LeetCode style), then bare fn_name.
    """
    if "Solution" in namespace:
        fn = getattr(namespace["Solution"](), fn_name, None)
        if fn is not None:
            return fn(*args)
    if fn_name in namespace:
        return namespace[fn_name](*args)
    raise AttributeError(f"Function '{fn_name}' not found in namespace")


def evaluate_function(code, fn_name, inputs, outputs):
    """
    inputs: list of list (args per test case)
    outputs: list of expected return values
    Returns (passed, total).
    """
    namespace, err = _run_with_timeout(_exec_code, args=(code,), timeout=EXEC_TIMEOUT)
    if err is not None:
        return 0, len(inputs)

    passed = 0
    for args, expected in zip(inputs, outputs):
        actual, err = _run_with_timeout(_call_fn, args=(namespace, fn_name, args), timeout=EXEC_TIMEOUT)
        if err is None and _values_match(actual, expected):
            passed += 1
    return passed, len(inputs)


# ---------------------------------------------------------------------------
# Top-level evaluate dispatcher
# ---------------------------------------------------------------------------

def evaluate_apps(model_output, io_data):
    """
    Returns (passed, total) or None if there are no test cases.
    """
    code = extract_code(model_output)

    try:
        ast.parse(code)
    except SyntaxError:
        return 0, 1  # treat as one failed test

    fn_name = io_data.get("fn_name")
    inputs = io_data.get("inputs", [])
    outputs = io_data.get("outputs", [])

    if not inputs:
        return None  # unevaluatable

    if fn_name:
        return evaluate_function(code, fn_name, inputs, outputs)

    if isinstance(inputs[0], str):
        return evaluate_stdin(code, inputs, outputs)
    else:
        return evaluate_list_stdin(code, inputs, outputs)


# ---------------------------------------------------------------------------
# Model call
# ---------------------------------------------------------------------------

def call_model(prompt, temperature=0.0, max_retries=3):
    payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature
    }

    for attempt in range(max_retries):
        try:
            start_time = time.time_ns()

            response = requests.post(
                URL,
                headers=HEADERS,
                data=json.dumps(payload),
                timeout=500
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


# ---------------------------------------------------------------------------
# Result saving
# ---------------------------------------------------------------------------

def save_results_to_csv(results, filename):
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(["ID", "Difficulty", "Problem", "Model Response", "Correct", "Latency", "Input Tokens", "Output Tokens"])
        writer.writerows(results)


# ---------------------------------------------------------------------------
# Per-problem processing
# ---------------------------------------------------------------------------

def process_problem(row):
    problem = row["question"]
    prob_id = row["id"]
    difficulty = row.get("difficulty", "unknown")
    starter_code = row.get("starter_code", "") or ""

    io_raw = row.get("input_output")
    io_data = None
    if io_raw:
        io_data = json.loads(io_raw) if isinstance(io_raw, str) else io_raw

    fn_name = io_data.get("fn_name") if io_data else None

    if fn_name:
        prompt = build_prompt_function(problem, fn_name, starter_code)
    else:
        prompt = build_prompt_stdin(problem)

    try:
        model_response, time_taken, input_tokens, output_tokens = call_model(prompt)
        if io_data:
            eval_result = evaluate_apps(model_response, io_data)
        else:
            eval_result = None
    except Exception as e:
        model_response = str(e)
        eval_result = (0, 1)
        time_taken = 0.0
        input_tokens = 0
        output_tokens = 0

    if eval_result is None:
        is_correct = None
    else:
        passed, total = eval_result
        is_correct = passed == total

    result = [
        prob_id,
        difficulty,
        problem.replace("\n", " "),
        model_response.replace("\n", " "),
        is_correct,
        time_taken,
        input_tokens,
        output_tokens
    ]
    return result, model_response, is_correct, time_taken, prob_id


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    df = pd.read_json(DATASET_PATH, lines=True)
    results = []
    total = 0
    correct = 0
    total_response_time = 0.0
    start_time = time.time()

    with concurrent.futures.ThreadPoolExecutor(max_workers=NUM_THREADS) as executor:
        futures = [executor.submit(process_problem, row) for _, row in df.iterrows()]

        for i, future in enumerate(concurrent.futures.as_completed(futures)):
            result, model_response, is_correct, time_taken, id = future.result()
            results.append(result)
            total += 1
            if is_correct:
                correct += 1
            total_response_time += time_taken
            print(f"Problem {i + 1}/{len(df)} | Correct: {is_correct} | Time: {time_taken:.2f}s | ID: {id}")

    end_time = time.time()
    total_time = end_time - start_time
    save_results_to_csv(results, f"dataset/APPS_results/APPS_evaluation_results_{MODEL}.csv")
    print(f"\nModel: {MODEL}")
    print(f"Final Accuracy: {correct}/{total} = {correct/total:.2%}")
    print(f"Total Time: {total_time:.2f}s")
    print(f"Average Time per Problem: {total_response_time/total:.2f}s")


if __name__ == "__main__":
    main()
