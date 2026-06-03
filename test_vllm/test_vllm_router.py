import os

import pandas as pd
from collections import Counter
import requests

# Semantic router proxy base URL (must be the same backend used by the dashboard)
ROUTER_BASE_URL = os.getenv("ROUTER_URL")
CHAT_COMPLETIONS_PATH = "/v1/chat/completions"

# --- User config ---
dataset_path = "dataset/MMLU-Pro_results/MMLU_Pro_results_merged.csv"
question_column = "problem"


def extract_route(data: dict):
    """Try common routing metadata locations in the vLLM response."""
    possible_paths = [
        ("route",),
        ("routing", "selected_model"),
        ("routing", "model"),
        ("metadata", "route"),
        ("metadata", "selected_model"),
        ("model",),
    ]

    for path in possible_paths:
        try:
            value = data
            for key in path:
                value = value[key]
            return value
        except Exception:
            continue
    return None


def assert_router_metadata(data: dict):
    if extract_route(data) is None:
        raise RuntimeError(
            "No routing metadata detected in response. "
            "Check that ROUTER_BASE_URL points to the semantic router proxy used by the dashboard."
        )

def route_bucket(route_value):
    if route_value is None:
        raise RuntimeError("Route value is missing; expected small or big.")

    r = str(route_value).strip().lower()

    # Map router output into only two supported buckets.
    if any(tag in r for tag in ["small", "slm", "qwen", "7b"]):
        return "slm"
    if any(tag in r for tag in ["big", "llm", "gpt", "large"]):
        return "llm"

    raise RuntimeError(f"Unexpected route value '{route_value}'. Expected small/big route.")


def to_float(value, default=0.0):
    try:
        if pd.isna(value):
            return default
        return float(value)
    except Exception:
        return default


def to_bool(value):
    if isinstance(value, bool):
        return value
    if pd.isna(value):
        return False

    normalized = str(value).strip().lower()
    if normalized in {"true", "1", "yes", "y", "correct"}:
        return True
    if normalized in {"false", "0", "no", "n", "incorrect"}:
        return False

    raise ValueError(f"Cannot parse boolean value: {value!r}")


def normalize_column_name(column):
    return (
        str(column)
        .strip()
        .lower()
        .replace(" ", "_")
        .replace("-", "_")
    )


def resolve_column(df: pd.DataFrame, wanted: str):
    normalized = {normalize_column_name(c): c for c in df.columns}
    return normalized.get(normalize_column_name(wanted))


def resolve_metric_columns(df: pd.DataFrame):
    needed = {
        "slm_correct": "Correct_qwen",
        "slm_latency": "Latency_qwen",
        "llm_correct": "Correct_gpt5",
        "llm_latency": "Latency_gpt5",
        "llm_input_tokens": "Input Tokens_gpt5",
        "llm_output_tokens": "Output Tokens_gpt5",
    }

    resolved = {}
    missing = []
    for key, wanted_name in needed.items():
        col = resolve_column(df, wanted_name)
        if col is None:
            missing.append(wanted_name)
        else:
            resolved[key] = col

    if missing:
        raise ValueError(
            "Missing expected columns in merged dataset: "
            + ", ".join(missing)
            + f"\nAvailable columns: {list(df.columns)}"
        )

    return resolved


df = pd.read_csv(dataset_path)
resolved_question_column = resolve_column(df, question_column)
if resolved_question_column is None:
    raise ValueError(f"Column '{question_column}' not found. Available columns: {list(df.columns)}")

metric_cols = resolve_metric_columns(df)

working_df = df[df[resolved_question_column].notna()].copy()
if working_df.empty:
    raise ValueError("No questions found after dropping NA values.")

counts = Counter()
raw_routes = Counter()
correct_counts = Counter()

# Requested aggregates
slm_latency_total = 0.0
llm_latency_total = 0.0
llm_input_tokens_total = 0.0
llm_output_tokens_total = 0.0

for i, (_, row) in enumerate(working_df.iterrows(), start=1):
    question = str(row[resolved_question_column])

    payload = {
        "model": "auto",
        "messages": [{"role": "user", "content": question}],
        "temperature": 0.2,
        "max_tokens": 300,
    }

    response = requests.post(
        f"{ROUTER_BASE_URL}{CHAT_COMPLETIONS_PATH}",
        headers={"Content-Type": "application/json"},
        json=payload,
        timeout=120,
    )
    response.raise_for_status()
    data = response.json()

    assert_router_metadata(data)
    route = extract_route(data)
    bucket = route_bucket(route)

    counts[bucket] += 1
    raw_routes[str(route)] += 1

    if bucket == "slm":
        slm_latency_total += to_float(row[metric_cols["slm_latency"]])
        correct_counts[bucket] += int(to_bool(row[metric_cols["slm_correct"]]))
    elif bucket == "llm":
        llm_latency_total += to_float(row[metric_cols["llm_latency"]])
        llm_input_tokens_total += to_float(row[metric_cols["llm_input_tokens"]])
        llm_output_tokens_total += to_float(row[metric_cols["llm_output_tokens"]])
        correct_counts[bucket] += int(to_bool(row[metric_cols["llm_correct"]]))
    if i % 25 == 0 or i == len(working_df):
        print(f"Processed {i}/{len(working_df)} questions...")


total = len(working_df)
slm_n = counts.get("slm", 0)
llm_n = counts.get("llm", 0)

routed_n = slm_n + llm_n
if routed_n != total:
    raise RuntimeError(f"Expected all rows to route to small/big, but got {routed_n}/{total}.")
total_time = slm_latency_total + llm_latency_total
avg_time_per_question = total_time / routed_n if routed_n else 0.0
correct_total = correct_counts["slm"] + correct_counts["llm"]
accuracy = correct_total / routed_n if routed_n else 0.0

print("========== ROUTING SUMMARY ==========")
print(f"Total questions: {total}")
print(f"SLM route:      {slm_n} ({slm_n/total:.2%})")
print(f"LLM route:      {llm_n} ({llm_n/total:.2%})")
print(f"Routed total:   {routed_n} ({routed_n/total:.2%})")

print("\n========== ACCURACY SUMMARY ==========")
print(f"SLM correct:    {correct_counts['slm']}/{slm_n} ({correct_counts['slm']/slm_n:.2%})" if slm_n else "SLM correct:    0/0 (n/a)")
print(f"LLM correct:    {correct_counts['llm']}/{llm_n} ({correct_counts['llm']/llm_n:.2%})" if llm_n else "LLM correct:    0/0 (n/a)")
print(f"Total correct:  {correct_total}/{routed_n} ({accuracy:.2%})")

print("\n========== COST/TIME SUMMARY ==========")
print(f"SLM total latency:      {slm_latency_total:.4f}")
print(f"LLM total latency:      {llm_latency_total:.4f}")
print(f"Average time/question:  {avg_time_per_question:.4f}")
print(f"Total time:             {total_time:.4f}")
print(f"Total input tokens:     {llm_input_tokens_total:.0f}")
print(f"Total output tokens:    {llm_output_tokens_total:.0f}")

print("\nRaw route values seen:")
for route_name, n in raw_routes.most_common():
    print(f"  {route_name}: {n}")

GPT5_INPUT_PRICE_PER_1M = 1.25
GPT5_OUTPUT_PRICE_PER_1M = 10.0

def calculate_cost(input_tokens, output_tokens, scalar=1.0):
    input_cost = (input_tokens / 1_000_000) * GPT5_INPUT_PRICE_PER_1M
    output_cost = (output_tokens / 1_000_000) * GPT5_OUTPUT_PRICE_PER_1M
    return (input_cost + output_cost) * scalar

print("\n========== COST ESTIMATE ==========")
estimated_cost = calculate_cost(llm_input_tokens_total, llm_output_tokens_total, 1)
print(f"Estimated total cost: ${estimated_cost:.4f}")
print(f"Average cost per question: ${(estimated_cost / routed_n):.6f}" if routed_n else "Average cost per question: n/a")