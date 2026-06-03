# For AMD GPU use:
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"  # must be set before torch/sentence_transformers imports on this setup

import joblib
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer
import matplotlib.pyplot as plt
from pathlib import Path
import re

# ── Constants from validation set (do not change) ──────────────────────────────
SQ_SLM_FILE = "dataset/Simple_questions_results/simple_questions_evaluation_results_unsloth_Qwen2.5-7B-Instruct-bnb-4bit.csv"
SQ_LLM_FILE = "dataset/Simple_questions_results/simple_questions_evaluation_results_gpt-5.csv"

MMLU_SLM_FILE = "dataset/MMLU-Pro_results/MMLU-Pro_results_unsloth_Qwen2.5-7B-Instruct-bnb-4bit.csv"
MMLU_LLM_FILE = "dataset/MMLU-Pro_results/MMLU-Pro_results_gpt-5.csv"

# Estimated probabilities from validation split of training data:
Q_SLM_COMPLEX = 0.36
Q_SLM_SIMPLE = 0.73
Q_LLM_COMPLEX = 0.81
Q_LLM_SIMPLE = 0.93

SLM_VAL_LATENCY = 3.1007
LLM_VAL_LATENCY = 23.8062
LLM_VAL_INPUT_TOKENS = 154.0009
LLM_VAL_OUTPUT_TOKENS = 1897.1786

GPT5_INPUT_PRICE = 1.25 / 1e6
GPT5_OUTPUT_PRICE = 10.0 / 1e6

LAMBDA_LATENCY = 0.11
MU_COST = 100
C_ERR = 14

MODEL_DIR = "all-mpnet-base-v2"
ROUTER_PATH = "trained_routers/trained_svr_regressor.joblib"
USE_UTILITY = False
MODEL_THRESHOLD = 0.6

# ── Derived routing threshold (closed-form, no dataset needed) ─────────────────
# Route to LLM when p >= THRESHOLD, else to SLM.
# Derived by setting E[C_SLM | p] = E[C_LLM | p] and solving for p.
_llm_val_cost = (GPT5_INPUT_PRICE * LLM_VAL_INPUT_TOKENS + GPT5_OUTPUT_PRICE * LLM_VAL_OUTPUT_TOKENS) * MU_COST
_delta_c = _llm_val_cost + LAMBDA_LATENCY * (LLM_VAL_LATENCY - SLM_VAL_LATENCY)
_d_simple = Q_LLM_SIMPLE - Q_SLM_SIMPLE
_d_complex = Q_LLM_COMPLEX - Q_SLM_COMPLEX
THRESHOLD = ((_delta_c / C_ERR) - _d_simple) / (_d_complex - _d_simple)
print(f"Routing threshold p*: {THRESHOLD:.4f}  (route to LLM when predicted score > p*)")
print(f"USE_UTILITY={USE_UTILITY}  (model-only threshold={MODEL_THRESHOLD:.4f})")


# ── Helpers ────────────────────────────────────────────────────────────────────
def llm_cost_per_row(input_tokens, output_tokens):
    return GPT5_INPUT_PRICE * input_tokens + GPT5_OUTPUT_PRICE * output_tokens


def error_cost(monetary_cost, latency, quality):
    return monetary_cost * MU_COST + LAMBDA_LATENCY * latency + C_ERR * (1 - quality)


def expected_quality(p_complex, simple_quality, complex_quality):
    return simple_quality + p_complex * (complex_quality - simple_quality)


def choose_router_route(p, slm_latency, llm_latency, llm_cost):
    if USE_UTILITY:
        slm_quality = expected_quality(p, Q_SLM_SIMPLE, Q_SLM_COMPLEX)
        llm_quality = expected_quality(p, Q_LLM_SIMPLE, Q_LLM_COMPLEX)
        slm_err = error_cost(0.0, slm_latency, slm_quality)
        llm_err = error_cost(llm_cost, llm_latency, llm_quality)
        return "slm" if slm_err <= llm_err else "llm"

    return "llm" if p > MODEL_THRESHOLD else "slm"



def _slugify_dataset_name(name):
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


def save_confusion_matrix_figure(conf, label):
    """Save routing confusion matrix figure for a dataset."""
    out_dir = Path("figures")
    out_dir.mkdir(parents=True, exist_ok=True)
    filename = f"regression_routing_confusion_matrix_{_slugify_dataset_name(label)}.png"
    out_path = out_dir / filename

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(conf.values, cmap="Blues")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax.set_xticks(range(len(conf.columns)))
    ax.set_xticklabels([c.upper() for c in conf.columns])
    ax.set_yticks(range(len(conf.index)))
    ax.set_yticklabels([i.upper() for i in conf.index])
    ax.set_xlabel("Router decision")
    ax.set_ylabel("Ground truth route")
    ax.set_title(f"Regression Routing Confusion Matrix - {label}")

    max_count = conf.values.max() if conf.values.size else 0
    for r in range(conf.shape[0]):
        for c in range(conf.shape[1]):
            v = int(conf.values[r, c])
            text_color = "white" if v > max_count / 2 else "black"
            ax.text(c, r, str(v), ha="center", va="center", color=text_color)

    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"  Saved confusion matrix figure: {out_path}")


def save_p_value_histogram(scores, label):
    """Save histogram of trained router p-values/scores for a dataset."""
    out_dir = Path("figures")
    out_dir.mkdir(parents=True, exist_ok=True)
    filename = f"regression_router_p_values_histogram_{_slugify_dataset_name(label)}.png"
    out_path = out_dir / filename

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.hist(scores, bins=20, range=(0.0, 1.0), color="#4c78a8", edgecolor="white")
    ax.set_xlabel("p")
    ax.set_ylabel("Count")
    ax.set_title(f"Distribution of Difficulty Predictor Score - {label}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"  Saved p-value histogram: {out_path}")


def load_and_merge(slm_file, llm_file):
    """Load SLM and LLM result CSVs and merge on ID."""
    slm = pd.read_csv(slm_file)
    llm = pd.read_csv(llm_file)
    slm = slm.rename(columns={
        "Problem": "question",
        "Correct": "slm_correct",
        "Latency": "slm_latency",
        "Input Tokens": "slm_input_tokens",
        "Output Tokens": "slm_output_tokens",
    })[["ID", "question", "slm_correct", "slm_latency", "slm_input_tokens", "slm_output_tokens"]]
    llm = llm.rename(columns={
        "Correct": "llm_correct",
        "Latency": "llm_latency",
        "Input Tokens": "llm_input_tokens",
        "Output Tokens": "llm_output_tokens",
    })[["ID", "llm_correct", "llm_latency", "llm_input_tokens", "llm_output_tokens"]]
    df = slm.merge(llm, on="ID", how="inner").reset_index(drop=True)
    df["slm_correct"] = df["slm_correct"].map(lambda x: 1 if str(x).strip().lower() == "true" else 0)
    df["llm_correct"] = df["llm_correct"].map(lambda x: 1 if str(x).strip().lower() == "true" else 0)
    return df


def evaluate_dataset(df, probs, label):
    """Compute router, oracle, all-SLM, and all-LLM statistics."""
    n = len(df)
    records = {"router": [], "oracle": [], "oracle_cost": [], "all_slm": [], "all_llm": []}
    routing_rows = []

    for i, row in df.iterrows():
        p = float(probs[i])
        slm_ok = int(row["slm_correct"])
        llm_ok = int(row["llm_correct"])
        slm_lat = float(row["slm_latency"])
        llm_lat = float(row["llm_latency"])
        llm_in = float(row["llm_input_tokens"])
        llm_out = float(row["llm_output_tokens"])
        llm_cost_row = llm_cost_per_row(llm_in, llm_out)

        slm_err = error_cost(0.0, slm_lat, slm_ok)
        llm_err = error_cost(llm_cost_row, llm_lat, llm_ok)

        decisions = {
            "router": choose_router_route(p, slm_lat, llm_lat, llm_cost_row),
            "oracle": "slm" if slm_ok else "llm",
            "oracle_cost": "slm" if slm_err <= llm_err else "llm",
            "all_slm": "slm",
            "all_llm": "llm",
        }

        gt_route = "slm" if slm_ok else "llm"
        router_route = decisions["router"]
        routing_rows.append({
            "gt_route": gt_route,
            "router_route": router_route,
            "slm_correct": slm_ok,
            "llm_correct": llm_ok,
            "slm_latency": slm_lat,
            "llm_latency": llm_lat,
            "llm_cost": llm_cost_row,
        })

        for strategy, decision in decisions.items():
            if decision == "slm":
                lat = slm_lat
                cost = 0.0
                correct = slm_ok
            else:
                lat = llm_lat
                cost = llm_cost_row
                correct = llm_ok

            records[strategy].append({
                "decision": decision,
                "correct": correct,
                "latency": lat,
                "cost": cost,
                "error": error_cost(cost, lat, correct),
            })

    print(f"\n{'=' * 60}")
    print(f"Dataset: {label}  (n={n})")
    print(f"{'=' * 60}")

    for strategy, rows in records.items():
        df_r = pd.DataFrame(rows)
        n_llm = (df_r["decision"] == "llm").sum()
        print(f"\n  [{strategy}]")
        print(f"    LLM decisions:  {n_llm} / {n}  ({n_llm / n * 100:.1f}%)")
        print(f"    Accuracy:       {df_r['correct'].mean():.2%}")
        print(f"    Avg latency:    {df_r['latency'].mean():.2f} s")
        print(f"    Avg cost:       {df_r['cost'].mean() * MU_COST:.4f} cents")
        print(f"    Avg error:      {df_r['error'].mean():.4f}")


    # Routing confusion matrix based on route ground truth:
    # - If SLM is correct => should route to SLM
    # - Else => should route to LLM
    routing_df = pd.DataFrame(routing_rows)
    conf = pd.crosstab(
        routing_df["gt_route"],
        routing_df["router_route"],
        rownames=["Ground truth route"],
        colnames=["Router decision"],
        dropna=False,
    ).reindex(index=["slm", "llm"], columns=["slm", "llm"], fill_value=0)

    missed_escalation = int(conf.loc["llm", "slm"])   # should be LLM, sent to SLM
    failed_escalation = int(conf.loc["slm", "llm"])   # should be SLM, sent to LLM

    failed_mask = (routing_df["gt_route"] == "slm") & (routing_df["router_route"] == "llm")
    failed_cost_cents = float(routing_df.loc[failed_mask, "llm_cost"].sum() * MU_COST)
    failed_time_seconds = float((routing_df.loc[failed_mask, "llm_latency"] - routing_df.loc[failed_mask, "slm_latency"]).sum())

    # Missed escalations always fail by definition of gt_route (slm_correct == 0).
    missed_accuracy_failures = missed_escalation

    print("\n  [routing_confusion_matrix] (counts)")
    print(conf.to_string())
    # save_confusion_matrix_figure(conf, label)
    print(f"\n  Missed Escalation (gt=LLM, router=SLM): {missed_escalation}")
    print(f"  Failed Escalation (gt=SLM, router=LLM): {failed_escalation}")
    print(
        f"  Failed Escalations wasted about {failed_cost_cents:.4f} cents and {failed_time_seconds:.2f} s "
        f"(could have been saved by routing to SLM)."
    )
    print(
        f"  Missed Escalations caused {missed_accuracy_failures} failed questions "
        f"(routed to SLM when SLM was incorrect)."
    )


# ── Main ───────────────────────────────────────────────────────────────────────
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")

router = joblib.load(ROUTER_PATH)
embedder = SentenceTransformer(MODEL_DIR, device=device)

datasets = [
    ("Simple Questions", SQ_SLM_FILE, SQ_LLM_FILE),
    ("MMLU-Pro", MMLU_SLM_FILE, MMLU_LLM_FILE),
]

for label, slm_file, llm_file in datasets:
    try:
        df = load_and_merge(slm_file, llm_file)
    except FileNotFoundError as e:
        print(f"\nSkipping {label}: {e}")
        continue

    questions = df["question"].tolist()
    embeddings = embedder.encode(questions, batch_size=64, show_progress_bar=True)
    scores = router.predict(embeddings).clip(0.0, 1.0)

    evaluate_dataset(df, scores, label)
    save_p_value_histogram(scores, label)
