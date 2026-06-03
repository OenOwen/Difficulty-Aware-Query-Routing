import os
import csv
import uuid
from flask import Flask, jsonify, request
from dotenv import load_dotenv
from flask_cors import CORS
import requests
import json
import time
import numpy as np
import joblib
from sentence_transformers import SentenceTransformer

load_dotenv()
API_KEY = os.getenv("API_KEY")
CSV_LOG_PATH = "responses.csv"
CSV_COLUMNS = ["response_id", "Question", "answer", "latency", "input_tokens", "output_tokens", "model", "is_good"]

def _one_line(text):
    return " ".join(str(text).splitlines())

def log_response_to_csv(response_id, question, answer, latency, input_tokens, output_tokens, model_used, is_good):
    model_label = "large" if model_used == "LLM" else "small"
    write_header = not os.path.exists(CSV_LOG_PATH)
    with open(CSV_LOG_PATH, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        if write_header:
            writer.writeheader()
        writer.writerow({
            "response_id": response_id,
            "Question": _one_line(question),
            "answer": _one_line(answer),
            "latency": latency,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "model": model_label,
            "is_good": is_good,
        })
LLM_URL = os.getenv("LLM_URL")
SLM_URL = os.getenv("SLM_URL")
MODEL = "gpt-5"
GPT5_INPUT_PRICE_PER_1M = 1.25
GPT5_OUTPUT_PRICE_PER_1M = 10.0
MU_COST = 100

LAMBDA_LATENCY = 0.11
C_ERR = 15

MANUAL_Q_S_SIMPLE = 0.76
MANUAL_Q_S_COMPLEX = 0.39
MANUAL_Q_L_SIMPLE = 0.93
MANUAL_Q_L_COMPLEX = 0.83

SLM_AVG_LATENCY = 3.10 
LLM_AVG_LATENCY = 23.81
LLM_AVG_IMPUT_TOKENS = 154.00 
LLM_AVG_OUTPUT_TOKENS = 1897.18

UTILITY_SETTING_DEFAULTS = {
    "LAMBDA_LATENCY": LAMBDA_LATENCY,
    "C_ERR": C_ERR,
    "MANUAL_Q_S_SIMPLE": MANUAL_Q_S_SIMPLE,
    "MANUAL_Q_S_COMPLEX": MANUAL_Q_S_COMPLEX,
    "MANUAL_Q_L_SIMPLE": MANUAL_Q_L_SIMPLE,
    "MANUAL_Q_L_COMPLEX": MANUAL_Q_L_COMPLEX,
    "SLM_AVG_LATENCY": SLM_AVG_LATENCY,
    "LLM_AVG_LATENCY": LLM_AVG_LATENCY,
    "LLM_AVG_IMPUT_TOKENS": LLM_AVG_IMPUT_TOKENS,
    "LLM_AVG_OUTPUT_TOKENS": LLM_AVG_OUTPUT_TOKENS,
}

ROUTER_MODEL = joblib.load("models/trained_svm_classifier.joblib")
EMBEDDER = SentenceTransformer("all-roberta-large-v1")

app = Flask(__name__)
CORS(app, origins=["http://localhost:5173"])

@app.get("/api/health")
def health():
    return jsonify({"status": "ok"})

@app.post("/api/chat")
def chat():
    payload = request.get_json(silent=True) or {}
    prompt = str(payload.get("prompt", "")).strip()
    use_utility = bool(payload.get("use_utility", True))
    utility_settings = parse_utility_settings(payload.get("utility_settings", {}))
    selected_model = normalize_selected_model(payload.get("selected_model"))
    print(f"Received prompt: {prompt}")
    answer, input_tokens, output_tokens, time_taken, model_used = decide_model(
        prompt,
        use_utility=use_utility,
        utility_settings=utility_settings,
        selected_model=selected_model,
    )
    cost = 0
    if model_used == "LLM":
        cost = compute_expected_cost(input_tokens, output_tokens)

    response_id = str(uuid.uuid4())
    model_size = "large model" if model_used == "LLM" else "small model"
    return jsonify(
        {
            "answer": answer,
            "time_taken": time_taken,
            "cost": cost,
            "model": model_used,
            "model_size": model_size,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "response_id": response_id,
        }
    )


@app.post("/api/route")
def route():
    payload = request.get_json(silent=True) or {}
    prompt = str(payload.get("prompt", "")).strip()
    use_utility = bool(payload.get("use_utility", True))
    utility_settings = parse_utility_settings(payload.get("utility_settings", {}))
    model_used = decide_model_route(
        prompt,
        use_utility=use_utility,
        utility_settings=utility_settings,
    )
    model_size = "large model" if model_used == "LLM" else "small model"
    return jsonify({"model": model_used, "model_size": model_size})


@app.post("/api/feedback")
def feedback():
    payload = request.get_json(silent=True) or {}
    feedback_choice = str(payload.get("feedback", "")).strip()
    response_id = str(payload.get("response_id", "")).strip()
    question = str(payload.get("question", "")).strip()
    answer = str(payload.get("answer", "")).strip()
    latency = payload.get("latency", 0)
    input_tokens = payload.get("input_tokens", 0)
    output_tokens = payload.get("output_tokens", 0)
    model_used = str(payload.get("model", "")).strip()
    is_good = "true" if feedback_choice == "up" else "false"
    print(f"Received feedback: {feedback_choice} for response_id: {response_id}")
    log_response_to_csv(response_id, question, answer, latency, input_tokens, output_tokens, model_used, is_good)
    return jsonify({"status": "ok"})


@app.post("/api/cascade/start")
def cascade_start():
    payload = request.get_json(silent=True) or {}
    prompt = str(payload.get("prompt", "")).strip()
    print(f"Received cascading prompt: {prompt}")
    answer, input_tokens, output_tokens, time_taken, model_used = decide_model(
        prompt,
        selected_model="SLM",
    )
    return jsonify(
        {
            "answer": answer,
            "time_taken": time_taken,
            "cost": 0,
            "model": model_used,
            "model_size": "small model",
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "response_id": str(uuid.uuid4()),
        }
    )


@app.post("/api/cascade/satisfied")
def cascade_satisfied():
    payload = request.get_json(silent=True) or {}
    response_id = str(payload.get("response_id", "")).strip() or str(uuid.uuid4())
    question = str(payload.get("question", "")).strip()
    answer = str(payload.get("answer", "")).strip()
    latency = payload.get("latency", 0)
    input_tokens = payload.get("input_tokens", 0)
    output_tokens = payload.get("output_tokens", 0)
    log_response_to_csv(response_id, question, answer, latency, input_tokens, output_tokens, "SLM", "true")
    return jsonify({"status": "ok"})


@app.post("/api/cascade/escalate")
def cascade_escalate():
    payload = request.get_json(silent=True) or {}
    question = str(payload.get("question", "")).strip()
    slm_response_id = str(payload.get("slm_response_id", "")).strip() or str(uuid.uuid4())
    slm_answer = str(payload.get("slm_answer", "")).strip()
    slm_latency = payload.get("slm_latency", 0)
    slm_input_tokens = payload.get("slm_input_tokens", 0)
    slm_output_tokens = payload.get("slm_output_tokens", 0)

    log_response_to_csv(
        slm_response_id,
        question,
        slm_answer,
        slm_latency,
        slm_input_tokens,
        slm_output_tokens,
        "SLM",
        "false",
    )

    answer, input_tokens, output_tokens, time_taken, model_used = decide_model(
        question,
        selected_model="LLM",
    )
    response_id = str(uuid.uuid4())
    log_response_to_csv(
        response_id,
        question,
        answer,
        time_taken,
        input_tokens,
        output_tokens,
        model_used,
        "true",
    )
    return jsonify(
        {
            "answer": answer,
            "time_taken": time_taken,
            "cost": compute_expected_cost(input_tokens, output_tokens),
            "model": model_used,
            "model_size": "large model",
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "response_id": response_id,
        }
    )


def parse_utility_settings(raw_settings):
    if not isinstance(raw_settings, dict):
        return UTILITY_SETTING_DEFAULTS.copy()

    settings = UTILITY_SETTING_DEFAULTS.copy()
    for key in settings:
        if key not in raw_settings:
            continue
        try:
            settings[key] = float(raw_settings[key])
        except (TypeError, ValueError):
            pass
    return settings

def normalize_selected_model(selected_model):
    if not selected_model:
        return None

    normalized = str(selected_model).strip().upper()
    if normalized in {"SLM", "LLM"}:
        return normalized
    return None

def decide_model_route(prompt, use_utility=True, utility_settings=None):
    utility_settings = utility_settings or UTILITY_SETTING_DEFAULTS
    prompt_embedding = EMBEDDER.encode([prompt])
    complexity_prob = get_complexity_probability(prompt_embedding)
    print(f"Complexity probability: {complexity_prob:.6f}")
    if use_utility:
        decision = bayesian_decision_2path(complexity_prob, utility_settings)
    else:
        decision = "big" if complexity_prob >= 0.5 else "small"

    return "SLM" if decision == "small" else "LLM"

def decide_model(prompt, use_utility=True, utility_settings=None, selected_model=None):
    model_used = selected_model or decide_model_route(
        prompt,
        use_utility=use_utility,
        utility_settings=utility_settings,
    )

    decision = "small" if model_used == "SLM" else "big"
    if decision == "small":
        start_time = time.time_ns()
        content, input_tokens, output_tokens = call_slm_model_api(prompt)
        end_time = time.time_ns()
        time_taken = (end_time - start_time) / 1e9
        return content, input_tokens, output_tokens, time_taken, "SLM"
    else:
        start_time = time.time_ns()
        content, input_tokens, output_tokens = call_llm_model_api(prompt)
        end_time = time.time_ns()
        time_taken = (end_time - start_time) / 1e9
        return content, input_tokens, output_tokens, time_taken, "LLM"

def call_llm_model_api(prompt):
    HEADERS = {
    "Content-Type": "application/json",
    "api-key": API_KEY
    }

    payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}]
    }

    response = requests.post(
                LLM_URL,
                headers=HEADERS,
                data=json.dumps(payload),
                timeout=300
            )
    
    response.raise_for_status()

    response_json = response.json()
    content = response_json["choices"][0]["message"]["content"].strip()
    input_tokens = response_json.get("usage", {}).get("prompt_tokens", 0)
    output_tokens = response_json.get("usage", {}).get("completion_tokens", 0)
    return content, input_tokens, output_tokens

def get_complexity_probability(prompt_embedding):
    # We use class label 1 as "complex". If classes are missing, default to index 1.
    complex_idx = 1
    if hasattr(ROUTER_MODEL, "classes_"):
        classes = np.asarray(ROUTER_MODEL.classes_)
        if 1 in classes:
            complex_idx = int(np.where(classes == 1)[0][0])

    if hasattr(ROUTER_MODEL, "predict_proba"):
        proba = ROUTER_MODEL.predict_proba(prompt_embedding)
        return float(proba[0][complex_idx])

    # Fallback for models without predict_proba: approximate from decision function.
    if hasattr(ROUTER_MODEL, "decision_function"):
        score = float(np.asarray(ROUTER_MODEL.decision_function(prompt_embedding)).reshape(-1)[0])
        return float(1.0 / (1.0 + np.exp(-score)))

    # Last resort: map hard class prediction to 0/1 probability.
    pred = int(np.asarray(ROUTER_MODEL.predict(prompt_embedding)).reshape(-1)[0])
    return 1.0 if pred == 1 else 0.0

def call_slm_model_api(prompt):
    HEADERS = {
    "Content-Type": "application/json",
    }
    payload = {
        "messages": [
            {"role": "user", "content": prompt}
        ],
    }
    response = requests.post(
        SLM_URL,
        headers=HEADERS,
        data=json.dumps(payload),
        timeout=300
    )
    response.raise_for_status()
    response_json = response.json()
    content = response_json["choices"][0]["message"]["content"].strip()
    input_tokens = response_json.get("usage", {}).get("prompt_tokens", 0)
    output_tokens = response_json.get("usage", {}).get("completion_tokens", 0)
    return content, input_tokens, output_tokens

def compute_expected_cost(input_tokens, output_tokens):
    total_cost = (input_tokens / 1_000_000) * GPT5_INPUT_PRICE_PER_1M + (output_tokens / 1_000_000) * GPT5_OUTPUT_PRICE_PER_1M
    # print(total_cost)
    return total_cost * MU_COST

def error_cost(cost, latency, q, lambd=LAMBDA_LATENCY, c_err=C_ERR):
    return cost + lambd * latency + c_err * (1 - q)

def compute_qs(p, settings=None):
    settings = settings or UTILITY_SETTING_DEFAULTS
    qs = (1-p) * settings["MANUAL_Q_S_SIMPLE"] + p * settings["MANUAL_Q_S_COMPLEX"]
    return qs

def compute_ql(p, settings=None):
    settings = settings or UTILITY_SETTING_DEFAULTS
    ql = (1-p) * settings["MANUAL_Q_L_SIMPLE"] + p * settings["MANUAL_Q_L_COMPLEX"]
    return ql
    

def bayesian_decision_2path(p, settings=None):
    settings = settings or UTILITY_SETTING_DEFAULTS
    p = float(np.asarray(p).reshape(-1)[-1])
    total_cost_small = 0
    total_cost_big = compute_expected_cost(
        settings["LLM_AVG_IMPUT_TOKENS"],
        settings["LLM_AVG_OUTPUT_TOKENS"],
    )
    ql = compute_ql(p, settings)
    qs = compute_qs(p, settings)
    error_big = error_cost(
        total_cost_big,
        settings["LLM_AVG_LATENCY"],
        ql,
        lambd=settings["LAMBDA_LATENCY"],
        c_err=settings["C_ERR"],
    )
    error_small = error_cost(
        total_cost_small,
        settings["SLM_AVG_LATENCY"],
        qs,
        lambd=settings["LAMBDA_LATENCY"],
        c_err=settings["C_ERR"],
    )
    if error_big < error_small:
        return "big"
    else:
        return "small"


if __name__ == "__main__":
    port = int(os.environ.get("FLASK_PORT", "8080"))
    app.run(host="127.0.0.1", port=port, debug=True)
