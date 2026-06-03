# Difficulty-Aware Query Routing

This repository contains the implementation of a Difficulty-Aware Query Router developed as part of a master thesis. The router estimates the difficulty of incoming queries and uses this prediction to decide whether a smaller or larger language model should answer the query. The goal is to reduce computational cost and latency while preserving answer quality. The project includes code for data processing, model training, routing logic, evaluation experiments, and an exploratory application for demonstrating the practical implementation of this router.


## Project Structure

- `application/` - Web application with a Flask backend and React/Vite frontend.
- `application/backend/` - API server, router model loading, model-calling logic, and feedback logging.
- `application/frontend/` - React TypeScript interface for router and cascade modes.
- `dataset/` - Evaluation datasets, model result CSVs, and prepared train/validation splits.
- `dataset_evaluations/` - Scripts for evaluating models on individual datasets.
- `router_evaluation/` - Scripts and notebooks for comparing router policies against baselines.
- `pipeline_classifier/` - Binary simple/complex router training pipeline.
- `pipeline_regression/` - Regression router training pipeline.
- `dense_network/` - Dense neural-network router experiments.
- `trained_routers/` - Saved router model artifacts.
- `images/` - Generated confusion matrices, histograms, and plots.
- `dataset_stats/` - Dataset and SLM result analysis notebooks.
- `embedding_stats/` - Embedding analysis notebooks.
- `test_vllm/` - vLLM router test script and notebook.
- `slm_results/` - SLM result aggregation and recalculation outputs.
- `vllm-sr-configs/` - vLLM server/router configuration files.

## Requirements

Use Python 3.10+ and Node.js 20+.

The backend requirements file installs Flask packages. `application/backend/main.py` also imports:

- `flask`
- `flask-cors`
- `python-dotenv`
- `requests`
- `numpy`
- `joblib`
- `scikit-learn`
- `sentence-transformers`

The frontend dependencies are managed with `npm` in `application/frontend/package.json`.

## How To Run The Application

Run the backend and frontend in separate terminals.

### 1. Configure The Backend

Create an environment file in `application/backend/.env`:

```env
API_KEY=your_llm_api_key
LLM_URL=https://your-llm-endpoint.example.com/chat/completions
SLM_URL=http://localhost:8000/v1/chat/completions
FLASK_PORT=8080
```

`LLM_URL` and `SLM_URL` must be OpenAI-compatible chat completion endpoints. The backend expects responses with `choices[0].message.content` and optional `usage.prompt_tokens` / `usage.completion_tokens` fields.

### 2. Start The Backend

Run these commands from the repository root:

```bash
cd application/backend
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install python-dotenv requests numpy joblib scikit-learn sentence-transformers
python main.py
```

The backend runs at `http://127.0.0.1:8080` by default. It must be started from `application/backend` because it loads `models/trained_svm_classifier.joblib` by relative path.

You can check it with:

```bash
curl http://127.0.0.1:8080/api/health
```

### 3. Start The Frontend

Open a second terminal and run:

```bash
cd application/frontend
npm install
VITE_API_URL=http://127.0.0.1:8080 npm run dev
```

Then open the Vite URL printed in the terminal, usually:

```text
http://localhost:5173
```

## Application Modes

The web app supports two main flows:

- Router mode: estimates prompt complexity and routes directly to either the SLM or LLM.
- Cascade mode: asks the SLM first, then allows escalation to the LLM if the SLM answer is not good enough.

Feedback is written by the backend to `application/backend/responses.csv`.

## Notes

- Several scripts use relative paths, so run them from the repository root unless the command example changes directory first.
- SentenceTransformer models may download on first use if they are not already cached.
- Some workflows require local or remote model-serving endpoints for the SLM and LLM.
