# AutoKaggle

AutoKaggle is a CLI tool that scaffolds and runs baseline Kaggle competition pipelines. Provide a competition URL, and AutoKaggle will download data, profile it, draft modeling scripts (via an LLM or local templates), and optionally execute the generated pipeline locally.

## Features

- **CLI-first workflow** for running a Kaggle baseline end-to-end.
- **Kaggle API integration** to fetch competition metadata and datasets.
- **Data profiling** to summarize schema, missingness, and target inference.
- **LLM-assisted strategy + code generation** (Gemini) with a local fallback.
- **Local execution** that produces `model.joblib`, `metrics.json`, and `submission.csv`.
- **Run metadata + logs** to track progress and artifacts.

## Requirements

- Python 3.10+
- Kaggle API credentials (see Configuration)
- Optional: Google API key for Gemini-based strategy + code generation

## Installation

```bash
pip install -r requirements.txt
```

## Quick start

```bash
# Authenticate with Kaggle (once per environment)
export KAGGLE_API_TOKEN="username:key"   # or JSON from kaggle.json
export GOOGLE_API_KEY="your_google_api_key"  # optional for LLM steps

# Run a competition end-to-end
python -m autokaggle run https://www.kaggle.com/competitions/{competition}

# Inspect a run
python -m autokaggle status {run_id}
python -m autokaggle logs {run_id}
```

## Configuration

AutoKaggle reads configuration from environment variables:

| Variable | Purpose |
| --- | --- |
| `KAGGLE_API_TOKEN` | Kaggle API token (`username:key` or JSON from `kaggle.json`). |
| `KAGGLE_USERNAME` / `KAGGLE_KEY` | Alternative to `KAGGLE_API_TOKEN`. |
| `GOOGLE_API_KEY` | Enables Gemini-powered strategy + code generation. |
| `AUTOKAGGLE_MODEL` | Gemini model name (default: `gemini-3-flash-preview`). |
| `AUTOKAGGLE_SKIP_DOWNLOAD` | Skip downloading competition data. |
| `AUTOKAGGLE_SKIP_CHAT` | Skip the chat-guided strategy step. |
| `AUTOKAGGLE_SKIP_EXECUTION` | Skip local execution of generated scripts. |

> Tip: put these in a `.env` file at the repo root if you want them loaded automatically.

## How it works

1. **Create a run** under `runs/`.
2. **Download data + metadata** via the Kaggle API.
3. **Profile the data** to infer column types and target candidates.
4. **Generate a strategy + pipeline** using Gemini (if configured) or local templates.
5. **Execute the pipeline** in an isolated virtualenv (optional).

## Run artifacts

Each run is stored under `runs/{run_id}`:

```
runs/{run_id}/
  input/
    competition.json
    data_profile.json
    chat_transcript.md
    chat_decisions.json
  code/
    data_loading.py
    preprocess.py
    train.py
    validate.py
    predict.py
    strategy.py
  env/
    requirements.txt
    venv/
  output/
    model.joblib
    metrics.json
    submission.csv
  logs/
    run.log
```

## CLI reference

```bash
python -m autokaggle run <competition_url>
python -m autokaggle status <run_id>
python -m autokaggle logs <run_id>
```

## Testing

```bash
pytest -q
```

## Notes

- The local pipeline uses scikit-learn baselines; LLM-generated scripts can adjust modeling choices and features.
- Kaggle competitions often require accepting rules on the competition page before downloads succeed.
