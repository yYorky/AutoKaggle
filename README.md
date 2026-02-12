![](https://raw.githubusercontent.com/yYorky/AutoKaggle/refs/heads/main/assets/autokaggle.png)

# AutoKaggle

AutoKaggle is a CLI-first tool that automates Kaggle competition workflows with minimal user input. You provide a competition URL, and AutoKaggle will download data, profile the dataset, ask an LLM for strategy guidance, generate a runnable baseline pipeline, and (optionally) execute it locally to produce a `submission.csv`.

## Why this design?

- **Why a CLI?** Kaggle workflows are naturally scriptable (download → train → submit). A CLI keeps runs reproducible, works well in CI or remote machines, and makes it easy to capture artifacts and logs without a UI dependency.
- **Why let an LLM evaluate data and propose steps?** Kaggle competitions vary widely. The LLM reads the competition rules, evaluation metric, and data profile so it can suggest model family, features, constraints, and hyperparameters that are more tailored than a one-size-fits-all template.
- **Why self-healing with retry steps?** Generated code can fail on edge cases (unexpected schemas, missing columns). AutoKaggle retries code generation with the failure context so the LLM can fix the specific error, improving reliability without manual intervention.
- **Why split run orchestration into helper functions?** The run flow now follows explicit stages (`download_data` → `build_decision` → `generate_code` → `execute_and_retry`) to reduce cognitive load and make each stage easier to test in isolation.
- **Why template prompt constraints as constants?** Centralizing codegen constraints avoids subtle drift when editing prompt policies and makes future prompt updates safer and more consistent.
- **Why bounded tail reads for logs?** Log tailing now reads files incrementally and keeps only the last N lines in memory, which scales better for long-running jobs with large logs.
- **Why always write artifacts into a per-run directory tree?** Each run (including retries) gets its own isolated `runs/{run_id}` workspace with `input/code/env/output/logs`, making failures reproducible and preventing cross-run contamination.
- **Why preserve a local-template fallback when LLM calls are unavailable?** If an API key is missing, AutoKaggle still generates a runnable baseline from local templates so users can progress offline or in restricted environments.
- **Why infer targets from sample submission first?** Many competitions encode target expectations in `sample_submission.csv`; inferring targets from it (with normalization + fallbacks) reduces wrong-target guesses on unfamiliar datasets.
- **Why combine Kaggle API metadata with scraped competition-page text?** API metadata provides structured fields, while page text often includes nuanced rules/metric details not exposed cleanly in the API.
- **Why enforce env-var-over-config precedence?** Environment overrides make CI/CD and ephemeral execution safer by letting operators inject runtime settings/secrets without mutating committed config files.
- **Why validate run metadata against a schema on read/write?** Early validation catches corrupted or partially written run records before downstream commands (`status`, retries) rely on them.
- **Why define LLM integrations behind lightweight protocols?** `ChatModel` and `CodegenModel` protocols allow dependency injection in tests, making chat/codegen logic testable without live network calls.

## Quickstart (clone → run → inspect)

1. **Clone the repo**
   ```bash
   git clone https://github.com/your-org/AutoKaggle.git
   cd AutoKaggle
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set credentials**
   AutoKaggle reads secrets from environment variables or a local `.env` file.

   ```bash
   export KAGGLE_API_TOKEN=your_kaggle_token
   export GOOGLE_API_KEY=your_google_api_key
   export AUTOKAGGLE_MODEL=gemini-3-flash-preview
   ```

   Notes:
   - `GOOGLE_API_KEY` enables LLM-backed chat strategy and LLM code generation.
   - `AUTOKAGGLE_MODEL` only selects the model name used for those LLM calls.
   - If `GOOGLE_API_KEY` is not set, AutoKaggle falls back to local template-based code generation.

5. **Run AutoKaggle**
   ```bash
   python -m autokaggle run https://www.kaggle.com/competitions/{competition}
   ```

6. **Inspect the run**
   ```bash
   python -m autokaggle status {run_id}
   python -m autokaggle logs {run_id}
   ls runs/{run_id}/output
   ```

## Workflow diagram

```mermaid
flowchart TD
    A[Competition URL] --> B[Download data + competition metadata]
    B --> C[Profile dataset + infer schema]
    C --> D{Chat-guided strategy}
    D -->|LLM enabled| E[LLM proposes model + features]
    D -->|LLM disabled| F[Default baseline strategy]
    E --> G[Generate pipeline code]
    F --> G
    G --> H{Execute locally?}
    H -->|Yes| I[Run train/predict scripts]
    I --> J{Success?}
    J -->|Yes| K[submission.csv + metrics]
    J -->|No| L[Capture failure context]
    L --> M[Retry codegen with error context]
    M --> G
    H -->|No| N[Artifacts ready for review]
```

## What happens during a run?

1. **Competition discovery**: metadata, evaluation metric, and rules are fetched via the Kaggle API.
2. **Data profiling**: AutoKaggle scans CSVs in the downloaded dataset and produces a `data_profile.json`.
3. **Chat-guided strategy**: an LLM (default: Gemini) reviews the competition and data profile to propose a baseline strategy and hyperparameters (or you can skip this step).
4. **Pipeline generation**: AutoKaggle generates `data_loading.py`, `preprocess.py`, `train.py`, and `predict.py` plus an `env/requirements.txt` (LLM-generated if a key is available, otherwise templated locally).
5. **Local execution (optional)**: the generated scripts are executed in a per-run virtualenv to produce `model_lightgbm.joblib`, `model_xgboost.joblib`, `model_catboost.joblib`, `model_meta.json`, and `submission.csv`.
6. **Self-healing retries**: if execution fails, AutoKaggle regenerates code with the error context for a configurable number of retries.

## Repository structure (run artifacts)

```
runs/
  {run_id}/
    input/
      competition.json
      data_profile.json
      chat_transcript.md
      chat_decisions.json
    code/
      data_loading.py
      preprocess.py
      train.py
      predict.py
      strategy.py
    env/
      requirements.txt
      venv/
    output/
      model_lightgbm.joblib
      model_xgboost.joblib
      model_catboost.joblib
      model_meta.json
      submission.csv
    logs/
      run.log
```

## Configuration

AutoKaggle reads secrets from environment variables (or a `.env` file) and non-secret runtime
settings from an optional `config.yaml`. Environment variables always take precedence.

### Secrets (`.env`)

- `KAGGLE_API_TOKEN`: Kaggle API token (required to download competition data).
- `GOOGLE_API_KEY`: Google GenAI API key (required for chat + LLM codegen).

### Runtime settings (`config.yaml`)

```yaml
autokaggle_model: gemini-3-flash-preview
skip_download: false
skip_chat: false
skip_execution: false
max_codegen_retries: 3
```

You can still configure these via environment variables instead:

- `AUTOKAGGLE_MODEL`: LLM model name (default is configured in code).
- `AUTOKAGGLE_SKIP_DOWNLOAD=1`: intentionally runs in "create-only" mode (creates a run folder and metadata, then exits without download/profile/chat/codegen/execution).
- `AUTOKAGGLE_SKIP_CHAT=1`: skip the LLM strategy step and use a default baseline.
- `AUTOKAGGLE_SKIP_EXECUTION=1`: generate code without executing it.
- `AUTOKAGGLE_MAX_CODEGEN_RETRIES=3`: retries for self-healing code generation.

## Testing

```bash
pytest -q
```
