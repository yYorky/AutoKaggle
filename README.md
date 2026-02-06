# AutoKaggle

AutoKaggle is a proposed end-to-end app for automating Kaggle competition workflows with minimal user input. The user provides a Kaggle competition URL, and the app orchestrates:

1. **Competition discovery** – fetch metadata, rules, dataset structure, and evaluation details.
2. **Project scaffolding** – generate a new workspace with reproducible environment, data cache, and run logs.
3. **LLM-assisted pipeline generation** – create Python scripts for data loading, feature engineering, modeling, and submission generation.
4. **Local execution** – run scripts in an isolated virtual environment with required dependencies.
5. **Submission output** – generate a `submission.csv` ready for Kaggle upload.

## Product goals

- **Minimal user input**: a single competition URL and optional constraints (time budget, GPU availability, model family preference).
- **CLI-first**: ship a command-line interface before any web UI.
- **Open architecture**: support most tabular competitions and allow extensibility for vision/NLP.
- **Reproducibility**: generate deterministic project folders with run metadata.
- **Safety**: constrain tool execution, manage credentials securely, and avoid leaking secrets to the LLM.

## High-level architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                          AutoKaggle App                           │
├──────────────────────────────────────────────────────────────────┤
│  CLI Layer                                                        │
│  - Competition URL intake                                         │
│  - Run status + artifacts                                         │
│  - Chat session controls                                          │
├──────────────────────────────────────────────────────────────────┤
│  Orchestrator                                                     │
│  - Job queue + run lifecycle                                      │
│  - Tool/agent routing                                             │
│  - State store (runs, prompts, outputs)                           │
├──────────────────────────────────────────────────────────────────┤
│  Skills / Tools                                                   │
│  - Kaggle metadata fetcher                                        │
│  - Data profiler                                                  │
│  - Chat session manager                                           │
│  - Pipeline generator (LLM)                                       │
│  - Executor + environment manager                                 │
│  - Evaluator (local CV / validation)                              │
├──────────────────────────────────────────────────────────────────┤
│  Storage                                                         │
│  - Run artifacts (code, logs, submissions)                        │
│  - Dataset cache                                                  │
│  - Secrets vault                                                  │
└──────────────────────────────────────────────────────────────────┘
```

## Workflow overview

1. **User submits competition URL**.
2. **Kaggle metadata fetcher** retrieves competition details (deadline, evaluation metric, dataset files, sample submission).
3. **Data profiler** downloads data (via Kaggle API), inspects schema, sizes, missingness.
4. **Chat session** lets the user discuss strategy with the LLM after it reads competition details.
5. **LLM prompt builder** constructs a prompt with rules + dataset context + chat decisions.
6. **LLM code generator** writes scripts under a run directory.
7. **Executor** installs deps in a virtualenv and runs scripts.
8. **Submission output** is stored with model info and metrics.

## Proposed run directory structure

```
runs/
  {run_id}/
    input/
      competition.json
      data_profile.json
      chat_transcript.md
    code/
      train.py
      predict.py
      features.py
      config.yaml
    env/
      requirements.txt
    output/
      model.pkl
      oof.csv
      submission.csv
    logs/
      run.log
```

## Key components

### 1) Competition metadata fetcher
- Uses Kaggle API to pull competition info and dataset.
- Extracts evaluation metric, submission format, rules.

### 2) Data profiler
- Builds schema summary, target column inference, split hints.
- Produces JSON for the LLM prompt.

### 3) Chat session manager
- Presents competition context to the user and LLM.
- Stores decisions (features, model family, constraints) for prompt building.

### 4) LLM pipeline generator
- Calls Gemini (or other provider) with a structured prompt.
- Generates code for:
  - data loading
  - preprocessing
  - model training
  - validation
  - prediction + submission format

### 5) Executor
- Creates a venv and installs dependencies.
- Runs scripts in sequence with resource limits.
- Logs stdout/stderr and artifacts.

## Extensibility

- **Model templates**: plug in baseline templates for tabular, NLP, CV.
- **Prompt templates**: per competition type (classification, regression, time-series).
- **Custom evaluators**: allow user-defined metrics.

## Security and compliance

- Never send Kaggle API keys to the LLM.
- Mask PII in dataset summaries.
- Cache dataset locally with restricted permissions.

## Target scope

- **Primary focus**: Kaggle community monthly competitions (primarily tabular).
- **Data access**: automated downloads via Kaggle API.
- **Interface**: CLI-only for the first version.

## Usage (planned)

> This section will be finalized once the CLI is implemented.

```
# 1) Authenticate with Kaggle (once)
export KAGGLE_USERNAME=your_name
export KAGGLE_KEY=your_key

# 2) Run AutoKaggle with a competition URL
autokaggle run https://www.kaggle.com/competitions/{competition}

# 3) Review the chat transcript and artifacts
ls runs/{run_id}/output
```

## Running & testing (by phase)

### Requirements

- Python 3.10+
- Install dependencies with `pip install -r requirements.txt`.

### Phase 1 (CLI skeleton + run store)

Run the CLI from the repo root:

```
python -m autokaggle run https://www.kaggle.com/competitions/{competition}
python -m autokaggle status {run_id}
python -m autokaggle logs {run_id}
```

Run tests from the repo root:

```
pytest -q
```

## Testing (planned)

> This section will be finalized once the CLI is implemented.

```
# Basic smoke test for CLI entrypoint
pytest -m smoke

# Lint + type checks
ruff check .
pyright
```

## Next steps

- Build a minimal CLI to process a Kaggle URL.
- Implement Kaggle API download + profiler for monthly competitions.
- Add chat loop to capture LLM + user decisions before code generation.
- Implement a baseline tabular template generator.
- Add a local runner that creates `submission.csv`.

## Discussion prompts

- Which chat output format should we treat as best practice for v1 (markdown transcript, structured JSON, or both)?
- Should we add optional manual dataset upload in a later phase, or keep Kaggle API-only longer?
- Baseline model priority: LightGBM.
