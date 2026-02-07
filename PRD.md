# Product Requirements Document (PRD)

## 1) Overview

AutoKaggle is a CLI-first application that automates Kaggle competition workflows. Users provide a competition URL; AutoKaggle retrieves competition metadata and datasets via the Kaggle API, profiles the data, runs a guided chat with an LLM to confirm strategy, generates a baseline ML pipeline, executes it locally in an isolated environment, and produces a submission file.

## 2) Goals

- **CLI-first**: ship a command-line tool before any web UI.
- **Minimal input**: a competition URL plus optional constraints.
- **Automated Kaggle API download**: no manual dataset handling in v1.
- **Scoped focus**: Kaggle community monthly competitions, primarily tabular.
- **Chat-guided decisions**: user + LLM agree on a baseline approach before code generation.
- **Reproducible runs**: deterministic folders with logs, configs, and artifacts.

## 3) Non-goals (v1)

- Web UI.
- Full support for CV/NLP competitions.
- Automated hyperparameter tuning at scale.
- Direct Kaggle submission automation (manual upload is fine).

## 4) Users & Use cases

- **Primary user**: Kaggle participants who want a fast baseline for tabular competitions.
- **Use case**: “Given a competition URL, generate a working baseline pipeline and a valid `submission.csv`.”

## 5) Functional requirements

### 5.1 CLI interface
- `autokaggle run <competition_url>` starts a run.
- `autokaggle status <run_id>` shows progress and key artifacts.
- `autokaggle logs <run_id>` tails run logs.

### 5.2 Kaggle API integration
- Support authentication via `KAGGLE_API_TOKEN`.
- Download competition data and sample submission.
- Fetch competition rules/metric details where possible.

### 5.3 Data profiling
- Detect schema, missingness, data types.
- Infer potential target column from sample submission.
- Persist a machine-readable profile JSON.

### 5.4 Chat-guided strategy step
- Provide the LLM with competition context + data profile.
- Include the competition URL page text so the LLM can review rules and evaluation details.
- Run a guided prompt to propose baseline approach and confirm the evaluation metric.
- Allow user to accept/edit key choices (model family, features, constraints).
- Persist transcript and decisions.

### 5.5 Code generation
- Generate baseline scripts for:
  - data loading
  - preprocessing
  - model training
  - validation
  - prediction and submission
- Write scripts into a run-scoped folder.

### 5.6 Local execution
- Create a virtual environment for each run.
- Install required dependencies.
- Execute the pipeline with logging (train/validate/predict).
- Capture metrics (`metrics.json`), model artifacts, and errors in the run logs.

### 5.7 Output artifacts
- `submission.csv` with the correct schema.
- `model.joblib` and `metrics.json` stored in `output/`.
- Logs, config, and run metadata stored in the run directory.

## 6) Non-functional requirements

- **Security**: never send Kaggle API keys to the LLM.
- **Reproducibility**: all runs create deterministic outputs + metadata.
- **Observability**: clear logs and step-by-step run status.
- **Extensibility**: modular components for future CV/NLP support.

## 7) Component breakdown

1. **CLI App**
   - Argument parsing, run lifecycle, command routing.
2. **Kaggle Client**
   - Authentication, download, metadata fetch.
3. **Data Profiler**
   - Dataset inspection + JSON profile.
4. **Chat Manager**
   - Prompt building, transcript storage, decisions extraction.
5. **Pipeline Generator**
   - LLM prompt templates + code scaffold output.
6. **Executor**
   - venv setup, dependency install, script execution.
7. **Run Store**
   - Folder layout, configs, logs, outputs.

## 8) Phased build plan (with tests)

> Each phase should be completed and tested before proceeding.

### Phase 1: CLI skeleton + run store
**Features**
- `autokaggle run` creates a run folder + metadata.
- Standard run directory structure.

**Tests**
- CLI smoke test creates folder.
- Run metadata file validates schema.

### Phase 2: Kaggle API integration
**Features**
- Auth via env vars.
- Download competition dataset and sample submission.

**Tests**
- Integration test (mocked or real credentials) downloads files.
- Dataset files exist and are non-empty.

### Phase 3: Data profiling
**Features**
- Basic schema + missingness summary.
- Target inference from sample submission.

**Tests**
- Profile JSON includes expected keys.
- Handles mixed numeric/categorical data.

### Phase 4: Chat-guided strategy
**Features**
- Prompt build from metadata + profile.
- Capture transcript + structured decisions.

**Tests**
- Transcript stored in run folder.
- Decisions JSON contains model family + features list.

### Phase 5: Code generation
**Features**
- Generate baseline scripts with configurable model family.
- Write requirements file.

**Tests**
- Scripts are created and importable.
- Requirements file lists dependencies.

### Phase 6: Local execution
**Features**
- Create venv and install dependencies.
- Run training and prediction scripts.

**Tests**
- `submission.csv` is generated.
- Logs include training metrics.

### Phase 7: Polish + docs
**Features**
- Improve errors, add CLI help.
- Expand README usage/testing sections.

**Tests**
- CLI help outputs expected commands.
- End-to-end run on a small competition.

## 9) Open questions

- Baseline model priority: LightGBM.
- What is the best-practice format for chat outputs (markdown transcript, structured JSON, or both)?
- Should we allow optional manual dataset upload in later phases, or keep Kaggle API-only longer?
