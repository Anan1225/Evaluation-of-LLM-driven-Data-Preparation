# Mini LLMFW

Mini LLMFW is a lightweight experiment framework for running LLM-based entity matching and data imputation experiments. This repository contains only source code, prompts, configuration templates, and tests. Datasets, raw model outputs, generated plots, and paper figures are intentionally excluded.

## Contents

- `run.py`: main experiment runner for entity matching sweeps.
- `run2.py`: earlier runner kept for reproducibility.
- `experiment/`: reusable sampling, prompt-building, and evaluation helpers.
- `metrics/`: metric utilities.
- `scripts/`: non-plotting batch and conversion scripts.
- `test/`: unit/integration tests with temporary toy data.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Copy the example config and edit local paths:

```bash
cp config.example.yaml config.yaml
```

Set API keys through environment variables instead of committing them:

```bash
export OPENAI_API_KEY="..."
export GEMINI_API_KEY="..."
```

## Data

No datasets are included. Put local datasets under `datasets/` or update the paths in `config.yaml`.

Expected entity matching files:

- `tableA.csv`
- `tableB.csv`
- `train.csv`
- `test.csv` or `test_label.csv`

## Run

Single entity matching run:

```bash
python run.py --config config.yaml --model gpt4o --mode single
```

Sweep:

```bash
python run.py --config config.yaml --model gpt4o --mode sweep
```

Outputs are written to `outputs/`, which is ignored by git.

## Tests

```bash
python -m unittest discover -s test
```
