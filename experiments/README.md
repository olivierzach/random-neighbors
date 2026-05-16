# Experiments

Run from repo root.

## Setup
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .[dev]
```

## Synthetic blobs
```bash
python -m experiments.run_blobs
```

## Iris demo
```bash
python -m experiments.run_iris
```

These scripts print best feature subset scores and produce small plots (if matplotlib is installed).
