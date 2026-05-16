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

## URF landmarks (scalable proximity)
```bash
python -m experiments.run_urf_landmarks
```

These scripts print best feature subset scores and produce small plots (if matplotlib is installed).
