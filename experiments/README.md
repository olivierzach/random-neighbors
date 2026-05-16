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

## Benchmarks (artifacts + plots)
```bash
python -m experiments.run_benchmarks
# outputs under artifacts/benchmarks/<run_id>/
```

These scripts print best feature subset scores and produce small plots (if matplotlib is installed).
