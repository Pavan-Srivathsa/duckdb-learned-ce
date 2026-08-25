"""Compute q-error metrics for native cardinality estimates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics

import pandas as pd


def q_error(prediction: float, actual: float) -> float:
    if actual <= 0 and prediction <= 0:
        return 1.0
    if actual <= 0:
        return float("inf")
    if prediction <= 0:
        return float("inf")
    return max(prediction / actual, actual / prediction)


def percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = int(round((pct / 100.0) * (len(ordered) - 1)))
    return ordered[max(0, min(idx, len(ordered) - 1))]


def analyze_dataset(path: Path) -> dict[str, object]:
    frame = pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_json(path, lines=True)
    errors = [
        q_error(float(row["native_estimate"]), float(row["actual_cardinality"]))
        for _, row in frame.iterrows()
    ]
    finite_errors = [e for e in errors if e != float("inf")]

    return {
        "records": len(frame),
        "median_q_error": statistics.median(finite_errors) if finite_errors else 0.0,
        "p75_q_error": percentile(finite_errors, 75),
        "p90_q_error": percentile(finite_errors, 90),
        "p95_q_error": percentile(finite_errors, 95),
        "p99_q_error": percentile(finite_errors, 99),
        "max_q_error": max(finite_errors) if finite_errors else 0.0,
        "infinite_errors": len(errors) - len(finite_errors),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze native estimator q-error on a training dataset")
    parser.add_argument("--dataset", type=Path, default=Path("data/training.parquet"))
    parser.add_argument("--output", type=Path, default=Path("experiments/results/baseline/cardinality_metrics.json"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metrics = analyze_dataset(args.dataset)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
