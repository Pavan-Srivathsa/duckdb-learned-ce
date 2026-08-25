"""Validate Python vs ONNX Runtime prediction parity."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import onnxruntime as ort
import xgboost as xgb

from learned_ce.dataset import feature_matrix, load_dataset, split_frame


def validate_parity(
    *,
    booster_path: Path,
    onnx_path: Path,
    dataset_path: Path,
    tolerance: float = 1e-4,
    split: str = "val",
) -> dict[str, object]:
    frame = split_frame(load_dataset(dataset_path), split)
    if frame.empty:
        raise ValueError(f"No rows found for split={split}")

    x_val = feature_matrix(frame).astype(np.float32)
    regressor = xgb.XGBRegressor()
    regressor.load_model(booster_path)
    python_preds = regressor.predict(x_val).astype(np.float64)

    session = ort.InferenceSession(onnx_path.read_bytes(), providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    onnx_preds = session.run(None, {input_name: x_val})[0].reshape(-1).astype(np.float64)

    diffs = np.abs(python_preds - onnx_preds)
    max_diff = float(np.max(diffs))
    mean_diff = float(np.mean(diffs))
    passed = max_diff <= tolerance

    return {
        "rows": len(frame),
        "max_abs_diff": max_diff,
        "mean_abs_diff": mean_diff,
        "tolerance": tolerance,
        "passed": passed,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate ONNX parity against XGBoost")
    parser.add_argument("--booster", type=Path, default=Path("artifacts/model.xgb.json"))
    parser.add_argument("--onnx", type=Path, default=Path("artifacts/model.onnx"))
    parser.add_argument("--dataset", type=Path, default=Path("data/training.parquet"))
    parser.add_argument("--split", default="val")
    parser.add_argument("--tolerance", type=float, default=1e-4)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    for path in (args.booster, args.onnx, args.dataset):
        if not path.exists():
            print(f"Missing required file: {path}", file=sys.stderr)
            raise SystemExit(1)

    result = validate_parity(
        booster_path=args.booster,
        onnx_path=args.onnx,
        dataset_path=args.dataset,
        tolerance=args.tolerance,
        split=args.split,
    )
    print(json.dumps(result, indent=2))
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
