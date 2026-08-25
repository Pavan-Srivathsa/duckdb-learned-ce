"""Train XGBoost regressor on log1p(cardinality) target."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import xgboost as xgb

from learned_ce.dataset import feature_matrix, load_dataset, split_frame, target_vector
from learned_ce.export_onnx import export_model_to_onnx
from learned_ce.metrics import predictions_from_log_target, q_error, summarize_q_errors
from learned_ce.schema import FEATURE_NAMES, FEATURE_SCHEMA_VERSION


def train_model(
    dataset_path: Path,
    *,
    output_dir: Path,
    seed: int = 42,
) -> dict[str, object]:
    frame = load_dataset(dataset_path)
    train = split_frame(frame, "train")
    val = split_frame(frame, "val")
    test = split_frame(frame, "test")

    if train.empty:
        raise ValueError("Dataset must contain a non-empty train template split")
    if val.empty:
        val = train

    x_train = feature_matrix(train)
    y_train = target_vector(train)
    x_val = feature_matrix(val)
    y_val = target_vector(val)

    model = xgb.XGBRegressor(
        objective="reg:squarederror",
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=seed,
        n_jobs=-1,
    )
    model.fit(x_train, y_train, eval_set=[(x_val, y_val)], verbose=False)

    output_dir.mkdir(parents=True, exist_ok=True)
    booster_path = output_dir / "model.xgb.json"
    model.get_booster().save_model(booster_path)

    def evaluate_split(name: str, split_frame_data) -> dict[str, object]:
        if split_frame_data.empty:
            return {"records": 0}
        x_split = feature_matrix(split_frame_data)
        preds = predictions_from_log_target(model.predict(x_split))
        actual = split_frame_data["actual_cardinality"].astype(float).tolist()
        native = split_frame_data["native_estimate"].astype(float).tolist()
        learned_errors = [q_error(p, a) for p, a in zip(preds, actual)]
        native_errors = [q_error(n, a) for n, a in zip(native, actual)]
        return {
            "records": len(split_frame_data),
            "learned": summarize_q_errors(learned_errors),
            "native": summarize_q_errors(native_errors),
        }

    metrics = {
        "seed": seed,
        "target": "log1p_cardinality",
        "train": evaluate_split("train", train),
        "val": evaluate_split("val", val),
        "test": evaluate_split("test", test),
    }

    feature_schema = {
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "features": FEATURE_NAMES,
        "feature_count": len(FEATURE_NAMES),
    }

    metadata = {
        "model_version": "v1",
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "target": "log1p_cardinality",
        "features": FEATURE_NAMES,
        "training_benchmarks": sorted(frame["benchmark"].dropna().unique().tolist()),
        "dataset": str(dataset_path),
        "booster_path": str(booster_path),
    }

    metrics_path = output_dir / "metrics.json"
    schema_path = output_dir / "feature_schema.json"
    metadata_path = output_dir / "model_metadata.json"
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    schema_path.write_text(json.dumps(feature_schema, indent=2), encoding="utf-8")
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    onnx_path = output_dir / "model.onnx"
    export_model_to_onnx(model, onnx_path, feature_count=len(FEATURE_NAMES))

    return {
        "metrics_path": str(metrics_path),
        "onnx_path": str(onnx_path),
        "val_median_q_error": metrics["val"]["learned"]["median_q_error"],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train learned CE XGBoost model")
    parser.add_argument("--dataset", type=Path, default=Path("data/training.parquet"))
    parser.add_argument("--output", type=Path, default=Path("artifacts"))
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.dataset.exists():
        print(f"Dataset not found: {args.dataset}", file=sys.stderr)
        raise SystemExit(1)

    summary = train_model(args.dataset, output_dir=args.output, seed=args.seed)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
