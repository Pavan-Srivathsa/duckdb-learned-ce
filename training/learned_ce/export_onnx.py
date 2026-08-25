"""Export trained XGBoost model to ONNX."""

from __future__ import annotations

import argparse
from pathlib import Path

import xgboost as xgb
from onnxmltools import convert_xgboost
from onnxmltools.convert.common.data_types import FloatTensorType


def export_model_to_onnx(model: xgb.XGBRegressor, output_path: Path, *, feature_count: int) -> None:
    initial_type = [("features", FloatTensorType([None, feature_count]))]
    onnx_model = convert_xgboost(model.get_booster(), initial_types=initial_type, target_opset=15)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("wb") as handle:
        handle.write(onnx_model.SerializeToString())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export XGBoost booster to ONNX")
    parser.add_argument("--booster", type=Path, default=Path("artifacts/model.xgb.json"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/model.onnx"))
    parser.add_argument("--feature-count", type=int, default=14)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    regressor = xgb.XGBRegressor()
    regressor.load_model(args.booster)
    export_model_to_onnx(regressor, args.output, feature_count=args.feature_count)
    print(f"Exported ONNX model to {args.output}")


if __name__ == "__main__":
    main()
