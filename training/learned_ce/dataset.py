"""Load and prepare training datasets."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from learned_ce.schema import FEATURE_NAMES


def load_dataset(path: Path) -> pd.DataFrame:
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    if path.suffix == ".jsonl":
        return pd.read_json(path, lines=True)
    raise ValueError(f"Unsupported dataset format: {path}")


def feature_matrix(frame: pd.DataFrame) -> np.ndarray:
    columns = []
    for name in FEATURE_NAMES:
        flat = f"features.{name}"
        if flat in frame.columns:
            columns.append(frame[flat].astype(float).to_numpy())
        elif name in frame.columns:
            columns.append(frame[name].astype(float).to_numpy())
        else:
            raise KeyError(f"Missing feature column: {flat}")
    return np.column_stack(columns)


def target_vector(frame: pd.DataFrame) -> np.ndarray:
    actual = frame["actual_cardinality"].astype(float).to_numpy()
    return np.log1p(actual)


def split_frame(frame: pd.DataFrame, split_name: str) -> pd.DataFrame:
    if "split" not in frame.columns:
        raise KeyError("Dataset missing template split column 'split'")
    return frame[frame["split"] == split_name].reset_index(drop=True)
