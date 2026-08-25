"""Cardinality estimation metrics."""

from __future__ import annotations

import math
import statistics
from typing import Iterable


def q_error(prediction: float, actual: float) -> float:
    if actual <= 0 and prediction <= 0:
        return 1.0
    if actual <= 0 or prediction <= 0:
        return float("inf")
    return max(prediction / actual, actual / prediction)


def percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = int(round((pct / 100.0) * (len(ordered) - 1)))
    return ordered[max(0, min(idx, len(ordered) - 1))]


def summarize_q_errors(errors: Iterable[float]) -> dict[str, float]:
    error_list = list(errors)
    finite = [e for e in error_list if math.isfinite(e)]
    if not finite:
        return {
            "median_q_error": 0.0,
            "p75_q_error": 0.0,
            "p90_q_error": 0.0,
            "p95_q_error": 0.0,
            "p99_q_error": 0.0,
            "max_q_error": 0.0,
            "infinite_errors": len(error_list),
        }
    return {
        "median_q_error": statistics.median(finite),
        "p75_q_error": percentile(finite, 75),
        "p90_q_error": percentile(finite, 90),
        "p95_q_error": percentile(finite, 95),
        "p99_q_error": percentile(finite, 99),
        "max_q_error": max(finite),
        "infinite_errors": len(error_list) - len(finite),
    }


def predictions_from_log_target(model_output) -> list[float]:
    import numpy as np

    values = np.expm1(np.asarray(model_output, dtype=float))
    return [max(1.0, float(v)) for v in values]
