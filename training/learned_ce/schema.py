"""Shared schemas and constants for the learned CE training pipeline."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

FEATURE_NAMES: list[str] = [
    "log_native_estimate",
    "log_left_cardinality",
    "log_right_cardinality",
    "log_left_ndv",
    "log_right_ndv",
    "ndv_ratio",
    "relation_count",
    "join_edge_count",
    "equality_predicate_count",
    "range_predicate_count",
    "filter_count",
    "join_graph_density",
    "has_left_stats",
    "has_right_stats",
]

FEATURE_SCHEMA_VERSION = 1


@dataclass
class TrainingRecord:
    query_id: str
    template: str
    benchmark: str
    relations: list[str]
    features: dict[str, float]
    native_estimate: float
    actual_cardinality: int
    sql: str = ""
    split: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
