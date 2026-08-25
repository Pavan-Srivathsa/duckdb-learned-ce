"""Collect exact cardinalities and native estimates for generated workloads."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import re
import sys

import duckdb

from learned_ce.generate_workload import iter_tpch_parameter_grid
from learned_ce.schema import FEATURE_NAMES, TrainingRecord
from learned_ce.splits import assign_template_splits, stable_query_id


def ensure_tpch(con: duckdb.DuckDBPyConnection, scale_factor: float) -> None:
    con.execute("INSTALL tpch")
    con.execute("LOAD tpch")
    con.execute(f"CALL dbgen(sf={scale_factor})")


def extract_estimated_cardinality(explain_text: str) -> float:
    """Best-effort parse of estimated rows from DuckDB EXPLAIN physical plan."""
    tilde_rows = re.findall(r"~\s*([0-9,]+)\s*rows", explain_text, flags=re.IGNORECASE)
    if tilde_rows:
        values = [float(value.replace(",", "")) for value in tilde_rows]
        return max(values)

    matches = re.findall(r"Cardinality:\s*([0-9,]+)", explain_text)
    if not matches:
        matches = re.findall(r"Estimated Cardinality:\s*([0-9,]+)", explain_text)
    if not matches:
        return 0.0
    values = [float(m.replace(",", "")) for m in matches]
    return max(values)


def build_features(
    *,
    native_estimate: float,
    actual_cardinality: int,
    relation_count: int,
    join_edge_count: int,
    filter_count: int,
) -> dict[str, float]:
    actual = max(actual_cardinality, 0)
    native = max(native_estimate, 0)
    return {
        "log_native_estimate": math.log1p(native),
        "log_left_cardinality": math.log1p(actual),
        "log_right_cardinality": math.log1p(actual),
        "log_left_ndv": 0.0,
        "log_right_ndv": 0.0,
        "ndv_ratio": 0.0,
        "relation_count": float(relation_count),
        "join_edge_count": float(join_edge_count),
        "equality_predicate_count": float(max(join_edge_count, 0)),
        "range_predicate_count": float(filter_count),
        "filter_count": float(filter_count),
        "join_graph_density": (
            join_edge_count / (relation_count * (relation_count - 1))
            if relation_count > 1
            else 0.0
        ),
        "has_left_stats": 1.0 if actual > 0 else 0.0,
        "has_right_stats": 1.0 if actual > 0 else 0.0,
    }


def collect_records(
    *,
    scale_factor: float,
    max_records: int,
    max_variants_per_template: int,
) -> list[TrainingRecord]:
    con = duckdb.connect(database=":memory:")
    ensure_tpch(con, scale_factor)

    records: list[TrainingRecord] = []
    templates = []
    workloads = list(iter_tpch_parameter_grid(max_variants_per_template=max_variants_per_template))
    templates = [w.template for w in workloads]
    split_map = assign_template_splits(templates)

    for idx, workload in enumerate(workloads):
        if len(records) >= max_records:
            break

        actual = int(con.execute(workload.sql).fetchone()[0])
        explain_text = con.execute(f"EXPLAIN {workload.sql}").fetchdf().to_string()
        native_estimate = extract_estimated_cardinality(explain_text)

        relation_count = len(workload.relations)
        join_edge_count = max(relation_count - 1, 0)
        filter_count = workload.sql.upper().count(" WHERE ")
        features = build_features(
            native_estimate=native_estimate,
            actual_cardinality=actual,
            relation_count=relation_count,
            join_edge_count=join_edge_count,
            filter_count=filter_count,
        )

        query_id = stable_query_id(workload.template, workload.params, idx)
        records.append(
            TrainingRecord(
                query_id=query_id,
                template=workload.template,
                benchmark=workload.benchmark,
                relations=list(workload.relations),
                features=features,
                native_estimate=native_estimate,
                actual_cardinality=actual,
                sql=workload.sql,
                split=split_map[workload.template],
                metadata={"scale_factor": scale_factor, "params": workload.params},
            )
        )
    return records


def write_records(records: list[TrainingRecord], output: Path, fmt: str) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    rows = [record.to_dict() for record in records]

    if fmt == "jsonl":
        with output.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row))
                handle.write("\n")
        return

    if fmt == "parquet":
        import pandas as pd

        frame = pd.json_normalize(rows)
        frame.to_parquet(output, index=False)
        return

    raise ValueError(f"Unsupported output format: {fmt}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect cardinality ground truth for learned CE training")
    parser.add_argument("--scale-factor", type=float, default=1.0)
    parser.add_argument("--max-records", type=int, default=1000)
    parser.add_argument("--max-variants-per-template", type=int, default=250)
    parser.add_argument("--output", type=Path, default=Path("data/training.parquet"))
    parser.add_argument("--format", choices=["parquet", "jsonl"], default="parquet")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    records = collect_records(
        scale_factor=args.scale_factor,
        max_records=args.max_records,
        max_variants_per_template=args.max_variants_per_template,
    )
    if not records:
        print("No records collected", file=sys.stderr)
        raise SystemExit(1)

    write_records(records, args.output, args.format)
    split_counts: dict[str, int] = {}
    for record in records:
        split_counts[record.split] = split_counts.get(record.split, 0) + 1

    manifest = {
        "records": len(records),
        "feature_names": FEATURE_NAMES,
        "split_counts": split_counts,
        "scale_factor": args.scale_factor,
        "output": str(args.output),
    }
    manifest_path = args.output.with_name(f"{args.output.stem}.manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
