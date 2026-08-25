# Benchmarking

## Workloads

- TPC-H (SF1, SF10)
- TPC-DS
- IMDB join-heavy workload

## Modes

Run each workload under: native, shadow, learned, hybrid (adaptive optional).

## Metrics

### Cardinality

- q-error: `max(p/a, a/p)`
- Report median, p75, p90, p95, p99, max

### Runtime

- Planning time, inference time, execution time, total latency
- Plan diffs and regression buckets (>10% improved / degraded)

## Reproducibility

Each run records: CPU, RAM, OS, compiler, DuckDB commit, model hash, scale factor, thread count, seed.

Results go to `experiments/results/<experiment-id>/` (never hand-edited).
