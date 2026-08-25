# Learned Cardinality Estimation

Experimental integration of an XGBoost/ONNX learned cardinality estimator into DuckDB's join-order optimizer.

## Upstream pin

```text
DuckDB upstream: 95697fa642c7ccd4514284ef0d7cdd2e82667d48 (2026-08-25)
Project remote:  https://github.com/Pavan-Srivathsa/duckdb-learned-ce
```

## Settings

| Setting | Default | Description |
|---------|---------|-------------|
| `learned_ce_enabled` | false | Enable experimental learned CE |
| `learned_ce_mode` | native | native, shadow, learned, hybrid, adaptive |
| `learned_ce_model_path` | "" | Path to ONNX model artifact |
| `learned_ce_confidence_threshold` | 0.5 | Hybrid/learned confidence gate |
| `learned_ce_collect_telemetry` | false | Log estimation records |

## Build and test

```bash
make -f learned-ce.mk build
make -f learned-ce.mk test-learned-ce
```

## Documentation

- [DESIGN.md](DESIGN.md) — problem, hypotheses, milestones
- [ARCHITECTURE.md](ARCHITECTURE.md) — C++/Python integration
- [MODEL.md](MODEL.md) — features, target, splits
- [BENCHMARKING.md](BENCHMARKING.md) — metrics and methodology
- [RESULTS.md](RESULTS.md) — experiment results template
