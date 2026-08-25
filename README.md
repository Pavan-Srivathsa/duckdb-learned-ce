# duckdb-learned-ce

Experimental learned cardinality estimation for DuckDB join-order optimization.

This repository contains **only project code**. Official DuckDB lives in the [`duckdb/`](duckdb/) submodule and is **never modified in git**. At build time, `scripts/apply_integration.sh` overlays the learned-CE C++ integration into the submodule working tree locally.

## Upstream pin

```text
DuckDB upstream: 95697fa642c7ccd4514284ef0d7cdd2e82667d48
GitHub:          https://github.com/Pavan-Srivathsa/duckdb-learned-ce
```

## Repository layout

```text
duckdb/                 # pristine DuckDB submodule
integration/            # C++ overlay + patches (your changes)
training/               # Python training pipeline
benchmarks/             # benchmark runners and analysis
documentation/learned-ce/
scripts/                # apply, build, reset helpers
```

## Quick start

```bash
git submodule update --init duckdb
make -f learned-ce.mk apply
make -f learned-ce.mk build
```

### Python ML pipeline

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install numpy pandas pyarrow scikit-learn xgboost onnxmltools onnx onnxruntime duckdb pytest
# macOS only: brew install libomp

make -f learned-ce.mk pipeline   # collect data → baseline → train → ONNX parity
```

Individual steps:

```bash
make -f learned-ce.mk generate-data
make -f learned-ce.mk baseline-analysis
make -f learned-ce.mk train
make -f learned-ce.mk validate-onnx
```

Reset the submodule to pristine upstream:

```bash
make -f learned-ce.mk reset-duckdb
```

## Documentation

- [documentation/learned-ce/DESIGN.md](documentation/learned-ce/DESIGN.md)
- [documentation/learned-ce/ARCHITECTURE.md](documentation/learned-ce/ARCHITECTURE.md)
- [documentation/learned-ce/MODEL.md](documentation/learned-ce/MODEL.md)
- [documentation/learned-ce/BENCHMARKING.md](documentation/learned-ce/BENCHMARKING.md)

## Settings (after build)

| Setting | Default | Description |
|---------|---------|-------------|
| `learned_ce_enabled` | false | Enable experimental learned CE |
| `learned_ce_mode` | native | native, shadow, learned, hybrid, adaptive |
| `learned_ce_model_path` | "" | Path to ONNX model |
| `learned_ce_confidence_threshold` | 0.5 | Hybrid/learned confidence gate |
| `learned_ce_collect_telemetry` | false | Log estimation records |
