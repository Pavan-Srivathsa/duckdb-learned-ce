# Architecture

## Repository model

This project is **not** a DuckDB fork in git. Official DuckDB lives in [`duckdb/`](../duckdb/) as a **pinned submodule**. All learned-CE changes live in [`integration/`](../integration/).

At build time:

```text
integration/include  -> duckdb/src/include   (overlay)
integration/src      -> duckdb/src           (overlay)
integration/tests    -> duckdb/test          (overlay)
integration/patches  -> duckdb/              (git apply)
generate_settings.py -> regenerates settings codegen
```

Reset with `make -f learned-ce.mk reset-duckdb` to return the submodule to pristine upstream.

## Runtime path

```text
JoinOrderOptimizer
  -> CardinalityEstimator::EstimateCardinalityWithSet
       -> native estimate (DuckDB)
       -> LearnedCEBridge::Apply
            -> CEFeatureExtractor
            -> LearnedCardinalityEstimator
            -> EstimatorGate (mode + eligibility)
            -> ONNXCardinalityModel (stub until Milestone 5)
```

## Modes

- **native** — DuckDB estimate only (control)
- **shadow** — compute learned estimate, optimizer uses native
- **learned** — use model when eligible
- **hybrid** — learned with rule-based fallback
- **adaptive** — Thompson Sampling selector (Milestone 10)

## Code layout

```text
integration/include/duckdb/optimizer/learned_ce/
integration/src/optimizer/learned_ce/
integration/patches/0001-learned-ce-hook.patch
training/learned_ce/
benchmarks/
experiments/results/
artifacts/
```

## Fallback cases

The learned model never blocks optimization. Fallback to native when the model is unavailable, ineligible, non-finite, or below confidence threshold.
