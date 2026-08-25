# Architecture

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
src/optimizer/learned_ce/          # C++ runtime module
src/include/duckdb/optimizer/learned_ce/
training/learned_ce/               # Python training pipeline
benchmarks/                        # TPC-H, TPC-DS, IMDB runners
experiments/results/               # Generated benchmark outputs
artifacts/                         # ONNX models and metadata
```

## Fallback cases

The learned model never blocks optimization. Fallback to native when the model is unavailable, ineligible, non-finite, or below confidence threshold.
