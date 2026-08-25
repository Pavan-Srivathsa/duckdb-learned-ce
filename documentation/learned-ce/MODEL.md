# Model

## Target

Train on `y = log1p(actual_cardinality)`, predict `expm1(model_output)`, clamp to `[1, max_bound]`.

## Feature schema v1

See [artifacts/model_metadata.json](../artifacts/model_metadata.json).

## Training splits

- Split by **query template**, not individual rows
- Cross-scale tests: e.g. TPC-H SF1 train → SF10 test
- Cross-workload tests: TPC-H/TPC-DS train → IMDB test

## Initial model

XGBoost regressor exported to ONNX. No neural architecture in v1.
