# Learned CE Makefile targets
# Usage: make -f learned-ce.mk <target>

DUCKDB_UPSTREAM_COMMIT ?= 95697fa642c7ccd4514284ef0d7cdd2e82667d48
BUILD_DIR ?= build/release
JOBS ?= $(shell sysctl -n hw.ncpu 2>/dev/null || echo 4)

.PHONY: build test-learned-ce generate-data train validate-onnx benchmark docs

build:
	cmake -DCMAKE_BUILD_TYPE=Release -B $(BUILD_DIR) -S .
	cmake --build $(BUILD_DIR) --parallel $(JOBS)

test-learned-ce: build
	./$(BUILD_DIR)/test/unittest "[learned_ce]"

generate-data:
	cd training && PYTHONPATH=. python3 -m learned_ce.collect_cardinalities --scale-factor 1 --max-records 200 --output ../data/training.parquet

baseline-analysis:
	python3 benchmarks/analysis.py --dataset data/training.parquet --output experiments/results/baseline/cardinality_metrics.json

train:
	python3 -m learned_ce.train --dataset data/training.parquet --output artifacts/model.onnx

validate-onnx:
	python3 training/learned_ce/validate_onnx.py

benchmark:
	python3 benchmarks/runner.py

docs:
	@echo "See docs/ for design, architecture, model, and benchmarking methodology."
