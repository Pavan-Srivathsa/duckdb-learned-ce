# Learned CE Makefile targets
# Usage: make -f learned-ce.mk <target>

DUCKDB_UPSTREAM_COMMIT ?= 95697fa642c7ccd4514284ef0d7cdd2e82667d48
BUILD_DIR ?= duckdb/build/release
JOBS ?= $(shell sysctl -n hw.ncpu 2>/dev/null || echo 4)

.PHONY: submodule apply reset build test-learned-ce generate-data baseline-analysis docs

submodule:
	git submodule update --init --recursive duckdb

apply:
	./scripts/apply_integration.sh

reset-duckdb:
	./scripts/reset_duckdb.sh

build:
	JOBS=$(JOBS) ./scripts/build.sh

test-learned-ce: build
	./$(BUILD_DIR)/test/unittest "[learned_ce]"

generate-data:
	cd training && PYTHONPATH=. python3 -m learned_ce.collect_cardinalities --scale-factor 1 --max-records 200 --output ../data/training.parquet

baseline-analysis:
	python3 benchmarks/analysis.py --dataset data/training.parquet --output experiments/results/baseline/cardinality_metrics.json

docs:
	@echo "See documentation/learned-ce/ for design and architecture."
