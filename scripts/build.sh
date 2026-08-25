#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DUCKDB_DIR="$ROOT/duckdb"
BUILD_DIR="$DUCKDB_DIR/build/release"
JOBS="${JOBS:-$(sysctl -n hw.ncpu 2>/dev/null || echo 4)}"
RUN_TESTS="${RUN_TESTS:-1}"

"$ROOT/scripts/apply_integration.sh"

if ! command -v cmake >/dev/null 2>&1; then
  echo "cmake is required to build DuckDB. Install cmake and retry." >&2
  exit 1
fi

echo "Configuring DuckDB..."
cmake -DCMAKE_BUILD_TYPE=Release -B "$BUILD_DIR" -S "$DUCKDB_DIR"

echo "Building DuckDB..."
cmake --build "$BUILD_DIR" --parallel "$JOBS"

if [ "$RUN_TESTS" = "1" ] && [ -x "$BUILD_DIR/test/unittest" ]; then
  echo "Running learned CE unit tests..."
  "$BUILD_DIR/test/unittest" "[learned_ce]"
fi

echo "Build complete: $BUILD_DIR"
