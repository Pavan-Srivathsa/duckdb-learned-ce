#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DUCKDB_DIR="$ROOT/duckdb"

if [ ! -e "$DUCKDB_DIR/.git" ]; then
  echo "DuckDB submodule missing." >&2
  exit 1
fi

git -C "$DUCKDB_DIR" reset --hard
git -C "$DUCKDB_DIR" clean -fd
echo "DuckDB submodule reset to pristine submodule commit."
