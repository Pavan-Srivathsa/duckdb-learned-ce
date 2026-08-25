#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DUCKDB_DIR="$ROOT/duckdb"
INTEGRATION_DIR="$ROOT/integration"

if [ ! -e "$DUCKDB_DIR/.git" ]; then
  echo "DuckDB submodule missing. Run: git submodule update --init duckdb" >&2
  exit 1
fi

if [ -n "$(git -C "$DUCKDB_DIR" status --porcelain)" ]; then
  echo "DuckDB submodule working tree is dirty. Reset it before applying integration:" >&2
  echo "  git -C duckdb reset --hard && git -C duckdb clean -fd" >&2
  exit 1
fi

echo "Applying learned CE integration overlay..."
rsync -a "$INTEGRATION_DIR/include/" "$DUCKDB_DIR/src/include/"
rsync -a "$INTEGRATION_DIR/src/" "$DUCKDB_DIR/src/"
rsync -a "$INTEGRATION_DIR/tests/" "$DUCKDB_DIR/test/"

echo "Applying learned CE hook patches..."
for patch in "$INTEGRATION_DIR/patches/"*.patch; do
  [ -e "$patch" ] || continue
  git -C "$DUCKDB_DIR" apply --whitespace=nowarn "$patch"
done

echo "Regenerating DuckDB settings..."
python3 "$DUCKDB_DIR/scripts/generate_settings.py"

echo "Integration applied to duckdb/ (local working tree only)."
