#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TEST_ENV="${TEST_ENV:-test}"

echo "[check] installed scbiomarker command in conda env: ${TEST_ENV}"
conda run -n "${TEST_ENV}" bash -lc 'command -v scbiomarker >/dev/null'
echo "[ok] scbiomarker command available in env ${TEST_ENV}"

echo "[check] scbiomarker --help"
conda run -n "${TEST_ENV}" scbiomarker --help >/dev/null
echo "[ok] scbiomarker --help"

echo "[run] TODO step-1 CLI/import smoke"
conda run -n "${TEST_ENV}" python "${SCRIPT_DIR}/smoke_test_todo_step1_examples.py"
