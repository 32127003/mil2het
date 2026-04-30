#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TEST_ENV="${TEST_ENV:-test}"

echo "[check] installed mil2het command in conda env: ${TEST_ENV}"
conda run -n "${TEST_ENV}" bash -lc 'command -v mil2het >/dev/null'
echo "[ok] mil2het command available in env ${TEST_ENV}"

echo "[check] mil2het --help"
conda run -n "${TEST_ENV}" mil2het --help >/dev/null
echo "[ok] mil2het --help"

echo "[run] TODO step-1 CLI/import smoke"
conda run -n "${TEST_ENV}" python "${SCRIPT_DIR}/smoke_test_todo_step1_examples.py"
