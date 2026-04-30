from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

TESTS_DIR = Path(__file__).resolve().parent
if str(TESTS_DIR) not in sys.path:
    sys.path.insert(0, str(TESTS_DIR))

collect_ignore = ["smoke_test_helpers.py"]


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--cpu-only",
        action="store_true",
        default=False,
        help="Skip tests marked as requiring actual CUDA execution.",
    )


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers",
        "gpu: requires actual CUDA execution and is skipped when --cpu-only is used",
    )
    if bool(config.getoption("--cpu-only")):
        os.environ["SCBIOMARKER_TEST_CPU_ONLY"] = "1"
    else:
        os.environ.pop("SCBIOMARKER_TEST_CPU_ONLY", None)


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    if not bool(config.getoption("--cpu-only")):
        return

    skip_gpu = pytest.mark.skip(reason="skipped in --cpu-only run")
    for item in items:
        if "gpu" in item.keywords:
            item.add_marker(skip_gpu)
