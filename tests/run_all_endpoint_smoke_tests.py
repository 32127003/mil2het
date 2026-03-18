from __future__ import annotations

import subprocess
import sys
from pathlib import Path

SMOKE_TEST_SCRIPTS = [
    "smoke_test_config_endpoints.py",
    "smoke_test_cell_encoder_endpoints.py",
    "smoke_test_multiple_instance_learning_endpoints.py",
    "smoke_test_prior_interface_find_endpoints.py",
    "smoke_test_biomarker_endpoints.py",
    "smoke_test_split_dataset_endpoints.py",
    "smoke_test_preselection_endpoints.py",
    "smoke_test_train_endpoints.py",
]


def main() -> None:
    current_dir = Path(__file__).resolve().parent
    failures: list[str] = []
    for script_name in SMOKE_TEST_SCRIPTS:
        script_path = current_dir / script_name
        if not script_path.is_file():
            raise FileNotFoundError(f"Smoke test script not found: {script_path}")
        print(f"[run] {script_name}", flush=True)
        completed = subprocess.run([sys.executable, str(script_path)], check=False)
        if completed.returncode != 0:
            failures.append(script_name)

    if failures:
        raise SystemExit(f"[fail] smoke tests failed: {', '.join(failures)}")

    print("[ok] all endpoint smoke tests completed")


if __name__ == "__main__":
    main()
