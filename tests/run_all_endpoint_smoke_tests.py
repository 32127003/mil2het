from __future__ import annotations

import subprocess
import sys
from pathlib import Path

def main(argv: list[str] | None = None) -> None:
    current_dir = Path(__file__).resolve().parent
    pytest_args = list(argv or [])
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            str(current_dir),
            *pytest_args,
        ],
        check=False,
    )
    if completed.returncode != 0:
        raise SystemExit(completed.returncode)


if __name__ == "__main__":
    main(sys.argv[1:])
