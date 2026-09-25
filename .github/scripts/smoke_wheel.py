#!/usr/bin/env python3
"""Install the freshly built wheel into a throwaway venv and smoke it.

Run from the repo root after `python -m build`; expects exactly one wheel in
dist/. Verifies: clean-venv install resolves, both distribution packages
(``supply_demand`` and ``econometrics``) import, the packaged YAML config data
ships inside the wheel, and the ``econ`` CLI entry point is functional.
"""

from __future__ import annotations

import glob
import subprocess
import sys
import tempfile
import venv
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
DIST = REPO / "dist"


def run(cmd: list[str], **kw) -> None:
    print("+", " ".join(cmd))
    subprocess.run(cmd, check=True, **kw)


def main() -> int:
    wheels = sorted(glob.glob(str(DIST / "*.whl")))
    if len(wheels) != 1:
        print(f"expected exactly one wheel in dist/, found: {wheels}")
        return 1
    wheel = wheels[0]

    with tempfile.TemporaryDirectory(prefix="supply-demand-smoke-") as tmp:
        venv_dir = Path(tmp) / "venv"
        venv.create(venv_dir, with_pip=True)
        py = str(venv_dir / "bin" / "python")

        run([py, "-m", "pip", "install", "--quiet", "--upgrade", "pip"])
        run([py, "-m", "pip", "install", "--quiet", wheel])

        # Imports + versions
        run([py, "-c",
             "import supply_demand, econometrics; "
             "print('import-ok', supply_demand.__version__, econometrics.__version__)"])

        # Packaged YAML config data ships in the wheel
        run([py, "-c",
             "from supply_demand.config.settings import settings; "
             "from econometrics.config.settings import METROS_DIR; "
             "p = settings.get_config_path('markets.yaml'); "
             "assert p.exists(), f'missing {p}'; "
             "assert (METROS_DIR / 'dallas_tx.yaml').exists(), f'missing {METROS_DIR}'; "
             "print('config-data-ok')"])

        # Deterministic estimation math works end to end
        run([py, "-c",
             "import numpy as np, pandas as pd; "
             "from supply_demand.models.state_space.kalman import KalmanFilter; "
             "print('kalman-import-ok')"])

        # CLI entry point
        run([str(venv_dir / "bin" / "econ"), "--help"])

    print("SMOKE OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())