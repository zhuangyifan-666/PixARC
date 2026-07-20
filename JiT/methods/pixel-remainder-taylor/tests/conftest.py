from __future__ import annotations

import sys
from pathlib import Path


BASE = Path(__file__).resolve().parents[1]
SCRIPTS = BASE / "scripts"
BASELINE = Path(__file__).resolve().parents[3] / "baselines" / "taylorseer-style"
for path in (BASE, SCRIPTS, BASELINE):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))
