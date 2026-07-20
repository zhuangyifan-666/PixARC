from __future__ import annotations

import sys
from pathlib import Path


BASE = Path(__file__).resolve().parents[1]
SCRIPTS = BASE / "scripts"
for path in (BASE, SCRIPTS):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))
