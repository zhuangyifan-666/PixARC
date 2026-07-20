from __future__ import annotations

import sys
from pathlib import Path


BASE = Path(__file__).resolve().parents[1]
UPSTREAM = Path(__file__).resolve().parents[4] / "third-party" / "PixelGen"
for path in (BASE, UPSTREAM):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))
