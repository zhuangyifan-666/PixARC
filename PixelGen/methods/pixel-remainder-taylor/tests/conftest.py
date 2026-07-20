from __future__ import annotations

import sys
from pathlib import Path


BASE = Path(__file__).resolve().parents[1]
SCRIPTS = BASE / "scripts"
UPSTREAM = Path(__file__).resolve().parents[4] / "third-party" / "PixelGen"
SHARED = Path(__file__).resolve().parents[4] / "JiT/methods/pixel-remainder-taylor"
SHARED_SCRIPTS = SHARED / "scripts"
BASELINE = Path(__file__).resolve().parents[3] / "baselines/taylorseer-style"
for path in reversed((BASE, SCRIPTS, UPSTREAM, SHARED, SHARED_SCRIPTS, BASELINE)):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))
