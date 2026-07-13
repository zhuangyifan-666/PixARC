from __future__ import annotations

import os
import sys
from pathlib import Path

if os.environ.get("CUDA_VISIBLE_DEVICES") not in {"", "-1"}:
    raise RuntimeError("DiCache unit tests require CUDA_VISIBLE_DEVICES='' or '-1'")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
