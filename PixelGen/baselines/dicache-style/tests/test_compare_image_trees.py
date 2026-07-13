from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "compare_image_trees.py"


def _environment() -> dict[str, str]:
    result = dict(os.environ)
    result["PYTHONPATH"] = os.pathsep.join(
        [str(ROOT), result.get("PYTHONPATH", "")]
    )
    return result


def test_compare_image_trees_writes_exact_report(tmp_path):
    reference = tmp_path / "reference"
    candidate = tmp_path / "candidate"
    reference.mkdir()
    candidate.mkdir()
    array = np.full((4, 4, 3), 17, dtype=np.uint8)
    Image.fromarray(array).save(reference / "000000.png")
    Image.fromarray(array).save(candidate / "000000.png")
    report = tmp_path / "report.json"
    completed = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--reference-dir",
            str(reference),
            "--candidate-dir",
            str(candidate),
            "--output-json",
            str(report),
            "--require-exact",
        ],
        capture_output=True,
        text=True,
        env=_environment(),
    )
    assert completed.returncode == 0, completed.stderr
    value = json.loads(report.read_text(encoding="utf-8"))
    assert value["exact"] is True
    assert value["sample_count"] == 1
    assert value["max_absolute_uint8_error"] == 0


def test_compare_image_trees_fails_but_persists_difference_report(tmp_path):
    reference = tmp_path / "reference"
    candidate = tmp_path / "candidate"
    reference.mkdir()
    candidate.mkdir()
    Image.fromarray(np.zeros((2, 2, 3), dtype=np.uint8)).save(
        reference / "000007.png"
    )
    changed = np.zeros((2, 2, 3), dtype=np.uint8)
    changed[0, 0, 0] = 2
    Image.fromarray(changed).save(candidate / "000007.png")
    report = tmp_path / "report.json"
    completed = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--reference-dir",
            str(reference),
            "--candidate-dir",
            str(candidate),
            "--output-json",
            str(report),
            "--require-exact",
        ],
        capture_output=True,
        text=True,
        env=_environment(),
    )
    assert completed.returncode != 0
    value = json.loads(report.read_text(encoding="utf-8"))
    assert value["exact"] is False
    assert value["differing_image_count"] == 1
    assert value["max_absolute_uint8_error"] == 2
