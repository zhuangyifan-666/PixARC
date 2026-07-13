#!/usr/bin/env python3
"""Compute SpeCa-minus-Full deltas from two distribution reports."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


BASELINE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASELINE_ROOT))

from speca_style.distribution_metrics import distribution_deltas  # noqa: E402
from speca_style.metadata import atomic_write_json, load_json  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full-json", required=True)
    parser.add_argument("--speca-json", required=True)
    parser.add_argument("--output-json", required=True)
    arguments = parser.parse_args()
    result = distribution_deltas(
        load_json(arguments.full_json), load_json(arguments.speca_json)
    )
    atomic_write_json(arguments.output_json, result)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
