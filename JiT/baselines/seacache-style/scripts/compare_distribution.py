#!/usr/bin/env python3
"""Compute SeaCache-minus-Full deltas from two distribution result JSONs."""

import argparse
import json

from seacache_style.distribution_metrics import distribution_deltas
from seacache_style.metadata import atomic_write_json, load_json


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full-json", required=True)
    parser.add_argument("--seacache-json", required=True)
    parser.add_argument("--output-json", required=True)
    args = parser.parse_args()
    result = distribution_deltas(load_json(args.full_json), load_json(args.seacache_json))
    atomic_write_json(args.output_json, result)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
