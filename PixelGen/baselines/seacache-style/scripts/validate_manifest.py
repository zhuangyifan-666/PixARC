#!/usr/bin/env python3
"""Validate an immutable deterministic manifest."""

import sys

from seacache_style.manifest import _main


if __name__ == "__main__":
    sys.argv.insert(1, "validate")
    _main()
