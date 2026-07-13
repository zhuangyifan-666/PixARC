#!/usr/bin/env python3
"""Build an immutable deterministic manifest."""

import sys

from seacache_style.manifest import _main


if __name__ == "__main__":
    sys.argv.insert(1, "build")
    _main()
