#!/usr/bin/env python3
"""Backward-compatible 1K wrapper around the generic run evaluator."""

from __future__ import annotations

import sys

from evaluate_run import (
    PAIRING_PROTOCOL_FIELDS,
    main,
    validate_pairing_protocol,
)


if __name__ == "__main__":
    if "--expected-count" not in sys.argv:
        sys.argv.extend(("--expected-count", "1000"))
    main()


__all__ = ["PAIRING_PROTOCOL_FIELDS", "main", "validate_pairing_protocol"]
