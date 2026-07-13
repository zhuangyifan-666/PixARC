#!/usr/bin/env bash
set -euo pipefail

BASELINE_ROOT="$(cd -- "$(dirname -- "$0")/.." && pwd -P)"
export PYTHONPATH="$BASELINE_ROOT${PYTHONPATH:+:$PYTHONPATH}"
exec python -m dicache_style.distribution_metrics "$@"

