#!/usr/bin/env bash
set -euo pipefail

if [[ "${SEACACHE_GPU_TESTS_ALLOWED:-0}" != "1" ]]; then
  echo "Refusing GPU benchmark. Set SEACACHE_GPU_TESTS_ALLOWED=1 only after GPUs are idle." >&2
  exit 2
fi
python -m seacache_style.latency "$@"

