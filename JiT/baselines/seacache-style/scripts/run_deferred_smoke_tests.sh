#!/usr/bin/env bash
set -euo pipefail

usage() {
  echo "Usage: $0 --full-config FILE --candidate-config FILE --manifest FILE --output-root DIR" >&2
}

FULL_CONFIG=""
CANDIDATE_CONFIG=""
MANIFEST=""
OUTPUT_ROOT=""
while (($#)); do
  case "$1" in
    --full-config) [[ $# -ge 2 ]] || { usage; exit 2; }; FULL_CONFIG="$2"; shift 2 ;;
    --candidate-config) [[ $# -ge 2 ]] || { usage; exit 2; }; CANDIDATE_CONFIG="$2"; shift 2 ;;
    --manifest) [[ $# -ge 2 ]] || { usage; exit 2; }; MANIFEST="$2"; shift 2 ;;
    --output-root) [[ $# -ge 2 ]] || { usage; exit 2; }; OUTPUT_ROOT="$2"; shift 2 ;;
    *) usage; exit 2 ;;
  esac
done
[[ -f "$FULL_CONFIG" && -f "$CANDIDATE_CONFIG" && -f "$MANIFEST" && -n "$OUTPUT_ROOT" ]] || { usage; exit 2; }
if [[ "${SEACACHE_GPU_TESTS_ALLOWED:-0}" != "1" ]]; then
  echo "Deferred GPU smoke tests are locked while reference generation is active." >&2
  exit 2
fi
[[ -n "${CUDA_VISIBLE_DEVICES:-}" && "$CUDA_VISIBLE_DEVICES" != *,* ]] || {
  echo "Smoke wrapper requires exactly one explicitly visible allocated GPU." >&2
  exit 2
}
for NAME in full candidate; do
  TARGET="$OUTPUT_ROOT/$NAME"
  if [[ -e "$TARGET" ]] && find "$TARGET" -mindepth 1 -print -quit | grep -q .; then
    echo "Refusing non-empty smoke output: $TARGET" >&2
    exit 3
  fi
done

python "$(dirname "$0")/generate_shard.py" \
  --config "$FULL_CONFIG" --manifest "$MANIFEST" --shard-id 0 --world-size 1 \
  --output-root "$OUTPUT_ROOT/full" --acknowledge-gpu-job
python "$(dirname "$0")/generate_shard.py" \
  --config "$CANDIDATE_CONFIG" --manifest "$MANIFEST" --shard-id 0 --world-size 1 \
  --output-root "$OUTPUT_ROOT/candidate" --acknowledge-gpu-job
