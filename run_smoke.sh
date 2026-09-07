#!/usr/bin/env bash
set -euo pipefail

# Resolve the release root from this script so it can be called from any cwd.
RELEASE_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "${RELEASE_ROOT}"

: "${CUDA_VISIBLE_DEVICES:=0}"
export CUDA_VISIBLE_DEVICES

exec python src/owp_infer.py \
  --input sample_data/owp_input.jsonl \
  --output results/owp_smoke.jsonl \
  --model videollama2 \
  --max-rows 1
