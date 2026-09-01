#!/usr/bin/env bash
set -euo pipefail

# Resolve the release root from this script so it can be called from any cwd.
RELEASE_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "${RELEASE_ROOT}"

: "${CUDA_VISIBLE_DEVICES:=0}"
export CUDA_VISIBLE_DEVICES

SMOKE_INPUT="data/samples/owp_smoke.jsonl"
SMOKE_VIDEO="data/AVHBench/videos/02060.mp4"
SMOKE_AUDIO="data/AVHBench/audios/02060.wav"
if [[ ! -f "${SMOKE_VIDEO}" || ! -f "${SMOKE_AUDIO}" ]]; then
  # The source-only release has no media payload.  Prepare one CMM row in the
  # user cache so the smoke test remains a single command.  Full AVHBench
  # preparation is intentionally separate because its official media release
  # is not currently mirrored on Hugging Face.
  SMOKE_INPUT="${OWP_CACHE_DIR:-${HOME}/.cache/omni-woodpecker}/manifests/cmm_smoke.jsonl"
  python tools/prepare_assets.py --dataset cmm --max-rows 1 --manifest "${SMOKE_INPUT}"
fi

exec python src/run_aaai27_videollama2_owp_5pass_efficiency_20260731.py \
  --input "${SMOKE_INPUT}" \
  --output results/owp_smoke.jsonl \
  --repeat 1 \
  --max-rows 1
