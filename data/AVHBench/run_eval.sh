#!/usr/bin/env bash
# =============================================================================
# run_eval.sh — Benchmark evaluation launcher for TTsHallucination
#
# Usage:
#   bash Benchmark/run_eval.sh [MODE] [OPTIONS]
#
# Modes:
#   quick         — 50 samples per task, base ASR, no audio saved  (default)
#   standard      — 200 samples, Whisper-small, audio saved
#   full          — all 6,408 items, Whisper-base, parallel
#   ablation      — runs 4 configs to isolate suppression contributions
#
# Examples:
#   bash Benchmark/run_eval.sh quick
#   bash Benchmark/run_eval.sh standard --save_audio
#   bash Benchmark/run_eval.sh full --resume
#   CUDA_VISIBLE_DEVICES=1 bash Benchmark/run_eval.sh quick
# =============================================================================

set -euo pipefail

# ── Paths ───────────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
VITA_CODE="${PROJECT_ROOT}"
MODEL_PATH="${PROJECT_ROOT}/models/Qwen2.5-Omni-7B"

export PYTHONPATH="${VITA_CODE}:${PROJECT_ROOT}/..:${PYTHONPATH:-}"

# ── GPU ─────────────────────────────────────────────────────────────────────
: "${CUDA_VISIBLE_DEVICES:=0}"
export CUDA_VISIBLE_DEVICES

# ── Default settings ─────────────────────────────────────────────────────────
MODE="${1:-quick}"
shift || true                    # remaining args forwarded to evaluate.py

DEVICE="cuda"
N_CANDIDATES=3
THRESHOLD=0.70
ASR_MODEL="base"
GRAD_STEPS=60
GRAD_LR=5e-3

log_header() {
    echo ""
    echo "╔══════════════════════════════════════════════════════════════════╗"
    printf "║  %-64s ║\n" "$1"
    echo "╚══════════════════════════════════════════════════════════════════╝"
    echo ""
}

run_eval() {
    local out_dir="$1"; shift
    log_header "Running: $out_dir"
    python3 "${SCRIPT_DIR}/evaluate.py" \
        --model_path   "${MODEL_PATH}" \
        --device       "${DEVICE}"     \
        --output_dir   "${out_dir}"    \
        --n_candidates "${N_CANDIDATES}" \
        --threshold    "${THRESHOLD}"  \
        --asr_model    "${ASR_MODEL}"  \
        --grad_steps   "${GRAD_STEPS}" \
        --grad_lr      "${GRAD_LR}"    \
        "$@"
}

# ── Mode dispatch ─────────────────────────────────────────────────────────────
case "${MODE}" in

# --------------------------------------------------------------------------
# quick: 50 samples/task for rapid sanity check (~10 min on A100)
# --------------------------------------------------------------------------
quick)
    log_header "QUICK EVAL — 50 samples/task"
    run_eval "Benchmark/results/quick" \
        --n_samples 50 \
        "$@"
    ;;

# --------------------------------------------------------------------------
# standard: 200 samples, Whisper-small, saves audio WAVs (~1 h on A100)
# --------------------------------------------------------------------------
standard)
    log_header "STANDARD EVAL — 200 samples/task"
    run_eval "Benchmark/results/standard" \
        --n_samples  200 \
        --asr_model  small \
        --save_audio \
        "$@"
    ;;

# --------------------------------------------------------------------------
# full: all 6,408 items (~8 h on A100)
# --------------------------------------------------------------------------
full)
    log_header "FULL EVAL — all items"
    run_eval "Benchmark/results/full" \
        --asr_model base \
        "$@"
    ;;

# --------------------------------------------------------------------------
# ablation: compare 4 configurations
#   A — No suppression (baseline only — pipeline disabled)
#   B — Signal processing correction only
#   C — Gradient correction (Path A waveform only)
#   D — Full pipeline (Path B hidden-state + Path A waveform)
# --------------------------------------------------------------------------
ablation)
    log_header "ABLATION — 4 suppression configurations"
    N_ABL=50

    # A: no gradient, no denoising, no prosody (pure baseline reproduced)
    log_header "Ablation A — No Suppression"
    python3 "${SCRIPT_DIR}/evaluate.py" \
        --model_path "${MODEL_PATH}" --device "${DEVICE}" \
        --output_dir "Benchmark/results/ablation_A_nosuppression" \
        --n_samples  ${N_ABL} --n_candidates 1 \
        --no_gradient "$@"

    # B: signal-processing correction only (no gradient)
    log_header "Ablation B — Signal Processing Only"
    python3 "${SCRIPT_DIR}/evaluate.py" \
        --model_path "${MODEL_PATH}" --device "${DEVICE}" \
        --output_dir "Benchmark/results/ablation_B_sigproc" \
        --n_samples  ${N_ABL} --n_candidates "${N_CANDIDATES}" \
        --no_gradient "$@"

    # C: waveform gradient only (Path A, no hidden-state Path B)
    log_header "Ablation C — Waveform Gradient (Path A only)"
    python3 "${SCRIPT_DIR}/evaluate.py" \
        --model_path "${MODEL_PATH}" --device "${DEVICE}" \
        --output_dir "Benchmark/results/ablation_C_waveform_grad" \
        --n_samples  ${N_ABL} --n_candidates "${N_CANDIDATES}" \
        --no_hidden_state_opt "$@"

    # D: full pipeline (Path B + Path A)
    log_header "Ablation D — Full Pipeline (Path B + Path A)"
    python3 "${SCRIPT_DIR}/evaluate.py" \
        --model_path "${MODEL_PATH}" --device "${DEVICE}" \
        --output_dir "Benchmark/results/ablation_D_full" \
        --n_samples  ${N_ABL} --n_candidates "${N_CANDIDATES}" \
        "$@"

    # ── Print ablation summary ─────────────────────────────────────────────
    log_header "ABLATION SUMMARY"
    python3 - <<'PYEOF'
import json, glob, sys
from pathlib import Path

configs = {
    "A (No Suppression)":   "Benchmark/results/ablation_A_nosuppression/report.json",
    "B (Signal Proc.)":     "Benchmark/results/ablation_B_sigproc/report.json",
    "C (Waveform Grad.)":   "Benchmark/results/ablation_C_waveform_grad/report.json",
    "D (Full Pipeline)":    "Benchmark/results/ablation_D_full/report.json",
}

print(f"\n{'Config':<25} {'Final':>8} {'TextAudio':>10} {'Emotion':>8} {'Speaker':>8} {'Noise':>8}")
print("─" * 75)
for name, rpath in configs.items():
    try:
        with open(rpath) as f:
            data = json.load(f)
        ov = data["overall"]["tts"]
        # For ablation: pipeline values = the "treatment" result
        p = ov["pipeline_final"]
        ta = ov["pipeline_text_audio"]
        em = ov["pipeline_emotion"]
        sp = ov["pipeline_speaker"]
        ns = ov["pipeline_noise"]
        print(f"{name:<25} {p:>8.4f} {ta:>10.4f} {em:>8.4f} {sp:>8.4f} {ns:>8.4f}")
    except Exception as e:
        print(f"{name:<25} {'(error)':>8}  {e}")
print()
PYEOF
    ;;

# --------------------------------------------------------------------------
# single-task shortcuts
# --------------------------------------------------------------------------
av_matching)
    log_header "TASK: AV Matching"
    run_eval "Benchmark/results/task_av_matching" \
        --tasks "AV Matching" --n_samples 100 "$@"
    ;;

video_ah)
    log_header "TASK: Video-driven Audio Hallucination"
    run_eval "Benchmark/results/task_video_ah" \
        --tasks "Video-driven Audio Hallucination" --n_samples 100 "$@"
    ;;

audio_ah)
    log_header "TASK: Audio-driven Video Hallucination"
    run_eval "Benchmark/results/task_audio_ah" \
        --tasks "Audio-driven Video Hallucination" --n_samples 100 "$@"
    ;;

captioning)
    log_header "TASK: AV Captioning"
    run_eval "Benchmark/results/task_captioning" \
        --tasks "AV Captioning" --n_samples 50 \
        --asr_model small "$@"
    ;;

*)
    echo "Unknown mode: ${MODE}"
    echo "Available modes: quick | standard | full | ablation | av_matching | video_ah | audio_ah | captioning"
    exit 1
    ;;
esac

log_header "Evaluation complete"
echo "Results saved to Benchmark/results/"
