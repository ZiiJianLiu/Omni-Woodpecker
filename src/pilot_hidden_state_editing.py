#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import json
import math
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import torch
from tqdm.auto import tqdm


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
for path in (SCRIPT_DIR, ROOT, ROOT.parent):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from patch_target_branch import (  # noqa: E402
    DEFAULT_MODEL_PATH,
    QwenOmniAdapter,
    capture_branch,
    non_target_branch_for,
    norm_yes_no,
    pred_from_margin,
    prepare_branch_inputs,
    safe_text,
    signed_margin_for_answer,
    target_branch_for,
    video_budget,
    write_json,
    write_jsonl,
    yes_no_token_ids,
)


DEFAULT_AUDIO_MANIFEST = ROOT / "data" / "manifests" / "unary_rebalance_manifest.jsonl"
DEFAULT_AUDIO_FINAL = ROOT / "data" / "manifests" / "audio_final_rows.jsonl"
DEFAULT_VISUAL_REPLAY = ROOT / "data" / "manifests" / "visual_replay_rows.jsonl"
DEFAULT_AVH_QA = ROOT / "data" / "AVHBench" / "QA.json"
DEFAULT_AVH_VIDEO_DIR = ROOT / "data" / "AVHBench" / "videos"
DEFAULT_OUTPUT_DIR = ROOT / "results" / "qwen_owp_hidden_editing"

TASK_AUDIO = "audio_grounded_presence"
TASK_VISUAL = "visual_grounded_presence"
BRANCHES = ("full", "audio_only", "visual_only", "text_only")
DEFAULT_DIRECTION_MODES = (
    "target_minus_full",
    "target_minus_nontarget",
    "full_minus_nontarget",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Unified unary hidden-state editing pilot. It captures full/target/non-target "
            "branch states for audio and visual unary rows, then applies additive hidden "
            "edits to the full branch answer position."
        )
    )
    parser.add_argument("--audio-manifest", type=Path, default=DEFAULT_AUDIO_MANIFEST)
    parser.add_argument("--audio-final", type=Path, default=DEFAULT_AUDIO_FINAL)
    parser.add_argument("--visual-replay", type=Path, default=DEFAULT_VISUAL_REPLAY)
    parser.add_argument("--avh-qa", type=Path, default=DEFAULT_AVH_QA)
    parser.add_argument("--avh-video-dir", type=Path, default=DEFAULT_AVH_VIDEO_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--model-path", type=str, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--seed", type=int, default=20260603)
    parser.add_argument("--max-audio", type=int, default=8)
    parser.add_argument("--max-visual", type=int, default=8)
    parser.add_argument("--repair-fraction", type=float, default=0.5)
    parser.add_argument("--layers", type=int, nargs="+", default=[20, 24, 26])
    parser.add_argument("--betas", type=float, nargs="+", default=[0.1, 0.25, 0.5])
    parser.add_argument(
        "--direction-modes",
        nargs="+",
        choices=DEFAULT_DIRECTION_MODES,
        default=list(DEFAULT_DIRECTION_MODES),
    )
    parser.add_argument(
        "--direction-scale",
        choices=("raw", "unit_full_norm"),
        default="raw",
        help="raw uses branch-vector differences directly; unit_full_norm rescales directions to the full-vector norm.",
    )
    parser.add_argument(
        "--target-capture-prompt",
        choices=("original", "evidence_focus"),
        default="original",
        help="Prompt used only when capturing the target branch state. The full branch is always edited under the original prompt.",
    )
    parser.add_argument(
        "--non-target-capture-prompt",
        choices=("original", "evidence_focus"),
        default="original",
        help="Prompt used only when capturing the non-target branch state.",
    )
    parser.add_argument("--video-fps", type=float, default=2.0)
    parser.add_argument("--video-max-frames", type=int, default=32)
    parser.add_argument("--video-max-pixels", type=int, default=602112)
    parser.add_argument("--video-min-pixels", type=int, default=100352)
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--no-progress", action="store_true")
    return parser.parse_args()


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def stable_unique_floats(values: Sequence[float]) -> List[float]:
    out: List[float] = []
    for value in values:
        number = float(value)
        if not math.isfinite(number):
            raise ValueError(f"beta must be finite, got {value!r}")
        if number not in out:
            out.append(number)
    return out


def normalize_binary_label(value: Any) -> Optional[str]:
    label = norm_yes_no(value)
    if label is not None:
        return label
    text = safe_text(value).lower()
    if text in {"1", "true"}:
        return "Yes"
    if text in {"0", "false"}:
        return "No"
    return None


def with_answer_instruction(question: str) -> str:
    text = safe_text(question)
    if "Answer with only Yes or No." in text:
        return text
    return f"{text}\nAnswer with only Yes or No."


def evidence_focus_prompt(question: str, *, target_modality: str) -> str:
    modality = safe_text(target_modality).lower()
    if modality == "audio":
        evidence_text = "Use only the auditory evidence relevant to the question. Do not rely on visual context or language priors."
    elif modality == "visual":
        evidence_text = "Use only the direct visual evidence relevant to the question. Do not rely on audio context or language priors."
    else:
        evidence_text = "Use only the direct target-modality evidence relevant to the question. Do not rely on non-target context or language priors."
    return f"{safe_text(question)}\n{evidence_text}\nAnswer with only Yes or No."


def capture_prompt_row(row: Dict[str, Any], *, prompt_mode: str) -> Dict[str, Any]:
    if prompt_mode == "original":
        return row
    if prompt_mode != "evidence_focus":
        raise ValueError(f"Unsupported capture prompt mode: {prompt_mode}")
    out = dict(row)
    out["formatted_question"] = evidence_focus_prompt(
        safe_text(row.get("question")),
        target_modality=safe_text(row.get("target_modality")),
    )
    return out


def sample_id_parts(sample_id: str) -> Tuple[int, str]:
    pieces = safe_text(sample_id).split(":")
    if len(pieces) != 3 or pieces[0] != "avhbench":
        raise ValueError(f"Unsupported AVHBench sample_id={sample_id!r}")
    return int(pieces[1]), pieces[2]


def qa_lookup_by_index(qa_rows: Sequence[Dict[str, Any]], sample_id: str) -> Dict[str, Any]:
    row_index, video_id = sample_id_parts(sample_id)
    if row_index < 0 or row_index >= len(qa_rows):
        raise ValueError(f"sample_id index out of range: {sample_id}")
    qa_row = dict(qa_rows[row_index])
    if safe_text(qa_row.get("video_id")) != video_id:
        raise ValueError(
            f"sample_id/video mismatch for {sample_id}: QA video_id={qa_row.get('video_id')!r}"
        )
    return qa_row


def joined_audio_rows(
    *,
    manifest_path: Path,
    final_path: Path,
) -> List[Dict[str, Any]]:
    manifest_by_id = {safe_text(row.get("sample_id")): row for row in read_jsonl(manifest_path)}
    out: List[Dict[str, Any]] = []
    for final_row in read_jsonl(final_path):
        sample_id = safe_text(final_row.get("sample_id"))
        manifest = manifest_by_id.get(sample_id)
        if manifest is None:
            continue
        reference = normalize_binary_label(final_row.get("reference_answer"))
        if reference is None:
            continue
        question = safe_text(manifest.get("question") or final_row.get("question"))
        if not question:
            continue
        video_path = safe_text(manifest.get("video_path"))
        if not video_path or not Path(video_path).exists():
            continue
        row = dict(manifest)
        row.update(
            {
                "source_dataset": "audio_unary_final",
                "task_family": TASK_AUDIO,
                "target_modality": "audio",
                "question": question,
                "formatted_question": safe_text(manifest.get("prompt_text")) or with_answer_instruction(question),
                "video_path": video_path,
                "audio_path": safe_text(manifest.get("audio_path")),
                "reference_answer": reference,
                "recorded_baseline_answer": normalize_binary_label(final_row.get("baseline_answer")),
                "recorded_policy_answer": normalize_binary_label(final_row.get("final_answer")),
                "recorded_target_branch_answer": normalize_binary_label(final_row.get("target_branch_answer")),
                "recorded_unary_answer": normalize_binary_label(final_row.get("unary_branch_contrastive_answer")),
            }
        )
        out.append(row)
    return out


def visual_rows(
    *,
    visual_replay_path: Path,
    qa_path: Path,
    video_dir: Path,
) -> List[Dict[str, Any]]:
    qa_rows = read_json(qa_path)
    out: List[Dict[str, Any]] = []
    for replay in read_jsonl(visual_replay_path):
        sample_id = safe_text(replay.get("sample_id"))
        try:
            qa_row = qa_lookup_by_index(qa_rows, sample_id)
        except ValueError:
            continue
        reference = normalize_binary_label(replay.get("reference_answer") or qa_row.get("label"))
        if reference is None:
            continue
        video_id = safe_text(qa_row.get("video_id"))
        video_path = video_dir / f"{video_id}.mp4"
        if not video_path.exists():
            continue
        question = safe_text(qa_row.get("text"))
        if not question:
            continue
        out.append(
            {
                "sample_id": sample_id,
                "source_dataset": "visual_prior_arbitration_replay",
                "task_family": TASK_VISUAL,
                "target_modality": "visual",
                "question": question,
                "formatted_question": with_answer_instruction(question),
                "video_path": str(video_path),
                "audio_path": "",
                "reference_answer": reference,
                "recorded_baseline_answer": normalize_binary_label(replay.get("baseline_answer")),
                "recorded_current_answer": normalize_binary_label(replay.get("current_answer")),
                "recorded_policy_answer": normalize_binary_label(replay.get("policy_answer")),
                "recorded_candidate_answer": normalize_binary_label(replay.get("candidate_answer")),
                "visual_replay_accepted": bool(replay.get("accepted")),
                "accepted_proof_domain": safe_text(replay.get("accepted_proof_domain")),
            }
        )
    return out


def take_mixed_rows(
    rows: Sequence[Dict[str, Any]],
    *,
    modality: str,
    limit: int,
    repair_fraction: float,
    seed: int,
) -> List[Dict[str, Any]]:
    if limit <= 0:
        return []
    rng = random.Random(int(seed))
    clean = [dict(row) for row in rows if normalize_binary_label(row.get("reference_answer")) is not None]
    for row in clean:
        reference = normalize_binary_label(row.get("reference_answer"))
        recorded_baseline = normalize_binary_label(
            row.get("recorded_current_answer")
            if modality == "visual"
            else row.get("recorded_baseline_answer")
        )
        recorded_policy = normalize_binary_label(row.get("recorded_policy_answer"))
        target_branch_answer = normalize_binary_label(row.get("recorded_target_branch_answer"))
        row["_selection_recorded_baseline"] = recorded_baseline
        row["_selection_recorded_policy"] = recorded_policy
        row["_selection_target_branch"] = target_branch_answer
        row["_selection_reference"] = reference

    repair_rows = [
        row
        for row in clean
        if row.get("_selection_recorded_baseline") != row.get("_selection_reference")
        and (
            row.get("_selection_recorded_policy") == row.get("_selection_reference")
            or row.get("_selection_target_branch") == row.get("_selection_reference")
            or bool(row.get("visual_replay_accepted"))
        )
    ]
    control_rows = [
        row
        for row in clean
        if row.get("_selection_recorded_baseline") == row.get("_selection_reference")
    ]
    fallback_rows = [row for row in clean if row not in repair_rows and row not in control_rows]
    for bucket in (repair_rows, control_rows, fallback_rows):
        rng.shuffle(bucket)

    repair_quota = min(len(repair_rows), int(round(limit * max(0.0, min(1.0, repair_fraction)))))
    selected = repair_rows[:repair_quota]
    remaining = limit - len(selected)
    selected.extend(control_rows[:remaining])
    remaining = limit - len(selected)
    selected.extend(repair_rows[repair_quota : repair_quota + remaining])
    remaining = limit - len(selected)
    selected.extend(fallback_rows[:remaining])
    return selected[:limit]


def direction_vector(
    *,
    mode: str,
    layer: int,
    captures: Dict[str, Dict[str, Any]],
    scale: str,
) -> torch.Tensor:
    full = captures["full"]["layer_vectors"][int(layer)].float()
    target = captures["target"]["layer_vectors"][int(layer)].float()
    non_target = captures["non_target"]["layer_vectors"][int(layer)].float()
    if mode == "target_minus_full":
        direction = target - full
    elif mode == "target_minus_nontarget":
        direction = target - non_target
    elif mode == "full_minus_nontarget":
        direction = full - non_target
    else:
        raise ValueError(f"Unknown direction mode: {mode}")
    if scale == "unit_full_norm":
        direction_norm = direction.norm().clamp_min(1e-12)
        full_norm = full.norm().clamp_min(1e-12)
        direction = direction / direction_norm * full_norm
    return direction.detach().float().cpu()


@torch.no_grad()
def patched_full_logits(
    adapter: QwenOmniAdapter,
    *,
    row: Dict[str, Any],
    layer: int,
    direction: torch.Tensor,
    beta: float,
    yes_id: int,
    no_id: int,
    budget: Dict[str, float],
) -> Dict[str, float]:
    inputs = prepare_branch_inputs(adapter, row=row, branch="full", budget=budget)
    thinker = adapter._model.thinker

    def patch_hook(_module, _inp, output):
        hidden_states = output[0] if isinstance(output, (tuple, list)) else output
        patched = hidden_states.clone()
        delta = direction.to(device=patched.device, dtype=patched.dtype) * float(beta)
        patched[0, -1, :] = patched[0, -1, :] + delta
        if isinstance(output, tuple):
            return (patched,) + tuple(output[1:])
        if isinstance(output, list):
            return [patched] + list(output[1:])
        return patched

    hook = thinker.model.layers[int(layer)].register_forward_hook(patch_hook)
    try:
        outputs = thinker(**inputs, use_cache=False)
    finally:
        hook.remove()

    logits = outputs.logits[:, -1, [yes_id, no_id]].detach().float().cpu()[0]
    probs = torch.softmax(logits, dim=0)
    margin = float(logits[0].item() - logits[1].item())
    result = {
        "yes_logit": float(logits[0].item()),
        "no_logit": float(logits[1].item()),
        "yes_prob": float(probs[0].item()),
        "no_prob": float(probs[1].item()),
        "margin_yes_minus_no": margin,
        "prediction": pred_from_margin(margin),
    }
    del inputs, outputs
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return result


def mean(values: Sequence[Any]) -> Optional[float]:
    clean: List[float] = []
    for value in values:
        if value is None:
            continue
        try:
            number = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(number):
            clean.append(number)
    if not clean:
        return None
    return float(sum(clean) / len(clean))


def summarize_records(records: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "n_edit_records": len(records),
        "by_spec": {},
        "by_modality_spec": {},
        "best_by_delta_accuracy": [],
        "best_by_net": [],
    }
    groupers = {
        "by_spec": lambda r: (
            safe_text(r.get("direction_mode")),
            str(r.get("layer")),
            str(r.get("beta")),
        ),
        "by_modality_spec": lambda r: (
            safe_text(r.get("target_modality")),
            safe_text(r.get("direction_mode")),
            str(r.get("layer")),
            str(r.get("beta")),
        ),
    }
    flat_specs: List[Dict[str, Any]] = []
    for group_name, key_fn in groupers.items():
        grouped: Dict[Tuple[str, ...], List[Dict[str, Any]]] = defaultdict(list)
        for record in records:
            grouped[key_fn(record)].append(record)
        group_out: Dict[str, Any] = {}
        for key, rows in sorted(grouped.items()):
            n = len(rows)
            baseline_correct = sum(1 for row in rows if row.get("baseline_correct"))
            edited_correct = sum(1 for row in rows if row.get("edited_correct"))
            w2c = sum(1 for row in rows if (not row.get("baseline_correct")) and row.get("edited_correct"))
            c2w = sum(1 for row in rows if row.get("baseline_correct") and (not row.get("edited_correct")))
            spec_summary = {
                "n": n,
                "baseline_correct": baseline_correct,
                "edited_correct": edited_correct,
                "baseline_accuracy": baseline_correct / n if n else None,
                "edited_accuracy": edited_correct / n if n else None,
                "delta_accuracy_pp": ((edited_correct - baseline_correct) / n * 100.0) if n else None,
                "wrong_to_correct": w2c,
                "correct_to_wrong": c2w,
                "net": w2c - c2w,
                "flips": sum(1 for row in rows if row.get("edited_prediction") != row.get("baseline_prediction")),
                "mean_delta_margin": mean([row.get("delta_margin") for row in rows]),
                "mean_delta_reference_aligned_margin": mean(
                    [row.get("delta_reference_aligned_margin") for row in rows]
                ),
                "mean_direction_norm": mean([row.get("direction_norm") for row in rows]),
            }
            key_text = " | ".join(key)
            group_out[key_text] = spec_summary
            if group_name == "by_spec":
                flat = dict(spec_summary)
                flat.update({"spec": key_text})
                flat_specs.append(flat)
        out[group_name] = group_out
    out["best_by_delta_accuracy"] = sorted(
        flat_specs,
        key=lambda item: (
            float(item.get("delta_accuracy_pp") or 0.0),
            int(item.get("net") or 0),
            -int(item.get("correct_to_wrong") or 0),
        ),
        reverse=True,
    )[:10]
    out["best_by_net"] = sorted(
        flat_specs,
        key=lambda item: (
            int(item.get("net") or 0),
            float(item.get("delta_accuracy_pp") or 0.0),
            -int(item.get("correct_to_wrong") or 0),
        ),
        reverse=True,
    )[:10]
    return out


def row_plan(row: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "sample_id": row.get("sample_id"),
        "source_dataset": row.get("source_dataset"),
        "target_modality": row.get("target_modality"),
        "task_family": row.get("task_family"),
        "question": row.get("question"),
        "reference_answer": row.get("reference_answer"),
        "recorded_baseline_answer": row.get("recorded_baseline_answer") or row.get("recorded_current_answer"),
        "recorded_policy_answer": row.get("recorded_policy_answer"),
        "recorded_candidate_answer": row.get("recorded_candidate_answer"),
        "recorded_target_branch_answer": row.get("recorded_target_branch_answer"),
        "visual_replay_accepted": row.get("visual_replay_accepted"),
        "accepted_proof_domain": row.get("accepted_proof_domain"),
    }


def run() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    layers = sorted({int(layer) for layer in args.layers})
    betas = stable_unique_floats(args.betas)
    direction_modes = [safe_text(mode) for mode in args.direction_modes]

    audio_pool = joined_audio_rows(manifest_path=args.audio_manifest, final_path=args.audio_final)
    visual_pool = visual_rows(
        visual_replay_path=args.visual_replay,
        qa_path=args.avh_qa,
        video_dir=args.avh_video_dir,
    )
    selected_audio = take_mixed_rows(
        audio_pool,
        modality="audio",
        limit=int(args.max_audio),
        repair_fraction=float(args.repair_fraction),
        seed=int(args.seed),
    )
    selected_visual = take_mixed_rows(
        visual_pool,
        modality="visual",
        limit=int(args.max_visual),
        repair_fraction=float(args.repair_fraction),
        seed=int(args.seed) + 17,
    )
    selected = selected_audio + selected_visual
    plans = [row_plan(row) for row in selected]
    write_jsonl(args.output_dir / "selected_rows.jsonl", plans)

    run_config = {
        "audio_manifest": str(args.audio_manifest),
        "audio_final": str(args.audio_final),
        "visual_replay": str(args.visual_replay),
        "avh_qa": str(args.avh_qa),
        "output_dir": str(args.output_dir),
        "model_path": args.model_path,
        "device": args.device,
        "seed": int(args.seed),
        "max_audio": int(args.max_audio),
        "max_visual": int(args.max_visual),
        "repair_fraction": float(args.repair_fraction),
        "layers": layers,
        "betas": betas,
        "direction_modes": direction_modes,
        "direction_scale": args.direction_scale,
        "target_capture_prompt": args.target_capture_prompt,
        "non_target_capture_prompt": args.non_target_capture_prompt,
        "video_budget": video_budget(args),
        "audio_pool": len(audio_pool),
        "visual_pool": len(visual_pool),
        "selected": len(selected),
        "selected_by_modality": dict(Counter(row.get("target_modality") for row in selected)),
    }
    write_json(args.output_dir / "run_config.json", run_config)
    if args.plan_only:
        print(f"[unary-hidden-edit] plan-only selected={len(selected)} output_dir={args.output_dir}")
        return

    adapter = QwenOmniAdapter(
        model_path=args.model_path,
        device=args.device,
        max_new_tokens=4,
    )
    yes_id, no_id = yes_no_token_ids(adapter)
    budget = video_budget(args)
    edit_records: List[Dict[str, Any]] = []
    branch_records: List[Dict[str, Any]] = []

    iterator = tqdm(
        selected,
        total=len(selected),
        desc="qwen_owp_hidden_edit",
        unit="sample",
        disable=bool(args.no_progress),
        dynamic_ncols=True,
    )
    try:
        for row in iterator:
            reference = normalize_binary_label(row.get("reference_answer"))
            if reference is None:
                continue
            target_branch = target_branch_for(row)
            non_target_branch = non_target_branch_for(row)
            full_capture = capture_branch(
                adapter,
                row=row,
                branch="full",
                layers=layers,
                yes_id=yes_id,
                no_id=no_id,
                budget=budget,
            )
            target_capture = capture_branch(
                adapter,
                row=capture_prompt_row(row, prompt_mode=args.target_capture_prompt),
                branch=target_branch,
                layers=layers,
                yes_id=yes_id,
                no_id=no_id,
                budget=budget,
            )
            non_target_capture = capture_branch(
                adapter,
                row=capture_prompt_row(row, prompt_mode=args.non_target_capture_prompt),
                branch=non_target_branch,
                layers=layers,
                yes_id=yes_id,
                no_id=no_id,
                budget=budget,
            )
            captures = {
                "full": full_capture,
                "target": target_capture,
                "non_target": non_target_capture,
            }
            full_margin = float(captures["full"]["final_yes_no_margin"])
            full_prediction = safe_text(captures["full"]["final_prediction"])
            full_correct = full_prediction == reference

            branch_capture_specs = (
                ("full", "full", "original", full_capture),
                ("target", target_branch, args.target_capture_prompt, target_capture),
                ("non_target", non_target_branch, args.non_target_capture_prompt, non_target_capture),
            )
            for branch_role, branch_name, prompt_mode, branch_capture in branch_capture_specs:
                branch_records.append(
                    {
                        "sample_id": row.get("sample_id"),
                        "target_modality": row.get("target_modality"),
                        "task_family": row.get("task_family"),
                        "branch_role": branch_role,
                        "branch": branch_name,
                        "capture_prompt": prompt_mode,
                        "token_count": branch_capture.get("token_count"),
                        "margin_yes_minus_no": branch_capture.get("final_yes_no_margin"),
                        "prediction": branch_capture.get("final_prediction"),
                        "reference_answer": reference,
                        "correct": branch_capture.get("final_prediction") == reference,
                    }
                )

            for mode in direction_modes:
                for layer in layers:
                    direction = direction_vector(
                        mode=mode,
                        layer=int(layer),
                        captures=captures,
                        scale=args.direction_scale,
                    )
                    direction_norm = float(direction.norm().item())
                    for beta in betas:
                        edited = patched_full_logits(
                            adapter,
                            row=row,
                            layer=int(layer),
                            direction=direction,
                            beta=float(beta),
                            yes_id=yes_id,
                            no_id=no_id,
                            budget=budget,
                        )
                        edited_margin = float(edited["margin_yes_minus_no"])
                        edited_prediction = safe_text(edited["prediction"])
                        edited_correct = edited_prediction == reference
                        baseline_ref_margin = signed_margin_for_answer(full_margin, reference)
                        edited_ref_margin = signed_margin_for_answer(edited_margin, reference)
                        record = {
                            "sample_id": row.get("sample_id"),
                            "source_dataset": row.get("source_dataset"),
                            "target_modality": row.get("target_modality"),
                            "task_family": row.get("task_family"),
                            "question": row.get("question"),
                            "reference_answer": reference,
                            "target_branch": target_branch,
                            "non_target_branch": non_target_branch,
                            "direction_mode": mode,
                            "direction_scale": args.direction_scale,
                            "target_capture_prompt": args.target_capture_prompt,
                            "non_target_capture_prompt": args.non_target_capture_prompt,
                            "layer": int(layer),
                            "beta": float(beta),
                            "direction_norm": direction_norm,
                            "baseline_margin": full_margin,
                            "baseline_prediction": full_prediction,
                            "baseline_correct": full_correct,
                            "edited_margin": edited_margin,
                            "edited_prediction": edited_prediction,
                            "edited_correct": edited_correct,
                            "delta_margin": edited_margin - full_margin,
                            "baseline_reference_aligned_margin": baseline_ref_margin,
                            "edited_reference_aligned_margin": edited_ref_margin,
                            "delta_reference_aligned_margin": (
                                edited_ref_margin - baseline_ref_margin
                                if edited_ref_margin is not None and baseline_ref_margin is not None
                                else None
                            ),
                            "target_branch_prediction": captures["target"]["final_prediction"],
                            "target_branch_margin": captures["target"]["final_yes_no_margin"],
                            "target_branch_correct": captures["target"]["final_prediction"] == reference,
                            "non_target_branch_prediction": captures["non_target"]["final_prediction"],
                            "non_target_branch_margin": captures["non_target"]["final_yes_no_margin"],
                            "non_target_branch_correct": captures["non_target"]["final_prediction"] == reference,
                            "recorded_baseline_answer": row.get("recorded_baseline_answer")
                            or row.get("recorded_current_answer"),
                            "recorded_policy_answer": row.get("recorded_policy_answer"),
                            "recorded_candidate_answer": row.get("recorded_candidate_answer"),
                            "recorded_target_branch_answer": row.get("recorded_target_branch_answer"),
                            "visual_replay_accepted": row.get("visual_replay_accepted"),
                            "accepted_proof_domain": row.get("accepted_proof_domain"),
                        }
                        edit_records.append(record)
            iterator.set_postfix(
                modality=safe_text(row.get("target_modality")),
                base=full_prediction,
                ref=reference,
            )
    finally:
        del adapter
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    write_jsonl(args.output_dir / "branch_records.jsonl", branch_records)
    write_jsonl(args.output_dir / "edit_records.jsonl", edit_records)
    summary = {
        "run_config": run_config,
        "yes_token_id": yes_id,
        "no_token_id": no_id,
        "selected_by_modality": dict(Counter(row.get("target_modality") for row in selected)),
        "selected_by_source_dataset": dict(Counter(row.get("source_dataset") for row in selected)),
        "branch_summary": {
            "n": len(branch_records),
            "by_modality_branch": {},
        },
        "edit_summary": summarize_records(edit_records),
    }
    grouped_branch: Dict[Tuple[str, str, str, str], List[Dict[str, Any]]] = defaultdict(list)
    for record in branch_records:
        grouped_branch[
            (
                safe_text(record.get("target_modality")),
                safe_text(record.get("branch_role")),
                safe_text(record.get("branch")),
                safe_text(record.get("capture_prompt")),
            )
        ].append(record)
    for key, rows in sorted(grouped_branch.items()):
        n = len(rows)
        correct = sum(1 for row in rows if row.get("correct"))
        summary["branch_summary"]["by_modality_branch"][" | ".join(key)] = {
            "n": n,
            "correct": correct,
            "accuracy": correct / n if n else None,
            "mean_margin_yes_minus_no": mean([row.get("margin_yes_minus_no") for row in rows]),
        }
    write_json(args.output_dir / "summary.json", summary)
    print(f"[unary-hidden-edit] wrote {args.output_dir / 'edit_records.jsonl'}")
    print(f"[unary-hidden-edit] wrote {args.output_dir / 'summary.json'}")


if __name__ == "__main__":
    run()
