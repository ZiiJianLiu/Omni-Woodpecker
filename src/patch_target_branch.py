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
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
PACKAGE_ROOT = ROOT.parent
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from owp.models.qwen_omni import QwenOmniAdapter
from owp.evaluation.answer_policy import (
    build_query_latent_state,
    format_question_for_answer_space,
)


DEFAULT_SAMPLES = ROOT / "data" / "manifests" / "unary_rebalance_manifest.jsonl"
DEFAULT_OUTPUT_DIR = ROOT / "results" / "qwen_owp_branch_patching"
DEFAULT_MODEL_PATH = "Qwen/Qwen2.5-Omni-7B"

TASK_AUDIO = "audio_grounded_presence"
TASK_VISUAL = "visual_grounded_presence"
BRANCHES = ("full", "audio_only", "visual_only", "text_only")
PATCH_SOURCES = ("target", "non_target", "text")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Causal patching probe: inject target-branch late final-token states "
            "into the full branch and compare against non-target/text controls."
        )
    )
    parser.add_argument("--samples", type=Path, default=DEFAULT_SAMPLES)
    parser.add_argument(
        "--filter-target-disagrees-from-records",
        type=Path,
        default=None,
        help=(
            "Optional internal_pattern_records.jsonl. If set, only samples where "
            "the recorded target branch prediction disagrees with the recorded full "
            "prediction are selected."
        ),
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--model-path", type=str, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--seed", type=int, default=20260427)
    parser.add_argument("--per-bucket", type=int, default=4)
    parser.add_argument("--max-total", type=int, default=0)
    parser.add_argument("--layers", type=int, nargs="+", default=[20, 24, 26])
    parser.add_argument("--sources", nargs="+", default=list(PATCH_SOURCES))
    parser.add_argument("--video-fps", type=float, default=4.0)
    parser.add_argument("--video-max-frames", type=int, default=32)
    parser.add_argument("--video-max-pixels", type=int, default=602112)
    parser.add_argument("--video-min-pixels", type=int, default=100352)
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--no-progress", action="store_true")
    return parser.parse_args()


def safe_text(value: Any) -> str:
    return "" if value is None else str(value).strip()


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def norm_yes_no(value: Any) -> Optional[str]:
    text = safe_text(value).lower()
    if text.startswith("yes"):
        return "Yes"
    if text.startswith("no"):
        return "No"
    return None


def pred_from_margin(value: float) -> str:
    return "Yes" if float(value) >= 0.0 else "No"


def signed_margin_for_answer(margin: float, answer: Optional[str]) -> Optional[float]:
    if answer == "Yes":
        return float(margin)
    if answer == "No":
        return -float(margin)
    return None


def target_branch_for(row: Dict[str, Any]) -> str:
    family = safe_text(row.get("task_family"))
    if family == TASK_AUDIO:
        return "audio_only"
    if family == TASK_VISUAL:
        return "visual_only"
    raise ValueError(f"Unsupported task_family={family!r} for sample={row.get('sample_id')}")


def non_target_branch_for(row: Dict[str, Any]) -> str:
    return "visual_only" if target_branch_for(row) == "audio_only" else "audio_only"


def branch_for_source(row: Dict[str, Any], source: str) -> str:
    if source == "target":
        return target_branch_for(row)
    if source == "non_target":
        return non_target_branch_for(row)
    if source == "text":
        return "text_only"
    if source in BRANCHES:
        return source
    raise ValueError(f"Unknown source spec: {source}")


def bucket_key(row: Dict[str, Any]) -> str:
    return safe_text(row.get("pattern_bucket")) or "unknown"


def select_rows(args: argparse.Namespace, rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rng = random.Random(int(args.seed))
    allowed_ids: Optional[set[str]] = None
    if args.filter_target_disagrees_from_records is not None:
        allowed_ids = target_disagreement_sample_ids(args.filter_target_disagrees_from_records)

    by_bucket: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if allowed_ids is not None and safe_text(row.get("sample_id")) not in allowed_ids:
            continue
        if safe_text(row.get("task_family")) in {TASK_AUDIO, TASK_VISUAL}:
            by_bucket[bucket_key(row)].append(row)

    wanted_order = [
        "audio_no_to_yes_wrong",
        "audio_yes_to_no_wrong",
        "audio_no_to_no_correct",
        "audio_yes_to_yes_correct",
        "visual_no_to_yes_wrong",
        "visual_yes_to_no_wrong",
        "visual_no_to_no_correct",
        "visual_yes_to_yes_correct",
    ]
    selected: List[Dict[str, Any]] = []
    for key in wanted_order:
        bucket_rows = list(by_bucket.get(key) or [])
        rng.shuffle(bucket_rows)
        selected.extend(bucket_rows[: max(0, int(args.per_bucket))])
    if args.max_total and args.max_total > 0:
        rng.shuffle(selected)
        selected = selected[: int(args.max_total)]
    return selected


def target_disagreement_sample_ids(path: Path) -> set[str]:
    out: set[str] = set()
    for row in read_jsonl(path):
        family = safe_text(row.get("task_family"))
        if family == TASK_AUDIO:
            target = "audio_only"
        elif family == TASK_VISUAL:
            target = "visual_only"
        else:
            continue
        traces = row.get("branch_traces") or {}
        full_pred = safe_text((traces.get("full") or {}).get("final_prediction"))
        target_pred = safe_text((traces.get(target) or {}).get("final_prediction"))
        if full_pred and target_pred and full_pred != target_pred:
            out.add(safe_text(row.get("sample_id")))
    return out


def formatted_question(row: Dict[str, Any]) -> str:
    question = safe_text(row.get("formatted_question") or row.get("question"))
    if "Answer with only Yes or No." in question:
        return question
    query_state = build_query_latent_state(
        safe_text(row.get("question")),
        eval_type=safe_text(row.get("eval_type")) or "yes_no",
        options=[],
    )
    return format_question_for_answer_space(safe_text(row.get("question")), query_state.answer_space)


def video_budget(args: argparse.Namespace) -> Dict[str, float]:
    return {
        "fps": float(args.video_fps),
        "max_frames": int(args.video_max_frames),
        "max_pixels": int(args.video_max_pixels),
        "min_pixels": int(args.video_min_pixels),
    }


def yes_no_token_ids(adapter: QwenOmniAdapter) -> Tuple[int, int]:
    tokenizer = adapter._processor.tokenizer
    yes_id = tokenizer.encode("Yes", add_special_tokens=False)[0]
    no_id = tokenizer.encode("No", add_special_tokens=False)[0]
    return int(yes_id), int(no_id)


def branch_masks(branch: str) -> Tuple[bool, bool]:
    if branch == "full":
        return False, False
    if branch == "audio_only":
        return False, True
    if branch == "visual_only":
        return True, False
    if branch == "text_only":
        return True, True
    raise ValueError(f"Unknown branch: {branch}")


def prepare_branch_inputs(
    adapter: QwenOmniAdapter,
    *,
    row: Dict[str, Any],
    branch: str,
    budget: Dict[str, float],
) -> Dict[str, torch.Tensor]:
    mask_audio, mask_visual = branch_masks(branch)
    video_path = safe_text(row.get("video_path")) or None
    audio_path = safe_text(row.get("audio_path")) or None
    if branch == "text_only":
        video_path = None
    audio_array = adapter._resolve_audio_array(
        video_path=video_path,
        audio_path=audio_path,
        mask_audio=mask_audio,
    )
    messages = adapter._build_messages(
        video_path,
        formatted_question(row),
        audio_array=audio_array,
        mask_visual=mask_visual,
    )
    return adapter._messages_to_inputs(
        messages,
        audio_array=audio_array,
        video_budget=budget,
    )


@torch.no_grad()
def capture_branch(
    adapter: QwenOmniAdapter,
    *,
    row: Dict[str, Any],
    branch: str,
    layers: Sequence[int],
    yes_id: int,
    no_id: int,
    budget: Dict[str, float],
) -> Dict[str, Any]:
    inputs = prepare_branch_inputs(adapter, row=row, branch=branch, budget=budget)
    thinker = adapter._model.thinker
    captures: Dict[int, torch.Tensor] = {}
    hooks = []

    def make_hook(layer_id: int):
        def hook_fn(_module, _inp, output):
            hidden_states = output[0] if isinstance(output, (tuple, list)) else output
            captures[int(layer_id)] = hidden_states[0, -1, :].detach().float().cpu()

        return hook_fn

    for layer in layers:
        hooks.append(thinker.model.layers[int(layer)].register_forward_hook(make_hook(int(layer))))
    try:
        outputs = thinker(**inputs, use_cache=False)
    finally:
        for hook in hooks:
            hook.remove()

    logits = outputs.logits[:, -1, [yes_id, no_id]].detach().float().cpu()[0]
    margin = float(logits[0].item() - logits[1].item())
    result = {
        "branch": branch,
        "token_count": int(inputs["input_ids"].shape[-1]),
        "final_yes_no_margin": margin,
        "final_prediction": pred_from_margin(margin),
        "layer_vectors": captures,
    }

    del inputs, outputs
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return result


@torch.no_grad()
def patched_full_margin(
    adapter: QwenOmniAdapter,
    *,
    row: Dict[str, Any],
    layer: int,
    source_vector: torch.Tensor,
    yes_id: int,
    no_id: int,
    budget: Dict[str, float],
) -> float:
    inputs = prepare_branch_inputs(adapter, row=row, branch="full", budget=budget)
    thinker = adapter._model.thinker

    def patch_hook(_module, _inp, output):
        hidden_states = output[0] if isinstance(output, (tuple, list)) else output
        patched = hidden_states.clone()
        replacement = source_vector.to(device=patched.device, dtype=patched.dtype)
        patched[0, -1, :] = replacement
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
    margin = float(logits[0].item() - logits[1].item())
    del inputs, outputs
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return margin


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
    return float(sum(clean) / len(clean)) if clean else None


def summarize_patch_records(records: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    summary: Dict[str, Any] = {
        "n_patch_records": len(records),
        "by_source": {},
        "by_source_layer": {},
        "by_source_layer_stored_correct": {},
        "by_source_layer_source_agreement": {},
        "by_pattern_bucket_source_layer": {},
    }

    group_specs = {
        "by_source": lambda r: (safe_text(r.get("source_role")),),
        "by_source_layer": lambda r: (safe_text(r.get("source_role")), str(r.get("layer"))),
        "by_source_layer_stored_correct": lambda r: (
            safe_text(r.get("source_role")),
            str(r.get("layer")),
            "stored_correct" if r.get("stored_base_correct") else "stored_wrong",
        ),
        "by_source_layer_source_agreement": lambda r: (
            safe_text(r.get("source_role")),
            str(r.get("layer")),
            "source_agrees_full" if r.get("source_prediction") == r.get("baseline_full_prediction") else "source_disagrees_full",
        ),
        "by_pattern_bucket_source_layer": lambda r: (
            safe_text(r.get("pattern_bucket")),
            safe_text(r.get("source_role")),
            str(r.get("layer")),
        ),
    }

    for group_name, key_fn in group_specs.items():
        grouped: Dict[Tuple[str, ...], List[Dict[str, Any]]] = defaultdict(list)
        for record in records:
            grouped[key_fn(record)].append(record)
        out: Dict[str, Any] = {}
        for key, rows in sorted(grouped.items()):
            key_text = " | ".join(key)
            source_disagrees = [r for r in rows if r.get("source_prediction") != r.get("baseline_full_prediction")]
            out[key_text] = {
                "n": len(rows),
                "source_disagrees_full": len(source_disagrees),
                "prediction_flips": sum(1 for r in rows if r.get("prediction_flipped")),
                "patched_matches_source": sum(1 for r in rows if r.get("patched_prediction") == r.get("source_prediction")),
                "source_control_rate_on_disagreement": (
                    sum(1 for r in source_disagrees if r.get("patched_prediction") == r.get("source_prediction"))
                    / len(source_disagrees)
                    if source_disagrees
                    else None
                ),
                "wrong_to_correct": sum(
                    1
                    for r in rows
                    if (not r.get("baseline_full_correct")) and r.get("patched_correct")
                ),
                "correct_to_wrong": sum(
                    1
                    for r in rows
                    if r.get("baseline_full_correct") and (not r.get("patched_correct"))
                ),
                "stored_wrong_to_reference": sum(
                    1
                    for r in rows
                    if (not r.get("stored_base_correct")) and r.get("patched_prediction") == r.get("reference_answer")
                ),
                "stored_correct_to_nonreference": sum(
                    1
                    for r in rows
                    if r.get("stored_base_correct") and r.get("patched_prediction") != r.get("reference_answer")
                ),
                "mean_delta_yes_no_margin": mean([r.get("delta_yes_no_margin") for r in rows]),
                "mean_delta_source_aligned_margin": mean([r.get("delta_source_aligned_margin") for r in rows]),
                "mean_delta_reference_aligned_margin": mean([r.get("delta_reference_aligned_margin") for r in rows]),
            }
        summary[group_name] = out
    return summary


def row_plan(row: Dict[str, Any], *, layers: Sequence[int], sources: Sequence[str]) -> Dict[str, Any]:
    return {
        "sample_id": row.get("sample_id"),
        "benchmark": row.get("benchmark"),
        "pattern_bucket": row.get("pattern_bucket"),
        "task_family": row.get("task_family"),
        "question": row.get("question"),
        "reference_answer": row.get("reference_answer"),
        "stored_base_prediction": row.get("stored_base_prediction"),
        "stored_base_correct": row.get("stored_base_correct"),
        "target_branch": target_branch_for(row),
        "non_target_branch": non_target_branch_for(row),
        "layers": list(layers),
        "sources": list(sources),
        "n_patch_specs": len(layers) * len(sources),
    }


def run() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    layers = sorted({int(layer) for layer in args.layers})
    sources = [safe_text(source) for source in args.sources]

    rows = read_jsonl(args.samples)
    selected = select_rows(args, rows)
    plans = [row_plan(row, layers=layers, sources=sources) for row in selected]
    write_jsonl(args.output_dir / "selected_patch_samples.jsonl", plans)
    write_json(
        args.output_dir / "plan_summary.json",
        {
            "samples": str(args.samples),
            "output_dir": str(args.output_dir),
            "selected": len(selected),
            "selected_by_bucket": dict(Counter(row.get("pattern_bucket") for row in selected)),
            "filter_target_disagrees_from_records": (
                str(args.filter_target_disagrees_from_records)
                if args.filter_target_disagrees_from_records is not None
                else None
            ),
            "layers": layers,
            "sources": sources,
            "patch_specs": sum(plan["n_patch_specs"] for plan in plans),
            "video_budget": video_budget(args),
        },
    )
    if args.plan_only:
        print(f"[target-branch-patching] plan-only selected={len(selected)} output_dir={args.output_dir}")
        return

    adapter = QwenOmniAdapter(
        model_path=args.model_path,
        device=args.device,
        max_new_tokens=4,
    )
    yes_id, no_id = yes_no_token_ids(adapter)
    budget = video_budget(args)

    patch_records: List[Dict[str, Any]] = []
    error_records: List[Dict[str, Any]] = []
    iterator = tqdm(
        selected,
        total=len(selected),
        desc="target_branch_patch",
        unit="sample",
        disable=bool(args.no_progress),
        dynamic_ncols=True,
    )
    try:
        for row in iterator:
            try:
                reference = norm_yes_no(row.get("reference_answer"))
                target_branch = target_branch_for(row)
                non_target_branch = non_target_branch_for(row)
                source_branches = sorted({"full"} | {branch_for_source(row, source) for source in sources})

                captures = {
                    branch: capture_branch(
                        adapter,
                        row=row,
                        branch=branch,
                        layers=layers,
                        yes_id=yes_id,
                        no_id=no_id,
                        budget=budget,
                    )
                    for branch in source_branches
                }
                full_margin = float(captures["full"]["final_yes_no_margin"])
                full_prediction = safe_text(captures["full"]["final_prediction"])
                full_correct = full_prediction == reference

                for source_role in sources:
                    source_branch = branch_for_source(row, source_role)
                    source_margin = float(captures[source_branch]["final_yes_no_margin"])
                    source_prediction = safe_text(captures[source_branch]["final_prediction"])
                    for layer in layers:
                        source_vector = captures[source_branch]["layer_vectors"].get(int(layer))
                        if source_vector is None:
                            continue
                        patched_margin = patched_full_margin(
                            adapter,
                            row=row,
                            layer=int(layer),
                            source_vector=source_vector,
                            yes_id=yes_id,
                            no_id=no_id,
                            budget=budget,
                        )
                        patched_prediction = pred_from_margin(patched_margin)
                        baseline_source_aligned = signed_margin_for_answer(full_margin, source_prediction)
                        patched_source_aligned = signed_margin_for_answer(patched_margin, source_prediction)
                        baseline_ref_aligned = signed_margin_for_answer(full_margin, reference)
                        patched_ref_aligned = signed_margin_for_answer(patched_margin, reference)

                        record = {
                            "sample_id": row.get("sample_id"),
                            "benchmark": row.get("benchmark"),
                            "pattern_bucket": row.get("pattern_bucket"),
                            "task_family": row.get("task_family"),
                            "question": row.get("question"),
                            "reference_answer": reference,
                            "stored_base_prediction": norm_yes_no(row.get("stored_base_prediction")),
                            "stored_base_correct": bool(row.get("stored_base_correct")),
                            "target_branch": target_branch,
                            "non_target_branch": non_target_branch,
                            "source_role": source_role,
                            "source_branch": source_branch,
                            "layer": int(layer),
                            "baseline_full_margin": full_margin,
                            "baseline_full_prediction": full_prediction,
                            "baseline_full_correct": full_correct,
                            "source_margin": source_margin,
                            "source_prediction": source_prediction,
                            "source_supports_reference": source_prediction == reference,
                            "patched_margin": patched_margin,
                            "patched_prediction": patched_prediction,
                            "patched_correct": patched_prediction == reference,
                            "prediction_flipped": patched_prediction != full_prediction,
                            "patched_matches_source": patched_prediction == source_prediction,
                            "delta_yes_no_margin": patched_margin - full_margin,
                            "baseline_source_aligned_margin": baseline_source_aligned,
                            "patched_source_aligned_margin": patched_source_aligned,
                            "delta_source_aligned_margin": (
                                patched_source_aligned - baseline_source_aligned
                                if patched_source_aligned is not None and baseline_source_aligned is not None
                                else None
                            ),
                            "baseline_reference_aligned_margin": baseline_ref_aligned,
                            "patched_reference_aligned_margin": patched_ref_aligned,
                            "delta_reference_aligned_margin": (
                                patched_ref_aligned - baseline_ref_aligned
                                if patched_ref_aligned is not None and baseline_ref_aligned is not None
                                else None
                            ),
                        }
                        patch_records.append(record)
                iterator.set_postfix(bucket=safe_text(row.get("pattern_bucket"))[:24])
            except Exception as exc:  # Keep a batch run auditable when one media item fails.
                error_records.append(
                    {
                        "sample_id": row.get("sample_id"),
                        "pattern_bucket": row.get("pattern_bucket"),
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                    }
                )
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                iterator.set_postfix(error=safe_text(row.get("sample_id"))[:24])
    finally:
        del adapter
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    write_jsonl(args.output_dir / "patch_records.jsonl", patch_records)
    write_jsonl(args.output_dir / "error_records.jsonl", error_records)
    summary = {
        "samples": str(args.samples),
        "output_dir": str(args.output_dir),
        "selected": len(selected),
        "completed_samples": len({safe_text(row.get("sample_id")) for row in patch_records}),
        "failed_samples": len(error_records),
        "selected_by_bucket": dict(Counter(row.get("pattern_bucket") for row in selected)),
        "filter_target_disagrees_from_records": (
            str(args.filter_target_disagrees_from_records)
            if args.filter_target_disagrees_from_records is not None
            else None
        ),
        "layers": layers,
        "sources": sources,
        "video_budget": budget,
        "yes_token_id": yes_id,
        "no_token_id": no_id,
        "errors_preview": error_records[:20],
        "summary": summarize_patch_records(patch_records),
    }
    write_json(args.output_dir / "summary.json", summary)
    print(f"[target-branch-patching] wrote {args.output_dir / 'patch_records.jsonl'}")
    print(f"[target-branch-patching] wrote {args.output_dir / 'summary.json'}")


if __name__ == "__main__":
    run()
