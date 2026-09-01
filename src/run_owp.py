#!/usr/bin/env python3
"""Five-pass, cache-reusing OWP efficiency runner for VideoLLaMA2.1-7B-AV.

Each timed sample performs exactly four diagnostic views (full, audio, visual,
and text) plus one intervened full-view pass.  Hidden states and source
attention measured during diagnosis are reused to construct the intervention.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping, Sequence

import torch


# Resolve all runtime resources relative to this release by default. Set
# OWP_REPO_ROOT when embedding the executor in another checkout.
ROOT = Path(os.environ.get("OWP_REPO_ROOT", str(Path(__file__).resolve().parents[1]))).resolve()
VIDEOLLAMA2_ROOT = ROOT / "third_party" / "VideoLLaMA2"
for _path in (ROOT, ROOT / "scripts", VIDEOLLAMA2_ROOT):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from videollama2 import model_init  # noqa: E402
from owp.assets import build_dataset_manifest, cache_root, prepare_dataset  # noqa: E402

import intervention_videollama2 as executor  # noqa: E402
from probe_qwen import online_target_modality_from_question  # noqa: E402
from videollama2_runtime import normalize_yes_no, safe_text, torch_dtype_from_name  # noqa: E402
from multimodal_inputs import (  # noqa: E402
    candidate_token_ids,
    language_layers,
    prepare_inputs,
)
from probe_videollama2 import (  # noqa: E402
    FROZEN_ALL_LAYERS,
    _qwen2_last_query_attention_row,
    build_base_conflict_budget,
    build_frozen_carriers,
    infer_stc_video_feature_len,
    modality_presence,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Five-pass cached OWP latency runner for VideoLLaMA2.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeat", type=int, required=True)
    parser.add_argument(
        "--model-path",
        default="DAMO-NLP-SG/VideoLLaMA2.1-7B-AV",
        help="Local checkpoint path or Hugging Face model id; missing local paths are fetched automatically.",
    )
    parser.add_argument("--dtype", choices=("float16", "bfloat16"), default="bfloat16")
    parser.add_argument("--max-rows", type=int, default=0)
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    # Source-only checkouts retain small manifests but omit media.  Resolve a
    # missing benchmark payload automatically and use a cache-local manifest.
    missing_media = any(
        value and not Path(str(value)).is_absolute() and not (ROOT / str(value)).is_file()
        for row in rows
        for value in (row.get("video_path"), row.get("audio_path"))
    )
    if missing_media and rows:
        benchmark = str(rows[0].get("benchmark") or "").strip().lower()
        if benchmark in {"cmm", "avhbench"}:
            layout = prepare_dataset(benchmark)
            runtime_manifest = cache_root() / "manifests" / f"{benchmark}_runtime.jsonl"
            build_dataset_manifest(layout, runtime_manifest)
            with runtime_manifest.open("r", encoding="utf-8") as handle:
                rows = [json.loads(line) for line in handle if line.strip()]
    for row in rows:
        for key in ("video_path", "audio_path"):
            value = row.get(key)
            if value and not Path(str(value)).is_absolute():
                row[key] = str((ROOT / str(value)).resolve())
    return sorted(rows, key=lambda row: int(row.get("_efficiency_order") or 0))


def append_jsonl(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(dict(payload), ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()


def logprob_pair(logits: torch.Tensor, yes_id: int, no_id: int) -> tuple[float, float]:
    scores = torch.log_softmax(logits.float(), dim=-1)
    return float(scores[int(yes_id)].item()), float(scores[int(no_id)].item())


def final_attention_payload(
    *,
    model: Any,
    input_ids: torch.Tensor,
    images: Any,
    seq_len: int,
    attention_rows: Mapping[int, torch.Tensor],
) -> dict[str, Any]:
    if images is None:
        return {
            "dominant_modality": "language",
            "modality_dominance": {"video": 0.0, "audio": 0.0, "language": 1.0},
        }
    mm_positions = torch.where(input_ids[0] < 0)[0]
    if int(mm_positions.numel()) <= 0:
        raise RuntimeError("cannot locate VideoLLaMA2 multimodal placeholder")
    mm_start = int(mm_positions[0].item())
    original_len_without_mm = int(input_ids.shape[1]) - int(mm_positions.numel())
    total_mm_len = int(seq_len) - original_len_without_mm
    if total_mm_len <= 0:
        raise RuntimeError(f"invalid multimodal length: {total_mm_len}")
    present = modality_presence(images)
    if present["video"] and present["audio"]:
        video_len = infer_stc_video_feature_len(model)
        audio_len = total_mm_len - video_len
    elif present["video"]:
        video_len, audio_len = total_mm_len, 0
    elif present["audio"]:
        video_len, audio_len = 0, total_mm_len
    else:
        video_len, audio_len = 0, 0
    if video_len < 0 or audio_len < 0:
        raise RuntimeError(f"invalid modal spans: video={video_len}, audio={audio_len}")
    video_positions = torch.arange(mm_start, mm_start + video_len, dtype=torch.long)
    audio_positions = torch.arange(mm_start + video_len, mm_start + video_len + audio_len, dtype=torch.long)
    language_positions = torch.arange(mm_start + video_len + audio_len, seq_len, dtype=torch.long)

    def mean_mass(row: torch.Tensor, positions: torch.Tensor) -> float:
        if int(positions.numel()) <= 0:
            return 0.0
        return float(row[positions].sum().item()) / float(positions.numel())

    items: list[dict[str, float]] = []
    for layer in sorted(attention_rows):
        row = attention_rows[layer].float().mean(dim=0)
        if bool(torch.isfinite(row).all().item()) and float(row.sum().item()) > 1.0e-12:
            items.append(
                {
                    "video": mean_mass(row, video_positions),
                    "audio": mean_mass(row, audio_positions),
                    "language": mean_mass(row, language_positions),
                }
            )
    if not items:
        raise RuntimeError("no usable full-view attention rows")
    masses = {key: sum(item[key] for item in items) / len(items) for key in ("video", "audio", "language")}
    return {
        "dominant_modality": max(masses, key=masses.get),
        "modality_dominance": masses,
        "captured_layers": sorted(attention_rows),
    }


@torch.inference_mode()
def diagnose_view(
    *,
    model: Any,
    tokenizer: Any,
    processor: Mapping[str, Any],
    row: Mapping[str, Any],
    branch: str,
    dtype: torch.dtype,
    yes_id: int,
    no_id: int,
    capture_attention: bool,
) -> tuple[dict[str, Any], dict[int, torch.Tensor], dict[str, Any] | None]:
    prepared = prepare_inputs(
        model=model, tokenizer=tokenizer, processor=processor, row=row, branch=branch, dtype=dtype
    )
    input_ids = prepared["input_ids"]
    original_mask = prepared["attention_mask"]
    images = prepared["images"]
    _ids, attention_mask, _past, inputs_embeds, _labels = model.prepare_inputs_labels_for_multimodal(
        input_ids=input_ids,
        attention_mask=original_mask,
        past_key_values=None,
        labels=None,
        images=images,
    )
    modules = language_layers(model)
    hidden: dict[int, torch.Tensor] = {}
    attention_rows: dict[int, torch.Tensor] = {}
    hooks: list[Any] = []

    def make_hidden_hook(layer_id: int):
        def hook_fn(_module: torch.nn.Module, _args: tuple[Any, ...], output: Any):
            values = output[0] if isinstance(output, (tuple, list)) else output
            if not isinstance(values, torch.Tensor):
                return None
            vector = values[0, -1, :].detach().float().cpu()
            if not bool(torch.isfinite(vector).all().item()):
                raise FloatingPointError(f"non-finite hidden state at layer={layer_id}, branch={branch}")
            hidden[int(layer_id)] = vector
            return None

        return hook_fn

    def make_attention_hook(layer_id: int):
        def hook_fn(module: torch.nn.Module, args: tuple[Any, ...], kwargs: Mapping[str, Any]):
            states = args[0] if args else kwargs.get("hidden_states")
            position_embeddings = kwargs.get("position_embeddings")
            if position_embeddings is None and len(args) >= 2:
                position_embeddings = args[1]
            mask = kwargs.get("attention_mask")
            if mask is None and len(args) >= 3:
                mask = args[2]
            if states is not None and position_embeddings is not None:
                attention_rows[int(layer_id)] = _qwen2_last_query_attention_row(
                    module, hidden_states=states, attention_mask=mask, position_embeddings=position_embeddings
                )
            return None

        return hook_fn

    try:
        for layer in FROZEN_ALL_LAYERS:
            hooks.append(modules[int(layer)].register_forward_hook(make_hidden_hook(int(layer))))
        if capture_attention:
            for layer_id, module in enumerate(modules):
                hooks.append(module.self_attn.register_forward_pre_hook(make_attention_hook(layer_id), with_kwargs=True))
        if inputs_embeds is None:
            out = model(
                input_ids=input_ids,
                attention_mask=original_mask,
                images=images,
                use_cache=False,
                return_dict=True,
                pad_token_id=tokenizer.eos_token_id,
            )
        else:
            out = model(
                input_ids=None,
                attention_mask=attention_mask,
                inputs_embeds=inputs_embeds,
                use_cache=False,
                return_dict=True,
                pad_token_id=tokenizer.eos_token_id,
            )
        yes_score, no_score = logprob_pair(out.logits[0, -1, :], yes_id, no_id)
        attention_payload = (
            final_attention_payload(
                model=model,
                input_ids=input_ids,
                images=images,
                seq_len=int(out.logits.shape[1]),
                attention_rows=attention_rows,
            )
            if capture_attention
            else None
        )
    finally:
        for hook in hooks:
            hook.remove()
    missing = sorted(set(FROZEN_ALL_LAYERS) - set(hidden))
    if missing:
        raise RuntimeError(f"missing hidden captures for branch={branch}: {missing}")
    answer = "Yes" if yes_score >= no_score else "No"
    score = {
        "answer": answer,
        "yes_score": yes_score,
        "no_score": no_score,
        "margin": yes_score - no_score,
    }
    return score, hidden, attention_payload


def choose_target(*, row: Mapping[str, Any], scores: Mapping[str, Mapping[str, Any]], attention: Mapping[str, Any]) -> str:
    question_target, _ = online_target_modality_from_question(safe_text(row.get("question")))
    if question_target in {"audio", "visual"}:
        return question_target
    full = normalize_yes_no(scores["full"].get("answer"))
    audio = normalize_yes_no(scores["audio_only"].get("answer"))
    visual = normalize_yes_no(scores["visual_only"].get("answer"))
    if full == audio and full != visual:
        return "visual"
    if full == visual and full != audio:
        return "audio"
    masses = dict(attention.get("modality_dominance") or {})
    return "audio" if float(masses.get("audio", 0.0)) <= float(masses.get("video", 0.0)) else "visual"


def carriers_from_cache(
    *,
    model: Any,
    tokenizer: Any,
    scores: Mapping[str, Mapping[str, Any]],
    hidden: Mapping[str, Mapping[int, torch.Tensor]],
    target: str,
) -> tuple[list[dict[str, Any]], dict[int, dict[str, torch.Tensor]], dict[str, Any], str, str]:
    target_key = "audio_only" if target == "audio" else "visual_only"
    non_target_key = "visual_only" if target == "audio" else "audio_only"
    captures = {
        "full": {"layer_vectors": hidden["full"], "final_prediction": scores["full"]["answer"], "final_yes_no_margin": scores["full"]["margin"]},
        "target": {"layer_vectors": hidden[target_key], "final_prediction": scores[target_key]["answer"], "final_yes_no_margin": scores[target_key]["margin"]},
        "non_target": {"layer_vectors": hidden[non_target_key], "final_prediction": scores[non_target_key]["answer"], "final_yes_no_margin": scores[non_target_key]["margin"]},
        "text": {"layer_vectors": hidden["text_only"], "final_prediction": scores["text_only"]["answer"], "final_yes_no_margin": scores["text_only"]["margin"]},
    }
    current = normalize_yes_no(scores["full"]["answer"])
    candidate = normalize_yes_no(scores[target_key]["answer"])
    if current is None or candidate is None:
        raise RuntimeError("invalid binary branch prediction")
    base_budget = build_base_conflict_budget(
        branch_scores=scores,
        current_answer=current,
        candidate_answer=candidate,
        target_branch="audio" if target == "audio" else "visual",
        non_target_branch="visual" if target == "audio" else "audio",
    )
    lm_head = model.get_output_embeddings() if hasattr(model, "get_output_embeddings") else getattr(model, "lm_head")
    rows, tensors, budget = build_frozen_carriers(
        captures=captures,
        lm_head=lm_head,
        yes_id=int(candidate_token_ids(tokenizer, "Yes")[0]),
        no_id=int(candidate_token_ids(tokenizer, "No")[0]),
        current_answer=current,
        candidate_answer=candidate,
        layers=FROZEN_ALL_LAYERS,
        base_conflict_budget=base_budget,
    )
    return rows, tensors, budget, current, candidate


def run_row(*, model: Any, tokenizer: Any, processor: Mapping[str, Any], row: Mapping[str, Any], dtype: torch.dtype) -> dict[str, Any]:
    yes_tokens = candidate_token_ids(tokenizer, "Yes")
    no_tokens = candidate_token_ids(tokenizer, "No")
    if len(yes_tokens) != 1 or len(no_tokens) != 1:
        raise RuntimeError("five-pass protocol requires one-token Yes/No candidates")
    scores: dict[str, dict[str, Any]] = {}
    hidden: dict[str, dict[int, torch.Tensor]] = {}
    full_score, full_hidden, attention = diagnose_view(
        model=model, tokenizer=tokenizer, processor=processor, row=row, branch="full", dtype=dtype,
        yes_id=yes_tokens[0], no_id=no_tokens[0], capture_attention=True,
    )
    scores["full"], hidden["full"] = full_score, full_hidden
    for key, branch in (("audio_only", "audio"), ("visual_only", "visual"), ("text_only", "text")):
        score, vectors, _ = diagnose_view(
            model=model, tokenizer=tokenizer, processor=processor, row=row, branch=branch, dtype=dtype,
            yes_id=yes_tokens[0], no_id=no_tokens[0], capture_attention=False,
        )
        scores[key], hidden[key] = score, vectors
    if attention is None:
        raise RuntimeError("missing full-view attention")
    target = choose_target(row=row, scores=scores, attention=attention)
    carrier_rows, carrier_tensors, budget, current, candidate = carriers_from_cache(
        model=model, tokenizer=tokenizer, scores=scores, hidden=hidden, target=target
    )
    carrier_map = {int(item["layer"]): item for item in carrier_rows}
    manifest = {
        "current_answer_y0": current,
        "candidate_answer_y": candidate,
        "target_answer": candidate,
        "target_modality": target,
        "oelpr_conflict_budget": budget,
    }
    beta, alpha = (1.7, 0.25) if row.get("benchmark") == "cmm" else (0.95, 0.10)
    edited = executor.frozen_structured_edit_yesno(
        model=model,
        tokenizer=tokenizer,
        processor=processor,
        row=row,
        manifest=manifest,
        carrier_tensors=carrier_tensors,
        carrier_rows=carrier_map,
        dtype=dtype,
        beta=beta,
        alpha=alpha,
        prepare_retries=0,
        decision_mode="constrained_yesno",
        max_new_tokens=1,
        evidence_prior_subspace_purification_lambda=0.25,
        hidden_update_mode="apply",
        token_head_update_mode="apply",
        token_head_value_operator="allpath_token_scaling",
        prior_suppression_operator="path_residual_minimal",
        evidence_transfer_mode="path_answer_margin_minimal",
    )
    return {
        "target_modality": target,
        "baseline_prediction": current,
        "intervened_prediction": edited["prediction"],
        "attention_dominant_modality": attention["dominant_modality"],
        "conflict_budget": float(budget.get("budget") or 0.0),
    }


def main() -> None:
    args = parse_args()
    dtype = torch_dtype_from_name(args.dtype)
    rows = read_jsonl(args.input)
    if args.max_rows > 0:
        rows = rows[: int(args.max_rows)]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("", encoding="utf-8")
    model, processor, tokenizer = model_init(
        args.model_path, device_map=torch.device("cuda"), use_flash_attn=True, torch_dtype=dtype
    )
    model.eval()
    forward_counter = [0]

    def count_forward(_module: Any, _inputs: Any) -> None:
        forward_counter[0] += 1

    handle = model.register_forward_pre_hook(count_forward)
    try:
        for row in rows:
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            start = time.perf_counter()
            counter_start = int(forward_counter[0])
            error = ""
            details: dict[str, Any] = {}
            try:
                details = run_row(model=model, tokenizer=tokenizer, processor=processor, row=row, dtype=dtype)
            except Exception as exc:
                error = repr(exc)
            torch.cuda.synchronize()
            calls = int(forward_counter[0]) - counter_start
            if not error and calls != 5:
                error = f"expected exactly 5 model forwards, observed {calls}"
            append_jsonl(
                args.output,
                {
                    "sample_id": row.get("sample_id"),
                    "method": "owp_5pass_cached",
                    "repeat": int(args.repeat),
                    "benchmark": row.get("benchmark"),
                    "cohort_role": row.get("_efficiency_cohort_role"),
                    "runtime_seconds": float(time.perf_counter() - start),
                    "forward_calls": calls,
                    "peak_memory_allocated_bytes": int(torch.cuda.max_memory_allocated()),
                    "peak_memory_reserved_bytes": int(torch.cuda.max_memory_reserved()),
                    "error": error,
                    **details,
                },
            )
            if error:
                raise RuntimeError(f"sample={row.get('sample_id')}: {error}")
    finally:
        handle.remove()


if __name__ == "__main__":
    main()
