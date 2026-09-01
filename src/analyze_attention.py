#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import json
import math
import os
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import torch
from tqdm.auto import tqdm
from transformers.models.qwen2_5_omni.modeling_qwen2_5_omni import (
    apply_multimodal_rotary_pos_emb,
    repeat_kv,
)


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
DEFAULT_OUTPUT_DIR = ROOT / "results" / "qwen_owp_attention"
DEFAULT_MODEL_PATH = "Qwen/Qwen2.5-Omni-7B"

DEFAULT_LAYERS = (0, 4, 8, 12, 16, 18, 20, 22, 24, 26, 27)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Full-run token attention/provenance probe. This does not split branches; "
            "it observes answer-token attention to audio/video/text tokens inside the full input."
        )
    )
    parser.add_argument("--samples", type=Path, default=DEFAULT_SAMPLES)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--model-path", type=str, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--seed", type=int, default=20260427)
    parser.add_argument("--per-bucket", type=int, default=2)
    parser.add_argument("--max-total", type=int, default=0)
    parser.add_argument("--layers", type=int, nargs="+", default=list(DEFAULT_LAYERS))
    parser.add_argument(
        "--max-attention-tokens",
        type=int,
        default=4096,
        help=(
            "Deprecated safety knob. The script now computes only the final-query "
            "attention row with hooks, so it does not materialize seq_len^2 attentions."
        ),
    )
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


def select_rows(args: argparse.Namespace, rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rng = random.Random(int(args.seed))
    by_bucket: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        key = safe_text(row.get("pattern_bucket"))
        if key:
            by_bucket[key].append(row)

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


def prepare_full_inputs(
    adapter: QwenOmniAdapter,
    *,
    row: Dict[str, Any],
    budget: Dict[str, float],
) -> Dict[str, torch.Tensor]:
    video_path = safe_text(row.get("video_path")) or None
    audio_path = safe_text(row.get("audio_path")) or None
    audio_array = adapter._resolve_audio_array(
        video_path=video_path,
        audio_path=audio_path,
        mask_audio=False,
    )
    messages = adapter._build_messages(
        video_path,
        formatted_question(row),
        audio_array=audio_array,
        mask_visual=False,
    )
    return adapter._messages_to_inputs(messages, audio_array=audio_array, video_budget=budget)


def tokenizer_ids(adapter: QwenOmniAdapter) -> Dict[str, Optional[int]]:
    tok = adapter._processor.tokenizer
    names = {
        "audio": "<|AUDIO|>",
        "audio_bos": "<|audio_bos|>",
        "audio_eos": "<|audio_eos|>",
        "vision_bos": "<|vision_bos|>",
        "vision_eos": "<|vision_eos|>",
        "vision_pad": "<|vision_pad|>",
        "image": "<|IMAGE|>",
        "video": "<|VIDEO|>",
        "im_start": "<|im_start|>",
        "im_end": "<|im_end|>",
    }
    return {name: tok.convert_tokens_to_ids(token) for name, token in names.items()}


def mark_spans(ids: torch.Tensor, bos_id: Optional[int], eos_id: Optional[int]) -> torch.Tensor:
    mask = torch.zeros(ids.shape[0], dtype=torch.bool)
    if bos_id is None or eos_id is None:
        return mask
    inside = False
    for idx, token_id in enumerate(ids.tolist()):
        if token_id == bos_id:
            inside = True
            mask[idx] = True
            continue
        if inside:
            mask[idx] = True
        if token_id == eos_id:
            inside = False
    return mask


def build_token_group_masks(
    *,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    special_ids: Dict[str, Optional[int]],
) -> Dict[str, torch.Tensor]:
    ids = input_ids[0].detach().cpu()
    valid = attention_mask[0].detach().cpu().bool()
    seq_len = int(ids.shape[0])
    query = torch.zeros(seq_len, dtype=torch.bool)
    query[max(0, seq_len - 1)] = True

    audio_span = mark_spans(ids, special_ids.get("audio_bos"), special_ids.get("audio_eos"))
    vision_span = mark_spans(ids, special_ids.get("vision_bos"), special_ids.get("vision_eos"))

    audio_ids = [special_ids.get("audio"), special_ids.get("audio_bos"), special_ids.get("audio_eos")]
    vision_ids = [
        special_ids.get("vision_bos"),
        special_ids.get("vision_eos"),
        special_ids.get("vision_pad"),
        special_ids.get("image"),
        special_ids.get("video"),
    ]
    audio_id_mask = torch.zeros(seq_len, dtype=torch.bool)
    vision_id_mask = torch.zeros(seq_len, dtype=torch.bool)
    for token_id in audio_ids:
        if token_id is not None:
            audio_id_mask |= ids == int(token_id)
    for token_id in vision_ids:
        if token_id is not None:
            vision_id_mask |= ids == int(token_id)

    audio = valid & (audio_span | audio_id_mask)
    vision = valid & (vision_span | vision_id_mask)
    media = audio | vision
    special = audio | vision
    text = valid & ~special
    text &= ~query
    audio &= ~query
    vision &= ~query
    media &= ~query

    question_text = torch.zeros(seq_len, dtype=torch.bool)
    im_start = special_ids.get("im_start")
    im_end = special_ids.get("im_end")
    assistant_start_idx: Optional[int] = None
    user_end_idx: Optional[int] = None
    if im_start is not None:
        starts = torch.nonzero(ids == int(im_start), as_tuple=False).flatten()
        if starts.numel() > 0:
            assistant_start_idx = int(starts[-1].item())
    if im_end is not None and assistant_start_idx is not None:
        ends = torch.nonzero((ids == int(im_end)) & (torch.arange(seq_len) < assistant_start_idx), as_tuple=False).flatten()
        if ends.numel() > 0:
            user_end_idx = int(ends[-1].item())
    media_indices = torch.nonzero(media, as_tuple=False).flatten()
    if user_end_idx is not None:
        start_idx = int(media_indices[-1].item()) + 1 if media_indices.numel() > 0 else 0
        if start_idx < user_end_idx:
            question_text[start_idx:user_end_idx] = True
    question_text &= text
    structural_text = text & ~question_text

    return {
        "valid": valid,
        "query": query,
        "audio": audio,
        "vision": vision,
        "media": media,
        "text": text,
        "question_text": question_text,
        "structural_text": structural_text,
    }


def group_counts(masks: Dict[str, torch.Tensor]) -> Dict[str, int]:
    return {f"{key}_token_count": int(value.sum().item()) for key, value in masks.items()}


def top_token_summary(
    adapter: QwenOmniAdapter,
    *,
    ids: torch.Tensor,
    weights: torch.Tensor,
    k: int = 8,
) -> List[Dict[str, Any]]:
    tok = adapter._processor.tokenizer
    values, indices = torch.topk(weights.detach().cpu().float(), k=min(k, int(weights.numel())))
    out: List[Dict[str, Any]] = []
    for value, index in zip(values.tolist(), indices.tolist()):
        token_id = int(ids[int(index)].item())
        try:
            surface = tok.decode([token_id], skip_special_tokens=False)
        except Exception:
            surface = str(token_id)
        out.append(
            {
                "index": int(index),
                "token_id": token_id,
                "token": surface,
                "attention": float(value),
            }
        )
    return out


def entropy(weights: torch.Tensor) -> float:
    w = weights.detach().float().cpu()
    w = w / max(float(w.sum().item()), 1e-12)
    w = w.clamp_min(1e-12)
    return float(-(w * w.log()).sum().item())


def compute_final_query_head_attention_rows(
    module: torch.nn.Module,
    hidden_states: torch.Tensor,
    *,
    attention_mask: Optional[torch.Tensor],
    position_embeddings: Any,
) -> torch.Tensor:
    bsz, q_len, _ = hidden_states.size()
    query_states = module.q_proj(hidden_states)
    key_states = module.k_proj(hidden_states)

    query_states = query_states.view(bsz, q_len, -1, module.head_dim).transpose(1, 2)
    key_states = key_states.view(bsz, q_len, -1, module.head_dim).transpose(1, 2)

    cos, sin = position_embeddings
    query_states, key_states = apply_multimodal_rotary_pos_emb(
        query_states,
        key_states,
        cos,
        sin,
        module.rope_scaling["mrope_section"],
    )
    key_states = repeat_kv(key_states, module.num_key_value_groups)
    query_final = query_states[:, :, -1:, :]
    attn_weights = torch.matmul(query_final, key_states.transpose(2, 3)) * module.scaling
    if attention_mask is not None:
        attn_weights = attn_weights + attention_mask[:, :, -1:, : key_states.shape[-2]]
    attn_weights = torch.nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
    return attn_weights[0, :, 0, :].detach().float().cpu()


def compute_final_query_attention_row(
    module: torch.nn.Module,
    hidden_states: torch.Tensor,
    *,
    attention_mask: Optional[torch.Tensor],
    position_embeddings: Any,
) -> torch.Tensor:
    return compute_final_query_head_attention_rows(
        module,
        hidden_states,
        attention_mask=attention_mask,
        position_embeddings=position_embeddings,
    ).mean(dim=0)


@torch.no_grad()
def run_full_probe(
    adapter: QwenOmniAdapter,
    *,
    row: Dict[str, Any],
    layers: Sequence[int],
    yes_id: int,
    no_id: int,
    budget: Dict[str, float],
    max_attention_tokens: int,
    special_ids: Dict[str, Optional[int]],
) -> Dict[str, Any]:
    inputs = prepare_full_inputs(adapter, row=row, budget=budget)
    input_ids = inputs["input_ids"]
    attention_mask = inputs.get("attention_mask")
    if attention_mask is None:
        attention_mask = torch.ones_like(input_ids)
    seq_len = int(input_ids.shape[-1])
    masks = build_token_group_masks(
        input_ids=input_ids,
        attention_mask=attention_mask,
        special_ids=special_ids,
    )

    thinker = adapter._model.thinker
    layer_records: Dict[str, Any] = {}
    ids_cpu = input_ids[0].detach().cpu()
    selected_layers = sorted({int(layer) for layer in layers})
    hooks = []

    def make_pre_hook(layer_id: int):
        def hook_fn(module, args, kwargs):
            hidden_states = args[0] if args else kwargs.get("hidden_states")
            position_embeddings = kwargs.get("position_embeddings")
            if hidden_states is None or position_embeddings is None:
                return None
            attention_mask_arg = kwargs.get("attention_mask")
            try:
                final_attn = compute_final_query_attention_row(
                    module,
                    hidden_states,
                    attention_mask=attention_mask_arg,
                    position_embeddings=position_embeddings,
                )
            except RuntimeError as exc:
                layer_records[str(layer_id)] = {
                    "available": False,
                    "error": f"{type(exc).__name__}: {exc}",
                }
                return None
            total = max(float(final_attn.sum().item()), 1e-12)
            rec: Dict[str, Any] = {
                "available": True,
                "attention_entropy": entropy(final_attn),
                "top_tokens": top_token_summary(adapter, ids=ids_cpu, weights=final_attn, k=8),
            }
            for group in ("audio", "vision", "media", "text", "question_text", "structural_text"):
                group_mask = masks[group]
                count = int(group_mask.sum().item())
                mass = float(final_attn[group_mask].sum().item()) / total if count else 0.0
                rec[f"{group}_attention_mass"] = mass
                rec[f"{group}_attention_per_token"] = mass / count if count else None
            layer_records[str(layer_id)] = rec
            return None

        return hook_fn

    for layer in selected_layers:
        if 0 <= layer < len(thinker.model.layers):
            hooks.append(
                thinker.model.layers[int(layer)].self_attn.register_forward_pre_hook(
                    make_pre_hook(int(layer)),
                    with_kwargs=True,
                )
            )

    try:
        outputs = thinker(
            **inputs,
            use_cache=False,
            output_attentions=False,
            return_dict=True,
        )
    finally:
        for hook in hooks:
            hook.remove()

    logits = outputs.logits[:, -1, [yes_id, no_id]].detach().float().cpu()[0]
    margin = float(logits[0].item() - logits[1].item())

    del inputs, outputs
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return {
        "seq_len": seq_len,
        "attention_available": bool(layer_records),
        "token_counts": group_counts(masks),
        "final_yes_no_margin": margin,
        "final_prediction": pred_from_margin(margin),
        "layers": layer_records,
    }


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


def auc_correct_greater(correct: Sequence[float], wrong: Sequence[float]) -> Optional[float]:
    if not correct or not wrong:
        return None
    total = 0
    score = 0.0
    for p in correct:
        for n in wrong:
            total += 1
            if p > n:
                score += 1.0
            elif p == n:
                score += 0.5
    return score / total if total else None


def numeric_values(rows: Sequence[Dict[str, Any]], key_path: Sequence[str]) -> List[float]:
    out: List[float] = []
    for row in rows:
        value: Any = row
        for key in key_path:
            if not isinstance(value, dict) or key not in value:
                value = None
                break
            value = value[key]
        if value is None:
            continue
        try:
            number = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(number):
            out.append(number)
    return out


def summarize_metric(rows: Sequence[Dict[str, Any]], key_path: Sequence[str]) -> Dict[str, Any]:
    wrong_rows = [row for row in rows if not bool(row.get("stored_base_correct"))]
    correct_rows = [row for row in rows if bool(row.get("stored_base_correct"))]
    wrong = numeric_values(wrong_rows, key_path)
    correct = numeric_values(correct_rows, key_path)
    return {
        "metric": ".".join(key_path),
        "wrong_n": len(wrong),
        "correct_n": len(correct),
        "wrong_mean": mean(wrong),
        "correct_mean": mean(correct),
        "gap_correct_minus_wrong": (
            float(mean(correct)) - float(mean(wrong)) if wrong and correct else None
        ),
        "auc_correct_greater": auc_correct_greater(correct, wrong),
    }


def summarize(records: Sequence[Dict[str, Any]], layers: Sequence[int]) -> Dict[str, Any]:
    metrics: List[List[str]] = [
        ["probe", "seq_len"],
        ["probe", "token_counts", "audio_token_count"],
        ["probe", "token_counts", "vision_token_count"],
        ["probe", "token_counts", "media_token_count"],
        ["probe", "token_counts", "text_token_count"],
    ]
    for layer in layers:
        layer_key = str(int(layer))
        for group in ("audio", "vision", "media", "text", "question_text", "structural_text"):
            metrics.append(["probe", "layers", layer_key, f"{group}_attention_mass"])
            metrics.append(["probe", "layers", layer_key, f"{group}_attention_per_token"])
        metrics.append(["probe", "layers", layer_key, "attention_entropy"])

    return {
        "n_records": len(records),
        "record_buckets": dict(Counter(row.get("pattern_bucket") for row in records)),
        "attention_available": sum(1 for row in records if (row.get("probe") or {}).get("attention_available")),
        "attention_skipped": sum(1 for row in records if not (row.get("probe") or {}).get("attention_available")),
        "metrics": [summarize_metric(records, metric) for metric in metrics],
    }


def run() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = read_jsonl(args.samples)
    selected = select_rows(args, rows)
    layers = sorted({int(layer) for layer in args.layers})
    plan_rows = [
        {
            "sample_id": row.get("sample_id"),
            "pattern_bucket": row.get("pattern_bucket"),
            "task_family": row.get("task_family"),
            "reference_answer": row.get("reference_answer"),
            "stored_base_prediction": row.get("stored_base_prediction"),
            "stored_base_correct": row.get("stored_base_correct"),
            "video_path": row.get("video_path"),
            "audio_path": row.get("audio_path"),
        }
        for row in selected
    ]
    write_jsonl(args.output_dir / "selected_samples.jsonl", plan_rows)
    write_json(
        args.output_dir / "plan_summary.json",
        {
            "samples": str(args.samples),
            "output_dir": str(args.output_dir),
            "selected": len(selected),
            "selected_by_bucket": dict(Counter(row.get("pattern_bucket") for row in selected)),
            "layers": layers,
            "max_attention_tokens": int(args.max_attention_tokens),
            "video_budget": video_budget(args),
            "note": "Attention is recorded only when seq_len <= max_attention_tokens.",
        },
    )
    if args.plan_only:
        print(f"[full-token-attn] plan-only selected={len(selected)} output_dir={args.output_dir}")
        return

    adapter = QwenOmniAdapter(
        model_path=args.model_path,
        device=args.device,
        max_new_tokens=4,
    )
    yes_id, no_id = yes_no_token_ids(adapter)
    special_ids = tokenizer_ids(adapter)
    budget = video_budget(args)

    records: List[Dict[str, Any]] = []
    iterator = tqdm(
        selected,
        total=len(selected),
        desc="full_token_attention",
        unit="sample",
        disable=bool(args.no_progress),
        dynamic_ncols=True,
    )
    try:
        for row in iterator:
            probe = run_full_probe(
                adapter,
                row=row,
                layers=layers,
                yes_id=yes_id,
                no_id=no_id,
                budget=budget,
                max_attention_tokens=int(args.max_attention_tokens),
                special_ids=special_ids,
            )
            record = {
                "sample_id": row.get("sample_id"),
                "benchmark": row.get("benchmark"),
                "pattern_bucket": row.get("pattern_bucket"),
                "task_family": row.get("task_family"),
                "question": row.get("question"),
                "reference_answer": norm_yes_no(row.get("reference_answer")),
                "stored_base_prediction": norm_yes_no(row.get("stored_base_prediction")),
                "stored_base_correct": bool(row.get("stored_base_correct")),
                "probe": probe,
            }
            records.append(record)
            iterator.set_postfix(
                bucket=safe_text(row.get("pattern_bucket"))[:18],
                seq=probe.get("seq_len"),
                attn=int(bool(probe.get("attention_available"))),
            )
    finally:
        del adapter
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    write_jsonl(args.output_dir / "attention_records.jsonl", records)
    payload = {
        "samples": str(args.samples),
        "output_dir": str(args.output_dir),
        "selected": len(selected),
        "selected_by_bucket": dict(Counter(row.get("pattern_bucket") for row in selected)),
        "layers": layers,
        "max_attention_tokens": int(args.max_attention_tokens),
        "video_budget": budget,
        "special_token_ids": special_ids,
        "summary": summarize(records, layers),
    }
    write_json(args.output_dir / "summary.json", payload)
    print(f"[full-token-attn] wrote {args.output_dir / 'attention_records.jsonl'}")
    print(f"[full-token-attn] wrote {args.output_dir / 'summary.json'}")


if __name__ == "__main__":
    run()
