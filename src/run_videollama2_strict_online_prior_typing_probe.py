#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import json
import math
import sys
import time
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterable, Mapping, Sequence

import torch
import transformers
from transformers.models.qwen2.modeling_qwen2 import apply_rotary_pos_emb, repeat_kv
from tqdm.auto import tqdm


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
for path in (SCRIPT_DIR, ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

VIDEOLLAMA2_ROOT = ROOT / "third_party" / "VideoLLaMA2"
if str(VIDEOLLAMA2_ROOT) not in sys.path:
    sys.path.insert(0, str(VIDEOLLAMA2_ROOT))

from videollama2 import model_init  # noqa: E402
from videollama2.constants import DEFAULT_VIDEO_TOKEN  # noqa: E402
from videollama2.mm_utils import tokenizer_multimodal_token  # noqa: E402

from run_strict_online_prior_typing_probe import (  # noqa: E402
    attribute_directed_prior,
    branch_answer,
    branch_support_for_answer,
    classify_target_evidence_state,
    infer_repair_target_online,
    infer_target_modality_online,
)
from run_prior_carrier_suppression_executor import (  # noqa: E402
    candidate_reliability_from_carrier_rows,
    candidate_reliable_conflict_budget,
    orthogonal_prior_direction,
    soft_conflict_budget_from_row,
)
from run_videollama2_lccs_cross_model import (  # noqa: E402
    normalize_yes_no,
    safe_text,
    torch_dtype_from_name,
)
from run_videollama2_lccs_strict_mad_path import (  # noqa: E402
    answer_correct_strict,
    build_official_prompt,
    candidate_token_ids,
    capture_hidden,
    generate_answer,
    language_layers,
    official_video_id,
    prepare_inputs,
    prepare_media,
    score_candidate_sequence,
    select_cmm_unary_full_rows,
    target_branch_for as row_target_branch_for,
    tensor_to_official_cuda,
    write_json,
)


MODALITY_QUERY_PROMPT = "To answer this question, which modality is needed (audio, video, or both): "
ANSWER_QUERY_PROMPT = " Answer only 'Yes' or 'No'. Do not include any explanation."

FROZEN_SELF_LOGIT_LAYERS = [12, 16, 20]
FROZEN_EVIDENCE_LAYERS = [12, 16, 20]
FROZEN_PRIOR_LAYERS = [24, 26, 27]
FROZEN_ALL_LAYERS = sorted(set(FROZEN_SELF_LOGIT_LAYERS + FROZEN_EVIDENCE_LAYERS + FROZEN_PRIOR_LAYERS))
FROZEN_OELPR_CONFLICT_BUDGET_TAU = 1.0
FROZEN_OELPR_CONFLICT_BUDGET_STRONG_FLOOR = 0.75
FROZEN_OELPR_CONFLICT_BUDGET_WEAK_FLOOR = 0.35


def append_jsonl(path: Path, row: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(dict(row), ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), ensure_ascii=False, sort_keys=True) + "\n")


def finite_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float(default)
    if not math.isfinite(out):
        return float(default)
    return float(out)


def release_cuda_cache() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def probability_pair_from_log_scores(yes_score: float, no_score: float) -> tuple[float, float]:
    scores = torch.tensor([float(yes_score), float(no_score)], dtype=torch.float32)
    probs = torch.softmax(scores, dim=0).tolist()
    return float(probs[0]), float(probs[1])


def yes_no_token_id(tokenizer: Any, text: str) -> int:
    ids = candidate_token_ids(tokenizer, text)
    if not ids:
        raise ValueError(f"empty tokenization for {text!r}")
    return int(ids[0])


def target_branch_for_modality(modality: str) -> str | None:
    value = safe_text(modality).lower()
    if value == "audio":
        return "audio"
    if value in {"visual", "video", "vision"}:
        return "visual"
    return None


def non_target_branch_for_modality(modality: str) -> str | None:
    branch = target_branch_for_modality(modality)
    if branch == "audio":
        return "visual"
    if branch == "visual":
        return "audio"
    return None


def typing_branch_key(videollama_branch: str) -> str:
    if videollama_branch == "full":
        return "full"
    if videollama_branch == "audio":
        return "audio_only"
    if videollama_branch == "visual":
        return "visual_only"
    if videollama_branch == "text":
        return "text_only"
    raise ValueError(f"unknown branch: {videollama_branch}")


@torch.inference_mode()
def score_yesno_branch(
    *,
    model: Any,
    tokenizer: Any,
    processor: Mapping[str, Any],
    row: Mapping[str, Any],
    branch: str,
    dtype: torch.dtype,
) -> dict[str, Any]:
    raw, debug = generate_answer(
        model=model,
        tokenizer=tokenizer,
        processor=processor,
        row=row,
        branch=branch,
        max_new_tokens=1,
        decision_mode="constrained_yesno",
        dtype=dtype,
    )
    yes_score = finite_float(debug.get("yes_score"))
    no_score = finite_float(debug.get("no_score"))
    p_yes, p_no = probability_pair_from_log_scores(yes_score, no_score)
    answer = normalize_yes_no(raw)
    return {
        "branch": typing_branch_key(branch),
        "videollama2_branch": branch,
        "answer": answer,
        "raw_output": raw,
        "confidence_on_yes": p_yes,
        "confidence_on_no": p_no,
        "yes_score": yes_score,
        "no_score": no_score,
        "margin": yes_score - no_score,
    }


@torch.inference_mode()
def score_modality_probe(
    *,
    model: Any,
    tokenizer: Any,
    processor: Mapping[str, Any],
    row: Mapping[str, Any],
    dtype: torch.dtype,
) -> dict[str, Any]:
    question = safe_text(row.get("question"))
    if "answer" not in question.lower()[-60:]:
        question = question + ANSWER_QUERY_PROMPT
    head_question = "Question: " + question + "\n" + MODALITY_QUERY_PROMPT
    media, modal, modal_token = prepare_media(processor, row, "full")
    prompt = build_official_prompt(tokenizer, model, head_question, modal_token or DEFAULT_VIDEO_TOKEN)
    input_ids = tokenizer_multimodal_token(
        prompt,
        tokenizer,
        modal_token or DEFAULT_VIDEO_TOKEN,
        return_tensors="pt",
    ).unsqueeze(0).long().cuda()
    attention_mask = input_ids.ne(tokenizer.pad_token_id).long().cuda()
    images = tensor_to_official_cuda(media, modal, dtype=dtype)
    out = model(
        input_ids,
        attention_mask=attention_mask,
        images=images,
        use_cache=False,
        return_dict=True,
        pad_token_id=tokenizer.eos_token_id,
    )
    logits = out.logits[0, -1, :].detach().float().cpu()
    audio_id = int(tokenizer.encode("audio", add_special_tokens=False)[0])
    video_id = int(tokenizer.encode("video", add_special_tokens=False)[0])
    both_id = int(tokenizer.encode("both", add_special_tokens=False)[0])
    audio_logit = float(logits[audio_id].item())
    video_logit = float(logits[video_id].item())
    both_logit = float(logits[both_id].item())
    probs = torch.softmax(torch.tensor([audio_logit, video_logit, both_logit]), dim=0).tolist()
    dist = {"audio": float(probs[0]), "video": float(probs[1]), "both": float(probs[2])}
    return {
        "mode": "videollama2_head_only_modality_probe",
        "head_query_prompt": MODALITY_QUERY_PROMPT,
        "head_modality_logits": {
            "audio": audio_logit,
            "video": video_logit,
            "both": both_logit,
        },
        "modality_distribution": dist,
        "predicted_modality": max(dist.items(), key=lambda item: item[1])[0],
    }


@torch.inference_mode()
def _video_feature_len(model: Any, video_tensor: torch.Tensor) -> int:
    features = model.encode_images_or_videos([(video_tensor, "video")])
    return int(features.shape[1])


@torch.inference_mode()
def _audio_feature_len(model: Any, audio_tensor: torch.Tensor) -> int:
    device = next(model.parameters()).device
    audio_padding_mask = torch.zeros(audio_tensor.shape, device=device).bool()
    audio_embedding, _t, _f = model.get_model().get_audio_tower().extract_features(
        audio_tensor,
        padding_mask=audio_padding_mask,
        feature_only=True,
    )
    features = model.get_model().mm_projector_a(audio_embedding)
    return int(features.view(1, -1, features.shape[-1]).shape[1])


@torch.inference_mode()
def modal_feature_lengths(model: Any, images: Any) -> dict[str, int]:
    if images is None:
        return {"video": 0, "audio": 0}
    if not isinstance(images, list) or not images:
        return {"video": 0, "audio": 0}
    media, modal = images[0]
    if isinstance(media, dict):
        return {
            "video": _video_feature_len(model, media["video"]),
            "audio": _audio_feature_len(model, media["audio"]),
        }
    if modal == "video":
        return {"video": _video_feature_len(model, media), "audio": 0}
    if modal == "audio":
        return {"video": 0, "audio": _audio_feature_len(model, media)}
    return {"video": 0, "audio": 0}


def infer_stc_video_feature_len(model: Any) -> int:
    vision_tower = model.get_vision_tower() if hasattr(model, "get_vision_tower") else model.get_model().get_vision_tower()
    projector_type = safe_text(getattr(model.config, "mm_projector_type", "")).lower()
    num_frames = int(getattr(model.config, "num_frames", 0) or 0)
    patches_per_side = int(getattr(vision_tower, "num_patches_per_side", 0) or 0)
    num_patches = int(getattr(vision_tower, "num_patches", 0) or 0)
    if "stc_connector" in projector_type:
        if num_frames <= 0 or patches_per_side <= 0:
            raise RuntimeError("cannot infer STC video feature length from VideoLLaMA2 config")
        return int((num_frames // 2) * (patches_per_side // 2) * (patches_per_side // 2))
    if num_patches <= 0:
        raise RuntimeError("cannot infer video feature length from VideoLLaMA2 vision tower")
    return int(num_patches)


def modality_presence(images: Any) -> dict[str, bool]:
    has_video = False
    has_audio = False
    if not isinstance(images, list):
        return {"video": False, "audio": False}
    for media, modal in images:
        if isinstance(media, dict):
            has_video = has_video or "video" in media
            has_audio = has_audio or "audio" in media
            continue
        has_video = has_video or modal in {"video", "image"}
        has_audio = has_audio or modal == "audio"
    return {"video": has_video, "audio": has_audio}


def _qwen2_last_query_attention_row(
    module: torch.nn.Module,
    *,
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor | None,
    position_embeddings: Any,
) -> torch.Tensor:
    input_shape = hidden_states.shape[:-1]
    hidden_shape = (*input_shape, -1, int(module.head_dim))
    query_states = module.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
    key_states = module.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
    cos, sin = position_embeddings
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
    query_states = query_states.to(torch.float32)
    key_states = key_states.to(torch.float32)
    key_states = repeat_kv(key_states, int(module.num_key_value_groups))
    attn_row = torch.matmul(query_states[:, :, -1:, :], key_states.transpose(2, 3)) * float(module.scaling)
    if attention_mask is not None:
        attn_row = attn_row + attention_mask[:, :, -1:, : attn_row.shape[-1]]
    attn_row = torch.nn.functional.softmax(attn_row, dim=-1, dtype=torch.float32)
    return attn_row[0, :, 0, :].detach().float().cpu()


@torch.inference_mode()
def compute_avcd_attention_mass(
    *,
    model: Any,
    tokenizer: Any,
    processor: Mapping[str, Any],
    row: Mapping[str, Any],
    dtype: torch.dtype,
) -> dict[str, Any]:
    inputs = prepare_inputs(
        model=model,
        tokenizer=tokenizer,
        processor=processor,
        row=row,
        branch="full",
        dtype=dtype,
    )
    images = inputs["images"]
    if images is None:
        return {
            "dominant_modality": "language",
            "modality_dominance": {"video": 0.0, "audio": 0.0, "language": 1.0},
            "avg_dominance": [["language", 1.0]],
            "captured_layers": [],
            "span_source": "text_only",
        }
    input_ids = inputs["input_ids"]
    mm_positions = torch.where(input_ids[0] < 0)[0]
    if int(mm_positions.numel()) <= 0:
        raise RuntimeError("cannot locate VideoLLaMA2 multimodal placeholder token")
    mm_start = int(mm_positions[0].item())
    _ids, attention_mask, _past, inputs_embeds, _labels = model.prepare_inputs_labels_for_multimodal(
        input_ids=input_ids,
        attention_mask=inputs["attention_mask"],
        past_key_values=None,
        labels=None,
        images=images,
    )
    if not bool(torch.isfinite(inputs_embeds.detach().float()).all().item()):
        bad = int((~torch.isfinite(inputs_embeds.detach().float())).sum().item())
        raise FloatingPointError(f"non-finite multimodal inputs_embeds before AVCD reader: bad_entries={bad}")
    layer_rows: dict[int, torch.Tensor] = {}
    hooks: list[Any] = []

    def make_pre_hook(layer_idx: int):
        def hook_fn(module: torch.nn.Module, args: tuple[Any, ...], kwargs: Mapping[str, Any]):
            hidden_states = args[0] if args else kwargs.get("hidden_states")
            position_embeddings = kwargs.get("position_embeddings")
            if position_embeddings is None and len(args) >= 2:
                position_embeddings = args[1]
            attn_mask = kwargs.get("attention_mask")
            if attn_mask is None and len(args) >= 3:
                attn_mask = args[2]
            if hidden_states is None or position_embeddings is None:
                return None
            layer_rows[int(layer_idx)] = _qwen2_last_query_attention_row(
                module,
                hidden_states=hidden_states,
                attention_mask=attn_mask,
                position_embeddings=position_embeddings,
            )
            return None

        return hook_fn

    for layer_idx, layer_module in enumerate(language_layers(model)):
        hooks.append(
            layer_module.self_attn.register_forward_pre_hook(
                make_pre_hook(int(layer_idx)),
                with_kwargs=True,
            )
        )
    outputs = None
    reader_logits_finite: bool | None = None
    try:
        outputs = model(
            input_ids=None,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            use_cache=False,
            output_attentions=False,
            return_dict=True,
            pad_token_id=tokenizer.eos_token_id,
        )
        logits = outputs.logits[:, -1, :].detach().float()
        reader_logits_finite = bool(torch.isfinite(logits).all().item())
    finally:
        for hook in hooks:
            hook.remove()
        del outputs

    seq_len = int(inputs_embeds.shape[1])
    original_len_without_mm = int(input_ids.shape[1]) - int(mm_positions.numel())
    total_mm_len = int(seq_len) - int(original_len_without_mm)
    if total_mm_len <= 0:
        raise RuntimeError(f"invalid expanded multimodal length: total_mm_len={total_mm_len}")
    present = modality_presence(images)
    if present["video"] and present["audio"]:
        video_len = infer_stc_video_feature_len(model)
        if video_len <= 0 or video_len > total_mm_len:
            raise RuntimeError(f"invalid inferred video_len={video_len} for total_mm_len={total_mm_len}")
        audio_len = total_mm_len - video_len
    elif present["video"]:
        video_len = total_mm_len
        audio_len = 0
    elif present["audio"]:
        video_len = 0
        audio_len = total_mm_len
    else:
        video_len = 0
        audio_len = 0
    video_positions = torch.arange(mm_start, mm_start + video_len, dtype=torch.long)
    audio_positions = torch.arange(mm_start + video_len, mm_start + video_len + audio_len, dtype=torch.long)
    language_positions = torch.arange(mm_start + video_len + audio_len, seq_len, dtype=torch.long)
    effective_layers = sorted(int(layer_id) for layer_id in layer_rows)
    if len(effective_layers) > 1:
        effective_layers = effective_layers[:-1]
    if not effective_layers:
        raise RuntimeError("AVCD dominance reader did not capture any usable attention layer")

    per_layer: list[dict[str, float]] = []
    skipped_layers: list[dict[str, Any]] = []

    def mean_attention_per_position(row: torch.Tensor, positions: torch.Tensor) -> float:
        if int(positions.numel()) <= 0:
            return 0.0
        return float(row[positions].sum().item()) / float(positions.numel())

    for layer_idx in effective_layers:
        # [heads, key] -> mean over heads for the final query.
        row_attn = layer_rows[int(layer_idx)].float().mean(dim=0)
        if not bool(torch.isfinite(row_attn).all().item()):
            bad = int((~torch.isfinite(row_attn)).sum().item())
            skipped_layers.append({"layer": int(layer_idx), "reason": "nonfinite_attention", "bad_entries": bad})
            continue
        total = float(row_attn.sum().item())
        if not math.isfinite(total) or total <= 1.0e-12:
            skipped_layers.append({"layer": int(layer_idx), "reason": "invalid_attention_total", "total": total})
            continue
        video_mass = mean_attention_per_position(row_attn, video_positions) if video_len > 0 else 0.0
        audio_mass = mean_attention_per_position(row_attn, audio_positions) if audio_len > 0 else 0.0
        language_mass = mean_attention_per_position(row_attn, language_positions)
        per_layer.append(
            {
                "layer": int(layer_idx),
                "video": video_mass,
                "audio": audio_mass,
                "language": language_mass,
            }
        )
    if not per_layer:
        raise FloatingPointError(f"no finite AVCD attention layers; skipped_layers={skipped_layers[:8]}")
    avg = {
        "video": sum(item["video"] for item in per_layer) / max(1, len(per_layer)),
        "audio": sum(item["audio"] for item in per_layer) / max(1, len(per_layer)),
        "language": sum(item["language"] for item in per_layer) / max(1, len(per_layer)),
    }
    ranked = sorted(avg.items(), key=lambda item: item[1], reverse=True)
    return {
        "dominant_modality": ranked[0][0],
        "modality_dominance": {key: float(value) for key, value in avg.items()},
        "video_mass": float(avg["video"]),
        "audio_mass": float(avg["audio"]),
        "language_mass": float(avg["language"]),
        "avg_dominance": [[key, float(value)] for key, value in ranked],
        "captured_layers": [int(item["layer"]) for item in per_layer],
        "raw_captured_layers": sorted(int(layer_id) for layer_id in layer_rows),
        "skipped_layers": skipped_layers,
        "reader_logits_finite": reader_logits_finite,
        "modal_spans": {
            "mm_start": int(mm_start),
            "video_len": int(video_len),
            "audio_len": int(audio_len),
            "language_len": int(language_positions.numel()),
        },
        "aggregation": "mean_attention_per_token_over_inserted_spans",
        "span_source": "videollama2_manual_fp32_last_query_attention",
    }


def tensor_payload_to_json_scalars(payload: Mapping[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in payload.items():
        if isinstance(value, torch.Tensor):
            tensor = value.detach().float().cpu()
            out[f"{key}_norm"] = float(tensor.norm().item())
            continue
        if isinstance(value, (str, int, float, bool)) or value is None:
            out[key] = value
        elif isinstance(value, (list, tuple)):
            if all(isinstance(item, (str, int, float, bool)) or item is None for item in value):
                out[key] = list(value)
        elif isinstance(value, Mapping):
            clean = {
                str(k): v
                for k, v in value.items()
                if isinstance(v, (str, int, float, bool)) or v is None
            }
            if clean:
                out[key] = clean
    return out


def build_captures(
    *,
    model: Any,
    tokenizer: Any,
    processor: Mapping[str, Any],
    row: Mapping[str, Any],
    dtype: torch.dtype,
    branch_scores: Mapping[str, Mapping[str, Any]],
    target_modality: str,
    layers: Sequence[int],
) -> dict[str, dict[str, Any]]:
    target_branch = target_branch_for_modality(target_modality)
    non_target_branch = non_target_branch_for_modality(target_modality)
    if target_branch is None or non_target_branch is None:
        raise ValueError(f"cannot build captures for target_modality={target_modality!r}")
    branch_map = {
        "full": "full",
        "target": target_branch,
        "non_target": non_target_branch,
        "text": "text",
    }
    captures: dict[str, dict[str, Any]] = {}
    for role, vl_branch in branch_map.items():
        hidden = capture_hidden(
            model=model,
            tokenizer=tokenizer,
            processor=processor,
            row=row,
            branch=vl_branch,
            layers=layers,
            dtype=dtype,
        )
        score = branch_scores[typing_branch_key(vl_branch)]
        captures[role] = {
            "layer_vectors": {int(layer): hidden[int(layer)] for layer in layers},
            "final_prediction": score.get("answer"),
            "final_yes_no_margin": float(score.get("margin") or 0.0),
        }
    return captures


def branch_support_sum(
    branch_scores: Mapping[str, Mapping[str, Any]],
    *,
    answer: str | None,
) -> tuple[float | None, int]:
    label = normalize_yes_no(answer)
    if label is None:
        return None, 0
    support = 0.0
    count = 0
    for branch_record in branch_scores.values():
        if not isinstance(branch_record, Mapping):
            continue
        branch_label = normalize_yes_no(branch_record.get("answer"))
        if branch_label is None:
            continue
        count += 1
        if branch_label != label:
            continue
        value = branch_support_for_answer(dict(branch_record), candidate_answer=label)
        if value is None:
            margin = branch_record.get("margin")
            value = abs(finite_float(margin, 0.0)) if margin is not None else 1.0
        support += max(0.0, float(value))
    if count <= 0:
        return None, 0
    return float(support), int(count)


def build_base_conflict_budget(
    *,
    branch_scores: Mapping[str, Mapping[str, Any]],
    current_answer: str | None,
    candidate_answer: str | None,
    target_branch: str | None,
    non_target_branch: str | None,
) -> dict[str, Any]:
    conflict = bool(
        current_answer in {"Yes", "No"}
        and candidate_answer in {"Yes", "No"}
        and current_answer != candidate_answer
    )
    conflict_state = "target_vs_mad_baseline_conflict" if conflict else "no_target_prior_conflict"
    conflict_strength = "strong" if conflict else "none"
    cur_support, cur_count = branch_support_sum(branch_scores, answer=current_answer)
    alt_support, alt_count = branch_support_sum(branch_scores, answer=candidate_answer)
    margin = None
    if cur_support is not None and alt_support is not None:
        margin = float(alt_support) - float(cur_support)
    budget_row = {
        "conflict_state": conflict_state,
        "conflict_strength": conflict_strength,
        "current_answer_y0": current_answer,
        "baseline_answer": current_answer,
        "base_protocol_answer": current_answer,
        "mad_protocol_baseline_answer": current_answer,
        "candidate_answer_y": candidate_answer,
        "target_answer": candidate_answer,
        "target_branch": target_branch,
        "non_target_branch": non_target_branch,
        "branches": dict(branch_scores),
    }
    payload = soft_conflict_budget_from_row(
        budget_row,
        tau=FROZEN_OELPR_CONFLICT_BUDGET_TAU,
        strong_floor=FROZEN_OELPR_CONFLICT_BUDGET_STRONG_FLOOR,
        weak_floor=FROZEN_OELPR_CONFLICT_BUDGET_WEAK_FLOOR,
        current_answer=current_answer,
        candidate_answer=candidate_answer,
    )
    payload.update(
        {
            "conflict_state": conflict_state,
            "conflict_strength": conflict_strength,
            "soft_conflict_cur_branch_count": (
                payload.get("soft_conflict_cur_branch_count")
                if payload.get("soft_conflict_cur_branch_count") is not None
                else cur_count
            ),
            "soft_conflict_alt_branch_count": (
                payload.get("soft_conflict_alt_branch_count")
                if payload.get("soft_conflict_alt_branch_count") is not None
                else alt_count
            ),
            "budget_source": "videollama2_branch_support_margin",
            "oelpr_conflict_budget_tau": FROZEN_OELPR_CONFLICT_BUDGET_TAU,
            "oelpr_conflict_budget_strong_floor": FROZEN_OELPR_CONFLICT_BUDGET_STRONG_FLOOR,
            "oelpr_conflict_budget_weak_floor": FROZEN_OELPR_CONFLICT_BUDGET_WEAK_FLOOR,
        }
    )
    return payload


def build_frozen_carriers(
    *,
    captures: Mapping[str, Mapping[str, Any]],
    lm_head: torch.nn.Module,
    yes_id: int,
    no_id: int,
    current_answer: str | None,
    candidate_answer: str | None,
    layers: Sequence[int],
    base_conflict_budget: Mapping[str, Any] | None = None,
    evidence_validation_mode: str = "cross_modal_self_logit_continuous",
    evidence_prior_subspace_purification_lambda: float = 1.0,
) -> tuple[list[dict[str, Any]], dict[int, dict[str, torch.Tensor]], dict[str, Any]]:
    carrier_rows: list[dict[str, Any]] = []
    carrier_tensors: dict[int, dict[str, torch.Tensor]] = {}
    for layer in layers:
        payload = orthogonal_prior_direction(
            captures,
            lm_head=lm_head,
            yes_id=int(yes_id),
            no_id=int(no_id),
            current_answer=current_answer,
            candidate_answer=candidate_answer,
            layer=int(layer),
            self_logit_layers=FROZEN_SELF_LOGIT_LAYERS,
            evidence_direction_mode="target_minus_text",
            evidence_source="primary",
            evidence_source_blend_weight=0.5,
            cross_path_active_residual_keep=0.25,
            cross_path_active_prior_clean_lambda=0.75,
            evidence_validation_mode=str(evidence_validation_mode),
            evidence_validation_support_weight=0.5,
            evidence_validation_interaction_weight=0.25,
            evidence_validation_shortcut_lambda=0.5,
            evidence_validation_min_agreement=0.0,
            evidence_validation_trust_radius=1.0,
            evidence_validation_non_target_support_weight=0.75,
            evidence_validation_answer_margin_tau=1.0,
            evidence_validation_non_target_support_min_lift=0.0,
            prior_direction_mode="context_invalidated_shortcut",
            prior_factor_main_weight=1.0,
            prior_factor_fusion_weight=1.0,
            prior_protection="orthogonalized",
            prior_orthogonalization_lambda=1.0,
            prior_orthogonalization_policy="fixed",
            evidence_risk_tolerance=0.0,
            evidence_risk_grid_size=41,
            evidence_anchor_subspace="none",
            evidence_anchor_subspace_rank=2,
            evidence_prior_subspace_purification_lambda=float(evidence_prior_subspace_purification_lambda),
            evidence_transfer_mode="path_answer_margin_minimal",
        )
        row = tensor_payload_to_json_scalars(payload)
        row["layer"] = int(layer)
        carrier_rows.append(row)
        carrier_tensors[int(layer)] = {
            key: torch.as_tensor(payload[key]).detach().float().cpu()
            for key in (
                "u_e",
                "u_e_legacy",
                "u_e_perp_prior",
                "u_p_raw",
                "u_p_perp",
                "u_prior_non_target_main",
                "u_prior_fusion_residual",
                "evidence_transfer_vector",
            )
            if key in payload
        }
    reliability = candidate_reliability_from_carrier_rows(carrier_rows, tau=1.0)
    base_budget = dict(base_conflict_budget or {})
    if not base_budget:
        conflict = bool(
            current_answer in {"Yes", "No"}
            and candidate_answer in {"Yes", "No"}
            and current_answer != candidate_answer
        )
        base_budget = {
            "budget": 1.0 if conflict else 0.0,
            "mode_applied": conflict,
            "budget_reason": "fallback_binary_target_current_conflict",
            "conflict_state": "target_vs_mad_baseline_conflict" if conflict else "no_target_prior_conflict",
            "conflict_strength": "strong" if conflict else "none",
        }
    budget = candidate_reliable_conflict_budget(
        base_budget,
        reliability,
        weak_floor=FROZEN_OELPR_CONFLICT_BUDGET_WEAK_FLOOR,
    )
    return carrier_rows, carrier_tensors, budget


def build_manifest_row(
    *,
    row: Mapping[str, Any],
    record: Mapping[str, Any],
    current_answer: str | None,
    candidate_answer: str | None,
    target_modality: str,
    target_branch: str | None,
    non_target_branch: str | None,
    conflict_budget: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    conflict = bool(current_answer in {"Yes", "No"} and candidate_answer in {"Yes", "No"} and current_answer != candidate_answer)
    role = "unified_conflict" if conflict and target_modality in {"audio", "visual"} else "no_conflict_fallback"
    budget_payload = dict(conflict_budget or {})
    conflict_state = safe_text(budget_payload.get("conflict_state")) or (
        "target_vs_mad_baseline_conflict" if conflict else "no_target_prior_conflict"
    )
    conflict_strength = safe_text(budget_payload.get("conflict_strength")) or ("strong" if conflict else "none")
    return {
        "sample_id": row.get("sample_id"),
        "source_dataset": f"videollama2_typing_{safe_text(row.get('benchmark'))}",
        "manifest_role": role,
        "target_modality": target_modality,
        "effective_target_modality": target_modality if target_modality in {"audio", "visual"} else "unknown",
        "repair_target_modality": record.get("repair_target_modality"),
        "online_target_modality": record.get("online_target_modality"),
        "target_modality_dispatch_source": record.get("target_modality_dispatch_source"),
        "current_answer_y0": current_answer,
        "candidate_answer_y": candidate_answer,
        "current_best_accepts": False,
        "conflict_state": conflict_state,
        "conflict_strength": conflict_strength,
        "soft_conflict_score": budget_payload.get("soft_conflict_score"),
        "soft_conflict_cur_support": budget_payload.get("soft_conflict_cur_support"),
        "soft_conflict_alt_support": budget_payload.get("soft_conflict_alt_support"),
        "soft_conflict_margin": budget_payload.get("soft_conflict_margin"),
        "soft_conflict_cur_branch_count": budget_payload.get("soft_conflict_cur_branch_count"),
        "soft_conflict_alt_branch_count": budget_payload.get("soft_conflict_alt_branch_count"),
        "target_branch": target_branch,
        "non_target_branch": non_target_branch,
        "target_branch_answer": candidate_answer,
        "non_target_branch_answer": record.get("non_target_branch_answer"),
        "text_branch_answer": record.get("text_branch_answer"),
        "full_branch_answer": current_answer,
        "baseline_answer": current_answer,
        "base_protocol_answer": current_answer,
        "mad_protocol_baseline_answer": current_answer,
        "directed_prior_label": record.get("directed_prior_label"),
        "attribution_reason": record.get("attribution_reason"),
        "attribution_source_branch": record.get("attribution_source_branch"),
        "target_evidence_state": record.get("target_evidence_state"),
        "target_evidence_reason": record.get("target_evidence_reason"),
        "oelpr_conflict_budget": budget_payload,
        "runtime_uses_reference_answer": False,
        "selection_uses_reference_answer": False,
        "selection_uses_task_family": False,
    }


def build_runtime_row(row: Mapping[str, Any], manifest: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "sample_id": row.get("sample_id"),
        "runtime_source": "videollama2_strict_online_prior_typing",
        "source_dataset": manifest.get("source_dataset"),
        "question": row.get("question"),
        "formatted_question": row.get("question"),
        "prompt_text": row.get("question"),
        "video_path": row.get("video_path"),
        "audio_path": row.get("audio_path"),
        "target_modality": manifest.get("target_modality"),
        "baseline_answer": manifest.get("current_answer_y0"),
        "target_answer": manifest.get("candidate_answer_y"),
        "oelpr_conflict_budget": manifest.get("oelpr_conflict_budget"),
        "eval_type": "yes_no",
        "runtime_uses_reference_answer": False,
        "selection_uses_reference_answer": False,
        "selection_uses_task_family": False,
    }


def summarize(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    valid = [row for row in records if not row.get("error")]
    return {
        "n_rows": len(records),
        "n_valid": len(valid),
        "n_errors": len(records) - len(valid),
        "online_target_modality_counts": dict(Counter(safe_text(row.get("online_target_modality")) for row in valid)),
        "repair_target_modality_counts": dict(Counter(safe_text(row.get("repair_target_modality")) for row in valid)),
        "directed_prior_label_counts": dict(Counter(safe_text(row.get("directed_prior_label")) for row in valid)),
        "target_evidence_state_counts": dict(Counter(safe_text(row.get("target_evidence_state")) for row in valid)),
        "manifest_role_counts": dict(Counter(safe_text(row.get("manifest_role")) for row in valid)),
        "carrier_built": sum(1 for row in valid if row.get("carrier_built") is True),
        "policy_candidate_conflicts": sum(1 for row in valid if row.get("candidate_answer_y") != row.get("current_answer_y0")),
        "baseline_correct": sum(1 for row in valid if row.get("baseline_correct") is True),
        "target_branch_correct": sum(1 for row in valid if row.get("target_branch_correct") is True),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Strict VideoLLaMA2 online prior typing and frozen carrier probe.")
    parser.add_argument("--benchmark", choices=["avh", "cmm"], default="avh")
    parser.add_argument("--avhbench-dir", type=Path, default=ROOT / "data" / "AVHBench")
    parser.add_argument("--cmm-data-path", type=Path, default=ROOT / "data" / "CMM" / "all_data_final_reorg.json")
    parser.add_argument("--cmm-media-base-dir", type=Path, default=ROOT / "data" / "CMM")
    parser.add_argument("--selected-rows-path", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--model-path", type=str, default="DAMO-NLP-SG/VideoLLaMA2.1-7B-AV"
    )
    parser.add_argument("--max-rows", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--init-mode", choices=["eager", "sdpa", "official_flash"], default="eager")
    parser.add_argument("--dtype", choices=["float16", "bfloat16", "float32"], default="bfloat16")
    parser.add_argument("--warmup-rows-path", type=Path, default=None)
    parser.add_argument("--warmup-max-rows", type=int, default=1)
    parser.add_argument("--warmup-retries", type=int, default=2)
    parser.add_argument("--target-modality-margin-threshold", type=float, default=0.12)
    parser.add_argument("--source-confidence-gap-threshold", type=float, default=0.05)
    parser.add_argument("--source-tie-threshold", type=float, default=0.03)
    parser.add_argument("--avcd-source-mass-weight", type=float, default=0.35)
    parser.add_argument("--repair-joint-both-threshold", type=float, default=0.25)
    parser.add_argument("--repair-joint-support-threshold", type=float, default=0.4)
    parser.add_argument("--repair-joint-tie-threshold", type=float, default=0.3)
    parser.add_argument(
        "--evidence-validation-mode",
        choices=["cross_modal_self_logit_continuous", "cross_modal_self_logit_candidate_only_decontam"],
        default="cross_modal_self_logit_continuous",
    )
    parser.add_argument("--evidence-prior-subspace-purification-lambda", type=float, default=1.0)
    parser.add_argument("--allow-avcd-fallback", action="store_true")
    parser.add_argument("--no-carriers", action="store_true")
    parser.add_argument("--no-save-carrier-tensors", action="store_true")
    parser.add_argument("--fail-on-errors", action="store_true")
    parser.add_argument("--allow-warmup-errors", action="store_true")
    parser.add_argument("--no-progress", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dtype = torch_dtype_from_name(args.dtype)
    max_rows = int(args.max_rows) if int(args.max_rows) > 0 else None
    if args.selected_rows_path is not None:
        selected_all = [
            json.loads(line)
            for line in args.selected_rows_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    elif args.benchmark == "avh":
        from run_videollama2_lccs_cross_model import select_avh_unary_full_rows  # noqa: WPS433

        selected_all = select_avh_unary_full_rows(avhbench_dir=args.avhbench_dir, max_rows=max_rows)
    else:
        selected_all = select_cmm_unary_full_rows(
            cmm_data_path=args.cmm_data_path,
            cmm_media_base_dir=args.cmm_media_base_dir,
            max_rows=max_rows,
        )
    selected = [row for idx, row in enumerate(selected_all) if idx % int(args.num_shards) == int(args.shard_index)]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows_path = args.output_dir / "records.jsonl"
    manifest_path = args.output_dir / "manifest.jsonl"
    runtime_path = args.output_dir / "runtime_rows.jsonl"
    eval_sidecar_path = args.output_dir / "eval_sidecar.jsonl"
    carrier_rows_path = args.output_dir / "carrier_rows.jsonl"
    for path in (rows_path, manifest_path, runtime_path, eval_sidecar_path, carrier_rows_path):
        path.write_text("", encoding="utf-8")

    run_config = {
        "kind": "videollama2_strict_online_prior_typing_probe_v1",
        "benchmark": args.benchmark,
        "model_path": args.model_path,
        "torch_version": torch.__version__,
        "transformers_version": transformers.__version__,
        "dtype": args.dtype,
        "init_mode": args.init_mode,
        "n_selected_all": len(selected_all),
        "n_selected_shard": len(selected),
        "num_shards": int(args.num_shards),
        "shard_index": int(args.shard_index),
        "runtime_uses_reference_answer": False,
        "selection_uses_reference_answer": False,
        "warmup": {
            "rows_path": str(args.warmup_rows_path) if args.warmup_rows_path is not None else "",
            "max_rows": int(args.warmup_max_rows),
            "retries": int(args.warmup_retries),
            "n_rows": 0,
            "n_ok": 0,
            "n_errors": 0,
            "n_attempts": 0,
            "errors": [],
            "runtime_uses_reference_answer": False,
            "selection_uses_reference_answer": False,
        },
        "typing_contract": (
            "VideoLLaMA2 produces full/audio/visual/text branch answers, MAD head-only modality "
            "distribution, AVCD-style inserted-token attention dominance, online target/repair typing, "
            "directed prior attribution, and frozen context-invalidated carrier rows."
        ),
        "frozen_mainline": {
            "prior_direction": "context_invalidated_shortcut",
            "prior_protection": "orthogonalized",
            "evidence_direction": "target_minus_text",
            "evidence_source": "primary",
            "evidence_validation_mode": str(args.evidence_validation_mode),
            "evidence_validation_self_logit_layers": FROZEN_SELF_LOGIT_LAYERS,
            "evidence_validation_non_target_support_weight": 0.75,
            "evidence_prior_subspace_purification_lambda": float(
                args.evidence_prior_subspace_purification_lambda
            ),
            "evidence_transfer_mode": "path_answer_margin_minimal",
            "oelpr_conflict_budget_mode": "candidate_safe_evidence_budget",
        },
    }
    write_json(args.output_dir / "run_config.json", run_config)
    write_jsonl(args.output_dir / "selected_rows.jsonl", selected)

    if args.init_mode == "official_flash":
        model, processor, tokenizer = model_init(
            args.model_path,
            device_map=torch.device("cuda"),
            use_flash_attn=True,
            torch_dtype=dtype,
        )
    else:
        attn_impl = "eager" if args.init_mode == "eager" else "sdpa"
        model, processor, tokenizer = model_init(
            args.model_path,
            device_map=torch.device("cuda"),
            use_flash_attn=False,
            attn_implementation=attn_impl,
            torch_dtype=dtype,
        )
    model.eval()
    if max(FROZEN_ALL_LAYERS) >= len(language_layers(model)):
        raise ValueError(
            f"Layer index out of range: layers={FROZEN_ALL_LAYERS}, model_layers={len(language_layers(model))}"
        )
    lm_head = model.get_output_embeddings() if hasattr(model, "get_output_embeddings") else getattr(model, "lm_head")
    if lm_head is None:
        lm_head = getattr(model, "lm_head")
    yes_id = yes_no_token_id(tokenizer, "Yes")
    no_id = yes_no_token_id(tokenizer, "No")
    answer_space = SimpleNamespace(kind="yes_no")

    if args.warmup_rows_path is not None:
        warmup_rows = [
            json.loads(line)
            for line in args.warmup_rows_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if int(args.warmup_max_rows) > 0:
            warmup_rows = warmup_rows[: int(args.warmup_max_rows)]
        warmup_report = dict(run_config["warmup"])
        warmup_report["n_rows"] = len(warmup_rows)
        max_attempts = max(1, int(args.warmup_retries) + 1)
        for warmup_row in warmup_rows:
            warmup_ok = False
            last_exc: Exception | None = None
            for attempt in range(max_attempts):
                warmup_report["n_attempts"] = int(warmup_report.get("n_attempts") or 0) + 1
                try:
                    compute_avcd_attention_mass(
                        model=model,
                        tokenizer=tokenizer,
                        processor=processor,
                        row=warmup_row,
                        dtype=dtype,
                    )
                    release_cuda_cache()
                    for branch in ("full", "visual", "audio", "text"):
                        score_yesno_branch(
                            model=model,
                            tokenizer=tokenizer,
                            processor=processor,
                            row=warmup_row,
                            branch=branch,
                            dtype=dtype,
                        )
                        release_cuda_cache()
                    warmup_report["n_ok"] = int(warmup_report.get("n_ok") or 0) + 1
                    warmup_ok = True
                    break
                except Exception as exc:
                    last_exc = exc
                    release_cuda_cache()
                    if attempt + 1 < max_attempts:
                        continue
            if not warmup_ok:
                exc = last_exc if last_exc is not None else RuntimeError("unknown warmup failure")
                warmup_report["n_errors"] = int(warmup_report.get("n_errors") or 0) + 1
                errors = list(warmup_report.get("errors") or [])
                errors.append({"sample_id": warmup_row.get("sample_id"), "error": repr(exc)})
                warmup_report["errors"] = errors
                if not args.allow_warmup_errors:
                    run_config["warmup"] = warmup_report
                    write_json(args.output_dir / "run_config.json", run_config)
                    raise exc
        run_config["warmup"] = warmup_report
        write_json(args.output_dir / "run_config.json", run_config)

    records: list[dict[str, Any]] = []
    carrier_tensor_store: dict[str, dict[int, dict[str, torch.Tensor]]] = {}
    iterator = tqdm(
        selected,
        desc=f"vl2_typing_s{args.shard_index}",
        disable=bool(args.no_progress),
        unit="sample",
    )
    for row in iterator:
        start = time.time()
        sample_id = safe_text(row.get("sample_id"))
        try:
            avcd_error: Exception | None = None
            try:
                avcd_payload = compute_avcd_attention_mass(
                    model=model,
                    tokenizer=tokenizer,
                    processor=processor,
                    row=row,
                    dtype=dtype,
                )
            except Exception as exc:
                if not args.allow_avcd_fallback:
                    raise
                avcd_error = exc
                avcd_payload = {}
            release_cuda_cache()
            branch_scores = {
                "full": score_yesno_branch(model=model, tokenizer=tokenizer, processor=processor, row=row, branch="full", dtype=dtype),
                "visual_only": score_yesno_branch(model=model, tokenizer=tokenizer, processor=processor, row=row, branch="visual", dtype=dtype),
                "audio_only": score_yesno_branch(model=model, tokenizer=tokenizer, processor=processor, row=row, branch="audio", dtype=dtype),
                "text_only": score_yesno_branch(model=model, tokenizer=tokenizer, processor=processor, row=row, branch="text", dtype=dtype),
            }
            mad_payload = score_modality_probe(
                model=model,
                tokenizer=tokenizer,
                processor=processor,
                row=row,
                dtype=dtype,
            )
            if avcd_error is not None:
                # This is an explicit diagnostic fallback, not the intended path.
                full_answer = branch_answer(branch_scores["full"])
                avcd_payload = {
                    "dominant_modality": "unknown",
                    "modality_dominance": {"video": 0.0, "audio": 0.0, "language": 0.0},
                    "avg_dominance": [],
                    "captured_layers": [],
                    "span_source": "fallback_unavailable",
                    "fallback_reason": repr(avcd_error),
                    "full_answer_support_visual": branch_support_for_answer(branch_scores["visual_only"], candidate_answer=full_answer),
                    "full_answer_support_audio": branch_support_for_answer(branch_scores["audio_only"], candidate_answer=full_answer),
                    "full_answer_support_text": branch_support_for_answer(branch_scores["text_only"], candidate_answer=full_answer),
                }
            release_cuda_cache()
            online_target = infer_target_modality_online(
                question=safe_text(row.get("question")),
                branch_scores=branch_scores,
                mad_payload=mad_payload,
                avcd_payload=avcd_payload,
                margin_threshold=float(args.target_modality_margin_threshold),
                avcd_target_mass_weight=0.35,
            )
            repair_target = infer_repair_target_online(
                question=safe_text(row.get("question")),
                branch_scores=branch_scores,
                mad_payload=mad_payload,
                avcd_payload=avcd_payload,
                margin_threshold=float(args.target_modality_margin_threshold),
                joint_both_threshold=float(args.repair_joint_both_threshold),
                joint_support_threshold=float(args.repair_joint_support_threshold),
                joint_tie_threshold=float(args.repair_joint_tie_threshold),
            )
            effective_target = safe_text(repair_target.get("repair_target_modality"))
            if effective_target not in {"audio", "visual"}:
                effective_target = safe_text(online_target.get("target_modality"))
            target_branch = target_branch_for_modality(effective_target)
            non_target_branch = non_target_branch_for_modality(effective_target)
            attribution = attribute_directed_prior(
                target_modality=effective_target,
                branch_scores=branch_scores,
                answer_space=answer_space,
                avcd_payload=avcd_payload,
                source_confidence_gap_threshold=float(args.source_confidence_gap_threshold),
                source_tie_threshold=float(args.source_tie_threshold),
                avcd_source_mass_weight=float(args.avcd_source_mass_weight),
            )
            target_evidence_state = classify_target_evidence_state(
                target_modality=effective_target,
                branch_scores=branch_scores,
                answer_space=answer_space,
            )
            current_answer = normalize_yes_no(branch_scores["full"].get("answer"))
            candidate_answer = (
                normalize_yes_no(branch_scores[typing_branch_key(target_branch)].get("answer"))
                if target_branch is not None
                else None
            )
            non_target_answer = (
                normalize_yes_no(branch_scores[typing_branch_key(non_target_branch)].get("answer"))
                if non_target_branch is not None
                else None
            )
            base_conflict_budget = build_base_conflict_budget(
                branch_scores=branch_scores,
                current_answer=current_answer,
                candidate_answer=candidate_answer,
                target_branch=target_branch,
                non_target_branch=non_target_branch,
            )
            carrier_rows: list[dict[str, Any]] = []
            conflict_budget: dict[str, Any] = {
                "budget": 0.0,
                "budget_reason": "carrier_not_built",
                "mode_applied": False,
            }
            carrier_built = False
            if not args.no_carriers and effective_target in {"audio", "visual"}:
                captures = build_captures(
                    model=model,
                    tokenizer=tokenizer,
                    processor=processor,
                    row=row,
                    dtype=dtype,
                    branch_scores=branch_scores,
                    target_modality=effective_target,
                    layers=FROZEN_ALL_LAYERS,
                )
                carrier_rows, carrier_tensors, conflict_budget = build_frozen_carriers(
                    captures=captures,
                    lm_head=lm_head,
                    yes_id=yes_id,
                    no_id=no_id,
                    current_answer=current_answer,
                    candidate_answer=candidate_answer,
                    layers=FROZEN_ALL_LAYERS,
                    base_conflict_budget=base_conflict_budget,
                    evidence_validation_mode=str(args.evidence_validation_mode),
                    evidence_prior_subspace_purification_lambda=float(
                        args.evidence_prior_subspace_purification_lambda
                    ),
                )
                carrier_built = True
                if not args.no_save_carrier_tensors:
                    carrier_tensor_store[sample_id] = carrier_tensors
                for carrier_row in carrier_rows:
                    append_jsonl(
                        carrier_rows_path,
                        {
                            "sample_id": sample_id,
                            "benchmark": row.get("benchmark"),
                            "target_modality": effective_target,
                            "current_answer_y0": current_answer,
                            "candidate_answer_y": candidate_answer,
                            **carrier_row,
                        },
                    )
                release_cuda_cache()
            record = {
                "sample_id": row.get("sample_id"),
                "video_id": official_video_id(row),
                "benchmark": row.get("benchmark"),
                "mad_protocol_task": row.get("mad_protocol_task"),
                "question": row.get("question"),
                "video_path": row.get("video_path"),
                "audio_path": row.get("audio_path"),
                "reference_answer_offline": row.get("reference_answer"),
                "online_target_modality": safe_text(online_target.get("target_modality")),
                "online_target_reason": safe_text(online_target.get("target_modality_reason")),
                "online_target_meta": online_target,
                "repair_target_modality": safe_text(repair_target.get("repair_target_modality")),
                "repair_target_reason": safe_text(repair_target.get("repair_target_reason")),
                "repair_target_confidence": repair_target.get("repair_target_confidence"),
                "repair_target_meta": repair_target,
                "target_modality": effective_target if effective_target in {"audio", "visual"} else "unknown",
                "target_modality_dispatch_source": (
                    "repair_target_modality"
                    if safe_text(repair_target.get("repair_target_modality")) in {"audio", "visual"}
                    else "online_target_modality"
                ),
                "target_branch": target_branch,
                "non_target_branch": non_target_branch,
                "current_answer_y0": current_answer,
                "candidate_answer_y": candidate_answer,
                "non_target_branch_answer": non_target_answer,
                "text_branch_answer": normalize_yes_no(branch_scores["text_only"].get("answer")),
                "full_branch_answer": current_answer,
                "baseline_prediction": current_answer,
                "baseline_correct": answer_correct_strict(current_answer, row.get("reference_answer")),
                "target_branch_correct": answer_correct_strict(candidate_answer, row.get("reference_answer"))
                if candidate_answer is not None
                else None,
                "branches": branch_scores,
                "mad_runtime": mad_payload,
                "avcd_attention_mass": avcd_payload,
                "directed_prior_label": safe_text(attribution.get("directed_prior_label")),
                "attribution_reason": safe_text(attribution.get("reason")),
                "attribution_target_branch": attribution.get("target_branch"),
                "attribution_source_branch": attribution.get("source_branch"),
                "attribution_avcd_source_branch": (attribution.get("avcd_source_meta") or {}).get("avcd_source_branch"),
                "attribution_avcd_source_meta": attribution.get("avcd_source_meta"),
                "source_candidates": attribution.get("source_candidates"),
                "target_evidence_state": safe_text(target_evidence_state.get("target_evidence_state")),
                "target_evidence_reason": safe_text(target_evidence_state.get("target_evidence_reason")),
                "target_evidence_meta": target_evidence_state,
                "carrier_built": bool(carrier_built),
                "carrier_layers": FROZEN_ALL_LAYERS if carrier_built else [],
                "carrier_rows_in_jsonl": len(carrier_rows),
                "oelpr_base_conflict_budget": base_conflict_budget,
                "oelpr_conflict_budget": conflict_budget,
                "runtime_uses_reference_answer": False,
                "selection_uses_reference_answer": False,
                "selection_uses_task_family": False,
                "runtime_seconds": time.time() - start,
                "error": "",
            }
            manifest = build_manifest_row(
                row=row,
                record=record,
                current_answer=current_answer,
                candidate_answer=candidate_answer,
                target_modality=record["target_modality"],
                target_branch=target_branch,
                non_target_branch=non_target_branch,
                conflict_budget=conflict_budget,
            )
            record["manifest_role"] = manifest["manifest_role"]
            runtime = build_runtime_row(row, manifest)
            eval_sidecar = {
                "sample_id": row.get("sample_id"),
                "benchmark": row.get("benchmark"),
                "mad_protocol_task": row.get("mad_protocol_task"),
                "eval_reference_answer": row.get("reference_answer"),
                "eval_target_modality": row.get("target_modality"),
                "eval_baseline_correct": record["baseline_correct"],
                "eval_probe_full_correct": record["baseline_correct"],
                "runtime_uses_reference_answer": False,
                "selection_uses_reference_answer": False,
            }
        except Exception as exc:
            record = {
                "sample_id": row.get("sample_id"),
                "benchmark": row.get("benchmark"),
                "mad_protocol_task": row.get("mad_protocol_task"),
                "question": row.get("question"),
                "reference_answer_offline": row.get("reference_answer"),
                "runtime_seconds": time.time() - start,
                "runtime_uses_reference_answer": False,
                "selection_uses_reference_answer": False,
                "error": repr(exc),
            }
            manifest = {
                "sample_id": row.get("sample_id"),
                "manifest_role": "typing_error",
                "target_modality": "unknown",
                "current_answer_y0": None,
                "candidate_answer_y": None,
                "runtime_uses_reference_answer": False,
                "selection_uses_reference_answer": False,
                "error": repr(exc),
            }
            runtime = build_runtime_row(row, manifest)
            eval_sidecar = {
                "sample_id": row.get("sample_id"),
                "benchmark": row.get("benchmark"),
                "mad_protocol_task": row.get("mad_protocol_task"),
                "eval_reference_answer": row.get("reference_answer"),
                "runtime_uses_reference_answer": False,
                "selection_uses_reference_answer": False,
                "error": repr(exc),
            }
        append_jsonl(rows_path, record)
        append_jsonl(manifest_path, manifest)
        append_jsonl(runtime_path, runtime)
        append_jsonl(eval_sidecar_path, eval_sidecar)
        records.append(record)

    if not args.no_save_carrier_tensors:
        torch.save(carrier_tensor_store, args.output_dir / "carrier_tensors.pt")
    summary = summarize(records)
    summary["run_config"] = run_config
    write_json(args.output_dir / "summary.json", summary)
    if args.fail_on_errors and int(summary.get("n_errors") or 0) > 0:
        raise RuntimeError(f"typing shard produced n_errors={summary.get('n_errors')} in {args.output_dir}")
    print(json.dumps(summary, ensure_ascii=False, indent=2)[:4000])


if __name__ == "__main__":
    main()
