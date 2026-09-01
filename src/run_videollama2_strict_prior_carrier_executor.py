#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import json
import math
import os
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import torch
import transformers
from tqdm.auto import tqdm
from transformers.models.qwen2.modeling_qwen2 import apply_rotary_pos_emb, repeat_kv


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
for path in (SCRIPT_DIR, ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

VIDEOLLAMA2_ROOT = ROOT / "third_party" / "VideoLLaMA2"
if str(VIDEOLLAMA2_ROOT) not in sys.path:
    sys.path.insert(0, str(VIDEOLLAMA2_ROOT))

from videollama2 import model_init  # noqa: E402

from run_prior_carrier_suppression_executor import (  # noqa: E402
    _allpath_head_scaled_output,
    _allpath_token_scaled_output,
    _head_answer_contribution_scores,
    _masked_value_contribution_from_mask,
    _path_answer_margin_evidence_delta,
    _path_constrained_hidden_delta_from_paths,
    _path_soft_hidden_carriers_from_token_heads,
    _renorm_like,
    _select_evidence_support_heads,
    _select_patch_heads,
    _token_head_alpha_weights_for_policy,
    _token_selection_mask_for_answer_path,
    adaptive_intervention_layer_weights,
    orthonormal_basis,
    soft_minimal_suppression_delta,
    subspace_projection,
    up_r_projection_delta,
)
from run_videollama2_lccs_cross_model import (  # noqa: E402
    normalize_yes_no,
    safe_text,
    torch_dtype_from_name,
    unit,
)
from run_videollama2_lccs_strict_mad_path import (  # noqa: E402
    candidate_token_ids,
    language_layers,
    official_cmm_score_row,
    official_score_row,
    official_video_id,
    prepare_inputs,
    score_candidate_sequence,
    write_json,
)
from run_videollama2_strict_online_prior_typing_probe import (  # noqa: E402
    infer_stc_video_feature_len,
    modality_presence,
    yes_no_token_id,
)


FROZEN_LAYERS = [12, 16, 20, 24, 26, 27]
FROZEN_LAYER_GROUP_WEIGHTS = {12: 0.10, 16: 0.15, 20: 0.20, 24: 0.15, 26: 0.30, 27: 0.10}
FROZEN_EVIDENCE_LAYER_WEIGHTS = {12: 0.25, 16: 0.35, 20: 0.40}
FROZEN_PRIOR_LAYER_WEIGHTS = {24: 0.23, 26: 0.61, 27: 0.16}
FROZEN_BETA = 0.95
FROZEN_ALPHA = 0.10
_YES_NO_PATTERN = re.compile(r"\b(yes|no)\b", re.IGNORECASE)


def parse_int_list(value: str | None) -> list[int] | None:
    if value is None or not str(value).strip():
        return None
    out: list[int] = []
    for item in str(value).split(","):
        item = item.strip()
        if item:
            out.append(int(item))
    return out


def parse_layer_weight_map(value: str | None) -> dict[int, float] | None:
    if value is None or not str(value).strip():
        return None
    out: dict[int, float] = {}
    for item in str(value).split(","):
        item = item.strip()
        if not item:
            continue
        if ":" not in item:
            raise ValueError(f"layer weight must be layer:weight, got {item!r}")
        layer, weight = item.split(":", 1)
        out[int(layer.strip())] = float(weight.strip())
    return out


def restrict_weight_map(layers: Sequence[int], weights: Mapping[int, float]) -> dict[int, float]:
    layer_set = {int(layer) for layer in layers}
    return {int(layer): float(weight) for layer, weight in weights.items() if int(layer) in layer_set}


def configure_frozen_layers_from_args(args: argparse.Namespace) -> None:
    global FROZEN_LAYERS, FROZEN_LAYER_GROUP_WEIGHTS, FROZEN_EVIDENCE_LAYER_WEIGHTS, FROZEN_PRIOR_LAYER_WEIGHTS

    layers = parse_int_list(args.layers)
    if layers is not None:
        if not layers:
            raise ValueError("--layers must contain at least one layer")
        FROZEN_LAYERS = layers

    parsed_group = parse_layer_weight_map(args.layer_group_weights)
    parsed_evidence = parse_layer_weight_map(args.evidence_layer_weights)
    parsed_prior = parse_layer_weight_map(args.prior_layer_weights)
    if parsed_group is not None:
        FROZEN_LAYER_GROUP_WEIGHTS = restrict_weight_map(FROZEN_LAYERS, parsed_group)
    else:
        FROZEN_LAYER_GROUP_WEIGHTS = restrict_weight_map(FROZEN_LAYERS, FROZEN_LAYER_GROUP_WEIGHTS)
    if parsed_evidence is not None:
        FROZEN_EVIDENCE_LAYER_WEIGHTS = restrict_weight_map(FROZEN_LAYERS, parsed_evidence)
    else:
        FROZEN_EVIDENCE_LAYER_WEIGHTS = restrict_weight_map(FROZEN_LAYERS, FROZEN_EVIDENCE_LAYER_WEIGHTS)
    if parsed_prior is not None:
        FROZEN_PRIOR_LAYER_WEIGHTS = restrict_weight_map(FROZEN_LAYERS, parsed_prior)
    else:
        FROZEN_PRIOR_LAYER_WEIGHTS = restrict_weight_map(FROZEN_LAYERS, FROZEN_PRIOR_LAYER_WEIGHTS)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), ensure_ascii=False, sort_keys=True) + "\n")


def append_jsonl(path: Path, row: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(dict(row), ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()


def release_cuda_cache() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def finite_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float(default)
    if not math.isfinite(out):
        return float(default)
    return float(out)


def finite_mean(values: Iterable[Any]) -> float | None:
    cleaned: list[float] = []
    for value in values:
        try:
            item = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(item):
            cleaned.append(float(item))
    if not cleaned:
        return None
    return sum(cleaned) / float(len(cleaned))


def clamp01(value: Any, default: float = 0.0) -> float:
    return max(0.0, min(1.0, finite_float(value, default)))


def cosine(a: torch.Tensor, b: torch.Tensor) -> float | None:
    a = torch.as_tensor(a).detach().float().cpu()
    b = torch.as_tensor(b).detach().float().cpu()
    na = float(a.norm().item())
    nb = float(b.norm().item())
    if na <= 1.0e-12 or nb <= 1.0e-12:
        return None
    return float(torch.dot(a, b).item() / (na * nb))


def vector_projection(vec: torch.Tensor, direction: torch.Tensor) -> torch.Tensor:
    vec = torch.as_tensor(vec).detach().float().cpu()
    direction = torch.as_tensor(direction).detach().float().cpu()
    denom = float(torch.dot(direction, direction).item())
    if denom <= 1.0e-12:
        return torch.zeros_like(vec)
    return float(torch.dot(vec, direction).item() / denom) * direction


def safe_correct(prediction: Any, reference: Any) -> bool | None:
    pred = normalize_yes_no(prediction)
    ref = normalize_yes_no(reference)
    if pred is None or ref is None:
        return None
    return pred == ref


def answer_sign(answer: Any) -> int | None:
    label = normalize_yes_no(answer)
    if label == "Yes":
        return 1
    if label == "No":
        return -1
    return None


def signed_margin(answer: Any, margin_yes_minus_no: float | None) -> float | None:
    sign = answer_sign(answer)
    if sign is None or margin_yes_minus_no is None:
        return None
    return float(sign) * float(margin_yes_minus_no)


def normalize_weights(layers: Sequence[int], weights: Mapping[int, float]) -> dict[int, float]:
    raw = {int(layer): max(0.0, float(weights.get(int(layer), 0.0))) for layer in layers}
    total = sum(raw.values())
    if total <= 1.0e-12:
        return {int(layer): 1.0 / float(len(layers)) for layer in layers}
    return {int(layer): value / total for layer, value in raw.items()}


def typing_paths(args: argparse.Namespace) -> dict[str, Path]:
    if args.typing_dir is None:
        required = {
            "manifest": args.manifest_path,
            "runtime": args.runtime_rows_path,
            "carrier_rows": args.carrier_rows_path,
            "carrier_tensors": args.carrier_tensors_path,
        }
        missing = [name for name, value in required.items() if value is None]
        if missing:
            raise ValueError(f"missing input paths without --typing-dir: {missing}")
        out = {name: Path(value) for name, value in required.items()}
        if args.selected_rows_path is not None:
            out["selected"] = args.selected_rows_path
        if args.eval_sidecar_path is not None:
            out["eval_sidecar"] = args.eval_sidecar_path
        if args.records_path is not None:
            out["records"] = args.records_path
        return out

    root = Path(args.typing_dir)
    out = {
        "manifest": args.manifest_path or root / "manifest.jsonl",
        "runtime": args.runtime_rows_path or root / "runtime_rows.jsonl",
        "carrier_rows": args.carrier_rows_path or root / "carrier_rows.jsonl",
        "carrier_tensors": args.carrier_tensors_path or root / "carrier_tensors.pt",
    }
    selected = args.selected_rows_path or root / "selected_rows.jsonl"
    eval_sidecar = args.eval_sidecar_path or root / "eval_sidecar.jsonl"
    records = args.records_path or root / "records.jsonl"
    if selected.exists():
        out["selected"] = selected
    if eval_sidecar.exists():
        out["eval_sidecar"] = eval_sidecar
    if records.exists():
        out["records"] = records
    return out


def load_carrier_tensors(path: Path) -> dict[str, dict[int, dict[str, torch.Tensor]]]:
    raw = torch.load(path, map_location="cpu")
    if not isinstance(raw, Mapping):
        raise TypeError(f"carrier_tensors must be a mapping, got {type(raw)!r}")
    out: dict[str, dict[int, dict[str, torch.Tensor]]] = {}
    for sample_id, by_layer in raw.items():
        if not isinstance(by_layer, Mapping):
            continue
        layer_payload: dict[int, dict[str, torch.Tensor]] = {}
        for layer_key, payload in by_layer.items():
            if not isinstance(payload, Mapping):
                continue
            layer = int(layer_key)
            layer_payload[layer] = {
                str(key): torch.as_tensor(value).detach().float().cpu()
                for key, value in payload.items()
                if isinstance(value, torch.Tensor) or hasattr(value, "__array__")
            }
        out[safe_text(sample_id)] = layer_payload
    return out


def current_evidence_source_mode() -> str:
    mode = safe_text(os.getenv("EVIDENCE_SOURCE_MODE", "raw")).lower()
    if mode not in {"raw", "clean"}:
        raise ValueError(f"unsupported EVIDENCE_SOURCE_MODE={mode!r}")
    return mode


def current_evidence_validation_mode() -> str:
    mode = safe_text(os.getenv("EVIDENCE_VALIDATION_MODE", "none")).lower()
    if mode not in {"none", "cross_modal_self_logit_candidate_only_decontam"}:
        raise ValueError(f"unsupported EVIDENCE_VALIDATION_MODE={mode!r}")
    return mode


def apply_evidence_validation_from_tensors(
    *,
    evidence: torch.Tensor,
    raw_prior: torch.Tensor,
    payload: Mapping[str, torch.Tensor],
    mode: str,
    prior_subspace_purification_lambda: float,
) -> tuple[torch.Tensor, dict[str, Any]]:
    evidence = torch.as_tensor(evidence).detach().float().cpu()
    raw_prior = torch.as_tensor(raw_prior).detach().float().cpu()
    lam = max(0.0, min(1.0, float(prior_subspace_purification_lambda)))
    debug: dict[str, Any] = {
        "evidence_validation_mode": str(mode),
        "evidence_validation_applied_in_executor": False,
        "evidence_prior_subspace_purification_lambda": float(lam),
        "active_decontam_applied": False,
        "active_decontam_lambda": 0.0,
        "active_decontam_projection_norm": 0.0,
        "active_decontam_pre_norm": float(evidence.norm().item()),
        "active_decontam_post_norm": float(evidence.norm().item()),
        "active_decontam_cos_pre_post": 1.0,
        "active_decontam_basis_dim": 0,
        "active_decontam_basis_source": "none",
    }
    if str(mode) == "none":
        return evidence, debug
    if str(mode) != "cross_modal_self_logit_candidate_only_decontam":
        raise ValueError(f"unsupported evidence validation mode: {mode!r}")
    basis_vectors: list[torch.Tensor] = [raw_prior]
    basis_source = "raw_prior"
    if "u_prior_non_target_main" in payload:
        basis_vectors.append(torch.as_tensor(payload["u_prior_non_target_main"]).detach().float().cpu())
        basis_source = "raw_prior_plus_non_target_main"
    basis = orthonormal_basis(basis_vectors, max_rank=2)
    projection = subspace_projection(evidence, basis)
    pre = evidence.clone()
    candidate = evidence - lam * projection
    if float(candidate.norm().item()) > 1.0e-12:
        evidence = _renorm_like(candidate, pre)
    debug.update(
        {
            "evidence_validation_applied_in_executor": True,
            "active_decontam_applied": bool(lam > 0.0),
            "active_decontam_lambda": float(lam),
            "active_decontam_projection_norm": float(projection.norm().item()),
            "active_decontam_pre_norm": float(pre.norm().item()),
            "active_decontam_post_norm": float(evidence.norm().item()),
            "active_decontam_cos_pre_post": cosine(pre, evidence),
            "active_decontam_basis_dim": len(basis),
            "active_decontam_basis_source": basis_source,
        }
    )
    return evidence, debug


def index_rows(rows: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for row in rows:
        sample_id = safe_text(row.get("sample_id"))
        if sample_id:
            out[sample_id] = dict(row)
    return out


def carrier_rows_by_sample(rows: Sequence[Mapping[str, Any]]) -> dict[str, dict[int, dict[str, Any]]]:
    out: dict[str, dict[int, dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        sample_id = safe_text(row.get("sample_id"))
        if not sample_id or row.get("layer") is None:
            continue
        out[sample_id][int(row["layer"])] = dict(row)
    return dict(out)


def merged_runtime_rows(
    *,
    manifest_rows: Sequence[Mapping[str, Any]],
    runtime_rows: Sequence[Mapping[str, Any]],
    selected_rows: Sequence[Mapping[str, Any]],
    eval_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    manifest_by_id = index_rows(manifest_rows)
    selected_by_id = index_rows(selected_rows)
    eval_by_id = index_rows(eval_rows)
    merged: list[dict[str, Any]] = []
    for runtime in runtime_rows:
        sample_id = safe_text(runtime.get("sample_id"))
        if not sample_id:
            continue
        manifest = manifest_by_id.get(sample_id, {})
        selected = selected_by_id.get(sample_id, {})
        sidecar = eval_by_id.get(sample_id, {})
        row: dict[str, Any] = {}
        row.update(selected)
        row.update(runtime)
        row.update(
            {
                "sample_id": sample_id,
                "manifest": manifest,
                "reference_answer": (
                    selected.get("reference_answer")
                    or sidecar.get("eval_reference_answer")
                    or runtime.get("reference_answer")
                ),
                "mad_protocol_task": (
                    selected.get("mad_protocol_task")
                    or sidecar.get("mad_protocol_task")
                    or manifest.get("mad_protocol_task")
                    or runtime.get("mad_protocol_task")
                ),
                "target_modality": manifest.get("target_modality") or runtime.get("target_modality"),
            }
        )
        merged.append(row)
    return merged


def _qwen2_last_query_attention_row_and_values(
    module: torch.nn.Module,
    *,
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor | None,
    position_embeddings: Any,
) -> tuple[torch.Tensor, torch.Tensor]:
    input_shape = hidden_states.shape[:-1]
    head_dim = int(getattr(module, "head_dim"))
    hidden_shape = (*input_shape, -1, head_dim)
    query_states = module.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
    key_states = module.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
    value_states = module.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
    cos, sin = position_embeddings
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
    query_states = query_states.to(torch.float32)
    key_states = key_states.to(torch.float32)
    key_states = repeat_kv(key_states, int(getattr(module, "num_key_value_groups", 1)))
    value_states = repeat_kv(value_states, int(getattr(module, "num_key_value_groups", 1))).to(torch.float32)
    scaling = float(getattr(module, "scaling", head_dim ** -0.5))
    attn_row = torch.matmul(query_states[:, :, -1:, :], key_states.transpose(2, 3)) * scaling
    if attention_mask is not None:
        attn_row = attn_row + attention_mask[:, :, -1:, : attn_row.shape[-1]]
    attn_row = torch.nn.functional.softmax(attn_row, dim=-1, dtype=torch.float32)
    return attn_row.detach().float(), value_states.detach().float()


def expanded_full_inputs(
    *,
    model: Any,
    tokenizer: Any,
    processor: Mapping[str, Any],
    row: Mapping[str, Any],
    dtype: torch.dtype,
    prepare_retries: int = 0,
) -> dict[str, Any]:
    last_exc: Exception | None = None
    max_attempts = max(1, int(prepare_retries) + 1)
    for attempt in range(max_attempts):
        inputs = prepare_inputs(model=model, tokenizer=tokenizer, processor=processor, row=row, branch="full", dtype=dtype)
        input_ids = inputs["input_ids"]
        images = inputs["images"]
        mm_positions = torch.where(input_ids[0] < 0)[0]
        if int(mm_positions.numel()) <= 0:
            raise RuntimeError(f"cannot locate VideoLLaMA2 multimodal placeholder for {row.get('sample_id')}")
        mm_start = int(mm_positions[0].item())
        _ids, attention_mask, _past, inputs_embeds, _labels = model.prepare_inputs_labels_for_multimodal(
            input_ids=input_ids,
            attention_mask=inputs["attention_mask"],
            past_key_values=None,
            labels=None,
            images=images,
        )
        if attention_mask is None:
            attention_mask = torch.ones(inputs_embeds.shape[:2], device=inputs_embeds.device, dtype=torch.long)
        if bool(torch.isfinite(inputs_embeds.detach().float()).all().item()):
            break
        bad = int((~torch.isfinite(inputs_embeds.detach().float())).sum().item())
        last_exc = FloatingPointError(f"non-finite multimodal inputs_embeds: bad_entries={bad}")
        release_cuda_cache()
        if attempt + 1 >= max_attempts:
            raise last_exc
    else:
        raise last_exc if last_exc is not None else RuntimeError("unknown multimodal prepare failure")

    seq_len = int(inputs_embeds.shape[1])
    original_len_without_mm = int(input_ids.shape[1]) - int(mm_positions.numel())
    total_mm_len = seq_len - original_len_without_mm
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
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "inputs_embeds": inputs_embeds,
        "images": images,
        "prompt_question": inputs["prompt_question"],
        "spans": {
            "mm_start": int(mm_start),
            "seq_len": int(seq_len),
            "video_positions": video_positions,
            "audio_positions": audio_positions,
            "language_positions": language_positions,
            "video_len": int(video_len),
            "audio_len": int(audio_len),
            "language_len": int(language_positions.numel()),
        },
    }


def resolve_group_positions(spans: Mapping[str, Any], *, target_modality: str, group: str) -> torch.Tensor:
    target = safe_text(target_modality).lower()
    key = safe_text(group).lower()
    video = torch.as_tensor(spans["video_positions"]).detach().long()
    audio = torch.as_tensor(spans["audio_positions"]).detach().long()
    language = torch.as_tensor(spans["language_positions"]).detach().long()
    if key == "auto_target":
        positions = audio if target == "audio" else video
    elif key in {"auto_target_non_target", "auto_target_plus_non_target", "audio_video", "audio_visual"}:
        positions = torch.cat([video, audio]) if int(audio.numel()) else video
    elif key == "auto_non_target":
        positions = video if target == "audio" else audio
    elif key in {"video", "visual", "vision"}:
        positions = video
    elif key == "audio":
        positions = audio
    elif key == "language":
        positions = language
    elif key == "language_non_target":
        non_target = video if target == "audio" else audio
        positions = torch.cat([language, non_target])
    else:
        raise ValueError(f"unsupported token group={group!r} target_modality={target_modality!r}")
    if int(positions.numel()) <= 0:
        return torch.empty(0, dtype=torch.long)
    return positions.unique(sorted=True).detach().cpu().long()


def build_budget(payload: Mapping[str, Any]) -> dict[str, float]:
    budget = clamp01(payload.get("budget"), 0.0)
    base_budget = payload.get("base_budget")
    suppression = clamp01(base_budget if base_budget is not None else budget, budget)
    evidence = budget
    return {
        "conflict_budget": budget,
        "suppression_budget": suppression,
        "evidence_budget": evidence,
    }


def canonical_branch_key(value: Any) -> str:
    key = safe_text(value).lower()
    if key in {"audio", "audio_only"}:
        return "audio_only"
    if key in {"visual", "video", "vision", "visual_only", "video_only"}:
        return "visual_only"
    if key in {"text", "text_only"}:
        return "text_only"
    if key in {"full", "joint"}:
        return "full"
    return key


def branch_payload(record: Mapping[str, Any], branch: Any) -> Mapping[str, Any]:
    branches = record.get("branches") or {}
    if not isinstance(branches, Mapping):
        return {}
    key = canonical_branch_key(branch)
    payload = branches.get(key)
    if isinstance(payload, Mapping):
        return payload
    if key == "visual_only":
        payload = branches.get("video_only") or branches.get("visual")
    elif key == "audio_only":
        payload = branches.get("audio")
    elif key == "text_only":
        payload = branches.get("text")
    return payload if isinstance(payload, Mapping) else {}


def branch_answer(record: Mapping[str, Any], branch: Any) -> str | None:
    payload = branch_payload(record, branch)
    answer = normalize_yes_no(payload.get("answer") or payload.get("raw_output"))
    return answer


def branch_confidence_for_answer(record: Mapping[str, Any], branch: Any, answer: Any) -> float:
    payload = branch_payload(record, branch)
    label = normalize_yes_no(answer)
    if label == "Yes":
        return clamp01(payload.get("confidence_on_yes"), 0.0)
    if label == "No":
        return clamp01(payload.get("confidence_on_no"), 0.0)
    return 0.0


def branch_margin_for_answer(record: Mapping[str, Any], branch: Any, answer: Any) -> float | None:
    payload = branch_payload(record, branch)
    margin = payload.get("margin")
    if margin is None:
        return None
    return signed_margin(answer, finite_float(margin, 0.0))


def branch_raw_margin(record: Mapping[str, Any], branch: Any) -> float | None:
    payload = branch_payload(record, branch)
    margin = payload.get("margin")
    if margin is None:
        return None
    value = finite_float(margin, float("nan"))
    return value if math.isfinite(value) else None


def soft_conflict_candidate_from_record(
    *,
    manifest: Mapping[str, Any],
    record: Mapping[str, Any],
) -> dict[str, Any] | None:
    target_modality = safe_text(manifest.get("target_modality") or record.get("target_modality")).lower()
    target_branch = canonical_branch_key(manifest.get("target_branch") or record.get("target_branch") or target_modality)
    if target_branch not in {"audio_only", "visual_only"}:
        target_branch = "audio_only" if target_modality == "audio" else "visual_only" if target_modality == "visual" else ""
    if target_branch not in {"audio_only", "visual_only"}:
        return None
    non_target_branch = "visual_only" if target_branch == "audio_only" else "audio_only"
    target_answer = normalize_yes_no(branch_answer(record, target_branch) or manifest.get("candidate_answer_y"))
    full_answer = normalize_yes_no(branch_answer(record, "full") or manifest.get("current_answer_y0"))
    non_target_answer = normalize_yes_no(branch_answer(record, non_target_branch))
    text_answer = normalize_yes_no(branch_answer(record, "text_only"))
    if target_answer is None:
        return None

    source_answers = [
        ("full", full_answer),
        (non_target_branch, non_target_answer),
        ("text_only", text_answer),
    ]
    disagree_sources = [(name, ans) for name, ans in source_answers if ans in {"Yes", "No"} and ans != target_answer]
    if not disagree_sources:
        return None

    source_branch, source_answer = disagree_sources[0]
    target_support = branch_confidence_for_answer(record, target_branch, target_answer)
    full_support = branch_confidence_for_answer(record, "full", target_answer)
    non_target_support = branch_confidence_for_answer(record, non_target_branch, target_answer)
    text_support = branch_confidence_for_answer(record, "text_only", target_answer)
    source_support = branch_confidence_for_answer(record, source_branch, target_answer)
    target_margin = branch_margin_for_answer(record, target_branch, target_answer)
    full_margin = branch_margin_for_answer(record, "full", target_answer)
    disagree_count = len(disagree_sources)
    # Same shape as the original soft expansion: target-branch support wins over
    # full/current support, with branch disagreement as the coarse conflict signal.
    score = (
        float(target_support - full_support)
        + 0.50 * float(disagree_count)
        + max(0.0, float(target_support - max(non_target_support, text_support, source_support)))
    )
    return {
        "bucket": "soft",
        "reason": "derived_soft_runtime_branch_conflict_from_records",
        "score": float(score),
        "target_answer": target_answer,
        "source_answer": source_answer,
        "source_branch": source_branch,
        "target_branch": target_branch,
        "non_target_branch": non_target_branch,
        "target_support": float(target_support),
        "full_support": float(full_support),
        "non_target_support": float(non_target_support),
        "text_support": float(text_support),
        "source_support": float(source_support),
        "target_margin": target_margin,
        "full_margin": full_margin,
        "disagree_count": int(disagree_count),
        "full_answer": full_answer,
        "non_target_answer": non_target_answer,
        "text_answer": text_answer,
    }


def parse_soft_top_k(value: str, benchmark: str) -> int:
    text = safe_text(value).lower()
    if not text or text == "auto":
        return 208 if benchmark == "avh" else 92 if benchmark == "cmm" else 0
    return max(0, int(text))


def assign_frozen_granularity(
    *,
    rows: list[dict[str, Any]],
    records_by_id: Mapping[str, Mapping[str, Any]],
    benchmark: str,
    soft_top_k: str,
    soft_score_threshold: float | None,
    global_soft_candidates: Sequence[tuple[float, str, dict[str, Any]]] | None = None,
) -> dict[str, Any]:
    hard_states = {"target_vs_mad_baseline_conflict", "target_vs_current_conflict", "target_vs_full_branch_conflict"}
    soft_candidates: list[tuple[float, str, dict[str, Any]]] = []
    for row in rows:
        manifest = dict(row.get("manifest") or {})
        state = safe_text(manifest.get("conflict_state") or row.get("conflict_state"))
        current = normalize_yes_no(manifest.get("current_answer_y0") or row.get("baseline_answer"))
        candidate = normalize_yes_no(manifest.get("candidate_answer_y") or row.get("target_answer"))
        if state in hard_states and current is not None and candidate is not None and current != candidate:
            row["frozen_granularity_bucket"] = "hard"
            row["frozen_granularity_reason"] = f"{state}_strong_answer_contrast"
            continue
        record = records_by_id.get(safe_text(row.get("sample_id")), {})
        soft = soft_conflict_candidate_from_record(manifest=manifest, record=record) if record else None
        if soft is not None:
            soft_candidates.append((float(soft["score"]), safe_text(row.get("sample_id")), soft))
        row["frozen_granularity_bucket"] = "outside"
        row["frozen_granularity_reason"] = "outside_hard_soft_preserve_baseline"

    threshold = soft_score_threshold
    top_k = parse_soft_top_k(soft_top_k, benchmark)
    selected: dict[str, dict[str, Any]] = {}
    ranking_candidates = list(global_soft_candidates) if global_soft_candidates is not None else soft_candidates
    if threshold is not None:
        for score, sid, payload in ranking_candidates:
            if score >= float(threshold):
                selected[sid] = payload
    if top_k > 0:
        ranked = sorted(ranking_candidates, key=lambda item: (-item[0], item[1]))
        for _score, sid, payload in ranked[:top_k]:
            selected[sid] = payload

    for row in rows:
        sid = safe_text(row.get("sample_id"))
        payload = selected.get(sid)
        if payload is None or row.get("frozen_granularity_bucket") == "hard":
            continue
        row["frozen_granularity_bucket"] = "soft"
        row["frozen_granularity_reason"] = payload["reason"]
        row["frozen_soft_conflict_score"] = payload["score"]
        row["frozen_soft_target_answer"] = payload["target_answer"]
        row["frozen_soft_source_answer"] = payload["source_answer"]
        row["frozen_soft_source_branch"] = payload["source_branch"]
        row["frozen_soft_target_branch"] = payload["target_branch"]
        row["frozen_soft_non_target_branch"] = payload["non_target_branch"]
        row["frozen_soft_disagree_count"] = payload["disagree_count"]
        row["frozen_soft_target_support"] = payload["target_support"]
        row["frozen_soft_full_support"] = payload["full_support"]
        row["frozen_soft_non_target_support"] = payload["non_target_support"]
        row["frozen_soft_text_support"] = payload["text_support"]
        row["frozen_soft_source_support"] = payload["source_support"]
        row["frozen_soft_full_answer"] = payload["full_answer"]
        row["frozen_soft_non_target_answer"] = payload["non_target_answer"]
        row["frozen_soft_text_answer"] = payload["text_answer"]

    counts = Counter(safe_text(row.get("frozen_granularity_bucket")) for row in rows)
    selected_prefix_counts = Counter(safe_text(sid).split(":", 1)[0] for sid in selected)
    return {
        "policy": "frozen_hard_soft_outside",
        "soft_top_k": soft_top_k,
        "soft_top_k_resolved": int(top_k),
        "soft_score_threshold": threshold,
        "soft_candidate_count": len(soft_candidates),
        "global_soft_candidate_count": len(ranking_candidates),
        "selected_global_soft_count": len(selected),
        "selected_global_soft_id_prefix_counts": dict(selected_prefix_counts),
        "selected_soft_count": int(counts.get("soft", 0)),
        "bucket_counts": dict(counts),
    }


def soft_candidates_from_typing_dir(typing_dir: Path) -> list[tuple[float, str, dict[str, Any]]]:
    manifest_rows = read_jsonl(typing_dir / "manifest.jsonl")
    runtime_rows = read_jsonl(typing_dir / "runtime_rows.jsonl")
    selected_rows = read_jsonl(typing_dir / "selected_rows.jsonl")
    eval_rows = read_jsonl(typing_dir / "eval_sidecar.jsonl")
    records_by_id = index_rows(read_jsonl(typing_dir / "records.jsonl"))
    rows = merged_runtime_rows(
        manifest_rows=manifest_rows,
        runtime_rows=runtime_rows,
        selected_rows=selected_rows,
        eval_rows=eval_rows,
    )
    hard_states = {"target_vs_mad_baseline_conflict", "target_vs_current_conflict", "target_vs_full_branch_conflict"}
    out: list[tuple[float, str, dict[str, Any]]] = []
    for row in rows:
        manifest = dict(row.get("manifest") or {})
        state = safe_text(manifest.get("conflict_state") or row.get("conflict_state"))
        current = normalize_yes_no(manifest.get("current_answer_y0") or row.get("baseline_answer"))
        candidate = normalize_yes_no(manifest.get("candidate_answer_y") or row.get("target_answer"))
        if state in hard_states and current is not None and candidate is not None and current != candidate:
            continue
        sid = safe_text(row.get("sample_id"))
        record = records_by_id.get(sid, {})
        payload = soft_conflict_candidate_from_record(manifest=manifest, record=record) if record else None
        if payload is not None:
            out.append((float(payload["score"]), sid, payload))
    return out


def adaptive_granularity_scales(
    *,
    signal: float | None,
    tau: float,
    temperature: float,
    hidden_floor: float,
    token_floor: float,
    reason: str = "runtime_full_margin",
) -> dict[str, Any]:
    hidden_floor_value = max(0.0, min(1.0, float(hidden_floor)))
    token_floor_value = max(0.0, min(1.0, float(token_floor)))
    if signal is None or not math.isfinite(float(signal)):
        return {
            "adaptive_gate": 0.0,
            "hidden_scale": 1.0,
            "token_scale": token_floor_value,
            "signal": None,
            "reason": "missing_signal",
        }
    z = (float(signal) - float(tau)) / max(1.0e-6, float(temperature))
    if z >= 60.0:
        gate = 0.0
    elif z <= -60.0:
        gate = 1.0
    else:
        gate = 1.0 / (1.0 + math.exp(z))
    gate = max(0.0, min(1.0, float(gate)))
    return {
        "adaptive_gate": gate,
        "hidden_scale": hidden_floor_value + (1.0 - hidden_floor_value) * (1.0 - gate),
        "token_scale": token_floor_value + (1.0 - token_floor_value) * gate,
        "signal": float(signal),
        "reason": reason,
    }


def target_modality_from_payloads(*payloads: Mapping[str, Any]) -> str:
    for payload in payloads:
        for key in (
            "repair_target_modality",
            "target_modality",
            "effective_target_modality",
            "online_target_modality",
        ):
            value = safe_text(payload.get(key)).lower()
            if value in {"audio", "audio_only"}:
                return "audio"
            if value in {"visual", "vision", "video", "visual_only", "video_only"}:
                return "visual"
        branch = canonical_branch_key(payload.get("target_branch"))
        if branch == "audio_only":
            return "audio"
        if branch == "visual_only":
            return "visual"
    return ""


def hard_modality_confidence_adjusted_signal(
    *,
    signal: float | None,
    row: Mapping[str, Any],
    manifest: Mapping[str, Any],
    record: Mapping[str, Any],
    mode: str,
    audio_center: float,
    visual_center: float,
    audio_weight: float,
    visual_weight: float,
) -> tuple[float | None, dict[str, Any]]:
    mode_key = safe_text(mode).strip().lower() or "off"
    debug: dict[str, Any] = {
        "mode": mode_key,
        "enabled": False,
        "base_signal": float(signal) if signal is not None and math.isfinite(float(signal)) else None,
    }
    if mode_key == "off":
        debug["reason"] = "disabled"
        return signal, debug
    if mode_key != "signal_shift":
        raise ValueError(f"unknown frozen hard modality confidence mode: {mode!r}")
    if signal is None or not math.isfinite(float(signal)):
        debug["reason"] = "missing_adaptive_signal"
        return signal, debug

    modality = target_modality_from_payloads(record, manifest, row)
    if modality not in {"audio", "visual"}:
        debug["reason"] = "missing_target_modality"
        return signal, debug

    confidence: float | None = None
    confidence_source = ""
    for payload in (record, manifest, row):
        for key in ("repair_target_confidence", "target_confidence", "target_support"):
            if key in payload:
                value = finite_float(payload.get(key), float("nan"))
                if math.isfinite(value):
                    confidence = max(0.0, min(1.0, float(value)))
                    confidence_source = key
                    break
        if confidence is not None:
            break

    if confidence is None and record:
        target_branch = canonical_branch_key(manifest.get("target_branch") or record.get("target_branch") or modality)
        target_answer = normalize_yes_no(manifest.get("candidate_answer_y") or row.get("target_answer") or record.get("candidate_answer_y"))
        if target_branch in {"audio_only", "visual_only"} and target_answer in {"Yes", "No"}:
            confidence = branch_confidence_for_answer(record, target_branch, target_answer)
            confidence_source = f"{target_branch}_confidence_for_candidate_answer"

    if confidence is None:
        debug["reason"] = "missing_modality_confidence"
        debug["target_modality"] = modality
        return signal, debug

    center = float(audio_center) if modality == "audio" else float(visual_center)
    weight = float(audio_weight) if modality == "audio" else float(visual_weight)
    adjustment = float(weight) * (float(confidence) - float(center))
    adjusted_signal = float(signal) + adjustment
    debug.update(
        {
            "enabled": True,
            "reason": "modality_confidence_signal_shift",
            "target_modality": modality,
            "confidence": float(confidence),
            "confidence_source": confidence_source,
            "center": float(center),
            "weight": float(weight),
            "signal_adjustment": float(adjustment),
            "adjusted_signal": float(adjusted_signal),
        }
    )
    return float(adjusted_signal), debug


def frozen_hard_adaptive_signal(
    *,
    row: Mapping[str, Any],
    manifest: Mapping[str, Any],
    record: Mapping[str, Any],
    source: str,
    baseline_margin: float | None,
) -> tuple[float | None, str]:
    source_key = safe_text(source).strip().lower() or "runtime_full_margin"
    if source_key == "runtime_full_margin":
        for payload in (record, manifest, row):
            value = branch_raw_margin(payload, "full")
            if value is not None:
                return float(value), "runtime_full_margin"
            for key in (
                "runtime_full_margin",
                "full_margin",
                "full_branch_margin",
                "full_context_margin",
                "baseline_margin",
            ):
                if key in payload:
                    candidate = finite_float(payload.get(key), float("nan"))
                    if math.isfinite(candidate):
                        return float(candidate), f"runtime_full_margin:{key}"
        return None, "runtime_full_margin_missing"
    if source_key == "candidate_aligned_baseline_margin":
        return (
            signed_margin(manifest.get("candidate_answer_y"), baseline_margin),
            "candidate_aligned_baseline_margin",
        )
    raise ValueError(f"unknown frozen hard adaptive signal source: {source!r}")


def adaptive_layer_weights(carriers: Mapping[int, Mapping[str, Any]]) -> tuple[dict[int, float], dict[int, float], dict[int, float]]:
    layers = list(FROZEN_LAYERS)
    group_weights = normalize_weights(layers, FROZEN_LAYER_GROUP_WEIGHTS)
    evidence_weights = adaptive_intervention_layer_weights(
        layers,
        FROZEN_EVIDENCE_LAYER_WEIGHTS,
        carriers,
        mode="evidence_lift",
        kind="evidence",
    ) or normalize_weights(layers, FROZEN_EVIDENCE_LAYER_WEIGHTS)
    prior_weights = adaptive_intervention_layer_weights(
        layers,
        FROZEN_PRIOR_LAYER_WEIGHTS,
        carriers,
        mode="conflict_bottleneck",
        kind="prior",
    ) or normalize_weights(layers, FROZEN_PRIOR_LAYER_WEIGHTS)
    alpha_weights = _token_head_alpha_weights_for_policy(
        policy="auto",
        operator="allpath_token_scaling",
        layers=layers,
        layer_weights=group_weights,
        evidence_layer_weights=evidence_weights,
        prior_layer_weights=prior_weights,
    )
    beta_total = sum(float(prior_weights.get(layer, 0.0)) for layer in layers)
    if beta_total <= 1.0e-12:
        beta_weights = normalize_weights(layers, FROZEN_PRIOR_LAYER_WEIGHTS)
    else:
        beta_weights = {layer: float(prior_weights.get(layer, 0.0)) / beta_total for layer in layers}
    return group_weights, alpha_weights, beta_weights


@torch.inference_mode()
def constrained_yesno_from_prepared(
    *,
    model: Any,
    tokenizer: Any,
    prepared: Mapping[str, Any],
) -> tuple[str, dict[str, Any]]:
    out = model(
        input_ids=None,
        attention_mask=prepared["attention_mask"],
        inputs_embeds=prepared["inputs_embeds"],
        use_cache=True,
        return_dict=True,
        pad_token_id=tokenizer.eos_token_id,
    )
    logits = out.logits[0, -1, :]
    yes_score = score_candidate_sequence(
        model=model,
        first_logits=logits,
        first_past=out.past_key_values,
        token_ids=candidate_token_ids(tokenizer, "Yes"),
    )
    no_score = score_candidate_sequence(
        model=model,
        first_logits=logits,
        first_past=out.past_key_values,
        token_ids=candidate_token_ids(tokenizer, "No"),
    )
    prediction = "Yes" if yes_score >= no_score else "No"
    return prediction, {
        "yes_score": float(yes_score),
        "no_score": float(no_score),
        "margin_yes_minus_no": float(yes_score - no_score),
        "decode_mode": "constrained_yesno",
    }


def has_yes_no_answer(tokenizer: Any, token_ids: Sequence[int]) -> bool:
    if not token_ids:
        return False
    text = tokenizer.decode(list(token_ids), skip_special_tokens=True)
    return bool(_YES_NO_PATTERN.search(text))


@torch.inference_mode()
def greedy_generate_from_prepared(
    *,
    model: Any,
    tokenizer: Any,
    prepared: Mapping[str, Any],
    max_new_tokens: int,
    yes_no_early_stop: bool = True,
) -> tuple[str, dict[str, Any]]:
    generated: list[int] = []
    out = model(
        input_ids=None,
        attention_mask=prepared["attention_mask"],
        inputs_embeds=prepared["inputs_embeds"],
        use_cache=True,
        return_dict=True,
        pad_token_id=tokenizer.eos_token_id,
    )
    logits = out.logits[0, -1, :]
    yes_score = score_candidate_sequence(
        model=model,
        first_logits=logits,
        first_past=out.past_key_values,
        token_ids=candidate_token_ids(tokenizer, "Yes"),
    )
    no_score = score_candidate_sequence(
        model=model,
        first_logits=logits,
        first_past=out.past_key_values,
        token_ids=candidate_token_ids(tokenizer, "No"),
    )
    past_key_values = out.past_key_values
    step_logits = logits
    stop_reason = "max_new_tokens"
    eos_id = tokenizer.eos_token_id
    for _ in range(max(0, int(max_new_tokens))):
        next_token = int(torch.argmax(step_logits, dim=-1).item())
        generated.append(next_token)
        if next_token == eos_id:
            stop_reason = "eos_token"
            break
        if yes_no_early_stop and has_yes_no_answer(tokenizer, generated):
            stop_reason = "yes_no_answer"
            break
        next_token_tensor = torch.tensor([[next_token]], device=prepared["inputs_embeds"].device, dtype=torch.long)
        out = model(
            input_ids=next_token_tensor,
            attention_mask=None,
            past_key_values=past_key_values,
            use_cache=True,
            return_dict=True,
            pad_token_id=tokenizer.eos_token_id,
        )
        past_key_values = out.past_key_values
        step_logits = out.logits[0, -1, :]
    text = tokenizer.decode(generated, skip_special_tokens=True).strip()
    return text, {
        "yes_score": float(yes_score),
        "no_score": float(no_score),
        "margin_yes_minus_no": float(yes_score - no_score),
        "decode_mode": "generate",
        "max_new_tokens": int(max_new_tokens),
        "yes_no_early_stop": bool(yes_no_early_stop),
        "generated_token_count": int(len(generated)),
        "stop_reason": stop_reason,
    }


def decode_from_prepared(
    *,
    model: Any,
    tokenizer: Any,
    prepared: Mapping[str, Any],
    decision_mode: str,
    max_new_tokens: int,
    evidence_prior_subspace_purification_lambda: float | None = None,
    yes_no_early_stop: bool = True,
) -> tuple[str, dict[str, Any]]:
    if decision_mode == "constrained_yesno":
        return constrained_yesno_from_prepared(model=model, tokenizer=tokenizer, prepared=prepared)
    if decision_mode == "generate":
        return greedy_generate_from_prepared(
            model=model,
            tokenizer=tokenizer,
            prepared=prepared,
            max_new_tokens=int(max_new_tokens),
            yes_no_early_stop=bool(yes_no_early_stop),
        )
    raise ValueError(f"unsupported decision_mode={decision_mode!r}")


@torch.inference_mode()
def frozen_structured_edit_yesno(
    *,
    model: Any,
    tokenizer: Any,
    processor: Mapping[str, Any],
    row: Mapping[str, Any],
    manifest: Mapping[str, Any],
    carrier_tensors: Mapping[int, Mapping[str, torch.Tensor]],
    carrier_rows: Mapping[int, Mapping[str, Any]],
    dtype: torch.dtype,
    beta: float,
    alpha: float,
    prepare_retries: int,
    decision_mode: str,
    max_new_tokens: int,
    evidence_prior_subspace_purification_lambda: float,
    hidden_update_mode: str = "apply",
    token_head_update_mode: str = "trace_only",
    token_head_value_operator: str = "allpath_token_scaling",
    prior_suppression_operator: str = "path_residual_minimal",
    evidence_transfer_mode: str = "path_answer_margin_minimal",
    hidden_strength_scale: float = 1.0,
    token_strength_scale: float = 1.0,
    max_patched_heads_per_layer: int = 16,
    head_answer_contribution_min: float = 0.0,
    yes_no_early_stop: bool = True,
) -> dict[str, Any]:
    current = normalize_yes_no(manifest.get("current_answer_y0") or row.get("baseline_answer"))
    candidate = normalize_yes_no(manifest.get("candidate_answer_y") or row.get("target_answer"))
    if current is None or candidate is None:
        raise ValueError(f"missing yes/no current/candidate for sample={row.get('sample_id')}")
    target_modality = safe_text(manifest.get("target_modality") or row.get("target_modality")).lower()
    if target_modality not in {"audio", "visual", "video", "vision"}:
        raise ValueError(f"unsupported target_modality={target_modality!r}")
    for layer in FROZEN_LAYERS:
        if int(layer) not in carrier_tensors:
            raise ValueError(f"missing carrier tensor for layer={layer}")
        for key in ("u_e", "u_p_perp", "evidence_transfer_vector"):
            if key not in carrier_tensors[int(layer)]:
                raise ValueError(f"missing carrier tensor key={key!r} layer={layer}")

    budget_payload = manifest.get("oelpr_conflict_budget") or row.get("oelpr_conflict_budget") or {}
    budgets = build_budget(budget_payload if isinstance(budget_payload, Mapping) else {})
    prepared = expanded_full_inputs(
        model=model,
        tokenizer=tokenizer,
        processor=processor,
        row=row,
        dtype=dtype,
        prepare_retries=prepare_retries,
    )
    source_positions = resolve_group_positions(
        prepared["spans"],
        target_modality=target_modality,
        group="auto_non_target",
    )
    evidence_positions = resolve_group_positions(
        prepared["spans"],
        target_modality=target_modality,
        group="auto_target_non_target",
    )
    if int(source_positions.numel()) <= 0:
        raise ValueError(f"empty source-prior token group for sample={row.get('sample_id')}")
    if int(evidence_positions.numel()) <= 0:
        raise ValueError(f"empty evidence token group for sample={row.get('sample_id')}")

    modules = language_layers(model)
    lm_head = model.get_output_embeddings() if hasattr(model, "get_output_embeddings") else getattr(model, "lm_head")
    if lm_head is None:
        lm_head = getattr(model, "lm_head")
    yes_id = yes_no_token_id(tokenizer, "Yes")
    no_id = yes_no_token_id(tokenizer, "No")
    group_weights, alpha_weights, beta_weights = adaptive_layer_weights(carrier_rows)

    evidence_source_mode = current_evidence_source_mode()
    evidence_validation_mode = current_evidence_validation_mode()
    evidence_directions: dict[int, torch.Tensor] = {}
    prior_directions: dict[int, torch.Tensor] = {}
    transfer_vectors: dict[int, torch.Tensor] = {}
    evidence_direction_debug: dict[str, dict[str, Any]] = {}
    for layer in FROZEN_LAYERS:
        payload = carrier_tensors[int(layer)]
        raw_evidence = torch.as_tensor(payload["u_e"]).detach().float().cpu()
        raw_prior = torch.as_tensor(payload.get("u_p_raw", payload["u_p_perp"])).detach().float().cpu()
        stored_prior = torch.as_tensor(payload["u_p_perp"]).detach().float().cpu()
        stored_transfer = torch.as_tensor(payload["evidence_transfer_vector"]).detach().float().cpu()
        raw_gap = finite_float(carrier_rows.get(layer, {}).get("evidence_transfer_gap"), float(stored_transfer.norm().item()))
        if evidence_source_mode == "clean":
            if "u_e_perp_prior" not in payload:
                raise ValueError(f"missing clean evidence tensor key='u_e_perp_prior' layer={layer}")
            clean_evidence = torch.as_tensor(payload["u_e_perp_prior"]).detach().float().cpu()
            if float(clean_evidence.norm().item()) <= 1.0e-12:
                clean_evidence = raw_evidence - vector_projection(raw_evidence, raw_prior)
            if float(clean_evidence.norm().item()) <= 1.0e-12:
                clean_evidence = raw_evidence
            clean_prior = raw_prior - vector_projection(raw_prior, clean_evidence)
            if float(clean_prior.norm().item()) <= 1.0e-12:
                clean_prior = stored_prior
            evidence = clean_evidence
            prior = clean_prior
            transfer = unit(clean_evidence) * float(raw_gap)
        else:
            evidence = raw_evidence
            prior = stored_prior
            transfer = stored_transfer
        evidence, validation_debug = apply_evidence_validation_from_tensors(
            evidence=evidence,
            raw_prior=raw_prior,
            payload=payload,
            mode=evidence_validation_mode,
            prior_subspace_purification_lambda=float(evidence_prior_subspace_purification_lambda),
        )
        if evidence_validation_mode != "none":
            prior = raw_prior - vector_projection(raw_prior, evidence)
            if float(prior.norm().item()) <= 1.0e-12:
                prior = stored_prior
            transfer = unit(evidence) * float(raw_gap)
        evidence_directions[int(layer)] = evidence
        prior_directions[int(layer)] = prior
        transfer_vectors[int(layer)] = transfer
        evidence_direction_debug[str(layer)] = {
            "layer": int(layer),
            "evidence_source_mode": evidence_source_mode,
            **validation_debug,
            "raw_evidence_norm": float(raw_evidence.norm().item()),
            "selected_evidence_norm": float(evidence.norm().item()),
            "raw_prior_norm": float(raw_prior.norm().item()),
            "selected_prior_norm": float(prior.norm().item()),
            "stored_prior_perp_norm": float(stored_prior.norm().item()),
            "raw_transfer_vector_l2": float(stored_transfer.norm().item()),
            "selected_transfer_vector_l2": float(transfer.norm().item()),
            "raw_evidence_raw_prior_cos": cosine(raw_evidence, raw_prior),
            "raw_evidence_stored_prior_cos": cosine(raw_evidence, stored_prior),
            "selected_evidence_selected_prior_cos": cosine(evidence, prior),
            "selected_evidence_raw_prior_cos": cosine(evidence, raw_prior),
        }
    evidence_gaps = {
        layer: finite_float(carrier_rows.get(layer, {}).get("evidence_transfer_gap"), float(transfer_vectors[layer].norm().item()))
        for layer in FROZEN_LAYERS
    }

    path_carriers_by_layer: dict[int, dict[str, Any]] = {}
    token_debug_by_layer: dict[str, dict[str, Any]] = {}
    hidden_debug_by_layer: dict[str, dict[str, Any]] = {}
    token_patch_by_layer: dict[int, dict[str, Any]] = {}
    hooks: list[Any] = []
    hidden_update_key = safe_text(hidden_update_mode) or "apply"
    token_update_key = safe_text(token_head_update_mode) or "trace_only"
    token_operator_key = safe_text(token_head_value_operator) or "allpath_token_scaling"
    prior_operator_key = safe_text(prior_suppression_operator) or "path_residual_minimal"
    evidence_transfer_key = safe_text(evidence_transfer_mode) or "path_answer_margin_minimal"
    if hidden_update_key not in {"apply", "diagnose_only"}:
        raise ValueError(f"unknown hidden_update_mode={hidden_update_mode!r}")
    if token_update_key not in {"apply", "trace_only"}:
        raise ValueError(f"unknown token_head_update_mode={token_head_update_mode!r}")
    if token_operator_key not in {"allpath_head_scaling", "allpath_token_scaling"}:
        raise ValueError(f"unknown token_head_value_operator={token_head_value_operator!r}")
    if prior_operator_key not in {"path_residual_minimal", "up_r_projection"}:
        raise ValueError(f"unknown prior_suppression_operator={prior_suppression_operator!r}")
    if evidence_transfer_key not in {"path_answer_margin_minimal", "legacy_projection_gap"}:
        raise ValueError(f"unknown evidence_transfer_mode={evidence_transfer_mode!r}")

    def make_trace_hook(layer_id: int):
        def hook_fn(module: torch.nn.Module, args: tuple[Any, ...], kwargs: Mapping[str, Any]):
            hidden_states = args[0] if args else kwargs.get("hidden_states")
            position_embeddings = kwargs.get("position_embeddings")
            if position_embeddings is None and len(args) >= 2:
                position_embeddings = args[1]
            attn_mask = kwargs.get("attention_mask")
            if attn_mask is None and len(args) >= 3:
                attn_mask = args[2]
            if hidden_states is None or position_embeddings is None or hidden_states.shape[1] <= 1:
                return None
            attn_row, value_states = _qwen2_last_query_attention_row_and_values(
                module,
                hidden_states=hidden_states,
                attention_mask=attn_mask,
                position_embeddings=position_embeddings,
            )
            row_before = attn_row[:, :, 0, :].float()
            source_mask, source_gate, source_debug = _token_selection_mask_for_answer_path(
                row_before,
                value_states,
                source_positions,
                module=module,
                lm_head=lm_head,
                yes_id=yes_id,
                no_id=no_id,
                positive_answer=current,
                negative_answer=candidate,
                mode="attention_topk_per_head",
                top_k=16,
                min_attn=0.0,
            )
            evidence_mask, evidence_gate, evidence_debug = _token_selection_mask_for_answer_path(
                row_before,
                value_states,
                evidence_positions,
                module=module,
                lm_head=lm_head,
                yes_id=yes_id,
                no_id=no_id,
                positive_answer=candidate,
                negative_answer=current,
                mode="answer_effect_soft_topk_per_head",
                top_k=16,
                min_attn=0.0,
            )
            source_mass = (row_before * source_gate.float()).sum(dim=-1).mean(dim=0).detach().cpu()
            evidence_mass = (row_before * evidence_gate.float()).sum(dim=-1).mean(dim=0).detach().cpu()
            source_contrib = _masked_value_contribution_from_mask(row_before, value_states, source_gate)
            evidence_contrib = _masked_value_contribution_from_mask(row_before, value_states, evidence_gate)
            original_head_output = torch.matmul(attn_row.float(), value_states.float())
            source_answer = _head_answer_contribution_scores(
                module=module,
                contribution=source_contrib,
                lm_head=lm_head,
                yes_id=yes_id,
                no_id=no_id,
                current_answer=current,
                candidate_answer=candidate,
            )
            evidence_answer = _head_answer_contribution_scores(
                module=module,
                contribution=evidence_contrib,
                lm_head=lm_head,
                yes_id=yes_id,
                no_id=no_id,
                current_answer=current,
                candidate_answer=candidate,
            )
            payload = _path_soft_hidden_carriers_from_token_heads(
                module=module,
                source_contribution=source_contrib,
                evidence_contribution=evidence_contrib,
                source_mass=source_mass,
                evidence_mass=evidence_mass,
                prior_direction=prior_directions[int(layer_id)],
                evidence_direction=evidence_directions[int(layer_id)],
                source_current_over_candidate=(
                    source_answer["current_minus_candidate"] if bool(source_answer.get("available")) else None
                ),
                evidence_candidate_over_current=(
                    evidence_answer["candidate_minus_current"] if bool(evidence_answer.get("available")) else None
                ),
            )
            path_carriers_by_layer[int(layer_id)] = payload
            token_patch_debug: dict[str, Any] = {}
            if token_update_key == "apply":
                answer_current_over_candidate = (
                    source_answer["current_minus_candidate"] if bool(source_answer.get("available")) else None
                )
                answer_candidate_over_current = (
                    evidence_answer["candidate_minus_current"] if bool(evidence_answer.get("available")) else None
                )
                selected_heads = _select_patch_heads(
                    source_mass=source_mass,
                    evidence_mass=evidence_mass,
                    mode="answer_contribution",
                    max_heads=int(max_patched_heads_per_layer),
                    min_source_margin=0.0,
                    min_source_mass=0.0,
                    seed=0,
                    answer_current_over_candidate=answer_current_over_candidate,
                    min_answer_contribution=float(head_answer_contribution_min),
                )
                if int(selected_heads.numel()) <= 0:
                    selected_heads = _select_patch_heads(
                        source_mass=source_mass,
                        evidence_mass=evidence_mass,
                        mode="source_mass_topk",
                        max_heads=int(max_patched_heads_per_layer),
                        min_source_margin=0.0,
                        min_source_mass=0.0,
                        seed=0,
                    )
                good_heads = _select_evidence_support_heads(
                    source_mass=source_mass,
                    evidence_mass=evidence_mass,
                    max_heads=int(max_patched_heads_per_layer),
                    exclude_heads=selected_heads,
                    answer_candidate_over_current=answer_candidate_over_current,
                    min_answer_contribution=float(head_answer_contribution_min),
                )
                beta_for_token = float(beta) * float(budgets["suppression_budget"]) * float(beta_weights.get(int(layer_id), 0.0)) * float(token_strength_scale)
                alpha_for_token = float(alpha) * float(budgets["evidence_budget"]) * float(alpha_weights.get(int(layer_id), 0.0)) * float(token_strength_scale)
                if token_operator_key == "allpath_head_scaling":
                    scaled = _allpath_head_scaled_output(
                        original_head_output=original_head_output,
                        hallu_heads=selected_heads,
                        good_heads=good_heads,
                        beta_for_layer=beta_for_token,
                        alpha_for_layer=alpha_for_token,
                        alpha_mode="evidence_value_boost",
                    )
                else:
                    if int(selected_heads.numel()) > 0:
                        selected_dev = selected_heads.to(device=row_before.device)
                        source_contrib_selected = _masked_value_contribution_from_mask(
                            row_before[:, selected_dev, :],
                            value_states[:, selected_dev, :, :],
                            source_gate[:, selected_dev, :],
                        )
                    else:
                        source_contrib_selected = torch.zeros(
                            row_before.shape[0],
                            0,
                            1,
                            value_states.shape[-1],
                            device=row_before.device,
                            dtype=torch.float32,
                        )
                    if int(good_heads.numel()) > 0:
                        good_dev = good_heads.to(device=row_before.device)
                        evidence_contrib_good = _masked_value_contribution_from_mask(
                            row_before[:, good_dev, :],
                            value_states[:, good_dev, :, :],
                            evidence_gate[:, good_dev, :],
                        )
                    else:
                        evidence_contrib_good = torch.zeros(
                            row_before.shape[0],
                            0,
                            1,
                            value_states.shape[-1],
                            device=row_before.device,
                            dtype=torch.float32,
                        )
                    scaled = _allpath_token_scaled_output(
                        original_head_output=original_head_output,
                        hallu_heads=selected_heads,
                        good_heads=good_heads,
                        source_contribution=source_contrib_selected,
                        evidence_contribution_good=evidence_contrib_good,
                        beta_for_layer=beta_for_token,
                        alpha_for_layer=alpha_for_token,
                        alpha_mode="evidence_value_boost",
                    )
                patched_head_output = torch.as_tensor(scaled["patched_head_output"]).to(
                    device=hidden_states.device,
                    dtype=torch.float32,
                )
                patched_flat = patched_head_output.transpose(1, 2).reshape(
                    patched_head_output.shape[0],
                    1,
                    patched_head_output.shape[1] * patched_head_output.shape[-1],
                )
                patched_attn_output = module.o_proj(patched_flat.to(dtype=hidden_states.dtype))
                token_patch_by_layer[int(layer_id)] = {
                    "attn_output_last": patched_attn_output.detach(),
                }
                token_patch_debug = {
                    "token_head_update_operator": token_operator_key,
                    "token_head_update_applied": True,
                    "token_beta_for_layer": float(beta_for_token),
                    "token_alpha_for_layer": float(alpha_for_token),
                    "token_strength_scale": float(token_strength_scale),
                    "selected_hallu_heads": [int(x) for x in selected_heads.detach().cpu().tolist()],
                    "selected_good_heads": [int(x) for x in good_heads.detach().cpu().tolist()],
                    **{k: v for k, v in scaled.items() if k not in {"patched_head_output", "delta"}},
                }
            token_debug_by_layer[str(layer_id)] = {
                "layer": int(layer_id),
                "token_head_update_mode": token_update_key,
                "token_head_update_applied": token_update_key == "apply",
                "source_token_count": int(source_positions.numel()),
                "evidence_token_count": int(evidence_positions.numel()),
                "source_mass_mean_all_heads": float(source_mass.float().mean().item()),
                "evidence_mass_mean_all_heads": float(evidence_mass.float().mean().item()),
                "source_token_selection": source_debug,
                "evidence_token_selection": evidence_debug,
                "source_answer_available": bool(source_answer.get("available")),
                "evidence_answer_available": bool(evidence_answer.get("available")),
                **token_patch_debug,
                **dict(payload.get("debug") or {}),
            }
            return None

        return hook_fn

    def make_token_apply_hook(layer_id: int):
        def hook_fn(_module: torch.nn.Module, _args: tuple[Any, ...], output: Any):
            if token_update_key != "apply":
                return None
            payload = token_patch_by_layer.pop(int(layer_id), None)
            if not payload:
                return None
            attn_output = output[0] if isinstance(output, (tuple, list)) else output
            if not isinstance(attn_output, torch.Tensor) or attn_output.shape[1] <= 0:
                return None
            patched = attn_output.clone()
            patch_last = torch.as_tensor(payload["attn_output_last"]).to(
                device=patched.device,
                dtype=patched.dtype,
            )
            patched[:, -1:, :] = patch_last
            if isinstance(output, tuple):
                return (patched,) + tuple(output[1:])
            if isinstance(output, list):
                return [patched] + list(output[1:])
            return patched

        return hook_fn

    def make_hidden_hook(layer_id: int):
        beta_weight = float(beta_weights.get(int(layer_id), 0.0))
        alpha_weight = float(alpha_weights.get(int(layer_id), 0.0))
        beta_for_layer = float(beta) * float(budgets["suppression_budget"]) * beta_weight * float(hidden_strength_scale)
        alpha_for_layer = float(alpha) * float(budgets["evidence_budget"]) * alpha_weight * float(hidden_strength_scale)
        p_cpu = prior_directions[int(layer_id)]
        e_cpu = evidence_directions[int(layer_id)]
        transfer_cpu = transfer_vectors[int(layer_id)]

        def hook_fn(_module: torch.nn.Module, _args: tuple[Any, ...], output: Any):
            hidden_states = output[0] if isinstance(output, (tuple, list)) else output
            if not isinstance(hidden_states, torch.Tensor):
                return None
            if hidden_states.shape[1] <= 1:
                return None
            patched = hidden_states.clone()
            base = patched[0, -1, :].float()
            p_dir = unit(p_cpu).to(device=base.device, dtype=torch.float32)
            e_dir = unit(e_cpu).to(device=base.device, dtype=torch.float32)
            original_p_dir = p_dir
            original_e_dir = e_dir
            transfer_vec = transfer_cpu.to(device=base.device, dtype=torch.float32)
            path_payload = path_carriers_by_layer.get(int(layer_id), {})
            path_debug = dict(path_payload.get("debug") or {})
            path_soft_prior_used = False
            path_soft_evidence_used = False
            path_soft_prior_blend = 0.0
            path_soft_evidence_blend = 0.0
            use_path_hidden = prior_operator_key == "path_residual_minimal" or evidence_transfer_key == "path_answer_margin_minimal"
            if use_path_hidden:
                path_prior = path_payload.get("prior_direction")
                if bool(path_payload.get("prior_available")) and path_prior is not None:
                    path_prior_tensor = torch.as_tensor(path_prior).to(device=base.device, dtype=torch.float32)
                    if float(path_prior_tensor.norm().detach().cpu().item()) > 1.0e-12:
                        path_prior_dir = unit(path_prior_tensor).to(device=base.device, dtype=torch.float32)
                        prior_alignment = torch.dot(path_prior_dir, original_p_dir).clamp(-1.0, 1.0)
                        path_soft_prior_blend = abs(float(prior_alignment.detach().cpu().item()))
                        oriented = path_prior_dir if float(prior_alignment.item()) >= 0.0 else -path_prior_dir
                        p_dir = unit((1.0 - path_soft_prior_blend) * original_p_dir + path_soft_prior_blend * oriented)
                        path_soft_prior_used = True
                path_evidence = path_payload.get("evidence_direction")
                if bool(path_payload.get("evidence_available")) and path_evidence is not None:
                    path_evidence_tensor = torch.as_tensor(path_evidence).to(device=base.device, dtype=torch.float32)
                    if float(path_evidence_tensor.norm().detach().cpu().item()) > 1.0e-12:
                        path_evidence_dir = unit(path_evidence_tensor).to(device=base.device, dtype=torch.float32)
                        evidence_alignment = torch.dot(path_evidence_dir, original_e_dir).clamp(-1.0, 1.0)
                        path_soft_evidence_blend = abs(float(evidence_alignment.detach().cpu().item()))
                        oriented = path_evidence_dir if float(evidence_alignment.item()) >= 0.0 else -path_evidence_dir
                        e_dir = unit((1.0 - path_soft_evidence_blend) * original_e_dir + path_soft_evidence_blend * oriented)
                        path_soft_evidence_used = True

            if prior_operator_key == "up_r_projection":
                up_r_payload = up_r_projection_delta(
                    base=base,
                    prior_direction=p_dir,
                    evidence_direction=e_dir,
                    beta=beta_for_layer,
                )
                prior_delta = torch.as_tensor(up_r_payload["delta"]).to(device=base.device, dtype=torch.float32)
                soft_payload = {
                    "mu": None,
                    "cos_prior_evidence": float(torch.dot(p_dir, e_dir).item()),
                    "delta_prior_projection": float(torch.dot(prior_delta, p_dir).item()),
                    "delta_evidence_projection": float(torch.dot(prior_delta, e_dir).item()),
                }
                path_prior_payload: dict[str, Any] = {}
            else:
                soft_payload = soft_minimal_suppression_delta(
                    base=base,
                    prior_direction=p_dir,
                    evidence_direction=e_dir,
                    beta=beta_for_layer,
                    evidence_lambda=0.0,
                )
                prior_delta = torch.as_tensor(soft_payload["delta"]).to(device=base.device, dtype=torch.float32)
                path_prior_payload = _path_constrained_hidden_delta_from_paths(
                    target_delta=prior_delta,
                    path_vectors=path_payload.get("prior_paths"),
                    path_scores=path_payload.get("prior_scores"),
                    prefix="prior",
                    residual_penalty=0.5,
                )
                if bool(path_prior_payload.get("available")):
                    prior_delta = torch.as_tensor(path_prior_payload["delta"]).to(device=base.device, dtype=torch.float32)

            if evidence_transfer_key == "legacy_projection_gap":
                evidence_delta = alpha_for_layer * max(0.0, float(evidence_gaps[int(layer_id)])) * e_dir
                evidence_payload = {
                    "path_answer_margin_available": False,
                    "path_answer_margin_reason": "legacy_projection_gap",
                    "path_answer_margin_active_count": 0,
                    "path_answer_margin_answer_gain": None,
                }
                path_evidence_payload: dict[str, Any] = {}
            else:
                evidence_payload = _path_answer_margin_evidence_delta(
                    base=base,
                    path_vectors=path_payload.get("evidence_paths"),
                    path_scores=path_payload.get("evidence_scores"),
                    lm_head=lm_head,
                    yes_id=yes_id,
                    no_id=no_id,
                    current_answer=current,
                    candidate_answer=candidate,
                    alpha_for_layer=alpha_for_layer,
                    solution_mode="least_norm_answer_axis",
                )
                evidence_delta = torch.as_tensor(evidence_payload["delta"]).to(device=base.device, dtype=torch.float32)
                path_evidence_payload = _path_constrained_hidden_delta_from_paths(
                    target_delta=evidence_delta,
                    path_vectors=path_payload.get("evidence_paths"),
                    path_scores=path_payload.get("evidence_scores"),
                    prefix="evidence",
                    residual_penalty=0.5,
                )
                if bool(path_evidence_payload.get("available")):
                    evidence_delta = torch.as_tensor(path_evidence_payload["delta"]).to(device=base.device, dtype=torch.float32)

            raw_update_delta = prior_delta + evidence_delta
            update_delta = raw_update_delta if hidden_update_key == "apply" else torch.zeros_like(raw_update_delta)
            updated = base + update_delta
            patched[0, -1, :] = updated.to(dtype=patched.dtype)
            hidden_debug_by_layer[str(layer_id)] = {
                "layer": int(layer_id),
                "layer_weight": float(group_weights.get(int(layer_id), 0.0)),
                "evidence_layer_weight": alpha_weight,
                "prior_layer_weight": beta_weight,
                "beta_for_layer": float(beta_for_layer),
                "alpha_for_layer": float(alpha_for_layer),
                "hidden_update_mode": hidden_update_key,
                "hidden_update_applied": hidden_update_key == "apply",
                "hidden_strength_scale": float(hidden_strength_scale),
                "hidden_beta_budget_scale": float(budgets["suppression_budget"]),
                "hidden_alpha_budget_scale": float(budgets["evidence_budget"]),
                "prior_suppression_operator": prior_operator_key,
                "evidence_transfer_mode": evidence_transfer_key,
                "evidence_source_mode": evidence_source_mode,
                "path_soft_prior_used": bool(path_soft_prior_used),
                "path_soft_evidence_used": bool(path_soft_evidence_used),
                "path_soft_prior_blend": float(path_soft_prior_blend),
                "path_soft_evidence_blend": float(path_soft_evidence_blend),
                **evidence_direction_debug.get(str(layer_id), {}),
                "prior_projection_before": float(torch.dot(base, p_dir).item()),
                "prior_projection_after": float(torch.dot(updated, p_dir).item()),
                "evidence_projection_before": float(torch.dot(base, e_dir).item()),
                "evidence_projection_after": float(torch.dot(updated, e_dir).item()),
                "soft_suppression_mu": soft_payload.get("mu"),
                "soft_suppression_cos_prior_evidence": soft_payload.get("cos_prior_evidence"),
                "soft_suppression_delta_prior_projection": soft_payload.get("delta_prior_projection"),
                "soft_suppression_delta_evidence_projection": soft_payload.get("delta_evidence_projection"),
                "prior_path_projected_available": bool(path_prior_payload.get("prior_path_projected_available")),
                "prior_path_projected_active_count": path_prior_payload.get("prior_path_projected_active_count"),
                "prior_path_projected_fit_cos": path_prior_payload.get("prior_path_projected_fit_cos"),
                "prior_path_projected_fit_ratio": path_prior_payload.get("prior_path_projected_fit_ratio"),
                "evidence_path_projected_available": bool(path_evidence_payload.get("evidence_path_projected_available")),
                "evidence_path_projected_active_count": path_evidence_payload.get("evidence_path_projected_active_count"),
                "evidence_path_projected_fit_cos": path_evidence_payload.get("evidence_path_projected_fit_cos"),
                "evidence_path_projected_fit_ratio": path_evidence_payload.get("evidence_path_projected_fit_ratio"),
                "path_answer_margin_available": bool(evidence_payload.get("path_answer_margin_available")),
                "path_answer_margin_reason": evidence_payload.get("path_answer_margin_reason"),
                "path_answer_margin_active_count": evidence_payload.get("path_answer_margin_active_count"),
                "path_answer_margin_answer_gain": evidence_payload.get("path_answer_margin_answer_gain"),
                "raw_update_l2": float(raw_update_delta.norm().item()),
                "prior_delta_l2": float(prior_delta.norm().item()),
                "evidence_delta_l2": float(evidence_delta.norm().item()),
                "delta_l2": float(update_delta.norm().item()),
                "base_norm": float(base.norm().item()),
                "updated_norm": float(updated.norm().item()),
                "evidence_gap": float(evidence_gaps[int(layer_id)]),
                "evidence_transfer_vector_l2": float(transfer_vec.norm().item()),
                **path_debug,
            }
            if isinstance(output, tuple):
                return (patched,) + tuple(output[1:])
            if isinstance(output, list):
                return [patched] + list(output[1:])
            return patched

        return hook_fn

    try:
        for layer in FROZEN_LAYERS:
            hooks.append(
                modules[int(layer)].self_attn.register_forward_pre_hook(
                    make_trace_hook(int(layer)),
                    with_kwargs=True,
                )
            )
            hooks.append(modules[int(layer)].self_attn.register_forward_hook(make_token_apply_hook(int(layer))))
            hooks.append(modules[int(layer)].register_forward_hook(make_hidden_hook(int(layer))))
        edited_prediction, edited_scores = decode_from_prepared(
            model=model,
            tokenizer=tokenizer,
            prepared=prepared,
            decision_mode=decision_mode,
            max_new_tokens=int(max_new_tokens),
            yes_no_early_stop=bool(yes_no_early_stop),
        )
    finally:
        for hook in hooks:
            hook.remove()

    debug_rows = list(hidden_debug_by_layer.values())
    return {
        "prediction": edited_prediction,
        "margin_yes_minus_no": edited_scores["margin_yes_minus_no"],
        "yes_score": edited_scores["yes_score"],
        "no_score": edited_scores["no_score"],
        "decode_mode": edited_scores.get("decode_mode"),
        "generated_token_count": edited_scores.get("generated_token_count"),
        "stop_reason": edited_scores.get("stop_reason"),
        "debug_by_layer": hidden_debug_by_layer,
        "token_debug_by_layer": token_debug_by_layer,
        "debug_layer_weights": {str(layer): float(group_weights.get(layer, 0.0)) for layer in FROZEN_LAYERS},
        "debug_evidence_layer_weights": {str(layer): float(alpha_weights.get(layer, 0.0)) for layer in FROZEN_LAYERS},
        "debug_prior_layer_weights": {str(layer): float(beta_weights.get(layer, 0.0)) for layer in FROZEN_LAYERS},
        "debug_evidence_source_mode": evidence_source_mode,
        "debug_evidence_direction_by_layer": evidence_direction_debug,
        "debug_num_layers_patched": len(debug_rows),
        "debug_delta_l2": sum(float(item.get("delta_l2") or 0.0) for item in debug_rows),
        "debug_prior_delta_l2": sum(float(item.get("prior_delta_l2") or 0.0) for item in debug_rows),
        "debug_evidence_delta_l2": sum(float(item.get("evidence_delta_l2") or 0.0) for item in debug_rows),
        "debug_token_head_delta_l2": sum(
            float(item.get("head_transport_delta_l2") or 0.0) for item in token_debug_by_layer.values()
        ),
        "debug_prior_path_projected_available_count": sum(
            1 for item in debug_rows if bool(item.get("prior_path_projected_available"))
        ),
        "debug_evidence_path_projected_available_count": sum(
            1 for item in debug_rows if bool(item.get("evidence_path_projected_available"))
        ),
        "debug_oelpr_conflict_budget": float(budgets["conflict_budget"]),
        "debug_oelpr_suppression_budget": float(budgets["suppression_budget"]),
        "debug_oelpr_evidence_budget": float(budgets["evidence_budget"]),
        "debug_source_prior_token_count": int(source_positions.numel()),
        "debug_evidence_token_count": int(evidence_positions.numel()),
        "span_debug": {
            "video_len": int(prepared["spans"]["video_len"]),
            "audio_len": int(prepared["spans"]["audio_len"]),
            "language_len": int(prepared["spans"]["language_len"]),
        },
    }


def apply_frozen_answer_preserve(
    *,
    raw_prediction: str,
    raw_margin: float,
    baseline_prediction: str | None,
    baseline_margin: float | None,
    budget_payload: Mapping[str, Any],
    suppression_budget: float,
    low_risk_answer_preserve_threshold: float,
    answer_change_cross_support_min: float,
) -> dict[str, Any]:
    threshold = max(0.0, min(1.0, float(low_risk_answer_preserve_threshold)))
    cross_support_min = max(0.0, min(1.0, float(answer_change_cross_support_min)))
    cross_support = budget_payload.get("candidate_reliability_cross_prob")
    cross_support_value = None if cross_support is None else finite_float(cross_support, float("nan"))
    low_risk_budget = suppression_budget <= threshold
    cross_support_low = cross_support_value is None or not math.isfinite(cross_support_value) or cross_support_value < cross_support_min
    eligible = bool(low_risk_budget or cross_support_low)
    edited_label = normalize_yes_no(raw_prediction)
    baseline_label = normalize_yes_no(baseline_prediction)
    active = bool(eligible and edited_label is not None and baseline_label is not None and edited_label != baseline_label)
    if active:
        return {
            "prediction": baseline_label,
            "margin": baseline_margin if baseline_margin is not None else raw_margin,
            "active": True,
            "reason": "cross_support_flip_reverted_to_baseline"
            if cross_support_low and not low_risk_budget
            else "low_risk_cross_support_flip_reverted_to_baseline"
            if cross_support_low and low_risk_budget
            else "low_risk_flip_reverted_to_baseline",
        }
    if eligible:
        reason = "cross_support_no_answer_flip" if cross_support_low and not low_risk_budget else "low_risk_no_answer_flip"
    else:
        reason = "budget_above_low_risk_threshold_and_cross_support_passed"
    return {
        "prediction": raw_prediction,
        "margin": raw_margin,
        "active": False,
        "reason": reason,
    }


def summarize(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    valid = [row for row in rows if not row.get("error")]

    def acc(prefix: str, group: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        eval_rows = [row for row in group if row.get(f"{prefix}_correct") is not None]
        total = len(eval_rows)
        correct = sum(1 for row in eval_rows if row.get(f"{prefix}_correct") is True)
        return {"correct": int(correct), "total": int(total), "accuracy": correct / total if total else None}

    baseline = acc("baseline", valid)
    raw = acc("raw_edited", valid)
    policy = acc("policy", valid)
    by_task: dict[str, Any] = {}
    for row in valid:
        key = safe_text(row.get("mad_protocol_task")) or "UNKNOWN"
        by_task.setdefault(key, []).append(row)
    by_task = {
        key: {
            "baseline": acc("baseline", group),
            "raw_edited": acc("raw_edited", group),
            "policy": acc("policy", group),
        }
        for key, group in sorted(by_task.items())
    }
    delta = None
    if baseline["accuracy"] is not None and policy["accuracy"] is not None:
        delta = 100.0 * (policy["accuracy"] - baseline["accuracy"])
    return {
        "n_rows": len(rows),
        "n_valid": len(valid),
        "n_errors": len(rows) - len(valid),
        "baseline": baseline,
        "raw_edited": raw,
        "policy": policy,
        "delta_pp": delta,
        "manifest_roles": dict(Counter(safe_text(row.get("manifest_role")) for row in valid)),
        "target_modalities": dict(Counter(safe_text(row.get("target_modality")) for row in valid)),
        "frozen_granularity_buckets": dict(Counter(safe_text(row.get("frozen_granularity_bucket")) for row in valid)),
        "preserve_reasons": dict(Counter(safe_text(row.get("answer_preserve_reason")) for row in valid)),
        "raw_flips": sum(1 for row in valid if row.get("raw_edited_prediction") != row.get("baseline_prediction")),
        "policy_flips": sum(1 for row in valid if row.get("policy_prediction") != row.get("baseline_prediction")),
        "by_task": by_task,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Strict frozen VideoLLaMA2 prior-carrier executor.")
    parser.add_argument("--benchmark", choices=["avh", "cmm"], default="avh")
    parser.add_argument("--typing-dir", type=Path, default=None)
    parser.add_argument("--manifest-path", type=Path, default=None)
    parser.add_argument("--runtime-rows-path", type=Path, default=None)
    parser.add_argument("--carrier-rows-path", type=Path, default=None)
    parser.add_argument("--carrier-tensors-path", type=Path, default=None)
    parser.add_argument("--selected-rows-path", type=Path, default=None)
    parser.add_argument("--eval-sidecar-path", type=Path, default=None)
    parser.add_argument("--records-path", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--model-path", type=str, default="DAMO-NLP-SG/VideoLLaMA2.1-7B-AV"
    )
    parser.add_argument("--max-rows", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--init-mode", choices=["eager", "sdpa", "official_flash"], default="eager")
    parser.add_argument("--dtype", choices=["float16", "bfloat16", "float32"], default="bfloat16")
    parser.add_argument("--beta", type=float, default=FROZEN_BETA)
    parser.add_argument("--alpha", type=float, default=FROZEN_ALPHA)
    parser.add_argument("--layers", type=str, default=None, help="Comma-separated frozen layer list, e.g. 16,20,24,26.")
    parser.add_argument("--layer-group-weights", type=str, default=None, help="Comma-separated layer:weight map.")
    parser.add_argument("--evidence-layer-weights", type=str, default=None, help="Comma-separated layer:weight map.")
    parser.add_argument("--prior-layer-weights", type=str, default=None, help="Comma-separated layer:weight map.")
    parser.add_argument(
        "--evidence-validation-mode",
        choices=["none", "cross_modal_self_logit_candidate_only_decontam"],
        default=None,
        help="Optional executor-side evidence validation/decontamination mode.",
    )
    parser.add_argument("--evidence-prior-subspace-purification-lambda", type=float, default=0.25)
    parser.add_argument("--decision-mode", choices=["constrained_yesno", "generate"], default="constrained_yesno")
    parser.add_argument("--max-new-tokens", type=int, default=1)
    parser.add_argument(
        "--disable-yesno-early-stop",
        action="store_true",
        help="For generate mode, continue until EOS or max_new_tokens instead of stopping after a yes/no answer.",
    )
    parser.add_argument("--oelpr-low-risk-answer-preserve-threshold", type=float, default=0.35)
    parser.add_argument("--oelpr-answer-change-cross-support-min", type=float, default=0.5)
    parser.add_argument("--oelpr-answer-change-cross-support-min-audio", type=float, default=None)
    parser.add_argument("--oelpr-answer-change-cross-support-min-visual", type=float, default=None)
    parser.add_argument(
        "--conflict-granularity-policy",
        choices=["legacy_candidate_safe_budget", "frozen_hard_soft_outside"],
        default="legacy_candidate_safe_budget",
    )
    parser.add_argument(
        "--frozen-soft-top-k",
        type=str,
        default="300",
        help="For frozen_hard_soft_outside, select derived soft rows by top-K. Tunable; auto uses AVH=208, CMM=92.",
    )
    parser.add_argument(
        "--global-soft-source",
        action="append",
        type=Path,
        default=[],
        help="Typing dir to include when ranking a global soft top-K across benchmarks; can be repeated.",
    )
    parser.add_argument(
        "--include-frozen-buckets",
        type=str,
        default="",
        help="Comma-separated frozen buckets to execute after granularity assignment, e.g. hard,soft. Empty runs all rows.",
    )
    parser.add_argument("--frozen-soft-score-threshold", type=float, default=None)
    parser.add_argument(
        "--frozen-hard-budget-mode",
        choices=["original", "unit"],
        default="original",
        help="For hard rows under frozen_hard_soft_outside, reuse the manifest OELPR budget or force unit budget.",
    )
    parser.add_argument("--frozen-hard-adaptive-tau", type=float, default=-0.5625)
    parser.add_argument("--frozen-hard-adaptive-temperature", type=float, default=0.10)
    parser.add_argument("--frozen-hard-adaptive-hidden-floor", type=float, default=0.0)
    parser.add_argument("--frozen-hard-adaptive-token-floor", type=float, default=0.0)
    parser.add_argument(
        "--frozen-hard-adaptive-signal-source",
        choices=["runtime_full_margin", "candidate_aligned_baseline_margin"],
        default="runtime_full_margin",
    )
    parser.add_argument(
        "--frozen-hard-modality-confidence-mode",
        choices=["off", "signal_shift"],
        default="off",
        help="Optionally shift the hard adaptive signal by target-modality confidence.",
    )
    parser.add_argument("--frozen-hard-modality-confidence-audio-center", type=float, default=0.55)
    parser.add_argument("--frozen-hard-modality-confidence-visual-center", type=float, default=0.55)
    parser.add_argument("--frozen-hard-modality-confidence-audio-weight", type=float, default=0.0)
    parser.add_argument("--frozen-hard-modality-confidence-visual-weight", type=float, default=0.50)
    parser.add_argument(
        "--frozen-hard-token-head-update-mode",
        choices=["trace_only", "apply"],
        default="apply",
        help="Hard rows follow the frozen OWP adaptive run: hidden apply plus token-head apply.",
    )
    parser.add_argument("--frozen-hard-token-head-strength", type=float, default=0.25)
    parser.add_argument(
        "--frozen-hard-prior-suppression-operator",
        choices=["up_r_projection", "path_residual_minimal"],
        default="up_r_projection",
    )
    parser.add_argument(
        "--frozen-hard-evidence-transfer-mode",
        choices=["legacy_projection_gap", "path_answer_margin_minimal"],
        default="legacy_projection_gap",
    )
    parser.add_argument(
        "--frozen-soft-adaptive-granularity-mode",
        choices=["off", "score_sigmoid"],
        default="off",
        help="For soft rows, optionally use the soft conflict score to trade hidden/full edit against token/local edit.",
    )
    parser.add_argument("--frozen-soft-adaptive-tau", type=float, default=1.88)
    parser.add_argument("--frozen-soft-adaptive-temperature", type=float, default=0.10)
    parser.add_argument("--frozen-soft-adaptive-hidden-max", type=float, default=0.25)
    parser.add_argument("--frozen-soft-token-head-strength", type=float, default=0.50)
    parser.add_argument("--max-patched-heads-per-layer", type=int, default=16)
    parser.add_argument("--head-answer-contribution-min", type=float, default=0.0)
    parser.add_argument("--warmup-rows-path", type=Path, default=None)
    parser.add_argument("--warmup-max-rows", type=int, default=1)
    parser.add_argument("--warmup-retries", type=int, default=2)
    parser.add_argument("--allow-warmup-errors", action="store_true")
    parser.add_argument("--prepare-retries", type=int, default=0)
    parser.add_argument("--fail-on-errors", action="store_true")
    parser.add_argument("--no-progress", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.evidence_validation_mode is not None:
        os.environ["EVIDENCE_VALIDATION_MODE"] = str(args.evidence_validation_mode)
    configure_frozen_layers_from_args(args)
    paths = typing_paths(args)
    dtype = torch_dtype_from_name(args.dtype)
    manifest_rows = read_jsonl(paths["manifest"])
    runtime_rows = read_jsonl(paths["runtime"])
    selected_rows = read_jsonl(paths.get("selected", Path("__missing__")))
    eval_rows = read_jsonl(paths.get("eval_sidecar", Path("__missing__")))
    records_rows = read_jsonl(paths.get("records", Path("__missing__")))
    records_by_id = index_rows(records_rows)
    carrier_rows_all = read_jsonl(paths["carrier_rows"])
    carrier_rows_lookup = carrier_rows_by_sample(carrier_rows_all)
    carrier_tensor_lookup = load_carrier_tensors(paths["carrier_tensors"])
    rows_all = merged_runtime_rows(
        manifest_rows=manifest_rows,
        runtime_rows=runtime_rows,
        selected_rows=selected_rows,
        eval_rows=eval_rows,
    )
    if args.max_rows and args.max_rows > 0:
        rows_all = rows_all[: int(args.max_rows)]
    granularity_summary: dict[str, Any] = {
        "policy": str(args.conflict_granularity_policy),
        "bucket_counts": {},
    }
    if args.conflict_granularity_policy == "frozen_hard_soft_outside":
        global_soft_candidates: list[tuple[float, str, dict[str, Any]]] | None = None
        if args.global_soft_source:
            global_soft_candidates = []
            for source_dir in args.global_soft_source:
                global_soft_candidates.extend(soft_candidates_from_typing_dir(Path(source_dir)))
        granularity_summary = assign_frozen_granularity(
            rows=rows_all,
            records_by_id=records_by_id,
            benchmark=str(args.benchmark),
            soft_top_k=str(args.frozen_soft_top_k),
            soft_score_threshold=args.frozen_soft_score_threshold,
            global_soft_candidates=global_soft_candidates,
        )
    include_frozen_buckets = {
        safe_text(item).strip().lower()
        for item in safe_text(args.include_frozen_buckets).split(",")
        if safe_text(item).strip()
    }
    if include_frozen_buckets:
        if args.conflict_granularity_policy != "frozen_hard_soft_outside":
            raise ValueError("--include-frozen-buckets requires --conflict-granularity-policy frozen_hard_soft_outside")
        allowed_buckets = {"hard", "soft", "outside"}
        unknown_buckets = sorted(include_frozen_buckets - allowed_buckets)
        if unknown_buckets:
            raise ValueError(f"unknown frozen buckets for --include-frozen-buckets: {unknown_buckets}")
        rows_all = [
            row
            for row in rows_all
            if safe_text(row.get("frozen_granularity_bucket")).strip().lower() in include_frozen_buckets
        ]
        granularity_summary["execution_bucket_filter"] = sorted(include_frozen_buckets)
        granularity_summary["execution_filtered_row_count"] = len(rows_all)
    rows = [row for idx, row in enumerate(rows_all) if idx % int(args.num_shards) == int(args.shard_index)]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows_path = args.output_dir / "rows.jsonl"
    rows_path.write_text("", encoding="utf-8")
    run_config = {
        "kind": "videollama2_strict_prior_carrier_executor_v1",
        "strict_contract": (
            "frozen_hard_soft_outside_adaptive_granularity"
            if args.conflict_granularity_policy == "frozen_hard_soft_outside"
            else "frozen_mainline_structured_subspace_path_residual_minimal_trace_only_candidate_safe_budget"
        ),
        "benchmark": args.benchmark,
        "typing_dir": str(args.typing_dir) if args.typing_dir is not None else None,
        "input_paths": {key: str(value) for key, value in paths.items()},
        "model_path": args.model_path,
        "torch_version": torch.__version__,
        "transformers_version": transformers.__version__,
        "dtype": args.dtype,
        "init_mode": args.init_mode,
        "include_frozen_buckets": sorted(include_frozen_buckets),
        "n_rows_all": len(rows_all),
        "n_rows_shard": len(rows),
        "n_records_rows": len(records_rows),
        "num_shards": int(args.num_shards),
        "shard_index": int(args.shard_index),
        "conflict_granularity_policy": str(args.conflict_granularity_policy),
        "conflict_granularity_summary": granularity_summary,
        "frozen_soft_top_k": str(args.frozen_soft_top_k),
        "frozen_soft_score_threshold": args.frozen_soft_score_threshold,
        "frozen_hard_budget_mode": str(args.frozen_hard_budget_mode),
        "frozen_hard_adaptive_tau": float(args.frozen_hard_adaptive_tau),
        "frozen_hard_adaptive_temperature": float(args.frozen_hard_adaptive_temperature),
        "frozen_hard_adaptive_hidden_floor": float(args.frozen_hard_adaptive_hidden_floor),
        "frozen_hard_adaptive_token_floor": float(args.frozen_hard_adaptive_token_floor),
        "frozen_hard_adaptive_signal_source": str(args.frozen_hard_adaptive_signal_source),
        "frozen_hard_modality_confidence_mode": str(args.frozen_hard_modality_confidence_mode),
        "frozen_hard_modality_confidence_audio_center": float(
            args.frozen_hard_modality_confidence_audio_center
        ),
        "frozen_hard_modality_confidence_visual_center": float(
            args.frozen_hard_modality_confidence_visual_center
        ),
        "frozen_hard_modality_confidence_audio_weight": float(
            args.frozen_hard_modality_confidence_audio_weight
        ),
        "frozen_hard_modality_confidence_visual_weight": float(
            args.frozen_hard_modality_confidence_visual_weight
        ),
        "frozen_hard_token_head_update_mode": str(args.frozen_hard_token_head_update_mode),
        "frozen_hard_token_head_strength": float(args.frozen_hard_token_head_strength),
        "frozen_hard_prior_suppression_operator": str(args.frozen_hard_prior_suppression_operator),
        "frozen_hard_evidence_transfer_mode": str(args.frozen_hard_evidence_transfer_mode),
        "frozen_soft_adaptive_granularity_mode": str(args.frozen_soft_adaptive_granularity_mode),
        "frozen_soft_adaptive_tau": float(args.frozen_soft_adaptive_tau),
        "frozen_soft_adaptive_temperature": float(args.frozen_soft_adaptive_temperature),
        "frozen_soft_adaptive_hidden_max": float(args.frozen_soft_adaptive_hidden_max),
        "frozen_soft_token_head_strength": float(args.frozen_soft_token_head_strength),
        "layers": FROZEN_LAYERS,
        "layer_group_weights": {str(layer): FROZEN_LAYER_GROUP_WEIGHTS[layer] for layer in FROZEN_LAYERS},
        "evidence_intervention_layer_weights": {str(k): v for k, v in FROZEN_EVIDENCE_LAYER_WEIGHTS.items()},
        "prior_intervention_layer_weights": {str(k): v for k, v in FROZEN_PRIOR_LAYER_WEIGHTS.items()},
        "adaptive_evidence_intervention_weights": "evidence_lift",
        "adaptive_prior_intervention_weights": "conflict_bottleneck",
        "evidence_source_mode": current_evidence_source_mode(),
        "evidence_validation_mode": current_evidence_validation_mode(),
        "evidence_prior_subspace_purification_lambda": float(args.evidence_prior_subspace_purification_lambda),
        "beta": float(args.beta),
        "alpha": float(args.alpha),
        "decision_mode": args.decision_mode,
        "max_new_tokens": int(args.max_new_tokens),
        "yes_no_early_stop": not bool(args.disable_yesno_early_stop),
        "official_decode_contract": (
            "generate uses greedy decoding with explicit max_new_tokens, do_sample=False semantics, "
            + (
                "and yes/no early stop matching the local MAD VideoLLaMA2 yes/no benchmark contract"
                if not args.disable_yesno_early_stop
                else "without yes/no early stop, matching the original free-generation protocol"
            )
            if args.decision_mode == "generate"
            else "constrained yes/no candidate scoring"
        ),
        "executor_mode": "structured_subspace",
        "prior_suppression_operator": {
            "hard": str(args.frozen_hard_prior_suppression_operator),
            "soft": "path_residual_minimal",
        },
        "evidence_transfer_mode": {
            "hard": str(args.frozen_hard_evidence_transfer_mode),
            "soft": "path_answer_margin_minimal",
        },
        "prior_protection": "orthogonalized",
        "structured_hidden_carrier_source": "branch",
        "structured_hidden_update_mode": (
            "bucket_dependent_hard_apply_soft_diagnose_only"
            if args.conflict_granularity_policy == "frozen_hard_soft_outside"
            else "apply"
        ),
        "structured_hidden_strength": 1.0,
        "structured_token_head_strength": {
            "hard": float(args.frozen_hard_token_head_strength),
            "soft": float(args.frozen_soft_token_head_strength),
        },
        "structured_token_head_update_mode": (
            "bucket_dependent_hard_soft_apply"
            if args.conflict_granularity_policy == "frozen_hard_soft_outside"
            else "trace_only"
        ),
        "structured_token_head_alpha_strength": 1.0,
        "token_head_value_operator": (
            {"hard": "allpath_token_scaling", "soft": "allpath_head_scaling"}
            if args.conflict_granularity_policy == "frozen_hard_soft_outside"
            else "allpath_token_scaling"
        ),
        "token_head_alpha_mode": "evidence_value_boost",
        "token_head_alpha_layer_policy": "auto",
        "max_patched_heads_per_layer": int(args.max_patched_heads_per_layer),
        "head_answer_contribution_min": float(args.head_answer_contribution_min),
        "source_prior_token_group": "auto_non_target",
        "evidence_token_group": "auto_target_non_target",
        "source_token_selection_mode": "attention_topk_per_head",
        "source_token_top_k": 16,
        "evidence_token_selection_mode": "answer_effect_soft_topk_per_head",
        "evidence_token_top_k": 16,
        "oelpr_conflict_budget_mode": "candidate_safe_evidence_budget",
        "oelpr_conflict_budget_scope": "all",
        "oelpr_low_risk_answer_preserve_threshold": float(args.oelpr_low_risk_answer_preserve_threshold),
        "oelpr_answer_change_cross_support_min": float(args.oelpr_answer_change_cross_support_min),
        "oelpr_answer_change_cross_support_min_audio": (
            None
            if args.oelpr_answer_change_cross_support_min_audio is None
            else float(args.oelpr_answer_change_cross_support_min_audio)
        ),
        "oelpr_answer_change_cross_support_min_visual": (
            None
            if args.oelpr_answer_change_cross_support_min_visual is None
            else float(args.oelpr_answer_change_cross_support_min_visual)
        ),
        "policy_mode": "always",
        "baseline_answer_source": (
            "final_baseline_generated_from_full_prompt; current_answer_y0 kept only as internal typing/current state"
            if args.decision_mode == "generate"
            else "current_answer_y0"
        ),
        "runtime_uses_reference_answer": False,
        "selection_uses_reference_answer": False,
        "selection_uses_task_family": False,
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
        "fail_on_errors": bool(args.fail_on_errors),
        "prepare_retries": int(args.prepare_retries),
    }
    write_json(args.output_dir / "run_config.json", run_config)
    write_jsonl(args.output_dir / "selected_rows.jsonl", rows)

    if args.init_mode == "official_flash":
        model, processor, tokenizer = model_init(
            args.model_path,
            device_map=torch.device("cuda"),
            use_flash_attn=True,
            torch_dtype=dtype,
        )
    else:
        model, processor, tokenizer = model_init(
            args.model_path,
            device_map=torch.device("cuda"),
            use_flash_attn=False,
            attn_implementation="eager" if args.init_mode == "eager" else "sdpa",
            torch_dtype=dtype,
        )
    model.eval()
    if max(FROZEN_LAYERS) >= len(language_layers(model)):
        raise ValueError(f"Layer index out of range for VideoLLaMA2: {FROZEN_LAYERS}")

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
                    prepared = expanded_full_inputs(
                        model=model,
                        tokenizer=tokenizer,
                        processor=processor,
                        row=warmup_row,
                        dtype=dtype,
                        prepare_retries=int(args.prepare_retries),
                    )
                    decode_from_prepared(
                        model=model,
                        tokenizer=tokenizer,
                        prepared=prepared,
                        decision_mode=args.decision_mode,
                        max_new_tokens=int(args.max_new_tokens),
                        yes_no_early_stop=not bool(args.disable_yesno_early_stop),
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

    outputs: list[dict[str, Any]] = []
    official_baseline_rows: list[dict[str, Any]] = []
    official_raw_rows: list[dict[str, Any]] = []
    official_policy_rows: list[dict[str, Any]] = []
    iterator = tqdm(rows, desc=f"vl2_strict_executor_s{args.shard_index}", disable=bool(args.no_progress), unit="sample")
    for row in iterator:
        start = time.time()
        sample_id = safe_text(row.get("sample_id"))
        manifest = dict(row.get("manifest") or {})
        reference = row.get("reference_answer")
        baseline_manifest_prediction = normalize_yes_no(manifest.get("current_answer_y0") or row.get("baseline_answer"))
        baseline_prediction = baseline_manifest_prediction
        baseline_raw_output = safe_text(baseline_prediction)
        raw_edited_output = ""
        baseline_margin = None
        baseline_decode_debug: dict[str, Any] = {}
        device_text = str(torch.device("cuda", torch.cuda.current_device())) if torch.cuda.is_available() else "cpu"
        bucket = safe_text(row.get("frozen_granularity_bucket")) or "unresolved"
        bucket_reason = safe_text(row.get("frozen_granularity_reason"))
        try:
            if not sample_id:
                raise ValueError("missing sample_id")
            if args.decision_mode == "generate":
                baseline_prepared = expanded_full_inputs(
                    model=model,
                    tokenizer=tokenizer,
                    processor=processor,
                    row=row,
                    dtype=dtype,
                    prepare_retries=int(args.prepare_retries),
                )
                baseline_raw_output, baseline_scores = decode_from_prepared(
                    model=model,
                    tokenizer=tokenizer,
                    prepared=baseline_prepared,
                    decision_mode=args.decision_mode,
                    max_new_tokens=int(args.max_new_tokens),
                    evidence_prior_subspace_purification_lambda=float(
                        args.evidence_prior_subspace_purification_lambda
                    ),
                    yes_no_early_stop=not bool(args.disable_yesno_early_stop),
                )
                baseline_prediction = normalize_yes_no(baseline_raw_output) or safe_text(baseline_raw_output)
                baseline_margin = float(baseline_scores["margin_yes_minus_no"])
                baseline_decode_debug = {
                    "raw_output": baseline_raw_output,
                    "decode_mode": baseline_scores.get("decode_mode"),
                    "max_new_tokens": baseline_scores.get("max_new_tokens"),
                    "generated_token_count": baseline_scores.get("generated_token_count"),
                    "stop_reason": baseline_scores.get("stop_reason"),
                    "yes_score": baseline_scores.get("yes_score"),
                    "no_score": baseline_scores.get("no_score"),
                }
            if sample_id not in carrier_tensor_lookup:
                raise ValueError(f"missing carrier_tensors for sample_id={sample_id}")
            if sample_id not in carrier_rows_lookup:
                raise ValueError(f"missing carrier_rows for sample_id={sample_id}")
            original_budget_payload = manifest.get("oelpr_conflict_budget") or row.get("oelpr_conflict_budget") or {}
            budget_payload = original_budget_payload if isinstance(original_budget_payload, Mapping) else {}
            effective_manifest = dict(manifest)
            bucket = "legacy"
            bucket_reason = "legacy_candidate_safe_budget"
            hidden_update_mode = "apply"
            token_update_mode = "trace_only"
            token_operator = "allpath_token_scaling"
            prior_operator = "path_residual_minimal"
            evidence_transfer_mode = "path_answer_margin_minimal"
            hidden_strength_scale = 1.0
            token_strength_scale = 1.0
            adaptive_scale_debug: dict[str, Any] = {}
            skip_answer_preserve_gate = False
            if args.conflict_granularity_policy == "frozen_hard_soft_outside":
                bucket = safe_text(row.get("frozen_granularity_bucket")) or "outside"
                bucket_reason = safe_text(row.get("frozen_granularity_reason")) or "frozen_granularity"
                unit_budget_payload = {
                    "budget": 1.0,
                    "base_budget": 1.0,
                    "budget_reason": f"frozen_{bucket}_unit_budget",
                }
                if bucket == "hard" and args.frozen_hard_budget_mode == "original" and budget_payload:
                    budget_payload = dict(budget_payload)
                    budget_payload["budget_reason"] = (
                        safe_text(budget_payload.get("budget_reason")) + "+frozen_hard_original_budget"
                    )
                elif bucket == "outside":
                    budget_payload = dict(budget_payload)
                else:
                    budget_payload = unit_budget_payload
                effective_manifest["oelpr_conflict_budget"] = budget_payload
                skip_answer_preserve_gate = True
                if bucket == "hard":
                    if (
                        str(args.frozen_hard_adaptive_signal_source) == "candidate_aligned_baseline_margin"
                        and baseline_margin is None
                    ):
                        baseline_score_prepared = expanded_full_inputs(
                            model=model,
                            tokenizer=tokenizer,
                            processor=processor,
                            row=row,
                            dtype=dtype,
                            prepare_retries=int(args.prepare_retries),
                        )
                        baseline_score_prediction, baseline_scores = decode_from_prepared(
                            model=model,
                            tokenizer=tokenizer,
                            prepared=baseline_score_prepared,
                            decision_mode="constrained_yesno",
                            max_new_tokens=1,
                            evidence_prior_subspace_purification_lambda=float(
                                args.evidence_prior_subspace_purification_lambda
                            ),
                            yes_no_early_stop=True,
                        )
                        baseline_margin = float(baseline_scores["margin_yes_minus_no"])
                        scored_baseline_prediction = normalize_yes_no(baseline_score_prediction) or safe_text(
                            baseline_score_prediction
                        )
                        if baseline_prediction is None:
                            baseline_prediction = scored_baseline_prediction
                        baseline_decode_debug = {
                            "raw_output": baseline_score_prediction,
                            "scored_prediction": scored_baseline_prediction,
                            "decode_mode": baseline_scores.get("decode_mode"),
                            "used_for": "frozen_hard_adaptive_margin",
                            "yes_score": baseline_scores.get("yes_score"),
                            "no_score": baseline_scores.get("no_score"),
                        }
                    effective_manifest["conflict_state"] = "target_vs_current_conflict"
                    effective_manifest["conflict_strength"] = "strong"
                    hidden_update_mode = "apply"
                    token_update_mode = str(args.frozen_hard_token_head_update_mode)
                    token_operator = "allpath_token_scaling"
                    prior_operator = str(args.frozen_hard_prior_suppression_operator)
                    evidence_transfer_mode = str(args.frozen_hard_evidence_transfer_mode)
                    adaptive_signal, adaptive_signal_reason = frozen_hard_adaptive_signal(
                        row=row,
                        manifest=effective_manifest,
                        record=records_by_id.get(sample_id, {}),
                        source=str(args.frozen_hard_adaptive_signal_source),
                        baseline_margin=baseline_margin,
                    )
                    adjusted_adaptive_signal, modality_confidence_debug = hard_modality_confidence_adjusted_signal(
                        signal=adaptive_signal,
                        row=row,
                        manifest=effective_manifest,
                        record=records_by_id.get(sample_id, {}),
                        mode=str(args.frozen_hard_modality_confidence_mode),
                        audio_center=float(args.frozen_hard_modality_confidence_audio_center),
                        visual_center=float(args.frozen_hard_modality_confidence_visual_center),
                        audio_weight=float(args.frozen_hard_modality_confidence_audio_weight),
                        visual_weight=float(args.frozen_hard_modality_confidence_visual_weight),
                    )
                    if modality_confidence_debug.get("enabled"):
                        adaptive_signal_reason = f"{adaptive_signal_reason}+modality_confidence_signal_shift"
                    adaptive_scale_debug = adaptive_granularity_scales(
                        signal=adjusted_adaptive_signal,
                        tau=float(args.frozen_hard_adaptive_tau),
                        temperature=float(args.frozen_hard_adaptive_temperature),
                        hidden_floor=float(args.frozen_hard_adaptive_hidden_floor),
                        token_floor=float(args.frozen_hard_adaptive_token_floor),
                        reason=adaptive_signal_reason,
                    )
                    adaptive_scale_debug["base_signal"] = (
                        float(adaptive_signal)
                        if adaptive_signal is not None and math.isfinite(float(adaptive_signal))
                        else None
                    )
                    adaptive_scale_debug["modality_confidence"] = modality_confidence_debug
                    hidden_strength_scale = float(adaptive_scale_debug["hidden_scale"])
                    token_strength_scale = float(args.frozen_hard_token_head_strength) * float(
                        adaptive_scale_debug["token_scale"]
                    )
                elif bucket == "soft":
                    effective_manifest["current_answer_y0"] = row.get("frozen_soft_source_answer")
                    effective_manifest["candidate_answer_y"] = row.get("frozen_soft_target_answer")
                    effective_manifest["target_answer"] = row.get("frozen_soft_target_answer")
                    effective_manifest["target_branch"] = row.get("frozen_soft_target_branch")
                    effective_manifest["non_target_branch"] = row.get("frozen_soft_non_target_branch")
                    effective_manifest["conflict_state"] = "soft_runtime_branch_conflict"
                    effective_manifest["conflict_strength"] = row.get("frozen_soft_conflict_score")
                    effective_manifest["soft_conflict_score"] = row.get("frozen_soft_conflict_score")
                    effective_manifest["soft_conflict_cur_support"] = row.get("frozen_soft_source_support")
                    effective_manifest["soft_conflict_alt_support"] = row.get("frozen_soft_target_support")
                    hidden_update_mode = "diagnose_only"
                    token_update_mode = "apply"
                    token_operator = "allpath_head_scaling"
                    prior_operator = "path_residual_minimal"
                    evidence_transfer_mode = "path_answer_margin_minimal"
                    hidden_strength_scale = 0.0
                    token_strength_scale = float(args.frozen_soft_token_head_strength)
                    if str(args.frozen_soft_adaptive_granularity_mode) == "score_sigmoid":
                        soft_adaptive_signal = finite_float(row.get("frozen_soft_conflict_score"), float("nan"))
                        if not math.isfinite(soft_adaptive_signal):
                            soft_adaptive_signal = None
                        adaptive_scale_debug = adaptive_granularity_scales(
                            signal=soft_adaptive_signal,
                            tau=float(args.frozen_soft_adaptive_tau),
                            temperature=float(args.frozen_soft_adaptive_temperature),
                            hidden_floor=0.0,
                            token_floor=0.0,
                            reason="frozen_soft_conflict_score_conflict_strength",
                        )
                        hidden_update_mode = "apply"
                        hidden_strength_scale = max(0.0, float(args.frozen_soft_adaptive_hidden_max)) * float(
                            adaptive_scale_debug["hidden_scale"]
                        )
                        token_strength_scale = float(args.frozen_soft_token_head_strength) * float(
                            adaptive_scale_debug["token_scale"]
                        )

            budgets = build_budget(budget_payload)
            current_label = normalize_yes_no(effective_manifest.get("current_answer_y0") or row.get("baseline_answer"))
            candidate_label = normalize_yes_no(effective_manifest.get("candidate_answer_y") or row.get("target_answer"))
            zero_budget_preserve = bool(
                float(budgets["suppression_budget"]) <= 1.0e-12
                and float(budgets["evidence_budget"]) <= 1.0e-12
            )
            no_candidate_contrast = bool(
                current_label is not None and candidate_label is not None and current_label == candidate_label
            )
            frozen_outside_preserve = bool(
                args.conflict_granularity_policy == "frozen_hard_soft_outside" and bucket == "outside"
            )
            legacy_preserve = bool(
                args.conflict_granularity_policy == "legacy_candidate_safe_budget"
                and (zero_budget_preserve or no_candidate_contrast)
            )
            if frozen_outside_preserve or legacy_preserve:
                raw_prediction = baseline_prediction
                raw_edited_output = baseline_raw_output
                raw_margin = baseline_margin
                policy_prediction = baseline_prediction
                policy_margin = baseline_margin
                preserved = {
                    "active": False,
                    "reason": "outside_hard_soft_preserve_baseline"
                    if frozen_outside_preserve
                    else "zero_candidate_safe_budget_preserve_baseline"
                    if zero_budget_preserve
                    else "no_candidate_current_contrast_preserve_baseline",
                }
                edit = {
                    "debug_by_layer": {},
                    "token_debug_by_layer": {},
                    "debug_layer_weights": {str(layer): FROZEN_LAYER_GROUP_WEIGHTS[layer] for layer in FROZEN_LAYERS},
                    "debug_evidence_layer_weights": {str(layer): 0.0 for layer in FROZEN_LAYERS},
                    "debug_prior_layer_weights": {str(layer): 0.0 for layer in FROZEN_LAYERS},
                    "debug_evidence_source_mode": current_evidence_source_mode(),
                    "debug_evidence_direction_by_layer": {},
                    "debug_delta_l2": 0.0,
                    "debug_prior_delta_l2": 0.0,
                    "debug_evidence_delta_l2": 0.0,
                    "debug_token_head_delta_l2": 0.0,
                    "debug_prior_path_projected_available_count": 0,
                    "debug_evidence_path_projected_available_count": 0,
                    "span_debug": {},
                }
            else:
                edit = frozen_structured_edit_yesno(
                    model=model,
                    tokenizer=tokenizer,
                    processor=processor,
                    row=row,
                    manifest=effective_manifest,
                    carrier_tensors=carrier_tensor_lookup[sample_id],
                    carrier_rows=carrier_rows_lookup[sample_id],
                    dtype=dtype,
                    beta=float(args.beta),
                    alpha=float(args.alpha),
                    prepare_retries=int(args.prepare_retries),
                    decision_mode=args.decision_mode,
                    max_new_tokens=int(args.max_new_tokens),
                    evidence_prior_subspace_purification_lambda=float(
                        args.evidence_prior_subspace_purification_lambda
                    ),
                    hidden_update_mode=hidden_update_mode,
                    token_head_update_mode=token_update_mode,
                    token_head_value_operator=token_operator,
                    prior_suppression_operator=prior_operator,
                    evidence_transfer_mode=evidence_transfer_mode,
                    hidden_strength_scale=hidden_strength_scale,
                    token_strength_scale=token_strength_scale,
                    max_patched_heads_per_layer=int(args.max_patched_heads_per_layer),
                    head_answer_contribution_min=float(args.head_answer_contribution_min),
                    yes_no_early_stop=not bool(args.disable_yesno_early_stop),
                )
                raw_edited_output = safe_text(edit["prediction"])
                raw_prediction = normalize_yes_no(raw_edited_output) or raw_edited_output
                raw_margin = float(edit["margin_yes_minus_no"])
                if skip_answer_preserve_gate:
                    preserved = {
                        "prediction": raw_prediction,
                        "margin": raw_margin,
                        "active": False,
                        "reason": "frozen_granularity_raw_policy_no_cross_support_gate",
                    }
                else:
                    answer_change_cross_support_min = float(args.oelpr_answer_change_cross_support_min)
                    target_modality_key = safe_text(manifest.get("target_modality") or row.get("target_modality")).lower()
                    if target_modality_key == "audio" and args.oelpr_answer_change_cross_support_min_audio is not None:
                        answer_change_cross_support_min = float(args.oelpr_answer_change_cross_support_min_audio)
                    elif target_modality_key == "visual" and args.oelpr_answer_change_cross_support_min_visual is not None:
                        answer_change_cross_support_min = float(args.oelpr_answer_change_cross_support_min_visual)
                    preserved = apply_frozen_answer_preserve(
                        raw_prediction=safe_text(raw_prediction),
                        raw_margin=raw_margin,
                        baseline_prediction=baseline_prediction,
                        baseline_margin=baseline_margin,
                        budget_payload=budget_payload,
                        suppression_budget=float(budgets["suppression_budget"]),
                        low_risk_answer_preserve_threshold=float(args.oelpr_low_risk_answer_preserve_threshold),
                        answer_change_cross_support_min=answer_change_cross_support_min,
                    )
                policy_prediction = normalize_yes_no(preserved["prediction"]) or safe_text(preserved["prediction"])
                policy_margin = float(preserved["margin"])
            runtime_seconds = time.time() - start
            out = {
                "sample_id": sample_id,
                "video_id": official_video_id(row),
                "benchmark": args.benchmark,
                "mad_protocol_task": row.get("mad_protocol_task"),
                "target_modality": manifest.get("target_modality") or row.get("target_modality"),
                "manifest_role": manifest.get("manifest_role"),
                "conflict_granularity_policy": str(args.conflict_granularity_policy),
                "frozen_granularity_bucket": bucket,
                "frozen_granularity_reason": bucket_reason,
                "effective_current_answer_y0": effective_manifest.get("current_answer_y0"),
                "effective_candidate_answer_y": effective_manifest.get("candidate_answer_y"),
                "effective_conflict_state": effective_manifest.get("conflict_state"),
                "effective_conflict_strength": effective_manifest.get("conflict_strength"),
                "effective_hidden_update_mode": hidden_update_mode,
                "effective_token_head_update_mode": token_update_mode,
                "effective_token_head_value_operator": token_operator,
                "effective_prior_suppression_operator": prior_operator,
                "effective_evidence_transfer_mode": evidence_transfer_mode,
                "effective_hidden_strength_scale": hidden_strength_scale,
                "effective_token_strength_scale": token_strength_scale,
                "adaptive_granularity_scale": adaptive_scale_debug,
                "frozen_soft_conflict_score": row.get("frozen_soft_conflict_score"),
                "frozen_soft_source_branch": row.get("frozen_soft_source_branch"),
                "frozen_soft_target_branch": row.get("frozen_soft_target_branch"),
                "question": row.get("question"),
                "video_path": row.get("video_path"),
                "audio_path": row.get("audio_path"),
                "reference_answer": reference,
                "current_answer_y0": manifest.get("current_answer_y0"),
                "candidate_answer_y": manifest.get("candidate_answer_y"),
                "baseline_manifest_prediction": baseline_manifest_prediction,
                "baseline_prediction": baseline_prediction,
                "baseline_raw_output": baseline_raw_output,
                "baseline_margin": baseline_margin,
                "baseline_decode_debug": baseline_decode_debug,
                "baseline_correct": safe_correct(baseline_prediction, reference),
                "raw_edited_prediction": raw_prediction,
                "raw_edited_output": raw_edited_output,
                "raw_edited_margin": raw_margin,
                "raw_edited_correct": safe_correct(raw_prediction, reference),
                "policy_prediction": policy_prediction,
                "policy_margin": policy_margin,
                "policy_correct": safe_correct(policy_prediction, reference),
                "edited_candidate_aligned_margin": signed_margin(effective_manifest.get("candidate_answer_y"), raw_margin),
                "policy_candidate_aligned_margin": signed_margin(effective_manifest.get("candidate_answer_y"), policy_margin),
                "edit_applied": not bool(frozen_outside_preserve or legacy_preserve),
                "policy_mode": "always",
                "answer_preserve_active": bool(preserved["active"]),
                "answer_preserve_reason": preserved["reason"],
                "oelpr_conflict_budget": budget_payload,
                "original_oelpr_conflict_budget": original_budget_payload,
                "oelpr_conflict_budget_value": budgets["conflict_budget"],
                "oelpr_suppression_budget": budgets["suppression_budget"],
                "oelpr_evidence_budget": budgets["evidence_budget"],
                "patch_debug_by_layer": edit["debug_by_layer"],
                "token_trace_debug_by_layer": edit["token_debug_by_layer"],
                "raw_edited_decode_mode": edit.get("decode_mode"),
                "raw_edited_generated_token_count": edit.get("generated_token_count"),
                "raw_edited_stop_reason": edit.get("stop_reason"),
                "debug_layer_weights": edit["debug_layer_weights"],
                "debug_evidence_layer_weights": edit["debug_evidence_layer_weights"],
                "debug_prior_layer_weights": edit["debug_prior_layer_weights"],
                "debug_evidence_source_mode": edit.get("debug_evidence_source_mode", current_evidence_source_mode()),
                "debug_evidence_direction_by_layer": edit.get("debug_evidence_direction_by_layer", {}),
                "debug_delta_l2": edit["debug_delta_l2"],
                "debug_prior_delta_l2": edit["debug_prior_delta_l2"],
                "debug_evidence_delta_l2": edit["debug_evidence_delta_l2"],
                "debug_token_head_delta_l2": edit.get("debug_token_head_delta_l2", 0.0),
                "debug_prior_path_projected_available_count": edit["debug_prior_path_projected_available_count"],
                "debug_evidence_path_projected_available_count": edit["debug_evidence_path_projected_available_count"],
                "span_debug": edit["span_debug"],
                "runtime_seconds": runtime_seconds,
                "runtime_uses_reference_answer": False,
                "selection_uses_reference_answer": False,
                "selection_uses_task_family": False,
                "error": "",
            }
            official_baseline_rows.append(
                official_cmm_score_row(row=row, prediction=baseline_prediction)
                if args.benchmark == "cmm"
                else official_score_row(row=row, prediction=baseline_prediction, runtime_seconds=runtime_seconds, device=device_text)
            )
            official_raw_rows.append(
                official_cmm_score_row(row=row, prediction=raw_prediction)
                if args.benchmark == "cmm"
                else official_score_row(row=row, prediction=raw_prediction, runtime_seconds=runtime_seconds, device=device_text)
            )
            official_policy_rows.append(
                official_cmm_score_row(row=row, prediction=policy_prediction)
                if args.benchmark == "cmm"
                else official_score_row(row=row, prediction=policy_prediction, runtime_seconds=runtime_seconds, device=device_text)
            )
        except Exception as exc:
            runtime_seconds = time.time() - start
            error_prediction = f"ERROR: {repr(exc)}"
            out = {
                "sample_id": sample_id,
                "video_id": official_video_id(row),
                "benchmark": args.benchmark,
                "mad_protocol_task": row.get("mad_protocol_task"),
                "target_modality": manifest.get("target_modality") or row.get("target_modality"),
                "frozen_granularity_bucket": bucket,
                "frozen_granularity_reason": bucket_reason,
                "question": row.get("question"),
                "reference_answer": reference,
                "baseline_prediction": baseline_prediction,
                "baseline_correct": safe_correct(baseline_prediction, reference),
                "runtime_seconds": runtime_seconds,
                "error": repr(exc),
            }
            official_baseline_rows.append(
                official_cmm_score_row(row=row, prediction=baseline_prediction or error_prediction)
                if args.benchmark == "cmm"
                else official_score_row(row=row, prediction=baseline_prediction or error_prediction, runtime_seconds=runtime_seconds, device=device_text)
            )
            official_raw_rows.append(
                official_cmm_score_row(row=row, prediction=error_prediction)
                if args.benchmark == "cmm"
                else official_score_row(row=row, prediction=error_prediction, runtime_seconds=runtime_seconds, device=device_text)
            )
            official_policy_rows.append(
                official_cmm_score_row(row=row, prediction=error_prediction)
                if args.benchmark == "cmm"
                else official_score_row(row=row, prediction=error_prediction, runtime_seconds=runtime_seconds, device=device_text)
            )
        append_jsonl(rows_path, out)
        outputs.append(out)
        release_cuda_cache()

    summary = summarize(outputs)
    summary["run_config"] = run_config
    write_json(args.output_dir / "summary.json", summary)
    if args.benchmark == "cmm":
        write_jsonl(args.output_dir / "official_baseline_results_cmm.jsonl", official_baseline_rows)
        write_jsonl(args.output_dir / "official_raw_edited_results_cmm.jsonl", official_raw_rows)
        write_jsonl(args.output_dir / "official_policy_results_cmm.jsonl", official_policy_rows)
    else:
        write_json(args.output_dir / "official_baseline_results_av.json", official_baseline_rows)
        write_json(args.output_dir / "official_raw_edited_results_av.json", official_raw_rows)
        write_json(args.output_dir / "official_policy_results_av.json", official_policy_rows)
    print(json.dumps(summary, ensure_ascii=False, indent=2)[:4000])
    if args.fail_on_errors and int(summary.get("n_errors") or 0) > 0:
        raise RuntimeError(f"strict executor produced errors: n_errors={summary.get('n_errors')}")


if __name__ == "__main__":
    main()
