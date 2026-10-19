#!/usr/bin/env python3
from __future__ import annotations

import gc
import math
import os
import sys
from pathlib import Path
from types import MethodType
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import torch
from transformers.models.qwen2_5_omni.modeling_qwen2_5_omni import (
    apply_multimodal_rotary_pos_emb,
    repeat_kv,
)


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
for path in (SCRIPT_DIR, ROOT, ROOT.parent):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from owp.models.qwen_omni import QwenOmniAdapter  # noqa: E402
from analyze_attention import (  # noqa: E402
    build_token_group_masks,
    formatted_question,
    safe_text,
    tokenizer_ids,
)


DEFAULT_PATTERN_BUCKETS = (
    "audio_no_to_yes_wrong",
    "audio_yes_to_no_wrong",
    "audio_no_to_no_correct",
    "audio_yes_to_yes_correct",
    "visual_no_to_yes_wrong",
    "visual_yes_to_no_wrong",
    "visual_no_to_no_correct",
    "visual_yes_to_yes_correct",
)
DEFAULT_TASK_FAMILIES: tuple[str, ...] = ()


def choose_rows(
    rows: Sequence[Dict[str, Any]],
    *,
    sample_id: str,
    max_samples: int,
    task_families: Sequence[str],
    pattern_buckets: Sequence[str],
) -> List[Dict[str, Any]]:
    chosen = [dict(row) for row in rows if not row.get("skipped")]
    if sample_id:
        chosen = [row for row in chosen if safe_text(row.get("sample_id")) == safe_text(sample_id)]
    # Method-purity contract:
    # runtime row selection must not filter on benchmark task_family metadata.
    # Keep the argument only for backward-compatible call sites.
    if pattern_buckets:
        allowed_buckets = {safe_text(x) for x in pattern_buckets if safe_text(x)}
        chosen = [row for row in chosen if safe_text(row.get("pattern_bucket")) in allowed_buckets]
    chosen.sort(key=lambda row: safe_text(row.get("sample_id")))
    if int(max_samples) > 0:
        chosen = chosen[: int(max_samples)]
    return chosen


def apply_shard(rows: Sequence[Dict[str, Any]], *, num_shards: int, shard_index: int) -> List[Dict[str, Any]]:
    total_shards = max(1, int(num_shards))
    shard = int(shard_index)
    if shard < 0 or shard >= total_shards:
        raise ValueError(f"invalid shard index {shard} for num_shards={total_shards}")
    if total_shards == 1:
        return [dict(row) for row in rows]
    return [dict(row) for idx, row in enumerate(rows) if int(idx) % total_shards == shard]


def norm_yes_no(value: Any) -> Optional[str]:
    text = safe_text(value).lower()
    if text.startswith("yes"):
        return "Yes"
    if text.startswith("no"):
        return "No"
    return None


def maybe_correct(prediction: Optional[str], reference_answer: Optional[str]) -> Optional[bool]:
    pred = norm_yes_no(prediction)
    ref = norm_yes_no(reference_answer)
    if pred is None or ref is None:
        return None
    return bool(pred == ref)


def actual_group(reference_answer: Optional[str], prediction: Optional[str]) -> str:
    ref = norm_yes_no(reference_answer)
    pred = norm_yes_no(prediction)
    if ref == "Yes" and pred == "No":
        return "yes_to_no_wrong"
    if ref == "Yes" and pred == "Yes":
        return "yes_to_yes_correct"
    if ref == "No" and pred == "No":
        return "no_to_no_correct"
    if ref == "No" and pred == "Yes":
        return "no_to_yes_wrong"
    return "unknown"


def mean(values: Iterable[Any]) -> Optional[float]:
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


def prepare_full_av_inputs(
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


def cache_multimodal_inputs_for_avcd_step(
    adapter: QwenOmniAdapter,
    inputs: Dict[str, torch.Tensor],
) -> Dict[str, torch.Tensor]:
    """Run deterministic media towers once for the four AVCD branches in a decode step."""
    thinker = adapter._model.thinker
    input_ids = inputs["input_ids"]
    attention_mask = inputs.get("attention_mask")
    with torch.inference_mode():
        inputs_embeds = thinker.get_input_embeddings()(input_ids)
        input_features = inputs.get("input_features")
        feature_attention_mask = inputs.get("feature_attention_mask")
        audio_feature_lengths = inputs.get("audio_feature_lengths")
        if input_features is not None:
            audio_features = thinker.get_audio_features(
                input_features,
                feature_attention_mask=feature_attention_mask,
                audio_feature_lengths=audio_feature_lengths,
            )
            audio_features = audio_features.to(inputs_embeds.device, inputs_embeds.dtype)
            _, _, audio_mask = thinker.get_placeholder_mask(input_ids, inputs_embeds=inputs_embeds)
            inputs_embeds = inputs_embeds.masked_scatter(audio_mask, audio_features)

        pixel_values = inputs.get("pixel_values")
        image_grid_thw = inputs.get("image_grid_thw")
        if pixel_values is not None:
            image_embeds = thinker.get_image_features(pixel_values, image_grid_thw)
            image_embeds = image_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
            image_mask, _, _ = thinker.get_placeholder_mask(
                input_ids,
                inputs_embeds=inputs_embeds,
                image_features=image_embeds,
            )
            inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)

        pixel_values_videos = inputs.get("pixel_values_videos")
        video_grid_thw = inputs.get("video_grid_thw")
        if pixel_values_videos is not None:
            video_embeds = thinker.get_video_features(pixel_values_videos, video_grid_thw)
            video_embeds = video_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
            _, video_mask, _ = thinker.get_placeholder_mask(
                input_ids,
                inputs_embeds=inputs_embeds,
                video_features=video_embeds,
            )
            inputs_embeds = inputs_embeds.masked_scatter(video_mask, video_embeds)

        effective_audio_lengths = (
            torch.sum(feature_attention_mask, dim=1)
            if feature_attention_mask is not None
            else None
        )
        position_ids = inputs.get("position_ids")
        if attention_mask is not None and position_ids is None:
            position_ids, _rope_deltas = thinker.get_rope_index(
                input_ids,
                image_grid_thw,
                video_grid_thw,
                attention_mask,
                inputs.get("use_audio_in_video"),
                effective_audio_lengths,
                inputs.get("video_second_per_grid"),
            )

    cached: Dict[str, torch.Tensor] = {
        "input_ids": input_ids,
        "inputs_embeds": inputs_embeds,
    }
    if attention_mask is not None:
        cached["attention_mask"] = attention_mask
    if position_ids is not None:
        cached["position_ids"] = position_ids
    return cached


def extend_inputs_for_generation(
    base_inputs: Dict[str, torch.Tensor],
    generated_token_ids: Sequence[int],
) -> Dict[str, torch.Tensor]:
    inputs: Dict[str, torch.Tensor] = {}
    append_ids = None
    if generated_token_ids:
        append_ids = torch.tensor(
            [list(int(token_id) for token_id in generated_token_ids)],
            dtype=base_inputs["input_ids"].dtype,
            device=base_inputs["input_ids"].device,
        )
    for key, value in base_inputs.items():
        if not torch.is_tensor(value):
            inputs[key] = value
            continue
        if key == "input_ids":
            inputs[key] = (
                torch.cat([value, append_ids], dim=1)
                if append_ids is not None
                else value
            )
            continue
        if key == "attention_mask":
            if append_ids is None:
                inputs[key] = value
            else:
                append_mask = torch.ones_like(append_ids, dtype=value.dtype, device=value.device)
                inputs[key] = torch.cat([value, append_mask], dim=1)
            continue
        inputs[key] = value
    if "attention_mask" not in inputs:
        seq_len = int(inputs["input_ids"].shape[1])
        inputs["attention_mask"] = torch.ones(
            (1, seq_len),
            dtype=inputs["input_ids"].dtype,
            device=inputs["input_ids"].device,
        )
    return inputs


def build_post_media_language_mask(masks: Dict[str, torch.Tensor]) -> torch.Tensor:
    valid = masks["valid"].detach().cpu().bool()
    media = masks["media"].detach().cpu().bool()
    seq_len = int(valid.shape[0])
    positions = torch.arange(seq_len, dtype=torch.long)
    if int(media.sum().item()) <= 0:
        return valid & ~media
    last_media_pos = int(torch.nonzero(media, as_tuple=False).flatten().max().item())
    return valid & ~media & (positions > last_media_pos)


def resolve_avcd_label_masks(
    *,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    special_ids: Dict[str, Optional[int]],
) -> Dict[str, torch.Tensor]:
    masks = build_token_group_masks(
        input_ids=input_ids,
        attention_mask=attention_mask,
        special_ids=special_ids,
    )
    language = build_post_media_language_mask(masks)
    vision = masks["vision"].detach().cpu().bool()
    audio = masks["audio"].detach().cpu().bool()
    return {
        "V": vision,
        "A": audio,
        "L": language,
        "VA": vision | audio,
        "LV": language | vision,
        "LA": language | audio,
    }


def _compute_attn_weights(
    module: torch.nn.Module,
    *,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    position_embeddings: Any,
) -> Tuple[torch.Tensor, torch.Tensor]:
    bsz, q_len, _ = hidden_states.size()
    query_states = module.q_proj(hidden_states)
    key_states = module.k_proj(hidden_states)
    value_states = module.v_proj(hidden_states)

    query_states = query_states.view(bsz, q_len, -1, module.head_dim).transpose(1, 2)
    key_states = key_states.view(bsz, q_len, -1, module.head_dim).transpose(1, 2)
    value_states = value_states.view(bsz, q_len, -1, module.head_dim).transpose(1, 2)

    cos, sin = position_embeddings
    query_states, key_states = apply_multimodal_rotary_pos_emb(
        query_states,
        key_states,
        cos,
        sin,
        module.rope_scaling["mrope_section"],
    )
    query_states = query_states.to(torch.float32)
    key_states = key_states.to(torch.float32)
    value_states = value_states.to(torch.float32)

    key_states = repeat_kv(key_states, module.num_key_value_groups)
    value_states = repeat_kv(value_states, module.num_key_value_groups)

    attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) * float(module.scaling)
    if attention_mask is not None:
        attn_weights = attn_weights + attention_mask[:, :, :, : key_states.shape[-2]]
    attn_weights = torch.nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32)
    return attn_weights, value_states


def _compute_last_query_attn_row(
    module: torch.nn.Module,
    *,
    hidden_states: torch.Tensor,
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
    query_states = query_states.to(torch.float32)
    key_states = key_states.to(torch.float32)

    key_states = repeat_kv(key_states, module.num_key_value_groups)
    last_query = query_states[:, :, -1:, :]
    attn_row = torch.matmul(last_query, key_states.transpose(2, 3)) * float(module.scaling)
    if attention_mask is not None:
        attn_row = attn_row + attention_mask[:, :, -1:, : key_states.shape[-2]]
    attn_row = torch.nn.functional.softmax(attn_row, dim=-1, dtype=torch.float32)
    return attn_row


def _masked_attn_output(
    module: torch.nn.Module,
    *,
    attn_weights: torch.Tensor,
    value_states: torch.Tensor,
    selected_query_positions: torch.Tensor,
    threshold: float,
    output_dtype: torch.dtype,
) -> torch.Tensor:
    masked = attn_weights
    valid_positions = selected_query_positions[
        (selected_query_positions >= 0) & (selected_query_positions < int(attn_weights.shape[-2]))
    ]
    if valid_positions.numel() > 0:
        valid_positions = valid_positions.to(device=attn_weights.device)
        modality_mask = torch.ones_like(attn_weights)
        query_sum = attn_weights[:, :, -1, valid_positions].unsqueeze(-1)
        keep_mask = (query_sum <= float(threshold)).to(attn_weights.dtype)
        modality_mask[:, :, valid_positions, :] = keep_mask
        masked = attn_weights * modality_mask
        masked = masked / masked.sum(dim=-1, keepdim=True).clamp(min=1e-6)

    attn_output = torch.matmul(masked, value_states)
    attn_output = attn_output.transpose(1, 2).contiguous()
    attn_output = attn_output.reshape(attn_output.shape[0], attn_output.shape[1], -1)
    attn_output = attn_output.to(output_dtype)
    return module.o_proj(attn_output)


def _masked_attn_output_chunked(
    module: torch.nn.Module,
    *,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    position_embeddings: Any,
    selected_query_positions: torch.Tensor,
    threshold: float,
    output_dtype: torch.dtype,
) -> torch.Tensor:
    """Compute the released AVCD attention exactly, without materializing all query rows at once."""
    bsz, q_len, _ = hidden_states.size()
    query_states = module.q_proj(hidden_states)
    key_states = module.k_proj(hidden_states)
    value_states = module.v_proj(hidden_states)
    query_states = query_states.view(bsz, q_len, -1, module.head_dim).transpose(1, 2)
    key_states = key_states.view(bsz, q_len, -1, module.head_dim).transpose(1, 2)
    value_states = value_states.view(bsz, q_len, -1, module.head_dim).transpose(1, 2)
    cos, sin = position_embeddings
    query_states, key_states = apply_multimodal_rotary_pos_emb(
        query_states,
        key_states,
        cos,
        sin,
        module.rope_scaling["mrope_section"],
    )
    query_states = query_states.to(torch.float32)
    key_states = repeat_kv(key_states.to(torch.float32), module.num_key_value_groups)
    value_states = repeat_kv(value_states.to(torch.float32), module.num_key_value_groups)

    valid_positions = selected_query_positions[
        (selected_query_positions >= 0) & (selected_query_positions < q_len)
    ].to(device=query_states.device)
    row_scale = torch.ones(
        (bsz, query_states.shape[1], q_len, 1),
        dtype=torch.float32,
        device=query_states.device,
    )
    if valid_positions.numel() > 0:
        last_logits = torch.matmul(query_states[:, :, -1:, :], key_states.transpose(2, 3)) * float(module.scaling)
        if attention_mask is not None:
            last_logits = last_logits + attention_mask[:, :, -1:, : key_states.shape[-2]]
        last_weights = torch.nn.functional.softmax(last_logits, dim=-1, dtype=torch.float32)
        row_scale[:, :, valid_positions, :] = (
            last_weights[:, :, 0, valid_positions].unsqueeze(-1) <= float(threshold)
        ).to(torch.float32)

    chunk_size = max(1, int(os.environ.get("AVCD_ATTN_QUERY_CHUNK_SIZE", "128")))
    output_chunks: List[torch.Tensor] = []
    key_transposed = key_states.transpose(2, 3)
    for start in range(0, q_len, chunk_size):
        end = min(q_len, start + chunk_size)
        logits = torch.matmul(query_states[:, :, start:end, :], key_transposed) * float(module.scaling)
        if attention_mask is not None:
            logits = logits + attention_mask[:, :, start:end, : key_states.shape[-2]]
        weights = torch.nn.functional.softmax(logits, dim=-1, dtype=torch.float32)
        masked_weights = weights * row_scale[:, :, start:end, :]
        masked_weights = masked_weights / masked_weights.sum(dim=-1, keepdim=True).clamp(min=1e-6)
        chunk_output = torch.matmul(masked_weights, value_states)
        output_chunks.append(chunk_output)
    attn_output = torch.cat(output_chunks, dim=2)
    attn_output = attn_output.transpose(1, 2).contiguous()
    attn_output = attn_output.reshape(bsz, q_len, -1).to(output_dtype)
    return module.o_proj(attn_output)


def run_avcd_branch(
    adapter: QwenOmniAdapter,
    *,
    inputs: Dict[str, torch.Tensor],
    special_ids: Dict[str, Optional[int]],
    masked_modality: Optional[str] = None,
    threshold: Optional[float] = None,
) -> Dict[str, Any]:
    thinker = adapter._model.thinker
    input_ids = inputs["input_ids"].detach().cpu()
    attention_mask = inputs.get("attention_mask")
    if attention_mask is None:
        attention_mask = torch.ones_like(inputs["input_ids"])
    attention_mask_cpu = attention_mask.detach().cpu()
    label_masks = resolve_avcd_label_masks(
        input_ids=input_ids,
        attention_mask=attention_mask_cpu,
        special_ids=special_ids,
    )
    selected_positions = None
    if masked_modality is not None:
        selected_mask = label_masks.get(str(masked_modality))
        if selected_mask is None:
            raise ValueError(f"unsupported_avcd_modality:{masked_modality}")
        selected_positions = torch.nonzero(selected_mask, as_tuple=False).flatten().long()

    layer_last_query_rows: Dict[int, torch.Tensor] = {}
    hooks: List[Any] = []
    replaced_forwards: List[Tuple[torch.nn.Module, str, Any]] = []
    layers = getattr(getattr(thinker, "model", None), "layers", None)
    if layers is None:
        raise RuntimeError("missing_thinker_model_layers")
    num_layers = len(layers)

    def make_pre_hook(layer_idx: int):
        def hook_fn(module, args, kwargs):
            hidden_states = args[0] if args else kwargs.get("hidden_states")
            position_embeddings = kwargs.get("position_embeddings")
            if hidden_states is None or position_embeddings is None:
                return None
            attn_mask = kwargs.get("attention_mask")
            attn_row = _compute_last_query_attn_row(
                module,
                hidden_states=hidden_states,
                attention_mask=attn_mask,
                position_embeddings=position_embeddings,
            )
            layer_last_query_rows[int(layer_idx)] = attn_row[0, :, 0, :].detach().float().cpu()
            return None

        return hook_fn

    def make_masked_forward(layer_idx: int):
        def forward_fn(
            module,
            hidden_states,
            attention_mask=None,
            position_ids=None,
            past_key_values=None,
            output_attentions=False,
            use_cache=False,
            cache_position=None,
            position_embeddings=None,
            **kwargs,
        ):
            if past_key_values is not None or use_cache:
                raise RuntimeError("strict AVCD masked forward requires use_cache=False")
            if position_embeddings is None:
                raise RuntimeError("strict AVCD masked forward is missing position_embeddings")
            positions = selected_positions
            if hidden_states.shape[1] <= 1 or int(layer_idx) >= int(num_layers - 1):
                positions = torch.empty(0, dtype=torch.long)
            attn_output = _masked_attn_output_chunked(
                module,
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                position_embeddings=position_embeddings,
                selected_query_positions=positions,
                threshold=float(threshold),
                output_dtype=hidden_states.dtype,
            )
            return attn_output, None

        return forward_fn

    if masked_modality is None:
        for layer_idx, layer_module in enumerate(layers):
            hooks.append(
                layer_module.self_attn.register_forward_pre_hook(
                    make_pre_hook(int(layer_idx)),
                    with_kwargs=True,
                )
            )
    else:
        if threshold is None or selected_positions is None:
            raise RuntimeError("masked AVCD branch requires threshold and selected positions")
        for layer_idx, layer_module in enumerate(layers):
            attention = layer_module.self_attn
            forward_attr = "_old_forward" if hasattr(attention, "_old_forward") else "forward"
            original_forward = getattr(attention, forward_attr)
            setattr(attention, forward_attr, MethodType(make_masked_forward(int(layer_idx)), attention))
            replaced_forwards.append((attention, forward_attr, original_forward))

    try:
        thinker.rope_deltas = None
        with adapter.temporary_release_cuda_reserve("avcd_branch"):
            with torch.inference_mode():
                outputs = thinker(
                    **inputs,
                    use_cache=False,
                    output_attentions=False,
                    output_hidden_states=False,
                    return_dict=True,
                )
        logits = outputs.logits[:, -1, :].detach().float().cpu()
    finally:
        for hook in hooks:
            hook.remove()
        for module, forward_attr, original_forward in replaced_forwards:
            setattr(module, forward_attr, original_forward)
        thinker.rope_deltas = None

    del outputs
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    result: Dict[str, Any] = {
        "logits": logits,
        "captured_layers": sorted(int(layer_id) for layer_id in layer_last_query_rows.keys()),
    }
    if masked_modality is not None:
        result["masked_modality"] = str(masked_modality)
        return result

    effective_layers = sorted(int(layer_id) for layer_id in layer_last_query_rows.keys())
    if len(effective_layers) > 1:
        effective_layers = effective_layers[:-1]
    if not effective_layers:
        result["avg_dominance"] = []
        result["threshold"] = None
        result["modality_dominance"] = {}
        return result

    stacked_rows = torch.stack([layer_last_query_rows[int(layer_id)] for layer_id in effective_layers], dim=0)
    mean_last_query = stacked_rows.mean(dim=0)
    threshold_value = float(torch.quantile(mean_last_query, 0.5, dim=-1, keepdim=True).mean().item())

    dominance_values: Dict[str, List[float]] = {"video": [], "audio": [], "language": []}
    dominance_masks = {
        "video": label_masks["V"],
        "audio": label_masks["A"],
        "language": label_masks["L"],
    }
    for layer_id in effective_layers:
        row = layer_last_query_rows[int(layer_id)].float()
        layer_mean = row.mean(dim=0)
        for name, mask in dominance_masks.items():
            positions = torch.nonzero(mask, as_tuple=False).flatten().long()
            if positions.numel() <= 0:
                dominance_values[name].append(0.0)
                continue
            mass = float(layer_mean[positions].sum().item()) / float(positions.numel())
            dominance_values[name].append(mass)

    modality_dominance = {
        name: float(sum(values) / len(values)) if values else 0.0
        for name, values in dominance_values.items()
    }
    avg_dominance = sorted(
        [(name, float(score)) for name, score in modality_dominance.items()],
        key=lambda item: item[1],
        reverse=True,
    )
    result["avg_dominance"] = avg_dominance
    result["threshold"] = float(threshold_value)
    result["modality_dominance"] = modality_dominance
    return result


def run_avcd_dominance_reader(
    adapter: QwenOmniAdapter,
    *,
    inputs: Dict[str, torch.Tensor],
    special_ids: Dict[str, Optional[int]],
) -> Dict[str, Any]:
    thinker = adapter._model.thinker
    input_ids = inputs["input_ids"].detach().cpu()
    attention_mask = inputs.get("attention_mask")
    if attention_mask is None:
        attention_mask = torch.ones_like(inputs["input_ids"])
    attention_mask_cpu = attention_mask.detach().cpu()
    label_masks = resolve_avcd_label_masks(
        input_ids=input_ids,
        attention_mask=attention_mask_cpu,
        special_ids=special_ids,
    )

    layer_last_query_rows: Dict[int, torch.Tensor] = {}
    hooks: List[Any] = []
    layers = getattr(getattr(thinker, "model", None), "layers", None)
    if layers is None:
        raise RuntimeError("missing_thinker_model_layers")

    def make_pre_hook(layer_idx: int):
        def hook_fn(module, args, kwargs):
            hidden_states = args[0] if args else kwargs.get("hidden_states")
            position_embeddings = kwargs.get("position_embeddings")
            if hidden_states is None or position_embeddings is None:
                return None
            attn_mask = kwargs.get("attention_mask")
            attn_row = _compute_last_query_attn_row(
                module,
                hidden_states=hidden_states,
                attention_mask=attn_mask,
                position_embeddings=position_embeddings,
            )
            layer_last_query_rows[int(layer_idx)] = attn_row[0, :, 0, :].detach().float().cpu()
            return None

        return hook_fn

    for layer_idx, layer_module in enumerate(layers):
        hooks.append(
            layer_module.self_attn.register_forward_pre_hook(
                make_pre_hook(int(layer_idx)),
                with_kwargs=True,
            )
        )

    outputs = None
    try:
        thinker.rope_deltas = None
        with adapter.temporary_release_cuda_reserve("avcd_dominance_reader"):
            with torch.inference_mode():
                outputs = thinker(
                    **inputs,
                    use_cache=False,
                    output_attentions=False,
                    output_hidden_states=False,
                    return_dict=True,
                )
        logits = outputs.logits[:, -1, :].detach().float().cpu()
    finally:
        for hook in hooks:
            hook.remove()
        thinker.rope_deltas = None
        del outputs
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    result: Dict[str, Any] = {
        "logits": logits,
        "captured_layers": sorted(int(layer_id) for layer_id in layer_last_query_rows.keys()),
    }
    effective_layers = sorted(int(layer_id) for layer_id in layer_last_query_rows.keys())
    if len(effective_layers) > 1:
        effective_layers = effective_layers[:-1]
    if not effective_layers:
        result["avg_dominance"] = []
        result["modality_dominance"] = {}
        return result

    dominance_values: Dict[str, List[float]] = {"video": [], "audio": [], "language": []}
    dominance_masks = {
        "video": label_masks["V"],
        "audio": label_masks["A"],
        "language": label_masks["L"],
    }
    for layer_id in effective_layers:
        row = layer_last_query_rows[int(layer_id)].float()
        layer_mean = row.mean(dim=0)
        for name, mask in dominance_masks.items():
            positions = torch.nonzero(mask, as_tuple=False).flatten().long()
            if positions.numel() <= 0:
                dominance_values[name].append(0.0)
                continue
            mass = float(layer_mean[positions].sum().item()) / float(positions.numel())
            dominance_values[name].append(mass)

    modality_dominance = {
        name: float(sum(values) / len(values)) if values else 0.0
        for name, values in dominance_values.items()
    }
    avg_dominance = sorted(
        [(name, float(score)) for name, score in modality_dominance.items()],
        key=lambda item: item[1],
        reverse=True,
    )
    result["avg_dominance"] = avg_dominance
    result["modality_dominance"] = modality_dominance
    return result


def logits_entropy(logits: torch.Tensor) -> float:
    probs = torch.softmax(logits, dim=-1)
    log_probs = torch.log_softmax(logits, dim=-1)
    entropy = -(probs * log_probs).sum(dim=-1)
    return float(entropy.item())


def avcd_modality_triplet(dominant_modality: str) -> Tuple[str, str, str]:
    dominant = str(dominant_modality).lower()
    if dominant == "language":
        return "VA", "A", "V"
    if dominant == "video":
        return "LA", "A", "L"
    if dominant == "audio":
        return "LV", "V", "L"
    raise ValueError(f"unsupported_dominant_modality:{dominant_modality}")


def decode_token_surface(tokenizer, token_id: int) -> str:
    try:
        return tokenizer.decode([int(token_id)], skip_special_tokens=False)
    except Exception:
        return str(int(token_id))


def strict_greedy_decode(
    adapter: QwenOmniAdapter,
    *,
    base_inputs: Dict[str, torch.Tensor],
    special_ids: Dict[str, Optional[int]],
    max_new_tokens: int,
    use_avcd: bool,
    avcd_alpha: float,
    plausibility_beta: float,
    entropy_threshold: float,
    avcd_mask_mode: str = "dominant_triplet",
    allowed_token_ids: Optional[Sequence[int]] = None,
) -> Dict[str, Any]:
    tokenizer = adapter._processor.tokenizer
    eos_id = int(tokenizer.eos_token_id)
    generated_ids: List[int] = []
    steps: List[Dict[str, Any]] = []
    mask_mode = str(avcd_mask_mode or "dominant_triplet").strip().lower()
    if mask_mode not in {"dominant_triplet", "language_only"}:
        raise ValueError(f"unsupported_avcd_mask_mode:{avcd_mask_mode}")
    candidate_ids: Optional[torch.Tensor] = None
    if allowed_token_ids is not None:
        normalized_ids = [int(token_id) for token_id in allowed_token_ids]
        if len(normalized_ids) < 2 or len(set(normalized_ids)) != len(normalized_ids):
            raise ValueError("allowed_token_ids must contain at least two unique token IDs")
        candidate_ids = torch.tensor(normalized_ids, dtype=torch.long)

    for step_idx in range(max(0, int(max_new_tokens))):
        step_inputs = extend_inputs_for_generation(base_inputs, generated_ids)
        branch_inputs = cache_multimodal_inputs_for_avcd_step(adapter, step_inputs)
        full_branch = run_avcd_branch(
            adapter,
            inputs=branch_inputs,
            special_ids=special_ids,
            masked_modality=None,
            threshold=None,
        )
        orig_logits = full_branch["logits"].detach().float()
        entropy_value = logits_entropy(orig_logits)
        next_logits = orig_logits
        candidate_selection_logits: Optional[torch.Tensor] = None
        step_trace: Dict[str, Any] = {
            "step": int(step_idx),
            "mode": "baseline",
            "entropy": float(entropy_value),
            "threshold": full_branch.get("threshold"),
            "avg_dominance": [
                [str(name), float(score)]
                for name, score in (full_branch.get("avg_dominance") or [])
            ],
            "readout": "candidate_constrained" if candidate_ids is not None else "full_vocabulary",
        }

        if use_avcd:
            step_trace["mode"] = "skip_low_entropy"
            if entropy_value >= float(entropy_threshold):
                avg_dominance = full_branch.get("avg_dominance") or []
                if not avg_dominance:
                    raise RuntimeError("missing_full_branch_dominance")
                dominant_modality = str(avg_dominance[0][0])
                threshold_value = full_branch.get("threshold")
                if threshold_value is None:
                    raise RuntimeError("missing_full_branch_threshold")
                if mask_mode == "language_only":
                    branch_language = run_avcd_branch(
                        adapter,
                        inputs=branch_inputs,
                        special_ids=special_ids,
                        masked_modality="L",
                        threshold=float(threshold_value),
                    )
                    masked_logits = branch_language["logits"].detach().float()
                    contrastive_logits = (
                        (1.0 + float(avcd_alpha)) * orig_logits
                        - float(avcd_alpha) * masked_logits
                    )
                    step_trace.update(
                        {
                            "mode": "avcd_language_only",
                            "dominant_modality": dominant_modality,
                            "mask_triplet": ["L"],
                            "threshold": float(threshold_value),
                            "avcd_mask_mode": "language_only",
                        }
                    )
                else:
                    modality1, modality2, modality3 = avcd_modality_triplet(dominant_modality)
                    branch1 = run_avcd_branch(
                        adapter,
                        inputs=branch_inputs,
                        special_ids=special_ids,
                        masked_modality=modality1,
                        threshold=float(threshold_value),
                    )
                    branch2 = run_avcd_branch(
                        adapter,
                        inputs=branch_inputs,
                        special_ids=special_ids,
                        masked_modality=modality2,
                        threshold=float(threshold_value),
                    )
                    branch3 = run_avcd_branch(
                        adapter,
                        inputs=branch_inputs,
                        special_ids=special_ids,
                        masked_modality=modality3,
                        threshold=float(threshold_value),
                    )
                    logits1 = branch1["logits"].detach().float()
                    logits2 = branch2["logits"].detach().float()
                    logits3 = branch3["logits"].detach().float()
                    contrastive_logits = (
                        (2.0 + 2.0 * float(avcd_alpha)) * orig_logits
                        - 2.0 * float(avcd_alpha) * logits1
                        + logits2
                        + logits3
                    )
                    step_trace.update(
                        {
                            "mode": "avcd",
                            "dominant_modality": dominant_modality,
                            "mask_triplet": [modality1, modality2, modality3],
                            "threshold": float(threshold_value),
                            "avcd_mask_mode": "dominant_triplet",
                        }
                    )
                if candidate_ids is None:
                    cutoff_reference = orig_logits.max(dim=-1, keepdim=True).values
                    step_trace["plausibility_scope"] = "full_vocabulary"
                    cutoff = math.log(float(plausibility_beta)) + cutoff_reference
                    next_logits = contrastive_logits.masked_fill(orig_logits < cutoff, -1.0e-4)
                else:
                    candidate_ids_device = candidate_ids.to(device=orig_logits.device)
                    candidate_orig_logits = orig_logits.index_select(-1, candidate_ids_device)
                    candidate_contrastive_logits = contrastive_logits.index_select(
                        -1, candidate_ids_device
                    )
                    cutoff_reference = candidate_orig_logits.max(dim=-1, keepdim=True).values
                    cutoff = math.log(float(plausibility_beta)) + cutoff_reference
                    candidate_valid = candidate_orig_logits >= cutoff
                    candidate_selection_logits = candidate_contrastive_logits.masked_fill(
                        ~candidate_valid,
                        -float("inf"),
                    )
                    if not torch.isfinite(candidate_selection_logits).any():
                        candidate_selection_logits = candidate_contrastive_logits
                    step_trace["plausibility_scope"] = "allowed_candidates"
                    step_trace["candidate_valid"] = [
                        bool(value)
                        for value in candidate_valid.detach().cpu().reshape(-1).tolist()
                    ]

        if candidate_ids is None:
            next_token_id = int(torch.argmax(next_logits, dim=-1).item())
        else:
            candidate_ids_device = candidate_ids.to(device=next_logits.device)
            candidate_logits = candidate_selection_logits
            if candidate_logits is None:
                candidate_logits = next_logits.index_select(-1, candidate_ids_device)
            selected_index = int(torch.argmax(candidate_logits, dim=-1).item())
            next_token_id = int(candidate_ids_device[selected_index].item())
            step_trace["candidate_token_ids"] = [int(value) for value in candidate_ids.tolist()]
            step_trace["candidate_logits"] = [
                (float(value) if math.isfinite(float(value)) else None)
                for value in candidate_logits.detach().float().cpu().reshape(-1).tolist()
            ]
        step_trace["next_token_id"] = int(next_token_id)
        step_trace["next_token"] = decode_token_surface(tokenizer, next_token_id)
        steps.append(step_trace)
        if int(next_token_id) == int(eos_id):
            break
        generated_ids.append(int(next_token_id))

    answer = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
    return {
        "answer": answer,
        "generated_token_ids": [int(token_id) for token_id in generated_ids],
        "steps": steps,
        "avcd_applied": bool(any(str(step.get("mode")).startswith("avcd") for step in steps)),
        "avcd_mask_mode": mask_mode,
        "n_generated_tokens": int(len(generated_ids)),
    }
