#!/usr/bin/env python3
from __future__ import annotations

import gc
import math
import random
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, List, Optional, Sequence, Tuple

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

from OMhallucination.qwen_omni_adapter import QwenOmniAdapter  # noqa: E402
from analyze_full_run_token_attention import safe_text  # noqa: E402
from avcd_qwen_omni_common import (  # noqa: E402
    prepare_full_av_inputs,
    resolve_avcd_label_masks,
)


GROUP_KEY_MAP = {
    "language": "L",
    "audio": "A",
    "video": "V",
    "vision": "V",
    "language_audio": "LA",
    "language_video": "LV",
    "language_vision": "LV",
    "audio_video": "VA",
    "audio_vision": "VA",
}


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


def build_full_inputs_from_prompt(
    adapter: QwenOmniAdapter,
    *,
    row: Dict[str, Any],
    budget: Dict[str, float],
) -> Dict[str, torch.Tensor]:
    prompt_text = safe_text(row.get("prompt_text") or row.get("formatted_question") or row.get("question"))
    if not prompt_text:
        raise ValueError(f"missing_prompt_text sample_id={safe_text(row.get('sample_id'))}")
    if "Answer with only Yes or No." not in prompt_text:
        prompt_text = prompt_text.rstrip() + "\nAnswer with only Yes or No."
    row_for_prompt = dict(row)
    row_for_prompt["question"] = prompt_text
    row_for_prompt["formatted_question"] = prompt_text
    return prepare_full_av_inputs(adapter, row=row_for_prompt, budget=budget)


def yes_no_distribution_from_logits(
    adapter: QwenOmniAdapter,
    *,
    logits: torch.Tensor,
) -> Dict[str, Any]:
    tokenizer = adapter._processor.tokenizer
    yes_id = tokenizer.encode("Yes", add_special_tokens=False)[0]
    no_id = tokenizer.encode("No", add_special_tokens=False)[0]
    log_probs = torch.log_softmax(logits.float(), dim=-1)
    yes_logp = float(log_probs[0, int(yes_id)].item())
    no_logp = float(log_probs[0, int(no_id)].item())
    pair = torch.tensor([[yes_logp, no_logp]], dtype=torch.float32)
    pair_probs = torch.softmax(pair, dim=-1)
    p_yes = float(pair_probs[0, 0].item())
    p_no = float(pair_probs[0, 1].item())
    margin = float(p_yes - p_no)
    answer = "Yes" if margin >= 0.0 else "No"
    return {
        "answer": answer,
        "p_yes": p_yes,
        "p_no": p_no,
        "margin": margin,
        "yes_logp": yes_logp,
        "no_logp": no_logp,
    }


def branch_answer_from_row(row: Dict[str, Any], branch_name: str) -> Optional[str]:
    branches = dict(row.get("branches") or {})
    branch = dict(branches.get(branch_name) or {})
    return norm_yes_no(branch.get("answer"))


def target_branch_name(target_modality: str) -> Optional[str]:
    target = safe_text(target_modality).lower()
    if target == "audio":
        return "audio_only"
    if target == "visual":
        return "visual_only"
    return None


def resolve_group_mask(
    *,
    label_masks: Dict[str, torch.Tensor],
    group_name: str,
    target_modality: str,
) -> torch.Tensor:
    key = safe_text(group_name).lower()
    if key == "auto_target":
        key = "audio" if safe_text(target_modality).lower() == "audio" else "video"
    elif key == "auto_non_target":
        key = "video" if safe_text(target_modality).lower() == "audio" else "audio"
    mask_key = GROUP_KEY_MAP.get(key)
    if mask_key is None:
        raise ValueError(f"unsupported_group_name:{group_name}")
    selected = label_masks.get(mask_key)
    if selected is None:
        raise ValueError(f"missing_group_mask:{mask_key}")
    return selected.detach().cpu().bool()


def _positions_from_mask(mask: torch.Tensor) -> torch.Tensor:
    return torch.nonzero(mask, as_tuple=False).flatten().long()


def _normalize_row(row: torch.Tensor) -> torch.Tensor:
    return row / row.sum(dim=-1, keepdim=True).clamp(min=1e-6)


def _apply_source_edit(
    row: torch.Tensor,
    *,
    source_positions: torch.Tensor,
    target_positions: torch.Tensor,
    mode: str,
    rng: Optional[random.Random],
) -> torch.Tensor:
    edited = row.clone()
    if source_positions.numel() <= 0:
        return _normalize_row(edited)

    source_mass = edited[:, :, source_positions].sum(dim=-1, keepdim=True)
    if mode == "none":
        return _normalize_row(edited)
    if mode == "zero_source":
        edited[:, :, source_positions] = 0.0
        return _normalize_row(edited)
    if mode == "uniform_source":
        edited[:, :, source_positions] = source_mass / float(source_positions.numel())
        return _normalize_row(edited)
    if mode == "reverse_source":
        edited[:, :, source_positions] = edited[:, :, source_positions.flip(0)]
        return _normalize_row(edited)
    if mode == "shuffle_source":
        perm = list(range(int(source_positions.numel())))
        if rng is not None:
            rng.shuffle(perm)
        perm_tensor = torch.tensor(perm, dtype=torch.long, device=source_positions.device)
        edited[:, :, source_positions] = edited[:, :, source_positions[perm_tensor]]
        return _normalize_row(edited)
    if mode == "transfer_source_to_target":
        edited[:, :, source_positions] = 0.0
        if target_positions.numel() > 0:
            target_slice = edited[:, :, target_positions]
            target_mass = target_slice.sum(dim=-1, keepdim=True)
            if bool(torch.all(target_mass <= 1e-6)):
                addition = source_mass / float(target_positions.numel())
                edited[:, :, target_positions] = target_slice + addition
            else:
                edited[:, :, target_positions] = target_slice + (
                    target_slice / target_mass.clamp(min=1e-6)
                ) * source_mass
        return _normalize_row(edited)
    raise ValueError(f"unsupported_source_edit_mode:{mode}")


def _apply_target_edit(
    row: torch.Tensor,
    *,
    target_positions: torch.Tensor,
    mode: str,
) -> torch.Tensor:
    edited = row.clone()
    if target_positions.numel() <= 0 or mode == "none":
        return _normalize_row(edited)
    target_mass = edited[:, :, target_positions].sum(dim=-1, keepdim=True)
    if mode == "uniform_target":
        edited[:, :, target_positions] = target_mass / float(target_positions.numel())
        return _normalize_row(edited)
    if mode == "sharpen_target":
        target_slice = edited[:, :, target_positions]
        sharpened = torch.square(target_slice)
        sharpened = sharpened / sharpened.sum(dim=-1, keepdim=True).clamp(min=1e-6)
        edited[:, :, target_positions] = sharpened * target_mass
        return _normalize_row(edited)
    raise ValueError(f"unsupported_target_edit_mode:{mode}")


def _clean_selected_heads_by_layer(
    selected_heads_by_layer: Optional[Mapping[int, Sequence[int]]],
    *,
    selected_layers: Sequence[int],
    layers: Sequence[Any],
) -> Dict[int, List[int]]:
    if not selected_heads_by_layer:
        return {}
    selected_layer_set = {int(layer) for layer in selected_layers}
    clean: Dict[int, List[int]] = {}
    for raw_layer, raw_heads in selected_heads_by_layer.items():
        layer_idx = int(raw_layer)
        if layer_idx not in selected_layer_set or layer_idx < 0 or layer_idx >= len(layers):
            continue
        self_attn = getattr(layers[layer_idx], "self_attn", None)
        num_heads = int(getattr(self_attn, "num_heads", 0) or 0)
        heads = sorted({int(head) for head in raw_heads if 0 <= int(head) < num_heads})
        if heads:
            clean[layer_idx] = heads
    return clean


def _compute_last_query_attn_row_and_values(
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

    last_query = query_states[:, :, -1:, :]
    attn_row = torch.matmul(last_query, key_states.transpose(2, 3)) * float(module.scaling)
    if attention_mask is not None:
        attn_row = attn_row + attention_mask[:, :, -1:, : key_states.shape[-2]]
    attn_row = torch.nn.functional.softmax(attn_row, dim=-1, dtype=torch.float32)
    return attn_row, value_states


def run_causal_joint_attention_probe(
    adapter: QwenOmniAdapter,
    *,
    inputs: Dict[str, torch.Tensor],
    special_ids: Dict[str, Optional[int]],
    target_modality: str,
    source_group: str,
    target_group: str,
    source_edit_mode: str,
    target_edit_mode: str,
    layer_indices: Sequence[int],
    seed: int,
    selected_heads_by_layer: Optional[Mapping[int, Sequence[int]]] = None,
) -> Dict[str, Any]:
    thinker = adapter._model.thinker
    input_ids = inputs["input_ids"].detach().cpu()
    attention_mask = inputs.get("attention_mask")
    if attention_mask is None:
        attention_mask = torch.ones_like(inputs["input_ids"])
    label_masks = resolve_avcd_label_masks(
        input_ids=input_ids,
        attention_mask=attention_mask.detach().cpu(),
        special_ids=special_ids,
    )
    source_mask = resolve_group_mask(
        label_masks=label_masks,
        group_name=source_group,
        target_modality=target_modality,
    )
    target_mask = resolve_group_mask(
        label_masks=label_masks,
        group_name=target_group,
        target_modality=target_modality,
    )
    source_positions = _positions_from_mask(source_mask)
    target_positions = _positions_from_mask(target_mask)
    source_positions_device = None
    target_positions_device = None

    layers = getattr(getattr(thinker, "model", None), "layers", None)
    if layers is None:
        raise RuntimeError("missing_thinker_model_layers")
    selected_layers = {int(layer) for layer in layer_indices if 0 <= int(layer) < len(layers)}
    if not selected_layers:
        raise ValueError("empty_selected_layers")
    clean_selected_heads = _clean_selected_heads_by_layer(
        selected_heads_by_layer,
        selected_layers=sorted(selected_layers),
        layers=layers,
    )
    if selected_heads_by_layer and not clean_selected_heads:
        raise ValueError("empty_selected_heads")

    rng = random.Random(int(seed))
    hooks: List[Any] = []
    patched_outputs: Dict[int, torch.Tensor] = {}
    layer_traces: Dict[int, Dict[str, float]] = {}

    def make_pre_hook(layer_idx: int):
        def hook_fn(module, args, kwargs):
            nonlocal source_positions_device, target_positions_device
            hidden_states = args[0] if args else kwargs.get("hidden_states")
            position_embeddings = kwargs.get("position_embeddings")
            if hidden_states is None or position_embeddings is None:
                return None
            attn_mask = kwargs.get("attention_mask")
            if int(layer_idx) not in selected_layers or hidden_states.shape[1] <= 1:
                return None
            attn_row, value_states = _compute_last_query_attn_row_and_values(
                module,
                hidden_states=hidden_states,
                attention_mask=attn_mask,
                position_embeddings=position_embeddings,
            )
            if source_positions_device is None:
                source_positions_device = source_positions.to(device=attn_row.device)
                target_positions_device = target_positions.to(device=attn_row.device)
            row_before = attn_row[:, :, 0, :].clone()
            head_indices = clean_selected_heads.get(int(layer_idx))
            if clean_selected_heads and not head_indices:
                return None
            head_tensor = None
            metric_row_before = row_before
            if head_indices:
                head_tensor = torch.tensor(head_indices, dtype=torch.long, device=row_before.device)
                metric_row_before = row_before.index_select(1, head_tensor)

            source_mass_before = float(metric_row_before[:, :, source_positions_device].sum(dim=-1).mean().item()) if source_positions_device.numel() > 0 else 0.0
            target_mass_before = float(metric_row_before[:, :, target_positions_device].sum(dim=-1).mean().item()) if target_positions_device.numel() > 0 else 0.0
            edited_row = _apply_source_edit(
                row_before,
                source_positions=source_positions_device,
                target_positions=target_positions_device,
                mode=source_edit_mode,
                rng=rng,
            )
            edited_row = _apply_target_edit(
                edited_row,
                target_positions=target_positions_device,
                mode=target_edit_mode,
            )
            row = edited_row
            if head_tensor is not None:
                row = row_before.clone()
                row[:, head_tensor, :] = edited_row.index_select(1, head_tensor)
                metric_row_after = row.index_select(1, head_tensor)
            else:
                metric_row_after = row

            source_mass_after = float(metric_row_after[:, :, source_positions_device].sum(dim=-1).mean().item()) if source_positions_device.numel() > 0 else 0.0
            target_mass_after = float(metric_row_after[:, :, target_positions_device].sum(dim=-1).mean().item()) if target_positions_device.numel() > 0 else 0.0
            layer_traces[int(layer_idx)] = {
                "source_mass_before": source_mass_before,
                "target_mass_before": target_mass_before,
                "source_mass_after": source_mass_after,
                "target_mass_after": target_mass_after,
                "patched_head_count": int(len(head_indices) if head_indices else int(row_before.shape[1])),
                "patched_heads": [int(head) for head in head_indices] if head_indices else "all",
            }
            edited_attn_row = row.unsqueeze(2)
            attn_output = torch.matmul(edited_attn_row, value_states)
            attn_output = attn_output.transpose(1, 2).contiguous()
            attn_output = attn_output.reshape(attn_output.shape[0], attn_output.shape[1], -1)
            attn_output = attn_output.to(hidden_states.dtype)
            patched_outputs[id(module)] = module.o_proj(attn_output)
            return None

        return hook_fn

    def make_forward_hook():
        def hook_fn(module, _args, output):
            patched = patched_outputs.pop(id(module), None)
            if patched is None:
                return None
            if isinstance(output, tuple):
                if not output:
                    return output
                attn_output = output[0]
                attn_output[:, -1:, :].copy_(patched.to(dtype=attn_output.dtype))
                return (attn_output, *output[1:])
            if isinstance(output, list):
                if not output:
                    return output
                attn_output = output[0]
                attn_output[:, -1:, :].copy_(patched.to(dtype=attn_output.dtype))
                return output
            output[:, -1:, :].copy_(patched.to(dtype=output.dtype))
            return output

        return hook_fn

    for layer_idx, layer_module in enumerate(layers):
        hooks.append(
            layer_module.self_attn.register_forward_pre_hook(
                make_pre_hook(int(layer_idx)),
                with_kwargs=True,
            )
        )
        hooks.append(layer_module.self_attn.register_forward_hook(make_forward_hook()))

    outputs = None
    try:
        thinker.rope_deltas = None
        with adapter.temporary_release_cuda_reserve("causal_joint_attention_probe"):
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

    return {
        "logits": logits,
        "selected_layers": sorted(int(layer) for layer in selected_layers),
        "selected_heads_by_layer": {str(layer): [int(head) for head in heads] for layer, heads in sorted(clean_selected_heads.items())},
        "source_group": str(source_group),
        "target_group": str(target_group),
        "source_edit_mode": str(source_edit_mode),
        "target_edit_mode": str(target_edit_mode),
        "source_group_positions": int(source_positions.numel()),
        "target_group_positions": int(target_positions.numel()),
        "layer_traces": {str(layer): dict(trace) for layer, trace in sorted(layer_traces.items())},
        "mean_source_mass_before": mean(trace.get("source_mass_before") for trace in layer_traces.values()),
        "mean_target_mass_before": mean(trace.get("target_mass_before") for trace in layer_traces.values()),
        "mean_source_mass_after": mean(trace.get("source_mass_after") for trace in layer_traces.values()),
        "mean_target_mass_after": mean(trace.get("target_mass_after") for trace in layer_traces.values()),
    }
