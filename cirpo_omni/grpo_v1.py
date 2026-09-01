from __future__ import annotations

import math
import re
from typing import Any, Dict, Iterable, List, Optional, Sequence

import numpy as np
import torch

from cirpo_omni.rewards import (
    _full_branch_clean_score,
    _relation_satisfaction,
    _target_action,
    _target_utility,
    parse_episode_output,
)
from omni_dpo.media import MediaResolver
from omni_dpo.modeling import compose_question_text
from omni_dpo.preference_pairs import safe_text

GRPO_V1_SYSTEM_PROMPT = (
    "You are a precise multimodal analyst. "
    "When the answer depends on audio, video, or event order, cite one or more supporting timestamps in [MM:SS]. "
    "Keep the explanation brief and always end with a final answer line in the requested format."
)
GRPO_V1_USER_INSTRUCTION = (
    "Additional requirements:\n"
    "- Ground the answer in the available audio and visual evidence.\n"
    "- If timing matters, cite one or more supporting timestamps in [MM:SS].\n"
    "- Keep the explanation brief.\n"
    "- End with the final answer in the requested format."
)

_BRACKETED_TIMESTAMP_RE = re.compile(r"\[(\d{1,2}(?::\d{1,2}){1,2})\]")
_PLAIN_TIMESTAMP_RE = re.compile(r"(?<![\d\[])(\d{1,2}(?::\d{1,2}){1,2})(?![\d\]])")
_REASONING_KEYS = (
    "omnivideobench_reasoning_steps",
    "reasoning_steps",
)
_EVIDENCE_TEXT_KEYS = (
    "evidence",
    "evidece",
    "evience",
    "evodence",
    "nevidence",
    "text",
    "content",
)


def build_grpo_v1_system_prompt(custom_prompt: Optional[str] = None) -> str:
    text = safe_text(custom_prompt)
    return text or GRPO_V1_SYSTEM_PROMPT


def build_grpo_v1_question_text(question_text: str, custom_instruction: Optional[str] = None) -> str:
    base = safe_text(question_text)
    instruction = safe_text(custom_instruction) or GRPO_V1_USER_INSTRUCTION
    if not base:
        return instruction
    return f"{base}\n\n{instruction}".strip()


def iter_episode_branches(episode: Dict[str, Any]) -> Iterable[Dict[str, Any]]:
    for key in ("full_branch", "cf_branch_a", "cf_branch_b"):
        branch = episode.get(key)
        if isinstance(branch, dict):
            yield branch


def normalize_group_advantages(rewards: Sequence[float]) -> List[float]:
    reward_array = np.asarray(list(rewards), dtype=np.float32)
    if reward_array.size == 0:
        return []
    mean = float(reward_array.mean())
    std = float(reward_array.std())
    if std < 1e-6:
        return [0.0 for _ in reward_array.tolist()]
    return [float((reward - mean) / std) for reward in reward_array.tolist()]


def compute_grpo_clip_loss(
    *,
    actor_mean_logp: torch.Tensor,
    ref_mean_logp: torch.Tensor,
    advantage: float,
    clip_eps: float,
    kl_beta: float,
) -> tuple[torch.Tensor, Dict[str, float]]:
    log_ratio = actor_mean_logp - ref_mean_logp
    ratio = torch.exp(torch.clamp(log_ratio, min=-20.0, max=20.0))
    clipped_ratio = torch.clamp(ratio, 1.0 - float(clip_eps), 1.0 + float(clip_eps))
    advantage_tensor = torch.as_tensor(
        float(advantage),
        device=actor_mean_logp.device,
        dtype=actor_mean_logp.dtype,
    )
    surrogate = torch.minimum(ratio * advantage_tensor, clipped_ratio * advantage_tensor)
    kl_estimate = log_ratio
    loss = -(surrogate - float(kl_beta) * kl_estimate)
    diagnostics = {
        "ratio": float(ratio.detach().item()),
        "clipped_ratio": float(clipped_ratio.detach().item()),
        "kl_estimate": float(kl_estimate.detach().item()),
        "surrogate": float(surrogate.detach().item()),
    }
    return loss, diagnostics


def _parse_colon_time(token: str) -> Optional[float]:
    text = safe_text(token)
    if not text:
        return None
    parts = text.split(":")
    if len(parts) == 2:
        minutes, seconds = parts
        hours = "0"
    elif len(parts) == 3:
        hours, minutes, seconds = parts
    else:
        return None
    try:
        hours_value = float(hours)
        minutes_value = float(minutes)
        seconds_value = float(seconds)
    except ValueError:
        return None
    if minutes_value < 0 or seconds_value < 0:
        return None
    return float(hours_value * 3600.0 + minutes_value * 60.0 + seconds_value)


def extract_response_timestamps(response_text: str) -> List[float]:
    text = safe_text(response_text)
    values: List[float] = []
    seen: set[float] = set()
    for regex in (_BRACKETED_TIMESTAMP_RE, _PLAIN_TIMESTAMP_RE):
        for match in regex.finditer(text):
            seconds = _parse_colon_time(match.group(1))
            if seconds is None:
                continue
            rounded = round(seconds, 3)
            if rounded in seen:
                continue
            seen.add(rounded)
            values.append(float(seconds))
    return values


def _append_point(targets: List[Dict[str, Any]], value: Optional[float], *, source: str) -> None:
    if value is None:
        return
    targets.append(
        {
            "kind": "point",
            "time": float(value),
            "source": source,
        }
    )


def _append_interval(targets: List[Dict[str, Any]], start: Optional[float], end: Optional[float], *, source: str) -> None:
    if start is None or end is None:
        return
    low = min(float(start), float(end))
    high = max(float(start), float(end))
    targets.append(
        {
            "kind": "interval",
            "start": low,
            "end": high,
            "source": source,
        }
    )


def _extract_temporal_targets_from_steps(source_meta: Dict[str, Any], *, source: str) -> List[Dict[str, Any]]:
    targets: List[Dict[str, Any]] = []
    for key in _REASONING_KEYS:
        steps = source_meta.get(key) or []
        if not isinstance(steps, list):
            continue
        for step in steps:
            text_chunks: List[str] = []
            if isinstance(step, dict):
                for evidence_key in _EVIDENCE_TEXT_KEYS:
                    value = step.get(evidence_key)
                    if value:
                        text_chunks.append(str(value))
            elif step:
                text_chunks.append(str(step))
            for chunk in text_chunks:
                for ts in extract_response_timestamps(chunk):
                    _append_point(targets, ts, source=source)
    return targets


def _extract_temporal_targets_from_media(media: Dict[str, Any], *, source: str) -> List[Dict[str, Any]]:
    time_span = media.get("time_span")
    if not isinstance(time_span, list) or len(time_span) != 2:
        return []
    try:
        start = float(time_span[0])
        end = float(time_span[1])
    except (TypeError, ValueError):
        return []
    targets: List[Dict[str, Any]] = []
    _append_interval(targets, start, end, source=source)
    return targets


def extract_branch_temporal_targets(
    episode: Dict[str, Any],
    branch: Dict[str, Any],
) -> List[Dict[str, Any]]:
    branch_input = branch.get("input") or {}
    branch_media = branch_input.get("media") or {}
    branch_intervention = branch_input.get("intervention") or {}
    transform_type = safe_text(branch.get("transform_type"))
    swap_source = (branch_intervention.get("swap_source") or {}) if transform_type in {"swap_audio", "swap_video"} else {}

    targets: List[Dict[str, Any]] = []
    if swap_source:
        targets.extend(_extract_temporal_targets_from_steps(swap_source.get("source_meta") or {}, source="swap_source_steps"))
        targets.extend(_extract_temporal_targets_from_media(swap_source.get("media") or {}, source="swap_source_media"))
    if not targets:
        targets.extend(_extract_temporal_targets_from_steps(episode.get("source_meta") or {}, source="episode_steps"))
        targets.extend(_extract_temporal_targets_from_media(branch_media, source="branch_media"))

    deduped: List[Dict[str, Any]] = []
    seen: set[tuple[str, float, float]] = set()
    for target in targets:
        if target.get("kind") == "point":
            key = ("point", round(float(target["time"]), 3), round(float(target["time"]), 3))
        else:
            key = (
                "interval",
                round(float(target["start"]), 3),
                round(float(target["end"]), 3),
            )
        if key in seen:
            continue
        seen.add(key)
        deduped.append(target)
    return deduped


def _distance_to_target(prediction: float, target: Dict[str, Any]) -> float:
    if target.get("kind") == "interval":
        start = float(target.get("start", 0.0))
        end = float(target.get("end", start))
        if start <= prediction <= end:
            return 0.0
        return min(abs(prediction - start), abs(prediction - end))
    return abs(prediction - float(target.get("time", 0.0)))


def compute_temporal_alignment_reward(
    response_text: str,
    gt_targets: Sequence[Dict[str, Any]],
    *,
    penalty_weight: float = -1.0,
    missing_penalty: float = -5.0,
    extra_timestamp_penalty: float = -0.25,
) -> tuple[float, Dict[str, Any]]:
    targets = list(gt_targets)
    predictions = extract_response_timestamps(response_text)
    if not targets:
        return 0.0, {
            "predicted_timestamps": predictions,
            "target_count": 0,
            "mean_distance": 0.0,
        }
    if not predictions:
        return float(missing_penalty), {
            "predicted_timestamps": [],
            "target_count": len(targets),
            "mean_distance": None,
        }
    distances = [min(_distance_to_target(prediction, target) for target in targets) for prediction in predictions]
    mean_distance = float(sum(distances) / float(len(distances)))
    extra_count = max(0, len(predictions) - len(targets))
    reward = float(penalty_weight) * mean_distance + float(extra_timestamp_penalty) * float(extra_count)
    return reward, {
        "predicted_timestamps": predictions,
        "target_count": len(targets),
        "mean_distance": mean_distance,
        "extra_timestamp_count": int(extra_count),
    }


def collect_modality_token_ids(processor) -> List[int]:
    tokenizer = processor.tokenizer
    added_vocab = tokenizer.get_added_vocab()
    token_names = (
        "<|AUDIO|>",
        "<|IMAGE|>",
        "<|VIDEO|>",
        "<|vision_bos|>",
        "<|vision_eos|>",
        "<|vision_pad|>",
        "<|audio_bos|>",
        "<|audio_eos|>",
    )
    token_ids = {
        int(token_id)
        for token_name in token_names
        if (token_id := added_vocab.get(token_name)) is not None
    }
    for attr_name in ("audio_token_id", "image_token_id", "video_token_id"):
        token_id = getattr(tokenizer, attr_name, None)
        if token_id is not None:
            token_ids.add(int(token_id))
    return sorted(token_ids)


def build_attention_role_masks(
    *,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    prompt_length: int,
    modality_token_ids: Sequence[int],
    pad_token_id: Optional[int] = None,
) -> Dict[str, torch.Tensor]:
    token_ids = input_ids[0].detach().cpu()
    valid_mask = attention_mask[0].detach().cpu().bool()
    prompt_mask = torch.zeros_like(valid_mask, dtype=torch.bool)
    prompt_mask[: max(0, min(int(prompt_length), prompt_mask.shape[0]))] = True
    prompt_mask &= valid_mask

    if modality_token_ids:
        modality_ids = torch.tensor(list(modality_token_ids), dtype=token_ids.dtype)
        media_mask = (token_ids.unsqueeze(1) == modality_ids.unsqueeze(0)).any(dim=1)
    else:
        media_mask = torch.zeros_like(valid_mask, dtype=torch.bool)
    if pad_token_id is not None:
        media_mask &= token_ids != int(pad_token_id)

    text_prompt_mask = prompt_mask & ~media_mask
    generated_mask = valid_mask.clone()
    generated_mask[: max(0, min(int(prompt_length), generated_mask.shape[0]))] = False
    return {
        "prompt_mask": prompt_mask,
        "text_prompt_mask": text_prompt_mask,
        "media_prompt_mask": prompt_mask & media_mask,
        "generated_mask": generated_mask,
    }


def compute_overshadowing_reward(
    *,
    attentions,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    prompt_length: int,
    processor,
    tau: float = 2.0,
    penalty_scale: float = -1.0,
) -> tuple[float, Dict[str, Any]]:
    if attentions is None:
        return 0.0, {"available": False}
    if isinstance(attentions, tuple):
        if not attentions:
            return 0.0, {"available": False}
        layer_attn = attentions[-1]
    else:
        layer_attn = attentions
    if layer_attn is None or layer_attn.ndim != 4:
        return 0.0, {"available": False}

    modality_token_ids = collect_modality_token_ids(processor)
    masks = build_attention_role_masks(
        input_ids=input_ids,
        attention_mask=attention_mask,
        prompt_length=prompt_length,
        modality_token_ids=modality_token_ids,
        pad_token_id=getattr(processor.tokenizer, "pad_token_id", None),
    )
    gen_indices = torch.nonzero(masks["generated_mask"], as_tuple=False).flatten()
    text_indices = torch.nonzero(masks["text_prompt_mask"], as_tuple=False).flatten()
    media_indices = torch.nonzero(masks["media_prompt_mask"], as_tuple=False).flatten()
    if gen_indices.numel() == 0 or text_indices.numel() == 0 or media_indices.numel() == 0:
        return 0.0, {
            "available": False,
            "generated_count": int(gen_indices.numel()),
            "text_prompt_count": int(text_indices.numel()),
            "media_prompt_count": int(media_indices.numel()),
        }

    last_layer = layer_attn[0].detach().float().mean(dim=0).cpu()
    gen_to_prompt = last_layer.index_select(0, gen_indices)
    text_attn = float(gen_to_prompt.index_select(1, text_indices).mean().item())
    media_attn = float(gen_to_prompt.index_select(1, media_indices).mean().item())
    ratio = text_attn / max(media_attn, 1e-6)
    reward = float(penalty_scale) * float(max(0.0, ratio - float(tau)))
    return reward, {
        "available": True,
        "ratio": float(ratio),
        "text_attn": text_attn,
        "media_attn": media_attn,
        "generated_count": int(gen_indices.numel()),
        "text_prompt_count": int(text_indices.numel()),
        "media_prompt_count": int(media_indices.numel()),
    }


def compute_branch_answer_reward(
    *,
    episode: Dict[str, Any],
    branch: Dict[str, Any],
    raw_output: str,
) -> tuple[float, Dict[str, Any]]:
    clean_reference = safe_text(episode.get("clean_reference"))
    answer_type = safe_text(episode.get("answer_type"))
    parsed = parse_episode_output(branch, raw_output, clean_reference)
    role = safe_text(branch.get("branch_role"))

    if role == "full":
        clean_score = _full_branch_clean_score(answer_type, parsed, clean_reference)
        if bool(parsed.get("abstained")) or bool(parsed.get("uncertain")) or not bool(parsed.get("parseable")):
            reward = -1.0
        elif clean_score >= 0.999:
            reward = 1.0
        else:
            reward = max(-0.75, 2.0 * float(clean_score) - 1.0)
        return reward, {
            "role": role,
            "clean_score": float(clean_score),
            "parsed": parsed,
        }

    if role == "control_corruption":
        relation = _relation_satisfaction(branch, parsed, clean_reference, episode)
        return float(relation), {
            "role": role,
            "relation": float(relation),
            "parsed": parsed,
        }

    target_eval = _target_action(branch, parsed, clean_reference, episode)
    reward = _target_utility(
        branch=branch,
        target_eval=target_eval,
        episode=episode,
    )
    return float(reward), {
        "role": role,
        "parsed": parsed,
        "target_eval": target_eval,
    }


def resolve_branch_context(
    *,
    example: Dict[str, Any],
    resolver: MediaResolver,
) -> Dict[str, Any]:
    pair_like = {
        "dataset": example.get("dataset"),
        "sample_id": example.get("sample_id"),
        "input": (example.get("branch") or {}).get("input") or {},
    }
    prepared = resolver.prepare_pair_inputs(pair_like)
    prompt = ((example.get("branch") or {}).get("input") or {}).get("prompt") or {}
    return {
        "question_text": compose_question_text(prompt),
        "video_path": prepared.get("video_path"),
        "audio_array": prepared.get("audio_array"),
    }


def build_grpo_branch_examples(
    episodes: Sequence[Dict[str, Any]],
    *,
    split: Optional[str] = None,
    role_weights: Optional[Dict[str, float]] = None,
) -> List[Dict[str, Any]]:
    weights = {
        "full": 0.6,
        "control_corruption": 0.8,
        "target_corruption": 1.0,
    }
    if role_weights:
        weights.update({str(key): float(value) for key, value in role_weights.items()})

    examples: List[Dict[str, Any]] = []
    for episode in episodes:
        if split is not None and safe_text(episode.get("split")) != safe_text(split):
            continue
        for branch in iter_episode_branches(episode):
            branch_role = safe_text(branch.get("branch_role"))
            if branch_role not in weights:
                continue
            branch_legality = (
                ((episode.get("target_legality_spec") or {}).get("branch_legality") or {}).get(safe_text(branch.get("branch_id")))
                or {}
            )
            temporal_targets = extract_branch_temporal_targets(episode, branch)
            preferred_action = safe_text(branch_legality.get("preferred_action"))
            temporal_reward_enabled = bool(
                safe_text(episode.get("task_family")) == "temporal_alignment"
                and temporal_targets
                and (
                    branch_role in {"full", "control_corruption"}
                    or preferred_action == "retarget"
                )
            )
            example = {
                "example_id": f"{safe_text(episode.get('episode_id'))}:{safe_text(branch.get('branch_id'))}",
                "episode_id": episode.get("episode_id"),
                "sample_id": episode.get("sample_id"),
                "dataset": episode.get("dataset"),
                "split": episode.get("split"),
                "task_family": episode.get("task_family"),
                "target_modality": episode.get("target_modality"),
                "phenomenon": episode.get("phenomenon"),
                "answer_type": episode.get("answer_type"),
                "clean_reference": episode.get("clean_reference"),
                "branch": branch,
                "episode": episode,
                "branch_role": branch_role,
                "sample_weight": float(episode.get("sample_weight", 1.0) or 1.0) * float(weights[branch_role]),
                "temporal_targets": temporal_targets,
                "temporal_reward_enabled": bool(temporal_reward_enabled),
                "overshadow_reward_enabled": bool(safe_text(episode.get("phenomenon")) == "modality_overshadowing"),
                "preferred_action": preferred_action,
            }
            examples.append(example)
    return examples


def summarize_branch_examples(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    rows = list(rows)
    if not rows:
        return {
            "n_examples": 0,
            "branch_role_histogram": {},
            "task_family_histogram": {},
            "temporal_reward_enabled_rate": 0.0,
            "overshadow_reward_enabled_rate": 0.0,
        }

    role_histogram: Dict[str, int] = {}
    task_histogram: Dict[str, int] = {}
    for row in rows:
        role = safe_text(row.get("branch_role")) or "unknown"
        task = safe_text(row.get("task_family")) or "unknown"
        role_histogram[role] = role_histogram.get(role, 0) + 1
        task_histogram[task] = task_histogram.get(task, 0) + 1

    return {
        "n_examples": len(rows),
        "branch_role_histogram": role_histogram,
        "task_family_histogram": task_histogram,
        "temporal_reward_enabled_rate": sum(1 for row in rows if bool(row.get("temporal_reward_enabled"))) / float(len(rows)),
        "overshadow_reward_enabled_rate": sum(1 for row in rows if bool(row.get("overshadow_reward_enabled"))) / float(len(rows)),
    }
