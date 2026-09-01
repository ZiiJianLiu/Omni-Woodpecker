from __future__ import annotations

import copy
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn

from cirpo_omni.runtime import resolve_episode_contexts
from omni_dpo.media import MediaResolver
from omni_dpo.modeling import compose_question_text
from omni_dpo.preference_pairs import canonical_choice_answer, find_explicit_mismatch_choice, normalize_text, safe_text


GENERAL_ACTION_VALUES = (
    "retarget",
    "mismatch",
    "abstain",
)

GENERAL_DECISIVE_MODALITIES = (
    "audio",
    "visual",
    "cross_modal_relation",
    "unknown",
)

GENERAL_TRANSFORM_TYPES = (
    "swap_audio",
    "swap_video",
    "shift_audio",
    "drop_audio",
    "drop_video",
    "unknown",
)


def general_decisive_modality(episode: Dict[str, Any]) -> str:
    task_family = safe_text(episode.get("task_family"))
    target_modality = safe_text(episode.get("target_modality"))
    if task_family in {"audio_grounded_presence", "speaker_attribution"}:
        return "audio"
    if task_family == "visual_grounded_presence":
        return "visual"
    if task_family == "temporal_alignment":
        if target_modality == "audio_temporal":
            return "audio"
        if target_modality == "visual_temporal":
            return "visual"
        if target_modality == "joint_temporal":
            return "cross_modal_relation"
    if task_family == "av_matching":
        return "cross_modal_relation"
    if target_modality in {"audio", "audio_temporal"}:
        return "audio"
    if target_modality in {"visual", "visual_temporal"}:
        return "visual"
    if target_modality in {"joint_av", "joint_temporal"}:
        return "cross_modal_relation"
    return "unknown"


def branch_trace_vector(trace, layer_ids: Sequence[int]) -> torch.Tensor:
    parts: List[torch.Tensor] = []
    for layer_id in layer_ids:
        record = trace.layer_records.get(str(layer_id))
        if record is None:
            continue
        parts.append(record["pooled_post"].detach().float().reshape(-1).cpu())
    if not parts:
        raise RuntimeError(f"No layer features found for requested layers={list(layer_ids)}")
    return torch.cat(parts, dim=0)


def assemble_branch_feature(
    *,
    full_vec: torch.Tensor,
    target_vec: torch.Tensor,
    control_vec: torch.Tensor,
    donor_vec: torch.Tensor,
    has_control: bool,
    has_donor: bool,
    feature_layout: str,
) -> torch.Tensor:
    structural = torch.tensor(
        [1.0 if has_control else 0.0, 1.0 if has_donor else 0.0],
        dtype=torch.float32,
    )
    if feature_layout == "diffs_only":
        feature = torch.cat(
            [
                target_vec - full_vec,
                target_vec - control_vec,
                target_vec - donor_vec,
                structural,
            ],
            dim=0,
        )
    else:
        feature = torch.cat(
            [
                full_vec,
                target_vec,
                control_vec,
                donor_vec,
                target_vec - full_vec,
                target_vec - control_vec,
                target_vec - donor_vec,
                structural,
            ],
            dim=0,
        )
    return feature.float()


def resolve_general_branch_contexts(
    *,
    episode: Dict[str, Any],
    branch: Dict[str, Any],
    resolver: MediaResolver,
) -> Dict[str, Dict[str, Any]]:
    contexts = resolve_episode_contexts(episode=episode, resolver=resolver)
    full_branch_id = safe_text((episode.get("full_branch") or {}).get("branch_id"))
    target_branch_id = safe_text(branch.get("branch_id"))
    out = {
        "full": contexts[full_branch_id],
        "target": contexts[target_branch_id],
    }

    control_branch = None
    for candidate in (episode.get("cf_branch_a") or {}, episode.get("cf_branch_b") or {}):
        if safe_text(candidate.get("branch_role")) == "control_corruption":
            control_branch = candidate
            break
    if control_branch is not None:
        control_branch_id = safe_text(control_branch.get("branch_id"))
        if control_branch_id in contexts:
            out["control"] = contexts[control_branch_id]

    swap_source = ((((branch.get("input") or {}).get("intervention")) or {}).get("swap_source") or {})
    donor_dataset = safe_text(swap_source.get("dataset"))
    donor_sample_id = safe_text(swap_source.get("sample_id"))
    if donor_dataset and donor_sample_id:
        donor_pair = {
            "dataset": donor_dataset,
            "sample_id": donor_sample_id,
            "input": {
                "prompt": copy.deepcopy(swap_source.get("prompt", {}) or {}),
                "media": copy.deepcopy(swap_source.get("media", {}) or {}),
                "intervention": None,
            },
        }
        donor_prepared = resolver.prepare_pair_inputs(donor_pair)
        out["donor"] = {
            "question_text": compose_question_text(swap_source.get("prompt", {}) or {}),
            "video_path": donor_prepared.get("video_path"),
            "audio_array": donor_prepared.get("audio_array"),
        }
    return out


def resolve_general_branch_legality(episode: Dict[str, Any], branch: Dict[str, Any]) -> Dict[str, Any]:
    branch_legality = ((episode.get("target_legality_spec") or {}).get("branch_legality") or {})
    return branch_legality.get(safe_text(branch.get("branch_id"))) or {}


def _swap_source_payload(branch: Dict[str, Any]) -> Dict[str, Any]:
    return ((((branch.get("input") or {}).get("intervention")) or {}).get("swap_source") or {})


def _extract_choice_answer_text(answer: str) -> str:
    candidate = safe_text(answer)
    if not candidate:
        return ""
    if ":" not in candidate:
        return candidate
    suffix = candidate.split(":", 1)[1].strip()
    if ". " in suffix:
        return suffix.split(". ", 1)[1].strip()
    return suffix


def _swap_source_answer_text(branch: Dict[str, Any]) -> str:
    swap_source = _swap_source_payload(branch)
    clean_answer = swap_source.get("clean_answer") or {}
    answer_text = safe_text(clean_answer.get("answer_text"))
    if answer_text:
        return answer_text
    return _extract_choice_answer_text(safe_text(swap_source.get("canonical_answer")))


def _find_choice_by_text(prompt: Dict[str, Any], answer_text: str) -> Optional[Dict[str, Any]]:
    normalized_target = normalize_text(answer_text)
    if not normalized_target:
        return None
    for choice in prompt.get("choices") or []:
        if normalize_text(safe_text(choice.get("text"))) == normalized_target:
            return copy.deepcopy(choice)
    return None


def resolve_retarget_answer(episode: Dict[str, Any], branch: Dict[str, Any]) -> Tuple[str, str]:
    prompt = ((branch.get("input") or {}).get("prompt") or {})
    answer_text = _swap_source_answer_text(branch)
    if answer_text:
        choice = _find_choice_by_text(prompt, answer_text)
        if choice is not None:
            return canonical_choice_answer(choice), "current_choice_match"

    legality = resolve_general_branch_legality(episode, branch)
    preferred_answer = safe_text(legality.get("preferred_answer")) or safe_text(branch.get("preferred_answer"))
    if preferred_answer:
        return preferred_answer, "preferred_answer"

    canonical_answer = safe_text(_swap_source_payload(branch).get("canonical_answer"))
    if canonical_answer:
        return canonical_answer, "swap_source_canonical"

    return "", "none"


def resolve_mismatch_answer(episode: Dict[str, Any], branch: Dict[str, Any]) -> Tuple[str, str]:
    legality = resolve_general_branch_legality(episode, branch)
    mismatch_answer = safe_text(legality.get("mismatch_answer"))
    if mismatch_answer:
        return mismatch_answer, "branch_legality"

    preferred_action = safe_text(legality.get("preferred_action"))
    preferred_answer = safe_text(legality.get("preferred_answer")) or safe_text(branch.get("preferred_answer"))
    if preferred_action == "mismatch" and preferred_answer:
        return preferred_answer, "preferred_answer"

    prompt = ((branch.get("input") or {}).get("prompt") or {})
    mismatch_choice = find_explicit_mismatch_choice({"prompt": prompt})
    if mismatch_choice is not None:
        return canonical_choice_answer(mismatch_choice), "explicit_mismatch_choice"

    return "", "none"


class GeneralActionPolicyNet(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int) -> None:
        super().__init__()
        if hidden_dim > 0:
            self.encoder = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.GELU(),
                nn.LayerNorm(hidden_dim),
            )
            rep_dim = hidden_dim
        else:
            self.encoder = nn.Identity()
            rep_dim = input_dim
        self.action_head = nn.Linear(rep_dim, len(GENERAL_ACTION_VALUES))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rep = self.encoder(x)
        return self.action_head(rep)


def normalize_features(
    train_x: torch.Tensor,
    val_x: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
    mean = train_x.mean(dim=0, keepdim=True)
    std = train_x.std(dim=0, keepdim=True, unbiased=False).clamp_min(1e-6)
    return (train_x - mean) / std, (val_x - mean) / std, {
        "feature_mean_abs": float(mean.abs().mean().item()),
        "feature_std_mean": float(std.mean().item()),
        "mean": mean.cpu(),
        "std": std.cpu(),
    }


def evaluate_action_logits(
    logits: torch.Tensor,
    labels: torch.Tensor,
    label_names: Sequence[str],
) -> Dict[str, Any]:
    if labels.numel() == 0:
        return {
            "n_samples": 0,
            "accuracy": 0.0,
            "macro_accuracy": 0.0,
            "majority_accuracy": 0.0,
            "class_accuracy": {label: 0.0 for label in label_names},
            "label_counts": {label: 0 for label in label_names},
            "prediction_counts": {label: 0 for label in label_names},
            "confusion": {
                label: {other: 0 for other in label_names}
                for label in label_names
            },
        }
    pred = logits.argmax(dim=-1)
    accuracy = float((pred == labels).float().mean().item())
    counts = torch.bincount(labels, minlength=len(label_names))
    pred_counts = torch.bincount(pred, minlength=len(label_names))
    majority_accuracy = float(counts.max().item() / max(1, int(labels.numel())))
    class_accuracy: Dict[str, float] = {}
    confusion: Dict[str, Dict[str, int]] = {}
    for idx, label_name in enumerate(label_names):
        mask = labels == idx
        if int(mask.sum().item()) == 0:
            class_accuracy[label_name] = 0.0
        else:
            class_accuracy[label_name] = float((pred[mask] == labels[mask]).float().mean().item())
        confusion[label_name] = {}
        for other_idx, other_label in enumerate(label_names):
            confusion[label_name][other_label] = int(((labels == idx) & (pred == other_idx)).sum().item())
    present_acc = [class_accuracy[label_names[idx]] for idx in range(len(label_names)) if int(counts[idx].item()) > 0]
    macro_accuracy = float(sum(present_acc) / max(1, len(present_acc)))
    return {
        "n_samples": int(labels.numel()),
        "accuracy": accuracy,
        "macro_accuracy": macro_accuracy,
        "majority_accuracy": majority_accuracy,
        "class_accuracy": class_accuracy,
        "label_counts": {label_names[idx]: int(counts[idx].item()) for idx in range(len(label_names))},
        "prediction_counts": {label_names[idx]: int(pred_counts[idx].item()) for idx in range(len(label_names))},
        "confusion": confusion,
    }


def class_weights(labels: torch.Tensor, num_classes: int) -> torch.Tensor:
    counts = torch.bincount(labels, minlength=num_classes).float()
    weights = counts.sum() / counts.clamp_min(1.0)
    return weights / weights.mean().clamp_min(1e-8)
