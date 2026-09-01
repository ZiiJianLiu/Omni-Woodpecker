from __future__ import annotations

import copy
import hashlib
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from omni_dpo.preference_pairs import (
    ABSTAIN_TEMPLATE,
    SwapSelector,
    canonical_answer,
    canonical_choice_answer,
    enrich_intervention,
    find_explicit_mismatch_choice,
    is_abstention_text,
    normalize_text,
    read_jsonl,
    safe_text,
    write_json,
    write_jsonl,
)


PHENOMENON_TEMPORAL = "temporal_misalignment"
PHENOMENON_OVERSHADOW = "modality_overshadowing"

CCI_TASK_FAMILIES = {
    "audio_grounded_presence",
    "speaker_attribution",
    "av_matching",
}
CCI_DECISIVE_MODALITY_AUDIO = "audio"
CCI_DECISIVE_MODALITY_VISUAL = "visual"
CCI_DECISIVE_MODALITY_RELATION = "cross_modal_relation"
CCI_EXPECTED_TRANSITION_PRESERVE = "preserve"
CCI_EXPECTED_TRANSITION_RETARGET = "retarget"
CCI_EXPECTED_TRANSITION_MISMATCH = "mismatch"
CCI_EXPECTED_TRANSITION_ABSTAIN = "abstain"
CCI_TRAINING_BUCKET_RETARGET = "retarget_positive"
CCI_TRAINING_BUCKET_MISMATCH = "mismatch_legal"
CCI_TRAINING_BUCKET_PRESERVE = "preserve_control"

DEFAULT_MISMATCH_CHOICE_TEXT = "None of the above"
DEFAULT_ABSTAIN_CHOICE_TEXT = "Insufficient information from the available modalities."
AUTO_REPAIRABLE_EPISODE_TYPES = {
    "joint_swap_pair",
    "joint_drop_pair",
    "joint_temporal_shift_dropaudio",
    "joint_temporal_shift_dropvideo",
}
ABSTAIN_REPAIRABLE_EPISODE_TYPES = {
    "visual_target_drop_ctrl",
}
RETARGET_REPAIRABLE_EPISODE_TYPES = {
    "audio_target_swap_ctrl",
    "visual_target_swap_ctrl",
    "temporal_swap_ctrl",
    "joint_temporal_swapaudio_dropvideo",
    "joint_temporal_swapvideo_dropaudio",
}
AUDIT_REQUIRED_EPISODE_TYPES = {
    "temporal_shift_ctrl",
    "temporal_dropaudio_ctrl",
}


def load_episode_rows(path: Path) -> List[Dict[str, Any]]:
    return read_jsonl(path)


def _stable_hash(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:12]


def unit_in_scope(unit: Dict[str, Any]) -> bool:
    task_family = safe_text(unit.get("task_family"))
    target_modality = safe_text(unit.get("target_modality"))
    if task_family in {"audio_grounded_presence", "visual_grounded_presence", "speaker_attribution", "av_matching"}:
        return True
    if task_family == "temporal_alignment" and target_modality in {"audio_temporal", "joint_temporal"}:
        return True
    return False


def phenomenon_for_unit(unit: Dict[str, Any]) -> Optional[str]:
    task_family = safe_text(unit.get("task_family"))
    target_modality = safe_text(unit.get("target_modality"))
    if task_family == "temporal_alignment" and target_modality in {"audio_temporal", "joint_temporal"}:
        return PHENOMENON_TEMPORAL
    if task_family in {"audio_grounded_presence", "visual_grounded_presence", "speaker_attribution", "av_matching"}:
        return PHENOMENON_OVERSHADOW
    return None


def _branch_input(unit: Dict[str, Any], intervention: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    return {
        "prompt": copy.deepcopy(unit.get("prompt", {}) or {}),
        "media": copy.deepcopy(unit.get("media", {}) or {}),
        "intervention": copy.deepcopy(intervention) if intervention else None,
    }


def _branch_record(
    *,
    unit: Dict[str, Any],
    branch_id: str,
    branch_role: str,
    view: Optional[Dict[str, Any]],
    swap_selector: Optional[SwapSelector],
    preferred_answer: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    intervention = copy.deepcopy(view) if view is not None else None
    transform_type = safe_text(((intervention or {}).get("transform") or {}).get("type"))
    if transform_type in {"swap_audio", "swap_video"} and intervention is not None:
        if swap_selector is None:
            return None
        try:
            intervention = enrich_intervention(unit, intervention, swap_selector)
        except ValueError:
            return None
    transform = ((intervention or {}).get("transform") or {})
    return {
        "branch_id": branch_id,
        "branch_role": branch_role,
        "view_id": safe_text((view or {}).get("view_id")),
        "transform_type": safe_text(transform.get("type")) if transform else "none",
        "expected_relation": safe_text((view or {}).get("expected_relation")) if view else "clean_reference",
        "preferred_answer": preferred_answer,
        "input": _branch_input(unit, intervention),
    }


def _view_map(unit: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    return {
        safe_text(view.get("view_id")): copy.deepcopy(view)
        for view in unit.get("counterfactual_views", []) or []
        if safe_text(view.get("view_id"))
    }


def _preferred_mismatch_answer(unit: Dict[str, Any]) -> Optional[str]:
    mismatch_choice = find_explicit_mismatch_choice(unit)
    if mismatch_choice is None:
        return None
    return canonical_choice_answer(mismatch_choice)


def _episode_base(unit: Dict[str, Any], phenomenon: str) -> Dict[str, Any]:
    return {
        "unit_id": unit.get("unit_id"),
        "sample_id": unit.get("sample_id"),
        "split": unit.get("split"),
        "dataset": unit.get("dataset"),
        "task_family": unit.get("task_family"),
        "target_modality": unit.get("target_modality"),
        "sample_weight": float(unit.get("sample_weight", 1.0) or 1.0),
        "phenomenon": phenomenon,
        "answer_type": safe_text((unit.get("reward_spec") or {}).get("answer_type")),
        "clean_reference": canonical_answer(unit),
        "reward_weights": copy.deepcopy((unit.get("reward_spec") or {}).get("weights") or {}),
        "source_meta": copy.deepcopy(unit.get("source_meta", {}) or {}),
    }


def _make_episode(
    *,
    unit: Dict[str, Any],
    phenomenon: str,
    episode_type: str,
    full_branch: Dict[str, Any],
    cf_branch_a: Dict[str, Any],
    cf_branch_b: Dict[str, Any],
) -> Dict[str, Any]:
    base = _episode_base(unit, phenomenon)
    episode_id = (
        f"{safe_text(unit.get('sample_id'))}__{episode_type}"
        f"__{_stable_hash(safe_text(cf_branch_a.get('branch_id')) + ':' + safe_text(cf_branch_b.get('branch_id')))}"
    )
    base.update(
        {
            "episode_id": episode_id,
            "episode_type": episode_type,
            "full_branch": full_branch,
            "cf_branch_a": cf_branch_a,
            "cf_branch_b": cf_branch_b,
            "relation_spec": {
                "full_branch_id": safe_text(full_branch.get("branch_id")),
                "cf_branch_ids": [
                    safe_text(cf_branch_a.get("branch_id")),
                    safe_text(cf_branch_b.get("branch_id")),
                ],
                "requires_clean_match": True,
                "requires_control_keep": any(
                    branch.get("branch_role") == "control_corruption"
                    for branch in (cf_branch_a, cf_branch_b)
                ),
                "requires_temporal_break": phenomenon == PHENOMENON_TEMPORAL,
            },
        }
    )
    return base


def _build_unit_episodes(unit: Dict[str, Any]) -> List[Dict[str, Any]]:
    return _build_unit_episodes_with_selector(unit=unit, swap_selector=None)


def _build_unit_episodes_with_selector(
    *,
    unit: Dict[str, Any],
    swap_selector: Optional[SwapSelector],
) -> List[Dict[str, Any]]:
    if not unit_in_scope(unit):
        return []
    phenomenon = phenomenon_for_unit(unit)
    if phenomenon is None:
        return []
    views = _view_map(unit)
    full_branch = _branch_record(
        unit=unit,
        branch_id="full",
        branch_role="full",
        view=None,
        swap_selector=swap_selector,
        preferred_answer=canonical_answer(unit),
    )
    if full_branch is None:
        return []
    mismatch_answer = _preferred_mismatch_answer(unit)
    target_modality = safe_text(unit.get("target_modality"))
    task_family = safe_text(unit.get("task_family"))
    dataset = safe_text(unit.get("dataset"))
    answer_type = safe_text((unit.get("reward_spec") or {}).get("answer_type") or (unit.get("prompt") or {}).get("answer_format"))
    episodes: List[Dict[str, Any]] = []
    enable_temporal_swap_retarget = (
        dataset == "omnivideobench"
        and task_family == "temporal_alignment"
        and answer_type == "choice_label"
        and target_modality in {"audio_temporal", "joint_temporal"}
    )

    def maybe_add(
        *,
        episode_type: str,
        a_id: str,
        a_role: str,
        b_id: str,
        b_role: str,
        a_pref: Optional[str] = None,
        b_pref: Optional[str] = None,
    ) -> None:
        view_a = views.get(a_id)
        view_b = views.get(b_id)
        if view_a is None or view_b is None:
            return
        branch_a = _branch_record(
            unit=unit,
            branch_id=a_id,
            branch_role=a_role,
            view=view_a,
            swap_selector=swap_selector,
            preferred_answer=a_pref,
        )
        branch_b = _branch_record(
            unit=unit,
            branch_id=b_id,
            branch_role=b_role,
            view=view_b,
            swap_selector=swap_selector,
            preferred_answer=b_pref,
        )
        if branch_a is None or branch_b is None:
            return
        episodes.append(
            _make_episode(
                unit=unit,
                phenomenon=phenomenon,
                episode_type=episode_type,
                full_branch=copy.deepcopy(full_branch),
                cf_branch_a=branch_a,
                cf_branch_b=branch_b,
            )
        )

    if task_family in {"audio_grounded_presence", "speaker_attribution"}:
        maybe_add(
            episode_type="audio_target_drop_ctrl",
            a_id="drop_audio",
            a_role="target_corruption",
            b_id="drop_video_ctrl",
            b_role="control_corruption",
        )
        maybe_add(
            episode_type="audio_target_swap_ctrl",
            a_id="swap_audio",
            a_role="target_corruption",
            b_id="drop_video_ctrl",
            b_role="control_corruption",
            a_pref=mismatch_answer,
        )
        return episodes

    if task_family == "visual_grounded_presence":
        maybe_add(
            episode_type="visual_target_drop_ctrl",
            a_id="drop_video",
            a_role="target_corruption",
            b_id="drop_audio_ctrl",
            b_role="control_corruption",
        )
        maybe_add(
            episode_type="visual_target_swap_ctrl",
            a_id="swap_video",
            a_role="target_corruption",
            b_id="drop_audio_ctrl",
            b_role="control_corruption",
            a_pref=mismatch_answer,
        )
        return episodes

    if target_modality == "audio_temporal":
        if enable_temporal_swap_retarget:
            maybe_add(
                episode_type="temporal_swap_ctrl",
                a_id="swap_audio",
                a_role="target_corruption",
                b_id="drop_video_ctrl",
                b_role="control_corruption",
            )
        maybe_add(
            episode_type="temporal_shift_ctrl",
            a_id="shift_audio",
            a_role="target_corruption",
            b_id="drop_video_ctrl",
            b_role="control_corruption",
            a_pref=mismatch_answer,
        )
        maybe_add(
            episode_type="temporal_dropaudio_ctrl",
            a_id="drop_audio",
            a_role="target_corruption",
            b_id="drop_video_ctrl",
            b_role="control_corruption",
        )
        return episodes

    if target_modality == "joint_av":
        maybe_add(
            episode_type="joint_swap_pair",
            a_id="swap_audio",
            a_role="target_corruption",
            b_id="swap_video",
            b_role="target_corruption",
            a_pref=mismatch_answer,
            b_pref=mismatch_answer,
        )
        maybe_add(
            episode_type="joint_drop_pair",
            a_id="drop_audio",
            a_role="target_corruption",
            b_id="drop_video",
            b_role="target_corruption",
        )
        return episodes

    if target_modality == "joint_temporal":
        if enable_temporal_swap_retarget:
            maybe_add(
                episode_type="joint_temporal_swapaudio_dropvideo",
                a_id="swap_audio",
                a_role="target_corruption",
                b_id="drop_video",
                b_role="target_corruption",
            )
            maybe_add(
                episode_type="joint_temporal_swapvideo_dropaudio",
                a_id="swap_video",
                a_role="target_corruption",
                b_id="drop_audio",
                b_role="target_corruption",
            )
        maybe_add(
            episode_type="joint_temporal_shift_dropaudio",
            a_id="shift_audio",
            a_role="target_corruption",
            b_id="drop_audio",
            b_role="target_corruption",
            a_pref=mismatch_answer,
        )
        maybe_add(
            episode_type="joint_temporal_shift_dropvideo",
            a_id="shift_audio",
            a_role="target_corruption",
            b_id="drop_video",
            b_role="target_corruption",
            a_pref=mismatch_answer,
        )
        return episodes

    return episodes


def _cf_branches(episode: Dict[str, Any]) -> List[Dict[str, Any]]:
    return [episode.get("cf_branch_a") or {}, episode.get("cf_branch_b") or {}]


def _all_branches(episode: Dict[str, Any]) -> List[Dict[str, Any]]:
    return [episode.get("full_branch") or {}, *_cf_branches(episode)]


def _target_branches(episode: Dict[str, Any]) -> List[Dict[str, Any]]:
    return [branch for branch in _cf_branches(episode) if safe_text(branch.get("branch_role")) == "target_corruption"]


def _control_branches(episode: Dict[str, Any]) -> List[Dict[str, Any]]:
    return [branch for branch in _cf_branches(episode) if safe_text(branch.get("branch_role")) == "control_corruption"]


def _episode_prompt(episode: Dict[str, Any]) -> Dict[str, Any]:
    return (((episode.get("full_branch") or {}).get("input") or {}).get("prompt") or {})


def _has_target_corruption(episode: Dict[str, Any]) -> bool:
    return bool(_target_branches(episode))


def _has_control_corruption(episode: Dict[str, Any]) -> bool:
    return bool(_control_branches(episode))


def _requires_temporal_break(episode: Dict[str, Any]) -> bool:
    relation_spec = episode.get("relation_spec") or {}
    phenomenon = safe_text(episode.get("phenomenon"))
    return bool(relation_spec.get("requires_temporal_break")) or phenomenon == PHENOMENON_TEMPORAL


def _is_choice_label_episode(episode: Dict[str, Any]) -> bool:
    return safe_text(episode.get("answer_type")) == "choice_label"


def _is_joint_episode(episode: Dict[str, Any]) -> bool:
    return safe_text(episode.get("episode_type")).startswith("joint_")


def _target_transform_types(episode: Dict[str, Any]) -> List[str]:
    return [safe_text(branch.get("transform_type")) for branch in _target_branches(episode)]


def _next_choice_label(choices: Sequence[Dict[str, Any]]) -> str:
    labels = [safe_text(choice.get("label")).upper() for choice in choices]
    valid = [ord(label) for label in labels if len(label) == 1 and "A" <= label <= "Z"]
    next_ord = (max(valid) + 1) if valid else ord("A")
    if next_ord <= ord("Z"):
        return chr(next_ord)
    suffix = next_ord - ord("Z")
    return f"Z{suffix}"


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


def _canonical_mismatch_answer(prompt: Dict[str, Any]) -> str:
    mismatch_choice = find_explicit_mismatch_choice({"prompt": prompt})
    if mismatch_choice is None:
        return ""
    return canonical_choice_answer(mismatch_choice)


def _find_explicit_abstain_choice(prompt: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    for choice in prompt.get("choices") or []:
        text = safe_text(choice.get("text"))
        if not text:
            continue
        if is_abstention_text(text):
            return copy.deepcopy(choice)
    return None


def _canonical_abstain_answer(prompt: Dict[str, Any]) -> str:
    abstain_choice = _find_explicit_abstain_choice(prompt)
    if abstain_choice is not None:
        return canonical_choice_answer(abstain_choice)
    return ""


def _find_choice_by_text(prompt: Dict[str, Any], answer_text: str) -> Optional[Dict[str, Any]]:
    normalized_target = normalize_text(answer_text)
    if not normalized_target:
        return None
    for choice in prompt.get("choices") or []:
        if normalize_text(safe_text(choice.get("text"))) == normalized_target:
            return copy.deepcopy(choice)
    return None


def _append_choice_to_all_branches(episode: Dict[str, Any], choice: Dict[str, Any]) -> None:
    label = safe_text(choice.get("label")).upper()
    for branch in _all_branches(episode):
        branch_prompt = (((branch.get("input") or {}).get("prompt")) or {})
        branch_choices = branch_prompt.setdefault("choices", [])
        if any(safe_text(existing.get("label")).upper() == label for existing in branch_choices):
            continue
        branch_choices.append(copy.deepcopy(choice))


def _inject_mismatch_choice(episode: Dict[str, Any], *, mismatch_text: str = DEFAULT_MISMATCH_CHOICE_TEXT) -> str:
    prompt = _episode_prompt(episode)
    existing_choice = find_explicit_mismatch_choice({"prompt": prompt})
    if existing_choice is not None:
        mismatch_answer = canonical_choice_answer(existing_choice)
    else:
        choice_label = _next_choice_label(prompt.get("choices") or [])
        new_choice = {"label": choice_label, "text": mismatch_text}
        mismatch_answer = canonical_choice_answer(new_choice)
        for branch in _all_branches(episode):
            branch_prompt = (((branch.get("input") or {}).get("prompt")) or {})
            branch_choices = branch_prompt.setdefault("choices", [])
            if not any(safe_text(choice.get("label")).upper() == choice_label for choice in branch_choices):
                branch_choices.append(copy.deepcopy(new_choice))
    for branch in _target_branches(episode):
        branch["preferred_answer"] = mismatch_answer
    return mismatch_answer


def _inject_abstain_choice(episode: Dict[str, Any], *, abstain_text: str = DEFAULT_ABSTAIN_CHOICE_TEXT) -> str:
    prompt = _episode_prompt(episode)
    existing_choice = _find_explicit_abstain_choice(prompt)
    if existing_choice is not None:
        abstain_answer = canonical_choice_answer(existing_choice)
    else:
        choice_label = _next_choice_label(prompt.get("choices") or [])
        new_choice = {"label": choice_label, "text": abstain_text}
        abstain_answer = canonical_choice_answer(new_choice)
        _append_choice_to_all_branches(episode, new_choice)
    for branch in _target_branches(episode):
        if safe_text(branch.get("transform_type")) in {"drop_audio", "drop_video"} and not safe_text(branch.get("preferred_answer")):
            branch["preferred_answer"] = abstain_answer
    return abstain_answer


def _swap_source_answer_text(branch: Dict[str, Any]) -> str:
    swap_source = ((((branch.get("input") or {}).get("intervention")) or {}).get("swap_source") or {})
    clean_answer = swap_source.get("clean_answer") or {}
    answer_text = safe_text(clean_answer.get("answer_text"))
    if answer_text:
        return answer_text
    return _extract_choice_answer_text(safe_text(swap_source.get("canonical_answer")))


def _inject_retarget_choice(episode: Dict[str, Any]) -> Optional[str]:
    prompt = _episode_prompt(episode)
    clean_reference = safe_text(episode.get("clean_reference"))
    clean_answer_text = _extract_choice_answer_text(clean_reference)
    target_branches = [
        branch
        for branch in _target_branches(episode)
        if safe_text(branch.get("transform_type")) in {"swap_audio", "swap_video"}
    ]
    repaired_answers: List[str] = []
    for branch in target_branches:
        answer_text = _swap_source_answer_text(branch)
        if not answer_text:
            continue
        if normalize_text(answer_text) == normalize_text(clean_answer_text):
            continue
        choice = _find_choice_by_text(prompt, answer_text)
        if choice is None:
            choice = {
                "label": _next_choice_label(prompt.get("choices") or []),
                "text": answer_text,
            }
            _append_choice_to_all_branches(episode, choice)
            prompt = _episode_prompt(episode)
        preferred_answer = canonical_choice_answer(choice)
        if normalize_text(preferred_answer) == normalize_text(clean_reference):
            continue
        branch["preferred_answer"] = preferred_answer
        repaired_answers.append(preferred_answer)
    if not repaired_answers:
        return None
    return repaired_answers[0]


def _can_inject_retarget_choice(episode: Dict[str, Any]) -> bool:
    if safe_text(episode.get("episode_type")) not in RETARGET_REPAIRABLE_EPISODE_TYPES:
        return False
    if safe_text(episode.get("answer_type")) != "choice_label":
        return False
    clean_reference = safe_text(episode.get("clean_reference"))
    clean_answer_text = _extract_choice_answer_text(clean_reference)
    for branch in _target_branches(episode):
        if safe_text(branch.get("transform_type")) not in {"swap_audio", "swap_video"}:
            continue
        answer_text = _swap_source_answer_text(branch)
        if not answer_text:
            continue
        if normalize_text(answer_text) == normalize_text(clean_answer_text):
            continue
        return True
    return False


def _can_inject_abstain_choice(episode: Dict[str, Any]) -> bool:
    if safe_text(episode.get("episode_type")) not in ABSTAIN_REPAIRABLE_EPISODE_TYPES:
        return False
    if safe_text(episode.get("answer_type")) != "choice_label":
        return False
    if _find_explicit_abstain_choice(_episode_prompt(episode)) is not None:
        return False
    for branch in _target_branches(episode):
        if safe_text(branch.get("transform_type")) in {"drop_audio", "drop_video"} and not safe_text(branch.get("preferred_answer")):
            return True
    return False


def inspect_episode_answer_space(episode: Dict[str, Any]) -> Dict[str, Any]:
    answer_type = safe_text(episode.get("answer_type"))
    has_target = _has_target_corruption(episode)
    has_control = _has_control_corruption(episode)
    has_temporal = _requires_temporal_break(episode)
    has_explicit_mismatch_choice = find_explicit_mismatch_choice({"prompt": _episode_prompt(episode)}) is not None
    has_explicit_abstain_choice = _find_explicit_abstain_choice(_episode_prompt(episode)) is not None
    has_explicit_change_exit = has_explicit_mismatch_choice or has_explicit_abstain_choice
    target_branches = _target_branches(episode)
    target_preferred_answer_count = sum(1 for branch in target_branches if safe_text(branch.get("preferred_answer")))
    choice_target_under_specified = (
        answer_type == "choice_label"
        and has_target
        and not has_explicit_change_exit
        and target_preferred_answer_count <= 0
    )
    issue_types: List[str] = []
    if not has_control:
        issue_types.append("no_control_branch")
    if answer_type == "choice_label" and has_target:
        if not has_explicit_mismatch_choice and not has_explicit_abstain_choice:
            issue_types.append("missing_mismatch_choice")
        if not has_explicit_abstain_choice and safe_text(episode.get("episode_type")) in ABSTAIN_REPAIRABLE_EPISODE_TYPES:
            issue_types.append("missing_abstain_choice")
        if target_preferred_answer_count <= 0:
            issue_types.append("missing_target_preferred_answer")
        if choice_target_under_specified:
            issue_types.append("choice_target_under_specified")
            if has_temporal:
                issue_types.append("temporal_choice_under_specified")
        target_transform_types = _target_transform_types(episode)
        if any(transform in {"swap_audio", "swap_video"} for transform in target_transform_types) and target_preferred_answer_count <= 0:
            issue_types.append("swap_without_explicit_retarget")
        if any(transform in {"drop_audio", "drop_video"} for transform in target_transform_types) and not (
            has_explicit_change_exit or target_preferred_answer_count > 0
        ):
            issue_types.append("drop_without_abstain_or_mismatch_exit")

    episode_type = safe_text(episode.get("episode_type"))
    if answer_type != "choice_label" or not has_target:
        recommended_action = "keep"
    elif episode_type in RETARGET_REPAIRABLE_EPISODE_TYPES and target_preferred_answer_count <= 0 and _can_inject_retarget_choice(episode):
        recommended_action = "inject_retarget_choice"
    elif episode_type in ABSTAIN_REPAIRABLE_EPISODE_TYPES and target_preferred_answer_count <= 0 and _can_inject_abstain_choice(episode):
        recommended_action = "inject_abstain_choice"
    elif episode_type in AUTO_REPAIRABLE_EPISODE_TYPES and not has_explicit_mismatch_choice:
        recommended_action = "inject_mismatch_choice"
    elif episode_type in AUDIT_REQUIRED_EPISODE_TYPES and choice_target_under_specified:
        recommended_action = "audit_required"
    elif choice_target_under_specified:
        recommended_action = "exclude_from_core"
    else:
        recommended_action = "keep"

    return {
        "answer_type": answer_type,
        "has_target_corruption": has_target,
        "has_explicit_mismatch_choice": has_explicit_mismatch_choice,
        "has_explicit_abstain_choice": has_explicit_abstain_choice,
        "target_preferred_answer_count": target_preferred_answer_count,
        "choice_target_under_specified": choice_target_under_specified,
        "issue_types": issue_types,
        "recommended_action": recommended_action,
        "requires_control_keep": has_control,
        "requires_temporal_break": has_temporal,
    }


def _branch_legality_snapshot(episode: Dict[str, Any], branch: Dict[str, Any]) -> Dict[str, Any]:
    prompt = (((branch.get("input") or {}).get("prompt")) or {})
    clean_reference = safe_text(episode.get("clean_reference"))
    preferred_answer = safe_text(branch.get("preferred_answer"))
    mismatch_answer = _canonical_mismatch_answer(prompt)
    abstain_answer = _canonical_abstain_answer(prompt)
    transform_type = safe_text(branch.get("transform_type"))
    legal_actions: List[str] = []
    preferred_action = "abstain"
    preferred_is_mismatch = bool(
        preferred_answer and mismatch_answer and normalize_text(preferred_answer) == normalize_text(mismatch_answer)
    )
    preferred_is_abstain = bool(
        preferred_answer
        and (
            (abstain_answer and normalize_text(preferred_answer) == normalize_text(abstain_answer))
            or is_abstention_text(preferred_answer)
        )
    )

    if transform_type in {"swap_audio", "swap_video"}:
        if preferred_is_abstain:
            legal_actions = ["abstain"]
            preferred_action = "abstain"
            if mismatch_answer:
                legal_actions.insert(0, "mismatch")
        elif preferred_answer and not preferred_is_mismatch:
            legal_actions = ["retarget"]
            preferred_action = "retarget"
            if mismatch_answer:
                legal_actions.append("mismatch")
        else:
            legal_actions = ["mismatch"] if mismatch_answer else ["change"]
            preferred_action = "mismatch" if mismatch_answer else "change"
    elif transform_type == "shift_audio":
        if preferred_is_abstain:
            legal_actions = ["mismatch", "abstain"] if mismatch_answer else ["abstain"]
            preferred_action = "abstain"
        elif preferred_answer and not preferred_is_mismatch:
            legal_actions = ["retarget", "abstain"]
            preferred_action = "retarget"
        elif mismatch_answer:
            legal_actions = ["mismatch", "abstain"]
            preferred_action = "mismatch" if preferred_answer else "abstain"
        else:
            legal_actions = ["abstain"]
            preferred_action = "abstain"
    elif transform_type in {"drop_audio", "drop_video"}:
        if preferred_is_abstain:
            legal_actions = ["mismatch", "abstain"] if mismatch_answer else ["abstain"]
            preferred_action = "abstain"
        elif preferred_answer and normalize_text(preferred_answer) != normalize_text(clean_reference):
            if preferred_is_mismatch:
                legal_actions = ["mismatch", "abstain"]
                preferred_action = "mismatch"
            else:
                legal_actions = ["retarget", "abstain"]
                preferred_action = "retarget"
        elif mismatch_answer:
            legal_actions = ["mismatch", "abstain"]
            preferred_action = "mismatch"
        else:
            legal_actions = ["abstain"]
            preferred_action = "abstain"
    else:
        legal_actions = ["keep"]
        preferred_action = "keep"

    legal_actions = list(dict.fromkeys(action for action in legal_actions if safe_text(action)))
    preferred_answer_kind = "none"
    if preferred_answer:
        if preferred_is_abstain:
            preferred_answer_kind = "abstain"
        elif preferred_is_mismatch:
            preferred_answer_kind = "mismatch"
        else:
            preferred_answer_kind = "retarget"
    retarget_positive = preferred_action == "retarget"
    mismatch_dominant = preferred_action == "mismatch" and "retarget" not in legal_actions
    return {
        "branch_id": safe_text(branch.get("branch_id")),
        "transform_type": transform_type,
        "legal_actions": legal_actions,
        "preferred_action": preferred_action,
        "preferred_answer_kind": preferred_answer_kind,
        "preferred_answer": preferred_answer,
        "mismatch_answer": mismatch_answer,
        "retarget_positive": retarget_positive,
        "mismatch_dominant": mismatch_dominant,
    }


def _build_target_legality_spec(episode: Dict[str, Any]) -> Dict[str, Any]:
    branch_rows = [_branch_legality_snapshot(episode, branch) for branch in _target_branches(episode)]
    branch_legality = {row["branch_id"]: row for row in branch_rows}
    legal_actions = sorted({action for row in branch_rows for action in row["legal_actions"]})
    retarget_positive = any(bool(row["retarget_positive"]) for row in branch_rows)
    mismatch_dominant = bool(branch_rows) and all(bool(row["mismatch_dominant"]) for row in branch_rows)
    has_abstain = any("abstain" in row["legal_actions"] for row in branch_rows)
    if retarget_positive and any("mismatch" in row["legal_actions"] for row in branch_rows):
        target_legality_type = "retarget_or_mismatch"
    elif retarget_positive and has_abstain:
        target_legality_type = "retarget_or_abstain"
    elif retarget_positive:
        target_legality_type = "retarget_only"
    elif mismatch_dominant and has_abstain:
        target_legality_type = "mismatch_or_abstain"
    elif mismatch_dominant:
        target_legality_type = "mismatch_only"
    elif has_abstain:
        target_legality_type = "change_or_abstain"
    else:
        target_legality_type = "keep"
    return {
        "target_legality_type": target_legality_type,
        "legal_actions": legal_actions,
        "preferred_actions": {
            row["branch_id"]: row["preferred_action"] for row in branch_rows
        },
        "branch_legality": branch_legality,
        "retarget_positive": retarget_positive,
        "mismatch_dominant": mismatch_dominant,
        "non_refusal_preferred_target_count": sum(
            1 for row in branch_rows if row["preferred_action"] in {"retarget", "mismatch"}
        ),
    }


def _build_evidence_spec(episode: Dict[str, Any]) -> Dict[str, Any]:
    requires_temporal_break = _requires_temporal_break(episode)
    has_control = _has_control_corruption(episode)
    return {
        "evidence_anchor_type": "temporal_order" if requires_temporal_break else "cross_modal_dependency",
        "requires_temporal_localization": requires_temporal_break,
        "counterfactual_compare_available": True,
        "supports_evidence_verifier": True,
        "temporal_evidence_ready": requires_temporal_break,
        "has_control_branch": has_control,
    }


def _build_observation_spec(episode: Dict[str, Any]) -> Dict[str, Any]:
    bank = ["global_av", "audio_focus", "visual_focus", "counterfactual_compare"]
    if _requires_temporal_break(episode):
        bank.append("temporal_window")
    if _has_control_corruption(episode):
        bank.append("control_compare")
    return {
        "supports_active_perception": True,
        "observation_bank": bank,
        "per_branch_budget": {
            "full": 4,
            "target": 4,
            "control": 3,
        },
    }


def _build_curriculum_spec(
    episode: Dict[str, Any],
    *,
    action_taken: str,
    benchmark_flags: Dict[str, Any],
    target_legality_spec: Dict[str, Any],
    evidence_spec: Dict[str, Any],
) -> Dict[str, Any]:
    has_control = _has_control_corruption(episode)
    retarget_positive = bool(target_legality_spec.get("retarget_positive"))
    mismatch_dominant = bool(target_legality_spec.get("mismatch_dominant"))
    temporal_evidence_ready = bool(evidence_spec.get("temporal_evidence_ready"))
    auto_repaired = action_taken in {"inject_mismatch_choice", "inject_retarget_choice", "inject_abstain_choice"}
    hardness_score = 1.0
    if temporal_evidence_ready:
        hardness_score += 0.45
    if _is_joint_episode(episode):
        hardness_score += 0.25
    if auto_repaired:
        hardness_score += 0.20
    if not has_control:
        hardness_score += 0.10
    hardness_score += min(0.50, float(int(benchmark_flags.get("audit_priority", 0))) / 200.0)

    sampling_weight_multiplier = 1.0
    if retarget_positive:
        sampling_weight_multiplier += 1.00
    if has_control:
        sampling_weight_multiplier += 0.65
    if temporal_evidence_ready:
        sampling_weight_multiplier += 0.35
    if auto_repaired:
        sampling_weight_multiplier += 0.10
    if mismatch_dominant:
        sampling_weight_multiplier -= 0.45
    sampling_weight_multiplier = max(0.25, sampling_weight_multiplier)

    return {
        "retarget_positive": retarget_positive,
        "mismatch_dominant": mismatch_dominant,
        "has_control": has_control,
        "temporal_evidence_ready": temporal_evidence_ready,
        "auto_repaired": auto_repaired,
        "hardness_score": float(hardness_score),
        "sampling_weight_multiplier": float(sampling_weight_multiplier),
    }


def is_cci_episode(episode: Dict[str, Any]) -> bool:
    return (
        safe_text(episode.get("phenomenon")) == PHENOMENON_OVERSHADOW
        and safe_text(episode.get("task_family")) in CCI_TASK_FAMILIES
    )


def _cci_decisive_modality(task_family: str) -> str:
    if task_family in {"audio_grounded_presence", "speaker_attribution"}:
        return CCI_DECISIVE_MODALITY_AUDIO
    if task_family == "av_matching":
        return CCI_DECISIVE_MODALITY_RELATION
    return ""


def _corrupted_modality(transform_type: str) -> str:
    if transform_type in {"swap_audio", "drop_audio", "shift_audio"}:
        return "audio"
    if transform_type in {"swap_video", "drop_video"}:
        return "visual"
    return ""


def _preserved_modality(decisive_modality: str, corrupted_modality: str) -> str:
    if decisive_modality == CCI_DECISIVE_MODALITY_RELATION:
        return "cross_modal_relation"
    if corrupted_modality == "audio":
        return "visual"
    if corrupted_modality == "visual":
        return "audio"
    return ""


def _transition_from_branch_legality(branch_legality: Dict[str, Any]) -> str:
    preferred_action = safe_text(branch_legality.get("preferred_action"))
    if preferred_action == "retarget":
        return CCI_EXPECTED_TRANSITION_RETARGET
    if preferred_action == "mismatch":
        return CCI_EXPECTED_TRANSITION_MISMATCH
    if preferred_action == "abstain":
        return CCI_EXPECTED_TRANSITION_ABSTAIN
    return CCI_EXPECTED_TRANSITION_PRESERVE


def _training_bucket_from_transition(expected_transition: str) -> str:
    if expected_transition == CCI_EXPECTED_TRANSITION_RETARGET:
        return CCI_TRAINING_BUCKET_RETARGET
    if expected_transition == CCI_EXPECTED_TRANSITION_MISMATCH:
        return CCI_TRAINING_BUCKET_MISMATCH
    return CCI_TRAINING_BUCKET_PRESERVE


def _swap_source_payload(branch: Dict[str, Any]) -> Dict[str, Any]:
    return ((((branch.get("input") or {}).get("intervention")) or {}).get("swap_source") or {})


def _donor_supported_answer(branch: Dict[str, Any]) -> str:
    preferred_answer = safe_text(branch.get("preferred_answer"))
    if preferred_answer:
        return preferred_answer
    swap_source = _swap_source_payload(branch)
    canonical_answer = safe_text(swap_source.get("canonical_answer"))
    if canonical_answer:
        return canonical_answer
    clean_answer = swap_source.get("clean_answer") or {}
    if clean_answer:
        return canonical_choice_answer(clean_answer)
    return ""


def _build_quartet_spec(episode: Dict[str, Any]) -> Dict[str, Any]:
    if not is_cci_episode(episode):
        return {
            "eligible_for_cci": False,
            "quartet_count": 0,
            "decisive_modality": "",
            "target_quartets": [],
        }

    task_family = safe_text(episode.get("task_family"))
    decisive_modality = _cci_decisive_modality(task_family)
    control_branches = _control_branches(episode)
    control_branch_id = safe_text(control_branches[0].get("branch_id")) if control_branches else ""
    branch_legality = ((episode.get("target_legality_spec") or {}).get("branch_legality") or {})
    target_quartets: List[Dict[str, Any]] = []
    for branch in _target_branches(episode):
        branch_id = safe_text(branch.get("branch_id"))
        transform_type = safe_text(branch.get("transform_type"))
        legality = branch_legality.get(branch_id) or _branch_legality_snapshot(episode, branch)
        expected_transition = _transition_from_branch_legality(legality)
        corrupted_modality = _corrupted_modality(transform_type)
        preserved_modality = _preserved_modality(decisive_modality, corrupted_modality)
        swap_source = _swap_source_payload(branch)
        donor_supported_answer = _donor_supported_answer(branch)
        donor_available = bool(swap_source.get("dataset")) and bool(swap_source.get("sample_id"))
        target_quartets.append(
            {
                "quartet_id": f"{safe_text(episode.get('episode_id'))}::{branch_id}",
                "target_branch_id": branch_id,
                "control_branch_id": control_branch_id,
                "intervention_operator": transform_type,
                "corrupted_modality": corrupted_modality,
                "preserved_modality": preserved_modality,
                "decisive_modality": decisive_modality,
                "expected_transition": expected_transition,
                "donor_available": donor_available,
                "donor_dataset": safe_text(swap_source.get("dataset")),
                "donor_sample_id": safe_text(swap_source.get("sample_id")),
                "donor_prompt": copy.deepcopy(swap_source.get("prompt", {}) or {}),
                "donor_media": copy.deepcopy(swap_source.get("media", {}) or {}),
                "donor_supported_answer": donor_supported_answer,
                "training_bucket": _training_bucket_from_transition(expected_transition),
            }
        )
    return {
        "eligible_for_cci": bool(target_quartets),
        "quartet_count": len(target_quartets),
        "full_branch_id": safe_text((episode.get("full_branch") or {}).get("branch_id")),
        "control_branch_ids": [safe_text(branch.get("branch_id")) for branch in control_branches],
        "decisive_modality": decisive_modality,
        "target_quartets": target_quartets,
    }


def _build_cci_spec(episode: Dict[str, Any], quartet_spec: Dict[str, Any]) -> Dict[str, Any]:
    quartets = list(quartet_spec.get("target_quartets") or [])
    transfer_only = safe_text(episode.get("dataset")) == "omnivideobench"
    expected_transition_histogram = Counter(
        safe_text(quartet.get("expected_transition")) for quartet in quartets if safe_text(quartet.get("expected_transition"))
    )
    training_bucket_histogram = Counter(
        safe_text(quartet.get("training_bucket")) for quartet in quartets if safe_text(quartet.get("training_bucket"))
    )
    donor_coverage_rate = (
        sum(1 for quartet in quartets if bool(quartet.get("donor_available"))) / float(len(quartets))
        if quartets
        else 0.0
    )
    return {
        "eligible_for_training": bool(quartet_spec.get("eligible_for_cci")) and not transfer_only,
        "eligible_for_core": bool((episode.get("benchmark_flags") or {}).get("eligible_for_core")) and bool(quartet_spec.get("eligible_for_cci")) and not transfer_only,
        "transfer_only": bool(transfer_only),
        "quartet_count": int(quartet_spec.get("quartet_count", 0) or 0),
        "decisive_modality": safe_text(quartet_spec.get("decisive_modality")),
        "expected_transition_histogram": dict(expected_transition_histogram),
        "training_bucket_histogram": dict(training_bucket_histogram),
        "donor_coverage_rate": float(donor_coverage_rate),
    }


def _build_benchmark_flags(
    episode: Dict[str, Any],
    *,
    snapshot: Dict[str, Any],
    action_taken: str,
) -> Dict[str, Any]:
    answer_type = safe_text(snapshot.get("answer_type"))
    has_target = bool(snapshot.get("has_target_corruption"))
    has_explicit_mismatch_choice = bool(snapshot.get("has_explicit_mismatch_choice"))
    target_preferred_answer_count = int(snapshot.get("target_preferred_answer_count") or 0)
    choice_target_under_specified = bool(snapshot.get("choice_target_under_specified"))
    auto_repaired = action_taken in {"inject_mismatch_choice", "inject_retarget_choice", "inject_abstain_choice"}
    eligible_for_core = (
        answer_type != "choice_label"
        or not has_target
        or has_explicit_mismatch_choice
        or target_preferred_answer_count > 0
    )
    audit_priority = 0
    if choice_target_under_specified:
        audit_priority += 100
    if snapshot.get("recommended_action") == "audit_required":
        audit_priority += 60
    if auto_repaired:
        audit_priority += 40
    if bool(snapshot.get("requires_temporal_break")):
        audit_priority += 30
    if _is_joint_episode(episode):
        audit_priority += 20
    if not bool(snapshot.get("requires_control_keep")):
        audit_priority += 10
    needs_human_audit = (
        choice_target_under_specified
        or auto_repaired
        or bool(snapshot.get("requires_temporal_break"))
        or _is_joint_episode(episode)
    )
    return {
        "eligible_for_core": bool(eligible_for_core),
        "needs_human_audit": bool(needs_human_audit),
        "audit_priority": int(audit_priority),
    }


def upgrade_episode_to_v2(episode: Dict[str, Any]) -> Dict[str, Any]:
    upgraded = copy.deepcopy(episode)
    pre_snapshot = inspect_episode_answer_space(upgraded)
    action_taken = safe_text(pre_snapshot.get("recommended_action")) or "keep"
    if action_taken == "inject_retarget_choice":
        injected = _inject_retarget_choice(upgraded)
        if injected is None:
            action_taken = "keep"
    if action_taken == "inject_abstain_choice":
        _inject_abstain_choice(upgraded)
    if action_taken == "inject_mismatch_choice":
        _inject_mismatch_choice(upgraded)
    post_snapshot = inspect_episode_answer_space(upgraded)
    upgraded["answer_space_spec"] = {
        "has_explicit_mismatch_choice": bool(post_snapshot["has_explicit_mismatch_choice"]),
        "has_explicit_abstain_choice": bool(post_snapshot["has_explicit_abstain_choice"]),
        "target_preferred_answer_count": int(post_snapshot["target_preferred_answer_count"]),
        "choice_target_under_specified": bool(post_snapshot["choice_target_under_specified"]),
        "issue_types": list(post_snapshot["issue_types"]),
        "recommended_action": action_taken,
        "auto_repaired": action_taken in {"inject_mismatch_choice", "inject_retarget_choice", "inject_abstain_choice"},
        "repair_action_taken": action_taken if action_taken in {"inject_mismatch_choice", "inject_retarget_choice", "inject_abstain_choice"} else "none",
    }
    upgraded["benchmark_flags"] = _build_benchmark_flags(
        upgraded,
        snapshot=post_snapshot,
        action_taken=action_taken,
    )
    upgraded["target_legality_spec"] = _build_target_legality_spec(upgraded)
    upgraded["evidence_spec"] = _build_evidence_spec(upgraded)
    upgraded["observation_spec"] = _build_observation_spec(upgraded)
    upgraded["curriculum_spec"] = _build_curriculum_spec(
        upgraded,
        action_taken=action_taken,
        benchmark_flags=upgraded["benchmark_flags"],
        target_legality_spec=upgraded["target_legality_spec"],
        evidence_spec=upgraded["evidence_spec"],
    )
    upgraded["quartet_spec"] = _build_quartet_spec(upgraded)
    upgraded["cci_spec"] = _build_cci_spec(upgraded, upgraded["quartet_spec"])
    return upgraded


def upgrade_episodes_to_v2(episodes: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [upgrade_episode_to_v2(episode) for episode in episodes]


def build_phase1_episodes(units: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    swap_selector = SwapSelector(units)
    episodes: List[Dict[str, Any]] = []
    for unit in units:
        episodes.extend(_build_unit_episodes_with_selector(unit=unit, swap_selector=swap_selector))
    return episodes


def build_phase2_episodes(units: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return upgrade_episodes_to_v2(build_phase1_episodes(units))


def summarize_episodes(episodes: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    return {
        "total_episodes": len(episodes),
        "by_split": dict(Counter(safe_text(row.get("split")) for row in episodes)),
        "by_dataset": dict(Counter(safe_text(row.get("dataset")) for row in episodes)),
        "by_task_family": dict(Counter(safe_text(row.get("task_family")) for row in episodes)),
        "by_phenomenon": dict(Counter(safe_text(row.get("phenomenon")) for row in episodes)),
        "by_answer_type": dict(Counter(safe_text(row.get("answer_type")) for row in episodes)),
        "by_episode_type": dict(Counter(safe_text(row.get("episode_type")) for row in episodes)),
    }


def _summarize_v2_group(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    total = len(rows)
    if total <= 0:
        return {
            "n_episodes": 0,
            "core_eligible_rate": 0.0,
            "needs_human_audit_rate": 0.0,
            "auto_repaired_rate": 0.0,
            "choice_target_under_specified_rate": 0.0,
            "retarget_positive_rate": 0.0,
            "mismatch_dominant_rate": 0.0,
            "control_coverage_rate": 0.0,
            "temporal_evidence_ready_rate": 0.0,
            "by_recommended_action": {},
            "by_issue_type": {},
            "by_target_legality_type": {},
        }
    return {
        "n_episodes": total,
        "core_eligible_rate": sum(
            1 for row in rows if bool((row.get("benchmark_flags") or {}).get("eligible_for_core"))
        ) / float(total),
        "needs_human_audit_rate": sum(
            1 for row in rows if bool((row.get("benchmark_flags") or {}).get("needs_human_audit"))
        ) / float(total),
        "auto_repaired_rate": sum(
            1 for row in rows if bool((row.get("answer_space_spec") or {}).get("auto_repaired"))
        ) / float(total),
        "choice_target_under_specified_rate": sum(
            1 for row in rows if bool((row.get("answer_space_spec") or {}).get("choice_target_under_specified"))
        ) / float(total),
        "retarget_positive_rate": sum(
            1 for row in rows if bool((row.get("curriculum_spec") or {}).get("retarget_positive"))
        ) / float(total),
        "mismatch_dominant_rate": sum(
            1 for row in rows if bool((row.get("curriculum_spec") or {}).get("mismatch_dominant"))
        ) / float(total),
        "control_coverage_rate": sum(
            1 for row in rows if bool((row.get("curriculum_spec") or {}).get("has_control"))
        ) / float(total),
        "temporal_evidence_ready_rate": sum(
            1 for row in rows if bool((row.get("curriculum_spec") or {}).get("temporal_evidence_ready"))
        ) / float(total),
        "by_recommended_action": dict(
            Counter(safe_text((row.get("answer_space_spec") or {}).get("recommended_action")) for row in rows)
        ),
        "by_issue_type": dict(
            Counter(
                issue
                for row in rows
                for issue in ((row.get("answer_space_spec") or {}).get("issue_types") or [])
                if safe_text(issue)
            )
        ),
        "by_target_legality_type": dict(
            Counter(safe_text((row.get("target_legality_spec") or {}).get("target_legality_type")) for row in rows)
        ),
    }


def summarize_episodes_v2(episodes: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    by_split: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    by_task_family: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    by_episode_type: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in episodes:
        by_split[safe_text(row.get("split"))].append(row)
        by_task_family[safe_text(row.get("task_family"))].append(row)
        by_episode_type[safe_text(row.get("episode_type"))].append(row)
    return {
        **summarize_episodes(episodes),
        "validity": _summarize_v2_group(list(episodes)),
        "validity_by_split": {key: _summarize_v2_group(rows) for key, rows in sorted(by_split.items())},
        "validity_by_task_family": {key: _summarize_v2_group(rows) for key, rows in sorted(by_task_family.items())},
        "validity_by_episode_type": {key: _summarize_v2_group(rows) for key, rows in sorted(by_episode_type.items())},
    }


_CCI_REQUIRED_V2_FIELDS = (
    "answer_space_spec",
    "benchmark_flags",
    "target_legality_spec",
    "evidence_spec",
    "observation_spec",
    "curriculum_spec",
    "quartet_spec",
    "cci_spec",
)


def _ensure_cci_ready_episode(episode: Dict[str, Any]) -> Dict[str, Any]:
    if all(field in episode for field in _CCI_REQUIRED_V2_FIELDS):
        return copy.deepcopy(episode)
    return upgrade_episode_to_v2(episode)


def build_cci_episodes(episodes: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    cci_rows: List[Dict[str, Any]] = []
    for row in episodes:
        prepared = _ensure_cci_ready_episode(row)
        if bool((prepared.get("quartet_spec") or {}).get("eligible_for_cci")) and bool(
            (prepared.get("cci_spec") or {}).get("eligible_for_training")
        ):
            cci_rows.append(prepared)
    return cci_rows


def _summarize_cci_group(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    rows = list(rows)
    quartets = [
        quartet
        for row in rows
        for quartet in ((row.get("quartet_spec") or {}).get("target_quartets") or [])
    ]
    if not rows:
        return {
            "n_episodes": 0,
            "n_quartets": 0,
            "by_task_family": {},
            "by_episode_type": {},
            "by_decisive_modality": {},
            "by_expected_transition": {},
            "by_training_bucket": {},
            "donor_coverage_rate": 0.0,
            "control_coverage_rate": 0.0,
        }
    donor_coverage_rate = (
        sum(1 for quartet in quartets if bool(quartet.get("donor_available"))) / float(len(quartets))
        if quartets
        else 0.0
    )
    control_coverage_rate = sum(
        1 for row in rows if bool((row.get("curriculum_spec") or {}).get("has_control"))
    ) / float(len(rows))
    return {
        "n_episodes": len(rows),
        "n_quartets": len(quartets),
        "by_task_family": dict(Counter(safe_text(row.get("task_family")) for row in rows)),
        "by_episode_type": dict(Counter(safe_text(row.get("episode_type")) for row in rows)),
        "by_decisive_modality": dict(
            Counter(safe_text((row.get("cci_spec") or {}).get("decisive_modality")) for row in rows)
        ),
        "by_expected_transition": dict(
            Counter(safe_text(quartet.get("expected_transition")) for quartet in quartets)
        ),
        "by_training_bucket": dict(
            Counter(safe_text(quartet.get("training_bucket")) for quartet in quartets)
        ),
        "donor_coverage_rate": float(donor_coverage_rate),
        "control_coverage_rate": float(control_coverage_rate),
    }


def summarize_episodes_cci(episodes: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    by_split: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    by_task_family: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in episodes:
        by_split[safe_text(row.get("split"))].append(row)
        by_task_family[safe_text(row.get("task_family"))].append(row)
    return {
        "summary": _summarize_cci_group(episodes),
        "by_split": {key: _summarize_cci_group(rows) for key, rows in sorted(by_split.items())},
        "by_task_family": {key: _summarize_cci_group(rows) for key, rows in sorted(by_task_family.items())},
    }


def _audit_reason(episode: Dict[str, Any]) -> str:
    reasons: List[str] = []
    answer_space_spec = episode.get("answer_space_spec") or {}
    if bool(answer_space_spec.get("choice_target_under_specified")):
        reasons.append("choice_target_under_specified")
    if bool(answer_space_spec.get("auto_repaired")):
        reasons.append("auto_repaired")
    if bool((episode.get("benchmark_flags") or {}).get("needs_human_audit")) and not reasons:
        reasons.append("needs_human_audit")
    if _requires_temporal_break(episode):
        reasons.append("temporal")
    if _is_joint_episode(episode):
        reasons.append("joint")
    if not reasons:
        reasons.append("representative")
    return ",".join(dict.fromkeys(reasons))


def build_audit_candidates(
    episodes: Sequence[Dict[str, Any]],
    *,
    split: str = "val",
    per_episode_type: int = 24,
) -> List[Dict[str, Any]]:
    buckets: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in episodes:
        if safe_text(row.get("split")) != split:
            continue
        candidate = copy.deepcopy(row)
        candidate["audit_reason"] = _audit_reason(candidate)
        buckets[safe_text(candidate.get("episode_type"))].append(candidate)

    selected: List[Dict[str, Any]] = []
    for episode_type, rows in sorted(buckets.items()):
        ranked = sorted(
            rows,
            key=lambda row: (
                -int((row.get("benchmark_flags") or {}).get("audit_priority", 0)),
                safe_text(row.get("episode_id")),
            ),
        )
        selected.extend(ranked[:per_episode_type])
    return selected


def write_episode_artifacts(
    *,
    output_dir: Path,
    input_units_path: Path,
    episodes: Sequence[Dict[str, Any]],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(output_dir / "episodes_v1.jsonl", episodes)
    write_json(
        output_dir / "summary.json",
        {
            "input_rl_units": str(input_units_path),
            "summary": summarize_episodes(episodes),
        },
    )
    write_json(output_dir / "example_episodes.json", list(episodes)[:12])


def write_episode_v2_artifacts(
    *,
    output_dir: Path,
    input_units_path: Path,
    full_episodes: Sequence[Dict[str, Any]],
    core_episodes: Sequence[Dict[str, Any]],
    audit_candidates: Sequence[Dict[str, Any]],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(output_dir / "episodes_v2.full.jsonl", full_episodes)
    write_jsonl(output_dir / "episodes_v2.core.jsonl", core_episodes)
    write_jsonl(output_dir / "episodes_v2.audit_candidates.jsonl", audit_candidates)
    write_json(
        output_dir / "summary_v2.json",
        {
            "input_rl_units": str(input_units_path),
            "full_summary": summarize_episodes_v2(full_episodes),
            "core_summary": summarize_episodes_v2(core_episodes),
            "audit_candidates": {
                "total": len(audit_candidates),
                "by_episode_type": dict(Counter(safe_text(row.get("episode_type")) for row in audit_candidates)),
            },
        },
    )
    example_payload = {
        "auto_repaired_examples": [row for row in full_episodes if bool((row.get("answer_space_spec") or {}).get("auto_repaired"))][:4],
        "retarget_positive_examples": [
            row
            for row in full_episodes
            if bool((row.get("curriculum_spec") or {}).get("retarget_positive"))
        ][:4],
        "core_eligible_examples": [row for row in core_episodes][:4],
        "audit_required_examples": [
            row
            for row in full_episodes
            if safe_text((row.get("answer_space_spec") or {}).get("recommended_action")) == "audit_required"
        ][:4],
        "under_specified_examples": [
            row
            for row in full_episodes
            if bool((row.get("answer_space_spec") or {}).get("choice_target_under_specified"))
        ][:4],
    }
    write_json(output_dir / "example_episodes_v2.json", example_payload)


def write_episode_cci_artifacts(
    *,
    output_dir: Path,
    input_units_path: Path,
    full_episodes: Sequence[Dict[str, Any]],
    core_episodes: Sequence[Dict[str, Any]],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(output_dir / "episodes_cci.full.jsonl", full_episodes)
    write_jsonl(output_dir / "episodes_cci.core.jsonl", core_episodes)
    write_json(
        output_dir / "summary_cci.json",
        {
            "input_rl_units": str(input_units_path),
            "full_summary": summarize_episodes_cci(full_episodes),
            "core_summary": summarize_episodes_cci(core_episodes),
        },
    )
    write_json(
        output_dir / "example_episodes_cci.json",
        {
            "retarget_examples": [
                row
                for row in full_episodes
                if CCI_TRAINING_BUCKET_RETARGET in ((row.get("cci_spec") or {}).get("training_bucket_histogram") or {})
            ][:4],
            "mismatch_examples": [
                row
                for row in full_episodes
                if CCI_TRAINING_BUCKET_MISMATCH in ((row.get("cci_spec") or {}).get("training_bucket_histogram") or {})
            ][:4],
            "preserve_examples": [
                row
                for row in full_episodes
                if CCI_TRAINING_BUCKET_PRESERVE in ((row.get("cci_spec") or {}).get("training_bucket_histogram") or {})
            ][:4],
        },
    )
