from __future__ import annotations

import math
import re
from collections import Counter
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from omni_dpo.preference_pairs import (
    ABSTAIN_TEMPLATE,
    find_explicit_mismatch_choice,
    is_abstention_text,
    normalize_model_answer,
    normalize_text,
    safe_text,
)


_UNCERTAINTY_RE = re.compile(
    r"uncertain|unclear|not sure|unsure|cannot tell|can't tell|unable to tell|"
    r"not enough information|insufficient information|cannot determine|can't determine",
    re.IGNORECASE,
)
_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _tokenize(text: str) -> List[str]:
    return _TOKEN_RE.findall(normalize_text(text))


def _token_f1(pred: str, ref: str) -> float:
    pred_toks = _tokenize(pred)
    ref_toks = _tokenize(ref)
    if not pred_toks and not ref_toks:
        return 1.0
    if not pred_toks or not ref_toks:
        return 0.0
    pred_counter = Counter(pred_toks)
    ref_counter = Counter(ref_toks)
    overlap = sum(min(pred_counter[tok], ref_counter[tok]) for tok in pred_counter)
    if overlap <= 0:
        return 0.0
    precision = overlap / float(len(pred_toks))
    recall = overlap / float(len(ref_toks))
    if precision + recall <= 0:
        return 0.0
    return 2.0 * precision * recall / (precision + recall)


def is_uncertainty_text(text: str) -> bool:
    candidate = safe_text(text)
    if not candidate:
        return False
    return bool(_UNCERTAINTY_RE.search(candidate))


def parse_episode_output(branch: Dict[str, Any], raw_output: str, clean_reference: str) -> Dict[str, Any]:
    prompt = ((branch.get("input") or {}).get("prompt") or {})
    normalized = normalize_model_answer(raw_output, prompt)
    abstained = is_abstention_text(normalized or raw_output)
    uncertain = is_uncertainty_text(normalized or raw_output)
    parseable = normalized is not None or abstained
    normalized_output = normalized if normalized is not None else (ABSTAIN_TEMPLATE if abstained else None)
    word_count = len(safe_text(raw_output).split())
    copied_clean = (
        normalized_output is not None
        and normalize_text(normalized_output) == normalize_text(clean_reference)
    )
    return {
        "raw_output": raw_output,
        "normalized_output": normalized_output,
        "parseable": parseable,
        "abstained": abstained,
        "uncertain": uncertain,
        "word_count": word_count,
        "copied_clean": copied_clean,
    }


def _full_branch_clean_score(answer_type: str, parsed: Dict[str, Any], clean_reference: str) -> float:
    normalized = safe_text(parsed.get("normalized_output"))
    if not normalized:
        return 0.0
    if answer_type == "short_text":
        em = 1.0 if normalize_text(normalized) == normalize_text(clean_reference) else 0.0
        f1 = _token_f1(normalized, clean_reference)
        return max(0.0, min(1.0, 0.5 * em + 0.5 * f1))
    return 1.0 if normalize_text(normalized) == normalize_text(clean_reference) else 0.0


def _mismatch_answer_for_branch(branch: Dict[str, Any]) -> str:
    prompt = ((branch.get("input") or {}).get("prompt") or {})
    mismatch_choice = find_explicit_mismatch_choice({"prompt": prompt})
    if mismatch_choice is None:
        return ""
    label = safe_text(mismatch_choice.get("label")).upper()
    text = safe_text(mismatch_choice.get("text"))
    return f"Final answer: {label}. {text}".strip()


def _branch_legality(branch: Dict[str, Any], episode: Dict[str, Any]) -> Dict[str, Any]:
    target_legality_spec = episode.get("target_legality_spec") or {}
    branch_legality = (target_legality_spec.get("branch_legality") or {}).get(safe_text(branch.get("branch_id"))) or {}
    if branch_legality:
        return branch_legality
    mismatch_answer = _mismatch_answer_for_branch(branch)
    preferred_answer = safe_text(branch.get("preferred_answer"))
    transform_type = safe_text(branch.get("transform_type"))
    if transform_type in {"swap_audio", "swap_video"}:
        if preferred_answer and preferred_answer and normalize_text(preferred_answer) != normalize_text(mismatch_answer):
            legal_actions = ["retarget"] + (["mismatch"] if mismatch_answer else [])
            preferred_action = "retarget"
        else:
            legal_actions = ["mismatch"] if mismatch_answer else ["change"]
            preferred_action = "mismatch" if mismatch_answer else "change"
    elif transform_type == "shift_audio":
        legal_actions = ["retarget", "abstain"] if preferred_answer else (["mismatch", "abstain"] if mismatch_answer else ["abstain"])
        preferred_action = "retarget" if preferred_answer else ("mismatch" if mismatch_answer else "abstain")
    else:
        legal_actions = ["retarget", "abstain"] if preferred_answer else (["mismatch", "abstain"] if mismatch_answer else ["abstain"])
        preferred_action = "retarget" if preferred_answer and normalize_text(preferred_answer) != normalize_text(mismatch_answer) else ("mismatch" if mismatch_answer else "abstain")
    return {
        "branch_id": safe_text(branch.get("branch_id")),
        "legal_actions": legal_actions,
        "preferred_action": preferred_action,
        "preferred_answer": preferred_answer,
        "mismatch_answer": mismatch_answer,
    }


def _target_action(branch: Dict[str, Any], parsed: Dict[str, Any], clean_reference: str, episode: Dict[str, Any]) -> Dict[str, Any]:
    legality = _branch_legality(branch, episode)
    preferred_answer = safe_text(legality.get("preferred_answer"))
    mismatch_answer = safe_text(legality.get("mismatch_answer"))
    normalized = safe_text(parsed.get("normalized_output"))
    parseable = bool(parsed.get("parseable"))
    abstained = bool(parsed.get("abstained")) or bool(parsed.get("uncertain"))
    copied_clean = bool(parsed.get("copied_clean"))
    action = "unparseable"
    if copied_clean:
        action = "copy_clean"
    elif abstained:
        action = "abstain"
    elif preferred_answer and normalized and normalize_text(normalized) == normalize_text(preferred_answer):
        if mismatch_answer and normalize_text(preferred_answer) == normalize_text(mismatch_answer):
            action = "mismatch"
        else:
            action = "retarget"
    elif mismatch_answer and normalized and normalize_text(normalized) == normalize_text(mismatch_answer):
        action = "mismatch"
    elif parseable and normalized:
        action = "arbitrary_change"
    legal_actions = set(legality.get("legal_actions") or [])
    legal_change = action in legal_actions and action not in {"copy_clean", "unparseable"}
    preferred_action = safe_text(legality.get("preferred_action"))
    preferred_followed = action == preferred_action if preferred_action else legal_change
    evidence_supported = preferred_followed or (legal_change and action in {"retarget", "mismatch"})
    return {
        "action": action,
        "legal_change": legal_change,
        "preferred_followed": preferred_followed,
        "evidence_supported": evidence_supported,
        "preferred_action": preferred_action,
        "legal_actions": sorted(legal_actions),
        "copied_clean": copied_clean,
        "abstained": abstained,
        "parseable": parseable,
    }


def _target_utility(
    *,
    branch: Dict[str, Any],
    target_eval: Dict[str, Any],
    episode: Dict[str, Any],
) -> float:
    requires_temporal_break = bool((episode.get("evidence_spec") or {}).get("requires_temporal_localization"))
    action = safe_text(target_eval.get("action"))
    preferred_action = safe_text(target_eval.get("preferred_action"))
    legal_change = bool(target_eval.get("legal_change"))
    if action == "copy_clean":
        return -1.0
    if action == "unparseable":
        return -0.6
    if action == preferred_action:
        if action == "retarget":
            return 1.0
        if action == "mismatch":
            return 0.9
        if action == "abstain":
            return 0.35 if requires_temporal_break else 0.55
    if legal_change:
        if action == "retarget":
            return 0.85
        if action == "mismatch":
            return 0.7
        if action == "abstain":
            return 0.15 if requires_temporal_break else 0.35
        return 0.1
    if action == "abstain":
        return -0.35
    return -0.2


def _relation_satisfaction(
    branch: Dict[str, Any],
    parsed: Dict[str, Any],
    clean_reference: str,
    episode: Dict[str, Any],
) -> float:
    role = safe_text(branch.get("branch_role"))
    if role == "control_corruption":
        if bool(parsed.get("copied_clean")):
            return 1.0
        if bool(parsed.get("abstained")) or bool(parsed.get("uncertain")):
            return -1.0
        return -0.75 if bool(parsed.get("parseable")) else -1.0
    if role == "target_corruption":
        target_eval = _target_action(branch, parsed, clean_reference, episode)
        if bool(target_eval.get("legal_change")):
            return 1.0 if bool(target_eval.get("preferred_followed")) else 0.6
        if safe_text(target_eval.get("action")) == "copy_clean":
            return -1.0
        if safe_text(target_eval.get("action")) == "abstain":
            return -0.25
        return -0.4
    return 0.0


def _format_score(answer_type: str, parsed_rows: Sequence[Dict[str, Any]]) -> float:
    scores: List[float] = []
    max_words = 18 if answer_type == "choice_label" else 24
    for parsed in parsed_rows:
        parseable = bool(parsed.get("parseable"))
        words = int(parsed.get("word_count") or 0)
        if not parseable:
            scores.append(-1.0)
            continue
        scores.append(1.0 if words <= max_words else 0.2)
    return float(sum(scores) / max(1, len(scores)))


def _calibration_score(full_parsed: Dict[str, Any], cf_rows: Sequence[Tuple[Dict[str, Any], Dict[str, Any]]]) -> float:
    scores: List[float] = []
    if bool(full_parsed.get("abstained")) or bool(full_parsed.get("uncertain")):
        scores.append(-1.0)
    else:
        scores.append(0.75)
    for branch, parsed in cf_rows:
        role = safe_text(branch.get("branch_role"))
        abstain_or_uncertain = bool(parsed.get("abstained")) or bool(parsed.get("uncertain"))
        if role == "control_corruption":
            if abstain_or_uncertain:
                scores.append(-0.5)
            elif bool(parsed.get("copied_clean")):
                scores.append(0.75)
            else:
                scores.append(-0.25)
        else:
            if abstain_or_uncertain:
                scores.append(0.15)
            elif bool(parsed.get("copied_clean")):
                scores.append(-0.5)
            else:
                scores.append(0.55)
    return float(sum(scores) / max(1, len(scores)))


def score_episode_bundle(
    episode: Dict[str, Any],
    branch_outputs: Dict[str, str],
) -> Dict[str, Any]:
    clean_reference = safe_text(episode.get("clean_reference"))
    answer_type = safe_text(episode.get("answer_type"))
    full_branch = episode["full_branch"]
    cf_branches = [episode["cf_branch_a"], episode["cf_branch_b"]]

    parsed_full = parse_episode_output(full_branch, branch_outputs.get("full", ""), clean_reference)
    parsed_cf = [
        (branch, parse_episode_output(branch, branch_outputs.get(safe_text(branch.get("branch_id")), ""), clean_reference))
        for branch in cf_branches
    ]

    r_clean = _full_branch_clean_score(answer_type, parsed_full, clean_reference)
    target_rows = [(branch, parsed) for branch, parsed in parsed_cf if safe_text(branch.get("branch_role")) == "target_corruption"]
    control_rows = [(branch, parsed) for branch, parsed in parsed_cf if safe_text(branch.get("branch_role")) == "control_corruption"]
    target_evals = [
        (branch, parsed, _target_action(branch, parsed, clean_reference, episode))
        for branch, parsed in target_rows
    ]
    target_scores = [
        _target_utility(branch=branch, target_eval=target_eval, episode=episode)
        for branch, _parsed, target_eval in target_evals
    ]
    r_cf = float(sum(target_scores) / max(1, len(target_scores)))
    dep_scores = [
        _relation_satisfaction(branch, parsed, clean_reference, episode)
        for branch, parsed in parsed_cf
    ]
    r_dep = float(sum(dep_scores) / max(1, len(dep_scores)))
    r_fmt = _format_score(answer_type, [parsed_full] + [parsed for _branch, parsed in parsed_cf])
    r_cal = _calibration_score(parsed_full, parsed_cf)

    weights = episode.get("reward_weights") or {}
    has_control = bool(control_rows)
    full_correct = r_clean >= (0.8 if answer_type == "short_text" else 1.0)
    control_keep_rate_raw = sum(1 for _branch, row in control_rows if row.get("copied_clean")) / float(max(1, len(control_rows)))
    control_gate = 1.0 if (not has_control or control_keep_rate_raw >= 0.999) else 0.0
    clean_gate = 1.0 if full_correct else 0.0
    gated_target_utility = float(r_cf) * clean_gate * control_gate
    reward_total = (
        float(weights.get("R_cf", 0.35)) * gated_target_utility
        + float(weights.get("R_fmt", 0.05)) * r_fmt
        + float(weights.get("R_cal", 0.05)) * r_cal
        - float(weights.get("R_clean", 0.35)) * (1.0 - float(r_clean))
        - float(weights.get("R_dep", 0.20)) * (1.0 - max(0.0, control_gate if has_control else 1.0))
        - 0.15 * (1.0 if bool(parsed_full.get("abstained")) or bool(parsed_full.get("uncertain")) else 0.0)
    )

    branch_metrics: Dict[str, Any] = {
        "full": parsed_full,
    }
    for branch, parsed in parsed_cf:
        branch_metrics[safe_text(branch.get("branch_id"))] = parsed

    target_copy_rate = sum(1 for _branch, row, target_eval in target_evals if target_eval.get("copied_clean")) / float(max(1, len(target_evals)))
    target_abstention_rate = sum(
        1 for _branch, row, target_eval in target_evals if target_eval.get("abstained")
    ) / float(max(1, len(target_rows)))
    target_legal_change_rate = sum(
        1 for _branch, _row, target_eval in target_evals if bool(target_eval.get("legal_change"))
    ) / float(max(1, len(target_rows)))
    target_retarget_rate = sum(
        1 for _branch, _row, target_eval in target_evals if safe_text(target_eval.get("action")) == "retarget"
    ) / float(max(1, len(target_rows)))
    target_mismatch_rate = sum(
        1 for _branch, _row, target_eval in target_evals if safe_text(target_eval.get("action")) == "mismatch"
    ) / float(max(1, len(target_rows)))
    evidence_support_rate = sum(
        1 for _branch, _row, target_eval in target_evals if bool(target_eval.get("evidence_supported"))
    ) / float(max(1, len(target_rows)))
    target_preferred_follow_rate = sum(
        1 for _branch, _row, target_eval in target_evals if bool(target_eval.get("preferred_followed"))
    ) / float(max(1, len(target_rows)))
    control_keep_rate = control_keep_rate_raw if has_control else 1.0
    control_keep_rate_controlled = control_keep_rate_raw if has_control else 1.0
    clean_overrefusal = bool(parsed_full.get("abstained")) or bool(parsed_full.get("uncertain"))
    verifier_scores = {
        "V_clean": float(r_clean),
        "V_ctrl": float(control_keep_rate),
        "V_tgt": float(target_legal_change_rate),
        "V_evidence": float(evidence_support_rate),
        "V_refuse": 0.0 if clean_overrefusal else 1.0,
    }

    return {
        "reward_total": float(reward_total),
        "reward_components": {
            "R_clean": float(r_clean),
            "R_cf": float(target_legal_change_rate),
            "R_dep": float(control_keep_rate),
            "R_fmt": float(r_fmt),
            "R_cal": float(evidence_support_rate),
        },
        "branch_metrics": branch_metrics,
        "target_branch_metrics": {
            safe_text(branch.get("branch_id")): target_eval
            for branch, _parsed, target_eval in target_evals
        },
        "verifier_scores": verifier_scores,
        "constraint_metrics": {
            "clean_error_rate": float(1.0 - float(r_clean)),
            "control_error_rate": float(0.0 if not has_control else (1.0 - control_keep_rate_controlled)),
            "clean_overrefusal_rate": float(1.0 if clean_overrefusal else 0.0),
            "perception_budget_violation_rate": 0.0,
        },
        "ceg_metrics": {
            "gated_target_utility": float(gated_target_utility),
            "target_utility": float(r_cf),
            "clean_gate": float(clean_gate),
            "control_gate": float(control_gate),
        },
        "episode_metrics": {
            "full_correct": bool(full_correct),
            "target_copy_rate": float(target_copy_rate),
            "target_legal_change_rate": float(target_legal_change_rate),
            "target_retarget_rate": float(target_retarget_rate),
            "target_mismatch_rate": float(target_mismatch_rate),
            "target_abstention_rate": float(target_abstention_rate),
            "control_keep_rate": float(control_keep_rate),
            "control_keep_rate_controlled": float(control_keep_rate_controlled),
            "with_control": bool(has_control),
            "clean_overrefusal": bool(clean_overrefusal),
            "evidence_support_rate": float(evidence_support_rate),
            "target_preferred_follow_rate": float(target_preferred_follow_rate),
        },
    }


def summarize_episode_records(records: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    if not records:
        return {
            "n_records": 0,
            "reward_total": 0.0,
            "reward_mean": 0.0,
            "reward_std": 0.0,
            "clean_accuracy": 0.0,
            "target_copy_rate": 0.0,
            "target_legal_change_rate": 0.0,
            "target_retarget_rate": 0.0,
            "target_mismatch_rate": 0.0,
            "target_abstention_rate": 0.0,
            "control_keep_rate": 0.0,
            "control_keep_rate_controlled": 0.0,
            "with_control_rate": 0.0,
            "clean_overrefusal_rate": 0.0,
            "evidence_support_rate": 0.0,
            "target_preferred_follow_rate": 0.0,
        }

    clean_accuracy = sum(1 for row in records if bool((row.get("episode_metrics") or {}).get("full_correct"))) / float(len(records))
    clean_overrefusal_rate = sum(
        1 for row in records if bool((row.get("episode_metrics") or {}).get("clean_overrefusal"))
    ) / float(len(records))
    target_copy_rate = sum(float((row.get("episode_metrics") or {}).get("target_copy_rate", 0.0)) for row in records) / float(len(records))
    target_legal_change_rate = sum(
        float((row.get("episode_metrics") or {}).get("target_legal_change_rate", 0.0)) for row in records
    ) / float(len(records))
    target_retarget_rate = sum(
        float((row.get("episode_metrics") or {}).get("target_retarget_rate", 0.0)) for row in records
    ) / float(len(records))
    target_mismatch_rate = sum(
        float((row.get("episode_metrics") or {}).get("target_mismatch_rate", 0.0)) for row in records
    ) / float(len(records))
    target_abstention_rate = sum(float((row.get("episode_metrics") or {}).get("target_abstention_rate", 0.0)) for row in records) / float(len(records))
    control_keep_rate = sum(float((row.get("episode_metrics") or {}).get("control_keep_rate", 1.0)) for row in records) / float(len(records))
    controlled_records = [row for row in records if bool((row.get("episode_metrics") or {}).get("with_control"))]
    control_keep_rate_controlled = (
        sum(float((row.get("episode_metrics") or {}).get("control_keep_rate_controlled", 1.0)) for row in controlled_records)
        / float(len(controlled_records))
        if controlled_records
        else 1.0
    )
    with_control_rate = sum(
        1 for row in records if bool((row.get("episode_metrics") or {}).get("with_control"))
    ) / float(len(records))
    evidence_support_rate = sum(
        float((row.get("episode_metrics") or {}).get("evidence_support_rate", 0.0)) for row in records
    ) / float(len(records))
    target_preferred_follow_rate = sum(
        float((row.get("episode_metrics") or {}).get("target_preferred_follow_rate", 0.0)) for row in records
    ) / float(len(records))
    reward_mean = sum(float(row.get("reward_total", 0.0)) for row in records) / float(len(records))
    reward_std = math.sqrt(
        sum((float(row.get("reward_total", 0.0)) - reward_mean) ** 2 for row in records) / float(len(records))
    )
    return {
        "n_records": len(records),
        "reward_total": float(sum(float(row.get("reward_total", 0.0)) for row in records)),
        "reward_mean": float(reward_mean),
        "reward_std": float(reward_std),
        "clean_accuracy": float(clean_accuracy),
        "target_copy_rate": float(target_copy_rate),
        "target_legal_change_rate": float(target_legal_change_rate),
        "target_retarget_rate": float(target_retarget_rate),
        "target_mismatch_rate": float(target_mismatch_rate),
        "target_abstention_rate": float(target_abstention_rate),
        "control_keep_rate": float(control_keep_rate),
        "control_keep_rate_controlled": float(control_keep_rate_controlled),
        "with_control_rate": float(with_control_rate),
        "clean_overrefusal_rate": float(clean_overrefusal_rate),
        "evidence_support_rate": float(evidence_support_rate),
        "target_preferred_follow_rate": float(target_preferred_follow_rate),
    }
