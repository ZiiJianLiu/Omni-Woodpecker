from __future__ import annotations

import copy
import hashlib
import json
import os
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


ABSTAIN_TEMPLATE = "Final answer: Insufficient information from the available modalities."

PAIR_WEIGHT_BY_FAMILY = {
    "clean_pref": 1.0,
    "target_abstain": 1.0,
    "target_retarget": 1.2,
    "control_keep": 0.8,
    "hallucination_reject": 1.3,
}

ALLOWED_PAIR_FAMILIES = set(PAIR_WEIGHT_BY_FAMILY)
ALLOWED_CHOSEN_SOURCES = {"gold", "abstain_template", "counterfactual_label"}
ALLOWED_REJECTED_SOURCES = {
    "choice_negative",
    "pooled_negative",
    "clean_answer",
    "baseline_hallucination",
}

_FINAL_ANSWER_RE = re.compile(r"final answer\s*:\s*", re.IGNORECASE)
_CHOICE_LABEL_RE = re.compile(r"\b([A-Z])\b")
_ABSTAIN_RE = re.compile(
    r"insufficient information|not enough information|cannot determine|can't determine|"
    r"unable to determine|cannot answer|can't answer|do not know|don't know|not sure|unsure",
    re.IGNORECASE,
)
_MISMATCH_PATTERNS = (
    re.compile(r"\bunrelated\b", re.IGNORECASE),
    re.compile(r"\bmismatch(?:ed)?\b", re.IGNORECASE),
    re.compile(r"\bunmatched\b", re.IGNORECASE),
    re.compile(r"\bnot\s+matched\b", re.IGNORECASE),
    re.compile(r"\bnot\s+(?:a\s+)?match(?:ed)?\b", re.IGNORECASE),
    re.compile(r"\bno\s+match\b", re.IGNORECASE),
    re.compile(r"^none(?:\.)?$", re.IGNORECASE),
    re.compile(r"^none of the above(?:\.)?$", re.IGNORECASE),
)


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, default=_json_default)


def _json_default(obj: Any) -> Any:
    if isinstance(obj, os.PathLike):
        return os.fspath(obj)
    raise TypeError(f"Object of type {obj.__class__.__name__} is not JSON serializable")


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, default=_json_default) + "\n")


def append_jsonl(path: Path, row: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False, default=_json_default) + "\n")


def safe_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def normalize_text(text: str) -> str:
    text = safe_text(text).lower()
    text = _FINAL_ANSWER_RE.sub("", text)
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return " ".join(text.split())


def is_abstention_text(text: str) -> bool:
    candidate = safe_text(text)
    if not candidate:
        return True
    return bool(_ABSTAIN_RE.search(candidate))


def format_choice_answer(answer: Dict[str, Any]) -> str:
    label = safe_text(answer.get("answer_label")).upper()
    text = safe_text(answer.get("answer_text"))
    if label and text:
        return f"Final answer: {label}. {text}"
    if label:
        return f"Final answer: {label}"
    if text:
        return f"Final answer: {text}"
    return ABSTAIN_TEMPLATE


def format_short_answer(answer: Dict[str, Any]) -> str:
    text = safe_text(answer.get("answer_text"))
    if text:
        return f"Final answer: {text}"
    return ABSTAIN_TEMPLATE


def format_abstain() -> str:
    return ABSTAIN_TEMPLATE


def answer_format(unit: Dict[str, Any]) -> str:
    return safe_text(unit.get("reward_spec", {}).get("answer_type") or unit.get("prompt", {}).get("answer_format"))


def canonical_answer(unit: Dict[str, Any]) -> str:
    if answer_format(unit) == "choice_label":
        return format_choice_answer(unit.get("clean_answer", {}))
    return format_short_answer(unit.get("clean_answer", {}))


def canonical_choice_answer(choice: Dict[str, Any]) -> str:
    return format_choice_answer(
        {
            "answer_label": safe_text(choice.get("label")).upper(),
            "answer_text": safe_text(choice.get("text")),
        }
    )


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


def _find_choice_by_text(prompt: Dict[str, Any], answer_text: str) -> Optional[Dict[str, Any]]:
    normalized_target = normalize_text(answer_text)
    if not normalized_target:
        return None
    for choice in prompt.get("choices") or []:
        if normalize_text(safe_text(choice.get("text"))) == normalized_target:
            return copy.deepcopy(choice)
    return None


def extract_final_answer_text(text: str) -> str:
    text = safe_text(text)
    if not text:
        return ""
    match = _FINAL_ANSWER_RE.search(text)
    if match:
        return text[match.end():].strip()
    return text


def normalize_model_answer(output_text: str, prompt: Dict[str, Any]) -> Optional[str]:
    answer_kind = safe_text(prompt.get("answer_format"))
    extracted = extract_final_answer_text(output_text)
    if not extracted:
        return None
    if answer_kind == "choice_label":
        choices = prompt.get("choices") or []
        label_map = {safe_text(choice.get("label")).upper(): choice for choice in choices}
        label_match = re.match(r"^\(?([A-Za-z])\)?(?:[\.\):,\s]|$)", extracted)
        if label_match:
            label = label_match.group(1).upper()
            choice = label_map.get(label)
            if choice:
                return canonical_choice_answer(choice)
        normalized_extracted = normalize_text(extracted)
        matched: List[Dict[str, Any]] = []
        for choice in choices:
            choice_text = safe_text(choice.get("text"))
            if not choice_text:
                continue
            if normalize_text(choice_text) in normalized_extracted:
                matched.append(choice)
        if len(matched) == 1:
            return canonical_choice_answer(matched[0])
        return None
    if is_abstention_text(extracted):
        return format_abstain()
    return f"Final answer: {extracted}"


def choice_negative(unit: Dict[str, Any]) -> Optional[str]:
    gold_label = safe_text(unit.get("clean_answer", {}).get("answer_label")).upper()
    for choice in unit.get("prompt", {}).get("choices") or []:
        label = safe_text(choice.get("label")).upper()
        if label and label != gold_label:
            return canonical_choice_answer(choice)
    return None


def build_short_answer_negative_pools(units: Sequence[Dict[str, Any]]) -> Dict[Tuple[str, str], List[str]]:
    pools: Dict[Tuple[str, str], List[str]] = defaultdict(list)
    for unit in units:
        if answer_format(unit) != "short_text":
            continue
        key = (safe_text(unit.get("task_family")), safe_text(unit.get("question_type")))
        answer = format_short_answer(unit.get("clean_answer", {}))
        if answer not in pools[key]:
            pools[key].append(answer)
    return pools


def short_answer_negative(
    unit: Dict[str, Any],
    pools: Dict[Tuple[str, str], List[str]],
    rng: random.Random,
) -> str:
    key = (safe_text(unit.get("task_family")), safe_text(unit.get("question_type")))
    pool = [item for item in pools.get(key, []) if normalize_text(item) != normalize_text(canonical_answer(unit))]
    if pool:
        return rng.choice(pool)
    return "Final answer: A different event occurs."


def clean_negative(
    unit: Dict[str, Any],
    pools: Dict[Tuple[str, str], List[str]],
    rng: random.Random,
) -> Tuple[str, str]:
    if answer_format(unit) == "choice_label":
        alt = choice_negative(unit)
        if alt:
            return alt, "choice_negative"
    return short_answer_negative(unit, pools, rng), "pooled_negative"


def find_explicit_mismatch_choice(unit: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    for choice in unit.get("prompt", {}).get("choices") or []:
        text = safe_text(choice.get("text"))
        if not text:
            continue
        if any(pattern.search(text) for pattern in _MISMATCH_PATTERNS):
            return choice
    return None


def media_identity(unit: Dict[str, Any]) -> str:
    media = unit.get("media", {}) or {}
    source_meta = unit.get("source_meta", {}) or {}
    candidates = [
        safe_text(media.get("video_resolved")),
        safe_text(media.get("video_ref")),
        safe_text(source_meta.get("video_id")),
        safe_text(source_meta.get("omnimmi_task")),
        safe_text(unit.get("sample_id")),
    ]
    return next((value for value in candidates if value), safe_text(unit.get("sample_id")))


class SwapSelector:
    def __init__(self, units: Sequence[Dict[str, Any]]) -> None:
        self.units = list(units)
        self.by_split: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        self.by_split_dataset_task_answer: Dict[Tuple[str, str, str, str], List[Dict[str, Any]]] = defaultdict(list)
        self.by_split_dataset_answer: Dict[Tuple[str, str, str], List[Dict[str, Any]]] = defaultdict(list)
        self.by_split_answer: Dict[Tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)
        self.by_answer: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for unit in self.units:
            split = safe_text(unit.get("split"))
            dataset = safe_text(unit.get("dataset"))
            task_family = safe_text(unit.get("task_family"))
            ans = answer_format(unit)
            self.by_split[split].append(unit)
            self.by_split_dataset_task_answer[(split, dataset, task_family, ans)].append(unit)
            self.by_split_dataset_answer[(split, dataset, ans)].append(unit)
            self.by_split_answer[(split, ans)].append(unit)
            self.by_answer[ans].append(unit)

    @staticmethod
    def _pick_from_pool(
        pool: Sequence[Dict[str, Any]],
        *,
        unit: Dict[str, Any],
        transform_type: str,
        require_different_answer: bool = False,
    ) -> Optional[Dict[str, Any]]:
        if not pool:
            return None
        sample_id = safe_text(unit.get("sample_id"))
        source_id = media_identity(unit)
        source_answer = normalize_text(canonical_answer(unit)) if require_different_answer else ""
        start = int(stable_hash(f"{sample_id}:{transform_type}")[:8], 16) % len(pool)
        for offset in range(len(pool)):
            cand = pool[(start + offset) % len(pool)]
            if safe_text(cand.get("sample_id")) == sample_id:
                continue
            if media_identity(cand) == source_id:
                continue
            if require_different_answer and normalize_text(canonical_answer(cand)) == source_answer:
                continue
            return cand
        return None

    def select(self, unit: Dict[str, Any], transform_type: str) -> Dict[str, Any]:
        split = safe_text(unit.get("split"))
        answer_kind = answer_format(unit)
        dataset = safe_text(unit.get("dataset"))
        task_family = safe_text(unit.get("task_family"))
        candidate_pools = [
            self.by_split_dataset_task_answer.get((split, dataset, task_family, answer_kind), []),
            self.by_split_dataset_answer.get((split, dataset, answer_kind), []),
            self.by_split_answer.get((split, answer_kind), []),
            self.by_answer.get(answer_kind, []),
            self.by_split.get(split, []),
            self.units,
        ]
        prefer_distinct_answer = transform_type in {"swap_audio", "swap_video"}
        passes = [True, False] if prefer_distinct_answer else [False]
        for require_different_answer in passes:
            for pool in candidate_pools:
                picked = self._pick_from_pool(
                    pool,
                    unit=unit,
                    transform_type=transform_type,
                    require_different_answer=require_different_answer,
                )
                if picked is not None:
                    return picked
        raise ValueError(f"No swap candidate available for sample {safe_text(unit.get('sample_id'))}")


def stable_hash(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def build_prompt_payload(unit: Dict[str, Any], intervention: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    return {
        "prompt": {
            "question": unit.get("prompt", {}).get("question"),
            "choices": copy.deepcopy(unit.get("prompt", {}).get("choices") or []),
            "answer_format": unit.get("prompt", {}).get("answer_format"),
        },
        "media": copy.deepcopy(unit.get("media", {}) or {}),
        "intervention": copy.deepcopy(intervention) if intervention else None,
    }


def enrich_intervention(
    unit: Dict[str, Any],
    cf_view: Dict[str, Any],
    swap_selector: SwapSelector,
) -> Dict[str, Any]:
    intervention = copy.deepcopy(cf_view)
    transform_type = safe_text(intervention.get("transform", {}).get("type"))
    if transform_type not in {"swap_audio", "swap_video"}:
        return intervention
    donor = swap_selector.select(unit, transform_type=transform_type)
    intervention["swap_source"] = {
        "dataset": donor.get("dataset"),
        "sample_id": donor.get("sample_id"),
        "media": copy.deepcopy(donor.get("media", {}) or {}),
        "source_meta": copy.deepcopy(donor.get("source_meta", {}) or {}),
        "clean_answer": copy.deepcopy(donor.get("clean_answer") or {}),
        "canonical_answer": canonical_answer(donor),
        "prompt": copy.deepcopy(donor.get("prompt", {}) or {}),
    }
    return intervention


def build_pair_record(
    *,
    unit: Dict[str, Any],
    pair_id: str,
    pair_family: str,
    chosen: str,
    rejected: str,
    chosen_source: str,
    rejected_source: str,
    input_payload: Dict[str, Any],
) -> Dict[str, Any]:
    if pair_family not in ALLOWED_PAIR_FAMILIES:
        raise ValueError(f"Unsupported pair family: {pair_family}")
    if chosen_source not in ALLOWED_CHOSEN_SOURCES:
        raise ValueError(f"Unsupported chosen source: {chosen_source}")
    if rejected_source not in ALLOWED_REJECTED_SOURCES:
        raise ValueError(f"Unsupported rejected source: {rejected_source}")
    return {
        "pair_id": pair_id,
        "pair_family": pair_family,
        "sample_id": unit.get("sample_id"),
        "dataset": unit.get("dataset"),
        "split": unit.get("split"),
        "task_family": unit.get("task_family"),
        "target_modality": unit.get("target_modality"),
        "input": input_payload,
        "chosen": chosen,
        "rejected": rejected,
        "sample_weight": float(unit.get("sample_weight", 1.0)),
        "pair_weight": float(PAIR_WEIGHT_BY_FAMILY[pair_family]),
        "chosen_source": chosen_source,
        "rejected_source": rejected_source,
        "source_meta": copy.deepcopy(unit.get("source_meta", {}) or {}),
    }


def _build_clean_pair(
    unit: Dict[str, Any],
    short_answer_pools: Dict[Tuple[str, str], List[str]],
    rng: random.Random,
) -> Dict[str, Any]:
    rejected, rejected_source = clean_negative(unit, short_answer_pools, rng)
    return build_pair_record(
        unit=unit,
        pair_id=f"{safe_text(unit.get('sample_id'))}__clean_pref",
        pair_family="clean_pref",
        chosen=canonical_answer(unit),
        rejected=rejected,
        chosen_source="gold",
        rejected_source=rejected_source,
        input_payload=build_prompt_payload(unit),
    )


def _build_counterfactual_pair(
    unit: Dict[str, Any],
    cf_view: Dict[str, Any],
    swap_selector: SwapSelector,
    short_answer_pools: Dict[Tuple[str, str], List[str]],
    rng: random.Random,
) -> Optional[Dict[str, Any]]:
    task_family = safe_text(unit.get("task_family"))
    transform_type = safe_text(cf_view.get("transform", {}).get("type"))
    target_modality = safe_text(unit.get("target_modality"))
    base_answer = canonical_answer(unit)
    mismatch_choice = find_explicit_mismatch_choice(unit)
    intervention = enrich_intervention(unit, cf_view, swap_selector)
    input_payload = build_prompt_payload(unit, intervention=intervention)
    pair_id_prefix = f"{safe_text(unit.get('sample_id'))}__{safe_text(cf_view.get('view_id') or transform_type)}"
    temporal_retarget_answer: Optional[str] = None

    if (
        task_family == "temporal_alignment"
        and safe_text(unit.get("dataset")) == "omnivideobench"
        and answer_format(unit) == "choice_label"
        and transform_type in {"swap_audio", "swap_video"}
    ):
        prompt = input_payload.get("prompt") or {}
        clean_answer_text = _extract_choice_answer_text(base_answer)
        swap_source = (intervention.get("swap_source") or {}) if intervention else {}
        answer_text = safe_text(((swap_source.get("clean_answer") or {}).get("answer_text")))
        if not answer_text:
            answer_text = _extract_choice_answer_text(safe_text(swap_source.get("canonical_answer")))
        if answer_text and normalize_text(answer_text) != normalize_text(clean_answer_text):
            choice = _find_choice_by_text(prompt, answer_text)
            if choice is None:
                choice = {
                    "label": _next_choice_label(prompt.get("choices") or []),
                    "text": answer_text,
                }
                prompt.setdefault("choices", []).append(copy.deepcopy(choice))
            temporal_retarget_answer = canonical_choice_answer(choice)

    if task_family == "audio_grounded_presence":
        if transform_type in {"drop_audio", "swap_audio"}:
            pair_family = "target_abstain"
        elif transform_type == "drop_video":
            pair_family = "control_keep"
        else:
            return None
    elif task_family == "visual_grounded_presence":
        if transform_type in {"drop_video", "swap_video"}:
            pair_family = "target_abstain"
        elif transform_type == "drop_audio":
            pair_family = "control_keep"
        else:
            return None
    elif task_family == "speaker_attribution":
        if transform_type in {"drop_audio", "swap_audio", "shift_audio"}:
            pair_family = "target_abstain"
        elif transform_type == "drop_video":
            pair_family = "control_keep"
        else:
            return None
    elif task_family == "av_matching":
        if transform_type in {"drop_audio", "drop_video"}:
            pair_family = "target_abstain"
        elif transform_type in {"swap_audio", "swap_video"}:
            pair_family = "target_retarget" if mismatch_choice else "target_abstain"
        else:
            return None
    elif task_family == "temporal_alignment":
        if transform_type == "shift_audio":
            pair_family = "target_abstain"
        elif transform_type in {"swap_audio", "swap_video"}:
            if temporal_retarget_answer:
                pair_family = "target_retarget"
            else:
                pair_family = "target_retarget" if mismatch_choice else "target_abstain"
        elif transform_type == "drop_audio":
            pair_family = "target_abstain" if "audio" in target_modality or target_modality == "joint_temporal" else "control_keep"
        elif transform_type == "drop_video":
            pair_family = "target_abstain" if "visual" in target_modality or target_modality == "joint_temporal" else "control_keep"
        else:
            return None
    else:
        return None

    if pair_family == "target_abstain":
        return build_pair_record(
            unit=unit,
            pair_id=f"{pair_id_prefix}__target_abstain",
            pair_family=pair_family,
            chosen=format_abstain(),
            rejected=base_answer,
            chosen_source="abstain_template",
            rejected_source="clean_answer",
            input_payload=input_payload,
        )
    if pair_family == "target_retarget":
        chosen_answer = temporal_retarget_answer
        if chosen_answer is None and mismatch_choice is not None:
            chosen_answer = canonical_choice_answer(mismatch_choice)
        if chosen_answer is None:
            return build_pair_record(
                unit=unit,
                pair_id=f"{pair_id_prefix}__target_abstain",
                pair_family="target_abstain",
                chosen=format_abstain(),
                rejected=base_answer,
                chosen_source="abstain_template",
                rejected_source="clean_answer",
                input_payload=input_payload,
            )
        return build_pair_record(
            unit=unit,
            pair_id=f"{pair_id_prefix}__target_retarget",
            pair_family=pair_family,
            chosen=chosen_answer,
            rejected=base_answer,
            chosen_source="counterfactual_label",
            rejected_source="clean_answer",
            input_payload=input_payload,
        )

    rejected, rejected_source = clean_negative(unit, short_answer_pools, rng)
    return build_pair_record(
        unit=unit,
        pair_id=f"{pair_id_prefix}__control_keep",
        pair_family=pair_family,
        chosen=base_answer,
        rejected=rejected,
        chosen_source="gold",
        rejected_source=rejected_source,
        input_payload=input_payload,
    )


def build_base_pairs(units: Sequence[Dict[str, Any]], seed: int = 42) -> List[Dict[str, Any]]:
    rng = random.Random(seed)
    short_answer_pools = build_short_answer_negative_pools(units)
    swap_selector = SwapSelector(units)
    pairs: List[Dict[str, Any]] = []
    for unit in units:
        pairs.append(_build_clean_pair(unit, short_answer_pools=short_answer_pools, rng=rng))
        for cf_view in unit.get("counterfactual_views", []) or []:
            pair = _build_counterfactual_pair(
                unit,
                cf_view=cf_view,
                swap_selector=swap_selector,
                short_answer_pools=short_answer_pools,
                rng=rng,
            )
            if pair is not None:
                pairs.append(pair)
    return pairs


def load_baseline_hallucination_cache(path: Optional[Path]) -> List[Dict[str, Any]]:
    if path is None or not path.exists():
        return []
    rows = read_jsonl(path)
    valid_rows: List[Dict[str, Any]] = []
    for row in rows:
        if safe_text(row.get("pair_id")) and safe_text(row.get("rejected")):
            valid_rows.append(row)
    return valid_rows


def attach_hallucination_reject_pairs(
    base_pairs: Sequence[Dict[str, Any]],
    cache_rows: Sequence[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    base_by_id = {safe_text(pair.get("pair_id")): pair for pair in base_pairs}
    out = list(base_pairs)
    for row in cache_rows:
        pair_id = safe_text(row.get("pair_id"))
        base_pair = base_by_id.get(pair_id)
        if base_pair is None:
            continue
        if safe_text(base_pair.get("pair_family")) not in {"clean_pref", "target_abstain", "target_retarget"}:
            continue
        hallucinated = safe_text(row.get("rejected"))
        if not hallucinated:
            continue
        merged = copy.deepcopy(base_pair)
        merged["pair_id"] = f"{pair_id}__hallucination_reject"
        merged["pair_family"] = "hallucination_reject"
        merged["pair_weight"] = float(PAIR_WEIGHT_BY_FAMILY["hallucination_reject"])
        merged["rejected"] = hallucinated
        merged["rejected_source"] = "baseline_hallucination"
        source_meta = merged.get("source_meta", {}) or {}
        source_meta["hallucination_cache"] = {
            "cache_id": row.get("cache_id"),
            "raw_output": row.get("raw_output"),
            "normalized_output": row.get("normalized_output"),
            "base_pair_family": base_pair.get("pair_family"),
        }
        merged["source_meta"] = source_meta
        out.append(merged)
    return out


def summarize_pairs(pairs: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    return {
        "total_pairs": len(pairs),
        "by_pair_family": dict(Counter(safe_text(pair.get("pair_family")) for pair in pairs)),
        "by_task_family": dict(Counter(safe_text(pair.get("task_family")) for pair in pairs)),
        "by_split": dict(Counter(safe_text(pair.get("split")) for pair in pairs)),
        "by_dataset": dict(Counter(safe_text(pair.get("dataset")) for pair in pairs)),
        "by_chosen_source": dict(Counter(safe_text(pair.get("chosen_source")) for pair in pairs)),
        "by_rejected_source": dict(Counter(safe_text(pair.get("rejected_source")) for pair in pairs)),
    }
