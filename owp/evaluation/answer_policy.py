from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

def safe_text(value: Any) -> str:
    """Convert an optional annotation value to normalized display text."""
    if value is None:
        return ""
    return str(value).strip()


_TEMPORAL_ORDER_CUE_RE = re.compile(
    r"\b(before|after|earlier|later|first|second|precede|follow|happen first|happens first)\b",
    re.IGNORECASE,
)
_TEMPORAL_SYNC_CUE_RE = re.compile(
    r"\b(simultaneous|synchronized|sync|aligned|at the same time|together|concurrent)\b",
    re.IGNORECASE,
)
_MATCH_CUE_RE = re.compile(
    r"\b(match|matching|matched|consistent|correspond|fit|same context|same event)\b",
    re.IGNORECASE,
)
_AUDIO_CUE_RE = re.compile(r"\b(audio|sound|hear|voice|speaker|speaking)\b", re.IGNORECASE)
_VISUAL_CUE_RE = re.compile(r"\b(video|visible|see|shown|look|wearing|holding)\b", re.IGNORECASE)
_IDENTITY_CUE_RE = re.compile(r"\b(who|whose|speaker|voice|person|man|woman|boy|girl)\b", re.IGNORECASE)
_VISUAL_ACTION_CUE_RE = re.compile(
    r"\b(do|doing|action|moving|running|walking|jumping|playing|talking|speaking|singing|dancing)\b",
    re.IGNORECASE,
)
_VISUAL_ATTRIBUTE_CUE_RE = re.compile(
    r"\b(color|wearing|holding|looking|appearance|shown|visible|looks like)\b",
    re.IGNORECASE,
)
_YN_INSTRUCTION_RE = re.compile(
    r"\b(answer|respond)\s+with\s+(only\s+)?yes\s+or\s+no\b|\byes\s+or\s+no\b",
    re.IGNORECASE,
)
_YN_PREFIXES = (
    "is ",
    "are ",
    "was ",
    "were ",
    "do ",
    "does ",
    "did ",
    "can ",
    "could ",
    "will ",
    "would ",
    "has ",
    "have ",
    "had ",
    "should ",
)
_OPTION_PATTERN_RE = re.compile(
    r"([A-H]|\d{1,2})[\.\):]\s*(.+?)(?=(?:\s+(?:[A-H]|\d{1,2})[\.\):]\s)|$)",
    re.IGNORECASE | re.DOTALL,
)
_MULTI_SELECT_PATTERNS = (
    r"\bselect all that apply\b",
    r"\bchoose all that apply\b",
    r"\bmultiple answers?\b",
    r"\bmulti[\s-]?select\b",
    r"\bwhich of the following are\b",
    r"\bwhich options are\b",
    r"\ball correct\b",
    r"\bone or more\b",
)
_CHOICE_LABEL_PREFIX_RE = re.compile(
    r'^\s*["\'(\[]*(?:option\s+)?([A-Z]|\d{1,2})\b',
    re.IGNORECASE,
)
_CHOICE_LABEL_ANY_RE = re.compile(r"\b(?:option\s+)?([A-Z]|\d{1,2})\b", re.IGNORECASE)


@dataclass
class AnswerOption:
    label: str
    text: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class AnswerSpaceSpec:
    kind: str
    options: List[AnswerOption] = field(default_factory=list)
    labels_only: bool = False
    question_form: str = "open_ended"
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["options"] = [option.to_dict() for option in self.options]
        return payload


@dataclass
class QueryLatentState:
    query_family: str
    task_family: str
    predicate: str
    primary_modality: str
    required_modalities: Tuple[str, ...]
    corroborative_modalities: Tuple[str, ...]
    question_form: str
    answer_space: AnswerSpaceSpec
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["answer_space"] = self.answer_space.to_dict()
        return payload


def _normalize_yes_no(text: Optional[str]) -> Optional[str]:
    if text is None:
        return None
    normalized = str(text).strip()
    if not normalized:
        return None
    match = re.match(r'^\s*["\'(\[]*(yes|no)\b', normalized, flags=re.IGNORECASE)
    if match:
        return match.group(1).capitalize()
    match = re.search(r"\b(?:final answer|answer)\s*[:\-]?\s*(yes|no)\b", normalized, flags=re.IGNORECASE)
    if match:
        return match.group(1).capitalize()
    return None


def _normalize_choice_label(text: Optional[str]) -> str:
    value = safe_text(text)
    if not value:
        return ""
    return value.upper()


def _normalize_free_text(text: Optional[str]) -> str:
    candidate = safe_text(text)
    if not candidate:
        return ""
    normalized = re.sub(r"[^a-z0-9]+", " ", candidate.lower())
    return " ".join(normalized.split())


def infer_query_family(question: str) -> str:
    text = safe_text(question)
    if not text:
        return "unknown"
    if _TEMPORAL_ORDER_CUE_RE.search(text):
        return "temporal_order"
    if _TEMPORAL_SYNC_CUE_RE.search(text):
        return "temporal_sync"
    if _MATCH_CUE_RE.search(text):
        return "cross_modal_match"

    has_audio_cue = bool(_AUDIO_CUE_RE.search(text))
    has_visual_cue = bool(_VISUAL_CUE_RE.search(text))
    if has_audio_cue and has_visual_cue:
        return "cross_modal_relation"
    if has_audio_cue:
        if _IDENTITY_CUE_RE.search(text):
            return "audio_identity"
        return "audio_presence"
    if has_visual_cue:
        if _VISUAL_ACTION_CUE_RE.search(text):
            return "visual_action"
        if _VISUAL_ATTRIBUTE_CUE_RE.search(text):
            return "visual_attribute"
        if _IDENTITY_CUE_RE.search(text):
            return "visual_identity"
        return "visual_presence"
    return "unknown"


def infer_task_family_from_query_family(query_family: str) -> str:
    family = safe_text(query_family)
    if family in {"audio_presence", "audio_identity"}:
        return "audio_grounded_presence"
    if family in {"visual_presence", "visual_identity", "visual_attribute", "visual_action"}:
        return "visual_grounded_presence"
    if family in {"temporal_order", "temporal_sync"}:
        return "temporal_alignment"
    if family in {"cross_modal_match", "cross_modal_relation"}:
        return "av_matching"
    return "unknown"


def _normalize_question_text(text: str) -> str:
    normalized = re.sub(r"[^a-z0-9']+", " ", (text or "").lower())
    return " ".join(normalized.split())


def _strip_yes_no_instruction(question: str) -> str:
    return re.sub(
        r"\s*(answer|respond)\s+with\s+(only\s+)?yes\s+or\s+no\.?\s*$",
        "",
        question or "",
        flags=re.IGNORECASE,
    ).strip()


def _is_yes_no_question(question: str, max_new_tokens: Optional[int] = None) -> bool:
    normalized = (question or "").strip().lower()
    if _YN_INSTRUCTION_RE.search(normalized):
        return True
    if max_new_tokens is not None and max_new_tokens <= 10:
        return True
    return normalized.startswith(_YN_PREFIXES)


def _extract_options(question: str) -> List[AnswerOption]:
    raw = re.sub(r"\s+", " ", question or "").strip()
    options: List[AnswerOption] = []
    for match in _OPTION_PATTERN_RE.finditer(raw):
        label = _normalize_choice_label(match.group(1))
        text = re.sub(r"\s+", " ", match.group(2) or "").strip(" ;,")
        if label and text:
            options.append(AnswerOption(label=label, text=text))
    return options


def _infer_predicate(normalized_question: str) -> str:
    if any(re.search(pattern, normalized_question) for pattern in (r"\bwhen\b", r"\bbefore\b", r"\bafter\b", r"\bwhile\b", r"\bduring\b", r"\bfirst\b", r"\bthen\b", r"\blater\b", r"\bearlier\b", r"\btiming\b", r"\border\b", r"\bsequence\b")):
        return "temporal_order"
    if any(token in normalized_question for token in ("emotion", "mood", "feeling", "tone", "sentiment")):
        return "emotion"
    if any(token in normalized_question for token in ("same event", "same context", "match", "consistent", "align", "correspond")):
        return "cross_modal_consistency"
    if any(token in normalized_question for token in ("what is being said", "what does", "say", "saying", "spoken", "transcript")):
        return "speech_content"
    if any(token in normalized_question for token in ("sound", "audio", "hear", "heard", "audible", "noise")):
        return "sound_source"
    if any(token in normalized_question for token in ("visible", "see", "seen", "shown", "appear", "in the video", "in the scene")):
        return "visibility"
    return "attribute"


def _infer_primary_modality(normalized_question: str, predicate: str) -> str:
    audio_hits = bool(re.search(r"\b(audio|sound|sounds|hear|heard|audible|speech|voice|voices)\b", normalized_question))
    visual_hits = bool(re.search(r"\b(video|visual|scene|frame|frames|visible|see|seen|shown)\b", normalized_question))
    if predicate in {"cross_modal_consistency", "temporal_order"}:
        return "cross_modal"
    if predicate in {"speech_content", "sound_source"}:
        return "audio"
    if predicate == "visibility":
        return "visual"
    if predicate == "emotion":
        if audio_hits and not visual_hits:
            return "audio"
        if visual_hits and not audio_hits:
            return "visual"
        return "cross_modal"
    if audio_hits and not visual_hits:
        return "audio"
    if visual_hits and not audio_hits:
        return "visual"
    return "cross_modal"


def _required_modalities(predicate: str, primary_modality: str) -> Tuple[str, ...]:
    if predicate == "visibility":
        return ("visual",)
    if predicate in {"sound_source", "speech_content"}:
        return ("audio",)
    if predicate in {"cross_modal_consistency", "temporal_order"}:
        return ("audio", "visual")
    if primary_modality == "cross_modal":
        return ("audio", "visual")
    return (primary_modality,)


def _corroborative_modalities(predicate: str, primary_modality: str) -> Tuple[str, ...]:
    if predicate == "visibility":
        return ("audio",)
    if predicate in {"sound_source", "speech_content"}:
        return ("visual",)
    if primary_modality == "cross_modal":
        return tuple()
    return ("audio",) if primary_modality == "visual" else ("visual",)


def _parse_question_spec(question: str, *, max_new_tokens: Optional[int] = None) -> Dict[str, Any]:
    stripped = _strip_yes_no_instruction(question)
    normalized = _normalize_question_text(stripped)
    options = _extract_options(question)
    if _is_yes_no_question(question, max_new_tokens=max_new_tokens):
        form = "yes_no"
    elif len(options) >= 2:
        form = "multi_select" if any(re.search(pattern, normalized) for pattern in _MULTI_SELECT_PATTERNS) else "single_choice"
    else:
        form = "open_ended"
    predicate = _infer_predicate(normalized)
    primary_modality = _infer_primary_modality(normalized, predicate)
    return {
        "raw_question": question,
        "normalized_question": normalized,
        "form": form,
        "predicate": predicate,
        "primary_modality": primary_modality,
        "required_modalities": _required_modalities(predicate, primary_modality),
        "corroborative_modalities": _corroborative_modalities(predicate, primary_modality),
        "options": [option.to_dict() for option in options],
    }


def _coerce_options(options: Optional[Sequence[Dict[str, Any]]]) -> List[AnswerOption]:
    coerced: List[AnswerOption] = []
    for option in options or []:
        label = _normalize_choice_label(option.get("label"))
        text = safe_text(option.get("text"))
        if label or text:
            coerced.append(AnswerOption(label=label, text=text))
    return coerced


def build_answer_space_spec(
    question: str,
    *,
    eval_type: Optional[str] = None,
    options: Optional[Sequence[Dict[str, Any]]] = None,
    max_new_tokens: Optional[int] = None,
) -> AnswerSpaceSpec:
    parsed = _parse_question_spec(question, max_new_tokens=max_new_tokens)
    resolved_options = _coerce_options(options)
    if not resolved_options:
        resolved_options = _coerce_options(parsed.get("options") or [])

    normalized_eval_type = safe_text(eval_type)
    if normalized_eval_type == "yes_no":
        kind = "yes_no"
    elif normalized_eval_type in {"single_choice", "multi_select"}:
        kind = normalized_eval_type
    elif safe_text(parsed.get("form")) in {"yes_no", "single_choice", "multi_select"}:
        kind = safe_text(parsed.get("form"))
    elif len(resolved_options) >= 2:
        kind = "single_choice"
    else:
        kind = "open_ended"

    labels_only = kind in {"single_choice", "multi_select"} and all(option.label for option in resolved_options)
    return AnswerSpaceSpec(
        kind=kind,
        options=resolved_options,
        labels_only=labels_only,
        question_form=safe_text(parsed.get("form")) or "open_ended",
        metadata={"query_spec": parsed},
    )


def format_question_for_answer_space(question: str, answer_space: AnswerSpaceSpec) -> str:
    def _render_options_block() -> str:
        lines: List[str] = []
        for option in answer_space.options:
            label = safe_text(option.label)
            text = safe_text(option.text)
            if label and text:
                lines.append(f"{label}. {text}")
            elif label:
                lines.append(label)
            elif text:
                lines.append(text)
        if not lines:
            return ""
        return "Options:\n" + "\n".join(lines)

    def _has_rendered_options(prompt_text: str) -> bool:
        if not answer_space.options:
            return False
        return all(
            (not safe_text(option.label))
            or (f"{safe_text(option.label)}." in prompt_text)
            or (f"{safe_text(option.label)})" in prompt_text)
            or (f"{safe_text(option.label)}:" in prompt_text)
            for option in answer_space.options
        )

    prompt = safe_text(question)
    if not prompt:
        if answer_space.kind == "yes_no":
            return "Answer with only Yes or No."
        if answer_space.kind == "multi_select":
            rendered_options = _render_options_block()
            prefix = f"{rendered_options}\n" if rendered_options else ""
            return f"{prefix}Answer with only the option labels separated by commas."
        if answer_space.kind == "single_choice":
            rendered_options = _render_options_block()
            prefix = f"{rendered_options}\n" if rendered_options else ""
            return f"{prefix}Answer with only the option label."
        return ""
    if answer_space.kind == "yes_no":
        if _YN_INSTRUCTION_RE.search(prompt):
            return prompt
        return f"{prompt}\nAnswer with only Yes or No."
    if answer_space.kind == "single_choice":
        rendered_options = _render_options_block()
        option_block = ""
        if rendered_options and not _has_rendered_options(prompt):
            option_block = f"\n{rendered_options}"
        if re.search(r"\boption label\b", prompt, flags=re.IGNORECASE):
            return f"{prompt}{option_block}"
        return f"{prompt}{option_block}\nAnswer with only the option label."
    if answer_space.kind == "multi_select":
        rendered_options = _render_options_block()
        option_block = ""
        if rendered_options and not _has_rendered_options(prompt):
            option_block = f"\n{rendered_options}"
        if re.search(r"\boption labels\b", prompt, flags=re.IGNORECASE):
            return f"{prompt}{option_block}"
        return f"{prompt}{option_block}\nAnswer with only the option labels separated by commas."
    return prompt


def build_query_latent_state(
    question: str,
    *,
    eval_type: Optional[str] = None,
    options: Optional[Sequence[Dict[str, Any]]] = None,
    max_new_tokens: Optional[int] = None,
) -> QueryLatentState:
    answer_space = build_answer_space_spec(
        question,
        eval_type=eval_type,
        options=options,
        max_new_tokens=max_new_tokens,
    )
    query_spec = (answer_space.metadata or {}).get("query_spec") or {}
    query_family = infer_query_family(question)
    return QueryLatentState(
        query_family=query_family,
        task_family=infer_task_family_from_query_family(query_family),
        predicate=safe_text(query_spec.get("predicate")) or "unknown",
        primary_modality=safe_text(query_spec.get("primary_modality")) or "unknown",
        required_modalities=tuple(query_spec.get("required_modalities") or ()),
        corroborative_modalities=tuple(query_spec.get("corroborative_modalities") or ()),
        question_form=answer_space.kind,
        answer_space=answer_space,
        metadata={"query_spec": query_spec},
    )


def legacy_query_latent_state(task_family: str, *, answer_space: Optional[AnswerSpaceSpec] = None) -> QueryLatentState:
    answer_space = answer_space or AnswerSpaceSpec(kind="yes_no", question_form="yes_no")
    family = safe_text(task_family) or "unknown"
    if family in {"audio_grounded_presence", "speaker_attribution"}:
        required = ("audio",)
        primary = "audio"
    elif family == "visual_grounded_presence":
        required = ("visual",)
        primary = "visual"
    elif family in {"av_matching", "temporal_alignment"}:
        required = ("audio", "visual")
        primary = "cross_modal"
    else:
        required = tuple()
        primary = "unknown"
    return QueryLatentState(
        query_family="unknown",
        task_family=family,
        predicate="unknown",
        primary_modality=primary,
        required_modalities=required,
        corroborative_modalities=tuple(),
        question_form=answer_space.kind,
        answer_space=answer_space,
    )


def override_query_answer_space(
    query_state: QueryLatentState,
    answer_space: AnswerSpaceSpec,
    *,
    metadata: Optional[Dict[str, Any]] = None,
) -> QueryLatentState:
    merged_metadata = dict(query_state.metadata or {})
    if metadata:
        merged_metadata.update(metadata)
    return QueryLatentState(
        query_family=query_state.query_family,
        task_family=query_state.task_family,
        predicate=query_state.predicate,
        primary_modality=query_state.primary_modality,
        required_modalities=tuple(query_state.required_modalities),
        corroborative_modalities=tuple(query_state.corroborative_modalities),
        question_form=answer_space.kind,
        answer_space=answer_space,
        metadata=merged_metadata,
    )


def candidate_specs_from_query_state(query_state: QueryLatentState) -> List[Dict[str, str]]:
    required = tuple(safe_text(modality) for modality in query_state.required_modalities if safe_text(modality))
    primary = safe_text(query_state.primary_modality)
    include_audio = "audio" in required
    include_visual = "visual" in required

    if not include_audio and not include_visual:
        if primary == "audio":
            include_audio = True
        elif primary == "visual":
            include_visual = True
        else:
            include_audio = True
            include_visual = True

    specs: List[Dict[str, str]] = []
    if include_audio:
        specs.append(
            {
                "target_branch_id": "no_audio",
                "support_branch_id": "no_visual",
                "candidate_role": "drop_required_audio",
                "dropped_modality": "audio",
                "support_modality": "audio",
            }
        )
    if include_visual:
        specs.append(
            {
                "target_branch_id": "no_visual",
                "support_branch_id": "no_audio",
                "candidate_role": "drop_required_visual",
                "dropped_modality": "visual",
                "support_modality": "visual",
            }
        )
    return specs


def _single_choice_option_maps(answer_space: AnswerSpaceSpec) -> tuple[Dict[str, AnswerOption], Dict[str, str]]:
    label_map = {option.label: option for option in answer_space.options if option.label}
    text_map = {
        _normalize_free_text(option.text): option.label
        for option in answer_space.options
        if option.text and option.label
    }
    return label_map, text_map


def _normalize_single_choice_answer(answer: Optional[str], answer_space: AnswerSpaceSpec) -> Optional[str]:
    candidate = safe_text(answer)
    if not candidate:
        return None
    label_map, text_map = _single_choice_option_maps(answer_space)
    if not label_map and not text_map:
        return candidate

    prefix_match = _CHOICE_LABEL_PREFIX_RE.match(candidate)
    if prefix_match:
        normalized_label = _normalize_choice_label(prefix_match.group(1))
        if normalized_label in label_map:
            return normalized_label

    stripped = candidate.strip().strip("()[]{}\"' ")
    normalized_label = _normalize_choice_label(stripped.rstrip(".,:;)"))
    if normalized_label in label_map:
        return normalized_label

    normalized_text = _normalize_free_text(candidate)
    if normalized_text in text_map:
        return text_map[normalized_text]

    seen_labels = [
        _normalize_choice_label(match.group(1))
        for match in _CHOICE_LABEL_ANY_RE.finditer(candidate)
        if _normalize_choice_label(match.group(1)) in label_map
    ]
    if len(seen_labels) == 1:
        return seen_labels[0]
    return None


def _normalize_multi_select_answer(answer: Optional[str], answer_space: AnswerSpaceSpec) -> Optional[str]:
    candidate = safe_text(answer)
    if not candidate:
        return None
    label_map, text_map = _single_choice_option_maps(answer_space)
    ordered_labels = [option.label for option in answer_space.options if option.label]
    seen = {
        _normalize_choice_label(match.group(1))
        for match in _CHOICE_LABEL_ANY_RE.finditer(candidate)
        if _normalize_choice_label(match.group(1)) in label_map
    }
    if seen:
        return ",".join(label for label in ordered_labels if label in seen)

    normalized_chunks = {
        _normalize_free_text(chunk)
        for chunk in re.split(r"[;,]| and ", candidate)
        if _normalize_free_text(chunk)
    }
    matched = {text_map[chunk] for chunk in normalized_chunks if chunk in text_map}
    if matched:
        return ",".join(label for label in ordered_labels if label in matched)
    return None


def normalize_answer_for_space(answer: Optional[str], answer_space: AnswerSpaceSpec) -> Optional[str]:
    if answer_space.kind == "yes_no":
        return _normalize_yes_no(answer)
    if answer_space.kind == "single_choice":
        return _normalize_single_choice_answer(answer, answer_space)
    if answer_space.kind == "multi_select":
        return _normalize_multi_select_answer(answer, answer_space)
    candidate = safe_text(answer)
    return candidate or None


def render_answer_for_space(answer: Optional[str], answer_space: AnswerSpaceSpec) -> str:
    normalized = normalize_answer_for_space(answer, answer_space)
    if not normalized:
        return ""
    if answer_space.kind == "yes_no":
        return f"{normalized}."
    if answer_space.kind == "single_choice":
        label_map, _ = _single_choice_option_maps(answer_space)
        option = label_map.get(normalized)
        if option is None or answer_space.labels_only or not option.text:
            return normalized
        return f"{option.label}. {option.text}"
    if answer_space.kind == "multi_select":
        labels = [piece.strip() for piece in normalized.split(",") if piece.strip()]
        label_map, _ = _single_choice_option_maps(answer_space)
        if answer_space.labels_only:
            return ", ".join(labels)
        rendered = []
        for label in labels:
            option = label_map.get(label)
            rendered.append(f"{label}. {option.text}" if option is not None and option.text else label)
        return ", ".join(rendered)
    return normalized


def branch_answer(
    branch_scores: Dict[str, Dict[str, Any]],
    branch_id: str,
    answer_space: AnswerSpaceSpec,
) -> Optional[str]:
    branch_payload = branch_scores.get(branch_id) or {}
    return normalize_answer_for_space(branch_payload.get("answer"), answer_space)


def agreement_answer(
    branch_scores: Dict[str, Dict[str, Any]],
    answer_space: AnswerSpaceSpec,
) -> Optional[str]:
    no_audio = branch_answer(branch_scores, "no_audio", answer_space)
    no_visual = branch_answer(branch_scores, "no_visual", answer_space)
    if no_audio and no_audio == no_visual:
        return no_audio
    return None


def support_modality_from_branch_id(branch_id: str) -> str:
    normalized = safe_text(branch_id)
    if normalized == "no_visual":
        return "audio"
    if normalized == "no_audio":
        return "visual"
    return ""


def mismatch_answer_availability(
    *,
    query_state: QueryLatentState,
    branch_scores: Dict[str, Dict[str, Any]],
    target_branch_id: str,
    support_branch_id: str,
    baseline_answer: Optional[str],
) -> tuple[bool, str]:
    answer_space = query_state.answer_space
    full_answer = branch_answer(branch_scores, "full", answer_space) or normalize_answer_for_space(baseline_answer, answer_space)
    support_answer = branch_answer(branch_scores, support_branch_id, answer_space)
    target_answer = branch_answer(branch_scores, target_branch_id, answer_space)
    shared_unimodal_answer = agreement_answer(branch_scores, answer_space)

    required = tuple(safe_text(modality) for modality in query_state.required_modalities if safe_text(modality))
    cross_modal_required = len(required) >= 2 or safe_text(query_state.primary_modality) == "cross_modal"
    unimodal_required = len(required) == 1

    if unimodal_required:
        if support_answer and support_answer != full_answer:
            return True, "support_branch_change"
        return False, "none"

    if cross_modal_required:
        if shared_unimodal_answer and shared_unimodal_answer != full_answer:
            return True, "unimodal_agreement_change"
        if target_answer and target_answer != full_answer:
            return True, "target_branch_change"
        if support_answer and support_answer != full_answer:
            return True, "support_branch_change"
        return False, "none"

    if support_answer and support_answer != full_answer:
        return True, "generic_support_branch_change"
    if shared_unimodal_answer and shared_unimodal_answer != full_answer:
        return True, "generic_unimodal_agreement_change"
    if target_answer and target_answer != full_answer:
        return True, "generic_target_branch_change"
    return False, "none"


def decide_wrapper_output(
    *,
    baseline_raw_output: str,
    baseline_answer: Optional[str],
    branch_scores: Dict[str, Dict[str, Any]],
    query_state: QueryLatentState,
    selected_record: Optional[Dict[str, Any]],
    unimodal_abstain_mode: str,
) -> tuple[str, str, Dict[str, Any]]:
    metadata: Dict[str, Any] = {
        "selection_strategy": "keep_full",
        "selected_branch": "",
        "selected_action": "",
        "fallback_reason": "",
    }
    if selected_record is None:
        return baseline_raw_output, "keep_full", metadata

    if safe_text(selected_record.get("policy_family")) == "macs_omni_v1_1":
        return decide_wrapper_output_macs_phase1(
            baseline_raw_output=baseline_raw_output,
            baseline_answer=baseline_answer,
            branch_scores=branch_scores,
            query_state=query_state,
            selected_record=selected_record,
        )

    answer_space = query_state.answer_space
    full_answer = branch_answer(branch_scores, "full", answer_space) or normalize_answer_for_space(baseline_answer, answer_space)
    target_branch_id = safe_text(selected_record.get("target_branch_id"))
    support_branch_id = safe_text(selected_record.get("support_branch_id"))
    target_answer = branch_answer(branch_scores, target_branch_id, answer_space)
    support_answer = branch_answer(branch_scores, support_branch_id, answer_space)
    shared_unimodal_answer = agreement_answer(branch_scores, answer_space)
    selected_action = safe_text(selected_record.get("predicted_action"))
    metadata["selected_action"] = selected_action
    metadata["selected_branch"] = support_branch_id

    if selected_action == "abstain":
        if safe_text(unimodal_abstain_mode) == "keep_full":
            metadata["selection_strategy"] = "keep_full_abstain"
            if baseline_raw_output:
                return baseline_raw_output, "keep_full_abstain", metadata
            rendered = render_answer_for_space(full_answer, answer_space)
            if rendered:
                return rendered, "keep_full_abstain_label", metadata
            return "", "fallback_empty", metadata
        if support_answer:
            metadata["selection_strategy"] = "support_branch_abstain"
            return render_answer_for_space(support_answer, answer_space), "support_branch_abstain", metadata

    required = tuple(safe_text(modality) for modality in query_state.required_modalities if safe_text(modality))
    cross_modal_required = len(required) >= 2 or safe_text(query_state.primary_modality) == "cross_modal"
    unimodal_required = len(required) == 1

    if selected_action in {"mismatch", "retarget"}:
        if cross_modal_required and shared_unimodal_answer and shared_unimodal_answer != full_answer:
            metadata["selection_strategy"] = "unimodal_agreement"
            metadata["selected_branch"] = "no_audio+no_visual"
            return render_answer_for_space(shared_unimodal_answer, answer_space), "unimodal_agreement", metadata
        if selected_action == "mismatch" and cross_modal_required and target_answer and target_answer != full_answer:
            metadata["selection_strategy"] = "target_branch_mismatch"
            metadata["selected_branch"] = target_branch_id
            return render_answer_for_space(target_answer, answer_space), "target_branch_mismatch", metadata
        if unimodal_required and support_answer and support_answer != full_answer:
            strategy = "support_branch"
            metadata["selection_strategy"] = strategy
            return render_answer_for_space(support_answer, answer_space), strategy, metadata
        # For unimodal queries, dropping the required modality is not a reliable executable
        # mismatch path on benchmark. If the support branch does not expose a changed answer,
        # preserve the baseline instead of forcing the target branch answer.
        if selected_action == "mismatch" and unimodal_required:
            metadata["selection_strategy"] = "keep_full_unimodal_mismatch_guard"
            if baseline_raw_output:
                return baseline_raw_output, "keep_full_unimodal_mismatch_guard", metadata
            rendered = render_answer_for_space(full_answer, answer_space)
            if rendered:
                return rendered, "keep_full_unimodal_mismatch_guard_label", metadata
            return "", "fallback_empty", metadata
        if shared_unimodal_answer and shared_unimodal_answer != full_answer:
            metadata["selection_strategy"] = "generic_unimodal_agreement"
            metadata["selected_branch"] = "no_audio+no_visual"
            return render_answer_for_space(shared_unimodal_answer, answer_space), "generic_unimodal_agreement", metadata
        if selected_action == "mismatch" and target_answer and target_answer != full_answer:
            metadata["selection_strategy"] = "generic_target_branch"
            metadata["selected_branch"] = target_branch_id
            return render_answer_for_space(target_answer, answer_space), "generic_target_branch", metadata
        if support_answer and support_answer != full_answer:
            metadata["selection_strategy"] = "generic_support_branch"
            return render_answer_for_space(support_answer, answer_space), "generic_support_branch", metadata
        if selected_action == "retarget":
            full_margin = float((branch_scores.get("full") or {}).get("margin", 0.0))
            branch_ids = ("no_audio", "no_visual")
            valid_branch_ids = [
                branch_id
                for branch_id in branch_ids
                if branch_answer(branch_scores, branch_id, answer_space)
                and branch_answer(branch_scores, branch_id, answer_space) != full_answer
            ]
            if valid_branch_ids:
                best_branch_id = max(
                    valid_branch_ids,
                    key=lambda branch_id: float((branch_scores.get(branch_id) or {}).get("margin", -1.0)),
                )
                best_margin = float((branch_scores.get(best_branch_id) or {}).get("margin", 0.0))
                best_answer = branch_answer(branch_scores, best_branch_id, answer_space)
                if best_answer and best_margin >= full_margin + 0.05:
                    metadata["selection_strategy"] = "max_margin_branch"
                    metadata["selected_branch"] = best_branch_id
                    return render_answer_for_space(best_answer, answer_space), "max_margin_branch", metadata

    metadata["selection_strategy"] = "keep_full"
    if baseline_raw_output:
        return baseline_raw_output, "keep_full", metadata
    rendered = render_answer_for_space(full_answer, answer_space)
    if rendered:
        return rendered, "keep_full_label", metadata
    return "", "fallback_empty", metadata


def derive_oracle_action(
    *,
    query_state: QueryLatentState,
    branch_scores: Dict[str, Dict[str, Any]],
    reference_answer: Optional[str],
    target_branch_id: str,
    support_branch_id: str,
) -> Dict[str, Any]:
    answer_space = query_state.answer_space
    reference = normalize_answer_for_space(reference_answer, answer_space)
    full_answer = branch_answer(branch_scores, "full", answer_space)
    support_answer = branch_answer(branch_scores, support_branch_id, answer_space)
    target_answer = branch_answer(branch_scores, target_branch_id, answer_space)
    shared_unimodal_answer = agreement_answer(branch_scores, answer_space)
    required = tuple(safe_text(modality) for modality in query_state.required_modalities if safe_text(modality))
    cross_modal_required = len(required) >= 2 or safe_text(query_state.primary_modality) == "cross_modal"
    unimodal_required = len(required) == 1

    action = "abstain"
    reason = "no_executable_fix"
    executable_answer = full_answer

    if reference is None:
        reason = "invalid_reference"
    elif unimodal_required:
        if support_answer and support_answer == reference and support_answer != full_answer:
            action = "retarget"
            reason = "support_branch_matches_reference"
            executable_answer = support_answer
        elif full_answer == reference:
            reason = "full_already_correct"
        else:
            reason = "no_reliable_unimodal_fix"
    elif cross_modal_required:
        if shared_unimodal_answer and shared_unimodal_answer == reference and shared_unimodal_answer != full_answer:
            action = "mismatch"
            reason = "unimodal_agreement_matches_reference"
            executable_answer = shared_unimodal_answer
        elif target_answer and target_answer == reference and target_answer != full_answer:
            action = "mismatch"
            reason = "target_branch_matches_reference"
            executable_answer = target_answer
        elif support_answer and support_answer == reference and support_answer != full_answer:
            action = "retarget"
            reason = "support_branch_matches_reference"
            executable_answer = support_answer
        elif full_answer == reference:
            reason = "full_already_correct"
        else:
            reason = "no_cross_modal_fix"
    else:
        if shared_unimodal_answer and shared_unimodal_answer == reference and shared_unimodal_answer != full_answer:
            action = "mismatch"
            reason = "generic_unimodal_agreement_matches_reference"
            executable_answer = shared_unimodal_answer
        elif target_answer and target_answer == reference and target_answer != full_answer:
            action = "mismatch"
            reason = "generic_target_branch_matches_reference"
            executable_answer = target_answer
        elif support_answer and support_answer == reference and support_answer != full_answer:
            action = "retarget"
            reason = "generic_support_branch_matches_reference"
            executable_answer = support_answer
        elif full_answer == reference:
            reason = "full_already_correct"

    return {
        "action": action,
        "reason": reason,
        "reference_answer": reference,
        "full_answer": full_answer,
        "support_answer": support_answer,
        "target_answer": target_answer,
        "agreement_answer": shared_unimodal_answer,
        "executable_answer": executable_answer,
        "executable_matches_reference": executable_answer is not None and executable_answer == reference,
    }


def decide_wrapper_output_macs_phase1(
    *,
    baseline_raw_output: str,
    baseline_answer: Optional[str],
    branch_scores: Dict[str, Dict[str, Any]],
    query_state: QueryLatentState,
    selected_record: Optional[Dict[str, Any]],
) -> tuple[str, str, Dict[str, Any]]:
    metadata: Dict[str, Any] = {
        "selection_strategy": "keep_full",
        "selected_branch": "",
        "selected_action": "",
        "fallback_reason": "",
    }
    if selected_record is None:
        return baseline_raw_output, "keep_full", metadata

    answer_space = query_state.answer_space
    full_answer = branch_answer(branch_scores, "full", answer_space) or normalize_answer_for_space(
        baseline_answer,
        answer_space,
    )
    full_margin = float((branch_scores.get("full") or {}).get("margin", 0.0))
    selected_action = safe_text(selected_record.get("predicted_action"))
    relevant_modality = safe_text(selected_record.get("relevant_modality"))
    support_branch_id = safe_text(selected_record.get("support_branch_id"))

    metadata["selected_action"] = selected_action

    if selected_action != "retarget":
        metadata["fallback_reason"] = "selected_action_not_retarget"
        if baseline_raw_output:
            return baseline_raw_output, "keep_full", metadata
        rendered = render_answer_for_space(full_answer, answer_space)
        if rendered:
            return rendered, "keep_full_label", metadata
        return "", "fallback_empty", metadata

    if relevant_modality == "joint":
        metadata["fallback_reason"] = "joint_modality_requires_keep"
        if baseline_raw_output:
            return baseline_raw_output, "keep_full", metadata
        rendered = render_answer_for_space(full_answer, answer_space)
        if rendered:
            return rendered, "keep_full_label", metadata
        return "", "fallback_empty", metadata

    support_answer = branch_answer(branch_scores, support_branch_id, answer_space)
    support_modality = support_modality_from_branch_id(support_branch_id)
    if (
        support_answer
        and support_answer != full_answer
        and support_modality
        and support_modality == relevant_modality
    ):
        metadata["selection_strategy"] = "macs_support_branch"
        metadata["selected_branch"] = support_branch_id
        return render_answer_for_space(support_answer, answer_space), "macs_support_branch", metadata

    if support_answer and support_answer == full_answer:
        metadata["fallback_reason"] = "support_branch_no_change"
    elif not support_answer:
        metadata["fallback_reason"] = "support_branch_missing_answer"
    elif support_modality and support_modality != relevant_modality:
        metadata["fallback_reason"] = "support_modality_mismatch"
    else:
        metadata["fallback_reason"] = "support_branch_unusable"

    valid_branch_ids = [
        branch_id
        for branch_id in ("no_audio", "no_visual")
        if branch_answer(branch_scores, branch_id, answer_space)
        and branch_answer(branch_scores, branch_id, answer_space) != full_answer
    ]
    if valid_branch_ids:
        best_branch_id = max(
            valid_branch_ids,
            key=lambda branch_id: float((branch_scores.get(branch_id) or {}).get("margin", -1.0)),
        )
        best_margin = float((branch_scores.get(best_branch_id) or {}).get("margin", 0.0))
        best_answer = branch_answer(branch_scores, best_branch_id, answer_space)
        if best_answer and best_margin >= full_margin + 0.05:
            metadata["selection_strategy"] = "macs_max_margin_branch"
            metadata["selected_branch"] = best_branch_id
            metadata["fallback_reason"] = ""
            return render_answer_for_space(best_answer, answer_space), "macs_max_margin_branch", metadata
        metadata["fallback_reason"] = "max_margin_below_guard"
    else:
        metadata["fallback_reason"] = metadata["fallback_reason"] or "no_valid_unimodal_fallback"

    metadata["selection_strategy"] = "keep_full"
    if baseline_raw_output:
        return baseline_raw_output, "keep_full", metadata
    rendered = render_answer_for_space(full_answer, answer_space)
    if rendered:
        return rendered, "keep_full_label", metadata
    return "", "fallback_empty", metadata
