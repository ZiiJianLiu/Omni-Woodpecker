from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from cirpo_omni.benchmark_policy import (
    AnswerOption,
    AnswerSpaceSpec,
    format_question_for_answer_space,
    normalize_answer_for_space,
    render_answer_for_space,
)
from omni_dpo.preference_pairs import ABSTAIN_TEMPLATE, safe_text


GENERAL_POLICY_V2_FORMAT = "general_policy_v2_decoder"
GENERAL_V2_SELECTOR_VALUES = (
    "keep",
    "retarget",
    "abstain",
)
GENERAL_V2_SOURCE_VALUES = (
    "support",
    "target",
    "agreement",
)
GENERAL_V2_SOURCE_WITH_NONE = GENERAL_V2_SOURCE_VALUES + ("none",)
GENERAL_V2_SELECTOR_TO_ID = {label: idx for idx, label in enumerate(GENERAL_V2_SELECTOR_VALUES)}
GENERAL_V2_SOURCE_TO_ID = {label: idx for idx, label in enumerate(GENERAL_V2_SOURCE_VALUES)}

_REASON_TO_SOURCE = {
    "support_branch_matches_reference": "support",
    "generic_support_branch_matches_reference": "support",
    "target_branch_matches_reference": "target",
    "generic_target_branch_matches_reference": "target",
    "unimodal_agreement_matches_reference": "agreement",
    "generic_unimodal_agreement_matches_reference": "agreement",
}


def answer_space_spec_from_dict(payload: Optional[Dict[str, Any]]) -> AnswerSpaceSpec:
    payload = dict(payload or {})
    options = [
        AnswerOption(label=safe_text(option.get("label")), text=safe_text(option.get("text")))
        for option in (payload.get("options") or [])
        if safe_text((option or {}).get("label")) or safe_text((option or {}).get("text"))
    ]
    return AnswerSpaceSpec(
        kind=safe_text(payload.get("kind")) or "open_ended",
        options=options,
        labels_only=bool(payload.get("labels_only")),
        question_form=safe_text(payload.get("question_form")) or "open_ended",
        metadata=dict(payload.get("metadata") or {}),
    )


def canonical_answer_text(answer: Optional[str], answer_space: AnswerSpaceSpec) -> str:
    return safe_text(render_answer_for_space(answer, answer_space))


def canonical_allowed_texts(answer_space: AnswerSpaceSpec) -> List[str]:
    if answer_space.kind == "yes_no":
        return [text for text in [canonical_answer_text("Yes", answer_space), canonical_answer_text("No", answer_space)] if text]
    if answer_space.kind == "single_choice":
        allowed: List[str] = []
        for option in answer_space.options:
            raw_candidate = option.label or option.text
            rendered = canonical_answer_text(raw_candidate, answer_space)
            if rendered and rendered not in allowed:
                allowed.append(rendered)
        return allowed
    return []


def canonical_candidate_answers(
    answer_space: AnswerSpaceSpec,
    *,
    full_answer: Optional[str],
    support_answer: Optional[str],
    target_answer: Optional[str],
    agreement_answer: Optional[str],
) -> Dict[str, str]:
    return {
        "full": canonical_answer_text(full_answer, answer_space),
        "support": canonical_answer_text(support_answer, answer_space),
        "target": canonical_answer_text(target_answer, answer_space),
        "agreement": canonical_answer_text(agreement_answer, answer_space),
    }


def build_constrained_candidate_choices(
    *,
    answer_space: AnswerSpaceSpec,
    candidate_answers: Dict[str, str],
    baseline_answer: Optional[str],
) -> List[Tuple[str, str]]:
    allowed = set(canonical_allowed_texts(answer_space))
    baseline_text = canonical_answer_text(baseline_answer, answer_space)
    if not baseline_text:
        baseline_text = safe_text((candidate_answers or {}).get("full"))
    ordered: List[Tuple[str, str]] = []
    for key in ("support", "target", "agreement"):
        value = safe_text((candidate_answers or {}).get(key))
        if not value:
            continue
        if baseline_text and value == baseline_text:
            continue
        if allowed and value not in allowed:
            continue
        ordered.append((key, value))
    return ordered


def select_candidate_texts_for_sources(
    candidate_choices: Sequence[Tuple[str, str]],
    *,
    preferred_sources: Optional[Sequence[str]] = None,
    fallback_to_all: bool = True,
) -> Tuple[List[str], str, bool]:
    ordered_preferences = iter_nonempty(preferred_sources or [])
    by_source: Dict[str, List[str]] = {}
    for source, text in candidate_choices:
        normalized_source = safe_text(source)
        normalized_text = safe_text(text)
        if not normalized_source or not normalized_text:
            continue
        bucket = by_source.setdefault(normalized_source, [])
        if normalized_text not in bucket:
            bucket.append(normalized_text)
    for source in ordered_preferences:
        source_texts = list(by_source.get(source) or [])
        if source_texts:
            return source_texts, source, True
    if ordered_preferences and not fallback_to_all:
        return [], "", False
    return iter_nonempty(text for _source, text in candidate_choices), "", False


def selector_action_v2(
    *,
    reference_answer: Optional[str],
    full_answer: Optional[str],
    executable_answer: Optional[str],
    answer_space: AnswerSpaceSpec,
) -> str:
    normalized_reference = normalize_answer_for_space(reference_answer, answer_space)
    normalized_full = normalize_answer_for_space(full_answer, answer_space)
    normalized_executable = normalize_answer_for_space(executable_answer, answer_space)
    if normalized_reference and normalized_full and normalized_reference == normalized_full:
        return "keep"
    if normalized_executable and normalized_full and normalized_executable != normalized_full:
        return "retarget"
    if normalized_executable and not normalized_full:
        return "retarget"
    return "abstain"


def infer_decoder_target_source(
    *,
    oracle_reason: Optional[str],
    executable_answer: Optional[str],
    support_answer: Optional[str],
    target_answer: Optional[str],
    agreement_answer: Optional[str],
    answer_space: AnswerSpaceSpec,
) -> str:
    reason = safe_text(oracle_reason)
    if reason in _REASON_TO_SOURCE:
        return _REASON_TO_SOURCE[reason]
    executable_text = canonical_answer_text(executable_answer, answer_space)
    if not executable_text:
        return "none"
    if executable_text == canonical_answer_text(support_answer, answer_space):
        return "support"
    if executable_text == canonical_answer_text(target_answer, answer_space):
        return "target"
    if executable_text == canonical_answer_text(agreement_answer, answer_space):
        return "agreement"
    return "none"


def build_decoder_context_text(
    *,
    question_text: str,
    answer_space: AnswerSpaceSpec,
    candidate_answers: Dict[str, str],
) -> str:
    prompt = safe_text(format_question_for_answer_space(question_text, answer_space))
    allowed = canonical_allowed_texts(answer_space)
    lines = [
        "Policy decoding context.",
        f"Question: {prompt}",
        f"Answer space kind: {safe_text(answer_space.kind) or 'unknown'}",
    ]
    if allowed:
        lines.append("Allowed answers:")
        lines.extend(f"- {text}" for text in allowed)
    lines.append("Candidate answers:")
    for key in ("full", "support", "target", "agreement"):
        value = safe_text((candidate_answers or {}).get(key))
        if value:
            lines.append(f"- {key}: {value}")
    return "\n".join(lines).strip()


def build_constrained_candidate_texts(
    *,
    answer_space: AnswerSpaceSpec,
    candidate_answers: Dict[str, str],
    baseline_answer: Optional[str],
    preferred_sources: Optional[Sequence[str]] = None,
    fallback_to_all: bool = True,
) -> List[str]:
    candidate_choices = build_constrained_candidate_choices(
        answer_space=answer_space,
        candidate_answers=candidate_answers,
        baseline_answer=baseline_answer,
    )
    candidate_texts, _selected_source, _source_conditioning_applied = select_candidate_texts_for_sources(
        candidate_choices,
        preferred_sources=preferred_sources,
        fallback_to_all=fallback_to_all,
    )
    return candidate_texts


def tokenize_context_texts(
    tokenizer,
    texts: Sequence[str],
    *,
    max_length: int,
) -> Dict[str, torch.Tensor]:
    return tokenizer(
        list(texts),
        add_special_tokens=False,
        padding=True,
        truncation=True,
        max_length=max(8, int(max_length)),
        return_tensors="pt",
    )


def build_teacher_forcing_batch(
    tokenizer,
    texts: Sequence[str],
    *,
    max_length: int,
) -> Dict[str, torch.Tensor]:
    encoded = tokenizer(
        list(texts),
        add_special_tokens=False,
        padding=False,
        truncation=True,
        max_length=max(2, int(max_length) - 1),
    )
    pad_token_id = int(tokenizer.pad_token_id)
    eos_token_id = int(tokenizer.eos_token_id)
    start_token_id = eos_token_id
    input_rows: List[List[int]] = []
    target_rows: List[List[int]] = []
    max_seq_len = 1
    for token_ids in encoded.get("input_ids") or []:
        normalized_ids = [int(token_id) for token_id in token_ids]
        decoder_input = [start_token_id] + normalized_ids
        decoder_target = normalized_ids + [eos_token_id]
        max_seq_len = max(max_seq_len, len(decoder_input), len(decoder_target))
        input_rows.append(decoder_input)
        target_rows.append(decoder_target)
    if not input_rows:
        input_rows = [[start_token_id]]
        target_rows = [[eos_token_id]]
    input_ids = torch.full((len(input_rows), max_seq_len), pad_token_id, dtype=torch.long)
    target_ids = torch.full((len(target_rows), max_seq_len), -100, dtype=torch.long)
    attention_mask = torch.zeros((len(input_rows), max_seq_len), dtype=torch.long)
    for row_idx, (decoder_input, decoder_target) in enumerate(zip(input_rows, target_rows)):
        input_ids[row_idx, : len(decoder_input)] = torch.tensor(decoder_input, dtype=torch.long)
        target_ids[row_idx, : len(decoder_target)] = torch.tensor(decoder_target, dtype=torch.long)
        attention_mask[row_idx, : len(decoder_input)] = 1
    return {
        "decoder_input_ids": input_ids,
        "decoder_target_ids": target_ids,
        "decoder_attention_mask": attention_mask,
    }


class GeneralActionPolicyV2DecoderNet(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        vocab_size: int,
        *,
        decoder_dim: int = 128,
        pad_token_id: int = 0,
    ) -> None:
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
        self.selector_head = nn.Linear(rep_dim, len(GENERAL_V2_SELECTOR_VALUES))
        self.source_head = nn.Linear(rep_dim, len(GENERAL_V2_SOURCE_VALUES))
        self.token_embedding = nn.Embedding(vocab_size, decoder_dim, padding_idx=pad_token_id)
        self.context_projection = nn.Linear(rep_dim + decoder_dim, decoder_dim)
        self.decoder = nn.GRU(input_size=decoder_dim, hidden_size=decoder_dim, batch_first=True)
        self.lm_head = nn.Linear(decoder_dim, vocab_size)
        self.pad_token_id = int(pad_token_id)
        self.decoder_dim = int(decoder_dim)

    def encode_feature(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def _pool_context(self, context_input_ids: torch.Tensor, context_attention_mask: torch.Tensor) -> torch.Tensor:
        embedded = self.token_embedding(context_input_ids)
        mask = context_attention_mask.unsqueeze(-1).float()
        denom = mask.sum(dim=1).clamp_min(1.0)
        return (embedded * mask).sum(dim=1) / denom

    def decode_logits(
        self,
        rep: torch.Tensor,
        *,
        context_input_ids: torch.Tensor,
        context_attention_mask: torch.Tensor,
        decoder_input_ids: torch.Tensor,
    ) -> torch.Tensor:
        context_summary = self._pool_context(context_input_ids, context_attention_mask)
        init_hidden = torch.tanh(self.context_projection(torch.cat([rep, context_summary], dim=-1))).unsqueeze(0)
        decoder_emb = self.token_embedding(decoder_input_ids)
        decoder_out, _ = self.decoder(decoder_emb, init_hidden)
        return self.lm_head(decoder_out)

    def forward(
        self,
        x: torch.Tensor,
        *,
        context_input_ids: Optional[torch.Tensor] = None,
        context_attention_mask: Optional[torch.Tensor] = None,
        decoder_input_ids: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        rep = self.encode_feature(x)
        outputs: Dict[str, torch.Tensor] = {
            "selector_logits": self.selector_head(rep),
            "source_logits": self.source_head(rep),
        }
        if (
            context_input_ids is not None
            and context_attention_mask is not None
            and decoder_input_ids is not None
        ):
            outputs["decoder_logits"] = self.decode_logits(
                rep,
                context_input_ids=context_input_ids,
                context_attention_mask=context_attention_mask,
                decoder_input_ids=decoder_input_ids,
            )
        return outputs


def decoder_sequence_log_probs(
    model: GeneralActionPolicyV2DecoderNet,
    *,
    features: torch.Tensor,
    context_input_ids: torch.Tensor,
    context_attention_mask: torch.Tensor,
    decoder_input_ids: torch.Tensor,
    decoder_target_ids: torch.Tensor,
) -> torch.Tensor:
    outputs = model(
        features,
        context_input_ids=context_input_ids,
        context_attention_mask=context_attention_mask,
        decoder_input_ids=decoder_input_ids,
    )
    decoder_logits = outputs["decoder_logits"]
    log_probs = F.log_softmax(decoder_logits, dim=-1)
    safe_targets = decoder_target_ids.masked_fill(decoder_target_ids < 0, 0)
    gathered = torch.gather(log_probs, dim=-1, index=safe_targets.unsqueeze(-1)).squeeze(-1)
    mask = (decoder_target_ids >= 0).float()
    return (gathered * mask).sum(dim=-1)


def summarize_candidate_choices(rows: Sequence[Dict[str, Any]]) -> Dict[str, int]:
    counts = {key: 0 for key in ("support", "target", "agreement")}
    for row in rows:
        for key in counts:
            if safe_text(((row.get("candidate_answers_v2") or {}).get(key))):
                counts[key] += 1
    return counts


def iter_nonempty(values: Iterable[Optional[str]]) -> List[str]:
    ordered: List[str] = []
    for value in values:
        text = safe_text(value)
        if text and text not in ordered:
            ordered.append(text)
    return ordered


def v2_default_abstain_text() -> str:
    return safe_text(ABSTAIN_TEMPLATE)
