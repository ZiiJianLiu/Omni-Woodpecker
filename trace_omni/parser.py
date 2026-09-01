"""
Question parsing and responsibility assignment for TRACe-Omni.
"""

from __future__ import annotations

import re
from typing import List, Optional

from .types import ChoiceOption, Claim, QuerySpec

_YN_PREFIXES = (
    'is ',
    'are ',
    'was ',
    'were ',
    'do ',
    'does ',
    'did ',
    'can ',
    'could ',
    'will ',
    'would ',
    'has ',
    'have ',
    'had ',
    'should ',
)
_YN_INSTRUCTION_RE = re.compile(
    r"\b(answer|respond)\s+with\s+(only\s+)?yes\s+or\s+no\b|\byes\s+or\s+no\b",
    re.IGNORECASE,
)
_OPTION_PATTERN_RE = re.compile(
    r'([A-H]|\d{1,2})[\.\):]\s*(.+?)(?=(?:\s+(?:[A-H]|\d{1,2})[\.\):]\s)|$)',
    re.IGNORECASE | re.DOTALL,
)
_MULTI_SELECT_PATTERNS = (
    r'\bselect all that apply\b',
    r'\bchoose all that apply\b',
    r'\bmultiple answers?\b',
    r'\bmulti[\s-]?select\b',
    r'\bwhich of the following are\b',
    r'\bwhich options are\b',
    r'\ball correct\b',
    r'\bone or more\b',
)
_TEMPORAL_PATTERNS = (
    r'\bwhen\b',
    r'\bbefore\b',
    r'\bafter\b',
    r'\bwhile\b',
    r'\bduring\b',
    r'\bfirst\b',
    r'\bthen\b',
    r'\blater\b',
    r'\bearlier\b',
    r'\btiming\b',
    r'\border\b',
    r'\bsequence\b',
)


class ResponsibilityParser:
    """Parse a question into a task-agnostic responsibility specification."""

    @staticmethod
    def _normalize(text: str) -> str:
        text = re.sub(r"[^a-z0-9']+", ' ', (text or '').lower())
        return ' '.join(text.split())

    @staticmethod
    def _strip_yn_instruction(question: str) -> str:
        return re.sub(
            r'\s*(answer|respond)\s+with\s+(only\s+)?yes\s+or\s+no\.?\s*$',
            '',
            question or '',
            flags=re.IGNORECASE,
        ).strip()

    @classmethod
    def _is_yes_no_question(cls, question: str, max_new_tokens: Optional[int] = None) -> bool:
        q = (question or '').strip().lower()
        if _YN_INSTRUCTION_RE.search(q):
            return True
        if max_new_tokens is not None and max_new_tokens <= 10:
            return True
        return q.startswith(_YN_PREFIXES)

    def _extract_options(self, question: str) -> List[ChoiceOption]:
        raw = re.sub(r'\s+', ' ', question or '').strip()
        options: List[ChoiceOption] = []
        for match in _OPTION_PATTERN_RE.finditer(raw):
            label = str(match.group(1)).strip().upper()
            text = re.sub(r'\s+', ' ', match.group(2) or '').strip(' ;,')
            if label and text:
                options.append(ChoiceOption(label=label, text=text))
        return options

    def _infer_predicate(self, normalized: str) -> str:
        if any(re.search(pattern, normalized) for pattern in _TEMPORAL_PATTERNS):
            return 'temporal_order'
        if any(token in normalized for token in ('emotion', 'mood', 'feeling', 'tone', 'sentiment')):
            return 'emotion'
        if any(token in normalized for token in ('same event', 'same context', 'match', 'consistent', 'align', 'correspond')):
            return 'cross_modal_consistency'
        if any(token in normalized for token in ('what is being said', 'what does', 'say', 'saying', 'spoken', 'transcript')):
            return 'speech_content'
        if any(token in normalized for token in ('sound', 'audio', 'hear', 'heard', 'audible', 'noise')):
            return 'sound_source'
        if any(token in normalized for token in ('visible', 'see', 'seen', 'shown', 'appear', 'in the video', 'in the scene')):
            return 'visibility'
        return 'attribute'

    def _infer_primary_modality(self, normalized: str, predicate: str) -> str:
        audio_hits = bool(re.search(r'\b(audio|sound|sounds|hear|heard|audible|speech|voice|voices)\b', normalized))
        visual_hits = bool(re.search(r'\b(video|visual|scene|frame|frames|visible|see|seen|shown)\b', normalized))
        if predicate in {'cross_modal_consistency', 'temporal_order'}:
            return 'cross_modal'
        if predicate == 'speech_content':
            return 'audio'
        if predicate == 'sound_source':
            return 'audio'
        if predicate == 'visibility':
            return 'visual'
        if predicate == 'emotion':
            if audio_hits and not visual_hits:
                return 'audio'
            if visual_hits and not audio_hits:
                return 'visual'
            return 'cross_modal'
        if audio_hits and not visual_hits:
            return 'audio'
        if visual_hits and not audio_hits:
            return 'visual'
        return 'cross_modal'

    @staticmethod
    def _required_modalities(predicate: str, primary_modality: str) -> tuple[str, ...]:
        if predicate == 'visibility':
            return ('visual',)
        if predicate in {'sound_source', 'speech_content'}:
            return ('audio',)
        if predicate in {'cross_modal_consistency', 'temporal_order'}:
            return ('audio', 'visual')
        if primary_modality == 'cross_modal':
            return ('audio', 'visual')
        return (primary_modality,)

    @staticmethod
    def _corroborative_modalities(predicate: str, primary_modality: str) -> tuple[str, ...]:
        if predicate == 'visibility':
            return ('audio',)
        if predicate in {'sound_source', 'speech_content'}:
            return ('visual',)
        if primary_modality == 'cross_modal':
            return tuple()
        return ('audio',) if primary_modality == 'visual' else ('visual',)

    def parse(self, question: str, max_new_tokens: Optional[int] = None) -> QuerySpec:
        normalized = self._normalize(self._strip_yn_instruction(question))
        options = self._extract_options(question)
        if self._is_yes_no_question(question, max_new_tokens):
            form = 'yes_no'
        elif len(options) >= 2:
            form = 'multi_select' if any(re.search(pat, normalized) for pat in _MULTI_SELECT_PATTERNS) else 'single_choice'
        else:
            form = 'open_ended'

        predicate = self._infer_predicate(normalized)
        primary_modality = self._infer_primary_modality(normalized, predicate)
        required = self._required_modalities(predicate, primary_modality)
        corroborative = self._corroborative_modalities(predicate, primary_modality)
        asks_for_generation = form == 'open_ended'
        return QuerySpec(
            raw_question=question,
            normalized_question=normalized,
            form=form,
            predicate=predicate,
            primary_modality=primary_modality,
            required_modalities=required,
            corroborative_modalities=corroborative,
            asks_for_generation=asks_for_generation,
            options=options,
            metadata={
                'has_temporal_cue': predicate == 'temporal_order',
            },
        )

    def build_binary_claim(self, question_spec: QuerySpec) -> Claim:
        statement = self._strip_yn_instruction(question_spec.raw_question)
        statement = statement.rstrip(' ?')
        statement = re.sub(r'^(is|are|was|were|do|does|did|can|could|will|would|has|have|had|should)\s+', '', statement, flags=re.IGNORECASE)
        statement = statement.strip()
        if not statement:
            statement = question_spec.normalized_question
        text = statement[0].upper() + statement[1:] if statement else question_spec.raw_question
        return Claim(
            claim_id='claim_0',
            text=text,
            normalized_text=self._normalize(text),
            predicate=question_spec.predicate,
            primary_modality=question_spec.primary_modality,
            required_modalities=question_spec.required_modalities,
            corroborative_modalities=question_spec.corroborative_modalities,
            source='question',
            metadata={'form': question_spec.form},
        )
