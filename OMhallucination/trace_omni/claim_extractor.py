"""
Draft-answer decomposition into atomic claims.
"""

from __future__ import annotations

import re
from typing import List

from .types import Claim, QuerySpec

_SPLIT_RE = re.compile(r'[.;!?]+')
_CONNECTOR_RE = re.compile(r'\b(?:and|but|while|then|because|since|although)\b', re.IGNORECASE)
_FILLER_PREFIX_RE = re.compile(
    r'^(?:there is|there are|it is|it appears that|it seems that|we can see that|we can hear that)\s+',
    re.IGNORECASE,
)
_STOP_CLAUSE_RE = re.compile(r'^(?:yes|no|maybe|unclear|unknown)\.?$', re.IGNORECASE)


class ClaimDecomposer:
    """Decompose generated answers into atomic claims."""

    @staticmethod
    def _normalize(text: str) -> str:
        text = re.sub(r"[^a-z0-9']+", ' ', (text or '').lower())
        return ' '.join(text.split())

    def _clean_clause(self, clause: str) -> str:
        clause = ' '.join((clause or '').split()).strip(' ,')
        clause = _FILLER_PREFIX_RE.sub('', clause)
        return clause.strip()

    def _infer_clause_predicate(self, clause: str, fallback_predicate: str) -> str:
        normalized = self._normalize(clause)
        if any(token in normalized for token in ('before', 'after', 'while', 'when', 'timing', 'simultaneous')):
            return 'temporal_order'
        if any(token in normalized for token in ('emotion', 'mood', 'feeling', 'tone')):
            return 'emotion'
        if any(token in normalized for token in ('heard', 'hear', 'audible', 'sound', 'sounds', 'barking', 'beeping', 'speech', 'voice')):
            return 'sound_source'
        if any(token in normalized for token in ('said', 'saying', 'spoken', 'transcript')):
            return 'speech_content'
        if any(token in normalized for token in ('visible', 'shown', 'seen', 'appears', 'raise', 'raises', 'raising', 'holding', 'standing', 'walking')):
            return 'visibility'
        return fallback_predicate

    def _infer_primary_modality(self, clause: str, predicate: str, fallback_primary: str) -> str:
        normalized = self._normalize(clause)
        audio_hits = bool(re.search(r'\b(audio|sound|sounds|hear|heard|audible|speech|voice|barking|beeping)\b', normalized))
        visual_hits = bool(re.search(r'\b(video|visual|scene|visible|shown|seen|raising|holding|standing|walking|person|object)\b', normalized))
        if predicate == 'visibility':
            return 'visual'
        if predicate in {'sound_source', 'speech_content'}:
            return 'audio'
        if predicate in {'temporal_order', 'cross_modal_consistency'}:
            return 'cross_modal'
        if audio_hits and not visual_hits:
            return 'audio'
        if visual_hits and not audio_hits:
            return 'visual'
        return fallback_primary

    @staticmethod
    def _required_modalities(predicate: str, primary_modality: str) -> tuple[str, ...]:
        if predicate == 'visibility':
            return ('visual',)
        if predicate in {'sound_source', 'speech_content'}:
            return ('audio',)
        if predicate in {'temporal_order', 'cross_modal_consistency'}:
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

    def _split_text(self, text: str) -> List[str]:
        clauses: List[str] = []
        for sentence in _SPLIT_RE.split(text or ''):
            sentence = sentence.strip()
            if not sentence:
                continue
            parts = _CONNECTOR_RE.split(sentence)
            for part in parts:
                part = self._clean_clause(part)
                if not part or _STOP_CLAUSE_RE.match(part):
                    continue
                clauses.append(part)
        return clauses

    def decompose(self, text: str, query_spec: QuerySpec, *, max_claims: int = 8) -> List[Claim]:
        claims: List[Claim] = []
        for index, clause in enumerate(self._split_text(text)[:max_claims]):
            normalized = self._normalize(clause)
            if not normalized:
                continue
            predicate = self._infer_clause_predicate(clause, query_spec.predicate)
            primary_modality = self._infer_primary_modality(
                clause,
                predicate,
                query_spec.primary_modality,
            )
            claims.append(
                Claim(
                    claim_id=f'claim_{index}',
                    text=clause,
                    normalized_text=normalized,
                    predicate=predicate,
                    primary_modality=primary_modality,
                    required_modalities=self._required_modalities(predicate, primary_modality),
                    corroborative_modalities=self._corroborative_modalities(predicate, primary_modality),
                    source='generated',
                    metadata={'form': query_spec.form},
                )
            )
        return claims

    def option_claims(self, query_spec: QuerySpec) -> List[Claim]:
        claims: List[Claim] = []
        for index, option in enumerate(query_spec.options):
            claims.append(
                Claim(
                    claim_id=f'claim_{index}',
                    text=option.text,
                    normalized_text=self._normalize(option.text),
                    predicate=query_spec.predicate,
                    primary_modality=query_spec.primary_modality,
                    required_modalities=query_spec.required_modalities,
                    corroborative_modalities=query_spec.corroborative_modalities,
                    source='option',
                    option_label=option.label,
                    metadata={'form': query_spec.form},
                )
            )
        return claims
