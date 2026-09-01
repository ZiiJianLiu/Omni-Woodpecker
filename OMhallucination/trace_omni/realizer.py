"""
Commitment-controlled realization.
"""

from __future__ import annotations

import re
from typing import Dict, List

from .config import TraceOmniConfig
from .types import Claim, ClaimScore, QuerySpec, RealizedAnswer


class CommitmentControlledRealizer:
    def __init__(self, config: TraceOmniConfig):
        self.config = config

    @staticmethod
    def _normalize_yes_no(text: str) -> str:
        normalized = (text or '').strip().lower()
        if normalized.startswith('yes'):
            return 'Yes'
        if normalized.startswith('no'):
            return 'No'
        return ''

    @staticmethod
    def _clean_sentence(text: str) -> str:
        text = ' '.join((text or '').split()).strip()
        if not text:
            return ''
        if text[-1] not in '.!?':
            text += '.'
        return text[0].upper() + text[1:]

    def _fallback_text(self, draft_answer: str, query_spec: QuerySpec) -> str:
        if query_spec.form == 'yes_no':
            normalized = self._normalize_yes_no(draft_answer)
            return f'{normalized}.' if normalized else 'Uncertain.'
        draft_answer = ' '.join((draft_answer or '').split()).strip()
        return draft_answer or 'The available evidence is insufficient for a confident answer.'

    def realize(
        self,
        query_spec: QuerySpec,
        draft_answer: str,
        claims: List[Claim],
        scores: List[ClaimScore],
    ) -> RealizedAnswer:
        score_by_id: Dict[str, ClaimScore] = {score.claim_id: score for score in scores}
        ordered_pairs = [(claim, score_by_id.get(claim.claim_id)) for claim in claims if claim.claim_id in score_by_id]

        if query_spec.form == 'yes_no':
            if not ordered_pairs:
                return RealizedAnswer(text=self._fallback_text(draft_answer, query_spec), strategy='fallback')
            claim, score = ordered_pairs[0]
            if score.commitment_score >= self.config.scoring.strong_commitment_threshold and score.main_support >= self.config.scoring.localize_threshold:
                return RealizedAnswer(
                    text='Yes.',
                    strategy='assertive_binary',
                    kept_claim_ids=[claim.claim_id],
                )
            if score.contradiction_score >= self.config.scoring.contradiction_threshold and score.main_support < self.config.scoring.hedge_threshold:
                return RealizedAnswer(
                    text='No.',
                    strategy='negative_binary',
                    omitted_claim_ids=[claim.claim_id],
                    metadata={'reason': 'high_contradiction_or_low_support'},
                )
            return RealizedAnswer(
                text=self._fallback_text(draft_answer, query_spec),
                strategy='fallback_binary',
                metadata={'reason': 'insufficient_commitment'},
            )

        if query_spec.form == 'single_choice':
            if not ordered_pairs:
                return RealizedAnswer(text=self._fallback_text(draft_answer, query_spec), strategy='fallback_choice')
            ranked = sorted(
                ordered_pairs,
                key=lambda item: (
                    float(item[1].commitment_score if item[1] is not None else 0.0),
                    float(item[1].main_support if item[1] is not None else 0.0),
                    -float(item[1].contradiction_score if item[1] is not None else 0.0),
                ),
                reverse=True,
            )
            claim, score = ranked[0]
            if claim.option_label and not self.config.realizer.choice_return_labels_only:
                text = f'{claim.option_label}. {claim.text}'
            else:
                text = claim.option_label or claim.text
            return RealizedAnswer(
                text=text,
                strategy='top_commitment_choice',
                kept_claim_ids=[claim.claim_id],
                metadata={'commitment_score': score.commitment_score},
            )

        if query_spec.form == 'multi_select':
            selected: List[Claim] = []
            omitted: List[str] = []
            for claim, score in ordered_pairs:
                if score.decision in {'assertive', 'localized'}:
                    selected.append(claim)
                else:
                    omitted.append(claim.claim_id)
            if not selected and ordered_pairs:
                selected = [max(ordered_pairs, key=lambda item: item[1].commitment_score)[0]]
            pieces = []
            for claim in selected:
                if claim.option_label and not self.config.realizer.choice_return_labels_only:
                    pieces.append(f'{claim.option_label}. {claim.text}')
                else:
                    pieces.append(claim.option_label or claim.text)
            return RealizedAnswer(
                text=', '.join(pieces),
                strategy='supported_subset_choice',
                kept_claim_ids=[claim.claim_id for claim in selected],
                omitted_claim_ids=omitted,
            )

        kept: List[str] = []
        hedged: List[str] = []
        omitted: List[str] = []
        sentences: List[str] = []
        for claim, score in ordered_pairs[: self.config.realizer.open_ended_max_claims]:
            if score.decision == 'assertive':
                sentences.append(self._clean_sentence(claim.text))
                kept.append(claim.claim_id)
            elif score.decision == 'localized':
                if claim.predicate == 'temporal_order':
                    sentences.append(
                        self._clean_sentence(
                            f'{claim.text}, {self.config.realizer.uncertainty_suffix}'
                        )
                    )
                else:
                    sentences.append(self._clean_sentence(claim.text))
                kept.append(claim.claim_id)
            elif score.decision == 'hedged':
                sentences.append(
                    self._clean_sentence(f'{self.config.realizer.hedge_prefix} {claim.text}')
                )
                hedged.append(claim.claim_id)
            else:
                omitted.append(claim.claim_id)

        if not sentences:
            return RealizedAnswer(
                text=self._fallback_text(draft_answer, query_spec),
                strategy='fallback_open_ended',
                omitted_claim_ids=omitted,
            )

        return RealizedAnswer(
            text=' '.join(sentences),
            strategy='commitment_controlled_generation',
            kept_claim_ids=kept,
            hedged_claim_ids=hedged,
            omitted_claim_ids=omitted,
        )
