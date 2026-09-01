"""
Claim responsibility scoring for TRACe-Omni.
"""

from __future__ import annotations

import math
import re
from statistics import mean
from typing import Dict, List, Optional, Tuple

from .config import TraceOmniConfig
from .types import Claim, ClaimScore, DependencyEvidence, QuerySpec, WitnessGraph, WitnessNode

_STOPWORDS = {
    'a', 'an', 'the', 'is', 'are', 'was', 'were', 'in', 'on', 'at', 'to', 'of', 'for',
    'with', 'and', 'or', 'that', 'this', 'these', 'those', 'there', 'here', 'what',
    'which', 'who', 'when', 'where', 'why', 'how', 'be', 'being', 'been', 'it',
    'video', 'audio', 'scene',
}
_ALIAS_MAP = {
    'infant': {'baby', 'child'},
    'baby': {'infant', 'child'},
    'man': {'person', 'male'},
    'woman': {'person', 'female'},
    'car': {'vehicle', 'engine'},
    'vehicle': {'car', 'truck', 'engine'},
    'plane': {'airplane', 'aircraft'},
    'airplane': {'plane', 'aircraft'},
    'barking': {'dog', 'bark'},
    'meowing': {'cat', 'meow'},
    'beeping': {'beep', 'device'},
    'speech': {'voice', 'talking', 'speaking'},
}


class ClaimResponsibilityScorer:
    def __init__(self, config: TraceOmniConfig):
        self.config = config

    @staticmethod
    def _clip01(value: float) -> float:
        return max(0.0, min(1.0, float(value)))

    @staticmethod
    def _normalize(text: str) -> str:
        text = re.sub(r"[^a-z0-9']+", ' ', (text or '').lower())
        return ' '.join(text.split())

    def _tokens(self, text: str) -> List[str]:
        tokens = [token for token in self._normalize(text).split() if token and token not in _STOPWORDS]
        expanded: List[str] = []
        for token in tokens:
            expanded.append(token)
            expanded.extend(sorted(_ALIAS_MAP.get(token, set())))
        return expanded

    def _token_overlap(self, claim: Claim, node: WitnessNode) -> float:
        claim_tokens = set(self._tokens(claim.text))
        node_tokens = set(self._tokens(node.label))
        if not claim_tokens or not node_tokens:
            return 0.0
        shared = claim_tokens & node_tokens
        if not shared:
            return 0.0
        precision = len(shared) / max(1, len(claim_tokens))
        recall = len(shared) / max(1, len(node_tokens))
        return self._clip01(0.5 * (precision + recall))

    def _node_kind_bonus(self, claim: Claim, node: WitnessNode) -> float:
        if claim.predicate == 'visibility' and node.kind in {'object', 'scene'}:
            return 0.18
        if claim.predicate == 'sound_source' and node.kind in {'event', 'audio_type'}:
            return 0.20
        if claim.predicate == 'speech_content' and node.kind == 'speech':
            return 0.28
        if claim.predicate == 'emotion' and node.kind == 'emotion':
            return 0.16
        if claim.predicate == 'temporal_order' and node.start is not None and node.end is not None:
            return 0.18
        return 0.0

    def _support_for_modality(self, claim: Claim, graph: WitnessGraph, modality: str) -> Tuple[float, List[str]]:
        best = 0.0
        matched: List[str] = []
        floor = (
            self.config.scoring.default_audio_support
            if modality == 'audio' else self.config.scoring.default_visual_support
        )
        for node in graph.nodes:
            if node.modality != modality:
                continue
            overlap = self._token_overlap(claim, node)
            if overlap <= 0.0:
                continue
            score = self._clip01(
                self.config.scoring.token_overlap_floor
                + 0.65 * overlap
                + 0.20 * float(node.confidence or 0.0)
                + self._node_kind_bonus(claim, node)
            )
            if score > best:
                best = score
            matched.append(node.node_id)
        if matched:
            return best, matched
        return floor if modality in claim.corroborative_modalities else 0.0, matched

    def _corroboration_score(
        self,
        claim: Claim,
        support_by_modality: Dict[str, float],
    ) -> float:
        if not claim.corroborative_modalities:
            return 0.0
        corroborative = [support_by_modality.get(modality, 0.0) for modality in claim.corroborative_modalities]
        if not corroborative:
            return 0.0
        return self._clip01(0.75 * max(corroborative))

    def _temporal_attestation(
        self,
        claim: Claim,
        graph: WitnessGraph,
        matched_witnesses: Dict[str, List[str]],
    ) -> float:
        if claim.predicate != 'temporal_order':
            return 0.0
        audio_nodes = {
            node.node_id: node for node in graph.nodes
            if node.modality == 'audio'
        }
        visual_nodes = {
            node.node_id: node for node in graph.nodes
            if node.modality == 'visual'
        }
        audio_matches = [audio_nodes[node_id] for node_id in matched_witnesses.get('audio', []) if node_id in audio_nodes]
        visual_matches = [visual_nodes[node_id] for node_id in matched_witnesses.get('visual', []) if node_id in visual_nodes]
        if not audio_matches or not visual_matches:
            return self.config.scoring.cross_modal_temporal_floor

        overlap_hits = 0
        comparable = 0
        for audio_node in audio_matches:
            for visual_node in visual_matches:
                if audio_node.start is None or audio_node.end is None:
                    continue
                if visual_node.start is None or visual_node.end is None:
                    continue
                comparable += 1
                if max(audio_node.start, visual_node.start) <= min(audio_node.end, visual_node.end):
                    overlap_hits += 1
        if comparable == 0:
            return self.config.scoring.cross_modal_temporal_floor
        ratio = overlap_hits / float(comparable)
        return self._clip01(
            self.config.scoring.cross_modal_temporal_floor
            + self.config.scoring.temporal_overlap_bonus * ratio
        )

    def _entanglement_risk(
        self,
        claim: Claim,
        support_by_modality: Dict[str, float],
        dependency_by_modality: Dict[str, float],
        temporal_attestation: float,
    ) -> float:
        if claim.primary_modality == 'audio':
            dominance_gap = support_by_modality.get('visual', 0.0) - support_by_modality.get('audio', 0.0)
        elif claim.primary_modality == 'visual':
            dominance_gap = support_by_modality.get('audio', 0.0) - support_by_modality.get('visual', 0.0)
        else:
            dominance_gap = abs(support_by_modality.get('audio', 0.0) - support_by_modality.get('visual', 0.0))

        risk = max(0.0, dominance_gap)
        if claim.predicate == 'temporal_order':
            risk += max(0.0, 0.45 - temporal_attestation)
        if claim.primary_modality == 'cross_modal':
            dep_gap = abs(dependency_by_modality.get('audio', 0.0) - dependency_by_modality.get('visual', 0.0))
            risk += 0.5 * dep_gap
        return self._clip01(risk)

    def _prior_pressure(self, claim: Claim, matched_witnesses: Dict[str, List[str]]) -> float:
        claim_tokens = [token for token in self._tokens(claim.text) if len(token) >= 3]
        if not claim_tokens:
            return 0.0
        matched_fraction = 1.0 if any(matched_witnesses.values()) else 0.0
        unseen = max(0.0, 1.0 - matched_fraction)
        specificity = min(1.0, len(claim_tokens) / 6.0)
        return self._clip01(0.55 * unseen + 0.45 * specificity)

    def score_claim(
        self,
        claim: Claim,
        question_spec: QuerySpec,
        witness_graph: WitnessGraph,
        dependency_evidence: Optional[DependencyEvidence] = None,
    ) -> ClaimScore:
        support_by_modality: Dict[str, float] = {}
        matched_witnesses: Dict[str, List[str]] = {}
        for modality in ('audio', 'visual'):
            support, matches = self._support_for_modality(claim, witness_graph, modality)
            support_by_modality[modality] = support
            matched_witnesses[modality] = matches

        if dependency_evidence is None:
            dependency_by_modality = {'audio': 0.0, 'visual': 0.0}
        else:
            dependency_by_modality = {
                'audio': self._clip01(dependency_evidence.audio_dependency),
                'visual': self._clip01(dependency_evidence.visual_dependency),
            }

        main_modalities = list(claim.required_modalities) or [question_spec.primary_modality]
        main_support = min(support_by_modality.get(modality, 0.0) for modality in main_modalities if modality in {'audio', 'visual'})
        support_dependency_alignment = mean(
            min(
                support_by_modality.get(modality, 0.0),
                dependency_by_modality.get(modality, support_by_modality.get(modality, 0.0)),
            )
            for modality in main_modalities
            if modality in {'audio', 'visual'}
        ) if main_modalities else 0.0

        corroboration_score = self._corroboration_score(claim, support_by_modality)
        temporal_attestation = self._temporal_attestation(claim, witness_graph, matched_witnesses)
        entanglement_risk = self._entanglement_risk(
            claim,
            support_by_modality,
            dependency_by_modality,
            temporal_attestation,
        )
        prior_pressure = self._prior_pressure(claim, matched_witnesses)

        cfg = self.config.scoring
        commitment = (
            cfg.support_weight * main_support
            + cfg.dependency_weight * support_dependency_alignment
            + cfg.corroboration_weight * corroboration_score
            + cfg.temporal_weight * temporal_attestation
            - cfg.entanglement_penalty * entanglement_risk
            - cfg.prior_penalty * prior_pressure
        )
        commitment = self._clip01(commitment)

        contradiction = self._clip01(
            cfg.contradiction_weight * entanglement_risk
            + 0.30 * prior_pressure
            + max(0.0, cfg.localize_threshold - main_support)
        )

        rationale: List[str] = [
            f'predicate={claim.predicate}',
            f'main_support={main_support:.3f}',
            f'corroboration={corroboration_score:.3f}',
            f'temporal={temporal_attestation:.3f}',
            f'entanglement={entanglement_risk:.3f}',
            f'prior={prior_pressure:.3f}',
        ]
        if dependency_evidence is not None:
            rationale.append(f'dependency_method={dependency_evidence.method}')

        if commitment >= cfg.strong_commitment_threshold and contradiction < cfg.contradiction_threshold:
            decision = 'assertive'
        elif commitment >= cfg.localize_threshold:
            decision = 'localized'
        elif commitment >= cfg.hedge_threshold:
            decision = 'hedged'
        else:
            decision = 'omit'

        return ClaimScore(
            claim_id=claim.claim_id,
            support_by_modality=support_by_modality,
            dependency_by_modality=dependency_by_modality,
            matched_witnesses=matched_witnesses,
            corroboration_score=corroboration_score,
            temporal_attestation=temporal_attestation,
            entanglement_risk=entanglement_risk,
            prior_pressure=prior_pressure,
            main_support=main_support,
            contradiction_score=contradiction,
            commitment_score=commitment,
            decision=decision,
            rationale=rationale,
            dependency_evidence=dependency_evidence,
        )
