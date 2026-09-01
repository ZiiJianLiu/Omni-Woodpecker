"""
Core data structures for TRACe-Omni.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class ChoiceOption:
    label: str
    text: str


@dataclass
class QuerySpec:
    raw_question: str
    normalized_question: str
    form: str
    predicate: str
    primary_modality: str
    required_modalities: Tuple[str, ...]
    corroborative_modalities: Tuple[str, ...] = field(default_factory=tuple)
    asks_for_generation: bool = True
    options: List[ChoiceOption] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class Claim:
    claim_id: str
    text: str
    normalized_text: str
    predicate: str
    primary_modality: str
    required_modalities: Tuple[str, ...]
    corroborative_modalities: Tuple[str, ...] = field(default_factory=tuple)
    source: str = 'generated'
    polarity: str = 'positive'
    option_label: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class WitnessNode:
    node_id: str
    modality: str
    kind: str
    label: str
    normalized_label: str
    confidence: float
    start: Optional[float] = None
    end: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class WitnessEdge:
    source: str
    target: str
    relation: str
    score: float
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class WitnessGraph:
    nodes: List[WitnessNode] = field(default_factory=list)
    edges: List[WitnessEdge] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            'nodes': [node.to_dict() for node in self.nodes],
            'edges': [edge.to_dict() for edge in self.edges],
            'metadata': dict(self.metadata),
        }


@dataclass
class DependencyEvidence:
    full_support: float
    no_audio_support: float
    no_visual_support: float
    audio_dependency: float
    visual_dependency: float
    method: str
    branch_mean_logprobs: Dict[str, float] = field(default_factory=dict)
    branch_relative_support: Dict[str, float] = field(default_factory=dict)
    token_count: int = 0
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ClaimScore:
    claim_id: str
    support_by_modality: Dict[str, float]
    dependency_by_modality: Dict[str, float]
    matched_witnesses: Dict[str, List[str]]
    corroboration_score: float
    temporal_attestation: float
    entanglement_risk: float
    prior_pressure: float
    main_support: float
    contradiction_score: float
    commitment_score: float
    decision: str
    rationale: List[str] = field(default_factory=list)
    dependency_evidence: Optional[DependencyEvidence] = None

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        if self.dependency_evidence is not None:
            payload['dependency_evidence'] = self.dependency_evidence.to_dict()
        return payload


@dataclass
class RealizedAnswer:
    text: str
    strategy: str
    kept_claim_ids: List[str] = field(default_factory=list)
    hedged_claim_ids: List[str] = field(default_factory=list)
    omitted_claim_ids: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class TraceOmniResult:
    video_path: str
    question: str
    question_spec: QuerySpec
    draft_answer: str
    realized_answer: RealizedAnswer
    claims: List[Claim] = field(default_factory=list)
    claim_scores: List[ClaimScore] = field(default_factory=list)
    witness_graph: Optional[WitnessGraph] = None
    feature_metadata: Dict[str, Any] = field(default_factory=dict)
    runtime_metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            'video_path': self.video_path,
            'question': self.question,
            'question_spec': self.question_spec.to_dict(),
            'draft_answer': self.draft_answer,
            'realized_answer': self.realized_answer.to_dict(),
            'claims': [claim.to_dict() for claim in self.claims],
            'claim_scores': [score.to_dict() for score in self.claim_scores],
            'witness_graph': self.witness_graph.to_dict() if self.witness_graph is not None else None,
            'feature_metadata': dict(self.feature_metadata),
            'runtime_metadata': dict(self.runtime_metadata),
        }
