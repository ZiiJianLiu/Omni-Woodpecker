"""
Configuration for TRACe-Omni.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

from ..config import ModelPathConfig


@dataclass
class ProbeConfig:
    enabled: bool = True
    mode: str = 'heuristic'
    branch_probe_max_claims: int = 4
    branch_probe_max_new_tokens: int = 6
    branch_probe_use_masks: bool = True
    branch_probe_yes_weight: float = 1.0
    branch_probe_softmax_temperature: float = 0.35
    branch_probe_delta_scale: float = 0.25
    branch_probe_prompt_style: str = 'responsibility_clause'


@dataclass
class ScoringConfig:
    support_weight: float = 0.42
    dependency_weight: float = 0.23
    corroboration_weight: float = 0.10
    temporal_weight: float = 0.17
    entanglement_penalty: float = 0.20
    prior_penalty: float = 0.12
    contradiction_weight: float = 0.55
    strong_commitment_threshold: float = 0.62
    localize_threshold: float = 0.40
    hedge_threshold: float = 0.22
    contradiction_threshold: float = 0.58
    default_visual_support: float = 0.18
    default_audio_support: float = 0.18
    token_overlap_floor: float = 0.05
    cross_modal_temporal_floor: float = 0.25
    temporal_overlap_bonus: float = 0.20
    dominant_mismatch_penalty: float = 0.18


@dataclass
class RealizerConfig:
    fallback_to_draft: bool = True
    hedge_prefix: str = 'It appears that'
    uncertainty_suffix: str = 'but the exact cross-modal linkage is uncertain.'
    open_ended_max_claims: int = 8
    choice_return_labels_only: bool = False


@dataclass
class TraceOmniConfig:
    device: str = 'cuda:0'
    model_paths: ModelPathConfig = field(default_factory=ModelPathConfig)
    generation_max_new_tokens: int = 96
    probe: ProbeConfig = field(default_factory=ProbeConfig)
    scoring: ScoringConfig = field(default_factory=ScoringConfig)
    realizer: RealizerConfig = field(default_factory=RealizerConfig)

    @classmethod
    def from_pipeline_config(cls, pipeline_config: Optional[Any] = None) -> 'TraceOmniConfig':
        if pipeline_config is None:
            return cls()
        generation = getattr(pipeline_config, 'generation', None)
        return cls(
            device=str(getattr(pipeline_config, 'device', 'cuda:0')),
            model_paths=getattr(pipeline_config, 'model_paths', ModelPathConfig()),
            generation_max_new_tokens=int(getattr(generation, 'max_new_tokens', 96) or 96),
        )
