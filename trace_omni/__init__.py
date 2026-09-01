"""
TRACe-Omni
==========
Temporal-Responsibility Aware Claim Decoding for Omni models.

This package is intentionally standalone. It does not extend the legacy
suppression stack; instead, it provides a clean claim-level framework for:

1. parsing a question into responsibility-aware predicates,
2. building a temporal witness graph from multimodal evidence,
3. scoring atomic claims with support, dependency, and entanglement signals,
4. controlling commitment at realization time.
"""

from .config import TraceOmniConfig
from .pipeline import TraceOmniPipeline
from .types import (
    Claim,
    ClaimScore,
    QuerySpec,
    RealizedAnswer,
    TraceOmniResult,
    WitnessGraph,
)

__all__ = [
    'Claim',
    'ClaimScore',
    'QuerySpec',
    'RealizedAnswer',
    'TraceOmniConfig',
    'TraceOmniPipeline',
    'TraceOmniResult',
    'WitnessGraph',
]
