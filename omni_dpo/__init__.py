"""
Utilities for phase-1 OmniDPO style data construction, caching, and training.
"""

from .media import MediaResolver, ResolvedMedia
from .modeling import (
    compose_question_text,
    dpo_pair_loss,
    generate_response,
    prepare_supervised_inputs,
    resolve_model_input_device,
)
from .preference_pairs import (
    ABSTAIN_TEMPLATE,
    PAIR_WEIGHT_BY_FAMILY,
    attach_hallucination_reject_pairs,
    build_base_pairs,
    load_baseline_hallucination_cache,
    normalize_model_answer,
)

__all__ = [
    "ABSTAIN_TEMPLATE",
    "MediaResolver",
    "PAIR_WEIGHT_BY_FAMILY",
    "ResolvedMedia",
    "attach_hallucination_reject_pairs",
    "build_base_pairs",
    "compose_question_text",
    "dpo_pair_loss",
    "generate_response",
    "load_baseline_hallucination_cache",
    "normalize_model_answer",
    "prepare_supervised_inputs",
    "resolve_model_input_device",
]
