"""
Modules Package
===============
核心模块：特征提取、冲突检测、验证、修正
"""
from .async_validator import AsyncValidator
from .conflict_arbitrator import ConflictArbitrator
from .conflict_detector import CrossModalConflictDetector
from .corrector import AnswerCorrector
from .modality_extractor import ModalityExtractor
from .question_conditioned_evidence import QuestionConditionedEvidenceScorer
from .verifier import AnswerVerifier

__all__ = [
    'ModalityExtractor',
    'CrossModalConflictDetector',
    'AnswerCorrector',
    'AnswerVerifier',
    'AsyncValidator',
    'ConflictArbitrator',
    'QuestionConditionedEvidenceScorer',
]
