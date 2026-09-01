"""
OMhallucination: Multimodal Hallucination Suppression System
=============================================================
Training-free 多模态幻觉抑制系统

主要组件：
- MultimodalHallucinationSuppressionPipeline: 主 Pipeline
- PipelineConfig: 配置类
- ModalityFeatures, ConflictReport, VerificationResult: 数据类型
"""

__version__ = '0.1.0'

from .config import (
    ConflictDetectionConfig,
    CorrectionConfig,
    GenerationConfig,
    ModelPathConfig,
    PipelineConfig,
    SpeculativeDecodingConfig,
    VerificationConfig,
)
from .data_types import (
    ConflictReport,
    ModalityFeatures,
    PipelineResult,
    VerificationResult,
)
from .pipeline import MultimodalHallucinationSuppressionPipeline
from .qwen_omni_adapter import QwenOmniAdapter

__all__ = [
    'ModalityFeatures',
    'ConflictReport',
    'VerificationResult',
    'PipelineResult',
    'PipelineConfig',
    'ModelPathConfig',
    'ConflictDetectionConfig',
    'GenerationConfig',
    'VerificationConfig',
    'CorrectionConfig',
    'SpeculativeDecodingConfig',
    'MultimodalHallucinationSuppressionPipeline',
    'QwenOmniAdapter',
]
