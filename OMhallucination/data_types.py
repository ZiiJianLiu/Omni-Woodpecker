"""
Data Types for Multimodal Hallucination Suppression
====================================================
Training-free 多模态幻觉抑制系统的数据类型定义
"""
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class ModalityFeatures:
    """单个模态的特征（Training-free 提取）"""

    # ── 视觉特征 ──────────────────────────────────────────────────
    visual_emotion: Optional[str] = None
    visual_emotion_conf: float = 0.0
    visual_objects: List[str] = field(default_factory=list)
    visual_scene: Optional[str] = None
    visual_faces_count: int = 0

    # ── 音频特征 ──────────────────────────────────────────────────
    audio_emotion: Optional[str] = None
    audio_emotion_conf: float = 0.0
    audio_type: Optional[str] = None
    asr_text: Optional[str] = None
    asr_segments: List[Dict[str, Any]] = field(default_factory=list)
    asr_timestamp_index: Dict[str, List[Dict[str, float]]] = field(default_factory=dict)
    audio_events: List[str] = field(default_factory=list)
    audio_event_scores: Dict[str, float] = field(default_factory=dict)
    audio_event_timeline: List[Dict[str, Any]] = field(default_factory=list)
    audio_energy: float = 0.0
    audio_has_speech: bool = False

    # ── 文本特征（模型生成的答案）──────────────────────────────────
    text_emotion: Optional[str] = None
    text_emotion_conf: float = 0.0
    text_content: Optional[str] = None


@dataclass
class ConflictReport:
    """跨模态冲突检测报告"""

    # ── 冲突类型 ──────────────────────────────────────────────────
    audio_video_emotion_conflict: bool = False
    audio_video_content_conflict: bool = False
    audio_text_conflict: bool = False

    # ── 一致性分数 ────────────────────────────────────────────────
    emotion_consistency_score: float = 0.5
    content_consistency_score: float = 0.5

    # ── 冲突详情 ──────────────────────────────────────────────────
    conflict_details: Dict[str, Any] = field(default_factory=dict)
    emotion_distance: float = 0.0

    # ── 模态可信度 ────────────────────────────────────────────────
    dominant_modality: Optional[str] = None
    unreliable_modalities: List[str] = field(default_factory=list)

    # ── 修正策略建议 ──────────────────────────────────────────────
    suggested_correction: str = (
        'none'
    )


@dataclass
class VerificationResult:
    """答案验证结果"""

    visual_consistency: float = 0.0
    audio_consistency: float = 0.0
    emotion_consistency: float = 0.0

    final_score: float = 0.0
    passed: bool = False

    failure_reasons: List[str] = field(default_factory=list)
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PipelineResult:
    """Pipeline 运行结果"""

    # ── 输入信息 ──────────────────────────────────────────────────
    video_id: str
    question: str

    # ── Baseline 结果 ─────────────────────────────────────────────
    baseline_answer: str
    baseline_features: ModalityFeatures

    # ── 冲突检测 ──────────────────────────────────────────────────
    conflict_report: ConflictReport

    # ── Pipeline 结果 ─────────────────────────────────────────────
    corrected_answer: str
    corrected_features: ModalityFeatures

    # ── 验证结果 ──────────────────────────────────────────────────
    verification: VerificationResult

    # ── 生成期元信息 ──────────────────────────────────────────────
    generation_metadata: Dict[str, Any] = field(default_factory=dict)

    # ── DSAV 结果 ────────────────────────────────────────────────
    speculative_answer: str = ''
    speculative_metadata: Dict[str, Any] = field(default_factory=dict)

    # ── 修正信息 ──────────────────────────────────────────────────
    correction_applied: bool = False
    correction_type: str = 'none'
    n_correction_attempts: int = 0

    # ── 性能指标 ──────────────────────────────────────────────────
    inference_time_s: float = 0.0
    conflict_detection_time_s: float = 0.0
    speculative_time_s: float = 0.0
    dsav_trigger_count: int = 0
    dsav_rollback_count: int = 0
    dsav_validation_latency_s: float = 0.0
