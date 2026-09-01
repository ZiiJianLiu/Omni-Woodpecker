"""
Configuration for Multimodal Hallucination Suppression
======================================================
所有超参数和模型路径的集中配置
"""
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


# ══════════════════════════════════════════════════════════════════════════════
# 情感映射（Valence-Arousal 空间坐标）
# ══════════════════════════════════════════════════════════════════════════════
EMOTION_VALENCE_AROUSAL = {
    'happy':     (0.8, 0.6),
    'sad':       (-0.6, -0.4),
    'angry':     (-0.5, 0.7),
    'fear':      (-0.6, 0.5),
    'surprise':  (0.2, 0.8),
    'disgust':   (-0.7, 0.3),
    'neutral':   (0.0, 0.0),
}

EMOTION_LABEL_MAPPING = {
    'angry': 'angry',
    'calm': 'neutral',
    'disgust': 'disgust',
    'fearful': 'fear',
    'happy': 'happy',
    'neutral': 'neutral',
    'sad': 'sad',
    'surprised': 'surprise',
    'anger': 'angry',
    'fear': 'fear',
    'happiness': 'happy',
    'sadness': 'sad',
}


# ══════════════════════════════════════════════════════════════════════════════
# 模型路径配置
# ══════════════════════════════════════════════════════════════════════════════
@dataclass
class ModelPathConfig:
    """预训练模型路径"""

    qwen_omni_path: str = 'Qwen/Qwen2.5-Omni-7B'
    visual_emotion_model: str = 'trpakov/vit-face-expression'
    audio_emotion_model: str = 'ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition'
    asr_model_size: str = 'large-v3'
    audio_event_model: str = 'MIT/ast-finetuned-audioset-10-10-0.4593'
    audio_grounding_model: str = 'laion/clap-htsat-unfused'
    clip_model: str = 'openai/clip-vit-large-patch14'
    grounding_model: str = 'IDEA-Research/grounding-dino-tiny'
    grounding_fallback_model: str = 'google/owlv2-base-patch16-ensemble'
    text_emotion_model: str = 'j-hartmann/emotion-english-distilroberta-base'


# ══════════════════════════════════════════════════════════════════════════════
# 冲突检测配置
# ══════════════════════════════════════════════════════════════════════════════
@dataclass
class ConflictDetectionConfig:
    """跨模态冲突检测参数"""

    emotion_distance_threshold: float = 0.8
    emotion_consistency_threshold: float = 0.5
    asr_visual_similarity_threshold: float = 0.5
    clip_similarity_threshold: float = 0.6
    speech_energy_threshold: float = 0.01
    music_spectral_flatness_threshold: float = 0.3
    visual_confidence_weight: float = 1.2
    audio_confidence_weight: float = 1.0
    text_confidence_weight: float = 0.8
    min_visual_emotion_conf_for_correction: float = 0.55
    min_audio_emotion_conf_for_correction: float = 0.35
    strong_emotion_distance_threshold: float = 1.1
    min_object_mentions_for_content_conflict: int = 1


# ══════════════════════════════════════════════════════════════════════════════
# 答案生成配置
# ══════════════════════════════════════════════════════════════════════════════
@dataclass
class GenerationConfig:
    """答案生成参数"""

    n_candidates: int = 3
    temperature: float = 0.0
    candidate_temperatures: list = None
    max_new_tokens: int = 256

    def __post_init__(self):
        if self.candidate_temperatures is None:
            self.candidate_temperatures = [0.0, 0.7, 1.0]


# ══════════════════════════════════════════════════════════════════════════════
# 验证配置
# ══════════════════════════════════════════════════════════════════════════════
@dataclass
class VerificationConfig:
    """答案验证参数"""

    visual_weight: float = 0.4
    audio_weight: float = 0.3
    emotion_weight: float = 0.3
    pass_threshold: float = 0.7
    clip_n_frames: int = 8
    neutralize_yes_no_emotion: bool = True


# ══════════════════════════════════════════════════════════════════════════════
# 修正配置
# ══════════════════════════════════════════════════════════════════════════════
@dataclass
class CorrectionConfig:
    """答案修正参数"""

    max_correction_attempts: int = 1
    enable_joint_conflict_resolver: bool = False
    enable_legacy_retry: bool = False

    # ══════════════════════════════════════════════════════════════════════════════
    # 方案一：Token级别抑制增强配置
    # ══════════════════════════════════════════════════════════════════════════════
    enable_enhanced_token_suppression: bool = True
    enable_entity_level_suppression: bool = True
    enable_emotion_token_suppression: bool = True
    entity_suppression_strength: float = 3.0
    emotion_token_bias: float = -2.0
    emotion_focus_only: bool = False
    enable_emotion_conflict_resolver: bool = False
    emotion_focus_min_emotion_distance: float = 1.1
    emotion_focus_branch_margin: float = 0.18
    emotion_focus_branch_delta: float = 0.08
    emotion_focus_use_emotion_constraint: bool = False
    joint_strong_emotion_distance: float = 1.1
    joint_affect_conflict_penalty: float = 0.18
    joint_affect_consistency_bonus: float = 0.10
    joint_decision_margin: float = 0.10
    entity_emotion_risk_weight: float = 0.18
    relation_emotion_risk_weight: float = 0.72
    open_ended_emotion_risk_weight: float = 0.38
    joint_entity_support_threshold: float = 0.74
    joint_entity_contradiction_threshold: float = 0.74
    joint_entity_soft_contradiction_threshold: float = 0.60
    joint_entity_negative_risk_threshold: float = 0.54
    joint_entity_counterpart_threshold: float = 0.72
    joint_entity_counterpart_bonus: float = 0.08
    joint_entity_visual_counterpart_floor: float = 0.56
    joint_entity_visual_counterpart_bonus: float = 0.14
    joint_entity_audio_visual_floor: float = 0.52
    joint_entity_audio_visual_bonus: float = 0.10
    joint_entity_positive_risk_ceiling: float = 0.62
    joint_entity_driver_conflict_penalty: float = 0.14
    joint_entity_visual_synergy_bonus: float = 0.12
    joint_relation_support_threshold: float = 0.80
    joint_relation_proxy_support_threshold: float = 0.74
    joint_relation_contradiction_threshold: float = 0.66
    joint_relation_soft_contradiction_threshold: float = 0.58
    joint_relation_negative_risk_threshold: float = 0.55
    joint_relation_proxy_margin: float = 0.16
    joint_branch_margin: float = 0.12
    enable_open_ended_conflict_rewrite: bool = True
    open_ended_claim_guided_rewrite: bool = True
    open_ended_trigger_emotion_distance: float = 0.85
    open_ended_rewrite_max_new_tokens: int = 96
    open_ended_claim_audit_max_new_tokens: int = 160
    open_ended_claim_max_units: int = 8
    open_ended_claim_min_audit_coverage: float = 0.6
    open_ended_max_visual_items: int = 6
    open_ended_max_audio_items: int = 6
    enable_generation_conflict_guidance: bool = True
    enable_modality_masking: bool = False
    enable_separate_inference: bool = False
    enable_emotion_constraint: bool = False
    enable_question_conditioned_evidence: bool = False
    enable_relation_consistency_evidence: bool = True
    question_evidence_support_threshold: float = 0.74
    question_evidence_contradiction_threshold: float = 0.72
    question_evidence_flip_guard: float = 0.14
    relation_evidence_support_threshold: float = 0.80
    relation_evidence_contradiction_threshold: float = 0.72
    relation_evidence_flip_guard: float = 0.14
    question_evidence_visual_positive_clip_threshold: float = 0.45
    question_evidence_visual_generic_human_flip_threshold: float = 0.34
    question_evidence_visual_specific_human_face_clip_threshold: float = 0.32
    question_evidence_visual_specific_human_clip_threshold: float = 0.46
    question_evidence_visual_specific_human_proxy_cap: float = 0.40
    question_evidence_visual_specific_human_typed_grounding_threshold: float = 0.36
    question_evidence_visual_specific_human_typed_peak_threshold: float = 0.50
    question_evidence_specific_human_shadow_support_threshold: float = 0.72
    question_evidence_specific_human_shadow_proxy_cap: float = 0.30
    question_evidence_specific_human_shadow_contradiction_base: float = 0.44
    question_evidence_specific_human_shadow_contradiction_gain: float = 0.34
    question_evidence_visual_nondecisive_support_cap: float = 0.36
    question_evidence_visual_nondecisive_proxy_cap: float = 0.30
    question_evidence_visual_nondecisive_risk: float = 0.18
    question_evidence_audio_nondecisive_support_cap: float = 0.44
    question_evidence_audio_nondecisive_proxy_cap: float = 0.36
    question_evidence_audio_nondecisive_risk: float = 0.12
    question_evidence_allow_visual_negative_flip: bool = False
    question_evidence_skip_rgca_when_parsed: bool = True
    question_evidence_skip_rgca_on_abstain: bool = False
    choice_evidence_support_threshold: float = 0.56
    choice_evidence_margin_threshold: float = 0.10
    choice_evidence_top1_gap_threshold: float = 0.06
    relation_evidence_allow_no_object_positive_flip: bool = True
    relation_evidence_no_object_support_score: float = 0.82
    relation_evidence_audio_prompt_similarity_threshold: float = 0.42
    relation_evidence_audio_prompt_no_object_threshold: float = 0.68
    relation_evidence_audio_prompt_support_score: float = 0.84
    relation_evidence_multi_cue_threshold: float = 0.45
    relation_evidence_multi_cue_bonus: float = 0.08
    question_evidence_visual_contextual_peak_threshold: float = 0.28
    question_evidence_visual_contextual_clip_threshold: float = 0.30
    question_evidence_visual_contextual_clip_peak_threshold: float = 0.42
    enable_conflict_arbitration: bool = False
    arbitration_accept_margin: float = 0.10
    arbitration_min_branch_margin: float = 0.12
    arbitration_reliability_bonus: float = 0.22
    arbitration_full_branch_bonus: float = 0.04
    arbitration_flip_guard: float = 0.15
    arbitration_use_video_branch: bool = True
    arbitration_fallback_to_cmcd: bool = False
    enable_dcod: bool = True
    dcod_use_text_branch: bool = True
    dcod_top_k: int = 48
    dcod_repetition_penalty: float = 1.12
    dcod_credit_floor: float = 0.22
    dcod_debt_weight: float = 1.00
    dcod_prior_weight: float = 0.85
    dcod_irrelevant_weight: float = 0.65
    dcod_overlap_weight: float = 0.18
    dcod_modality_floor: float = 0.18
    dcod_yes_no_margin: float = 0.02
    enable_css: bool = True
    css_base_strength: float = 0.55
    css_affect_conflict_weight: float = 0.80
    css_detail_conflict_weight: float = 0.55
    css_causal_conflict_weight: float = 0.45
    enable_lwa: bool = True
    lwa_collapse_threshold: float = 0.30
    lwa_min_agreement: float = 0.24
    lwa_agreement_weight: float = 0.42
    lwa_counterpart_penalty: float = 0.30
    lwa_low_witness_penalty: float = 0.26
    lwa_relevance_weight: float = 0.38
    enable_muse: bool = True
    muse_logit_weight: float = 1.35
    muse_counterfactual_weight: float = 0.95
    muse_shadow_weight: float = 1.10
    muse_contradiction_weight: float = 0.85
    muse_insufficiency_weight: float = 0.70
    muse_cross_modal_weight: float = 0.55
    muse_nondecisive_weight: float = 0.75
    enable_tabs: bool = True
    tabs_yes_penalty_weight: float = 0.85
    tabs_no_bonus_weight: float = 0.40
    tabs_alignment_tolerance: float = 0.18
    tabs_max_gap_weight: float = 0.90
    enable_contrastive_decode: bool = False
    mcd_alpha: float = 0.5
    mcd_beta: float = 0.5
    mcd_adaptive: bool = True
    mcd_repetition_penalty: float = 1.2
    mcd_top_k: int = 50
    enable_cmcd: bool = False
    cmcd_collapse_threshold: float = 0.3
    cmcd_use_video_branch: bool = True
    mask_threshold: float = 0.3
    separate_inference_threshold: float = 0.5


# ══════════════════════════════════════════════════════════════════════════════
# Token级别抑制配置（方案一）
# ══════════════════════════════════════════════════════════════════════════════
@dataclass
class TokenLevelSuppressionConfig:
    """增强版Token级别抑制器配置（方案一）"""

    enable_enhanced_suppression: bool = True
    enable_entity_level_suppression: bool = True
    entity_suppression_strength: float = 3.0
    enable_emotion_token_suppression: bool = True
    emotion_token_bias: float = -2.0
    soft_mask_threshold: float = 0.3
    hard_mask_threshold: float = 0.7


# ══════════════════════════════════════════════════════════════════════════════
# 动态投机式解码配置
# ══════════════════════════════════════════════════════════════════════════════
@dataclass
class SpeculativeDecodingConfig:
    """Dynamic Speculative Audio Verification (DSAV) 配置。"""

    enable: bool = False
    enable_yes_no: bool = False
    sensitive_words_path: str = 'configs/sensitive_words.txt'
    validator_device: str = 'cuda:2'
    max_rollback_steps: int = 10
    validation_window_tokens: int = 6
    audio_event_chunk_duration_s: float = 2.0
    audio_event_hop_duration_s: float = 1.0
    audio_event_threshold: float = 0.18
    max_validation_contexts: int = 3
    blocked_token_ttl: int = 3
    enable_hidden_yn_draft: bool = True
    yn_draft_max_new_tokens: int = 48


# ══════════════════════════════════════════════════════════════════════════════
# 运行时裁剪配置
# ══════════════════════════════════════════════════════════════════════════════
@dataclass
class RuntimeControlConfig:
    """运行时主流程裁剪配置。"""

    prepare_validator_evidence: bool = False
    enable_verifier: bool = False
    extract_answer_conditioned_features: bool = False
    enable_stage4_correction: bool = False


# ══════════════════════════════════════════════════════════════════════════════
# Pipeline 总配置
# ══════════════════════════════════════════════════════════════════════════════
@dataclass
class PipelineConfig:
    """Pipeline 总配置"""

    model_paths: Optional[ModelPathConfig] = None
    conflict_detection: Optional[ConflictDetectionConfig] = None
    generation: Optional[GenerationConfig] = None
    verification: Optional[VerificationConfig] = None
    correction: Optional[CorrectionConfig] = None
    speculative_decoding: Optional[SpeculativeDecodingConfig] = None
    runtime: Optional[RuntimeControlConfig] = None
    device: str = 'cuda:0'
    verbose: bool = True

    def __post_init__(self):
        if self.model_paths is None:
            self.model_paths = ModelPathConfig()
        if self.conflict_detection is None:
            self.conflict_detection = ConflictDetectionConfig()
        if self.generation is None:
            self.generation = GenerationConfig()
        if self.verification is None:
            self.verification = VerificationConfig()
        if self.correction is None:
            self.correction = CorrectionConfig()
        if self.speculative_decoding is None:
            self.speculative_decoding = SpeculativeDecodingConfig()
        if self.runtime is None:
            self.runtime = RuntimeControlConfig()
