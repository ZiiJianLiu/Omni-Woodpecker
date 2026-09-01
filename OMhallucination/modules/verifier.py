"""
Answer Verifier
================
验证修正后的答案与各模态的一致性（Training-free）

验证维度：
1. 视觉一致性 — CLIP 视频帧 vs 答案文本相似度
2. 音频一致性 — ASR 文本 vs 答案文本语义相似度
3. 情感一致性 — 答案情感 vs 主导模态情感
"""
import logging
from typing import List, Optional

import numpy as np

from ..data_types import ModalityFeatures, ConflictReport, VerificationResult

logger = logging.getLogger(__name__)


class AnswerVerifier:
    """答案验证器"""

    def __init__(self, config, object_detector=None, device: str = "cuda:0"):
        """
        Parameters
        ----------
        config : VerificationConfig
        object_detector : ObjectDetector | None
            用于计算 CLIP 视频-文本相似度（可选，复用 extractor 中的实例）
        device : str
            SentenceTransformer 等小模型的设备
        """
        self.config = config
        self._clip = object_detector
        self._device = device
        self._sentence_model = None

    @staticmethod
    def _extract_yes_no(text: str) -> str:
        normalized = (text or '').strip().lower()
        if normalized.startswith('yes'):
            return 'Yes'
        if normalized.startswith('no'):
            return 'No'
        return 'Unknown'

    def verify(
        self,
        answer: str,
        features: ModalityFeatures,
        conflict_report: ConflictReport,
        frames: Optional[List[np.ndarray]] = None,
    ) -> VerificationResult:
        """验证答案与各模态的一致性

        Parameters
        ----------
        answer : str
            待验证的答案
        features : ModalityFeatures
            多模态特征
        conflict_report : ConflictReport
            冲突检测报告
        frames : list[np.ndarray] | None
            视频帧（用于 CLIP 相似度）

        Returns
        -------
        VerificationResult
        """
        result = VerificationResult()
        result.details = {}
        failure_reasons = []

        # 1. 视觉一致性（CLIP 视频-文本相似度）
        if self._clip and frames:
            try:
                vis_sim = self._clip.compute_video_text_similarity(frames, answer)
                result.visual_consistency = vis_sim
                result.details["clip_similarity"] = vis_sim
            except Exception as e:
                logger.debug("CLIP 相似度计算失败: %s", e)
                result.visual_consistency = 0.5  # 中性
        else:
            result.visual_consistency = 0.5

        # 2. 音频一致性（ASR vs 答案语义相似度）
        if features.asr_text and answer:
            try:
                audio_sim = self._compute_semantic_similarity(
                    features.asr_text, answer,
                )
                result.audio_consistency = audio_sim
                result.details["asr_answer_similarity"] = audio_sim
            except Exception as e:
                logger.debug("语义相似度计算失败: %s", e)
                result.audio_consistency = 0.5
        else:
            result.audio_consistency = 0.5

        # 3. 情感一致性
        dominant_emotion = conflict_report.dominant_modality
        if dominant_emotion == "visual":
            ref_emotion = features.visual_emotion
        elif dominant_emotion == "audio":
            ref_emotion = features.audio_emotion
        else:
            ref_emotion = features.visual_emotion or features.audio_emotion

        answer_label = self._extract_yes_no(answer)
        answer_emotion = features.text_emotion  # 已在 extractor 中提取
        if getattr(self.config, 'neutralize_yes_no_emotion', True) and answer_label in {'Yes', 'No'}:
            result.emotion_consistency = 0.5
            result.details["answer_emotion_skipped"] = True
        elif ref_emotion and answer_emotion:
            result.emotion_consistency = 1.0 if ref_emotion == answer_emotion else 0.0
            result.details["ref_emotion"] = ref_emotion
            result.details["answer_emotion"] = answer_emotion
        else:
            result.emotion_consistency = 0.5

        # 4. 综合分数
        result.final_score = (
            self.config.visual_weight * result.visual_consistency
            + self.config.audio_weight * result.audio_consistency
            + self.config.emotion_weight * result.emotion_consistency
        )

        # 5. 判断是否通过
        result.passed = result.final_score >= self.config.pass_threshold

        if result.visual_consistency < 0.3:
            failure_reasons.append("low_visual_consistency")
        if result.audio_consistency < 0.3:
            failure_reasons.append("low_audio_consistency")
        if result.emotion_consistency < 0.5:
            failure_reasons.append("emotion_mismatch")

        result.failure_reasons = failure_reasons

        logger.debug(
            "验证结果: vis=%.2f audio=%.2f emo=%.2f final=%.2f passed=%s",
            result.visual_consistency,
            result.audio_consistency,
            result.emotion_consistency,
            result.final_score,
            result.passed,
        )

        return result

    def _compute_semantic_similarity(self, text1: str, text2: str) -> float:
        """计算语义相似度"""
        try:
            if self._sentence_model is None:
                from sentence_transformers import SentenceTransformer
                self._sentence_model = SentenceTransformer("all-MiniLM-L6-v2", device=self._device)

            embs = self._sentence_model.encode([text1, text2])
            cos = float(np.dot(embs[0], embs[1]) / (
                np.linalg.norm(embs[0]) * np.linalg.norm(embs[1]) + 1e-8
            ))
            return max(0.0, min(1.0, cos))

        except Exception:
            # fallback: 词重叠
            w1 = set(text1.lower().split())
            w2 = set(text2.lower().split())
            union = len(w1 | w2)
            return len(w1 & w2) / union if union else 0.0
