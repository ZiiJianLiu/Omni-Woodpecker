"""
Cross-Modal Conflict Detector
==============================
检测音频-视频-文本之间的跨模态冲突（Training-free）
"""
import logging
import re
from typing import Tuple

import numpy as np

from ..config import EMOTION_VALENCE_AROUSAL
from ..data_types import ConflictReport, ModalityFeatures

logger = logging.getLogger(__name__)

_OBJECT_LEXICON = {
    "person", "people", "man", "woman", "child", "baby", "car", "truck", "bus",
    "motorcycle", "bike", "bicycle", "boat", "train", "airplane", "helicopter",
    "dog", "cat", "bird", "horse", "cow", "sheep", "computer", "laptop", "phone",
    "television", "tv", "book", "chair", "table", "door", "window", "guitar",
    "piano", "drum", "gun", "fireworks", "engine", "siren",
}


class CrossModalConflictDetector:
    """跨模态冲突检测器（Training-free）"""

    def __init__(self, config):
        self.config = config

    def detect(
        self,
        features: ModalityFeatures,
    ) -> ConflictReport:
        """检测跨模态冲突"""
        report = ConflictReport()
        report.conflict_details = {}
        report.emotion_consistency_score = 0.5
        report.content_consistency_score = 0.5

        if features.visual_emotion and features.audio_emotion:
            emotion_distance = self._compute_emotion_distance(
                features.visual_emotion,
                features.audio_emotion,
            )
            report.emotion_distance = emotion_distance
            if emotion_distance > self.config.emotion_distance_threshold:
                report.audio_video_emotion_conflict = True
                report.conflict_details['emotion_conflict'] = {
                    'visual': features.visual_emotion,
                    'audio': features.audio_emotion,
                    'distance': emotion_distance,
                }

        if features.asr_text and features.visual_objects:
            content_conflict = self._check_asr_visual_conflict(
                features.asr_text,
                features.visual_objects,
            )
            if content_conflict:
                report.audio_video_content_conflict = True
                report.conflict_details['content_conflict'] = content_conflict

        if features.asr_text and features.text_content:
            similarity = self._compute_text_similarity(
                features.asr_text,
                features.text_content,
            )
            if similarity < self.config.asr_visual_similarity_threshold:
                report.audio_text_conflict = True
                report.conflict_details['asr_answer_similarity'] = similarity

        emotions = [
            features.visual_emotion,
            features.audio_emotion,
            features.text_emotion,
        ]
        emotions = [e for e in emotions if e is not None]
        if len(emotions) >= 2:
            from collections import Counter
            emotion_counts = Counter(emotions)
            most_common_count = emotion_counts.most_common(1)[0][1]
            report.emotion_consistency_score = most_common_count / len(emotions)

        if features.visual_objects and features.asr_text:
            obj_score = min(len(features.visual_objects) / 5.0, 1.0)
            asr_score = min(len(features.asr_text.split()) / 20.0, 1.0)
            report.content_consistency_score = (obj_score + asr_score) / 2

        report.dominant_modality = self._determine_dominant_modality(features)
        report.unreliable_modalities = self._identify_unreliable_modalities(
            features, report,
        )
        report.suggested_correction = self._suggest_correction(report, features)
        return report

    def _compute_emotion_distance(self, emo1: str, emo2: str) -> float:
        v1, a1 = EMOTION_VALENCE_AROUSAL.get(emo1, (0, 0))
        v2, a2 = EMOTION_VALENCE_AROUSAL.get(emo2, (0, 0))
        distance = np.sqrt((v1 - v2) ** 2 + (a1 - a2) ** 2)
        return float(distance)

    def _check_asr_visual_conflict(
        self,
        asr_text: str,
        visual_objects: list,
    ) -> dict:
        """只在 ASR 明确提到对象且视觉侧缺失时，才判定内容冲突。"""
        asr_lower = asr_text.lower()
        visual_set = {obj.lower() for obj in visual_objects}
        min_mentions = max(
            1,
            int(getattr(self.config, 'min_object_mentions_for_content_conflict', 1)),
        )

        mentioned_in_asr = []
        for obj in sorted(_OBJECT_LEXICON, key=len, reverse=True):
            pattern = r'\b' + re.escape(obj) + r'\b'
            if re.search(pattern, asr_lower):
                mentioned_in_asr.append(obj)

        if len(mentioned_in_asr) < min_mentions:
            return {}

        missing = [obj for obj in mentioned_in_asr if obj not in visual_set]
        overlap = [obj for obj in mentioned_in_asr if obj in visual_set]

        if not missing:
            return {}
        if overlap and len(missing) < len(mentioned_in_asr):
            return {}

        return {
            'asr_mentions_not_visible': missing,
            'asr_mentions': mentioned_in_asr,
            'visual_objects': sorted(visual_set),
        }

    def _compute_text_similarity(self, text1: str, text2: str) -> float:
        try:
            from sentence_transformers import SentenceTransformer, util

            model = SentenceTransformer('all-MiniLM-L6-v2')
            emb1 = model.encode(text1, convert_to_tensor=True)
            emb2 = model.encode(text2, convert_to_tensor=True)
            similarity = util.cos_sim(emb1, emb2).item()
            return float(similarity)
        except Exception as e:
            logger.warning(f"文本相似度计算失败，使用词重叠: {e}")
            words1 = set(text1.lower().split())
            words2 = set(text2.lower().split())
            if not words1 or not words2:
                return 0.0
            overlap = len(words1 & words2)
            union = len(words1 | words2)
            return overlap / union if union > 0 else 0.0

    def _determine_dominant_modality(self, features: ModalityFeatures) -> str:
        scores = {}
        if features.visual_emotion:
            scores['visual'] = (
                features.visual_emotion_conf * self.config.visual_confidence_weight
            )
        if features.audio_emotion:
            scores['audio'] = (
                features.audio_emotion_conf * self.config.audio_confidence_weight
            )
        if features.text_emotion:
            scores['text'] = (
                features.text_emotion_conf * self.config.text_confidence_weight
            )
        if not scores:
            return 'visual'
        return max(scores, key=scores.get)

    def _identify_unreliable_modalities(
        self,
        features: ModalityFeatures,
        report: ConflictReport,
    ) -> list:
        unreliable = []
        if report.audio_video_emotion_conflict:
            if features.visual_emotion_conf > features.audio_emotion_conf:
                unreliable.append('audio')
            else:
                unreliable.append('visual')
        if features.audio_energy < self.config.speech_energy_threshold:
            if 'audio' not in unreliable:
                unreliable.append('audio')
        return unreliable

    def _has_reliable_audio_emotion(self, features: ModalityFeatures) -> bool:
        audio_type = (features.audio_type or '').lower()
        if audio_type in {'', 'silence'}:
            return False
        min_audio_conf = getattr(
            self.config,
            'min_audio_emotion_conf_for_correction',
            0.35,
        )
        return features.audio_emotion_conf >= min_audio_conf

    def _has_reliable_visual_emotion(self, features: ModalityFeatures) -> bool:
        min_visual_conf = getattr(
            self.config,
            'min_visual_emotion_conf_for_correction',
            0.55,
        )
        return features.visual_emotion_conf >= min_visual_conf

    def _suggest_correction(
        self,
        report: ConflictReport,
        features: ModalityFeatures,
    ) -> str:
        """输出统一 joint resolver 的建议入口，而不是 legacy 修正类型。"""
        strong_distance = getattr(
            self.config,
            'strong_emotion_distance_threshold',
            1.1,
        )
        reliable_visual_emotion = self._has_reliable_visual_emotion(features)
        reliable_audio_emotion = self._has_reliable_audio_emotion(features)

        if report.audio_video_emotion_conflict:
            if (
                report.emotion_distance >= strong_distance
                and reliable_visual_emotion
                and reliable_audio_emotion
            ):
                return 'joint_conflict'

        if report.audio_video_content_conflict:
            details = report.conflict_details.get('content_conflict', {})
            if details.get('asr_mentions_not_visible'):
                return 'joint_conflict'

        if report.audio_text_conflict:
            similarity = report.conflict_details.get('asr_answer_similarity', 1.0)
            if similarity < 0.2:
                return 'joint_conflict'

        if (
            report.emotion_distance >= strong_distance
            and reliable_visual_emotion
            and reliable_audio_emotion
        ):
            return 'joint_conflict'

        return 'none'
