"""Conflict-aware attention masking that hides the competing modality when a conflict is detected."""

import torch
import logging
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class ConflictSignal:
    """Conflict signals between the audio and visual streams."""
    has_emotion_conflict: bool
    emotion_distance: float  # [0, 1]
    has_content_conflict: bool
    content_conflict_score: float  # [0, 1]

    @property
    def overall_conflict_score(self) -> float:
        emotion_weight = 0.6
        content_weight = 0.4
        return (
            emotion_weight * (self.emotion_distance if self.has_emotion_conflict else 0.0) +
            content_weight * (self.content_conflict_score if self.has_content_conflict else 0.0)
        )

    @property
    def should_suppress(self) -> bool:
        return self.overall_conflict_score >= 0.3


class AttentionMaskSuppressor:
    """Mask the attention to the competing modality when a conflict is detected."""

    def __init__(
        self,
        emotion_conflict_threshold: float = 0.35,
        content_conflict_threshold: float = 0.30,
        min_conflict_for_masking: float = 0.3,
    ):
        self.emotion_threshold = emotion_conflict_threshold
        self.content_threshold = content_conflict_threshold
        self.min_conflict = min_conflict_for_masking

    def detect_conflict(
        self,
        visual_emotion: str,
        audio_emotion: str,
        visual_emotion_conf: float,
        audio_emotion_conf: float,
        visual_objects: List[str],
        asr_text: str,
    ) -> ConflictSignal:
        """Detect emotion and content conflicts between the visual and audio streams."""
        emotion_distance = self._compute_emotion_distance(
            visual_emotion, audio_emotion,
            visual_emotion_conf, audio_emotion_conf
        )
        has_emotion_conflict = emotion_distance > self.emotion_threshold

        content_conflict_score = self._compute_content_conflict(
            visual_objects, asr_text
        )
        has_content_conflict = content_conflict_score > self.content_threshold

        return ConflictSignal(
            has_emotion_conflict=has_emotion_conflict,
            emotion_distance=emotion_distance,
            has_content_conflict=has_content_conflict,
            content_conflict_score=content_conflict_score,
        )

    def _compute_emotion_distance(
        self,
        visual_emotion: str,
        audio_emotion: str,
        visual_conf: float,
        audio_conf: float,
    ) -> float:
        emotion_polarity = {
            'happy': 1.0, 'excited': 0.8, 'neutral': 0.0,
            'sad': -0.8, 'angry': -0.6, 'fear': -0.7,
        }

        v_polarity = emotion_polarity.get(visual_emotion.lower(), 0.0)
        a_polarity = emotion_polarity.get(audio_emotion.lower(), 0.0)

        polarity_diff = abs(v_polarity - a_polarity)

        avg_conf = (visual_conf + audio_conf) / 2.0

        return polarity_diff * avg_conf

    def _compute_content_conflict(
        self,
        visual_objects: List[str],
        asr_text: str,
    ) -> float:
        if not asr_text or not visual_objects:
            return 0.0

        asr_lower = asr_text.lower()
        visual_lower = [obj.lower() for obj in visual_objects]

        common_objects = [
            'person', 'man', 'woman', 'child', 'car', 'dog', 'cat',
            'tree', 'building', 'phone', 'computer', 'table', 'chair'
        ]

        mentioned_objects = [obj for obj in common_objects if obj in asr_lower]

        if not mentioned_objects:
            return 0.0

        missing_count = sum(
            1 for obj in mentioned_objects
            if not any(obj in v_obj for v_obj in visual_lower)
        )

        conflict_score = missing_count / len(mentioned_objects)
        return conflict_score

    def create_masked_inputs(
        self,
        processor,
        video_path: str,
        question: str,
        audio_array: Optional[object],
        conflict_signal: ConflictSignal,
        query_modality: str,  # 'visual' or 'audio'
    ) -> Tuple[Dict, bool]:
        """Build model inputs, masking the competing modality when the conflict is strong."""
        if not conflict_signal.should_suppress:
            from ..models.qwen_omni import _DEFAULT_SYSTEM_PROMPT
            messages = [
                {
                    'role': 'system',
                    'content': [{'type': 'text', 'text': _DEFAULT_SYSTEM_PROMPT}],
                },
                {
                    'role': 'user',
                    'content': [
                        {'type': 'video', 'video': video_path},
                        {'type': 'text', 'text': question},
                    ],
                },
            ]
            text_prompt = processor.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=False
            )
            inputs = processor(
                text=[text_prompt],
                videos=[video_path],
                audios=[audio_array] if audio_array is not None else None,
                return_tensors='pt',
                padding=True,
            )
            return inputs, False

        logger.info(
            f"Applying attention masking: conflict_score={conflict_signal.overall_conflict_score:.3f}, "
            f"query_modality={query_modality}"
        )

        if query_modality == 'audio':
            mask_visual = True
            mask_audio = False
        else:  # visual
            mask_visual = False
            mask_audio = True

        from ..models.qwen_omni import _DEFAULT_SYSTEM_PROMPT
        messages = [
            {
                'role': 'system',
                'content': [{'type': 'text', 'text': _DEFAULT_SYSTEM_PROMPT}],
            },
            {
                'role': 'user',
                'content': [
                    {'type': 'video', 'video': video_path},
                    {'type': 'text', 'text': question},
                ],
            },
        ]
        text_prompt = processor.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False
        )

        inputs = processor(
            text=[text_prompt],
            videos=[video_path] if not mask_visual else None,
            audios=[audio_array] if (audio_array is not None and not mask_audio) else None,
            return_tensors='pt',
            padding=True,
        )

        return inputs, True


def infer_query_modality(question: str) -> str:
    q_lower = question.lower()
    if 'visible' in q_lower or 'see' in q_lower or 'in the video' in q_lower:
        return 'visual'
    elif 'sound' in q_lower or 'hear' in q_lower or 'in the audio' in q_lower:
        return 'audio'
    else:
        return 'visual'
