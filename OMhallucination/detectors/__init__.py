"""
Detectors Package
=================
Training-free 检测器模块
"""
from .audio_emotion_detector import AudioEmotionDetector
from .audio_event_detector import AudioEventDetector
from .audio_text_grounding_detector import AudioTextGroundingDetector
from .asr_detector import ASRDetector
from .object_detector import ObjectDetector
from .visual_emotion_detector import VisualEmotionDetector

__all__ = [
    'VisualEmotionDetector',
    'AudioEmotionDetector',
    'AudioEventDetector',
    'AudioTextGroundingDetector',
    'ASRDetector',
    'ObjectDetector',
]
