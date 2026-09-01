"""Typed containers for modality-side evidence used by OWP."""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class ModalityFeatures:
    """Optional modality annotations accepted by the Qwen compatibility API."""

    visual_emotion: Optional[str] = None
    visual_emotion_conf: float = 0.0
    visual_objects: List[str] = field(default_factory=list)
    visual_scene: Optional[str] = None
    visual_faces_count: int = 0
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
    text_emotion: Optional[str] = None
    text_emotion_conf: float = 0.0
    text_content: Optional[str] = None
