"""
Temporal witness graph construction.
"""

from __future__ import annotations

import re
from typing import List, Optional

from ..data_types import ModalityFeatures
from .types import WitnessEdge, WitnessGraph, WitnessNode


class TemporalWitnessGraphBuilder:
    """Build a witness graph from extracted multimodal features."""

    @staticmethod
    def _normalize(text: str) -> str:
        text = re.sub(r"[^a-z0-9']+", ' ', (text or '').lower())
        return ' '.join(text.split())

    @staticmethod
    def _overlap(start_a: Optional[float], end_a: Optional[float], start_b: Optional[float], end_b: Optional[float]) -> bool:
        if start_a is None or end_a is None or start_b is None or end_b is None:
            return False
        return max(start_a, start_b) <= min(end_a, end_b)

    def build(self, features: ModalityFeatures) -> WitnessGraph:
        nodes: List[WitnessNode] = []
        edges: List[WitnessEdge] = []

        for index, label in enumerate(features.visual_objects or []):
            normalized = self._normalize(str(label))
            if not normalized:
                continue
            nodes.append(
                WitnessNode(
                    node_id=f'visual_object_{index}',
                    modality='visual',
                    kind='object',
                    label=str(label),
                    normalized_label=normalized,
                    confidence=0.72,
                )
            )

        if features.visual_scene:
            nodes.append(
                WitnessNode(
                    node_id='visual_scene_0',
                    modality='visual',
                    kind='scene',
                    label=str(features.visual_scene),
                    normalized_label=self._normalize(features.visual_scene),
                    confidence=0.62,
                )
            )

        if features.visual_emotion:
            nodes.append(
                WitnessNode(
                    node_id='visual_emotion_0',
                    modality='visual',
                    kind='emotion',
                    label=str(features.visual_emotion),
                    normalized_label=self._normalize(features.visual_emotion),
                    confidence=float(features.visual_emotion_conf or 0.55),
                )
            )

        timeline = list(features.audio_event_timeline or [])
        if timeline:
            for index, item in enumerate(timeline):
                label = str(item.get('label') or '').strip()
                normalized = self._normalize(label)
                if not normalized:
                    continue
                nodes.append(
                    WitnessNode(
                        node_id=f'audio_event_{index}',
                        modality='audio',
                        kind='event',
                        label=label,
                        normalized_label=normalized,
                        confidence=float(item.get('score', 0.6) or 0.6),
                        start=float(item.get('start', 0.0) or 0.0),
                        end=float(item.get('end', item.get('start', 0.0)) or 0.0),
                    )
                )
        else:
            for index, label in enumerate(features.audio_events or []):
                normalized = self._normalize(str(label))
                if not normalized:
                    continue
                nodes.append(
                    WitnessNode(
                        node_id=f'audio_event_{index}',
                        modality='audio',
                        kind='event',
                        label=str(label),
                        normalized_label=normalized,
                        confidence=float((features.audio_event_scores or {}).get(label, 0.6) or 0.6),
                    )
                )

        if features.asr_segments:
            for index, segment in enumerate(features.asr_segments):
                text = str(segment.get('text') or '').strip()
                normalized = self._normalize(text)
                if not normalized:
                    continue
                nodes.append(
                    WitnessNode(
                        node_id=f'asr_segment_{index}',
                        modality='audio',
                        kind='speech',
                        label=text,
                        normalized_label=normalized,
                        confidence=0.78,
                        start=float(segment.get('start', 0.0) or 0.0),
                        end=float(segment.get('end', segment.get('start', 0.0)) or 0.0),
                    )
                )
        elif features.asr_text:
            text = str(features.asr_text).strip()
            normalized = self._normalize(text)
            if normalized:
                nodes.append(
                    WitnessNode(
                        node_id='asr_text_0',
                        modality='audio',
                        kind='speech',
                        label=text,
                        normalized_label=normalized,
                        confidence=0.72,
                    )
                )

        if features.audio_type:
            nodes.append(
                WitnessNode(
                    node_id='audio_type_0',
                    modality='audio',
                    kind='audio_type',
                    label=str(features.audio_type),
                    normalized_label=self._normalize(features.audio_type),
                    confidence=0.58,
                )
            )

        if features.audio_emotion:
            nodes.append(
                WitnessNode(
                    node_id='audio_emotion_0',
                    modality='audio',
                    kind='emotion',
                    label=str(features.audio_emotion),
                    normalized_label=self._normalize(features.audio_emotion),
                    confidence=float(features.audio_emotion_conf or 0.55),
                )
            )

        for left in nodes:
            for right in nodes:
                if left.node_id >= right.node_id:
                    continue
                shared = set(left.normalized_label.split()) & set(right.normalized_label.split())
                if shared:
                    edges.append(
                        WitnessEdge(
                            source=left.node_id,
                            target=right.node_id,
                            relation='semantic_overlap',
                            score=min(left.confidence, right.confidence),
                            metadata={'shared_tokens': sorted(shared)},
                        )
                    )
                if self._overlap(left.start, left.end, right.start, right.end):
                    edges.append(
                        WitnessEdge(
                            source=left.node_id,
                            target=right.node_id,
                            relation='temporal_overlap',
                            score=min(left.confidence, right.confidence),
                        )
                    )

        return WitnessGraph(
            nodes=nodes,
            edges=edges,
            metadata={
                'n_visual_nodes': sum(node.modality == 'visual' for node in nodes),
                'n_audio_nodes': sum(node.modality == 'audio' for node in nodes),
                'has_timestamps': any(node.start is not None and node.end is not None for node in nodes),
            },
        )
