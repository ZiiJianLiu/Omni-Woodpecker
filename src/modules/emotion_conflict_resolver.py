"""
Emotion Conflict Resolver
=========================
Unified decision layer that combines:
1. direct question-conditioned evidence,
2. affect conflict as a global consistency prior,
3. modality reliability for conflict-aware calibration.

This module is not task-specific. It routes by question structure and uses
emotion conflict differently for entity, relation, and explicit emotion queries.
"""
import re
from typing import Dict, List, Optional, Tuple

from ..data_types import ConflictReport, ModalityFeatures
from .question_conditioned_evidence import QuestionConditionedEvidenceScorer

_RELATION_GENERIC_CUES = {
    'speech',
    'music',
    'sound',
    'sounds',
    'audio',
    'noise',
    'animal',
    'animals',
    'vehicle',
    'vehicles',
    'water',
    'liquid',
    'crowd',
    'laughter',
    'laughing',
    'voice',
    'voices',
    'narration',
    'narration monologue',
    'singing',
    'engine',
    'wind',
    'door',
}
_EMOTION_WORDS = {
    'emotion',
    'emotions',
    'emotional',
    'mood',
    'moods',
    'feeling',
    'feelings',
    'tone',
    'tones',
    'sentiment',
    'affect',
    'affective',
}
_RELATION_WORDS = {
    'match',
    'matching',
    'matched',
    'consistent',
    'consistency',
    'align',
    'aligned',
    'correspond',
    'corresponding',
    'fit',
}


class EmotionConflictResolver:
    """Conflict-aware joint decision resolver."""

    def __init__(self, config):
        self.config = config
        self.evidence = QuestionConditionedEvidenceScorer(config)

    @staticmethod
    def _normalize(text: str) -> str:
        text = re.sub(r"[^a-z0-9']+", ' ', (text or '').lower())
        return ' '.join(text.split())

    @staticmethod
    def _extract_yes_no(text: str) -> str:
        normalized = (text or '').strip().lower()
        if normalized.startswith('yes'):
            return 'Yes'
        if normalized.startswith('no'):
            return 'No'
        return 'Unknown'

    @staticmethod
    def _clip01(value: float) -> float:
        return max(0.0, min(1.0, float(value)))

    @staticmethod
    def _truncate_text(text: Optional[str], max_chars: int = 240) -> str:
        normalized = ' '.join((text or '').split())
        if len(normalized) <= max_chars:
            return normalized
        return normalized[: max_chars - 3].rstrip() + '...'

    def _format_items(self, items: Optional[List[str]], limit: int) -> str:
        seen = set()
        ordered = []
        for item in items or []:
            normalized = self._normalize(item)
            if not normalized or normalized in seen:
                continue
            seen.add(normalized)
            ordered.append(item.strip())
            if len(ordered) >= limit:
                break
        if not ordered:
            return 'none'
        return ', '.join(ordered)

    @staticmethod
    def _dedupe_contexts(
        contexts: Optional[List[str]],
        limit: int = 4,
    ) -> List[str]:
        seen = set()
        ordered: List[str] = []
        for context in contexts or []:
            normalized = ' '.join((context or '').split())
            if not normalized:
                continue
            key = normalized.lower()
            if key in seen:
                continue
            seen.add(key)
            ordered.append(normalized)
            if len(ordered) >= limit:
                break
        return ordered

    def _modality_signal_strengths(
        self,
        report: Optional[ConflictReport],
        features: ModalityFeatures,
    ) -> Dict[str, float]:
        report = report or ConflictReport()

        visual = 0.0
        if features.visual_emotion:
            visual += 0.26 + 0.30 * float(features.visual_emotion_conf or 0.0)
        if features.visual_objects:
            visual += 0.24
        if features.visual_scene:
            visual += 0.10
        face_count = int(getattr(features, 'visual_faces_count', 0) or 0)
        if face_count > 0:
            visual += min(0.16, 0.08 + 0.02 * float(face_count))

        audio = 0.0
        audio_type = (features.audio_type or '').lower()
        if audio_type and audio_type != 'silence':
            audio += 0.14
        if features.audio_emotion:
            audio += 0.22 + 0.26 * float(features.audio_emotion_conf or 0.0)
        if getattr(features, 'audio_has_speech', False):
            audio += 0.18
        if features.asr_text:
            audio += 0.20
        if features.audio_events or features.audio_event_scores:
            audio += 0.16
        if audio_type == 'silence':
            audio = min(audio, 0.10)

        unreliable = set(getattr(report, 'unreliable_modalities', []) or [])
        if 'visual' in unreliable:
            visual *= 0.78
        if 'audio' in unreliable:
            audio *= 0.72

        return {
            'visual': self._clip01(visual),
            'audio': self._clip01(audio),
        }

    def _modality_interference_state(
        self,
        queried_modality: str,
        report: Optional[ConflictReport],
        features: ModalityFeatures,
        affect: Dict[str, object],
    ) -> Dict[str, object]:
        if queried_modality not in {'audio', 'visual'}:
            return {
                'score': 0.0,
                'queried_modality': queried_modality,
                'counterpart_modality': None,
                'queried_strength': 0.0,
                'counterpart_strength': 0.0,
                'conflict_prior': 0.0,
            }

        report = report or ConflictReport()
        strengths = self._modality_signal_strengths(report, features)
        counterpart_modality = 'visual' if queried_modality == 'audio' else 'audio'
        queried_strength = float(strengths.get(queried_modality, 0.0))
        counterpart_strength = float(strengths.get(counterpart_modality, 0.0))

        conflict_prior = 0.0
        if getattr(report, 'audio_video_content_conflict', False):
            conflict_prior = max(conflict_prior, 0.72)
        if affect.get('conflict'):
            conflict_prior = max(conflict_prior, float(affect.get('conflict_strength', 0.0)))

        if conflict_prior <= 0.0:
            score = 0.0
        else:
            dominance_gap = max(0.0, counterpart_strength - queried_strength)
            score = self._clip01(
                0.30 * conflict_prior
                + 0.55 * conflict_prior * counterpart_strength
                + 0.40 * dominance_gap
            )
            unreliable = set(getattr(report, 'unreliable_modalities', []) or [])
            if counterpart_modality in unreliable:
                score = min(1.0, score + 0.10)
            if queried_modality in unreliable:
                score = min(1.0, score + 0.14)

        return {
            'score': score,
            'queried_modality': queried_modality,
            'counterpart_modality': counterpart_modality,
            'queried_strength': queried_strength,
            'counterpart_strength': counterpart_strength,
            'conflict_prior': conflict_prior,
        }

    def _question_policy(
        self,
        spec: Dict[str, object],
        question: Optional[str] = None,
    ) -> Dict[str, object]:
        kind = spec.get('kind')
        normalized_question = self._normalize(question or '')
        question_tokens = set(normalized_question.split())

        if kind in {'entity_audio', 'entity_visual'}:
            return {
                'name': 'entity',
                'emotion_relevance': 'low',
                'emotion_weight': float(getattr(self.config, 'entity_emotion_risk_weight', 0.18)),
                'primary_modality': spec.get('modality', 'cross_modal'),
                'positive_requires_direct': True,
                'allow_risk_negative': True,
                'allow_emotion_only_positive': False,
            }

        if kind == 'cross_modal_relation':
            return {
                'name': 'relation',
                'emotion_relevance': 'high',
                'emotion_weight': float(getattr(self.config, 'relation_emotion_risk_weight', 0.72)),
                'primary_modality': 'cross_modal',
                'positive_requires_direct': True,
                'allow_risk_negative': True,
                'allow_emotion_only_positive': False,
            }

        if kind == 'emotion_query':
            return {
                'name': 'emotion_query',
                'emotion_relevance': 'primary',
                'emotion_weight': 1.0,
                'primary_modality': spec.get('modality', 'cross_modal'),
                'positive_requires_direct': False,
                'allow_risk_negative': True,
                'allow_emotion_only_positive': True,
            }

        emotion_relevance = 'high' if (_EMOTION_WORDS & question_tokens) else 'medium'
        return {
            'name': 'open_ended',
            'emotion_relevance': emotion_relevance,
            'emotion_weight': float(
                getattr(
                    self.config,
                    'open_ended_emotion_risk_weight',
                    0.55 if emotion_relevance == 'high' else 0.38,
                )
            ),
            'primary_modality': 'cross_modal',
            'positive_requires_direct': False,
            'allow_risk_negative': True,
            'allow_emotion_only_positive': False,
        }

    def _risk_profile(
        self,
        spec: Dict[str, object],
        conflict_report: Optional[ConflictReport],
        features: ModalityFeatures,
        question: Optional[str] = None,
    ) -> Dict[str, object]:
        report = conflict_report or ConflictReport()
        affect = self._affect_state(report, features)
        policy = self._question_policy(spec, question)

        queried_modality = spec.get('modality', 'cross_modal')
        unreliable = set(getattr(report, 'unreliable_modalities', []) or [])
        queried_unreliable = queried_modality in unreliable if queried_modality in {'audio', 'visual'} else False
        dominant_modality = affect.get('dominant_modality')
        modality_conflict_state = self._modality_interference_state(
            queried_modality,
            report,
            features,
            affect,
        )

        risk = 0.0
        reasons: List[str] = []
        if report.audio_video_content_conflict:
            risk += 0.68
            reasons.append('audio_video_content_conflict')
        if report.audio_text_conflict:
            risk += 0.30
            reasons.append('audio_text_conflict')
        if affect['conflict']:
            affect_contribution = policy['emotion_weight'] * float(affect['conflict_strength'])
            if affect_contribution > 0.0:
                risk += affect_contribution
                reasons.append(f'emotion_conflict={affect_contribution:.2f}')
        if queried_unreliable:
            risk += 0.22
            reasons.append(f'{queried_modality}_unreliable')
        if (
            queried_modality in {'audio', 'visual'}
            and dominant_modality in {'audio', 'visual'}
            and dominant_modality != queried_modality
            and affect['conflict']
        ):
            risk += 0.08
            reasons.append('dominant_modality_disagrees')
        if modality_conflict_state['score'] > 0.0:
            modality_penalty = 0.16 * float(modality_conflict_state['score'])
            risk += modality_penalty
            reasons.append(f'modality_conflict={modality_penalty:.2f}')

        consistency_bonus = 0.0
        if not affect['conflict'] and affect['consistency_strength'] > 0.0:
            consistency_bonus = min(
                0.18,
                policy['emotion_weight'] * 0.18 * float(affect['consistency_strength']),
            )

        return {
            'score': self._clip01(risk),
            'reasons': reasons,
            'affect': affect,
            'policy': policy,
            'queried_modality': queried_modality,
            'queried_unreliable': queried_unreliable,
            'consistency_bonus': consistency_bonus,
            'modality_conflict_state': modality_conflict_state,
        }

    def _build_generation_evidence_contexts(
        self,
        features: ModalityFeatures,
        conflict_report: Optional[ConflictReport],
    ) -> Dict[str, str]:
        report = conflict_report or ConflictReport()

        visual_bits: List[str] = []
        if features.visual_objects:
            visual_bits.append(f'objects: {self._format_items(features.visual_objects, 4)}')
        if features.visual_scene:
            visual_bits.append(f'scene: {features.visual_scene}')
        face_count = int(getattr(features, 'visual_faces_count', 0) or 0)
        if face_count > 0:
            visual_bits.append(f'faces: {face_count}')
        if features.visual_emotion:
            visual_bits.append(f'visual emotion: {features.visual_emotion}')

        audio_bits: List[str] = []
        if features.audio_type:
            audio_bits.append(f'audio type: {features.audio_type}')
        if features.audio_events:
            audio_bits.append(f'events: {self._format_items(features.audio_events, 4)}')
        if features.asr_text:
            audio_bits.append(f'transcript: "{self._truncate_text(features.asr_text, 120)}"')
        if features.audio_emotion:
            audio_bits.append(f'audio emotion: {features.audio_emotion}')

        conflict_bits: List[str] = []
        if getattr(report, 'audio_video_emotion_conflict', False):
            conflict_bits.append(
                f'emotion conflict=true (distance {float(getattr(report, "emotion_distance", 0.0) or 0.0):.2f})'
            )
        if getattr(report, 'audio_video_content_conflict', False):
            conflict_bits.append('content conflict=true')
        if getattr(report, 'audio_text_conflict', False):
            conflict_bits.append('answer-vs-audio conflict=true')
        if getattr(report, 'dominant_modality', None):
            conflict_bits.append(f'dominant modality: {report.dominant_modality}')
        if getattr(report, 'unreliable_modalities', None):
            conflict_bits.append(
                'lower-trust modalities: ' + ', '.join(report.unreliable_modalities)
            )

        return {
            'visual': 'Visible evidence: ' + '; '.join(visual_bits) if visual_bits else '',
            'audio': 'Audio evidence: ' + '; '.join(audio_bits) if audio_bits else '',
            'conflict': 'Conflict summary: ' + '; '.join(conflict_bits) if conflict_bits else '',
        }

    def _build_audio_timeline_context(
        self,
        features: ModalityFeatures,
        *,
        max_items: int = 3,
    ) -> str:
        timeline = list(getattr(features, 'audio_event_timeline', []) or [])
        if not timeline:
            return ''

        entries: List[str] = []
        for item in sorted(
            timeline,
            key=lambda value: (
                float(value.get('score', 0.0) or 0.0),
                float(value.get('end', 0.0) or 0.0) - float(value.get('start', 0.0) or 0.0),
            ),
            reverse=True,
        ):
            label = self._normalize(str(item.get('label', '')))
            if not label:
                continue
            start = float(item.get('start', 0.0) or 0.0)
            end = float(item.get('end', start) or start)
            if end > start:
                span = f'{label} ({start:.1f}-{end:.1f}s)'
            else:
                span = f'{label} ({start:.1f}s)'
            entries.append(span)
            if len(entries) >= max_items:
                break

        if not entries:
            return ''
        return 'Audio timing anchors: ' + '; '.join(entries)

    @staticmethod
    def _build_query_focus_contexts(query_spec: Optional[Dict[str, object]]) -> List[str]:
        spec = query_spec or {}
        modality = str(spec.get('modality') or 'cross_modal')
        relation = str(spec.get('relation') or 'attribute')
        contexts: List[str] = []

        if modality == 'audio':
            contexts.append(
                'For sound, speech, or audio timing claims, rely on directly audible evidence. Visible actions or objects alone do not prove what was heard.'
            )
        elif modality == 'visual':
            contexts.append(
                'For object, action, or visibility claims, rely on directly visible evidence. Audio or transcript cues alone do not prove what is visible.'
            )
        else:
            contexts.append(
                'Keep audio-supported details and visual-supported details separate unless there is direct cross-modal alignment.'
            )

        if relation == 'sound':
            contexts.append(
                'Do not infer a sound source from visual presence alone. If the source is not clearly audible, avoid committing to it.'
            )
        elif relation == 'presence':
            contexts.append(
                'Do not infer visible objects or identities from sound alone. If the object is not clearly visible, avoid committing to it.'
            )
        elif relation == 'emotion':
            contexts.append(
                'Emotion or mood cues may describe tone, but they must not introduce concrete entities, actions, causes, or object presence.'
            )
        elif relation == 'consistency':
            contexts.append(
                'Judge cross-modal consistency by shared content or source cues, not by generic mood similarity alone.'
            )
        elif relation == 'temporal':
            contexts.append(
                'Do not force audio and visual events into the same timestamp unless both modalities independently support that alignment.'
            )
            contexts.append(
                'If timing is uncertain, describe the visual event and the audio event separately instead of asserting simultaneity or causation.'
            )
        else:
            contexts.append(
                'Use only directly supported facts. If a detail is unsupported or disputed across modalities, omit it or hedge briefly instead of guessing.'
            )
        return contexts

    def build_generation_guidance(
        self,
        question: str,
        conflict_report: Optional[ConflictReport],
        features: ModalityFeatures,
        max_new_tokens: Optional[int] = None,
    ) -> Dict[str, object]:
        spec = self.classify_question(question, max_new_tokens)
        query_spec = dict(spec.get('query_spec') or self.evidence.infer_query_spec(question, max_new_tokens))
        spec = {**spec, 'query_spec': query_spec}
        guidance = {
            'question_spec': spec,
            'safety_contexts': [],
            'emotion_constraint': None,
            'mask_audio': False,
            'mask_visual': False,
        }
        if not getattr(self.config, 'enable_generation_conflict_guidance', True):
            return guidance

        report = conflict_report or ConflictReport()
        affect = self._affect_state(report, features)
        strong_distance = float(getattr(self.config, 'joint_strong_emotion_distance', 1.1))
        evidence_contexts = self._build_generation_evidence_contexts(features, report)
        audio_timeline_context = self._build_audio_timeline_context(features)
        contexts: List[str] = []
        kind = spec.get('kind')

        if kind in {'entity_audio', 'entity_visual'}:
            queried_modality = str(spec.get('modality') or 'cross_modal')
            counterpart_modality = 'visual' if queried_modality == 'audio' else 'audio'
            modality_state = self._modality_interference_state(
                queried_modality,
                report,
                features,
                affect,
            )
            if queried_modality == 'audio':
                contexts.append(
                    'Answer this question using only audible evidence. Visible presence alone does not prove that the queried source is making sound.'
                )
                contexts.append('If the queried source is not clearly audible, answer No.')
                if evidence_contexts['audio']:
                    contexts.append(evidence_contexts['audio'])
            else:
                contexts.append(
                    'Answer this question using only visible evidence. Audio or transcript cues alone do not prove that the queried object is visible.'
                )
                contexts.append('If the queried object is not clearly visible, answer No.')
                if evidence_contexts['visual']:
                    contexts.append(evidence_contexts['visual'])

            if report.audio_video_content_conflict:
                contexts.append(
                    'Audio and visual content appear to refer to different entities or events in this sample.'
                )
            if affect.get('conflict'):
                contexts.append(
                    'Audio and visual emotion cues conflict here. Treat emotion as a soft consistency cue only, never as proof of entity presence or sound source identity.'
                )
            if modality_state.get('score', 0.0) >= 0.35:
                contexts.append(
                    f'The {counterpart_modality} modality may interfere with this question; prioritize the {queried_modality} modality when evidence disagrees.'
                )

            strong_conflict = bool(
                report.audio_video_content_conflict
                or (
                    affect.get('conflict')
                    and float(getattr(report, 'emotion_distance', 0.0) or 0.0) >= strong_distance
                )
                or modality_state.get('score', 0.0) >= 0.72
            )
            if getattr(self.config, 'enable_modality_masking', False) and strong_conflict:
                guidance['mask_visual'] = queried_modality == 'audio'
                guidance['mask_audio'] = queried_modality == 'visual'
        elif kind == 'cross_modal_relation':
            contexts.append('Judge whether the audio and video describe the same event or context.')
            contexts.append('Direct source or content alignment matters more than mood similarity alone.')
            if evidence_contexts['visual']:
                contexts.append(evidence_contexts['visual'])
            if evidence_contexts['audio']:
                contexts.append(evidence_contexts['audio'])
            if evidence_contexts['conflict']:
                contexts.append(evidence_contexts['conflict'])
            if report.audio_video_content_conflict:
                contexts.append(
                    'There is explicit evidence that audio mentions content not supported by the visible scene.'
                )
            elif affect.get('conflict'):
                contexts.append(
                    'Emotion disagreement is present, but it is only a soft mismatch cue unless content also diverges.'
                )
        elif kind == 'emotion_query':
            contexts.append(
                'Answer only about emotion or mood. Do not infer entities, actions, or object presence from emotion labels.'
            )
            queried_modality = str(spec.get('modality') or 'cross_modal')
            if queried_modality == 'audio':
                contexts.append('Prioritize the audio track when judging emotion for this question.')
                if evidence_contexts['audio']:
                    contexts.append(evidence_contexts['audio'])
            elif queried_modality == 'visual':
                contexts.append('Prioritize the visible scene when judging emotion for this question.')
                if evidence_contexts['visual']:
                    contexts.append(evidence_contexts['visual'])
            else:
                contexts.append(
                    'If audio and visual emotion disagree, report the conflict conservatively instead of guessing a unified mood.'
                )
                if evidence_contexts['visual']:
                    contexts.append(evidence_contexts['visual'])
                if evidence_contexts['audio']:
                    contexts.append(evidence_contexts['audio'])

            if evidence_contexts['conflict']:
                contexts.append(evidence_contexts['conflict'])

            if (
                getattr(self.config, 'enable_modality_masking', False)
                and affect.get('conflict')
                and float(getattr(report, 'emotion_distance', 0.0) or 0.0) >= strong_distance
            ):
                guidance['mask_visual'] = queried_modality == 'audio'
                guidance['mask_audio'] = queried_modality == 'visual'

            if getattr(self.config, 'enable_emotion_constraint', False) and not affect.get('conflict'):
                if queried_modality == 'audio' and features.audio_emotion:
                    guidance['emotion_constraint'] = features.audio_emotion
                elif queried_modality == 'visual' and features.visual_emotion:
                    guidance['emotion_constraint'] = features.visual_emotion
                elif (
                    queried_modality == 'cross_modal'
                    and features.visual_emotion
                    and features.visual_emotion == features.audio_emotion
                ):
                    guidance['emotion_constraint'] = features.visual_emotion
        else:
            contexts.extend(self._build_query_focus_contexts(query_spec))
            if evidence_contexts['visual']:
                contexts.append(evidence_contexts['visual'])
            if evidence_contexts['audio']:
                contexts.append(evidence_contexts['audio'])
            if query_spec.get('relation') == 'temporal' and audio_timeline_context:
                contexts.append(audio_timeline_context)
            if evidence_contexts['conflict']:
                contexts.append(evidence_contexts['conflict'])
            if report.audio_video_content_conflict or affect.get('conflict'):
                contexts.append(
                    'When modalities disagree, prefer the modality that directly supports the claimed fact instead of blending them.'
                )
            if query_spec.get('modality') in {'audio', 'visual'} and (
                report.audio_video_content_conflict or affect.get('conflict')
            ):
                counterpart = 'visual' if query_spec.get('modality') == 'audio' else 'audio'
                contexts.append(
                    f'The {counterpart} modality may be misleading for this question; use it only as corroboration when it directly agrees.'
                )

        limit = 8 if kind == 'open_ended' else (6 if kind in {'cross_modal_relation', 'emotion_query'} else 5)
        guidance['safety_contexts'] = self._dedupe_contexts(contexts, limit=limit)
        return guidance

    def _entity_abstain_detail(
        self,
        baseline_label: str,
        meta: Dict[str, object],
    ) -> str:
        if baseline_label != 'Yes':
            if meta.get('specific_human_visual_guard') and not meta.get('allow_positive_after_guard'):
                return 'specific_human_visual_guard'
            if not meta.get('allow_positive_after_guard'):
                return 'positive_flip_not_authorized'
            if not meta.get('direct_support'):
                return 'positive_requires_direct_support'
            if float(meta.get('support_score', 0.0)) < float(meta.get('positive_threshold', 0.0)):
                return 'support_below_positive_threshold'
            margin = float(meta.get('support_score', 0.0)) - float(meta.get('contradiction_score', 0.0))
            if margin < float(meta.get('decision_margin_threshold', 0.0)):
                return 'positive_margin_too_small'
            return 'positive_candidate_not_ready'

        if not meta.get('negative_ready'):
            return 'negative_not_observable'
        if float(meta.get('contradiction_score', 0.0)) < float(meta.get('soft_negative_threshold', 0.0)):
            return 'contradiction_below_soft_negative_threshold'
        if (
            float(meta.get('contradiction_score', 0.0)) < float(meta.get('negative_threshold', 0.0))
            and float((meta.get('risk_profile') or {}).get('score', 0.0))
            < float(meta.get('negative_risk_threshold', 0.0))
        ):
            return 'contradiction_and_risk_too_weak_for_negative'
        if meta.get('direct_support'):
            return 'query_modality_direct_support_blocks_negative'
        if meta.get('counterpart_direct'):
            return 'counterpart_direct_support_blocks_negative'
        reverse_margin = float(meta.get('contradiction_score', 0.0)) - float(meta.get('support_score', 0.0))
        if (
            float(meta.get('contradiction_score', 0.0)) >= float(meta.get('negative_threshold', 0.0))
            and reverse_margin < float(meta.get('decision_margin_threshold', 0.0))
        ):
            return 'negative_margin_too_small'
        return 'negative_candidate_not_ready'

    def _relation_abstain_detail(
        self,
        baseline_label: str,
        meta: Dict[str, object],
    ) -> str:
        if baseline_label != 'Yes':
            has_anchor = bool(meta.get('direct_anchor') or meta.get('typed_proxy_anchor'))
            if not has_anchor:
                return 'positive_requires_relation_anchor'
            positive_threshold = float(meta.get('positive_threshold', 0.0))
            margin_threshold = float(meta.get('decision_margin_threshold', 0.0))
            if meta.get('typed_proxy_anchor') and not meta.get('direct_anchor'):
                positive_threshold = float(meta.get('proxy_positive_threshold', positive_threshold))
                margin_threshold = float(meta.get('proxy_margin_threshold', margin_threshold))
            if float(meta.get('support_score', 0.0)) < positive_threshold:
                return 'support_below_positive_threshold'
            margin = float(meta.get('support_score', 0.0)) - float(meta.get('contradiction_score', 0.0))
            if margin < margin_threshold:
                return 'positive_margin_too_small'
            return 'positive_candidate_not_ready'

        if not meta.get('allow_negative_flip', True):
            return 'negative_flip_not_authorized_without_content_contradiction'
        if float(meta.get('contradiction_score', 0.0)) < float(meta.get('soft_negative_threshold', 0.0)):
            return 'contradiction_below_soft_negative_threshold'
        if (
            float(meta.get('contradiction_score', 0.0)) < float(meta.get('negative_threshold', 0.0))
            and float((meta.get('risk_profile') or {}).get('score', 0.0))
            < float(meta.get('negative_risk_threshold', 0.0))
        ):
            return 'contradiction_and_risk_too_weak_for_negative'
        positive_block_threshold = float(meta.get('positive_threshold', 0.0))
        if meta.get('typed_proxy_anchor') and not meta.get('direct_anchor'):
            positive_block_threshold = float(
                meta.get('proxy_positive_threshold', positive_block_threshold)
            )
        if float(meta.get('support_score', 0.0)) >= positive_block_threshold:
            return 'strong_positive_support_blocks_negative'
        reverse_margin = float(meta.get('contradiction_score', 0.0)) - float(meta.get('support_score', 0.0))
        if (
            float(meta.get('contradiction_score', 0.0)) >= float(meta.get('negative_threshold', 0.0))
            and reverse_margin < float(meta.get('decision_margin_threshold', 0.0))
        ):
            return 'negative_margin_too_small'
        return 'negative_candidate_not_ready'

    def _query_modality_observable(
        self,
        modality: str,
        features: ModalityFeatures,
        frames,
        object_detector,
    ) -> bool:
        if modality == 'audio':
            return bool(
                features.audio_type
                or features.audio_events
                or features.audio_event_scores
                or features.asr_text
            )
        if modality == 'visual':
            return bool(
                features.visual_objects
                or features.visual_scene
                or getattr(features, 'visual_faces_count', 0)
                or (frames and object_detector is not None)
            )
        return True

    def _open_ended_dominant_modality(
        self,
        report: Optional[ConflictReport],
        features: ModalityFeatures,
    ) -> str:
        dominant = getattr(report, 'dominant_modality', None)
        if dominant in {'audio', 'visual'}:
            return dominant

        visual_score = float(getattr(features, 'visual_emotion_conf', 0.0) or 0.0)
        audio_score = float(getattr(features, 'audio_emotion_conf', 0.0) or 0.0)
        if visual_score <= 0.0 and audio_score <= 0.0:
            return 'visual'
        return 'visual' if visual_score >= audio_score else 'audio'

    def _should_attempt_open_ended(
        self,
        conflict_report: Optional[ConflictReport],
    ) -> bool:
        if not getattr(self.config, 'enable_open_ended_conflict_rewrite', True):
            return False

        # Open-ended correction is claim-guided and abstains when evidence is not
        # informative enough, so we do not gate it with the coarse conflict
        # detector. Unsupported spans often do not surface as global conflicts.
        return True

    def _should_attempt_closed_form(
        self,
        spec: Dict[str, object],
        conflict_report: Optional[ConflictReport],
    ) -> bool:
        kind = spec.get('kind')
        if kind in {'entity_audio', 'entity_visual', 'cross_modal_relation', 'emotion_query'}:
            # Closed-form questions are now evidence-driven at resolve time.
            # We do not pre-block them with a coarse conflict gate, because many
            # entity/relation errors never surface as global content conflicts.
            return True
        return False

    def _build_open_ended_evidence_sections(
        self,
        features: ModalityFeatures,
        conflict_report: Optional[ConflictReport],
        policy: Dict[str, object],
    ) -> Tuple[List[str], List[str], List[str]]:
        max_visual_items = int(getattr(self.config, 'open_ended_max_visual_items', 6))
        max_audio_items = int(getattr(self.config, 'open_ended_max_audio_items', 6))

        visual_lines: List[str] = []
        if features.visual_objects:
            visual_lines.append(
                f'- visible objects: {self._format_items(features.visual_objects, max_visual_items)}'
            )
        if features.visual_scene:
            visual_lines.append(f'- scene type: {features.visual_scene}')
        if features.visual_faces_count:
            visual_lines.append(f'- visible faces/person cues: {features.visual_faces_count}')
        if features.visual_emotion:
            visual_lines.append(
                f'- visual emotion: {features.visual_emotion} (conf {features.visual_emotion_conf:.2f})'
            )
        if not visual_lines:
            visual_lines.append('- no reliable visual evidence extracted')

        audio_lines: List[str] = []
        if features.audio_type:
            audio_lines.append(f'- audio type: {features.audio_type}')
        if features.audio_events:
            audio_lines.append(
                f'- audio events: {self._format_items(features.audio_events, max_audio_items)}'
            )
        if features.asr_text:
            audio_lines.append(
                f'- transcript: "{self._truncate_text(features.asr_text, 220)}"'
            )
        if features.audio_emotion:
            audio_lines.append(
                f'- audio emotion: {features.audio_emotion} (conf {features.audio_emotion_conf:.2f})'
            )
        if not audio_lines:
            audio_lines.append('- no reliable audio evidence extracted')

        report = conflict_report or ConflictReport()
        dominant_modality = self._open_ended_dominant_modality(report, features)
        unreliable = ', '.join(report.unreliable_modalities) if report.unreliable_modalities else 'none'
        diagnosis_lines = [
            f"- primary evidence modality: {policy.get('primary_modality') or dominant_modality}",
            f"- emotion relevance to this question: {policy.get('emotion_relevance', 'medium')}",
            f'- audio-video emotion conflict: {str(bool(report.audio_video_emotion_conflict)).lower()}',
            f'- audio-video content conflict: {str(bool(report.audio_video_content_conflict)).lower()}',
            f'- answer-vs-audio conflict: {str(bool(report.audio_text_conflict)).lower()}',
            f'- emotion distance: {float(getattr(report, "emotion_distance", 0.0) or 0.0):.2f}',
            f'- dominant modality: {dominant_modality}',
            f'- unreliable modalities: {unreliable}',
        ]
        if features.text_emotion:
            diagnosis_lines.append(
                f'- answer emotion: {features.text_emotion} (conf {features.text_emotion_conf:.2f})'
            )
        return visual_lines, audio_lines, diagnosis_lines

    def _split_claim_units(
        self,
        text: str,
        max_units: int,
    ) -> List[str]:
        raw_segments = re.split(r'(?<=[.!?;])\s+|\n+', text or '')
        units: List[str] = []
        seen = set()

        for segment in raw_segments:
            segment = ' '.join(segment.strip().split())
            if not segment:
                continue

            subsegments = [segment]
            if len(segment.split()) > 14:
                subsegments = re.split(r'\s*(?:,|;|\bbut\b|\bwhile\b|\band\b)\s+', segment)

            for subsegment in subsegments:
                subsegment = ' '.join(subsegment.strip(' ,;').split())
                normalized = self._normalize(subsegment)
                if not normalized or normalized in seen:
                    continue
                if len(normalized.split()) < 3:
                    continue
                seen.add(normalized)
                units.append(subsegment)
                if len(units) >= max_units:
                    return units

        return units

    def _build_open_ended_claim_audit_prompt(
        self,
        question: str,
        baseline_answer: str,
        claim_units: List[str],
        features: ModalityFeatures,
        conflict_report: Optional[ConflictReport],
        policy: Dict[str, object],
    ) -> str:
        visual_lines, audio_lines, diagnosis_lines = self._build_open_ended_evidence_sections(
            features,
            conflict_report,
            policy,
        )
        claim_block = '\n'.join(
            f'{idx}. {self._truncate_text(claim, 160)}'
            for idx, claim in enumerate(claim_units, start=1)
        )
        return (
            'You are auditing atomic claims in a multimodal answer.\n'
            f'Question: {question}\n'
            f'Previous answer: {self._truncate_text(baseline_answer, 320)}\n\n'
            'Visual evidence:\n'
            + '\n'.join(visual_lines)
            + '\n\nAudio evidence:\n'
            + '\n'.join(audio_lines)
            + '\n\nCross-modal diagnosis:\n'
            + '\n'.join(diagnosis_lines)
            + '\n\nClaim units:\n'
            + claim_block
            + '\n\n'
            + 'Audit rules:\n'
            + '- KEEP: directly supported by the primary evidence modality or a strong cross-modal anchor.\n'
            + '- HEDGE: only weakly supported, coarse, or supported mainly by affect/tone cues.\n'
            + '- DROP: unsupported, contradicted, or overly specific.\n'
            + '- Emotion cues may support tone or mood, but cannot establish entities, identities, actions, causes, or object presence by themselves.\n'
            + '- For visibility or presence claims, require direct visual evidence, or at least moderate visual evidence that is explicitly corroborated by aligned audio context for a context-bound object; audio alone cannot prove that an object is visible.\n'
            + '- For sound-source claims, require direct audio evidence and at least some visual support that the named source is present; visual presence alone cannot prove that it is making the sound.\n'
            + '- If a claim names a person/object/action, require direct evidence for that factual detail.\n\n'
            + 'Output exactly one line per claim in this format:\n'
            + '<index>|<KEEP/HEDGE/DROP>|<short reason>'
        )

    def _parse_claim_audit(
        self,
        audit_text: str,
        claim_units: List[str],
    ) -> List[Dict[str, object]]:
        pattern = re.compile(
            r'^\s*(\d+)\s*[|:\-]\s*(KEEP|HEDGE|DROP)\b(?:\s*[|:\-]\s*(.*))?$',
            re.IGNORECASE,
        )
        decisions: Dict[int, Tuple[str, str]] = {}
        for line in (audit_text or '').splitlines():
            match = pattern.match(line.strip())
            if not match:
                continue
            index = int(match.group(1))
            if not (1 <= index <= len(claim_units)):
                continue
            decisions[index] = (
                match.group(2).upper(),
                (match.group(3) or '').strip(),
            )

        parsed = []
        for index, claim in enumerate(claim_units, start=1):
            if index not in decisions:
                continue
            decision, reason = decisions[index]
            parsed.append(
                {
                    'index': index,
                    'claim': claim,
                    'decision': decision,
                    'reason': reason,
                }
            )
        return parsed

    def _build_open_ended_revision_prompt(
        self,
        question: str,
        baseline_answer: str,
        features: ModalityFeatures,
        conflict_report: Optional[ConflictReport],
        policy: Dict[str, object],
        claim_units: List[str],
        audited_claims: Optional[List[Dict[str, object]]] = None,
    ) -> str:
        visual_lines, audio_lines, diagnosis_lines = self._build_open_ended_evidence_sections(
            features,
            conflict_report,
            policy,
        )
        previous_answer = self._truncate_text(baseline_answer, 320)
        claim_block = '\n'.join(
            f'{idx}. {self._truncate_text(claim, 160)}'
            for idx, claim in enumerate(claim_units, start=1)
        ) if claim_units else '- no explicit claim units extracted'
        if audited_claims:
            audit_block = '\n'.join(
                f"{item['index']}. [{item['decision']}] {self._truncate_text(item['claim'], 140)}"
                + (f" :: {self._truncate_text(item['reason'], 80)}" if item.get('reason') else '')
                for item in audited_claims
            )
        else:
            audit_block = '- audit unavailable; perform an internal claim audit before rewriting'

        return (
            'You are revising a multimodal answer using claim-level correction.\n'
            f'Question: {question}\n'
            f'Previous answer: {previous_answer}\n\n'
            'Visual evidence:\n'
            + '\n'.join(visual_lines)
            + '\n\nAudio evidence:\n'
            + '\n'.join(audio_lines)
            + '\n\nCross-modal diagnosis:\n'
            + '\n'.join(diagnosis_lines)
            + '\n\nClaim units:\n'
            + claim_block
            + '\n\nClaim audit:\n'
            + audit_block
            + '\n\n'
            + 'Rewrite the answer so it directly answers the question using only the supported evidence above.\n'
            + 'Rules:\n'
            + '- KEEP claims with direct support.\n'
            + '- Convert HEDGE claims into lower-commitment wording.\n'
            + '- DELETE unsupported or contradicted claims.\n'
            + '- Emotion cues may revise tone or mood, but must not introduce concrete entities, identities, actions, causes, or object presence.\n'
            + '- If audio and visual cues conflict, prefer the primary evidence modality and treat the other modality as corroboration only.\n'
            + '- Visibility claims need visual support; use aligned audio only as corroboration for borderline evidence on context-bound objects, never as the sole basis.\n'
            + '- Sound-source claims need direct audio support and at least some visual support that the source is present.\n'
            + '- If evidence is insufficient for a specific detail, omit it or hedge it briefly instead of guessing.\n'
            + '- Keep the final answer concise and factual.\n'
            + '- Output only the revised answer.\n'
        )

    def _rewrite_open_ended_answer(
        self,
        question: str,
        baseline_answer: str,
        features: ModalityFeatures,
        conflict_report: Optional[ConflictReport],
        adapter,
        max_new_tokens: Optional[int] = None,
    ) -> Tuple[Optional[str], Dict[str, object]]:
        if adapter is None:
            return None, {
                'policy': 'open_ended_claim_guided_rewrite',
                'reason': 'open_ended_missing_adapter',
            }

        policy = self._question_policy({'kind': 'open_ended'}, question)
        has_evidence = bool(
            features.visual_objects
            or features.visual_scene
            or features.visual_emotion
            or features.visual_faces_count
            or features.audio_events
            or features.audio_type
            or features.asr_text
            or features.audio_emotion
        )
        if not has_evidence:
            return None, {
                'policy': 'open_ended_claim_guided_rewrite',
                'reason': 'open_ended_no_evidence',
            }

        rewrite_tokens = int(getattr(self.config, 'open_ended_rewrite_max_new_tokens', 96))
        if max_new_tokens is not None:
            rewrite_tokens = max(32, min(rewrite_tokens, max_new_tokens))

        claim_units = self._split_claim_units(
            baseline_answer,
            max_units=int(getattr(self.config, 'open_ended_claim_max_units', 8)),
        )
        audited_claims: List[Dict[str, object]] = []
        if getattr(self.config, 'open_ended_claim_guided_rewrite', True) and claim_units:
            audit_prompt = self._build_open_ended_claim_audit_prompt(
                question,
                baseline_answer,
                claim_units,
                features,
                conflict_report,
                policy,
            )
            audit_tokens = int(getattr(self.config, 'open_ended_claim_audit_max_new_tokens', 160))
            audit_text = (adapter.text_answer(audit_prompt, max_new_tokens=audit_tokens) or '').strip()
            audited_claims = self._parse_claim_audit(audit_text, claim_units)

        min_coverage = float(getattr(self.config, 'open_ended_claim_min_audit_coverage', 0.6))
        audit_coverage = (
            float(len(audited_claims)) / float(len(claim_units))
            if claim_units else 0.0
        )

        prompt = self._build_open_ended_revision_prompt(
            question,
            baseline_answer,
            features,
            conflict_report,
            policy,
            claim_units,
            audited_claims if audit_coverage >= min_coverage else None,
        )
        revised = (adapter.text_answer(prompt, max_new_tokens=rewrite_tokens) or '').strip()
        meta = {
            'policy': 'open_ended_claim_guided_rewrite',
            'rewrite_max_new_tokens': rewrite_tokens,
            'claim_units': claim_units,
            'claim_audit': audited_claims,
            'claim_audit_coverage': audit_coverage,
            'reason': 'open_ended_claim_guided_rewrite_applied',
        }
        if not revised:
            meta['reason'] = 'open_ended_empty_rewrite'
            return None, meta
        if self._normalize(revised) == self._normalize(baseline_answer):
            meta['reason'] = 'open_ended_rewrite_unchanged'
            return None, meta
        if self._extract_yes_no(revised) in {'Yes', 'No'} and len(revised.split()) <= 2:
            meta['reason'] = 'open_ended_invalid_binary_output'
            return None, meta
        meta['revised_answer'] = revised
        return revised, meta

    def classify_question(
        self,
        question: str,
        max_new_tokens: Optional[int] = None,
    ) -> Dict[str, object]:
        query_spec = self.evidence.infer_query_spec(question, max_new_tokens)
        if not self.evidence.is_yes_no_question(question, max_new_tokens):
            return {
                'kind': 'open_ended',
                'question_kind': 'open_ended',
                'entity': None,
                'modality': str(query_spec.get('modality') or 'cross_modal'),
                'query_spec': query_spec,
                'supported': True,
                'normalized_question': self._normalize(question),
            }

        entity_spec = self.evidence._extract_entity_spec(question)
        if entity_spec is not None:
            kind = f"entity_{entity_spec['modality']}"
            return {
                **entity_spec,
                'kind': kind,
                'query_spec': query_spec,
                'supported': True,
            }

        if getattr(self.config, 'enable_relation_consistency_evidence', True):
            relation_spec = self.evidence._extract_relation_spec(question)
            if relation_spec is not None:
                return {
                    **relation_spec,
                    'kind': 'cross_modal_relation',
                    'query_spec': query_spec,
                    'supported': True,
                }

        q = self._normalize(question)
        if _EMOTION_WORDS & set(q.split()):
            has_audio = any(tok in q for tok in ('audio', 'sound', 'sounds'))
            has_visual = any(tok in q for tok in ('video', 'visual', 'scene', 'frame'))
            modality = 'cross_modal'
            if has_audio and not has_visual:
                modality = 'audio'
            elif has_visual and not has_audio:
                modality = 'visual'
            return {
                'kind': 'emotion_query',
                'question_kind': 'emotion_query',
                'entity': None,
                'modality': modality,
                'query_spec': query_spec,
                'supported': True,
                'normalized_question': q,
            }

        return {'kind': 'other_yes_no', 'query_spec': query_spec, 'supported': False}

    def should_attempt(
        self,
        question: str,
        conflict_report: Optional[ConflictReport],
        max_new_tokens: Optional[int] = None,
    ) -> bool:
        spec = self.classify_question(question, max_new_tokens)
        if not spec.get('supported', False):
            return False
        if spec.get('kind') == 'open_ended':
            return self._should_attempt_open_ended(conflict_report)
        return self._should_attempt_closed_form(spec, conflict_report)

    def _affect_state(
        self,
        report: Optional[ConflictReport],
        features: ModalityFeatures,
    ) -> Dict[str, object]:
        strong_distance = float(getattr(self.config, 'joint_strong_emotion_distance', 1.1))
        report = report or ConflictReport()
        distance = float(getattr(report, 'emotion_distance', 0.0) or 0.0)
        conflict = bool(getattr(report, 'audio_video_emotion_conflict', False))
        if conflict:
            conflict_strength = self._clip01(distance / max(strong_distance, 1e-6))
            consistency_strength = 0.0
        elif features.visual_emotion and features.audio_emotion:
            consistency_strength = self._clip01(1.0 - distance / max(strong_distance, 1e-6))
            conflict_strength = 0.0
        else:
            conflict_strength = 0.0
            consistency_strength = 0.0
        return {
            'conflict': conflict,
            'conflict_strength': conflict_strength,
            'consistency_strength': consistency_strength,
            'dominant_modality': getattr(report, 'dominant_modality', None),
            'unreliable_modalities': set(getattr(report, 'unreliable_modalities', []) or []),
        }

    def _relation_cue_is_specific(self, cue: Optional[str]) -> bool:
        cue = self._normalize(cue or '')
        if not cue:
            return False
        if cue in _RELATION_GENERIC_CUES:
            return False
        if len(cue) <= 3:
            return False
        return True

    def _score_entity_question(
        self,
        spec: Dict[str, object],
        baseline_label: str,
        features: ModalityFeatures,
        conflict_report: Optional[ConflictReport],
        frames,
        object_detector,
    ) -> Tuple[Optional[str], Dict[str, object]]:
        modality = spec['modality']
        entity = spec['entity']
        risk_profile = self._risk_profile(spec, conflict_report, features)
        affect = risk_profile['affect']
        policy = risk_profile['policy']
        specific_human_visual = (
            modality == 'visual'
            and self.evidence._is_specific_human_entity(entity)
        )

        if modality == 'audio':
            evidence = self.evidence._score_audio(entity, features)
            counterpart = self.evidence._score_visual(entity, features, frames, object_detector)
        else:
            evidence = self.evidence._score_visual(entity, features, frames, object_detector)
            counterpart = self.evidence._score_audio(entity, features)

        support = float(evidence.get('support_score', 0.0))
        contradiction = float(evidence.get('contradiction_score', 0.0))
        support_raw = support
        contradiction_raw = contradiction
        direct_support = bool(evidence.get('direct_support', False))
        proxy_support = bool(evidence.get('proxy_support', False)) or (support > 0.0 and not direct_support)
        allow_positive = bool(evidence.get('allow_positive_flip', False)) and direct_support
        if specific_human_visual:
            allow_positive = False
        allow_negative = bool(evidence.get('allow_negative_flip', modality == 'audio'))

        counterpart_support = float(counterpart.get('support_score', 0.0))
        counterpart_contradiction = float(counterpart.get('contradiction_score', 0.0))
        counterpart_direct = bool(counterpart.get('direct_support', False))
        counterpart_threshold = float(getattr(self.config, 'joint_entity_counterpart_threshold', 0.72))
        corroboration_bonus = float(getattr(self.config, 'joint_entity_counterpart_bonus', 0.08))
        visual_counterpart_floor = float(
            getattr(self.config, 'joint_entity_visual_counterpart_floor', 0.56)
        )
        visual_counterpart_bonus = float(
            getattr(self.config, 'joint_entity_visual_counterpart_bonus', 0.14)
        )
        audio_visual_floor = float(
            getattr(self.config, 'joint_entity_audio_visual_floor', 0.52)
        )
        audio_visual_bonus = float(
            getattr(self.config, 'joint_entity_audio_visual_bonus', 0.10)
        )
        positive_risk_ceiling = float(
            getattr(self.config, 'joint_entity_positive_risk_ceiling', 0.62)
        )
        modality_conflict_strength = float(
            (risk_profile.get('modality_conflict_state') or {}).get('score', 0.0)
        )

        pos_threshold = float(getattr(self.config, 'joint_entity_support_threshold', 0.78))
        neg_threshold = float(getattr(self.config, 'joint_entity_contradiction_threshold', 0.78))
        soft_neg_threshold = float(getattr(self.config, 'joint_entity_soft_contradiction_threshold', 0.60))
        risk_neg_threshold = float(getattr(self.config, 'joint_entity_negative_risk_threshold', 0.54))
        decision_margin = float(getattr(self.config, 'joint_decision_margin', 0.12))

        low_positive_risk = bool(
            risk_profile['score'] <= positive_risk_ceiling
            or affect.get('dominant_modality') == modality
        )
        counterpart_ready = bool(counterpart_direct or counterpart_support >= counterpart_threshold)
        calibrated_visual_positive = False
        audio_visual_gate = True
        contextual_visual_recovery = bool(
            self.evidence._allows_contextual_visual_recovery(entity)
        )

        # Query modality stays primary. The opposite modality is only a gate or
        # calibrator: it can rescue borderline evidence, but it does not replace
        # the queried modality.
        if modality == 'visual':
            if direct_support and counterpart_direct:
                support = min(1.0, support + corroboration_bonus)
            elif direct_support and counterpart_support >= counterpart_threshold:
                support = min(1.0, support + corroboration_bonus * 0.5)

            generic_human_gate = True
            if self.evidence._is_human_entity(entity) and not specific_human_visual:
                generic_human_gate = bool(int(evidence.get('face_count', 0) or 0) > 0)

            # Audio counterpart alone must NOT flip visual presence.
            # calibrated_visual_positive is disabled: the queried modality
            # (visual) must provide its own direct evidence for a positive
            # flip.  Audio may only serve as a corroboration bonus when
            # visual direct_support already exists (handled above).
            calibrated_visual_positive = False
        else:
            audio_visual_gate = bool(
                counterpart_direct or counterpart_support >= audio_visual_floor
            )
            allow_positive = bool(allow_positive and audio_visual_gate)
            if direct_support and audio_visual_gate:
                support = min(
                    1.0,
                    support + (audio_visual_bonus if counterpart_direct else audio_visual_bonus * 0.6),
                )
            if counterpart_contradiction >= soft_neg_threshold and not direct_support:
                contradiction = max(
                    contradiction,
                    min(0.84, 0.34 + 0.46 * counterpart_contradiction),
                )

        if (direct_support or calibrated_visual_positive) and risk_profile['consistency_bonus'] > 0.0:
            support = min(1.0, support + risk_profile['consistency_bonus'])
        if (direct_support or calibrated_visual_positive) and counterpart_direct:
            pos_threshold = max(0.72, pos_threshold - 0.02)
        if risk_profile['queried_unreliable']:
            pos_threshold += 0.04

        if modality_conflict_strength > 0.0:
            if modality == 'audio':
                pos_threshold = min(0.92, pos_threshold + 0.05 * modality_conflict_strength)
                if not direct_support:
                    contradiction = max(
                        contradiction,
                        min(0.78, 0.22 + 0.46 * modality_conflict_strength),
                    )
                    soft_neg_threshold = max(0.46, soft_neg_threshold - 0.10 * modality_conflict_strength)
                    risk_neg_threshold = max(0.40, risk_neg_threshold - 0.10 * modality_conflict_strength)
            elif modality == 'visual' and not direct_support and not contextual_visual_recovery:
                pos_threshold = min(0.94, pos_threshold + 0.04 * modality_conflict_strength)

        # Conservative guard: when queried modality is visual and there are
        # no matched visual objects, raise the positive threshold to avoid
        # flips driven solely by noisy CLIP/grounding scores.
        if (
            modality == 'visual'
            and not evidence.get('matched_objects')
            and not evidence.get('face_count', 0)
        ):
            pos_threshold = max(pos_threshold, 0.88)

        negative_ready = self._query_modality_observable(modality, features, frames, object_detector)

        margin = support - contradiction
        reverse_margin = contradiction - support
        candidate = None
        positive_candidate = (
            baseline_label != 'Yes'
            and allow_positive
            and (direct_support or calibrated_visual_positive)
            and support >= pos_threshold
            and margin >= decision_margin
        )
        hard_negative = (
            baseline_label == 'Yes'
            and negative_ready
            and allow_negative
            and contradiction >= neg_threshold
            and reverse_margin >= decision_margin
        )
        risk_negative = (
            baseline_label == 'Yes'
            and negative_ready
            and allow_negative
            and policy.get('allow_risk_negative', True)
            and contradiction >= soft_neg_threshold
            and risk_profile['score'] >= risk_neg_threshold
            and not direct_support
            and not counterpart_direct
        )

        if positive_candidate:
            candidate = 'Yes'
        elif hard_negative or risk_negative:
            candidate = 'No'

        meta = {
            'policy': 'entity_joint_decision',
            'entity': entity,
            'modality': modality,
            'support_score': support,
            'contradiction_score': contradiction,
            'support_score_raw': support_raw,
            'contradiction_score_raw': contradiction_raw,
            'candidate': candidate,
            'affect': affect,
            'question_policy': policy,
            'risk_profile': risk_profile,
            'evidence': evidence,
            'counterpart_evidence': counterpart,
            'direct_support': direct_support,
            'proxy_support': proxy_support,
            'counterpart_support': counterpart_support,
            'counterpart_contradiction': counterpart_contradiction,
            'counterpart_direct': counterpart_direct,
            'counterpart_ready': counterpart_ready,
            'calibrated_visual_positive': calibrated_visual_positive,
            'contextual_visual_recovery': contextual_visual_recovery,
            'audio_visual_gate': audio_visual_gate,
            'modality_conflict_strength': modality_conflict_strength,
            'counterpart_role': 'corroboration_only',
            'specific_human_visual_guard': specific_human_visual,
            'allow_positive_after_guard': allow_positive,
            'negative_ready': negative_ready,
            'positive_candidate': positive_candidate,
            'hard_negative': hard_negative,
            'risk_negative': risk_negative,
            'positive_threshold': pos_threshold,
            'negative_threshold': neg_threshold,
            'soft_negative_threshold': soft_neg_threshold,
            'negative_risk_threshold': risk_neg_threshold,
            'decision_margin_threshold': decision_margin,
        }
        if candidate is None or candidate == baseline_label:
            meta['reason'] = 'entity_abstained'
            meta['abstain_detail'] = self._entity_abstain_detail(baseline_label, meta)
            return None, meta
        meta['reason'] = 'entity_joint_decision_applied'
        return candidate, meta

    def _score_relation_question(
        self,
        question: str,
        baseline_label: str,
        video_path: str,
        features: ModalityFeatures,
        conflict_report: Optional[ConflictReport],
        frames,
        object_detector,
        adapter,
    ) -> Tuple[Optional[str], Dict[str, object]]:
        spec = {'kind': 'cross_modal_relation', 'modality': 'cross_modal'}
        risk_profile = self._risk_profile(spec, conflict_report, features, question)
        affect = risk_profile['affect']
        policy = risk_profile['policy']
        base_evidence = self.evidence._score_relation_consistency(
            features,
            conflict_report,
            frames,
            object_detector,
        )
        alignment = dict(base_evidence.get('alignment') or {})
        cue = alignment.get('cue')
        alignment_score = float(alignment.get('score', 0.0))
        specific_cue = bool(alignment.get('specific_cue', self._relation_cue_is_specific(cue)))

        support = float(base_evidence.get('support_score', 0.0))
        contradiction = float(base_evidence.get('contradiction_score', 0.0))
        support_sources = list(base_evidence.get('support_sources', []))
        contradiction_reasons = list(base_evidence.get('contradiction_reasons', []))
        allow_positive = bool(base_evidence.get('allow_positive_flip', False))
        allow_negative = bool(base_evidence.get('allow_negative_flip', True))
        direct_anchor = bool(base_evidence.get('direct_anchor', False))
        typed_proxy_anchor = bool(base_evidence.get('typed_proxy_anchor', False))
        explicit_negative_evidence = bool(base_evidence.get('negative_evidence_ready', False))

        if direct_anchor and risk_profile['consistency_bonus'] > 0.0:
            support = min(1.0, support + risk_profile['consistency_bonus'])
            support_sources.append(f"affect_consistency_bonus={risk_profile['consistency_bonus']:.2f}")
        elif typed_proxy_anchor and risk_profile['consistency_bonus'] > 0.0:
            proxy_bonus = min(0.05, 0.5 * risk_profile['consistency_bonus'])
            support = min(1.0, support + proxy_bonus)
            support_sources.append(f'affect_consistency_proxy_bonus={proxy_bonus:.2f}')

        if alignment_score > 0.0:
            if specific_cue and direct_anchor:
                support_sources.append(f'direct_alignment_verified={cue}:{alignment_score:.2f}')
            elif specific_cue and typed_proxy_anchor:
                support_sources.append(f'typed_proxy_alignment={cue}:{alignment_score:.2f}')
            elif specific_cue:
                support_sources.append(f'weak_alignment={cue}:{alignment_score:.2f}')
            else:
                support_sources.append(f'generic_alignment={cue}:{alignment_score:.2f}')

        branch_meta = {
            'skipped': True,
            'reason': 'direct_evidence_primary',
            'support_bonus': 0.0,
            'contradiction_bonus': 0.0,
        }

        yes_threshold = float(getattr(self.config, 'joint_relation_support_threshold', 0.72))
        proxy_yes_threshold = float(getattr(self.config, 'joint_relation_proxy_support_threshold', 0.74))
        no_threshold = float(getattr(self.config, 'joint_relation_contradiction_threshold', 0.72))
        soft_no_threshold = float(getattr(self.config, 'joint_relation_soft_contradiction_threshold', 0.58))
        risk_no_threshold = float(getattr(self.config, 'joint_relation_negative_risk_threshold', 0.55))
        decision_margin = float(getattr(self.config, 'joint_decision_margin', 0.12))
        proxy_margin = float(getattr(self.config, 'joint_relation_proxy_margin', 0.16))
        if typed_proxy_anchor and affect['conflict']:
            proxy_yes_threshold = min(
                0.88,
                proxy_yes_threshold + 0.04 * float(affect.get('conflict_strength', 0.0)),
            )
            proxy_margin = min(
                0.24,
                proxy_margin + 0.04 * float(affect.get('conflict_strength', 0.0)),
            )

        margin = support - contradiction
        reverse_margin = contradiction - support
        candidate = None
        positive_candidate = (
            baseline_label != 'Yes'
            and allow_positive
            and (
                (
                    direct_anchor
                    and support >= yes_threshold
                    and margin >= decision_margin
                )
                or (
                    typed_proxy_anchor
                    and support >= proxy_yes_threshold
                    and margin >= proxy_margin
                    and contradiction < soft_no_threshold
                )
            )
        )
        hard_negative = (
            baseline_label == 'Yes'
            and allow_negative
            and explicit_negative_evidence
            and contradiction >= no_threshold
            and reverse_margin >= decision_margin
        )
        risk_negative = (
            baseline_label == 'Yes'
            and allow_negative
            and explicit_negative_evidence
            and policy.get('allow_risk_negative', True)
            and contradiction >= soft_no_threshold
            and risk_profile['score'] >= risk_no_threshold
            and support < (yes_threshold if direct_anchor else proxy_yes_threshold)
        )

        if positive_candidate:
            candidate = 'Yes'
        elif hard_negative or risk_negative:
            candidate = 'No'

        meta = {
            'policy': 'relation_joint_decision',
            'support_score': support,
            'contradiction_score': contradiction,
            'base_evidence': base_evidence,
            'alignment': alignment,
            'specific_cue': specific_cue,
            'support_sources': support_sources,
            'contradiction_reasons': contradiction_reasons,
            'affect': affect,
            'question_policy': policy,
            'risk_profile': risk_profile,
            'branch_meta': branch_meta,
            'direct_anchor': direct_anchor,
            'typed_proxy_anchor': typed_proxy_anchor,
            'explicit_negative_evidence': explicit_negative_evidence,
            'allow_positive_flip': allow_positive,
            'allow_negative_flip': allow_negative,
            'candidate': candidate,
            'positive_candidate': positive_candidate,
            'hard_negative': hard_negative,
            'risk_negative': risk_negative,
            'positive_threshold': yes_threshold,
            'proxy_positive_threshold': proxy_yes_threshold,
            'negative_threshold': no_threshold,
            'soft_negative_threshold': soft_no_threshold,
            'negative_risk_threshold': risk_no_threshold,
            'decision_margin_threshold': decision_margin,
            'proxy_margin_threshold': proxy_margin,
        }
        if candidate is None or candidate == baseline_label:
            meta['reason'] = 'relation_abstained'
            meta['abstain_detail'] = self._relation_abstain_detail(baseline_label, meta)
            return None, meta
        meta['reason'] = 'relation_joint_decision_applied'
        return candidate, meta

    def _score_emotion_query(
        self,
        baseline_label: str,
        conflict_report: Optional[ConflictReport],
        features: ModalityFeatures,
    ) -> Tuple[Optional[str], Dict[str, object]]:
        affect = self._affect_state(conflict_report, features)
        candidate = None
        if affect['conflict'] and affect['conflict_strength'] >= 0.45:
            candidate = 'No'
            reason = 'emotion_conflict_detected'
        elif features.visual_emotion and features.audio_emotion and not affect['conflict']:
            candidate = 'Yes'
            reason = 'emotion_consistency_detected'
        else:
            reason = 'emotion_query_abstained'
        meta = {
            'policy': 'emotion_query_decision',
            'affect': affect,
            'candidate': candidate,
            'reason': reason,
        }
        if candidate is None or candidate == baseline_label:
            return None, meta
        return candidate, meta

    def resolve(
        self,
        video_path: str,
        question: str,
        baseline_answer: str,
        conflict_report: ConflictReport,
        features: ModalityFeatures,
        adapter,
        *,
        frames=None,
        object_detector=None,
        max_new_tokens: Optional[int] = None,
    ) -> Tuple[str, str, Dict[str, object]]:
        meta: Dict[str, object] = {'handled': False, 'reason': 'not_applicable'}
        spec = self.classify_question(question, max_new_tokens)
        meta['question_spec'] = spec
        if not spec.get('supported', False):
            meta['reason'] = 'unsupported_question_shape'
            return baseline_answer, 'none', meta

        meta['handled'] = True
        baseline_label = self._extract_yes_no(baseline_answer)
        meta['baseline_label'] = baseline_label

        if spec['kind'] in {'entity_audio', 'entity_visual'}:
            answer, local_meta = self._score_entity_question(
                spec,
                baseline_label,
                features,
                conflict_report,
                frames,
                object_detector,
            )
        elif spec['kind'] == 'cross_modal_relation':
            answer, local_meta = self._score_relation_question(
                question,
                baseline_label,
                video_path,
                features,
                conflict_report,
                frames,
                object_detector,
                adapter,
            )
        elif spec['kind'] == 'open_ended':
            answer, local_meta = self._rewrite_open_ended_answer(
                question,
                baseline_answer,
                features,
                conflict_report,
                adapter,
                max_new_tokens=max_new_tokens,
            )
        else:
            answer, local_meta = self._score_emotion_query(
                baseline_label,
                conflict_report,
                features,
            )

        meta.update(local_meta)
        if answer is None:
            return baseline_answer, 'none', meta
        return answer, str(local_meta.get('policy') or 'joint_conflict'), meta
