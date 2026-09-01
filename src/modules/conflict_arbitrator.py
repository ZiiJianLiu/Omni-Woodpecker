"""
Conflict Arbitrator
===================
基于模态可靠性与反事实分支的保守仲裁器。
"""
import logging
import re
from typing import Dict, Tuple

from ..data_types import ConflictReport, ModalityFeatures

logger = logging.getLogger(__name__)

_YES_RE = re.compile(
    r"\b(yes|yeah|yep|correct|right|true|affirmative|indeed|certainly|absolutely)\b",
    re.IGNORECASE,
)
_NO_RE = re.compile(
    r"\b(no|nah|nope|incorrect|wrong|false|negative|never)\b",
    re.IGNORECASE,
)
_YN_PREFIXES = (
    "is ",
    "are ",
    "was ",
    "were ",
    "do ",
    "does ",
    "did ",
    "can ",
    "could ",
    "will ",
    "would ",
    "has ",
    "have ",
    "had ",
    "should ",
)
_YN_INSTRUCTION_RE = re.compile(
    r"\b(answer|respond)\s+with\s+(only\s+)?yes\s+or\s+no\b|\byes\s+or\s+no\b",
    re.IGNORECASE,
)


class ConflictArbitrator:
    """Reliability-Gated Counterfactual Arbitration (RGCA)."""

    def __init__(self, config):
        self.config = config

    @staticmethod
    def _extract_yes_no(text: str) -> str:
        text = (text or '').strip()
        first = text.split()[0].strip('.,!?;:').lower() if text else ''
        if first in ('yes', 'yeah', 'yep'):
            return 'Yes'
        if first in ('no', 'nah', 'nope'):
            return 'No'
        yes_n = len(_YES_RE.findall(text))
        no_n = len(_NO_RE.findall(text))
        if yes_n > no_n:
            return 'Yes'
        if no_n > yes_n:
            return 'No'
        return 'Unknown'

    @staticmethod
    def _is_yes_no_question(question: str, max_new_tokens: int = None) -> bool:
        q = (question or '').strip().lower()
        if _YN_INSTRUCTION_RE.search(q):
            return True
        if max_new_tokens is not None and max_new_tokens <= 10:
            return True
        return q.startswith(_YN_PREFIXES)

    def _estimate_visual_reliability(
        self,
        features: ModalityFeatures,
        report: ConflictReport,
    ) -> float:
        score = 0.2
        score += 0.45 * float(features.visual_emotion_conf or 0.0)
        if features.visual_objects:
            score += 0.15
        if getattr(features, 'visual_faces_count', 0) > 0:
            score += 0.1
        if 'visual' in getattr(report, 'unreliable_modalities', []):
            score -= 0.2
        return max(0.0, min(1.0, score))

    def _estimate_audio_reliability(
        self,
        features: ModalityFeatures,
        report: ConflictReport,
    ) -> float:
        score = 0.1
        audio_type = (features.audio_type or '').lower()
        if audio_type and audio_type != 'silence':
            score += 0.15
        if getattr(features, 'audio_has_speech', False):
            score += 0.15
        if features.asr_text:
            score += 0.2
        if getattr(features, 'audio_events', None):
            score += 0.15
        score += 0.3 * float(features.audio_emotion_conf or 0.0)
        if 'audio' in getattr(report, 'unreliable_modalities', []):
            score -= 0.25
        if audio_type == 'silence':
            score -= 0.2
        return max(0.0, min(1.0, score))

    def _branch_alignment_strength(
        self,
        branch_name: str,
        visual_rel: float,
        audio_rel: float,
        report: ConflictReport,
    ) -> float:
        if branch_name == 'no_audio':
            strength = max(0.0, visual_rel - audio_rel)
            if 'audio' in getattr(report, 'unreliable_modalities', []):
                strength += 0.15
            return min(1.0, strength)
        if branch_name == 'no_video':
            strength = max(0.0, audio_rel - visual_rel)
            if 'visual' in getattr(report, 'unreliable_modalities', []):
                strength += 0.15
            return min(1.0, strength)
        return 0.0

    def _branch_conflict_bonus(self, branch_name: str, report: ConflictReport) -> float:
        bonus = 0.0
        if report.audio_video_emotion_conflict:
            if branch_name == 'no_audio' and 'audio' in getattr(report, 'unreliable_modalities', []):
                bonus += 0.08
            if branch_name == 'no_video' and 'visual' in getattr(report, 'unreliable_modalities', []):
                bonus += 0.08
        if report.audio_video_content_conflict:
            if branch_name == 'no_audio' and 'audio' in getattr(report, 'unreliable_modalities', []):
                bonus += 0.06
            if branch_name == 'no_video' and 'visual' in getattr(report, 'unreliable_modalities', []):
                bonus += 0.06
        return bonus

    @staticmethod
    def _consensus_bonus(branch_name: str, outputs: Dict[str, Dict[str, float]]) -> float:
        answer = outputs[branch_name]['answer']
        if answer == 'Unknown':
            return 0.0
        peers = [
            item['answer']
            for name, item in outputs.items()
            if name != branch_name and item['answer'] != 'Unknown'
        ]
        if not peers:
            return 0.0
        matches = sum(peer == answer for peer in peers)
        return 0.06 if matches >= 1 else 0.0

    def _score_branch(
        self,
        branch_name: str,
        outputs: Dict[str, Dict[str, float]],
        visual_rel: float,
        audio_rel: float,
        report: ConflictReport,
    ) -> float:
        item = outputs[branch_name]
        score = float(item.get('margin', 0.0))
        if branch_name == 'full':
            score += float(getattr(self.config, 'arbitration_full_branch_bonus', 0.04))
        score += float(getattr(self.config, 'arbitration_reliability_bonus', 0.22)) * self._branch_alignment_strength(
            branch_name,
            visual_rel,
            audio_rel,
            report,
        )
        score += self._branch_conflict_bonus(branch_name, report)
        score += self._consensus_bonus(branch_name, outputs)
        return score

    def arbitrate(
        self,
        video_path: str,
        question: str,
        baseline_answer: str,
        conflict_report: ConflictReport,
        features: ModalityFeatures,
        adapter,
        max_new_tokens: int = None,
    ) -> Tuple[str, str, Dict[str, object]]:
        meta: Dict[str, object] = {
            'handled': False,
            'reason': 'not_applicable',
        }
        if not self._is_yes_no_question(question, max_new_tokens):
            return baseline_answer, 'none', meta

        meta['handled'] = True
        try:
            outputs = adapter.score_yes_no_branches(
                video_path,
                question,
                use_video_branch=getattr(self.config, 'arbitration_use_video_branch', True),
            )
        except Exception as exc:
            logger.warning('RGCA 分支评分失败，保留 baseline: %s', exc)
            meta['reason'] = 'branch_scoring_failed'
            return baseline_answer, 'none', meta

        baseline_label = self._extract_yes_no(baseline_answer)
        if baseline_label == 'Unknown':
            baseline_label = outputs.get('full', {}).get('answer', 'Unknown')

        visual_rel = self._estimate_visual_reliability(features, conflict_report)
        audio_rel = self._estimate_audio_reliability(features, conflict_report)
        scores = {
            name: self._score_branch(name, outputs, visual_rel, audio_rel, conflict_report)
            for name in outputs
        }
        full_score = scores.get('full', 0.0)
        best_branch, best_score = max(scores.items(), key=lambda item: item[1])
        best_output = outputs[best_branch]
        best_answer = best_output.get('answer', 'Unknown')
        best_margin = float(best_output.get('margin', 0.0))
        score_delta = best_score - full_score
        alignment = self._branch_alignment_strength(
            best_branch,
            visual_rel,
            audio_rel,
            conflict_report,
        )

        meta.update(
            {
                'baseline_label': baseline_label,
                'branch_outputs': outputs,
                'branch_scores': scores,
                'best_branch': best_branch,
                'best_answer': best_answer,
                'best_margin': best_margin,
                'score_delta': score_delta,
                'visual_reliability': visual_rel,
                'audio_reliability': audio_rel,
                'alignment_strength': alignment,
            }
        )

        min_margin = float(getattr(self.config, 'arbitration_min_branch_margin', 0.12))
        accept_margin = float(getattr(self.config, 'arbitration_accept_margin', 0.10))
        flip_guard = float(getattr(self.config, 'arbitration_flip_guard', 0.15))

        if best_branch == 'full':
            meta['reason'] = 'full_branch_best'
            return baseline_answer, 'none', meta
        if best_answer == 'Unknown':
            meta['reason'] = 'unknown_best_answer'
            return baseline_answer, 'none', meta
        if best_answer == baseline_label:
            meta['reason'] = 'best_matches_baseline'
            return baseline_answer, 'none', meta
        if best_margin < min_margin:
            meta['reason'] = 'low_branch_margin'
            return baseline_answer, 'none', meta
        if score_delta < accept_margin:
            meta['reason'] = 'insufficient_score_delta'
            return baseline_answer, 'none', meta
        if alignment < flip_guard:
            meta['reason'] = 'insufficient_alignment'
            return baseline_answer, 'none', meta

        meta['reason'] = 'accepted_counterfactual_branch'
        return best_answer, 'rgca', meta
