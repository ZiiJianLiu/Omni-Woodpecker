"""
Answer Corrector
=================
默认纠正链路只保留 unified joint resolver：
1. 直接证据
2. 情感冲突/一致性校准
3. 模态可靠性约束

旧的 question_evidence / RGCA / CMCD fallback 不再作为默认路径。
"""
import logging

from ..data_types import ConflictReport, ModalityFeatures
from .emotion_conflict_resolver import EmotionConflictResolver

logger = logging.getLogger(__name__)


class AnswerCorrector:
    """统一答案修正器。"""

    def __init__(self, config):
        self.config = config
        self.resolver = EmotionConflictResolver(config)

    @staticmethod
    def _format_float(value) -> str:
        try:
            return f'{float(value):.2f}'
        except (TypeError, ValueError):
            return 'n/a'

    @staticmethod
    def _format_items(items, limit: int = 3) -> str:
        cleaned = []
        for item in items or []:
            text = str(item).strip()
            if not text:
                continue
            cleaned.append(text)
            if len(cleaned) >= limit:
                break
        if not cleaned:
            return 'none'
        return ','.join(cleaned)

    def _summarize_meta(self, meta) -> str:
        parts = []
        spec = meta.get('question_spec') or {}
        if spec.get('kind'):
            parts.append(f"kind={spec.get('kind')}")
        if meta.get('policy'):
            parts.append(f"policy={meta.get('policy')}")
        if meta.get('entity'):
            parts.append(f"entity={meta.get('entity')}")
        if meta.get('modality'):
            parts.append(f"mod={meta.get('modality')}")
        if 'support_score' in meta:
            parts.append(f"s={self._format_float(meta.get('support_score'))}")
        if 'contradiction_score' in meta:
            parts.append(f"c={self._format_float(meta.get('contradiction_score'))}")

        risk_profile = meta.get('risk_profile') or {}
        if risk_profile:
            parts.append(f"risk={self._format_float(risk_profile.get('score'))}")
            parts.append(f"risk_reasons={self._format_items(risk_profile.get('reasons'))}")

        affect = meta.get('affect') or {}
        if affect:
            affect_state = 'conflict' if affect.get('conflict') else 'consistent'
            strength_key = 'conflict_strength' if affect.get('conflict') else 'consistency_strength'
            parts.append(f"affect={affect_state}:{self._format_float(affect.get(strength_key))}")
            if affect.get('dominant_modality'):
                parts.append(f"dom={affect.get('dominant_modality')}")

        if meta.get('candidate') is not None:
            parts.append(f"cand={meta.get('candidate')}")
        if meta.get('abstain_detail'):
            parts.append(f"detail={meta.get('abstain_detail')}")

        policy = meta.get('policy')
        if policy == 'entity_joint_decision':
            parts.append(f"direct={bool(meta.get('direct_support'))}")
            parts.append(f"cp_direct={bool(meta.get('counterpart_direct'))}")
            parts.append(f"cp_ready={bool(meta.get('counterpart_ready'))}")
            parts.append(f"neg_ready={bool(meta.get('negative_ready'))}")
            parts.append(f"pos_gate={bool(meta.get('positive_candidate'))}")
            parts.append(f"calib_pos={bool(meta.get('calibrated_visual_positive'))}")
            parts.append(f"ctx_vis={bool(meta.get('contextual_visual_recovery'))}")
            parts.append(f"av_gate={bool(meta.get('audio_visual_gate', True))}")
            parts.append(f"hard_neg={bool(meta.get('hard_negative'))}")
            parts.append(f"risk_neg={bool(meta.get('risk_negative'))}")
            parts.append(f"thr+={self._format_float(meta.get('positive_threshold'))}")
            parts.append(f"thr-={self._format_float(meta.get('negative_threshold'))}")
        elif policy == 'relation_joint_decision':
            alignment = meta.get('alignment') or {}
            if alignment.get('cue'):
                parts.append(f"cue={alignment.get('cue')}")
            if 'score' in alignment:
                parts.append(f"align={self._format_float(alignment.get('score'))}")
            parts.append(f"anchor={bool(meta.get('direct_anchor'))}")
            parts.append(f"proxy={bool(meta.get('typed_proxy_anchor'))}")
            parts.append(f"neg_obs={bool(meta.get('explicit_negative_evidence'))}")
            parts.append(f"pos_gate={bool(meta.get('positive_candidate'))}")
            parts.append(f"hard_neg={bool(meta.get('hard_negative'))}")
            parts.append(f"risk_neg={bool(meta.get('risk_negative'))}")
            parts.append(f"thr+={self._format_float(meta.get('positive_threshold'))}")
            parts.append(f"thr+p={self._format_float(meta.get('proxy_positive_threshold'))}")
            parts.append(f"thr-={self._format_float(meta.get('negative_threshold'))}")
            parts.append(f"support_src={self._format_items(meta.get('support_sources'))}")
            parts.append(f"contra_src={self._format_items(meta.get('contradiction_reasons'))}")
        elif policy == 'open_ended_claim_guided_rewrite':
            claim_units = meta.get('claim_units') or []
            claim_audit = meta.get('claim_audit') or []
            if claim_units:
                parts.append(f"claims={len(claim_units)}")
            if claim_audit:
                parts.append(f"audit={len(claim_audit)}/{len(claim_units) or 0}")
            if 'claim_audit_coverage' in meta:
                parts.append(f"cov={self._format_float(meta.get('claim_audit_coverage'))}")

        return ' | '.join(parts)

    def should_attempt(
        self,
        question: str,
        conflict_report: ConflictReport,
        max_new_tokens: int = None,
    ) -> bool:
        if not getattr(self.config, 'enable_joint_conflict_resolver', True):
            return False
        return self.resolver.should_attempt(question, conflict_report, max_new_tokens)

    def build_generation_guidance(
        self,
        question: str,
        conflict_report: ConflictReport,
        features: ModalityFeatures,
        max_new_tokens: int = None,
    ) -> dict:
        return self.resolver.build_generation_guidance(
            question,
            conflict_report,
            features,
            max_new_tokens=max_new_tokens,
        )

    def correct(
        self,
        video_path: str,
        question: str,
        baseline_answer: str,
        conflict_report: ConflictReport,
        features: ModalityFeatures,
        adapter,
        max_new_tokens: int = None,
        frames=None,
        object_detector=None,
    ) -> tuple:
        """基于统一 joint resolver 进行一次修正。"""
        if not getattr(self.config, 'enable_joint_conflict_resolver', True):
            return baseline_answer, 'none'
        if not self.should_attempt(question, conflict_report, max_new_tokens):
            return baseline_answer, 'none'

        try:
            answer, method, meta = self.resolver.resolve(
                video_path=video_path,
                question=question,
                baseline_answer=baseline_answer,
                conflict_report=conflict_report,
                features=features,
                adapter=adapter,
                frames=frames,
                object_detector=object_detector,
                max_new_tokens=max_new_tokens,
            )
            if method == 'none':
                logger.info(
                    '统一 joint resolver 弃权: %s | %s',
                    meta.get('reason'),
                    self._summarize_meta(meta),
                )
                return baseline_answer, 'none'

            logger.info(
                '统一 joint resolver: %s → %s (%s) | %s',
                meta.get('baseline_label'),
                answer,
                meta.get('reason'),
                self._summarize_meta(meta),
            )
            return answer, method
        except Exception as exc:
            logger.error('统一 joint resolver 失败: %s', exc, exc_info=True)
            return baseline_answer, 'none'
