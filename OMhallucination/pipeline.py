"""
Multimodal Hallucination Suppression Pipeline
=============================================
Training-free 多模态幻觉抑制主流程

Pipeline 流程：
  Stage 1: 提取多模态特征（视觉情感 / 音频情感 / ASR / 物体）
  Stage 2: 检测跨模态冲突（情感冲突 / 内容冲突）
  Stage 3: 使用 MUSE + weak TABS + DCOD + CSS + LWA 进行生成期抑制

默认关闭的旧路径：
  - DSAV 动态投机式解码
  - validator 证据预取缓存
  - Stage 4 joint resolver / post-hoc correction
  - verifier / answer-conditioned feature re-extract
"""
import logging
import time
from typing import Dict, Optional

from .config import PipelineConfig
from .data_types import ConflictReport, ModalityFeatures, PipelineResult, VerificationResult
from .modules.async_validator import AsyncValidator
from .modules.conflict_detector import CrossModalConflictDetector
from .modules.corrector import AnswerCorrector
from .modules.modality_extractor import ModalityExtractor
from .modules.question_conditioned_evidence import QuestionConditionedEvidenceScorer
from .modules.verifier import AnswerVerifier
from .qwen_omni_adapter import QwenOmniAdapter

logger = logging.getLogger(__name__)


class MultimodalHallucinationSuppressionPipeline:
    """多模态幻觉抑制 Pipeline（Training-free）"""

    def __init__(self, config: PipelineConfig):
        self.config = config

        logger.info('初始化 Pipeline...')
        self.adapter = QwenOmniAdapter(
            model_path=config.model_paths.qwen_omni_path,
            device=config.device,
        )
        self.extractor = ModalityExtractor(
            visual_emotion_model=config.model_paths.visual_emotion_model,
            audio_emotion_model=config.model_paths.audio_emotion_model,
            asr_model_size=config.model_paths.asr_model_size,
            clip_model=config.model_paths.clip_model,
            grounding_model=config.model_paths.grounding_model,
            grounding_fallback_model=config.model_paths.grounding_fallback_model,
            device=config.device,
        )
        self.detector = CrossModalConflictDetector(config.conflict_detection)
        self.corrector = AnswerCorrector(config.correction)
        self.verifier = None
        if config.runtime.enable_verifier:
            self.verifier = AnswerVerifier(
                config.verification,
                object_detector=self.extractor.object_detector,
            )
        self.async_validator = None
        if config.speculative_decoding.enable or config.runtime.prepare_validator_evidence:
            self.async_validator = AsyncValidator(
                config.speculative_decoding.sensitive_words_path,
                audio_event_model_name=config.model_paths.audio_event_model,
                asr_model_size=config.model_paths.asr_model_size,
                device=config.speculative_decoding.validator_device,
                max_rollback_steps=config.speculative_decoding.max_rollback_steps,
                validation_window_tokens=config.speculative_decoding.validation_window_tokens,
                chunk_duration_s=config.speculative_decoding.audio_event_chunk_duration_s,
                hop_duration_s=config.speculative_decoding.audio_event_hop_duration_s,
                audio_support_threshold=config.speculative_decoding.audio_event_threshold,
            )
        logger.info('Pipeline 初始化完成')

    @staticmethod
    def _merge_validator_evidence(
        features: Optional[ModalityFeatures],
        evidence: Optional[Dict[str, object]],
    ) -> Optional[ModalityFeatures]:
        if features is None or not evidence:
            return features

        audio_event_scores = evidence.get('audio_event_scores') or {}
        top_audio_events = evidence.get('top_audio_events') or []
        audio_event_timeline = evidence.get('audio_event_timeline') or []
        asr_text = evidence.get('asr_text')
        asr_segments = evidence.get('asr_segments') or []
        asr_index = evidence.get('asr_index') or {}

        if audio_event_scores:
            features.audio_event_scores = dict(audio_event_scores)
        if top_audio_events:
            features.audio_events = list(top_audio_events)
        if audio_event_timeline:
            features.audio_event_timeline = list(audio_event_timeline)
        if asr_text and not features.asr_text:
            features.asr_text = asr_text
        if asr_segments and not features.asr_segments:
            features.asr_segments = list(asr_segments)
        if asr_index and not features.asr_timestamp_index:
            features.asr_timestamp_index = dict(asr_index)
        if features.asr_text or features.asr_segments:
            features.audio_has_speech = True
            if not features.audio_type:
                features.audio_type = 'speech'

        return features

    @staticmethod
    def _is_yes_no_question(question: str, max_new_tokens: Optional[int] = None) -> bool:
        return QuestionConditionedEvidenceScorer.is_yes_no_question(question, max_new_tokens)

    @classmethod
    def _canonicalize_yes_no_answer(
        cls,
        question: str,
        answer: Optional[str],
        max_new_tokens: Optional[int] = None,
    ) -> Optional[str]:
        if not answer:
            return answer
        answer = answer.strip()
        if not cls._is_yes_no_question(question, max_new_tokens):
            return answer
        label = QuestionConditionedEvidenceScorer._extract_yes_no(answer)
        if label in {'Yes', 'No'}:
            return f'{label}.'
        return answer

    @staticmethod
    def _should_raise_runtime_error(exc: Exception) -> bool:
        return isinstance(exc, (NameError, AttributeError, UnboundLocalError, KeyError, TypeError))

    def _should_attempt_correction(
        self,
        report: Optional[ConflictReport],
        question: str,
        max_new_tokens: Optional[int] = None,
    ) -> bool:
        return (
            bool(self.config.runtime.enable_stage4_correction)
            and self.corrector.should_attempt(question, report, max_new_tokens)
        )

    def run(
        self,
        video_path: str,
        question: str,
        video_id: str = '',
    ) -> PipelineResult:
        """运行完整的幻觉抑制 Pipeline。"""
        t_start = time.time()
        is_yes_no = self._is_yes_no_question(question, self.config.generation.max_new_tokens)
        answer_max_tokens = (
            min(self.config.generation.max_new_tokens, 10)
            if is_yes_no else self.config.generation.max_new_tokens
        )

        logger.info('[%s] Stage 1: 提取多模态特征', video_id)
        t0 = time.time()
        features = self.extractor.extract(video_path)
        frames = self.extractor._extract_frames(video_path, n_frames=8)
        validator_evidence = {}
        if self.async_validator is not None and self.config.runtime.prepare_validator_evidence:
            try:
                validator_evidence = self.async_validator.prepare_sync(video_path)
                self._merge_validator_evidence(features, validator_evidence)
            except Exception as exc:
                if self._should_raise_runtime_error(exc):
                    raise
                logger.debug('[%s] 证据缓存准备失败: %s', video_id, exc)
        _ = time.time() - t0

        logger.info('[%s] Stage 2: 检测跨模态冲突', video_id)
        t0 = time.time()
        conflict_report = self.detector.detect(features)
        t_conflict = time.time() - t0

        if self.config.verbose:
            logger.info(
                '[%s] 冲突检测: emotion_conflict=%s, content_conflict=%s, '
                'consistency=%.2f, suggestion=%s',
                video_id,
                conflict_report.audio_video_emotion_conflict,
                conflict_report.audio_video_content_conflict,
                conflict_report.emotion_consistency_score,
                conflict_report.suggested_correction,
            )

        generation_guidance = self.corrector.build_generation_guidance(
            question,
            conflict_report,
            features,
            max_new_tokens=answer_max_tokens,
        )
        if self.config.verbose and (
            generation_guidance.get('safety_contexts')
            or generation_guidance.get('mask_audio')
            or generation_guidance.get('mask_visual')
            or generation_guidance.get('emotion_constraint')
        ):
            question_spec = generation_guidance.get('question_spec') or {}
            logger.info(
                '[%s] Generation guidance: kind=%s, mask_audio=%s, mask_visual=%s, contexts=%d, emotion=%s',
                video_id,
                question_spec.get('kind'),
                bool(generation_guidance.get('mask_audio')),
                bool(generation_guidance.get('mask_visual')),
                len(generation_guidance.get('safety_contexts') or []),
                generation_guidance.get('emotion_constraint'),
            )

        logger.info('[%s] Stage 3: Baseline 推理', video_id)
        generation_metadata = {}
        baseline_result = self.adapter.answer(
            video_path,
            question,
            max_new_tokens=answer_max_tokens,
            mask_audio=bool(generation_guidance.get('mask_audio')),
            mask_visual=bool(generation_guidance.get('mask_visual')),
            emotion_constraint=generation_guidance.get('emotion_constraint'),
            modality_features=features,
            conflict_report=conflict_report,
            decoder_config=self.config.correction,
            return_metadata=True,
            safety_contexts=generation_guidance.get('safety_contexts'),
            frames=frames,
            object_detector=self.extractor.object_detector,
        )
        if isinstance(baseline_result, dict):
            baseline_answer = baseline_result.get('text') or ''
            generation_metadata = baseline_result.get('metadata') or {}
        else:
            baseline_answer = baseline_result
        baseline_answer = self._canonicalize_yes_no_answer(question, baseline_answer, answer_max_tokens)
        baseline_features = features

        speculative_answer = baseline_answer
        speculative_metadata = {
            'trigger_count': 0,
            'rollback_count': 0,
            'validation_latency_s': 0.0,
            'rebuild_count': 0,
            'events': [],
            'preserved_baseline_without_intervention': False,
        }
        t_speculative = 0.0
        run_dsav = self.config.speculative_decoding.enable and (
            not is_yes_no or getattr(self.config.speculative_decoding, 'enable_yes_no', False)
        ) and not (
            generation_guidance.get('mask_audio') or generation_guidance.get('mask_visual')
        )
        if run_dsav:
            logger.info('[%s] Stage 3.1: DSAV 动态投机式解码', video_id)
            t0 = time.time()
            speculative_result = self.adapter.answer(
                video_path,
                question,
                max_new_tokens=answer_max_tokens,
                emotion_constraint=generation_guidance.get('emotion_constraint'),
                async_validator=self.async_validator,
                speculative_config=self.config.speculative_decoding,
                return_metadata=True,
                baseline_answer=baseline_answer,
                safety_contexts=generation_guidance.get('safety_contexts'),
            )
            t_speculative = time.time() - t0
            if isinstance(speculative_result, dict):
                speculative_answer = speculative_result.get('text') or baseline_answer
                speculative_answer = self._canonicalize_yes_no_answer(question, speculative_answer, answer_max_tokens)
                speculative_metadata = speculative_result.get('metadata', speculative_metadata)
            elif isinstance(speculative_result, str) and speculative_result.strip():
                speculative_answer = self._canonicalize_yes_no_answer(question, speculative_result, answer_max_tokens)

        corrected_answer = speculative_answer or baseline_answer
        dsav_applied = bool(
            speculative_metadata.get('rollback_count', 0)
            or speculative_metadata.get('rebuild_count', 0)
        )
        correction_type = 'dsav' if dsav_applied else 'none'
        correction_applied = dsav_applied
        n_attempts = 0

        if self.config.runtime.enable_stage4_correction and self._should_attempt_correction(
            conflict_report,
            question,
            answer_max_tokens,
        ):
            logger.info('[%s] Stage 4: 应用 unified joint resolver', video_id)
            stage4_features = features
            stage4_report = conflict_report
            if (
                self.config.runtime.extract_answer_conditioned_features
                and not is_yes_no
            ):
                try:
                    stage4_features = self.extractor.extract(video_path, answer_text=corrected_answer)
                    if validator_evidence:
                        self._merge_validator_evidence(stage4_features, validator_evidence)
                    stage4_report = self.detector.detect(stage4_features)
                except Exception as exc:
                    if self._should_raise_runtime_error(exc):
                        raise
                    logger.debug('[%s] 开放式修正特征构建失败: %s', video_id, exc)
                    stage4_features = features
                    stage4_report = conflict_report
            stage4_answer, stage4_type = self.corrector.correct(
                video_path=video_path,
                question=question,
                baseline_answer=corrected_answer,
                conflict_report=stage4_report,
                features=stage4_features,
                adapter=self.adapter,
                max_new_tokens=answer_max_tokens,
                frames=frames,
                object_detector=self.extractor.object_detector,
            )
            if stage4_type != 'none':
                corrected_answer = self._canonicalize_yes_no_answer(question, stage4_answer, answer_max_tokens)
                correction_applied = True
                correction_type = stage4_type if correction_type == 'none' else f'dsav+{stage4_type}'
                n_attempts = 1

        corrected_features = features
        if (
            self.config.runtime.extract_answer_conditioned_features
            and corrected_answer
            and (self.verifier is not None or self.config.runtime.enable_stage4_correction)
        ):
            try:
                corrected_features = self.extractor.extract(video_path, answer_text=corrected_answer)
                if validator_evidence:
                    self._merge_validator_evidence(corrected_features, validator_evidence)
            except Exception as exc:
                if self._should_raise_runtime_error(exc):
                    raise
                logger.debug('[%s] 输出特征重提取失败: %s', video_id, exc)
                corrected_features = features

        verification = VerificationResult()
        if self.verifier is not None:
            verification = self.verifier.verify(
                corrected_answer,
                corrected_features,
                conflict_report,
                frames,
            )

        t_total = time.time() - t_start
        if self.config.verbose:
            logger.info(
                '[%s] 完成: baseline=%r speculative=%r corrected=%r '
                '(correction=%s, score=%.2f, %.1fs)',
                video_id,
                baseline_answer[:50],
                speculative_answer[:50],
                corrected_answer[:50],
                correction_type,
                verification.final_score,
                t_total,
            )

        return PipelineResult(
            video_id=video_id,
            question=question,
            baseline_answer=baseline_answer,
            baseline_features=baseline_features,
            conflict_report=conflict_report,
            corrected_answer=corrected_answer,
            corrected_features=corrected_features,
            generation_metadata=generation_metadata,
            verification=verification,
            speculative_answer=speculative_answer,
            speculative_metadata=speculative_metadata,
            correction_applied=correction_applied,
            correction_type=correction_type,
            n_correction_attempts=n_attempts,
            inference_time_s=t_total,
            conflict_detection_time_s=t_conflict,
            speculative_time_s=t_speculative,
            dsav_trigger_count=speculative_metadata.get('trigger_count', 0),
            dsav_rollback_count=speculative_metadata.get('rollback_count', 0),
            dsav_validation_latency_s=speculative_metadata.get('validation_latency_s', 0.0),
        )
