"""
Benchmark/evaluate.py
=====================
多模态幻觉抑制系统评估脚本

对比：
- Baseline: Qwen2.5-Omni 直接推理（无抑制）
- Pipeline: 跨模态冲突检测 + 幻觉抑制后推理

评估流程（per item）：
                   ┌─────────────────────────────────────┐
  video + question │     Qwen2.5-Omni Multimodal Infer    │
                   └─────────────────────────────────────┘
                              │
               ┌──────────────┴───────────────────┐
               │                                  │
          BASELINE                           PIPELINE
    (直接推理, 无抑制)                  (冲突检测 + 修正推理)
               │                                  │
        text answer                         text answer
               │                                  │
               └──────────────┬───────────────────┘
                              │
                     compare + report

指标：
  Yes/No 任务   : Accuracy, F1
  Captioning    : BLEU-4, ROUGE-L, METEOR, Semantic Similarity
  冲突检测      : 情感冲突检出率, 内容冲突检出率
  抑制效果      : 修正成功率, 准确率提升 Δ

Usage:
  python Benchmark/evaluate.py --n_samples 50 --output_dir results/quick
  python Benchmark/evaluate.py --tasks "AV Matching" --output_dir results/av
  python Benchmark/evaluate.py --resume --output_dir results/full
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from tqdm import tqdm

# ── project root on sys.path ─────────────────────────────────────────────────
_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from OMhallucination.config import PipelineConfig
from OMhallucination.data_types import ConflictReport
from OMhallucination.qwen_omni_adapter import QwenOmniAdapter
from OMhallucination.modules.modality_extractor import ModalityExtractor
from OMhallucination.modules.conflict_detector import CrossModalConflictDetector
from OMhallucination.modules.corrector import AnswerCorrector
from OMhallucination.modules.question_conditioned_evidence import QuestionConditionedEvidenceScorer
from OMhallucination.modules.verifier import AnswerVerifier
from OMhallucination.modules.async_validator import AsyncValidator
from OMhallucination.Benchmark.metrics import (
    BenchmarkResult,
    extract_yes_no,
    print_report,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger("benchmark")

BENCHMARK_JSON = Path(__file__).parent / "QA.json"
VIDEO_DIR      = Path(__file__).parent / "videos"

ALL_TASKS = [
    "AV Matching",
    "Video-driven Audio Hallucination",
    "Audio-driven Video Hallucination",
    "AV Captioning",
]

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


def is_yes_no_question(question: str, max_new_tokens: int = None) -> bool:
    return QuestionConditionedEvidenceScorer.is_yes_no_question(question, max_new_tokens)


def normalize_answer_for_question(question: str, answer: Optional[str], max_new_tokens: int = None) -> Optional[str]:
    if not answer:
        return answer
    answer = answer.strip()
    if not is_yes_no_question(question, max_new_tokens):
        return answer
    label = QuestionConditionedEvidenceScorer._extract_yes_no(answer)
    if label in {'Yes', 'No'}:
        return f'{label}.'
    return answer


def should_raise_runtime_error(exc: Exception) -> bool:
    return isinstance(exc, (NameError, AttributeError, UnboundLocalError, KeyError, TypeError))


# ============================================================================
# Per-item evaluator
# ============================================================================

class ItemEvaluator:
    """评估单个 QA 样本。"""

    def __init__(
        self,
        adapter: QwenOmniAdapter,
        extractor: Optional[ModalityExtractor],
        detector: Optional[CrossModalConflictDetector],
        corrector: Optional[AnswerCorrector],
        verifier: Optional[AnswerVerifier],
        async_validator: Optional[AsyncValidator],
        config: PipelineConfig,
        video_dir: Path,
        mode: str = 'both',
        enable_token_level_suppressor: bool = False,
    ):
        self._adapter = adapter
        self._extractor = extractor
        self._detector = detector
        self._corrector = corrector
        self._verifier = verifier
        self._async_validator = async_validator
        self._config = config
        self._vdir = video_dir
        self._mode = mode
        self._feature_cache: Dict[str, tuple] = {}

        # Token-level 选择性注意力抑制器
        self._token_suppressor = None
        if enable_token_level_suppressor:
            from modules.token_level_suppressor import TokenLevelSelectiveSuppressor
            self._token_suppressor = TokenLevelSelectiveSuppressor(
                emotion_conflict_threshold=0.35,
                content_conflict_threshold=0.30,
                min_conflict_for_suppression=0.3,
                yes_bias_range=(-3.0, 0.0),
                no_bias_range=(0.0, 2.5),
            )
            logger.info("Token-level 选择性注意力抑制器已启用")

    @staticmethod
    def _merge_validator_evidence(features, evidence):
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

    def _run_baseline(
        self,
        video_path: str,
        question: str,
        video_id: str,
        max_new_tokens: int = None,
    ) -> Dict:
        t0 = time.time()
        try:
            answer = self._adapter.answer(
                video_path,
                question,
                max_new_tokens=max_new_tokens,
            )
            answer = normalize_answer_for_question(question, answer, max_new_tokens)
        except Exception as exc:
            logger.warning(
                '[%s] Baseline 推理失败: %s: %s',
                video_id,
                type(exc).__name__,
                exc,
                exc_info=True,
            )
            answer = 'Unable to process.'
        return {'text_answer': answer, 'inference_time_s': time.time() - t0}

    def _run_pipeline(self, video_path: str, question: str, video_id: str) -> Dict:
        pipeline_start = time.time()
        features = None
        frames = None
        validator_evidence = {}

        feature_t0 = time.time()
        cached = self._feature_cache.get(video_id)
        if cached is not None:
            features, frames = cached
            logger.debug('[%s] 使用缓存特征', video_id)
        else:
            try:
                frames = self._extractor._extract_frames(video_path, n_frames=8)
                features = self._extractor.extract(video_path)
            except Exception as exc:
                logger.warning('[%s] 特征提取失败: %s', video_id, exc)
                features = None

        if self._async_validator is not None and self._config.runtime.prepare_validator_evidence:
            try:
                validator_evidence = self._async_validator.prepare_sync(video_path)
                if features is not None:
                    self._merge_validator_evidence(features, validator_evidence)
            except Exception as exc:
                if should_raise_runtime_error(exc):
                    raise
                logger.debug('[%s] 证据缓存准备失败: %s', video_id, exc)

        if features is not None and cached is None:
            self._feature_cache[video_id] = (features, frames)
        feature_time = time.time() - feature_t0

        conflict_t0 = time.time()
        try:
            conflict = self._detector.detect(features) if features else ConflictReport()
        except Exception as exc:
            logger.warning('[%s] 冲突检测失败: %s', video_id, exc)
            conflict = ConflictReport() if features else None
        conflict_time = time.time() - conflict_t0

        yn_tokens = (
            min(self._config.generation.max_new_tokens, 10)
            if is_yes_no_question(question, self._config.generation.max_new_tokens)
            else None
        )

        generation_guidance = {
            'question_spec': {},
            'safety_contexts': [],
            'emotion_constraint': None,
            'mask_audio': False,
            'mask_visual': False,
        }
        if self._corrector is not None and features is not None:
            try:
                generation_guidance = self._corrector.build_generation_guidance(
                    question,
                    conflict,
                    features,
                    max_new_tokens=yn_tokens,
                )
            except Exception as exc:
                if should_raise_runtime_error(exc):
                    raise
                logger.warning('[%s] generation guidance 构建失败: %s', video_id, exc)

        generation_guidance_applied = bool(
            generation_guidance.get('safety_contexts')
            or generation_guidance.get('mask_audio')
            or generation_guidance.get('mask_visual')
            or generation_guidance.get('emotion_constraint')
        )

        generation_decoder_metadata = {}
        try:
            baseline_result = self._adapter.answer(
                video_path,
                question,
                max_new_tokens=yn_tokens,
                mask_audio=bool(generation_guidance.get('mask_audio')),
                mask_visual=bool(generation_guidance.get('mask_visual')),
                emotion_constraint=generation_guidance.get('emotion_constraint'),
                modality_features=features,
                conflict_report=conflict,
                decoder_config=self._config.correction,
                return_metadata=True,
                safety_contexts=generation_guidance.get('safety_contexts'),
                frames=frames,
                object_detector=self._extractor.object_detector,
            )
            if isinstance(baseline_result, dict):
                baseline_answer = baseline_result.get('text') or ''
                generation_decoder_metadata = baseline_result.get('metadata') or {}
            else:
                baseline_answer = baseline_result
            baseline_answer = normalize_answer_for_question(question, baseline_answer, yn_tokens)
        except Exception as exc:
            logger.warning('[%s] Pipeline baseline 推理失败: %s', video_id, exc)
            baseline_answer = 'Unable to process.'

        speculative_answer = baseline_answer
        speculative_meta = {
            'trigger_count': 0,
            'rollback_count': 0,
            'validation_latency_s': 0.0,
            'rebuild_count': 0,
            'events': [],
            'preserved_baseline_without_intervention': False,
        }
        speculative_time = 0.0
        run_dsav = (
            self._async_validator is not None
            and self._config.speculative_decoding.enable
            and (
                not is_yes_no_question(question, self._config.generation.max_new_tokens)
                or getattr(self._config.speculative_decoding, 'enable_yes_no', False)
            )
            and not (
                generation_guidance.get('mask_audio') or generation_guidance.get('mask_visual')
            )
        )
        if run_dsav:
            dsav_t0 = time.time()
            try:
                dsav_result = self._adapter.answer(
                    video_path,
                    question,
                    max_new_tokens=yn_tokens,
                    emotion_constraint=generation_guidance.get('emotion_constraint'),
                    async_validator=self._async_validator,
                    speculative_config=self._config.speculative_decoding,
                    return_metadata=True,
                    baseline_answer=baseline_answer,
                    safety_contexts=generation_guidance.get('safety_contexts'),
                )
                speculative_time = time.time() - dsav_t0
                if isinstance(dsav_result, dict):
                    speculative_answer = dsav_result.get('text') or baseline_answer
                    speculative_answer = normalize_answer_for_question(question, speculative_answer, yn_tokens)
                    speculative_meta = dsav_result.get('metadata', speculative_meta)
                elif isinstance(dsav_result, str) and dsav_result.strip():
                    speculative_answer = normalize_answer_for_question(question, dsav_result, yn_tokens)
            except Exception as exc:
                if should_raise_runtime_error(exc):
                    raise
                logger.warning('[%s] DSAV 推理失败: %s，回退到 baseline', video_id, exc)
                speculative_answer = baseline_answer

        corrected_answer = speculative_answer
        dsav_applied = bool(
            speculative_meta.get('rollback_count', 0)
            or speculative_meta.get('rebuild_count', 0)
        )
        correction_type = 'dsav' if dsav_applied else 'none'
        correction_applied = dsav_applied

        if (
            self._config.runtime.enable_stage4_correction
            and self._corrector
            and getattr(self._config.correction, 'enable_joint_conflict_resolver', True)
            and features is not None
            and self._corrector.should_attempt(question, conflict, yn_tokens)
        ):
            stage4_features = features
            stage4_conflict = conflict
            if (
                self._config.runtime.extract_answer_conditioned_features
                and not is_yes_no_question(question, self._config.generation.max_new_tokens)
            ):
                try:
                    stage4_features = self._extractor.extract(video_path, answer_text=corrected_answer)
                    if stage4_features is not None:
                        self._merge_validator_evidence(stage4_features, validator_evidence)
                        stage4_conflict = self._detector.detect(stage4_features)
                except Exception as exc:
                    if should_raise_runtime_error(exc):
                        raise
                    logger.debug('[%s] 开放式修正特征构建失败: %s', video_id, exc)
                    stage4_features = features
                    stage4_conflict = conflict

            try:
                stage4_answer, stage4_type = self._corrector.correct(
                    video_path=video_path,
                    question=question,
                    baseline_answer=speculative_answer,
                    conflict_report=stage4_conflict,
                    features=stage4_features,
                    adapter=self._adapter,
                    max_new_tokens=yn_tokens,
                    frames=frames,
                    object_detector=self._extractor.object_detector if self._extractor else None,
                )
                if stage4_type != 'none':
                    corrected_answer = normalize_answer_for_question(question, stage4_answer, yn_tokens)
                    correction_applied = True
                    correction_type = stage4_type if correction_type == 'none' else f'dsav+{stage4_type}'
            except Exception as exc:
                logger.warning('[%s] 冲突修正失败: %s，保留 DSAV 结果', video_id, exc)

        pipeline_time = time.time() - pipeline_start

        verification_score = 0.0
        verification_features = features
        if (
            self._config.runtime.extract_answer_conditioned_features
            and corrected_answer
            and (self._verifier or self._config.runtime.enable_stage4_correction)
        ):
            try:
                verification_features = self._extractor.extract(video_path, answer_text=corrected_answer)
                if verification_features is not None:
                    self._merge_validator_evidence(verification_features, validator_evidence)
            except Exception as exc:
                if should_raise_runtime_error(exc):
                    raise
                logger.debug('[%s] 输出特征重提取失败: %s', video_id, exc)
                verification_features = features
        if verification_features and conflict and self._verifier:
            try:
                if corrected_answer and hasattr(self._extractor, '_classify_text_emotion'):
                    verification_features.text_emotion, verification_features.text_emotion_conf = self._extractor._classify_text_emotion(corrected_answer)
                    verification_features.text_content = corrected_answer
                verification = self._verifier.verify(
                    corrected_answer,
                    verification_features,
                    conflict,
                    frames,
                )
                verification_score = verification.final_score
            except Exception as exc:
                logger.debug('[%s] 验证失败: %s', video_id, exc)

        top_audio_events = validator_evidence.get('top_audio_events', features.audio_events if features else [])
        asr_segments = validator_evidence.get('asr_segments', features.asr_segments if features else [])

        return {
            'text_answer': corrected_answer,
            'baseline_answer': baseline_answer,
            'speculative_answer': speculative_answer,
            'generation_guidance_applied': generation_guidance_applied,
            'generation_guidance': {
                'question_kind': (generation_guidance.get('question_spec') or {}).get('kind'),
                'mask_audio': bool(generation_guidance.get('mask_audio')),
                'mask_visual': bool(generation_guidance.get('mask_visual')),
                'context_count': len(generation_guidance.get('safety_contexts') or []),
                'emotion_constraint': generation_guidance.get('emotion_constraint'),
            },
            'generation_decoder': generation_decoder_metadata.get('decoder', 'direct'),
            'generation_decoder_metadata': generation_decoder_metadata,
            'generation_decoder_summary': {
                'decoder': generation_decoder_metadata.get('decoder', 'direct'),
                'audio_relevance': generation_decoder_metadata.get('audio_relevance'),
                'visual_relevance': generation_decoder_metadata.get('visual_relevance'),
                'last_step_avg_debt': generation_decoder_metadata.get('last_step_avg_debt'),
                'last_step_avg_supported_credit': generation_decoder_metadata.get('last_step_avg_supported_credit'),
                'margin': generation_decoder_metadata.get('margin'),
                'lwa_profile': generation_decoder_metadata.get('lwa_profile'),
                'css_profile': generation_decoder_metadata.get('css_profile'),
                'muse_profile': generation_decoder_metadata.get('muse_profile'),
                'tabs_profile': generation_decoder_metadata.get('tabs_profile'),
                'question_evidence': generation_decoder_metadata.get('question_evidence'),
            },
            'correction_applied': correction_applied,
            'correction_type': correction_type,
            'inference_time_s': pipeline_time,
            'speculative_time_s': speculative_time,
            'async_validation_latency_s': speculative_meta.get('validation_latency_s', 0.0),
            'dsav_trigger_count': speculative_meta.get('trigger_count', 0),
            'dsav_rollback_count': speculative_meta.get('rollback_count', 0),
            'dsav_rebuild_count': speculative_meta.get('rebuild_count', 0),
            'verification_score': verification_score,
            'dsav_events': speculative_meta.get('events', []),
            'conflict_detection': {
                'audio_video_emotion_conflict': (
                    conflict.audio_video_emotion_conflict if conflict else False
                ),
                'audio_video_content_conflict': (
                    conflict.audio_video_content_conflict if conflict else False
                ),
                'emotion_consistency_score': (
                    conflict.emotion_consistency_score if conflict else 1.0
                ),
                'emotion_distance': (
                    conflict.emotion_distance if conflict else 0.0
                ),
                'dominant_modality': (
                    conflict.dominant_modality if conflict else 'visual'
                ),
                'suggested_correction': (
                    conflict.suggested_correction if conflict else 'none'
                ),
                'detection_time_s': conflict_time,
            },
            'features': {
                'visual_emotion': features.visual_emotion if features else None,
                'audio_emotion': features.audio_emotion if features else None,
                'asr_text': features.asr_text if features else None,
                'visual_objects': features.visual_objects if features else [],
                'audio_type': features.audio_type if features else None,
                'audio_events': top_audio_events,
                'asr_segment_count': len(asr_segments),
                'feature_time_s': feature_time,
            },
        }

    def evaluate(self, item: Dict) -> Dict:
        video_id = item['video_id']
        task = item['task']
        question = item['text']
        gt_label = item['label']
        video_path = str(self._vdir / f'{video_id}.mp4')

        is_yn = is_yes_no_question(question, self._config.generation.max_new_tokens)
        if is_yn and 'yes or no' not in question.lower():
            question = question.rstrip() + ' Answer with only Yes or No.'

        result = {
            'video_id': video_id,
            'task': task,
            'label': gt_label,
            'question': question,
            'mode': self._mode,
        }

        if self._mode == 'pipeline':
            pipe = self._run_pipeline(video_path, question, video_id)
            result['pipeline'] = {k: v for k, v in pipe.items() if k not in ('conflict_detection', 'features')}
            result['conflict_detection'] = pipe['conflict_detection']
            result['features'] = pipe['features']
            result['async_validation_latency_s'] = pipe.get('async_validation_latency_s', 0.0)

        elif self._mode == 'baseline':
            yn_tokens = 10 if is_yes_no_question(question, self._config.generation.max_new_tokens) else None
            base = self._run_baseline(video_path, question, video_id, max_new_tokens=yn_tokens)
            result['baseline'] = base

        else:
            pipe = self._run_pipeline(video_path, question, video_id)
            yn_tokens = 10 if is_yes_no_question(question, self._config.generation.max_new_tokens) else None
            base = self._run_baseline(video_path, question, video_id, max_new_tokens=yn_tokens)
            result['pipeline'] = {k: v for k, v in pipe.items() if k not in ('conflict_detection', 'features')}
            result['baseline'] = base
            result['conflict_detection'] = pipe['conflict_detection']
            result['features'] = pipe['features']
            result['async_validation_latency_s'] = pipe.get('async_validation_latency_s', 0.0)

        return result


# ============================================================================
# Data loading
# ============================================================================

def load_benchmark(
    tasks: Optional[List[str]] = None,
    n_samples: Optional[int] = None,
    video_dir: Path = VIDEO_DIR,
    seed: int = 42,
) -> List[Dict]:
    with open(BENCHMARK_JSON) as f:
        data = json.load(f)

    if tasks:
        data = [x for x in data if x["task"] in tasks]

    data = [x for x in data if (video_dir / f"{x['video_id']}.mp4").exists()]

    if n_samples:
        rng = np.random.default_rng(seed)
        by_task: Dict[str, List] = {}
        for x in data:
            by_task.setdefault(x["task"], []).append(x)
        per_task = max(1, n_samples // len(by_task))
        sampled = []
        for items in by_task.values():
            idx = rng.choice(len(items), min(per_task, len(items)), replace=False)
            sampled.extend([items[i] for i in idx])
        data = sampled

    logger.info("加载 %d 个样本, 任务=%s", len(data), list({x["task"] for x in data}))
    return data


# ============================================================================
# GPU device selection
# ============================================================================

def _pick_detector_device(qwen_device: str) -> str:
    """自动选择检测器 GPU，避免与 Qwen 模型争显存。"""
    try:
        import torch
        n_gpus = torch.cuda.device_count()
        if n_gpus <= 1:
            return qwen_device if qwen_device != "auto" else "cuda:0"

        # device_map="auto" 时，Qwen 可能占多卡，选空闲最大的
        qwen_idx = -1  # auto 模式不排除特定卡
        if qwen_device != "auto" and ":" in qwen_device:
            qwen_idx = int(qwen_device.split(":")[1])

        best_idx, best_free = 0, 0
        for i in range(n_gpus):
            if i == qwen_idx:
                continue
            free = torch.cuda.mem_get_info(i)[0]
            if free > best_free:
                best_free = free
                best_idx = i

        chosen = f"cuda:{best_idx}"
        logger.info("自动选择检测器设备: %s (空闲 %.1f GiB)", chosen, best_free / 1e9)
        return chosen
    except Exception:
        return qwen_device




# ============================================================================
# Main evaluation loop
# ============================================================================

def run_evaluation(args: argparse.Namespace) -> None:
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_path = out_dir / 'raw_results.jsonl'

    tasks = [t.strip() for t in args.tasks.split(',')] if args.tasks else None
    items = load_benchmark(tasks, args.n_samples, VIDEO_DIR)

    done_ids: set = set()
    if args.resume and raw_path.exists():
        with open(raw_path) as f:
            for line in f:
                row = json.loads(line)
                done_ids.add((row['video_id'], row['task'], row['question']))
        logger.info('恢复评估 — %d 个样本已完成', len(done_ids))
    items = [x for x in items if (x['video_id'], x['task'], x['text']) not in done_ids]

    mode = getattr(args, 'mode', 'both')
    if not items:
        logger.info('所有样本已评估完成，仅生成报告。')
    else:
        logger.info('初始化模型... (mode=%s)', mode)
        config = PipelineConfig(device=args.device)
        config.verbose = args.verbose
        config.model_paths.qwen_omni_path = args.model_path
        if args.asr_model:
            config.model_paths.asr_model_size = args.asr_model
        if args.dsav_device:
            config.speculative_decoding.validator_device = args.dsav_device
        if args.sensitive_words_path:
            config.speculative_decoding.sensitive_words_path = args.sensitive_words_path
        if args.max_rollback_steps is not None:
            config.speculative_decoding.max_rollback_steps = args.max_rollback_steps
        if args.disable_dsav:
            config.speculative_decoding.enable = False
        if args.disable_generation_guidance:
            config.correction.enable_generation_conflict_guidance = False
        if getattr(args, 'disable_dcod', False):
            config.correction.enable_dcod = False
        if getattr(args, 'disable_css', False):
            config.correction.enable_css = False
        if getattr(args, 'disable_lwa', False) or getattr(args, 'fast', False):
            config.correction.enable_lwa = False
        if getattr(args, 'disable_text_branch', False) or getattr(args, 'fast', False):
            config.correction.dcod_use_text_branch = False
        if args.generation_only:
            config.correction.enable_joint_conflict_resolver = False
        if args.disable_rgca:
            config.correction.enable_conflict_arbitration = False
        if args.enable_arbitration_cmcd_fallback:
            config.correction.arbitration_fallback_to_cmcd = True
        if args.arbitration_accept_margin is not None:
            config.correction.arbitration_accept_margin = args.arbitration_accept_margin
        if args.arbitration_min_branch_margin is not None:
            config.correction.arbitration_min_branch_margin = args.arbitration_min_branch_margin
        if args.arbitration_flip_guard is not None:
            config.correction.arbitration_flip_guard = args.arbitration_flip_guard
        config.runtime.prepare_validator_evidence = False
        config.runtime.enable_verifier = False
        config.runtime.extract_answer_conditioned_features = False
        config.runtime.enable_stage4_correction = False

        adapter = QwenOmniAdapter(
            model_path=args.model_path,
            device=args.device,
        )

        det_device = args.detector_device or 'cpu'
        logger.info('Qwen 设备: %s, 检测器设备: %s, DSAV 设备: %s', args.device, det_device, config.speculative_decoding.validator_device)
        logger.info(
            '运行时裁剪: validator_cache=%s verifier=%s answer_features=%s stage4=%s | decode_fast=%s lwa=%s text_branch=%s',
            config.runtime.prepare_validator_evidence,
            config.runtime.enable_verifier,
            config.runtime.extract_answer_conditioned_features,
            config.runtime.enable_stage4_correction,
            bool(getattr(args, 'fast', False)),
            config.correction.enable_lwa,
            config.correction.dcod_use_text_branch,
        )

        extractor = detector = corrector = verifier = async_validator = None
        if mode in ('pipeline', 'both'):
            extractor = ModalityExtractor(
                visual_emotion_model=config.model_paths.visual_emotion_model,
                audio_emotion_model=config.model_paths.audio_emotion_model,
                asr_model_size=config.model_paths.asr_model_size,
                clip_model=config.model_paths.clip_model,
                grounding_model=config.model_paths.grounding_model,
                grounding_fallback_model=config.model_paths.grounding_fallback_model,
                device=det_device,
            )
            detector = CrossModalConflictDetector(config.conflict_detection)
            corrector = AnswerCorrector(config.correction)
            if config.runtime.enable_verifier:
                verifier = AnswerVerifier(
                    config.verification,
                    object_detector=extractor.object_detector,
                    device=det_device,
                )
            if config.speculative_decoding.enable or config.runtime.prepare_validator_evidence:
                async_validator = AsyncValidator(
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

        evaluator = ItemEvaluator(
            adapter=adapter,
            extractor=extractor,
            detector=detector,
            corrector=corrector,
            verifier=verifier,
            async_validator=async_validator,
            config=config,
            video_dir=VIDEO_DIR,
            mode=mode,
            enable_token_level_suppressor=getattr(args, 'enable_token_level_suppressor', False),
        )

        n_done = 0
        errors = 0
        pbar = tqdm(items, desc=f'评估 ({mode})', unit='sample', dynamic_ncols=True)
        with open(raw_path, 'a') as fout:
            for item in pbar:
                try:
                    row = evaluator.evaluate(item)
                    fout.write(json.dumps(row, ensure_ascii=False) + '\n')
                    fout.flush()
                    n_done += 1

                    if mode == 'baseline':
                        ans = row['baseline']['text_answer'][:30]
                        pbar.set_postfix_str(f"{item['video_id']} | {ans}")
                    else:
                        corr = row.get('pipeline', {}).get('correction_type', '?')
                        rollbacks = row.get('pipeline', {}).get('dsav_rollback_count', 0)
                        pbar.set_postfix_str(f"{item['video_id']} | corr={corr} rb={rollbacks}")
                except Exception as exc:
                    logger.error('样本 %s 失败: %s', item.get('video_id'), exc, exc_info=True)
                    errors += 1
        pbar.close()
        logger.info('评估完成 (mode=%s)。%d 个样本，%d 个错误。', mode, n_done, errors)

    bench = BenchmarkResult(mode=mode)
    with open(raw_path) as f:
        for line in f:
            bench.add(json.loads(line))

    agg = bench.aggregate()
    print_report(agg, mode=mode)

    report_path = out_dir / 'report.json'
    with open(report_path, 'w') as f:
        json.dump(agg, f, indent=2, ensure_ascii=False)
    logger.info('报告已保存 → %s', report_path)

    if mode == 'both':
        _write_markdown(agg, out_dir / 'report.md')
        logger.info('Markdown 已保存 → %s', out_dir / 'report.md')


# ============================================================================
# Output helpers
# ============================================================================

def _write_markdown(agg: Dict, path: Path) -> None:
    lines = ["# 多模态幻觉抑制 Benchmark 评估报告\n"]

    # Overall
    ov = agg["overall"]
    lines.append("## 总体结果\n")
    lines.append("| 指标 | Baseline | Pipeline | Δ |")
    lines.append("|------|----------|----------|---|")
    lines.append(
        f"| Accuracy | {ov['baseline_accuracy']:.4f} "
        f"| {ov['pipeline_accuracy']:.4f} "
        f"| {ov['accuracy_delta']:+.4f} |"
    )
    lines.append(
        f"| 修正率 | — | {ov['correction_rate']:.1%} | — |"
    )
    lines.append(
        f"| DSAV 介入率 | — | {ov.get('dsav_intervention_rate', 0.0):.1%} | — |"
    )
    lines.append(
        f"| 平均异步验证时延 | — | {ov.get('avg_async_validation_latency_s', 0.0):.3f}s | — |"
    )
    lines.append(
        f"| 情感冲突检出率 | — | {ov['emotion_conflict_rate']:.1%} | — |"
    )
    lines.append(
        f"| 内容冲突检出率 | — | {ov['content_conflict_rate']:.1%} | — |"
    )
    lines.append(f"\n样本数: {ov['n_items']}\n")

    # By task
    for task, tr in agg["by_task"].items():
        lines.append(f"\n## 任务: {task}\n")
        lines.append("| 指标 | Baseline | Pipeline | Δ |")
        lines.append("|------|----------|----------|---|")
        lines.append(
            f"| Accuracy | {tr['baseline_accuracy']:.4f} "
            f"| {tr['pipeline_accuracy']:.4f} "
            f"| {tr['accuracy_delta']:+.4f} |"
        )
        lines.append(f"| 样本数 | {tr['n_items']} | — | — |")
        lines.append(f"| 修正数 | — | {tr['n_corrected']} | — |")
        lines.append(f"| DSAV 介入数 | — | {tr.get('n_dsav_intervened', 0)} | — |")
        lines.append(f"| 平均异步验证时延 | — | {tr.get('avg_async_validation_latency_s', 0.0):.3f}s | — |")

    # Correction strategy breakdown
    if "correction_breakdown" in agg:
        lines.append("\n## 修正策略分析\n")
        lines.append("| 策略 | 次数 | 改进 | 退化 | 无变化 |")
        lines.append("|------|------|------|------|--------|")
        for strategy, stats in agg["correction_breakdown"].items():
            lines.append(
                f"| {strategy} | {stats['count']} "
                f"| {stats['improved']} "
                f"| {stats['degraded']} "
                f"| {stats['unchanged']} |"
            )

    path.write_text("\n".join(lines), encoding="utf-8")


# ============================================================================
# CLI
# ============================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description='评估多模态幻觉抑制系统',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        '--model_path',
        default=str(_ROOT / 'models' / 'Qwen2.5-Omni-7B'),
        help='Qwen2.5-Omni 模型路径',
    )
    p.add_argument('--output_dir', default='Benchmark/results')
    p.add_argument('--tasks', default=None, help='逗号分隔的任务名称，None 表示全部')
    p.add_argument('--n_samples', type=int, default=None, help='最大样本数（分层采样），None 表示全部')
    p.add_argument('--resume', action='store_true', help='恢复中断的评估')
    p.add_argument('--device', default='cuda:4', help='Qwen 模型设备')
    p.add_argument('--detector_device', default=None, help='特征提取与验证器设备，默认使用 CPU')
    p.add_argument('--asr_model', default=None, help='ASR 模型大小，例如 base / small / large-v3')
    p.add_argument('--dsav_device', default=None, help='DSAV 异步验证器设备，默认沿用配置中的 cuda:2')
    p.add_argument('--sensitive_words_path', default=None, help='DSAV 敏感词表路径')
    p.add_argument('--max_rollback_steps', type=int, default=None, help='DSAV 最大回溯步数')
    p.add_argument('--disable_dsav', action='store_true', help='旧版兼容开关；当前默认已禁用 DSAV')
    p.add_argument('--disable_generation_guidance', action='store_true', help='禁用生成期 grounded guidance，做生成期消融')
    p.add_argument('--disable_dcod', action='store_true', help='禁用 DCOD，回退到原始 prompt-guided 生成')
    p.add_argument('--disable_css', action='store_true', help='禁用 CSS，仅保留 DCOD evidence-debt 解码')
    p.add_argument('--disable_lwa', action='store_true', help='禁用 LWA，减少一次重型 full-branch probe')
    p.add_argument('--disable_text_branch', action='store_true', help='禁用 text-only 分支，text logits 回退为 max(no_audio, no_visual)')
    p.add_argument('--fast', action='store_true', help='快速评测模式：禁用 LWA 和 text branch，保留主干 DCOD + CSS + MUSE + TABS')
    p.add_argument('--generation_only', action='store_true', help='旧版兼容开关；当前默认已禁用 Stage 4 joint resolver')
    p.add_argument('--disable_rgca', action='store_true', help='禁用 RGCA，仅保留 DSAV 和其他显式修正策略')
    p.add_argument('--arbitration_accept_margin', type=float, default=None, help='RGCA 接受反事实分支所需的最小总分差')
    p.add_argument('--arbitration_min_branch_margin', type=float, default=None, help='RGCA 分支 Yes/No 最小置信间隔')
    p.add_argument('--arbitration_flip_guard', type=float, default=None, help='RGCA 允许翻转答案所需的最小模态可靠性差')
    p.add_argument('--enable_arbitration_cmcd_fallback', action='store_true', help='RGCA 弃权时允许回退到 CMCD')
    p.add_argument(
        '--mode',
        choices=['pipeline', 'baseline', 'both'],
        default='both',
        help='运行模式: pipeline=仅抑制, baseline=仅直接推理, both=两者',
    )
    p.add_argument(
        '--enable_token_level_suppressor',
        action='store_true',
        help='启用 Token-level 选择性注意力抑制器',
    )
    p.add_argument('--verbose', action='store_true', help='开启调试日志')
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    else:
        logging.getLogger().setLevel(logging.INFO)
        logging.getLogger('numba').setLevel(logging.WARNING)
        logging.getLogger('numba.core').setLevel(logging.WARNING)
        logging.getLogger('faster_whisper').setLevel(logging.WARNING)
    run_evaluation(args)
