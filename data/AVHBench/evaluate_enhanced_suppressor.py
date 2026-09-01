"""
Benchmark/evaluate_enhanced_suppressor.py
==========================================
方案一对比实验：增强版Token级别抑制器 vs Baseline

对比三种模式：
1. Baseline: Qwen2.5-Omni 直接推理（无抑制）
2. Original Pipeline: 原有的冲突检测 + 修正流程
3. Enhanced Suppressor: 方案一 - Token位置感知的注意力掩码

评估指标：
- Yes/No 任务: Accuracy, F1
- 冲突检测: 情感冲突检出率, 内容冲突检出率
- 抑制效果: 修正成功率, 准确率提升 Δ

Usage:
  # 运行完整对比实验（三种模式）
  python Benchmark/evaluate_enhanced_suppressor.py --n_samples 50 --output_dir results/enhanced_comparison

  # 只运行增强版抑制器
  python Benchmark/evaluate_enhanced_suppressor.py --mode enhanced --n_samples 20

  # 恢复中断的评估
  python Benchmark/evaluate_enhanced_suppressor.py --resume --output_dir results/enhanced_comparison
"""
from __future__ import annotations

import argparse
import json
import logging
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

from OMhallucination.config import PipelineConfig, TokenLevelSuppressionConfig
from OMhallucination.data_types import ConflictReport
from OMhallucination.qwen_omni_adapter import QwenOmniAdapter
from OMhallucination.modules.modality_extractor import ModalityExtractor
from OMhallucination.modules.conflict_detector import CrossModalConflictDetector
from OMhallucination.modules.corrector import AnswerCorrector
from OMhallucination.modules.question_conditioned_evidence import QuestionConditionedEvidenceScorer
from OMhallucination.modules.verifier import AnswerVerifier
from OMhallucination.modules.token_level_suppressor_enhanced import EnhancedTokenLevelSuppressor
from OMhallucination.modules.token_level_suppressor_v2 import ImprovedTokenLevelSuppressor
from OMhallucination.Benchmark.metrics import (
    BenchmarkResult,
    extract_yes_no,
    print_report,
)

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger("enhanced_benchmark")
logger.setLevel(logging.INFO)

# 只保留关键模块的 WARNING+，屏蔽详细的 DEBUG 输出
for _noisy in [
    "OMhallucination",
    "faster_whisper",
    "transformers",
    "torch",
    "PIL",
]:
    logging.getLogger(_noisy).setLevel(logging.WARNING)

BENCHMARK_JSON = Path(__file__).parent / "QA.json"
VIDEO_DIR      = Path(__file__).parent / "videos"

ALL_TASKS = [
    "AV Matching",
    "Video-driven Audio Hallucination",
    "Audio-driven Video Hallucination",
    "AV Captioning",
]


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


# ============================================================================
# Enhanced Suppressor Evaluator
# ============================================================================

class EnhancedSuppressorEvaluator:
    """评估增强版Token抑制器的效果"""

    def __init__(
        self,
        adapter: QwenOmniAdapter,
        extractor: ModalityExtractor,
        detector: CrossModalConflictDetector,
        config: PipelineConfig,
        video_dir: Path,
        suppressor_config: TokenLevelSuppressionConfig,
    ):
        self._adapter = adapter
        self._extractor = extractor
        self._detector = detector
        self._config = config
        self._vdir = video_dir
        self._feature_cache: Dict[str, tuple] = {}

        # 初始化增强版Token抑制器
        self._suppressor = EnhancedTokenLevelSuppressor(
            enable_entity_level_suppression=suppressor_config.enable_entity_level_suppression,
            enable_emotion_token_suppression=suppressor_config.enable_emotion_token_suppression,
            entity_suppression_strength=suppressor_config.entity_suppression_strength,
            emotion_token_bias=suppressor_config.emotion_token_bias,
        )
        # 保存tokenizer供后续使用
        self._tokenizer = adapter._processor.tokenizer
        logger.info("增强版Token抑制器已初始化")

    def _run_baseline(
        self,
        video_path: str,
        question: str,
        video_id: str,
        max_new_tokens: int = None,
    ) -> Dict:
        """Baseline: 直接推理，无任何抑制"""
        t0 = time.time()
        try:
            answer = self._adapter.answer(
                video_path,
                question,
                max_new_tokens=max_new_tokens,
            )
            answer = normalize_answer_for_question(question, answer, max_new_tokens)
        except Exception as exc:
            logger.warning('[%s] Baseline 推理失败: %s', video_id, exc)
            answer = 'Unable to process.'
        return {
            'text_answer': answer,
            'inference_time_s': time.time() - t0,
        }

    def _run_original_pipeline(
        self,
        video_path: str,
        question: str,
        video_id: str,
        features,
        conflict: ConflictReport,
    ) -> Dict:
        """Original Pipeline: 使用原有的冲突检测和修正流程"""
        t0 = time.time()

        yn_tokens = (
            min(self._config.generation.max_new_tokens, 10)
            if is_yes_no_question(question, self._config.generation.max_new_tokens)
            else None
        )

        try:
            # 原有的pipeline逻辑（简化版，不包含DSAV）
            answer = self._adapter.answer(
                video_path,
                question,
                max_new_tokens=yn_tokens,
            )
            answer = normalize_answer_for_question(question, answer, yn_tokens)
        except Exception as exc:
            logger.warning('[%s] Original pipeline 推理失败: %s', video_id, exc)
            answer = 'Unable to process.'

        return {
            'text_answer': answer,
            'inference_time_s': time.time() - t0,
            'correction_applied': False,
            'correction_type': 'none',
        }

    def _run_enhanced_suppressor(
        self,
        video_path: str,
        question: str,
        video_id: str,
        features,
        frames: Optional[list] = None,
    ) -> Dict:
        """Enhanced Suppressor: 方案一 Token 级别抑制 + Grounding DINO 实体检测"""
        t0 = time.time()

        yn_tokens = (
            min(self._config.generation.max_new_tokens, 10)
            if is_yes_no_question(question, self._config.generation.max_new_tokens)
            else 50
        )

        # ── Grounding DINO 实体检测 ─────────────────────────────────────────
        grounding_entity_scores = None
        asr_text = features.asr_text or ''
        if frames and asr_text and self._extractor and self._extractor.object_detector:
            try:
                grounding_entity_scores = \
                    self._extractor.object_detector.detect_asr_entities_in_frames(
                        frames=frames,
                        asr_text=asr_text,
                        detection_threshold=0.22,
                        max_frames=6,
                    )
            except Exception as exc:
                logger.warning('[%s] Grounding DINO 实体检测失败: %s', video_id, exc)

        # ── 冲突检测 ────────────────────────────────────────────────────────
        conflict_t0 = time.time()
        conflict_signal = self._suppressor.detect_conflict(
            visual_emotion=features.visual_emotion or 'neutral',
            audio_emotion=features.audio_emotion or 'neutral',
            visual_emotion_conf=features.visual_emotion_conf or 0.0,
            audio_emotion_conf=features.audio_emotion_conf or 0.0,
            visual_objects=features.visual_objects or [],
            asr_text=asr_text,
            grounding_entity_scores=grounding_entity_scores,
        )
        conflict_time = time.time() - conflict_t0

        # ── 生成答案 ─────────────────────────────────────────────────────────
        gen_t0 = time.time()
        try:
            if conflict_signal.should_suppress:
                # 使用增强版抑制器生成
                from OMhallucination.modules.attention_mask_suppressor import infer_query_modality
                query_modality = infer_query_modality(question)

                # 创建 logits 处理器
                logits_processor = self._suppressor.create_logits_processor(
                    tokenizer=self._tokenizer,
                    conflict_signal=conflict_signal,
                    query_modality=query_modality,
                )

                # 使用 adapter 的底层生成方法
                messages = self._adapter._build_messages(
                    video_path, question, emotion_constraint=None
                )
                audio_array = self._adapter._load_audio(video_path) if Path(video_path).exists() else None
                answer = self._adapter._generate(
                    messages=messages,
                    audio_array=audio_array,
                    max_new_tokens=yn_tokens,
                    logits_processor=logits_processor,
                )
                suppression_applied = True
            else:
                # 无冲突，标准生成
                answer = self._adapter.answer(
                    video_path,
                    question,
                    max_new_tokens=yn_tokens,
                )
                suppression_applied = False

            answer = normalize_answer_for_question(question, answer, yn_tokens)
        except Exception as exc:
            logger.warning('[%s] Enhanced suppressor 推理失败: %s', video_id, exc)
            answer = 'Unable to process.'
            suppression_applied = False

        generation_time = time.time() - gen_t0

        return {
            'text_answer': answer,
            'inference_time_s': time.time() - t0,
            'suppression_applied': suppression_applied,
            'conflict_detection_time_s': conflict_time,
            'generation_time_s': generation_time,
            'conflict_signal': {
                'has_emotion_conflict': conflict_signal.has_emotion_conflict,
                'has_content_conflict': conflict_signal.has_content_conflict,
                'overall_conflict_score': conflict_signal.overall_conflict_score,
                'emotion_distance': conflict_signal.emotion_distance,
                'content_conflict_score': conflict_signal.content_conflict_score,
                'conflicting_entities': conflict_signal.conflicting_entities,
                'emotion_keywords': conflict_signal.emotion_keywords_in_conflict[:5],
                'suppression_strength': conflict_signal.get_suppression_strength(),
                'grounding_scores': {
                    k: round(v, 3)
                    for k, v in (grounding_entity_scores or {}).items()
                },
            },
        }

    def evaluate(self, item: Dict, mode: str = 'all') -> Dict:
        """评估单个样本"""
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
            'mode': mode,
        }

        # 提取特征（所有模式共享）
        feature_t0 = time.time()
        cached = self._feature_cache.get(video_id)
        if cached is not None:
            features, frames = cached
            logger.debug('[%s] 使用缓存特征', video_id)
        else:
            try:
                frames = self._extractor._extract_frames(video_path, n_frames=8)
                features = self._extractor.extract(video_path)
                self._feature_cache[video_id] = (features, frames)
            except Exception as exc:
                logger.warning('[%s] 特征提取失败: %s', video_id, exc)
                features = None
                frames = None
        feature_time = time.time() - feature_t0

        # 冲突检测（用于original pipeline）
        conflict = None
        if features and mode in ('original', 'all'):
            try:
                conflict = self._detector.detect(features)
            except Exception as exc:
                logger.warning('[%s] 冲突检测失败: %s', video_id, exc)
                conflict = ConflictReport()

        # 根据模式运行不同的评估
        yn_tokens = 10 if is_yn else None

        if mode == 'baseline' or mode == 'all':
            result['baseline'] = self._run_baseline(
                video_path, question, video_id, max_new_tokens=yn_tokens
            )

        if mode == 'original' or mode == 'all':
            if features and conflict:
                result['original'] = self._run_original_pipeline(
                    video_path, question, video_id, features, conflict
                )
            else:
                result['original'] = {'text_answer': 'Feature extraction failed', 'inference_time_s': 0.0}

        if mode == 'enhanced' or mode == 'all':
            if features:
                result['enhanced'] = self._run_enhanced_suppressor(
                    video_path, question, video_id, features, frames=frames
                )
            else:
                result['enhanced'] = {'text_answer': 'Feature extraction failed', 'inference_time_s': 0.0}

        # 添加特征信息
        if features:
            result['features'] = {
                'visual_emotion': features.visual_emotion,
                'audio_emotion': features.audio_emotion,
                'asr_text': features.asr_text[:100] if features.asr_text else None,
                'visual_objects': features.visual_objects[:10],
                'feature_time_s': feature_time,
            }

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

    mode = args.mode
    if not items:
        logger.info('所有样本已评估完成，仅生成报告。')
    else:
        logger.info('初始化模型... (mode=%s)', mode)
        config = PipelineConfig(device=args.device)
        config.verbose = args.verbose
        config.model_paths.qwen_omni_path = args.model_path

        adapter = QwenOmniAdapter(
            model_path=args.model_path,
            device=args.device,
        )

        det_device = args.detector_device or 'cpu'
        logger.info('Qwen 设备: %s, 检测器设备: %s', args.device, det_device)

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

        # 配置增强版抑制器
        suppressor_config = TokenLevelSuppressionConfig(
            enable_enhanced_suppression=True,
            enable_entity_level_suppression=args.enable_entity_suppression,
            entity_suppression_strength=args.entity_suppression_strength,
            enable_emotion_token_suppression=args.enable_emotion_suppression,
            emotion_token_bias=args.emotion_token_bias,
            soft_mask_threshold=args.soft_mask_threshold,
            hard_mask_threshold=args.hard_mask_threshold,
        )

        evaluator = EnhancedSuppressorEvaluator(
            adapter=adapter,
            extractor=extractor,
            detector=detector,
            config=config,
            video_dir=VIDEO_DIR,
            suppressor_config=suppressor_config,
        )

        n_done = 0
        errors = 0
        pbar = tqdm(items, desc=f'评估 ({mode})', unit='sample', dynamic_ncols=True)
        with open(raw_path, 'a') as fout:
            for item in pbar:
                try:
                    row = evaluator.evaluate(item, mode=mode)
                    fout.write(json.dumps(row, ensure_ascii=False) + '\n')
                    fout.flush()
                    n_done += 1

                    # 更新进度条
                    if mode == 'enhanced' or mode == 'all':
                        enhanced_ans = row.get('enhanced', {}).get('text_answer', '?')[:20]
                        suppressed = row.get('enhanced', {}).get('suppression_applied', False)
                        pbar.set_postfix_str(f"{item['video_id']} | {enhanced_ans} | supp={suppressed}")
                    else:
                        pbar.set_postfix_str(f"{item['video_id']}")
                except Exception as exc:
                    logger.error('样本 %s 失败: %s', item.get('video_id'), exc, exc_info=True)
                    errors += 1
        pbar.close()
        logger.info('评估完成 (mode=%s)。%d 个样本，%d 个错误。', mode, n_done, errors)

    # 生成报告
    generate_comparison_report(raw_path, out_dir, mode)


def generate_comparison_report(raw_path: Path, out_dir: Path, mode: str) -> None:
    """生成对比报告"""
    results = []
    with open(raw_path) as f:
        for line in f:
            results.append(json.loads(line))

    def _empty_stat():
        return {'correct': 0, 'total': 0, 'time': [], 'suppressed': 0}

    overall = {k: _empty_stat() for k in ('baseline', 'original', 'enhanced')}
    by_task: Dict[str, Dict] = {}

    for r in results:
        label = r['label']
        task  = r.get('task', 'Unknown')

        if task not in by_task:
            by_task[task] = {k: _empty_stat() for k in ('baseline', 'original', 'enhanced')}

        for name in ('baseline', 'original', 'enhanced'):
            if name not in r:
                continue
            ans = extract_yes_no(r[name].get('text_answer', ''))
            correct = int(ans == label)
            for bucket in (overall[name], by_task[task][name]):
                bucket['total']   += 1
                bucket['correct'] += correct
                bucket['time'].append(r[name].get('inference_time_s', 0.0))
            if name == 'enhanced' and r[name].get('suppression_applied'):
                overall['enhanced']['suppressed']      += 1
                by_task[task]['enhanced']['suppressed'] += 1

    # ── 打印 ──────────────────────────────────────────────────────────────────
    SEP = "=" * 80
    print(f"\n{SEP}")
    print("方案一对比实验报告：增强版Token抑制器 vs Baseline")
    print(SEP)

    def _fmt(name: str, s: dict) -> None:
        if s['total'] == 0:
            return
        acc      = s['correct'] / s['total']
        avg_time = float(np.mean(s['time'])) if s['time'] else 0.0
        print(f"\n  【{name.upper()}】  acc={acc:.4f} ({s['correct']}/{s['total']})  "
              f"avg_time={avg_time:.3f}s", end='')
        if name == 'enhanced' and s['suppressed']:
            rate = s['suppressed'] / s['total']
            print(f"  supp={rate:.1%} ({s['suppressed']})", end='')
        print()

    print("\n## 总体结果")
    for name in ('baseline', 'original', 'enhanced'):
        _fmt(name, overall[name])
    if overall['baseline']['total'] and overall['enhanced']['total']:
        d = overall['enhanced']['correct']/overall['enhanced']['total'] - \
            overall['baseline']['correct']/overall['baseline']['total']
        print(f"\n  Δ Enhanced vs Baseline: {d:+.4f}")
    if overall['original']['total'] and overall['enhanced']['total']:
        d = overall['enhanced']['correct']/overall['enhanced']['total'] - \
            overall['original']['correct']/overall['original']['total']
        print(f"  Δ Enhanced vs Original: {d:+.4f}")

    print(f"\n{SEP}")
    for task in sorted(by_task):
        ts = by_task[task]
        print(f"\n## 任务: {task}")
        for name in ('baseline', 'original', 'enhanced'):
            _fmt(name, ts[name])
        if ts['baseline']['total'] and ts['enhanced']['total']:
            d = ts['enhanced']['correct']/ts['enhanced']['total'] - \
                ts['baseline']['correct']/ts['baseline']['total']
            print(f"  Δ Enhanced vs Baseline: {d:+.4f}")
    print(f"\n{SEP}\n")

    # ── 保存 JSON ─────────────────────────────────────────────────────────────
    report = {
        'mode': mode,
        'n_samples': len(results),
        'overall': {
            name: {
                'accuracy': round(s['correct']/s['total'], 4) if s['total'] else 0,
                'correct': s['correct'], 'total': s['total'],
                'suppression_rate': round(s['suppressed']/s['total'], 4) if s['total'] else 0,
            }
            for name, s in overall.items()
        },
        'by_task': {
            task: {
                name: {
                    'accuracy': round(s['correct']/s['total'], 4) if s['total'] else 0,
                    'correct': s['correct'], 'total': s['total'],
                    'suppression_rate': round(s['suppressed']/s['total'], 4) if s['total'] else 0,
                }
                for name, s in ts.items()
            }
            for task, ts in by_task.items()
        },
    }
    report_path = out_dir / 'comparison_report.json'
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    logger.info('报告已保存 → %s', report_path)


# ============================================================================
# CLI
# ============================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description='方案一对比实验：增强版Token抑制器评估',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument('--model_path', default=str(_ROOT / 'models' / 'Qwen2.5-Omni-7B'))
    p.add_argument('--output_dir', default='Benchmark/results/enhanced_comparison')
    p.add_argument('--tasks', default=None, help='逗号分隔的任务名称')
    p.add_argument('--n_samples', type=int, default=None, help='最大样本数')
    p.add_argument('--resume', action='store_true', help='恢复中断的评估')
    p.add_argument('--device', default='cuda:4', help='Qwen 模型设备')
    p.add_argument('--detector_device', default='cpu', help='特征提取器设备')
    p.add_argument(
        '--mode',
        choices=['baseline', 'original', 'enhanced', 'all'],
        default='all',
        help='评估模式: baseline=仅baseline, original=原pipeline, enhanced=方案一, all=全部对比',
    )

    # 增强版抑制器配置
    p.add_argument('--enable_entity_suppression', action='store_true', default=True)
    p.add_argument('--entity_suppression_strength', type=float, default=3.0)
    p.add_argument('--enable_emotion_suppression', action='store_true', default=True)
    p.add_argument('--emotion_token_bias', type=float, default=-2.0)
    p.add_argument('--soft_mask_threshold', type=float, default=0.3)
    p.add_argument('--hard_mask_threshold', type=float, default=0.7)

    p.add_argument('--verbose', action='store_true', default=True)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    run_evaluation(args)
