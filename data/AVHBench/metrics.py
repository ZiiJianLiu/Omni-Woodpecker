"""
Benchmark/metrics.py
====================
多模态幻觉抑制 Benchmark 评估指标

指标：
  Yes/No 任务   : Accuracy, F1
  Captioning    : BLEU-4, ROUGE-L, METEOR, Semantic Similarity
  冲突检测      : 情感冲突检出率, 内容冲突检出率
  抑制效果      : 修正成功率, 准确率提升 Δ, 按策略分析
"""
from __future__ import annotations

import re
import string
from collections import defaultdict
from typing import Dict, List

import numpy as np


# ---------------------------------------------------------------------------
# Yes / No answer extraction
# ---------------------------------------------------------------------------

_YES_RE = re.compile(
    r"\b(yes|yeah|yep|correct|right|true|affirmative|indeed|certainly|absolutely)\b",
    re.IGNORECASE,
)
_NO_RE = re.compile(
    r"\b(no|nah|nope|incorrect|wrong|false|negative|never)\b",
    re.IGNORECASE,
)


def extract_yes_no(model_output: str) -> str:
    """Parse model response into 'Yes', 'No', or 'Unknown'."""
    text = model_output.strip()
    first = text.split()[0].strip(string.punctuation).lower() if text else ""
    if first in ("yes", "yeah", "yep"):
        return "Yes"
    if first in ("no", "nah", "nope"):
        return "No"
    yes_n = len(_YES_RE.findall(text))
    no_n  = len(_NO_RE.findall(text))
    if yes_n > no_n:
        return "Yes"
    if no_n > yes_n:
        return "No"
    return "Unknown"


def yn_accuracy(predictions: List[str], labels: List[str]) -> float:
    correct = total = 0
    for pred, gt in zip(predictions, labels):
        if pred == "Unknown":
            continue
        total += 1
        correct += int(pred == gt)
    return correct / total if total else 0.0


def yn_f1(
    predictions: List[str], labels: List[str], positive: str = "Yes",
) -> float:
    tp = fp = fn = 0
    for pred, gt in zip(predictions, labels):
        p, g = pred == positive, gt == positive
        if p and g:       tp += 1
        elif p and not g: fp += 1
        elif not p and g: fn += 1
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec  = tp / (tp + fn) if (tp + fn) else 0.0
    return 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0


# ---------------------------------------------------------------------------
# Caption metrics
# ---------------------------------------------------------------------------

def compute_bleu4(hypotheses: List[str], references: List[str]) -> float:
    try:
        from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
        refs = [[r.split()] for r in references]
        hyps = [h.split()   for h in hypotheses]
        return corpus_bleu(
            refs, hyps, weights=(0.25,) * 4,
            smoothing_function=SmoothingFunction().method1,
        )
    except Exception:
        return float("nan")


def compute_rouge_l(hypotheses: List[str], references: List[str]) -> float:
    try:
        from rouge_score import rouge_scorer
        scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
        return float(np.mean([
            scorer.score(r, h)["rougeL"].fmeasure
            for h, r in zip(hypotheses, references)
        ]))
    except Exception:
        return float("nan")


def compute_meteor(hypotheses: List[str], references: List[str]) -> float:
    try:
        from nltk.translate.meteor_score import meteor_score
        return float(np.mean([
            meteor_score([r.split()], h.split())
            for h, r in zip(hypotheses, references)
        ]))
    except Exception:
        return float("nan")


def compute_semantic_sim(hypotheses: List[str], references: List[str]) -> float:
    try:
        from sentence_transformers import SentenceTransformer, util
        m   = SentenceTransformer("all-MiniLM-L6-v2")
        e_h = m.encode(hypotheses, convert_to_tensor=True, show_progress_bar=False)
        e_r = m.encode(references,  convert_to_tensor=True, show_progress_bar=False)
        return float(util.cos_sim(e_h, e_r).diagonal().mean().item())
    except Exception:
        return float("nan")


def caption_metrics(hypotheses: List[str], references: List[str]) -> Dict[str, float]:
    return {
        "bleu4":        compute_bleu4(hypotheses, references),
        "rouge_l":      compute_rouge_l(hypotheses, references),
        "meteor":       compute_meteor(hypotheses, references),
        "semantic_sim": compute_semantic_sim(hypotheses, references),
    }


# ---------------------------------------------------------------------------
# WER (for reference)
# ---------------------------------------------------------------------------

def compute_wer(reference: str, hypothesis: str) -> float:
    try:
        from jiwer import wer
        return float(wer(reference, hypothesis))
    except ImportError:
        ref_t = reference.lower().split()
        hyp_t = hypothesis.lower().split()
        if not ref_t:
            return 0.0 if not hyp_t else 1.0
        m, n = len(ref_t), len(hyp_t)
        dp = list(range(n + 1))
        for i in range(1, m + 1):
            prev, dp[0] = dp[0], i
            for j in range(1, n + 1):
                t = dp[j]
                dp[j] = (
                    prev if ref_t[i - 1] == hyp_t[j - 1]
                    else 1 + min(prev, dp[j], dp[j - 1])
                )
                prev = t
        return dp[n] / m


def text_audio_score(reference: str, asr_transcript: str) -> float:
    """Combined text-audio consistency: 0.5*(1-WER) + 0.5*semantic_sim."""
    wer_val   = compute_wer(reference, asr_transcript)
    wer_score = max(0.0, 1.0 - wer_val)
    try:
        from sentence_transformers import SentenceTransformer, util
        _m  = SentenceTransformer("all-MiniLM-L6-v2")
        e1  = _m.encode([reference],     convert_to_tensor=True, show_progress_bar=False)
        e2  = _m.encode([asr_transcript], convert_to_tensor=True, show_progress_bar=False)
        sem = float(util.cos_sim(e1, e2).item())
    except Exception:
        sem = wer_score
    return 0.5 * wer_score + 0.5 * max(0.0, sem)


# ---------------------------------------------------------------------------
# Aggregate result container
# ---------------------------------------------------------------------------

def _mean(lst: list) -> float:
    valid = [x for x in lst if x == x]  # drop NaN
    return float(np.mean(valid)) if valid else float("nan")


def _check_answer(answer: str, gt_label: str, task: str) -> bool:
    """检查答案是否正确"""
    if task in (
        "AV Matching",
        "Video-driven Audio Hallucination",
        "Audio-driven Video Hallucination",
    ):
        pred = extract_yes_no(answer)
        return pred == gt_label
    elif task == "AV Captioning":
        return len(answer.strip()) > 10  # 简单启发式
    return False


class BenchmarkResult:
    """收集 per-item 结果并计算聚合统计

    mode:
      "both"     — 结果同时包含 baseline 和 pipeline（默认）
      "pipeline" — 仅包含 pipeline 结果
      "baseline" — 仅包含 baseline 结果
    """

    def __init__(self, mode: str = "both") -> None:
        self.items: List[Dict] = []
        self._mode = mode

    def add(self, item: Dict) -> None:
        self.items.append(item)

    def aggregate(self) -> Dict:
        by_task: Dict[str, List[Dict]] = defaultdict(list)
        for it in self.items:
            by_task[it['task']].append(it)

        has_baseline = self._mode in ('both', 'baseline')
        has_pipeline = self._mode in ('both', 'pipeline')
        out: Dict = {'overall': {}, 'by_task': {}, 'mode': self._mode}

        all_b_correct = 0
        all_p_correct = 0
        all_n = 0
        all_corrected = 0
        all_emotion_conflict = 0
        all_content_conflict = 0
        all_dsav_intervened = 0

        for task, items in by_task.items():
            n = len(items)
            is_yn = items[0]['label'] in ('Yes', 'No')
            tr: Dict = {'n_items': n}

            if has_baseline:
                b_correct = sum(
                    _check_answer(x['baseline']['text_answer'], x['label'], task)
                    for x in items
                )
                tr['baseline_accuracy'] = b_correct / n if n else 0
                all_b_correct += b_correct

                if is_yn:
                    bp = [extract_yes_no(x['baseline']['text_answer']) for x in items]
                    gs = [x['label'] for x in items]
                    tr['baseline_f1'] = yn_f1(bp, gs)

                if task == 'AV Captioning':
                    bh = [x['baseline']['text_answer'] for x in items]
                    rs = [x['label'] for x in items]
                    tr['caption_baseline'] = caption_metrics(bh, rs)

            if has_pipeline:
                p_correct = sum(
                    _check_answer(x['pipeline']['text_answer'], x['label'], task)
                    for x in items
                )
                n_corrected = sum(
                    x['pipeline'].get('correction_applied', False) for x in items
                )
                n_dsav_intervened = sum(
                    bool(
                        x['pipeline'].get('dsav_trigger_count', 0)
                        or x['pipeline'].get('dsav_rollback_count', 0)
                    )
                    for x in items
                )
                tr['pipeline_accuracy'] = p_correct / n if n else 0
                tr['n_corrected'] = n_corrected
                tr['n_dsav_intervened'] = n_dsav_intervened
                tr['avg_speculative_time_s'] = _mean([
                    x['pipeline'].get('speculative_time_s', 0.0) for x in items
                ])
                tr['avg_async_validation_latency_s'] = _mean([
                    x['pipeline'].get('async_validation_latency_s', 0.0) for x in items
                ])
                tr['avg_dsav_triggers'] = _mean([
                    x['pipeline'].get('dsav_trigger_count', 0) for x in items
                ])
                tr['avg_dsav_rollbacks'] = _mean([
                    x['pipeline'].get('dsav_rollback_count', 0) for x in items
                ])
                tr['avg_dsav_rebuilds'] = _mean([
                    x['pipeline'].get('dsav_rebuild_count', 0) for x in items
                ])
                all_p_correct += p_correct
                all_corrected += n_corrected
                all_dsav_intervened += n_dsav_intervened

                if is_yn:
                    pp = [extract_yes_no(x['pipeline']['text_answer']) for x in items]
                    gs = [x['label'] for x in items]
                    tr['pipeline_f1'] = yn_f1(pp, gs)

                if task == 'AV Captioning':
                    ph = [x['pipeline']['text_answer'] for x in items]
                    rs = [x['label'] for x in items]
                    tr['caption_pipeline'] = caption_metrics(ph, rs)

                n_emo = sum(
                    x.get('conflict_detection', {}).get('audio_video_emotion_conflict', False)
                    for x in items
                )
                n_cnt = sum(
                    x.get('conflict_detection', {}).get('audio_video_content_conflict', False)
                    for x in items
                )
                tr['emotion_conflict_count'] = n_emo
                tr['content_conflict_count'] = n_cnt
                all_emotion_conflict += n_emo
                all_content_conflict += n_cnt

            if has_baseline and has_pipeline:
                tr['accuracy_delta'] = tr['pipeline_accuracy'] - tr['baseline_accuracy']

            out['by_task'][task] = tr
            all_n += n

        overall: Dict = {'n_items': all_n}
        if has_baseline:
            overall['baseline_accuracy'] = all_b_correct / all_n if all_n else 0
        if has_pipeline:
            overall['pipeline_accuracy'] = all_p_correct / all_n if all_n else 0
            overall['n_corrected'] = all_corrected
            overall['correction_rate'] = all_corrected / all_n if all_n else 0
            overall['emotion_conflict_rate'] = all_emotion_conflict / all_n if all_n else 0
            overall['content_conflict_rate'] = all_content_conflict / all_n if all_n else 0
            overall['dsav_intervention_rate'] = all_dsav_intervened / all_n if all_n else 0
            overall['avg_speculative_time_s'] = _mean([
                x.get('pipeline', {}).get('speculative_time_s', 0.0)
                for x in self.items
            ])
            overall['avg_async_validation_latency_s'] = _mean([
                x.get('pipeline', {}).get('async_validation_latency_s', 0.0)
                for x in self.items
            ])
            overall['avg_dsav_triggers'] = _mean([
                x.get('pipeline', {}).get('dsav_trigger_count', 0)
                for x in self.items
            ])
            overall['avg_dsav_rollbacks'] = _mean([
                x.get('pipeline', {}).get('dsav_rollback_count', 0)
                for x in self.items
            ])
            overall['avg_dsav_rebuilds'] = _mean([
                x.get('pipeline', {}).get('dsav_rebuild_count', 0)
                for x in self.items
            ])
        if has_baseline and has_pipeline:
            overall['accuracy_delta'] = overall['pipeline_accuracy'] - overall['baseline_accuracy']
        out['overall'] = overall

        if has_pipeline and has_baseline:
            strategy_stats: Dict[str, Dict] = defaultdict(
                lambda: {'count': 0, 'improved': 0, 'degraded': 0, 'unchanged': 0}
            )
            for it in self.items:
                corr_type = it['pipeline'].get('correction_type', 'none')
                if corr_type == 'none':
                    continue
                stats = strategy_stats[corr_type]
                stats['count'] += 1
                b_ok = _check_answer(it['baseline']['text_answer'], it['label'], it['task'])
                p_ok = _check_answer(it['pipeline']['text_answer'], it['label'], it['task'])
                if p_ok and not b_ok:
                    stats['improved'] += 1
                elif not p_ok and b_ok:
                    stats['degraded'] += 1
                else:
                    stats['unchanged'] += 1
            out['correction_breakdown'] = dict(strategy_stats)

        return out


# ---------------------------------------------------------------------------
# Pretty-print report
# ---------------------------------------------------------------------------

def print_report(agg: Dict, mode: str = 'both') -> None:
    sep = '─' * 72
    sep2 = '═' * 72

    print()
    print(sep2)
    print('  多模态幻觉抑制 Benchmark 评估报告')
    if mode != 'both':
        print(f'  模式: {mode}')
    print(sep2)

    ov = agg['overall']
    n = ov['n_items']

    if mode == 'both':
        def row(name: str, b: float, p: float) -> None:
            delta = p - b
            arrow = '▲' if delta >= 0 else '▼'
            print(f'  {name:<30} {b:>10.4f} {p:>10.4f}  {arrow}{abs(delta):.4f}')

        print()
        print(f"  {'总体结果':^68}")
        print(f"  {'指标':<30} {'Baseline':>10} {'Pipeline':>10}  {'Delta':>8}")
        print('  ' + sep)
        row('Accuracy', ov['baseline_accuracy'], ov['pipeline_accuracy'])
        print('  ' + sep)
        print(f"  样本数: {n}   修正: {ov['n_corrected']} ({100 * ov['n_corrected'] / n:.1f}%)")
        print(
            f"  情感冲突: {ov['emotion_conflict_rate']:.1%}   内容冲突: {ov['content_conflict_rate']:.1%}   "
            f"DSAV 介入: {ov.get('dsav_intervention_rate', 0.0):.1%}"
        )
        print(
            f"  DSAV 平均触发: {ov.get('avg_dsav_triggers', 0.0):.2f}   "
            f"平均回溯: {ov.get('avg_dsav_rollbacks', 0.0):.2f}   "
            f"平均验证时延: {ov.get('avg_async_validation_latency_s', 0.0):.3f}s"
        )

        for task, tr in agg['by_task'].items():
            print()
            print(f'  {sep}')
            print(f'  任务: {task}')
            print(f'  {sep}')
            row('Accuracy', tr['baseline_accuracy'], tr['pipeline_accuracy'])
            if 'baseline_f1' in tr:
                row('F1', tr['baseline_f1'], tr['pipeline_f1'])
            if 'caption_baseline' in tr:
                for metric in ('bleu4', 'rouge_l', 'meteor', 'semantic_sim'):
                    row(metric.upper(), tr['caption_baseline'][metric], tr['caption_pipeline'][metric])
            print(
                f"  样本: {tr['n_items']}  修正: {tr.get('n_corrected', 0)}  DSAV 介入: {tr.get('n_dsav_intervened', 0)}  "
                f"情感冲突: {tr.get('emotion_conflict_count', 0)}  内容冲突: {tr.get('content_conflict_count', 0)}"
            )
            print(
                f"  DSAV 平均触发: {tr.get('avg_dsav_triggers', 0.0):.2f}  平均回溯: {tr.get('avg_dsav_rollbacks', 0.0):.2f}  "
                f"平均验证时延: {tr.get('avg_async_validation_latency_s', 0.0):.3f}s"
            )

    elif mode == 'baseline':
        print()
        print(f"  {'总体结果 (Baseline)':^68}")
        print(f"  {'指标':<30} {'Baseline':>10}")
        print('  ' + sep)
        print(f"  {'Accuracy':<30} {ov['baseline_accuracy']:>10.4f}")
        print('  ' + sep)
        print(f'  样本数: {n}')

        for task, tr in agg['by_task'].items():
            print()
            print(f'  {sep}')
            print(f'  任务: {task}')
            print(f'  {sep}')
            print(f"  {'Accuracy':<30} {tr['baseline_accuracy']:>10.4f}")
            if 'baseline_f1' in tr:
                print(f"  {'F1':<30} {tr['baseline_f1']:>10.4f}")
            print(f"  样本: {tr['n_items']}")

    elif mode == 'pipeline':
        print()
        print(f"  {'总体结果 (Pipeline)':^68}")
        print(f"  {'指标':<30} {'Pipeline':>10}")
        print('  ' + sep)
        print(f"  {'Accuracy':<30} {ov['pipeline_accuracy']:>10.4f}")
        print('  ' + sep)
        if n:
            print(f"  样本数: {n}   修正: {ov.get('n_corrected', 0)} ({100 * ov.get('n_corrected', 0) / n:.1f}%)")
        else:
            print(f'  样本数: {n}')
        print(
            f"  情感冲突: {ov.get('emotion_conflict_rate', 0):.1%}   内容冲突: {ov.get('content_conflict_rate', 0):.1%}   "
            f"DSAV 介入: {ov.get('dsav_intervention_rate', 0.0):.1%}"
        )
        print(
            f"  DSAV 平均触发: {ov.get('avg_dsav_triggers', 0.0):.2f}   平均回溯: {ov.get('avg_dsav_rollbacks', 0.0):.2f}   "
            f"平均验证时延: {ov.get('avg_async_validation_latency_s', 0.0):.3f}s"
        )

        for task, tr in agg['by_task'].items():
            print()
            print(f'  {sep}')
            print(f'  任务: {task}')
            print(f'  {sep}')
            print(f"  {'Accuracy':<30} {tr['pipeline_accuracy']:>10.4f}")
            if 'pipeline_f1' in tr:
                print(f"  {'F1':<30} {tr['pipeline_f1']:>10.4f}")
            print(
                f"  样本: {tr['n_items']}  修正: {tr.get('n_corrected', 0)}  DSAV 介入: {tr.get('n_dsav_intervened', 0)}  "
                f"情感冲突: {tr.get('emotion_conflict_count', 0)}  内容冲突: {tr.get('content_conflict_count', 0)}"
            )
            print(
                f"  DSAV 平均触发: {tr.get('avg_dsav_triggers', 0.0):.2f}  平均回溯: {tr.get('avg_dsav_rollbacks', 0.0):.2f}  "
                f"平均验证时延: {tr.get('avg_async_validation_latency_s', 0.0):.3f}s"
            )

    if agg.get('correction_breakdown'):
        print()
        print(f'  {sep}')
        print('  修正策略分析')
        print(f'  {sep}')
        print(f"  {'策略':<25} {'次数':>6} {'改进':>6} {'退化':>6} {'无变化':>8}")
        for strategy, stats in agg['correction_breakdown'].items():
            print(
                f"  {strategy:<25} {stats['count']:>6} {stats['improved']:>6} {stats['degraded']:>6} {stats['unchanged']:>8}"
            )

    print()
    print(sep2)
    print()
