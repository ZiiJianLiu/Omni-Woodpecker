"""
Benchmark comparison runner for TRACe-Omni.
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import re
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

from tqdm import tqdm

from ..config import PipelineConfig
from ..cirpo_omni.benchmark_policy import (
    build_answer_space_spec,
    format_question_for_answer_space,
    normalize_answer_for_space,
)
from ..qwen_omni_adapter import QwenOmniAdapter
from .config import TraceOmniConfig
from .pipeline import TraceOmniPipeline

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
AVHBENCH_ROOT = PROJECT_ROOT / 'data' / 'AVHBench'
CMM_ROOT = PROJECT_ROOT / 'data' / 'CMM'
WORLDSENSE_ROOT = PROJECT_ROOT / 'external_datasets' / 'omni_rl_sources' / 'worldsense'
OMNIVIDEOBENCH_ROOT = PROJECT_ROOT / 'external_datasets' / 'omni_rl_sources' / 'omnivideobench'
DEFAULT_AVHBENCH_TASKS = (
    'AV Matching',
    'Video-driven Audio Hallucination',
    'Audio-driven Video Hallucination',
)
GROUP_KEYS = {
    'avhbench': ('task',),
    'cmm': ('modality', 'category', 'sub_category', 'correlation_type'),
    'worldsense': ('domain', 'task_domain', 'task_type', 'audio_class'),
    'omnivideobench': ('question_type', 'audio_type', 'video_type', 'duration'),
}


@dataclass
class BenchmarkSample:
    sample_id: str
    benchmark: str
    question: str
    reference_answer: str
    eval_type: str
    video_path: Optional[str] = None
    audio_path: Optional[str] = None
    options: List[Dict[str, str]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _parse_csv_arg(values: Optional[Sequence[str]]) -> Optional[List[str]]:
    if not values:
        return None
    parsed: List[str] = []
    for value in values:
        for piece in str(value).split(','):
            normalized = piece.strip()
            if normalized:
                parsed.append(normalized)
    return parsed or None


def _normalize_yes_no(text: Optional[str]) -> Optional[str]:
    if text is None:
        return None
    normalized = str(text).strip()
    if not normalized:
        return None
    match = re.match(r'^\s*["\'(\[]*(yes|no)\b', normalized, flags=re.IGNORECASE)
    if not match:
        return None
    return match.group(1).capitalize()


def _parse_choice_option(option: Any) -> Optional[Dict[str, str]]:
    text = str(option or '').strip()
    if not text:
        return None
    match = re.match(r'^\s*([A-Z])[\.\)]\s*(.*)$', text, flags=re.IGNORECASE)
    if match:
        return {'label': match.group(1).upper(), 'text': match.group(2).strip()}
    return {'label': '', 'text': text}


def _format_choice_question(question: str, options: Sequence[Dict[str, str]], eval_type: str) -> str:
    answer_space = build_answer_space_spec(question, eval_type=eval_type, options=options)
    return format_question_for_answer_space(question, answer_space)


def _normalize_for_sample(answer: Optional[str], sample: BenchmarkSample) -> Optional[str]:
    if sample.eval_type == 'yes_no':
        return _normalize_yes_no(answer)
    if sample.eval_type in {'single_choice', 'multi_select'}:
        answer_space = build_answer_space_spec(sample.question, eval_type=sample.eval_type, options=sample.options)
        return normalize_answer_for_space(answer, answer_space)
    return None


def _resolve_relative_path(root: Path, value: Optional[str]) -> Optional[str]:
    if not value:
        return None
    candidate = (root / value).resolve() if not Path(value).is_absolute() else Path(value)
    return str(candidate)


def _slice_samples(
    samples: List[BenchmarkSample],
    *,
    n_samples: Optional[int],
    start_index: int,
    shuffle: bool,
    seed: int,
) -> List[BenchmarkSample]:
    selected = list(samples)
    if shuffle:
        rng = random.Random(seed)
        rng.shuffle(selected)
    if start_index > 0:
        selected = selected[start_index:]
    if n_samples is not None:
        selected = selected[:n_samples]
    return selected


def load_avhbench_samples(
    *,
    benchmark_root: Path = AVHBENCH_ROOT,
    tasks: Optional[Sequence[str]] = None,
    include_free_form: bool = False,
    n_samples: Optional[int] = None,
    start_index: int = 0,
    shuffle: bool = False,
    seed: int = 0,
) -> List[BenchmarkSample]:
    task_filter = set(_parse_csv_arg(tasks) or DEFAULT_AVHBENCH_TASKS)
    data = json.loads((benchmark_root / 'QA.json').read_text(encoding='utf-8'))
    samples: List[BenchmarkSample] = []
    for index, item in enumerate(data):
        task = str(item.get('task') or '').strip()
        if task not in task_filter:
            continue
        reference = str(item.get('label') or '').strip()
        eval_type = 'yes_no' if _normalize_yes_no(reference) is not None else 'free_form'
        if eval_type != 'yes_no' and not include_free_form:
            continue
        video_id = str(item.get('video_id') or '').strip()
        video_path = benchmark_root / 'videos' / f'{video_id}.mp4'
        samples.append(
            BenchmarkSample(
                sample_id=f'avhbench:{index:05d}:{video_id}',
                benchmark='avhbench',
                question=str(item.get('text') or '').strip(),
                reference_answer=reference,
                eval_type=eval_type,
                video_path=str(video_path),
                audio_path=None,
                metadata={
                    'task': task,
                    'video_id': video_id,
                },
            )
        )
    return _slice_samples(
        samples,
        n_samples=n_samples,
        start_index=start_index,
        shuffle=shuffle,
        seed=seed,
    )


def load_cmm_samples(
    *,
    benchmark_root: Path = CMM_ROOT,
    modalities: Optional[Sequence[str]] = None,
    categories: Optional[Sequence[str]] = None,
    n_samples: Optional[int] = None,
    start_index: int = 0,
    shuffle: bool = False,
    seed: int = 0,
) -> List[BenchmarkSample]:
    modality_filter = set(_parse_csv_arg(modalities) or [])
    category_filter = set(_parse_csv_arg(categories) or [])
    data = json.loads((benchmark_root / 'all_data_final_reorg.json').read_text(encoding='utf-8'))
    samples: List[BenchmarkSample] = []
    for index, item in enumerate(data):
        modality = str(item.get('modality') or '').strip()
        category = str(item.get('category') or '').strip()
        if modality_filter and modality not in modality_filter:
            continue
        if category_filter and category not in category_filter:
            continue
        video_path = _resolve_relative_path(benchmark_root, item.get('video_path'))
        audio_path = _resolve_relative_path(benchmark_root, item.get('audio_path'))
        samples.append(
            BenchmarkSample(
                sample_id=f'cmm:{index:05d}',
                benchmark='cmm',
                question=str(item.get('question') or '').strip(),
                reference_answer=str(item.get('answer') or '').strip(),
                eval_type='yes_no',
                video_path=video_path,
                audio_path=audio_path,
                metadata={
                    'category': category,
                    'sub_category': str(item.get('sub_category') or '').strip(),
                    'modality': modality,
                    'granularity': str(item.get('granularity') or '').strip(),
                    'correlation_type': str(item.get('correlation_type') or '').strip(),
                },
            )
        )
    return _slice_samples(
        samples,
        n_samples=n_samples,
        start_index=start_index,
        shuffle=shuffle,
        seed=seed,
    )


def load_worldsense_samples(
    *,
    benchmark_root: Path = WORLDSENSE_ROOT,
    n_samples: Optional[int] = None,
    start_index: int = 0,
    shuffle: bool = False,
    seed: int = 0,
) -> List[BenchmarkSample]:
    qa_path = benchmark_root / 'worldsense_qa.json'
    if not qa_path.exists():
        logger.info('WorldSense assets are not bundled; skipping this benchmark.')
        return []
    data = json.loads(qa_path.read_text(encoding='utf-8'))
    samples: List[BenchmarkSample] = []
    for video_key, item in data.items():
        video_id = str(item.get('video_id') or video_key).strip()
        video_path = _resolve_relative_path(benchmark_root, f'videos/{video_id}.mp4')
        audio_class = item.get('audio_class') or []
        audio_class_text = ','.join(str(value) for value in audio_class) if isinstance(audio_class, list) else str(audio_class)
        for task_key, task in item.items():
            if not (str(task_key).startswith('task') and isinstance(task, dict) and task.get('question')):
                continue
            options = [_parse_choice_option(option) for option in task.get('candidates') or []]
            clean_options = [option for option in options if option is not None]
            question = _format_choice_question(str(task.get('question') or '').strip(), clean_options, 'single_choice')
            samples.append(
                BenchmarkSample(
                    sample_id=f'worldsense:{video_id}:{task_key}',
                    benchmark='worldsense',
                    question=question,
                    reference_answer=str(task.get('answer') or '').strip(),
                    eval_type='single_choice',
                    video_path=video_path,
                    audio_path=None,
                    options=clean_options,
                    metadata={
                        'video_id': video_id,
                        'duration': str(item.get('duration') or '').strip(),
                        'video_duration': str(item.get('video_duration') or '').strip(),
                        'domain': str(item.get('domain') or '').strip(),
                        'sub_category': str(item.get('sub_category') or '').strip(),
                        'audio_class': audio_class_text,
                        'task_key': task_key,
                        'task_domain': str(task.get('task_domain') or '').strip(),
                        'task_type': str(task.get('task_type') or '').strip(),
                    },
                )
            )
    return _slice_samples(samples, n_samples=n_samples, start_index=start_index, shuffle=shuffle, seed=seed)


def load_omnivideobench_samples(
    *,
    benchmark_root: Path = OMNIVIDEOBENCH_ROOT,
    n_samples: Optional[int] = None,
    start_index: int = 0,
    shuffle: bool = False,
    seed: int = 0,
) -> List[BenchmarkSample]:
    jsonl_path = benchmark_root / 'data.jsonl'
    if jsonl_path.exists():
        data = load_jsonl(jsonl_path)
    else:
        parquet_path = benchmark_root / 'data.parquet'
        if not parquet_path.exists():
            logger.info('OmniVideoBench assets are not bundled; skipping this benchmark.')
            return []
        try:
            import pandas as pd
        except ImportError as exc:
            raise RuntimeError(
                'pandas is required to load OmniVideoBench parquet data when data.jsonl is absent'
            ) from exc
        data = pd.read_parquet(parquet_path).to_dict(orient='records')
    samples: List[BenchmarkSample] = []
    for index, item in enumerate(data):
        raw_options = item.get('options')
        if raw_options is None:
            raw_options = []
        elif not isinstance(raw_options, list):
            raw_options = list(raw_options)
        clean_options = [_parse_choice_option(option) for option in raw_options]
        options = [option for option in clean_options if option is not None]
        question = _format_choice_question(str(item.get('question') or '').strip(), options, 'single_choice')
        video_rel = str(item.get('video') or '').strip()
        samples.append(
            BenchmarkSample(
                sample_id=f'omnivideobench:{index:05d}',
                benchmark='omnivideobench',
                question=question,
                reference_answer=str(item.get('correct_option') or item.get('answer') or '').strip(),
                eval_type='single_choice',
                video_path=_resolve_relative_path(benchmark_root, video_rel),
                audio_path=None,
                options=options,
                metadata={
                    'question_type': str(item.get('question_type') or '').strip(),
                    'audio_type': str(item.get('audio_type') or '').strip(),
                    'video_type': str(item.get('video_type') or '').strip(),
                    'duration': str(item.get('duration') or '').strip(),
                    'video': video_rel,
                },
            )
        )
    return _slice_samples(samples, n_samples=n_samples, start_index=start_index, shuffle=shuffle, seed=seed)


def _safe_accuracy(correct: int, total: int) -> Optional[float]:
    if total <= 0:
        return None
    return correct / float(total)


def _compute_method_metrics(records: Iterable[Dict[str, Any]], method_key: str) -> Dict[str, Any]:
    scored = [
        record for record in records
        if record.get('eval_type') in {'yes_no', 'single_choice', 'multi_select'} and record.get('status') == 'ok'
    ]
    total = len(scored)
    valid = 0
    correct = 0
    errors = 0
    for record in scored:
        payload = record.get(method_key) or {}
        if payload.get('error'):
            errors += 1
        prediction = payload.get('normalized_answer')
        if prediction is None:
            prediction = payload.get('normalized_yes_no')
        if prediction:
            valid += 1
        if payload.get('correct') is True:
            correct += 1
    return {
        'n_scored': total,
        'n_valid_predictions': valid,
        'n_correct': correct,
        'n_errors': errors,
        'accuracy': _safe_accuracy(correct, total),
        'valid_prediction_rate': _safe_accuracy(valid, total),
    }


def _group_breakdown(records: List[Dict[str, Any]], method_keys: Sequence[str], benchmark: str) -> Dict[str, Any]:
    breakdown: Dict[str, Any] = {}
    for group_key in GROUP_KEYS.get(benchmark, ()):
        grouped: Dict[str, List[Dict[str, Any]]] = {}
        for record in records:
            value = str((record.get('metadata') or {}).get(group_key) or 'unknown')
            grouped.setdefault(value, []).append(record)
        breakdown[group_key] = {}
        for value, group_records in grouped.items():
            breakdown[group_key][value] = {
                method_key: _compute_method_metrics(group_records, method_key)
                for method_key in method_keys
            }
    return breakdown


def summarize_records(records: List[Dict[str, Any]], *, benchmark: str) -> Dict[str, Any]:
    baseline_metrics = _compute_method_metrics(records, 'baseline')
    trace_metrics = _compute_method_metrics(records, 'trace_omni')
    improved = 0
    regressed = 0
    for record in records:
        if record.get('eval_type') not in {'yes_no', 'single_choice', 'multi_select'}:
            continue
        baseline_correct = bool((record.get('baseline') or {}).get('correct'))
        trace_correct = bool((record.get('trace_omni') or {}).get('correct'))
        if trace_correct and not baseline_correct:
            improved += 1
        if baseline_correct and not trace_correct:
            regressed += 1

    return {
        'benchmark': benchmark,
        'n_records': len(records),
        'n_invalid_samples': sum(1 for record in records if record.get('status') != 'ok'),
        'n_yes_no_records': sum(
            1 for record in records
            if record.get('eval_type') == 'yes_no' and record.get('status') == 'ok'
        ),
        'baseline': baseline_metrics,
        'trace_omni': trace_metrics,
        'delta_accuracy': None
        if baseline_metrics.get('accuracy') is None or trace_metrics.get('accuracy') is None
        else trace_metrics['accuracy'] - baseline_metrics['accuracy'],
        'n_improved_over_baseline': improved,
        'n_regressed_from_baseline': regressed,
        'breakdown': _group_breakdown(records, ('baseline', 'trace_omni'), benchmark),
    }


class TraceOmniBenchmarkRunner:
    def __init__(
        self,
        *,
        benchmark: str,
        model_path: str,
        device: str,
        output_dir: Path,
        probe_mode: str,
        max_new_tokens: Optional[int],
    ):
        self.benchmark = benchmark
        self.model_path = model_path
        self.device = device
        self.output_dir = output_dir
        self.raw_path = self.output_dir / 'raw_results.jsonl'
        self.summary_path = self.output_dir / 'summary.json'
        self.max_new_tokens = max_new_tokens

        base_config = PipelineConfig()
        trace_config = TraceOmniConfig.from_pipeline_config(base_config)
        trace_config.device = device
        trace_config.model_paths.qwen_omni_path = model_path
        trace_config.probe.mode = probe_mode
        if max_new_tokens is not None:
            trace_config.generation_max_new_tokens = max_new_tokens

        self.adapter = QwenOmniAdapter(
            model_path=model_path,
            device=device,
            max_new_tokens=max_new_tokens or trace_config.generation_max_new_tokens,
        )
        self.pipeline = TraceOmniPipeline(
            config=trace_config,
            adapter=self.adapter,
        )

    def _token_budget(self, sample: BenchmarkSample) -> Optional[int]:
        if self.max_new_tokens is not None:
            return self.max_new_tokens
        if sample.eval_type == 'yes_no':
            return 8
        return self.pipeline.config.generation_max_new_tokens

    def _run_baseline(self, sample: BenchmarkSample) -> Dict[str, Any]:
        token_budget = self._token_budget(sample)
        start = time.time()
        answer = self.adapter.answer_media(
            sample.question,
            video_path=sample.video_path,
            audio_path=sample.audio_path,
            max_new_tokens=token_budget,
        )
        latency = time.time() - start
        normalized = _normalize_for_sample(answer, sample)
        reference = _normalize_for_sample(sample.reference_answer, sample)
        return {
            'text': answer,
            'normalized_answer': normalized,
            'normalized_yes_no': normalized,
            'correct': normalized == reference if reference is not None else None,
            'latency_s': latency,
            'max_new_tokens': token_budget,
        }

    def _run_trace_omni(self, sample: BenchmarkSample) -> Dict[str, Any]:
        token_budget = self._token_budget(sample)
        start = time.time()
        result = self.pipeline.run_media(
            sample.question,
            video_path=sample.video_path,
            audio_path=sample.audio_path,
            max_new_tokens=token_budget,
        )
        latency = time.time() - start
        final_text = result.realized_answer.text
        normalized = _normalize_for_sample(final_text, sample)
        reference = _normalize_for_sample(sample.reference_answer, sample)
        return {
            'text': final_text,
            'draft_answer': result.draft_answer,
            'normalized_answer': normalized,
            'normalized_yes_no': normalized,
            'correct': normalized == reference if reference is not None else None,
            'latency_s': latency,
            'strategy': result.realized_answer.strategy,
            'max_new_tokens': token_budget,
            'n_claims': len(result.claims),
            'runtime_metadata': dict(result.runtime_metadata),
            'feature_metadata': dict(result.feature_metadata),
            'claims': [claim.to_dict() for claim in result.claims],
            'claim_scores': [score.to_dict() for score in result.claim_scores],
        }

    @staticmethod
    def _validate_sample(sample: BenchmarkSample) -> Optional[str]:
        if not sample.video_path and not sample.audio_path:
            return 'missing_media'
        if sample.video_path and not Path(sample.video_path).exists():
            return f'missing_video:{sample.video_path}'
        if sample.audio_path and not Path(sample.audio_path).exists():
            return f'missing_audio:{sample.audio_path}'
        return None

    def _processed_ids(self) -> set[str]:
        if not self.raw_path.exists():
            return set()
        processed: set[str] = set()
        with self.raw_path.open('r', encoding='utf-8') as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    payload = json.loads(line)
                except json.JSONDecodeError:
                    continue
                sample_id = str(payload.get('sample_id') or '').strip()
                if sample_id:
                    processed.add(sample_id)
        return processed

    def run(self, samples: List[BenchmarkSample], *, resume: bool = False) -> Dict[str, Any]:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        processed_ids = self._processed_ids() if resume else set()
        mode = 'a' if resume and self.raw_path.exists() else 'w'

        with self.raw_path.open(mode, encoding='utf-8') as writer:
            for sample in tqdm(samples, desc=f'{self.benchmark} comparison'):
                if sample.sample_id in processed_ids:
                    continue

                record: Dict[str, Any] = {
                    **sample.to_dict(),
                    'reference_answer_normalized': _normalize_for_sample(sample.reference_answer, sample),
                    'reference_yes_no': _normalize_yes_no(sample.reference_answer),
                    'status': 'ok',
                }
                validation_error = self._validate_sample(sample)
                if validation_error is not None:
                    record['status'] = 'invalid_sample'
                    record['error'] = validation_error
                    writer.write(json.dumps(record, ensure_ascii=False) + '\n')
                    writer.flush()
                    continue

                try:
                    record['baseline'] = self._run_baseline(sample)
                except Exception as exc:
                    logger.exception('Baseline failed for %s', sample.sample_id)
                    record['baseline'] = {
                        'text': None,
                        'normalized_answer': None,
                        'normalized_yes_no': None,
                        'correct': False if sample.eval_type in {'yes_no', 'single_choice', 'multi_select'} else None,
                        'latency_s': None,
                        'error': f'{type(exc).__name__}: {exc}',
                    }

                try:
                    record['trace_omni'] = self._run_trace_omni(sample)
                except Exception as exc:
                    logger.exception('TRACe-Omni failed for %s', sample.sample_id)
                    record['trace_omni'] = {
                        'text': None,
                        'draft_answer': None,
                        'normalized_answer': None,
                        'normalized_yes_no': None,
                        'correct': False if sample.eval_type in {'yes_no', 'single_choice', 'multi_select'} else None,
                        'latency_s': None,
                        'error': f'{type(exc).__name__}: {exc}',
                    }

                writer.write(json.dumps(record, ensure_ascii=False) + '\n')
                writer.flush()

        records = load_jsonl(self.raw_path)
        summary = summarize_records(records, benchmark=self.benchmark)
        self.summary_path.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + '\n',
            encoding='utf-8',
        )
        return summary


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    records: List[Dict[str, Any]] = []
    with path.open('r', encoding='utf-8') as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Compare raw Qwen2.5-Omni against TRACe-Omni on supported omni benchmarks.')
    parser.add_argument('--benchmark', choices=['avhbench', 'cmm', 'worldsense', 'omnivideobench'], required=True)
    parser.add_argument('--output_dir', required=True, help='Directory for raw_results.jsonl and summary.json.')
    parser.add_argument(
        '--model_path',
        default=str(Path(__file__).resolve().parents[1] / 'models' / 'Qwen2.5-Omni-7B'),
    )
    parser.add_argument('--device', default='auto')
    parser.add_argument('--probe_mode', choices=['heuristic', 'branch_contrastive'], default='branch_contrastive')
    parser.add_argument('--max_new_tokens', type=int, default=None)
    parser.add_argument('--n_samples', type=int, default=None)
    parser.add_argument('--start_index', type=int, default=0)
    parser.add_argument('--shuffle', action='store_true')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--include_free_form', action='store_true')
    parser.add_argument('--tasks', nargs='*', default=None, help='AVHBench task filter.')
    parser.add_argument('--modalities', nargs='*', default=None, help='CMM modality filter.')
    parser.add_argument('--categories', nargs='*', default=None, help='CMM category filter.')
    return parser


def main() -> None:
    args = build_argparser().parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(name)s - %(message)s',
    )

    if args.benchmark == 'avhbench':
        samples = load_avhbench_samples(
            tasks=args.tasks,
            include_free_form=args.include_free_form,
            n_samples=args.n_samples,
            start_index=args.start_index,
            shuffle=args.shuffle,
            seed=args.seed,
        )
    else:
        if args.benchmark == 'worldsense':
            samples = load_worldsense_samples(
                n_samples=args.n_samples,
                start_index=args.start_index,
                shuffle=args.shuffle,
                seed=args.seed,
            )
        elif args.benchmark == 'omnivideobench':
            samples = load_omnivideobench_samples(
                n_samples=args.n_samples,
                start_index=args.start_index,
                shuffle=args.shuffle,
                seed=args.seed,
            )
        else:
            samples = load_cmm_samples(
                modalities=args.modalities,
                categories=args.categories,
                n_samples=args.n_samples,
                start_index=args.start_index,
                shuffle=args.shuffle,
                seed=args.seed,
            )

    logger.info('Loaded %d samples for %s', len(samples), args.benchmark)
    runner = TraceOmniBenchmarkRunner(
        benchmark=args.benchmark,
        model_path=args.model_path,
        device=args.device,
        output_dir=Path(args.output_dir),
        probe_mode=args.probe_mode,
        max_new_tokens=args.max_new_tokens,
    )
    summary = runner.run(samples, resume=args.resume)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
