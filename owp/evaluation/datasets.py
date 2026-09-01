"""Benchmark sample schemas and loaders used by the OWP evaluators."""

from __future__ import annotations

import json
import logging
import random
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence
from .answer_policy import (
    build_answer_space_spec,
    format_question_for_answer_space,
    normalize_answer_for_space,
)

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
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
