#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import torch
from tqdm.auto import tqdm


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
PACKAGE_ROOT = ROOT.parent
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from OMhallucination.qwen_omni_adapter import QwenOmniAdapter
from OMhallucination.trace_omni.benchmark_runner import (
    load_avhbench_samples,
    load_cmm_samples,
    load_omnivideobench_samples,
    load_worldsense_samples,
)
from cirpo_omni.benchmark_policy import (
    AnswerSpaceSpec,
    build_query_latent_state,
    format_question_for_answer_space,
    normalize_answer_for_space,
)
from avcd_qwen_omni_common import run_avcd_dominance_reader, tokenizer_ids
from mad_qwen_omni_common import (
    MODALITY_QUERY_PROMPT,
    question_with_answer_prompt,
    strict_mad_head_only_modality_probe,
)


DEFAULT_RECORDS = (
    ROOT
    / "results"
    / "general_action_policy_benchmark_visual_mismatch_v2_suite_full"
    / "00_base_shared"
    / "base_records.jsonl"
)
DEFAULT_OUTPUT_DIR = ROOT / "results" / "strict_online_prior_typing_probe_v1"
DEFAULT_MODEL_PATH = "Qwen/Qwen2.5-Omni-7B"

_AUDIO_QUERY_CUES = {
    "audio",
    "sound",
    "sounds",
    "hear",
    "heard",
    "hearing",
    "listen",
    "listening",
    "voice",
    "voices",
    "speech",
    "speaking",
    "music",
    "noise",
    "noisy",
    "loud",
    "quiet",
    "audible",
    "singing",
    "talking",
    "saying",
    "shouting",
    "crying",
    "barking",
    "meowing",
    "ringing",
    "engine",
    "siren",
}
_VISUAL_QUERY_CUES = {
    "video",
    "image",
    "visible",
    "visually",
    "scene",
    "look",
    "looks",
    "looking",
    "shown",
    "showing",
    "see",
    "seen",
    "watch",
    "wearing",
    "holding",
    "face",
    "person",
    "people",
    "object",
    "objects",
    "appear",
    "appears",
}
METHOD_SIDE_EVAL_FIELDS = (
    "benchmark",
    "dataset",
    "task_family",
    "reference_answer",
    "eval_reference_answer",
    "eval_baseline_correct",
    "eval_probe_full_correct",
    "eval_target_modality",
    "eval_target_modality_reason",
    "eval_prior_regime_label",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Strict online prior typing probe. "
            "Runtime features may only use question, media, MAD self-assessment, and "
            "full/audio_only/visual_only/text_only branch outputs. "
            "Benchmark task labels are retained only for offline evaluation."
        )
    )
    parser.add_argument("--records", type=Path, default=DEFAULT_RECORDS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--model-path", type=str, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--sample-id", type=str, default="")
    parser.add_argument("--sample-id-file", type=Path, default=None)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--save-every", type=int, default=5)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--skip-on-error",
        action="store_true",
        help="Continue processing and record a skipped row when a single sample fails.",
    )
    parser.add_argument("--only-incorrect", action="store_true")
    parser.add_argument("--benchmarks", nargs="+", default=[])
    parser.add_argument(
        "--task-families",
        nargs="+",
        default=[],
        help=(
            "Deprecated compatibility flag. Ignored for method purity; "
            "runtime row selection no longer uses task_family."
        ),
    )
    parser.add_argument(
        "--target-modality-margin-threshold",
        type=float,
        default=0.12,
        help="MAD-only target typing margin. If audio-vs-visual MAD margin is below this value, fall back to joint/question/branch tie-breaks.",
    )
    parser.add_argument(
        "--source-confidence-gap-threshold",
        type=float,
        default=0.05,
        help="Require the selected source branch confidence on the full answer to exceed the target branch by this amount.",
    )
    parser.add_argument(
        "--source-tie-threshold",
        type=float,
        default=0.03,
        help="If competing cross-modal source scores are within this gap, fall back to no_cross_modal_override.",
    )
    parser.add_argument(
        "--avcd-target-mass-weight",
        type=float,
        default=0.35,
        help="Legacy compatibility knob. Ignored under the strict MAD-target -> AVCD-source strategy.",
    )
    parser.add_argument(
        "--avcd-source-mass-weight",
        type=float,
        default=0.35,
        help="Legacy compatibility knob. Source typing now uses AVCD dominance first and branch outputs only for verification.",
    )
    parser.add_argument(
        "--repair-joint-both-threshold",
        type=float,
        default=0.25,
        help="If MAD both-probability exceeds this value and both unimodal branches strongly support the full answer, abstain from single-target repair typing.",
    )
    parser.add_argument(
        "--repair-joint-support-threshold",
        type=float,
        default=0.4,
        help="Minimum support that both audio-only and visual-only branches must assign to the full answer before repair typing may abstain as joint/ambiguous.",
    )
    parser.add_argument(
        "--repair-joint-tie-threshold",
        type=float,
        default=0.3,
        help="Maximum audio-vs-visual support gap allowed when abstaining as joint/ambiguous.",
    )
    parser.add_argument("--gamma", type=float, default=0.5)
    parser.add_argument("--video-fps", type=float, default=4.0)
    parser.add_argument("--video-max-frames", type=int, default=32)
    parser.add_argument("--video-max-pixels", type=int, default=602112)
    parser.add_argument("--video-min-pixels", type=int, default=100352)
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--no-progress", action="store_true")
    return parser.parse_args()


def safe_text(value: Any) -> str:
    return "" if value is None else str(value).strip()


def maybe_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if number != number or abs(number) == float("inf"):
        return None
    return float(number)


def normalize_yes_no(value: Any) -> Optional[str]:
    text = safe_text(value).lower()
    if text.startswith("yes"):
        return "Yes"
    if text.startswith("no"):
        return "No"
    return None


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def append_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            handle.flush()


def is_oom_error(exc: BaseException) -> bool:
    if isinstance(exc, torch.OutOfMemoryError):
        return True
    text = str(exc).lower()
    return "out of memory" in text or "cuda oom" in text


def cuda_memory_snapshot() -> Dict[str, Any]:
    if not torch.cuda.is_available():
        return {
            "cuda_available": False,
            "device": None,
            "allocated_bytes": 0,
            "reserved_bytes": 0,
            "max_allocated_bytes": 0,
            "max_reserved_bytes": 0,
            "free_bytes": None,
            "total_bytes": None,
        }
    device = torch.cuda.current_device()
    free_bytes, total_bytes = torch.cuda.mem_get_info(device)
    return {
        "cuda_available": True,
        "device": int(device),
        "allocated_bytes": int(torch.cuda.memory_allocated(device)),
        "reserved_bytes": int(torch.cuda.memory_reserved(device)),
        "max_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
        "max_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
        "free_bytes": int(free_bytes),
        "total_bytes": int(total_bytes),
    }


def reset_cuda_peak_memory_stats() -> None:
    if not torch.cuda.is_available():
        return
    device = torch.cuda.current_device()
    try:
        torch.cuda.reset_peak_memory_stats(device)
    except Exception:
        return


def write_oom_event(
    path: Path,
    *,
    sample_id: str,
    stage: str,
    exc: BaseException,
    runtime_memory: Dict[str, Any],
) -> None:
    payload = {
        "kind": "strict_online_prior_typing_oom_event",
        "sample_id": str(sample_id),
        "stage": str(stage),
        "error_type": type(exc).__name__,
        "error_message": str(exc),
        "runtime_memory": runtime_memory,
    }
    write_json(path, payload)


def base_payload(row: Dict[str, Any]) -> Dict[str, Any]:
    payload = row.get("base")
    return payload if isinstance(payload, dict) else {}


def stored_full_pred(row: Dict[str, Any]) -> Optional[str]:
    payload = base_payload(row)
    return normalize_yes_no(payload.get("normalized_prediction") or payload.get("normalized_yes_no"))


def reference_answer(row: Dict[str, Any]) -> Optional[str]:
    raw = safe_text(row.get("eval_reference_answer")) or safe_text(row.get("reference_answer"))
    normalized = normalize_yes_no(raw)
    return normalized or raw or None


def task_family(row: Dict[str, Any]) -> str:
    return safe_text((row.get("metadata") or {}).get("task_family")) or "unknown"


def build_eval_sidecar_row(row: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "sample_id": safe_text(row.get("sample_id")),
        "benchmark": safe_text(row.get("benchmark")),
        "eval_task_family": safe_text(row.get("task_family")),
        "eval_reference_answer": safe_text(row.get("eval_reference_answer")) or safe_text(row.get("reference_answer")),
        "eval_baseline_correct": row.get("eval_baseline_correct"),
        "eval_probe_full_correct": row.get("eval_probe_full_correct"),
        "eval_target_modality": safe_text(row.get("eval_target_modality")),
        "eval_target_modality_reason": safe_text(row.get("eval_target_modality_reason")),
        "eval_prior_regime_label": safe_text(row.get("eval_prior_regime_label")),
    }


def strip_method_side_eval_fields(row: Dict[str, Any]) -> Dict[str, Any]:
    payload = dict(row)
    for field in METHOD_SIDE_EVAL_FIELDS:
        payload.pop(field, None)
    return payload


def row_choices(row: Dict[str, Any]) -> List[Dict[str, str]]:
    choices = list(row.get("choices") or ((row.get("metadata") or {}).get("options") or []))
    normalized: List[Dict[str, str]] = []
    for choice in choices:
        label = safe_text((choice or {}).get("label"))
        text = safe_text((choice or {}).get("text"))
        if label or text:
            normalized.append({"label": label, "text": text})
    return normalized


def read_sample_id_file(path: Optional[Path]) -> Optional[set[str]]:
    if path is None:
        return None
    values: set[str] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            sample_id = safe_text(line)
            if sample_id:
                values.add(sample_id)
    return values


def load_sample_lookup() -> Dict[str, Dict[str, Any]]:
    samples = (
        list(load_avhbench_samples(include_free_form=False))
        + list(load_cmm_samples())
        + list(load_worldsense_samples())
        + list(load_omnivideobench_samples())
    )
    lookup: Dict[str, Dict[str, Any]] = {}
    for sample in samples:
        payload = sample.to_dict() if hasattr(sample, "to_dict") else dict(sample)
        lookup[safe_text(payload.get("sample_id"))] = payload
    return lookup


def merged_sample(row: Dict[str, Any], sample_lookup: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    sample_id = safe_text(row.get("sample_id"))
    sample = dict(sample_lookup.get(sample_id) or {})
    metadata = dict(sample.get("metadata") or {})
    metadata.update(dict(row.get("metadata") or {}))
    sample.update(
        {
            "sample_id": sample_id,
            "benchmark": row.get("benchmark") or sample.get("benchmark"),
            "question": row.get("question") or sample.get("question"),
            "reference_answer": row.get("reference_answer") or sample.get("reference_answer"),
            "eval_type": row.get("eval_type") or sample.get("eval_type") or "yes_no",
            "choices": list(row.get("choices") or sample.get("choices") or []),
            "video_path": row.get("video_path") or sample.get("video_path"),
            "audio_path": row.get("audio_path") or sample.get("audio_path"),
            "metadata": metadata,
        }
    )
    return sample


def answer_space_for_sample(sample: Dict[str, Any]) -> AnswerSpaceSpec:
    question = safe_text(sample.get("question"))
    eval_type = (safe_text(sample.get("eval_type")) or "yes_no").lower()
    choices = row_choices(sample)
    if eval_type == "yes_no":
        return AnswerSpaceSpec(kind="yes_no", question_form="yes_no")
    if choices:
        return build_query_latent_state(
            question,
            eval_type=eval_type,
            options=choices,
            max_new_tokens=8,
        ).answer_space
    # Current AVHBench/CMM base records are yes/no-only. Do not infer answer space from
    # benchmark gold answers when raw options are absent.
    return AnswerSpaceSpec(kind="yes_no", question_form="yes_no")


def formatted_question(sample: Dict[str, Any], answer_space: AnswerSpaceSpec) -> str:
    return format_question_for_answer_space(safe_text(sample.get("question")), answer_space)


def normalize_for_space(raw: Any, answer_space: AnswerSpaceSpec) -> Optional[str]:
    return normalize_answer_for_space(raw, answer_space)


def video_budget(args: argparse.Namespace) -> Dict[str, float]:
    return {
        "fps": float(args.video_fps),
        "max_frames": int(args.video_max_frames),
        "max_pixels": int(args.video_max_pixels),
        "min_pixels": int(args.video_min_pixels),
    }


def choose_rows(
    rows: Sequence[Dict[str, Any]],
    *,
    sample_id: str,
    sample_id_set: Optional[set[str]],
    max_samples: int,
    only_incorrect: bool,
    benchmarks: Sequence[str],
    task_families: Sequence[str],
) -> List[Dict[str, Any]]:
    chosen: List[Dict[str, Any]] = []
    benchmark_set = {safe_text(item) for item in benchmarks if safe_text(item)}
    del task_families
    for row in rows:
        if row.get("skipped"):
            continue
        row_sample_id = safe_text(row.get("sample_id"))
        if sample_id and row_sample_id != safe_text(sample_id):
            continue
        if sample_id_set is not None and row_sample_id not in sample_id_set:
            continue
        if benchmark_set and safe_text(row.get("benchmark")) not in benchmark_set:
            continue
        if only_incorrect and base_payload(row).get("correct") is True:
            continue
        chosen.append(dict(row))
    chosen.sort(key=lambda row: safe_text(row.get("sample_id")))
    if int(max_samples) > 0:
        chosen = chosen[: int(max_samples)]
    return chosen


def apply_shard(rows: Sequence[Dict[str, Any]], *, num_shards: int, shard_index: int) -> List[Dict[str, Any]]:
    total_shards = max(1, int(num_shards))
    shard = int(shard_index)
    if shard < 0 or shard >= total_shards:
        raise ValueError(f"invalid shard index {shard} for num_shards={total_shards}")
    if total_shards == 1:
        return [dict(row) for row in rows]
    return [dict(row) for idx, row in enumerate(rows) if idx % total_shards == shard]


def branch_masks(branch: str) -> Tuple[bool, bool]:
    if branch == "full":
        return False, False
    if branch == "audio_only":
        return False, True
    if branch == "visual_only":
        return True, False
    if branch == "text_only":
        return True, True
    raise ValueError(f"unsupported branch: {branch}")


def prepare_full_inputs_for_avcd(
    adapter: QwenOmniAdapter,
    *,
    sample: Dict[str, Any],
    formatted_question_text: str,
    budget: Dict[str, float],
) -> Dict[str, torch.Tensor]:
    video_path = safe_text(sample.get("video_path")) or None
    audio_path = safe_text(sample.get("audio_path")) or None
    audio_array = adapter._resolve_audio_array(
        video_path=video_path,
        audio_path=audio_path,
        mask_audio=False,
    )
    return adapter._prepare_inputs(
        video_path,
        formatted_question_text,
        audio_array=audio_array,
        mask_visual=False,
        video_budget=budget,
    )


def compute_avcd_attention_mass(
    adapter: QwenOmniAdapter,
    *,
    sample: Dict[str, Any],
    formatted_question_text: str,
    budget: Dict[str, float],
    special_ids: Dict[str, Optional[int]],
) -> Dict[str, Any]:
    inputs = prepare_full_inputs_for_avcd(
        adapter,
        sample=sample,
        formatted_question_text=formatted_question_text,
        budget=budget,
    )
    payload = run_avcd_dominance_reader(
        adapter,
        inputs=inputs,
        special_ids=special_ids,
    )
    avg_dominance = [
        [str(name), float(score)]
        for name, score in list(payload.get("avg_dominance") or [])
    ]
    modality_dominance = {
        "video": float((payload.get("modality_dominance") or {}).get("video", 0.0)),
        "audio": float((payload.get("modality_dominance") or {}).get("audio", 0.0)),
        "language": float((payload.get("modality_dominance") or {}).get("language", 0.0)),
    }
    return {
        "dominant_modality": str(avg_dominance[0][0]) if avg_dominance else "unknown",
        "avg_dominance": avg_dominance,
        "modality_dominance": modality_dominance,
        "captured_layers": [int(x) for x in list(payload.get("captured_layers") or [])],
    }


def avcd_mass_for_branch(branch: str, avcd_payload: Dict[str, Any]) -> float:
    masses = dict(avcd_payload.get("modality_dominance") or {})
    if branch == "text_only":
        return float(masses.get("language", 0.0))
    if branch == "audio_only":
        return float(masses.get("audio", 0.0))
    if branch == "visual_only":
        return float(masses.get("video", 0.0))
    return 0.0


def avcd_mass_for_target_modality(target_modality: str, avcd_payload: Dict[str, Any]) -> float:
    masses = dict(avcd_payload.get("modality_dominance") or {})
    if target_modality == "audio":
        return float(masses.get("audio", 0.0))
    if target_modality == "visual":
        return float(masses.get("video", 0.0))
    return 0.0


def score_branch(
    adapter: QwenOmniAdapter,
    *,
    sample: Dict[str, Any],
    formatted_question_text: str,
    answer_space: AnswerSpaceSpec,
    branch: str,
    budget: Dict[str, float],
) -> Dict[str, Any]:
    mask_audio, mask_visual = branch_masks(branch)
    video_path = safe_text(sample.get("video_path")) or None
    audio_path = safe_text(sample.get("audio_path")) or None
    if branch == "text_only":
        video_path = None
    audio_array = adapter._resolve_audio_array(
        video_path=video_path,
        audio_path=audio_path,
        mask_audio=mask_audio,
    )
    inputs: Optional[Dict[str, torch.Tensor]] = None
    try:
        with adapter.temporary_release_cuda_reserve(f"strict_online_prior_branch:{branch}"):
            inputs = adapter._prepare_inputs(
                video_path,
                formatted_question_text,
                audio_array=audio_array,
                mask_visual=mask_visual,
                video_budget=budget,
            )
            if answer_space.kind == "yes_no":
                payload = dict(adapter._score_yes_no_inputs(inputs))
                answer = normalize_yes_no(payload.get("answer"))
                return {
                    "branch": str(branch),
                    "answer": answer,
                    "confidence_on_yes": float(payload.get("p_yes", 0.0)),
                    "confidence_on_no": float(payload.get("p_no", 0.0)),
                    "margin": float(payload.get("margin", 0.0)),
                }
            if answer_space.kind == "single_choice":
                payload = dict(adapter._score_single_choice_inputs(inputs, answer_space_spec=answer_space))
                return {
                    "branch": str(branch),
                    "answer": normalize_for_space(payload.get("answer"), answer_space),
                    "answer_text": safe_text(payload.get("answer_text")),
                    "choice_probs": dict(payload.get("choice_probs") or {}),
                    "margin": float(payload.get("margin", 0.0)),
                    "top1_prob": float(payload.get("top1_prob", 0.0)),
                    "top2_prob": float(payload.get("top2_prob", 0.0)),
                }
            raise NotImplementedError(f"unsupported answer space kind: {answer_space.kind}")
    finally:
        if inputs is not None:
            del inputs
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def branch_answer(record: Dict[str, Any]) -> Optional[str]:
    return safe_text(record.get("answer")) or None


def branch_answer_confidence(
    branch_record: Dict[str, Any],
    *,
    candidate_answer: Optional[str],
    answer_space: AnswerSpaceSpec,
) -> Optional[float]:
    answer = safe_text(candidate_answer)
    if not answer:
        return None
    if answer_space.kind == "yes_no":
        norm = normalize_yes_no(answer)
        if norm == "Yes":
            return float(branch_record.get("confidence_on_yes", 0.0))
        if norm == "No":
            return float(branch_record.get("confidence_on_no", 0.0))
        return None
    if answer_space.kind == "single_choice":
        choices = dict(branch_record.get("choice_probs") or {})
        value = choices.get(answer)
        return None if value is None else float(value)
    return None


def branch_support_for_answer(
    branch_record: Dict[str, Any],
    *,
    candidate_answer: Optional[str],
) -> Optional[float]:
    answer = safe_text(candidate_answer)
    if not answer:
        return None
    norm = normalize_yes_no(answer)
    if norm == "Yes":
        return maybe_float(branch_record.get("confidence_on_yes"))
    if norm == "No":
        return maybe_float(branch_record.get("confidence_on_no"))
    choices = dict(branch_record.get("choice_probs") or {})
    value = choices.get(answer)
    return maybe_float(value)


def infer_repair_target_online(
    *,
    question: str,
    branch_scores: Dict[str, Dict[str, Any]],
    mad_payload: Dict[str, Any],
    avcd_payload: Dict[str, Any],
    margin_threshold: float,
    joint_both_threshold: float,
    joint_support_threshold: float,
    joint_tie_threshold: float,
) -> Dict[str, Any]:
    md = dict(mad_payload.get("modality_distribution") or {})
    masses = dict(avcd_payload.get("modality_dominance") or {})
    need_audio = float(md.get("audio", 0.0)) + 0.5 * float(md.get("both", 0.0))
    need_visual = float(md.get("video", 0.0)) + 0.5 * float(md.get("both", 0.0))
    use_audio = float(masses.get("audio", 0.0))
    use_visual = float(masses.get("video", 0.0))
    gap_audio = float(need_audio - use_audio)
    gap_visual = float(need_visual - use_visual)
    gap_margin = float(abs(gap_audio - gap_visual))
    max_gap = float(max(gap_audio, gap_visual))
    both_prob = float(md.get("both", 0.0))

    question_vote, cue_meta = online_target_modality_from_question(question)
    full_answer = branch_answer(branch_scores["full"])
    audio_answer = branch_answer(branch_scores["audio_only"])
    visual_answer = branch_answer(branch_scores["visual_only"])
    audio_support_on_full = branch_support_for_answer(
        branch_scores["audio_only"],
        candidate_answer=full_answer,
    )
    visual_support_on_full = branch_support_for_answer(
        branch_scores["visual_only"],
        candidate_answer=full_answer,
    )
    support_tie_gap = None
    if audio_support_on_full is not None and visual_support_on_full is not None:
        support_tie_gap = float(abs(audio_support_on_full - visual_support_on_full))

    branch_hint = "unknown"
    if full_answer and audio_answer and visual_answer:
        if full_answer == audio_answer and full_answer != visual_answer:
            branch_hint = "visual"
        elif full_answer == visual_answer and full_answer != audio_answer:
            branch_hint = "audio"

    repair_target = "unknown"
    reason = "repair_gap_ambiguous"
    confidence = 0.0
    threshold = float(margin_threshold)
    soft_threshold = float(max(0.03, threshold * 0.5))
    joint_both_threshold = float(joint_both_threshold)
    joint_support_threshold = float(joint_support_threshold)
    joint_tie_threshold = float(joint_tie_threshold)
    joint_ambiguous = bool(
        both_prob >= joint_both_threshold
        and audio_support_on_full is not None
        and visual_support_on_full is not None
        and audio_support_on_full >= joint_support_threshold
        and visual_support_on_full >= joint_support_threshold
        and support_tie_gap is not None
        and support_tie_gap <= joint_tie_threshold
    )

    if max_gap <= 0.0:
        reason = "repair_gap_nonpositive"
    elif joint_ambiguous:
        repair_target = "unknown"
        reason = "joint_multimodal_support_ambiguous"
        confidence = float(
            min(
                both_prob,
                float(audio_support_on_full or 0.0),
                float(visual_support_on_full or 0.0),
            )
        )
    elif gap_margin >= threshold:
        repair_target = "audio" if gap_audio > gap_visual else "visual"
        reason = "need_use_gap_margin"
        confidence = gap_margin
    elif branch_hint in {"audio", "visual"}:
        hinted_gap = gap_audio if branch_hint == "audio" else gap_visual
        if hinted_gap > 0.0:
            repair_target = str(branch_hint)
            reason = "branch_conflict_repair_hint"
            confidence = max(gap_margin, hinted_gap)
    elif question_vote in {"audio", "visual"}:
        hinted_gap = gap_audio if question_vote == "audio" else gap_visual
        if hinted_gap >= soft_threshold:
            repair_target = str(question_vote)
            reason = "question_cue_repair_hint"
            confidence = max(gap_margin, hinted_gap)

    return {
        "repair_target_modality": str(repair_target),
        "repair_target_reason": str(reason),
        "repair_target_confidence": float(confidence),
        "need_audio": float(need_audio),
        "need_visual": float(need_visual),
        "use_audio": float(use_audio),
        "use_visual": float(use_visual),
        "repair_gap_audio": float(gap_audio),
        "repair_gap_visual": float(gap_visual),
        "repair_gap_margin": float(gap_margin),
        "repair_gap_max": float(max_gap),
        "mad_both_prob": float(both_prob),
        "branch_hint": str(branch_hint),
        "full_answer": full_answer,
        "audio_answer": audio_answer,
        "visual_answer": visual_answer,
        "audio_support_on_full": audio_support_on_full,
        "visual_support_on_full": visual_support_on_full,
        "support_tie_gap": support_tie_gap,
        "joint_both_threshold": float(joint_both_threshold),
        "joint_support_threshold": float(joint_support_threshold),
        "joint_tie_threshold": float(joint_tie_threshold),
        "joint_ambiguous": bool(joint_ambiguous),
        **cue_meta,
    }


def online_target_modality_from_question(question: str) -> Tuple[str, Dict[str, Any]]:
    text = safe_text(question).lower()
    audio_hits = sum(1 for token in _AUDIO_QUERY_CUES if token in text)
    visual_hits = sum(1 for token in _VISUAL_QUERY_CUES if token in text)
    if audio_hits > visual_hits and audio_hits > 0:
        return "audio", {"audio_cues": int(audio_hits), "visual_cues": int(visual_hits), "question_vote": "audio"}
    if visual_hits > audio_hits and visual_hits > 0:
        return "visual", {"audio_cues": int(audio_hits), "visual_cues": int(visual_hits), "question_vote": "visual"}
    if audio_hits > 0 and visual_hits > 0:
        return "joint_av", {"audio_cues": int(audio_hits), "visual_cues": int(visual_hits), "question_vote": "joint_av"}
    return "unknown", {"audio_cues": int(audio_hits), "visual_cues": int(visual_hits), "question_vote": "unknown"}


def infer_target_modality_online(
    *,
    question: str,
    branch_scores: Dict[str, Dict[str, Any]],
    mad_payload: Dict[str, Any],
    avcd_payload: Dict[str, Any],
    margin_threshold: float,
    avcd_target_mass_weight: float,
) -> Dict[str, Any]:
    del avcd_payload, avcd_target_mass_weight
    md = dict(mad_payload.get("modality_distribution") or {})
    audio_prob = float(md.get("audio", 0.0))
    video_prob = float(md.get("video", 0.0))
    both_prob = float(md.get("both", 0.0))
    score_audio = audio_prob + 0.5 * both_prob
    score_visual = video_prob + 0.5 * both_prob
    margin = float(abs(score_audio - score_visual))
    question_vote, cue_meta = online_target_modality_from_question(question)
    full_answer = branch_answer(branch_scores["full"])
    audio_answer = branch_answer(branch_scores["audio_only"])
    visual_answer = branch_answer(branch_scores["visual_only"])
    target: str
    reason: str

    if margin >= float(margin_threshold):
        target = "audio" if score_audio > score_visual else "visual"
        reason = "mad_requirement_margin"
    else:
        if question_vote in {"audio", "visual"}:
            target = question_vote
            reason = "question_cue_tiebreak"
        elif full_answer and audio_answer and visual_answer:
            if full_answer == audio_answer and full_answer != visual_answer:
                target = "visual"
                reason = "branch_conflict_tiebreak_visual"
            elif full_answer == visual_answer and full_answer != audio_answer:
                target = "audio"
                reason = "branch_conflict_tiebreak_audio"
            else:
                target = "joint_av"
                reason = "branch_conflict_unresolved"
        else:
            target = "unknown"
            reason = "online_target_unresolved"
    return {
        "target_modality": str(target),
        "target_modality_reason": str(reason),
        "mad_audio_prob": float(audio_prob),
        "mad_video_prob": float(video_prob),
        "mad_both_prob": float(both_prob),
        "online_audio_score": float(score_audio),
        "online_visual_score": float(score_visual),
        "target_margin": float(margin),
        **cue_meta,
    }


def directed_label_for(target_modality: str, source_branch: str) -> Optional[str]:
    if target_modality == "audio" and source_branch == "visual_only":
        return "visual_over_audio"
    if target_modality == "visual" and source_branch == "audio_only":
        return "audio_over_visual"
    return None


def background_text_branch_for_target(target_modality: str) -> str:
    if target_modality == "audio":
        return "text_only"
    if target_modality == "visual":
        return "text_only"
    raise ValueError(f"unsupported target modality: {target_modality}")


def cross_modal_source_branch_for_target(target_modality: str) -> str:
    if target_modality == "audio":
        return "visual_only"
    if target_modality == "visual":
        return "audio_only"
    raise ValueError(f"unsupported target modality: {target_modality}")


def target_branch_for(target_modality: str) -> str:
    if target_modality == "audio":
        return "audio_only"
    if target_modality == "visual":
        return "visual_only"
    raise ValueError(f"unsupported target modality: {target_modality}")


def expected_source_branch_from_avcd(
    *,
    target_modality: str,
    avcd_payload: Dict[str, Any],
) -> Tuple[Optional[str], Dict[str, Any]]:
    masses = dict(avcd_payload.get("modality_dominance") or {})
    language_mass = float(masses.get("language", 0.0))
    audio_mass = float(masses.get("audio", 0.0))
    video_mass = float(masses.get("video", 0.0))
    if target_modality == "audio":
        candidates = [("visual_only", video_mass)]
    elif target_modality == "visual":
        candidates = [("audio_only", audio_mass)]
    else:
        return None, {
            "avcd_source_branch": None,
            "avcd_source_mass": 0.0,
            "avcd_target_mass": 0.0,
            "avcd_background_language_mass": 0.0,
            "avcd_mass_gap_vs_target": 0.0,
        }
    ranked = sorted(candidates, key=lambda item: float(item[1]), reverse=True)
    best_branch, best_mass = ranked[0]
    target_mass = avcd_mass_for_target_modality(target_modality, avcd_payload)
    second_mass = float(ranked[1][1]) if len(ranked) > 1 else 0.0
    return str(best_branch), {
        "avcd_source_branch": str(best_branch),
        "avcd_source_mass": float(best_mass),
        "avcd_second_source_mass": float(second_mass),
        "avcd_target_mass": float(target_mass),
        "avcd_background_language_mass": float(language_mass),
        "avcd_mass_gap_vs_target": float(best_mass - target_mass),
    }


def compute_source_score(
    source_record: Dict[str, Any],
    *,
    target_record: Dict[str, Any],
    full_answer: Optional[str],
    answer_space: AnswerSpaceSpec,
    source_mass: float,
    target_mass: float,
    avcd_source_mass_weight: float,
) -> Dict[str, Any]:
    source_answer = branch_answer(source_record)
    target_answer = branch_answer(target_record)
    source_conf = branch_answer_confidence(source_record, candidate_answer=full_answer, answer_space=answer_space)
    target_conf = branch_answer_confidence(target_record, candidate_answer=full_answer, answer_space=answer_space)
    if source_conf is None:
        source_conf = 0.0
    if target_conf is None:
        target_conf = 0.0
    agrees_with_full = bool(source_answer and full_answer and source_answer == full_answer)
    target_disagrees_with_full = bool(target_answer and full_answer and target_answer != full_answer)
    mass_gap = float(source_mass - target_mass)
    score = float(source_conf - target_conf) + float(avcd_source_mass_weight) * mass_gap
    if agrees_with_full:
        score += 1.0
    return {
        "source_answer": source_answer,
        "target_answer": target_answer,
        "agrees_with_full": agrees_with_full,
        "target_disagrees_with_full": target_disagrees_with_full,
        "source_confidence_on_full": float(source_conf),
        "target_confidence_on_full": float(target_conf),
        "confidence_gap_vs_target": float(source_conf - target_conf),
        "avcd_source_mass": float(source_mass),
        "avcd_target_mass": float(target_mass),
        "avcd_mass_gap": float(mass_gap),
        "score": float(score),
    }


def attribute_directed_prior(
    *,
    target_modality: str,
    branch_scores: Dict[str, Dict[str, Any]],
    answer_space: AnswerSpaceSpec,
    avcd_payload: Dict[str, Any],
    source_confidence_gap_threshold: float,
    source_tie_threshold: float,
    avcd_source_mass_weight: float,
) -> Dict[str, Any]:
    full_record = dict(branch_scores["full"])
    full_answer = branch_answer(full_record)
    if target_modality not in {"audio", "visual"}:
        return {
            "directed_prior_label": "no_cross_modal_override",
            "reason": "target_modality_not_single_target_online",
            "target_modality": str(target_modality),
            "target_branch": None,
            "source_branch": None,
            "source_candidates": {},
        }

    target_branch = target_branch_for(target_modality)
    text_branch = background_text_branch_for_target(target_modality)
    other_branch = cross_modal_source_branch_for_target(target_modality)
    target_record = dict(branch_scores[target_branch])
    text_record = dict(branch_scores[text_branch])
    other_record = dict(branch_scores[other_branch])
    target_answer = branch_answer(target_record)
    avcd_source_branch, avcd_source_meta = expected_source_branch_from_avcd(
        target_modality=target_modality,
        avcd_payload=avcd_payload,
    )
    target_mass = float(avcd_source_meta.get("avcd_target_mass", 0.0))

    if not full_answer:
        return {
            "directed_prior_label": "no_cross_modal_override",
            "reason": "full_answer_unparsed",
            "target_modality": str(target_modality),
            "target_branch": str(target_branch),
            "source_branch": None,
            "source_candidates": {},
            "avcd_source_meta": avcd_source_meta,
        }
    if target_answer == full_answer:
        return {
            "directed_prior_label": "no_cross_modal_override",
            "reason": "full_matches_target_branch",
            "target_modality": str(target_modality),
            "target_branch": str(target_branch),
            "source_branch": None,
            "source_candidates": {
                str(text_branch): compute_source_score(
                    text_record,
                    target_record=target_record,
                    full_answer=full_answer,
                    answer_space=answer_space,
                    source_mass=avcd_mass_for_branch(text_branch, avcd_payload),
                    target_mass=target_mass,
                    avcd_source_mass_weight=float(avcd_source_mass_weight),
                ),
                str(other_branch): compute_source_score(
                    other_record,
                    target_record=target_record,
                    full_answer=full_answer,
                    answer_space=answer_space,
                    source_mass=avcd_mass_for_branch(other_branch, avcd_payload),
                    target_mass=target_mass,
                    avcd_source_mass_weight=float(avcd_source_mass_weight),
                ),
            },
            "avcd_source_meta": avcd_source_meta,
        }

    text_candidate = compute_source_score(
        text_record,
        target_record=target_record,
        full_answer=full_answer,
        answer_space=answer_space,
        source_mass=avcd_mass_for_branch(text_branch, avcd_payload),
        target_mass=target_mass,
        avcd_source_mass_weight=float(avcd_source_mass_weight),
    )
    other_candidate = compute_source_score(
        other_record,
        target_record=target_record,
        full_answer=full_answer,
        answer_space=answer_space,
        source_mass=avcd_mass_for_branch(other_branch, avcd_payload),
        target_mass=target_mass,
        avcd_source_mass_weight=float(avcd_source_mass_weight),
    )
    source_candidates = {
        str(text_branch): {
            **text_candidate,
            "candidate_role": "background_language",
        },
        str(other_branch): {
            **other_candidate,
            "candidate_role": "cross_modal_source",
        },
    }

    cross_modal_payload = source_candidates[str(other_branch)]
    if not (
        bool(cross_modal_payload.get("agrees_with_full"))
        and bool(cross_modal_payload.get("target_disagrees_with_full"))
    ):
        return {
            "directed_prior_label": "no_cross_modal_override",
            "reason": "cross_modal_source_not_supported_by_branch_agreement",
            "target_modality": str(target_modality),
            "target_branch": str(target_branch),
            "source_branch": None,
            "source_candidates": source_candidates,
            "avcd_source_meta": avcd_source_meta,
        }

    if avcd_source_branch is None or avcd_source_branch != other_branch:
        return {
            "directed_prior_label": "no_cross_modal_override",
            "reason": "avcd_source_not_verified_by_branch_agreement",
            "target_modality": str(target_modality),
            "target_branch": str(target_branch),
            "source_branch": avcd_source_branch,
            "source_candidates": source_candidates,
            "avcd_source_meta": avcd_source_meta,
        }
    best_payload = cross_modal_payload
    if float(best_payload.get("confidence_gap_vs_target", 0.0)) < float(source_confidence_gap_threshold):
        return {
            "directed_prior_label": "no_cross_modal_override",
            "reason": "cross_modal_confidence_gap_below_threshold",
            "target_modality": str(target_modality),
            "target_branch": str(target_branch),
            "source_branch": str(avcd_source_branch),
            "source_candidates": source_candidates,
            "avcd_source_meta": avcd_source_meta,
        }
    cross_modal_score = float(best_payload.get("score", 0.0))
    background_language_score = float(text_candidate.get("score", 0.0))
    if cross_modal_score <= background_language_score:
        return {
            "directed_prior_label": "no_cross_modal_override",
            "reason": "background_language_outranks_cross_modal_source",
            "target_modality": str(target_modality),
            "target_branch": str(target_branch),
            "source_branch": None,
            "source_candidates": source_candidates,
            "avcd_source_meta": {
                **avcd_source_meta,
                "background_language_score": background_language_score,
                "cross_modal_score": cross_modal_score,
            },
        }
    if abs(cross_modal_score - background_language_score) < float(source_tie_threshold):
        return {
            "directed_prior_label": "no_cross_modal_override",
            "reason": "cross_modal_tie_with_background_language",
            "target_modality": str(target_modality),
            "target_branch": str(target_branch),
            "source_branch": None,
            "source_candidates": source_candidates,
            "avcd_source_meta": {
                **avcd_source_meta,
                "background_language_score": background_language_score,
                "cross_modal_score": cross_modal_score,
            },
        }

    label = directed_label_for(target_modality, avcd_source_branch)
    if not label:
        return {
            "directed_prior_label": "no_cross_modal_override",
            "reason": "verified_source_not_mappable_to_directed_label",
            "target_modality": str(target_modality),
            "target_branch": str(target_branch),
            "source_branch": str(avcd_source_branch),
            "source_candidates": source_candidates,
            "avcd_source_meta": avcd_source_meta,
        }
    return {
        "directed_prior_label": str(label),
        "reason": "directed_prior_detected",
        "target_modality": str(target_modality),
        "target_branch": str(target_branch),
        "source_branch": str(avcd_source_branch),
        "source_candidates": source_candidates,
        "avcd_source_meta": avcd_source_meta,
    }


def classify_target_evidence_state(
    *,
    target_modality: str,
    branch_scores: Dict[str, Dict[str, Any]],
    answer_space: AnswerSpaceSpec,
) -> Dict[str, Any]:
    if target_modality not in {"audio", "visual"}:
        return {
            "target_evidence_state": "undetermined",
            "target_evidence_reason": "target_modality_not_single_target_online",
            "target_branch_answer": None,
            "full_branch_answer": branch_answer(branch_scores["full"]),
            "target_confidence_on_full": None,
            "target_margin": None,
        }

    full_record = dict(branch_scores["full"])
    target_record = dict(branch_scores[target_branch_for(target_modality)])
    full_answer = branch_answer(full_record)
    target_answer = branch_answer(target_record)
    target_conf_on_full = branch_answer_confidence(
        target_record,
        candidate_answer=full_answer,
        answer_space=answer_space,
    )
    target_margin = float(target_record.get("margin", 0.0))

    if not full_answer or not target_answer:
        return {
            "target_evidence_state": "undetermined",
            "target_evidence_reason": "branch_answer_unparsed",
            "target_branch_answer": target_answer,
            "full_branch_answer": full_answer,
            "target_confidence_on_full": target_conf_on_full,
            "target_margin": target_margin,
        }

    if target_answer != full_answer:
        return {
            "target_evidence_state": "target_evidence_absent_or_unstable",
            "target_evidence_reason": "target_branch_disagrees_with_full",
            "target_branch_answer": target_answer,
            "full_branch_answer": full_answer,
            "target_confidence_on_full": target_conf_on_full,
            "target_margin": target_margin,
        }

    return {
        "target_evidence_state": "target_evidence_present",
        "target_evidence_reason": "target_branch_matches_full",
        "target_branch_answer": target_answer,
        "full_branch_answer": full_answer,
        "target_confidence_on_full": target_conf_on_full,
        "target_margin": target_margin,
    }


def eval_target_modality_from_metadata(sample: Dict[str, Any]) -> Tuple[str, str]:
    family = task_family(sample)
    if family == "audio_grounded_presence":
        return "audio", "task_family_audio_grounded_presence"
    if family == "visual_grounded_presence":
        return "visual", "task_family_visual_grounded_presence"
    if family == "av_matching":
        return "joint_av", "task_family_av_matching"
    return "unknown", "task_family_unresolved"


def eval_prior_regime_label(sample: Dict[str, Any]) -> Optional[str]:
    metadata = dict(sample.get("metadata") or {})
    subcat = safe_text(metadata.get("sub_category")).lower()
    modality = safe_text(metadata.get("modality")).lower()
    corr = safe_text(metadata.get("correlation_type")).lower()
    pool = " ".join(part for part in (subcat, modality, corr) if part)
    if "overrely_visual_ignore_audio" in pool:
        return "visual_over_audio"
    if "overrely_audio_ignore_visual" in pool:
        return "audio_over_visual"
    return None


def compact_plan_row(row: Dict[str, Any], sample: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "sample_id": safe_text(row.get("sample_id")),
        "question": safe_text(row.get("question")),
        "stored_full_prediction": stored_full_pred(row),
        "video_path": safe_text(sample.get("video_path")),
        "audio_path": safe_text(sample.get("audio_path")),
    }


def compact_plan_eval_row(row: Dict[str, Any], sample: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "sample_id": safe_text(row.get("sample_id")),
        "benchmark": safe_text(row.get("benchmark")) or safe_text(sample.get("benchmark")),
        "eval_task_family": task_family(row),
        "eval_reference_answer": safe_text(row.get("reference_answer")) or safe_text(sample.get("reference_answer")),
    }


def skipped_record(
    *,
    row: Dict[str, Any],
    sample: Dict[str, Any],
    formatted_question: str,
    reason: str,
) -> Dict[str, Any]:
    return {
        "sample_id": safe_text(row.get("sample_id")),
        "benchmark": safe_text(row.get("benchmark")),
        "task_family": task_family(row),
        "question": safe_text(row.get("question")),
        "formatted_question": formatted_question,
        "target_modality": "unknown",
        "target_modality_reason": "runtime_exception",
        "online_target_meta": {},
        "repair_target_modality": "unknown",
        "repair_target_reason": "runtime_exception",
        "repair_target_confidence": 0.0,
        "repair_target_meta": {},
        "directed_prior_label": "skipped",
        "attribution_reason": "runtime_exception",
        "attribution_target_branch": None,
        "attribution_source_branch": None,
        "attribution_avcd_source_branch": None,
        "attribution_avcd_source_meta": {},
        "source_candidates": [],
        "target_evidence_state": "undetermined",
        "target_evidence_reason": "runtime_exception",
        "target_evidence_meta": {},
        "branches": {},
        "mad_runtime": {},
        "avcd_attention_mass": {},
        "probe_full_answer": None,
        "stored_full_prediction": stored_full_pred(row),
        "probe_full_matches_stored": None,
        "eval_reference_answer": reference_answer(row),
        "eval_baseline_correct": base_payload(row).get("correct"),
        "eval_probe_full_correct": None,
        "eval_target_modality": "unknown",
        "eval_target_modality_reason": "runtime_exception",
        "eval_prior_regime_label": eval_prior_regime_label(sample),
        "skipped": True,
        "skipped_reason": reason,
    }


def summarize(
    records: Sequence[Dict[str, Any]],
    eval_sidecar_lookup: Optional[Dict[str, Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    by_label = Counter()
    by_reason = Counter()
    by_online_target = Counter()
    by_target_reason = Counter()
    by_target_evidence_state = Counter()
    by_eval_target = Counter()
    by_repair_target = Counter()
    by_repair_reason = Counter()
    by_eval_target_repair_target = Counter()
    by_benchmark_label = Counter()
    by_task_family_label = Counter()
    by_eval_baseline_correct_label = Counter()
    by_eval_regime_label = Counter()
    full_matches_stored = 0
    probe_full_correct_count = 0
    probe_full_wrong_count = 0
    target_match_count = 0
    target_match_denom = 0
    repair_match_count = 0
    repair_match_denom = 0
    joint_eval_unknown_count = 0
    joint_eval_unknown_denom = 0
    regime_match_count = 0
    regime_match_denom = 0

    for row in records:
        sample_id = safe_text(row.get("sample_id"))
        eval_row = dict((eval_sidecar_lookup or {}).get(sample_id) or {})
        label = safe_text(row.get("directed_prior_label")) or "missing"
        reason = safe_text(row.get("attribution_reason")) or "missing"
        benchmark = safe_text(eval_row.get("benchmark")) or safe_text(row.get("benchmark")) or "unknown"
        family = safe_text(eval_row.get("eval_task_family")) or safe_text(row.get("task_family")) or "unknown"
        online_target = safe_text(row.get("target_modality")) or "unknown"
        target_reason = safe_text(row.get("target_modality_reason")) or "missing"
        target_evidence_state = safe_text(row.get("target_evidence_state")) or "unknown"
        eval_target = safe_text(eval_row.get("eval_target_modality")) or safe_text(row.get("eval_target_modality")) or "unknown"
        repair_target = safe_text(row.get("repair_target_modality")) or "unknown"
        repair_reason = safe_text(row.get("repair_target_reason")) or "missing"
        eval_baseline_correct = (
            eval_row.get("eval_baseline_correct")
            if eval_row.get("eval_baseline_correct") is not None
            else row.get("eval_baseline_correct")
        )
        eval_regime = (
            safe_text(eval_row.get("eval_prior_regime_label"))
            or safe_text(row.get("eval_prior_regime_label"))
            or ""
        )

        by_label[label] += 1
        by_reason[reason] += 1
        by_online_target[online_target] += 1
        by_target_reason[target_reason] += 1
        by_target_evidence_state[target_evidence_state] += 1
        by_eval_target[eval_target] += 1
        by_repair_target[repair_target] += 1
        by_repair_reason[repair_reason] += 1
        by_eval_target_repair_target[f"{eval_target}|{repair_target}"] += 1
        by_benchmark_label[f"{benchmark}|{label}"] += 1
        by_task_family_label[f"{family}|{label}"] += 1
        by_eval_baseline_correct_label[f"{eval_baseline_correct}|{label}"] += 1
        if eval_regime:
            by_eval_regime_label[f"{eval_regime}|{label}"] += 1

        if row.get("probe_full_matches_stored") is True:
            full_matches_stored += 1
        probe_full_correct = (
            eval_row.get("eval_probe_full_correct")
            if eval_row.get("eval_probe_full_correct") is not None
            else row.get("eval_probe_full_correct")
        )
        if probe_full_correct is True:
            probe_full_correct_count += 1
        if probe_full_correct is False:
            probe_full_wrong_count += 1

        if online_target in {"audio", "visual", "joint_av"} and eval_target in {"audio", "visual", "joint_av"}:
            target_match_denom += 1
            if online_target == eval_target:
                target_match_count += 1
        if eval_target in {"audio", "visual"}:
            repair_match_denom += 1
            if repair_target == eval_target:
                repair_match_count += 1
        if eval_target == "joint_av":
            joint_eval_unknown_denom += 1
            if repair_target == "unknown":
                joint_eval_unknown_count += 1
        if eval_regime:
            regime_match_denom += 1
            if label == eval_regime:
                regime_match_count += 1

    return {
        "n_records": len(records),
        "directed_prior_label_counts": dict(by_label),
        "attribution_reason_counts": dict(by_reason),
        "online_target_modality_counts": dict(by_online_target),
        "target_modality_reason_counts": dict(by_target_reason),
        "target_evidence_state_counts": dict(by_target_evidence_state),
        "eval_target_modality_counts": dict(by_eval_target),
        "repair_target_modality_counts": dict(by_repair_target),
        "repair_target_reason_counts": dict(by_repair_reason),
        "by_eval_target_repair_target": dict(sorted(by_eval_target_repair_target.items())),
        "by_benchmark_label": dict(sorted(by_benchmark_label.items())),
        "by_task_family_label": dict(sorted(by_task_family_label.items())),
        "by_eval_baseline_correct_label": dict(sorted(by_eval_baseline_correct_label.items())),
        "by_eval_prior_regime_label": dict(sorted(by_eval_regime_label.items())),
        "probe_full_matches_stored_count": int(full_matches_stored),
        "probe_full_correct_count": int(probe_full_correct_count),
        "probe_full_wrong_count": int(probe_full_wrong_count),
        "online_target_match_count": int(target_match_count),
        "online_target_match_denom": int(target_match_denom),
        "online_target_match_rate": float(target_match_count / target_match_denom) if target_match_denom else None,
        "repair_target_match_count": int(repair_match_count),
        "repair_target_match_denom": int(repair_match_denom),
        "repair_target_match_rate": float(repair_match_count / repair_match_denom) if repair_match_denom else None,
        "joint_eval_unknown_count": int(joint_eval_unknown_count),
        "joint_eval_unknown_denom": int(joint_eval_unknown_denom),
        "joint_eval_unknown_rate": float(joint_eval_unknown_count / joint_eval_unknown_denom) if joint_eval_unknown_denom else None,
        "regime_recovery_match_count": int(regime_match_count),
        "regime_recovery_match_denom": int(regime_match_denom),
        "regime_recovery_match_rate": float(regime_match_count / regime_match_denom) if regime_match_denom else None,
    }


class ProbeOomAbort(RuntimeError):
    def __init__(
        self,
        *,
        sample_id: str,
        stage: str,
        original_exc: BaseException,
        runtime_memory: Dict[str, Any],
    ) -> None:
        super().__init__(f"OOM at stage={stage} sample_id={sample_id}: {type(original_exc).__name__}: {original_exc}")
        self.sample_id = str(sample_id)
        self.stage = str(stage)
        self.original_exc = original_exc
        self.runtime_memory = dict(runtime_memory)


def run() -> None:
    args = parse_args()
    progress_enabled = not args.no_progress
    args.output_dir.mkdir(parents=True, exist_ok=True)
    records_path = args.output_dir / "records.jsonl"

    rows = read_jsonl(args.records)
    sample_lookup = load_sample_lookup()
    sample_id_set = read_sample_id_file(args.sample_id_file)
    selected = choose_rows(
        rows,
        sample_id=args.sample_id,
        sample_id_set=sample_id_set,
        max_samples=args.max_samples,
        only_incorrect=bool(args.only_incorrect),
        benchmarks=args.benchmarks,
        task_families=args.task_families,
    )
    selected = apply_shard(selected, num_shards=args.num_shards, shard_index=args.shard_index)

    plan_rows: List[Dict[str, Any]] = []
    plan_eval_rows: List[Dict[str, Any]] = []
    selected_with_samples: List[Tuple[Dict[str, Any], Dict[str, Any]]] = []
    for row in selected:
        sample = merged_sample(row, sample_lookup)
        selected_with_samples.append((row, sample))
        plan_rows.append(compact_plan_row(row, sample))
        plan_eval_rows.append(compact_plan_eval_row(row, sample))

    write_jsonl(args.output_dir / "selected_samples.jsonl", plan_rows)
    write_jsonl(args.output_dir / "selected_samples_eval_sidecar.jsonl", plan_eval_rows)
    write_json(
        args.output_dir / "plan_summary.json",
        {
            "records": str(args.records),
            "selected": len(plan_rows),
            "num_shards": int(args.num_shards),
            "shard_index": int(args.shard_index),
            "only_incorrect": bool(args.only_incorrect),
            "legacy_task_families_cli_ignored": list(args.task_families),
            "benchmarks": list(args.benchmarks),
            "target_modality_margin_threshold": float(args.target_modality_margin_threshold),
            "source_confidence_gap_threshold": float(args.source_confidence_gap_threshold),
            "source_tie_threshold": float(args.source_tie_threshold),
            "avcd_target_mass_weight": float(args.avcd_target_mass_weight),
            "avcd_source_mass_weight": float(args.avcd_source_mass_weight),
            "video_budget": video_budget(args),
            "runtime_constraints": {
                "task_family_not_used_for_runtime": True,
                "benchmark_regime_not_used_for_runtime": True,
                "allowed_runtime_inputs": [
                    "question",
                    "media",
                    "mad_modality_distribution",
                    "avcd_attention_mass",
                    "full_audio_visual_text_branch_outputs",
                ],
            },
        },
    )

    if args.plan_only:
        print(
            f"[strict-online-prior-typing] plan-only selected={len(plan_rows)} "
            f"output_dir={args.output_dir}"
        )
        return

    completed: Dict[str, Dict[str, Any]] = {}
    if args.resume and records_path.exists():
        for record in read_jsonl(records_path):
            completed[safe_text(record.get("sample_id"))] = record
        print(f"[strict-online-prior-typing] resume loaded completed={len(completed)}")
    elif records_path.exists():
        raise RuntimeError(f"{records_path} already exists. Use --resume or choose a new output dir.")

    adapter = QwenOmniAdapter(
        model_path=args.model_path,
        device=args.device,
        max_new_tokens=8,
    )
    budget = video_budget(args)
    special_ids = tokenizer_ids(adapter)
    new_records: List[Dict[str, Any]] = []
    eval_sidecar_path = args.output_dir / "eval_sidecar.jsonl"
    new_eval_rows: List[Dict[str, Any]] = []
    iterator = tqdm(
        selected_with_samples,
        total=len(selected_with_samples),
        desc="strict_online_prior",
        unit="sample",
        dynamic_ncols=True,
        disable=not progress_enabled,
    )
    try:
        for row, sample in iterator:
            sample_id = safe_text(row.get("sample_id"))
            if sample_id in completed:
                continue
            answer_space = answer_space_for_sample(sample)
            question = formatted_question(sample, answer_space)
            stage_name = "init"
            memory_trace: Dict[str, Any] = {"sample_start": cuda_memory_snapshot()}
            reset_cuda_peak_memory_stats()
            try:
                branch_scores: Dict[str, Dict[str, Any]] = {}
                for branch in ("full", "audio_only", "visual_only", "text_only"):
                    stage_name = f"branch:{branch}"
                    reset_cuda_peak_memory_stats()
                    branch_scores[branch] = score_branch(
                        adapter,
                        sample=sample,
                        formatted_question_text=question,
                        answer_space=answer_space,
                        branch=branch,
                        budget=budget,
                    )
                    memory_trace[f"{stage_name}_memory"] = cuda_memory_snapshot()

                video_path = safe_text(sample.get("video_path"))
                audio_path = safe_text(sample.get("audio_path"))
                stage_name = "mad_head_only"
                reset_cuda_peak_memory_stats()
                mad_payload = strict_mad_head_only_modality_probe(
                    adapter,
                    question=question_with_answer_prompt(safe_text(sample.get("question"))),
                    video_path=video_path,
                    audio_path=audio_path,
                    budget=budget,
                    add_prompt=MODALITY_QUERY_PROMPT,
                )
                memory_trace[f"{stage_name}_memory"] = cuda_memory_snapshot()
                stage_name = "avcd_dominance"
                reset_cuda_peak_memory_stats()
                avcd_mass_payload = compute_avcd_attention_mass(
                    adapter,
                    sample=sample,
                    formatted_question_text=question,
                    budget=budget,
                    special_ids=special_ids,
                )
                memory_trace[f"{stage_name}_memory"] = cuda_memory_snapshot()
                stage_name = "target_inference"
                online_target = infer_target_modality_online(
                    question=safe_text(sample.get("question")),
                    branch_scores=branch_scores,
                    mad_payload=mad_payload,
                    avcd_payload=avcd_mass_payload,
                    margin_threshold=float(args.target_modality_margin_threshold),
                    avcd_target_mass_weight=float(args.avcd_target_mass_weight),
                )
                repair_target = infer_repair_target_online(
                    question=safe_text(sample.get("question")),
                    branch_scores=branch_scores,
                    mad_payload=mad_payload,
                    avcd_payload=avcd_mass_payload,
                    margin_threshold=float(args.target_modality_margin_threshold),
                    joint_both_threshold=float(args.repair_joint_both_threshold),
                    joint_support_threshold=float(args.repair_joint_support_threshold),
                    joint_tie_threshold=float(args.repair_joint_tie_threshold),
                )
                stage_name = "source_attribution"
                attribution = attribute_directed_prior(
                    target_modality=safe_text(online_target.get("target_modality")),
                    branch_scores=branch_scores,
                    answer_space=answer_space,
                    avcd_payload=avcd_mass_payload,
                    source_confidence_gap_threshold=float(args.source_confidence_gap_threshold),
                    source_tie_threshold=float(args.source_tie_threshold),
                    avcd_source_mass_weight=float(args.avcd_source_mass_weight),
                )
                target_evidence_state = classify_target_evidence_state(
                    target_modality=safe_text(online_target.get("target_modality")),
                    branch_scores=branch_scores,
                    answer_space=answer_space,
                )
                full_answer = branch_answer(branch_scores["full"])
                stored_answer = stored_full_pred(row)
                ref_answer = reference_answer(row)
                probe_full_correct = bool(full_answer == ref_answer) if full_answer and ref_answer else None
                eval_target, eval_target_reason = eval_target_modality_from_metadata(sample)
                eval_regime = eval_prior_regime_label(sample)
                record = {
                    "sample_id": sample_id,
                    "benchmark": safe_text(row.get("benchmark")),
                    "task_family": task_family(row),
                    "question": safe_text(row.get("question")),
                    "formatted_question": question,
                    "answer_space_kind": answer_space.kind,
                    "target_modality": safe_text(online_target.get("target_modality")),
                    "target_modality_reason": safe_text(online_target.get("target_modality_reason")),
                    "online_target_meta": online_target,
                    "repair_target_modality": safe_text(repair_target.get("repair_target_modality")),
                    "repair_target_reason": safe_text(repair_target.get("repair_target_reason")),
                    "repair_target_confidence": repair_target.get("repair_target_confidence"),
                    "repair_target_meta": repair_target,
                    "directed_prior_label": safe_text(attribution.get("directed_prior_label")),
                    "attribution_reason": safe_text(attribution.get("reason")),
                    "attribution_target_branch": attribution.get("target_branch"),
                    "attribution_source_branch": attribution.get("source_branch"),
                    "attribution_avcd_source_branch": (attribution.get("avcd_source_meta") or {}).get("avcd_source_branch"),
                    "attribution_avcd_source_meta": attribution.get("avcd_source_meta"),
                    "source_candidates": attribution.get("source_candidates"),
                    "target_evidence_state": safe_text(target_evidence_state.get("target_evidence_state")),
                    "target_evidence_reason": safe_text(target_evidence_state.get("target_evidence_reason")),
                    "target_evidence_meta": target_evidence_state,
                    "branches": branch_scores,
                    "mad_runtime": mad_payload,
                    "avcd_attention_mass": avcd_mass_payload,
                    "runtime_memory": memory_trace,
                    "probe_full_answer": full_answer,
                    "stored_full_prediction": stored_answer,
                    "probe_full_matches_stored": bool(full_answer == stored_answer) if full_answer and stored_answer else None,
                    "eval_reference_answer": ref_answer,
                    "eval_baseline_correct": base_payload(row).get("correct"),
                    "eval_probe_full_correct": probe_full_correct,
                    "eval_target_modality": str(eval_target),
                    "eval_target_modality_reason": str(eval_target_reason),
                    "eval_prior_regime_label": eval_regime,
                }
            except Exception as exc:
                if is_oom_error(exc):
                    oom_memory = cuda_memory_snapshot()
                    oom_memory["memory_trace"] = memory_trace
                    write_oom_event(
                        args.output_dir / "oom_event.json",
                        sample_id=sample_id,
                        stage=stage_name,
                        exc=exc,
                        runtime_memory=oom_memory,
                    )
                    raise ProbeOomAbort(
                        sample_id=sample_id,
                        stage=stage_name,
                        original_exc=exc,
                        runtime_memory=oom_memory,
                    ) from exc
                if not args.skip_on_error:
                    raise
                record = skipped_record(
                    row=row,
                    sample=sample,
                    formatted_question=question,
                    reason=f"{type(exc).__name__}: {exc}",
                )
            clean_record = strip_method_side_eval_fields(record)
            eval_row = build_eval_sidecar_row(record)
            new_records.append(clean_record)
            new_eval_rows.append(eval_row)
            completed[sample_id] = clean_record
            iterator.set_postfix(
                label=clean_record["directed_prior_label"],
                target=clean_record["target_modality"],
            )
            if args.save_every > 0 and len(new_records) >= args.save_every:
                append_jsonl(records_path, new_records)
                append_jsonl(eval_sidecar_path, new_eval_rows)
                new_records = []
                new_eval_rows = []
    finally:
        if new_records:
            append_jsonl(records_path, new_records)
        if new_eval_rows:
            append_jsonl(eval_sidecar_path, new_eval_rows)
        del adapter
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    final_records = read_jsonl(records_path)
    final_eval_rows = read_jsonl(eval_sidecar_path) if eval_sidecar_path.exists() else []
    final_eval_lookup = {
        safe_text(row.get("sample_id")): row
        for row in final_eval_rows
        if safe_text(row.get("sample_id"))
    }
    summary = summarize(final_records, final_eval_lookup)
    summary.update(
        {
            "records": str(args.records),
            "output_dir": str(args.output_dir),
            "selected_samples": len(plan_rows),
            "eval_sidecar_path": str(eval_sidecar_path),
            "num_shards": int(args.num_shards),
            "shard_index": int(args.shard_index),
            "target_modality_margin_threshold": float(args.target_modality_margin_threshold),
            "source_confidence_gap_threshold": float(args.source_confidence_gap_threshold),
            "source_tie_threshold": float(args.source_tie_threshold),
            "avcd_target_mass_weight": float(args.avcd_target_mass_weight),
            "avcd_source_mass_weight": float(args.avcd_source_mass_weight),
            "repair_joint_both_threshold": float(args.repair_joint_both_threshold),
            "repair_joint_support_threshold": float(args.repair_joint_support_threshold),
            "repair_joint_tie_threshold": float(args.repair_joint_tie_threshold),
            "video_budget": budget,
            "gamma": float(args.gamma),
            "runtime_constraints": {
                "task_family_not_used_for_runtime": True,
                "benchmark_regime_not_used_for_runtime": True,
            },
        }
    )
    write_json(args.output_dir / "summary.json", summary)
    print(f"[strict-online-prior-typing] wrote {records_path}")
    print(f"[strict-online-prior-typing] wrote {args.output_dir / 'summary.json'}")


if __name__ == "__main__":
    run()
