#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
from tqdm.auto import tqdm


ROOT = Path(__file__).resolve().parents[1]
VIDEOLLAMA2_ROOT = ROOT / "third_party" / "VideoLLaMA2"
if str(VIDEOLLAMA2_ROOT) not in sys.path:
    sys.path.insert(0, str(VIDEOLLAMA2_ROOT))

from videollama2 import model_init  # noqa: E402
from videollama2.constants import DEFAULT_AUDIO_TOKEN, DEFAULT_VIDEO_TOKEN  # noqa: E402
from videollama2.mm_utils import KeywordsStoppingCriteria, tokenizer_multimodal_token  # noqa: E402


DEFAULT_MANIFEST = ROOT / "data" / "manifests" / "unary_rebalance_manifest.jsonl"
DEFAULT_POLICY_ROWS: Path | None = None
DEFAULT_OUTPUT_DIR = ROOT / "results" / "videollama2_owp"
DEFAULT_AVHBENCH_DIR = ROOT / "data" / "AVHBench"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)


def append_jsonl(path: Path, row: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def safe_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def normalize_yes_no(value: Any) -> str | None:
    text = safe_text(value).lower()
    if not text:
        return None
    text = text.replace(".", " ").replace(",", " ").replace(":", " ").strip()
    words = text.split()
    if words:
        first = words[0]
        if first in {"yes", "y", "true", "correct", "1"}:
            return "Yes"
        if first in {"no", "n", "false", "incorrect", "0"}:
            return "No"
    if "yes" in text:
        return "Yes"
    if "no" in text:
        return "No"
    return None


def answer_correct(prediction: Any, reference: Any) -> bool | None:
    pred = normalize_yes_no(prediction)
    ref = normalize_yes_no(reference)
    if pred is None or ref is None:
        return None
    return pred == ref


def signed_margin(margin_yes_minus_no: float, answer: Any) -> float | None:
    answer_norm = normalize_yes_no(answer)
    if answer_norm == "Yes":
        return float(margin_yes_minus_no)
    if answer_norm == "No":
        return -float(margin_yes_minus_no)
    return None


def unit(vec: torch.Tensor) -> torch.Tensor:
    vec = vec.detach().float()
    if not bool(torch.isfinite(vec).all().item()):
        raise FloatingPointError("non-finite vector cannot be normalized")
    norm = vec.norm()
    if not math.isfinite(float(norm.item())):
        raise FloatingPointError("non-finite vector norm")
    if float(norm.item()) <= 1.0e-12:
        return torch.zeros_like(vec)
    return vec / norm.clamp_min(1.0e-12)


def parse_layer_weights(text: str) -> dict[int, float]:
    weights: dict[int, float] = {}
    for part in str(text).replace("+", ",").split(","):
        part = part.strip()
        if not part:
            continue
        if ":" not in part:
            raise ValueError(f"Layer weights must use layer:weight form, got {text!r}")
        layer_text, weight_text = part.split(":", 1)
        weights[int(layer_text.strip())] = float(weight_text.strip())
    total = sum(max(0.0, value) for value in weights.values())
    if total <= 0.0:
        raise ValueError(f"Layer weights must sum to a positive value: {text!r}")
    return {layer: max(0.0, value) / total for layer, value in weights.items()}


def derive_audio_path(row: Mapping[str, Any]) -> str:
    explicit = safe_text(row.get("audio_path"))
    if explicit and Path(explicit).exists():
        return explicit
    video_path = safe_text(row.get("video_path"))
    if video_path:
        video = Path(video_path)
        candidates = [
            video.parent.parent / "audios" / f"{video.stem}.wav",
            ROOT / "data" / "AVHBench" / "audios" / f"{video.stem}.wav",
        ]
        for candidate in candidates:
            if candidate.exists():
                return str(candidate)
    sample_id = safe_text(row.get("sample_id"))
    if sample_id:
        stem = sample_id.split(":")[-1]
        candidate = ROOT / "data" / "AVHBench" / "audios" / f"{stem}.wav"
        if candidate.exists():
            return str(candidate)
    return ""


def select_cross_model_rows(
    *,
    manifest_path: Path,
    policy_rows_path: Path,
    max_rows: int | None,
    benchmark: str,
) -> list[dict[str, Any]]:
    manifest_rows = read_jsonl(manifest_path)
    manifest_by_id = {safe_text(row.get("sample_id")): row for row in manifest_rows}
    policy_rows = read_jsonl(policy_rows_path)
    edited_policy_by_id = {
        safe_text(row.get("sample_id")): row
        for row in policy_rows
        if row.get("edit_applied") is True and safe_text(row.get("sample_id"))
    }
    selected: list[dict[str, Any]] = []
    for sample_id in sorted(edited_policy_by_id):
        row = dict(manifest_by_id.get(sample_id) or {})
        if not row:
            continue
        if benchmark and safe_text(row.get("benchmark")).lower() != benchmark.lower():
            continue
        video_path = safe_text(row.get("video_path"))
        if not video_path or not Path(video_path).exists():
            continue
        audio_path = derive_audio_path(row)
        if not audio_path or not Path(audio_path).exists():
            continue
        policy = edited_policy_by_id[sample_id]
        reference = (
            row.get("mad_protocol_reference_answer")
            or row.get("eval_reference_answer")
            or row.get("target_answer")
        )
        row.update(
            {
                "sample_id": sample_id,
                "audio_path": audio_path,
                "reference_answer": reference,
                "qwen_policy_current_answer": policy.get("current_answer_y0"),
                "qwen_policy_candidate_answer": policy.get("candidate_answer_y"),
                "qwen_policy_answer": policy.get("policy_answer") or policy.get("final_answer"),
                "qwen_policy_edit_applied": policy.get("edit_applied"),
            }
        )
        selected.append(row)
        if max_rows is not None and max_rows > 0 and len(selected) >= max_rows:
            break
    return selected


def select_avh_unary_full_rows(*, avhbench_dir: Path, max_rows: int | None) -> list[dict[str, Any]]:
    qa_path = avhbench_dir / "QA.json"
    qa_rows = json.loads(qa_path.read_text(encoding="utf-8"))
    selected: list[dict[str, Any]] = []
    task_to_target = {
        "Video-driven Audio Hallucination": "audio",
        "Audio-driven Video Hallucination": "visual",
    }
    for idx, item in enumerate(qa_rows):
        task = safe_text(item.get("task"))
        target = task_to_target.get(task)
        if target is None:
            continue
        video_id = safe_text(item.get("video_id"))
        video_path = avhbench_dir / "videos" / f"{video_id}.mp4"
        audio_path = avhbench_dir / "audios" / f"{video_id}.wav"
        if not video_path.exists() or not audio_path.exists():
            continue
        selected.append(
            {
                "sample_id": f"avhbench:{idx:05d}:{video_id}",
                "benchmark": "avhbench",
                "mad_protocol_task": task,
                "target_modality": target,
                "question": safe_text(item.get("text")),
                "video_path": str(video_path),
                "audio_path": str(audio_path),
                "reference_answer": item.get("label"),
                "qwen_policy_current_answer": None,
                "qwen_policy_candidate_answer": None,
                "qwen_policy_answer": None,
                "qwen_policy_edit_applied": None,
            }
        )
        if max_rows is not None and max_rows > 0 and len(selected) >= max_rows:
            break
    return selected


def language_layers(model: Any) -> torch.nn.ModuleList:
    base = model.get_model() if hasattr(model, "get_model") else getattr(model, "model")
    layers = getattr(base, "layers", None)
    if layers is None and hasattr(base, "model"):
        layers = getattr(base.model, "layers", None)
    if layers is None:
        raise AttributeError("Cannot locate VideoLLaMA2 language layers for hook-based editing")
    return layers


def single_token_id(tokenizer: Any, candidates: Sequence[str]) -> int:
    for text in candidates:
        token_ids = tokenizer.encode(text, add_special_tokens=False)
        if len(token_ids) == 1:
            return int(token_ids[0])
    encoded = tokenizer.encode(candidates[0], add_special_tokens=False)
    if not encoded:
        raise ValueError(f"Cannot encode token candidates: {candidates}")
    return int(encoded[-1])


def yes_no_token_ids(tokenizer: Any) -> tuple[int, int]:
    yes_id = single_token_id(tokenizer, ["Yes", " yes", "YES", " yes.", "yes"])
    no_id = single_token_id(tokenizer, ["No", " no", "NO", " no.", "no"])
    return yes_id, no_id


def system_message_for(model: Any) -> list[dict[str, str]]:
    if model.config.model_type in {"videollama2", "videollama2_mistral", "videollama2_mixtral"}:
        return [
            {
                "role": "system",
                "content": (
                    "<<SYS>>\nYou are a helpful, respectful and honest assistant. Always answer as helpfully "
                    "as possible, while being safe. Your answers should not include any harmful, unethical, "
                    "racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses "
                    "are socially unbiased and positive in nature.\nIf a question does not make any sense, "
                    "or is not factually coherent, explain why instead of answering something not correct. "
                    "If you don't know the answer to a question, please don't share false information.\n<</SYS>>"
                ),
            }
        ]
    return []


def build_prompt(tokenizer: Any, model: Any, question: str, modal: str) -> tuple[str, str | None]:
    if modal == "video":
        modal_token: str | None = DEFAULT_VIDEO_TOKEN
        content = f"{DEFAULT_VIDEO_TOKEN}\n{question}"
    elif modal == "audio":
        modal_token = DEFAULT_AUDIO_TOKEN
        content = f"{DEFAULT_AUDIO_TOKEN}\n{question}"
    elif modal == "text":
        modal_token = None
        content = question
    else:
        raise ValueError(f"Unsupported modal={modal!r}")
    message = system_message_for(model) + [{"role": "user", "content": content}]
    prompt = tokenizer.apply_chat_template(message, tokenize=False, add_generation_prompt=True)
    return prompt, modal_token


def preprocess_media(processor: Mapping[str, Any], row: Mapping[str, Any], branch: str) -> tuple[Any, str]:
    video_path = safe_text(row.get("video_path"))
    audio_path = safe_text(row.get("audio_path"))
    if branch == "full":
        return processor["video"](video_path, va=True), "video"
    if branch == "visual":
        return processor["video"](video_path, va=False), "video"
    if branch == "audio":
        return processor["audio"](audio_path), "audio"
    if branch == "text":
        return None, "text"
    raise ValueError(f"Unsupported branch={branch!r}")


def torch_dtype_from_name(name: str) -> torch.dtype:
    normalized = str(name).lower().strip()
    if normalized in {"fp16", "float16", "half"}:
        return torch.float16
    if normalized in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if normalized in {"fp32", "float32"}:
        return torch.float32
    raise ValueError(f"Unsupported dtype={name!r}")


def tensor_to_cuda(media: Any, modal: str, *, dtype: torch.dtype) -> list[tuple[Any, str]] | None:
    if modal == "text":
        return None
    if isinstance(media, dict):
        tensor = {key: value.to(dtype=dtype).cuda() for key, value in media.items()}
    else:
        tensor = media.to(dtype=dtype).cuda()
    return [(tensor, modal)]


def prepare_inputs(
    *,
    tokenizer: Any,
    model: Any,
    processor: Mapping[str, Any],
    row: Mapping[str, Any],
    branch: str,
    dtype: torch.dtype,
) -> dict[str, Any]:
    question = safe_text(row.get("question"))
    if "answer" not in question.lower()[-40:]:
        question = question + " Answer only 'Yes' or 'No'."
    media, modal = preprocess_media(processor, row, branch)
    prompt, modal_token = build_prompt(tokenizer, model, question, modal)
    input_ids = tokenizer_multimodal_token(prompt, tokenizer, modal_token, return_tensors="pt")
    input_ids = input_ids.unsqueeze(0).long().cuda()
    attention_mask = input_ids.ne(tokenizer.pad_token_id).long().cuda()
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "images": tensor_to_cuda(media, modal, dtype=dtype),
    }


def logits_payload(logits: torch.Tensor, yes_id: int, no_id: int) -> dict[str, Any]:
    pair = logits[:, -1, [yes_id, no_id]].detach().float().cpu().reshape(-1)
    if not bool(torch.isfinite(pair).all().item()):
        raise FloatingPointError("non-finite yes/no logits")
    yes_logit = float(pair[0].item())
    no_logit = float(pair[1].item())
    probs = torch.softmax(pair, dim=0)
    margin = yes_logit - no_logit
    return {
        "yes_logit": yes_logit,
        "no_logit": no_logit,
        "yes_prob": float(probs[0].item()),
        "no_prob": float(probs[1].item()),
        "margin_yes_minus_no": float(margin),
        "prediction": "Yes" if margin >= 0.0 else "No",
    }


@torch.no_grad()
def capture_branch(
    *,
    model: Any,
    tokenizer: Any,
    processor: Mapping[str, Any],
    row: Mapping[str, Any],
    branch: str,
    layers: Sequence[int],
    yes_id: int,
    no_id: int,
    dtype: torch.dtype,
) -> dict[str, Any]:
    inputs = prepare_inputs(
        tokenizer=tokenizer,
        model=model,
        processor=processor,
        row=row,
        branch=branch,
        dtype=dtype,
    )
    captured: dict[int, torch.Tensor] = {}
    hooks = []
    modules = language_layers(model)

    def make_hook(layer_id: int):
        def hook_fn(_module, _inp, output):
            hidden = output[0] if isinstance(output, (tuple, list)) else output
            if isinstance(hidden, torch.Tensor):
                vec = hidden[0, -1, :].detach().float().cpu()
                if not bool(torch.isfinite(vec).all().item()):
                    raise FloatingPointError(f"non-finite hidden state at layer {layer_id}")
                captured[int(layer_id)] = vec
            return None

        return hook_fn

    try:
        for layer in layers:
            hooks.append(modules[int(layer)].register_forward_hook(make_hook(int(layer))))
        outputs = model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            images=inputs["images"],
            use_cache=False,
            return_dict=True,
        )
    finally:
        for hook in hooks:
            hook.remove()
    payload = logits_payload(outputs.logits, yes_id, no_id)
    return {
        "branch": branch,
        "layer_vectors": captured,
        "final_prediction": payload["prediction"],
        "final_yes_no_margin": payload["margin_yes_minus_no"],
        "yes_logit": payload["yes_logit"],
        "no_logit": payload["no_logit"],
    }


@torch.no_grad()
def patched_full_logits(
    *,
    model: Any,
    tokenizer: Any,
    processor: Mapping[str, Any],
    row: Mapping[str, Any],
    layers: Sequence[int],
    layer_weights: Mapping[int, float],
    prior_directions: Mapping[int, torch.Tensor],
    beta: float,
    yes_id: int,
    no_id: int,
    dtype: torch.dtype,
) -> dict[str, Any]:
    inputs = prepare_inputs(
        tokenizer=tokenizer,
        model=model,
        processor=processor,
        row=row,
        branch="full",
        dtype=dtype,
    )
    modules = language_layers(model)
    hooks = []
    debug_by_layer: dict[str, dict[str, Any]] = {}

    def make_hook(layer_id: int):
        p_cpu = prior_directions[int(layer_id)].detach().float().cpu()
        if not bool(torch.isfinite(p_cpu).all().item()):
            raise FloatingPointError(f"non-finite prior direction at layer {layer_id}")
        beta_for_layer = float(beta) * float(layer_weights[int(layer_id)])

        def hook_fn(_module, _inp, output):
            hidden = output[0] if isinstance(output, (tuple, list)) else output
            if not isinstance(hidden, torch.Tensor):
                return None
            patched = hidden.clone()
            base = patched[0, -1, :].float()
            if not bool(torch.isfinite(base).all().item()):
                raise FloatingPointError(f"non-finite base hidden at layer {layer_id}")
            p_dir = unit(p_cpu).to(device=base.device, dtype=torch.float32)
            projection = torch.dot(base, p_dir) * p_dir
            updated = base - beta_for_layer * projection
            patched[0, -1, :] = updated.to(dtype=patched.dtype)
            debug_by_layer[str(layer_id)] = {
                "layer": int(layer_id),
                "layer_weight": float(layer_weights[int(layer_id)]),
                "beta_for_layer": beta_for_layer,
                "prior_direction_norm": float(p_cpu.norm().item()),
                "prior_projection_before": float(torch.dot(base, p_dir).item()),
                "prior_projection_after": float(torch.dot(updated, p_dir).item()),
                "delta_l2": float((updated - base).norm().item()),
                "base_norm": float(base.norm().item()),
                "updated_norm": float(updated.norm().item()),
            }
            if isinstance(output, tuple):
                return (patched,) + tuple(output[1:])
            if isinstance(output, list):
                return [patched] + list(output[1:])
            return patched

        return hook_fn

    try:
        for layer in layers:
            hooks.append(modules[int(layer)].register_forward_hook(make_hook(int(layer))))
        outputs = model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            images=inputs["images"],
            use_cache=False,
            return_dict=True,
        )
    finally:
        for hook in hooks:
            hook.remove()
    payload = logits_payload(outputs.logits, yes_id, no_id)
    payload["debug_by_layer"] = debug_by_layer
    return payload


def target_branch_for(row: Mapping[str, Any]) -> str:
    target = safe_text(row.get("target_modality")).lower()
    if target == "audio":
        return "audio"
    if target in {"visual", "video", "vision"}:
        return "visual"
    raise ValueError(f"Unsupported target_modality={target!r} sample_id={row.get('sample_id')}")


def non_target_branch_for(row: Mapping[str, Any]) -> str:
    target = target_branch_for(row)
    return "visual" if target == "audio" else "audio"


def decide_runtime_gate(
    *,
    baseline_prediction: str,
    target_prediction: str,
    non_target_prediction: str,
    text_prediction: str,
) -> dict[str, Any]:
    supporting_branches: list[str] = []
    if non_target_prediction == baseline_prediction:
        supporting_branches.append("non_target")
    if text_prediction == baseline_prediction:
        supporting_branches.append("text")
    target_conflicts = target_prediction != baseline_prediction
    edit_applied = bool(target_conflicts and supporting_branches)
    if edit_applied:
        reason = "target_conflicts_with_current_and_prior_supports_current"
    elif not target_conflicts:
        reason = "target_agrees_with_current"
    else:
        reason = "target_conflicts_but_no_prior_support"
    return {
        "edit_applied": edit_applied,
        "gate_reason": reason,
        "gate_target_conflicts_current": bool(target_conflicts),
        "gate_prior_supporting_branches": supporting_branches,
    }


def summarize(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    def acc(prefix: str, items: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        valid = [row for row in items if row.get(f"{prefix}_correct") is not None]
        correct = sum(1 for row in valid if row.get(f"{prefix}_correct") is True)
        total = len(valid)
        return {
            "correct": int(correct),
            "total": int(total),
            "accuracy": float(correct / total) if total else None,
        }

    baseline = acc("baseline", rows)
    edited = acc("edited", rows)
    w2c = sum(
        1
        for row in rows
        if row.get("baseline_correct") is False and row.get("edited_correct") is True
    )
    c2w = sum(
        1
        for row in rows
        if row.get("baseline_correct") is True and row.get("edited_correct") is False
    )
    unchanged = sum(
        1
        for row in rows
        if row.get("baseline_prediction") == row.get("edited_prediction")
    )
    raw_edited = acc("raw_edited", rows)
    edit_applied = sum(1 for row in rows if row.get("edit_applied") is True)
    gate_reasons = Counter(safe_text(row.get("gate_reason")) for row in rows)
    by_target = {}
    for target, group in groupby_key(rows, "target_modality").items():
        by_target[target] = {
            "baseline": acc("baseline", group),
            "edited": acc("edited", group),
            "w2c": sum(1 for row in group if row.get("baseline_correct") is False and row.get("edited_correct") is True),
            "c2w": sum(1 for row in group if row.get("baseline_correct") is True and row.get("edited_correct") is False),
        }
        if by_target[target]["baseline"]["accuracy"] is not None and by_target[target]["edited"]["accuracy"] is not None:
            by_target[target]["delta_pp"] = 100.0 * (
                by_target[target]["edited"]["accuracy"] - by_target[target]["baseline"]["accuracy"]
            )
    by_task = {}
    for task, group in groupby_key(rows, "mad_protocol_task").items():
        by_task[task] = {
            "baseline": acc("baseline", group),
            "edited": acc("edited", group),
            "w2c": sum(1 for row in group if row.get("baseline_correct") is False and row.get("edited_correct") is True),
            "c2w": sum(1 for row in group if row.get("baseline_correct") is True and row.get("edited_correct") is False),
        }
        if by_task[task]["baseline"]["accuracy"] is not None and by_task[task]["edited"]["accuracy"] is not None:
            by_task[task]["delta_pp"] = 100.0 * (
                by_task[task]["edited"]["accuracy"] - by_task[task]["baseline"]["accuracy"]
            )
    delta_pp = None
    if baseline["accuracy"] is not None and edited["accuracy"] is not None:
        delta_pp = 100.0 * (edited["accuracy"] - baseline["accuracy"])
    return {
        "n_rows": len(rows),
        "baseline": baseline,
        "edited": edited,
        "raw_edited": raw_edited,
        "delta_pp": delta_pp,
        "w2c": int(w2c),
        "c2w": int(c2w),
        "unchanged_prediction": int(unchanged),
        "edit_applied": int(edit_applied),
        "edit_applied_rate": float(edit_applied / len(rows)) if rows else None,
        "gate_reasons": dict(gate_reasons),
        "by_target_modality": by_target,
        "by_task": by_task,
        "qwen_policy_reference": {
            "note": "Qwen policy fields are copied only for offline comparison; they are not used for VideoLLaMA2 runtime routing or editing.",
            "qwen_policy_answer_counts": dict(Counter(safe_text(row.get("qwen_policy_answer")) for row in rows)),
        },
    }


def groupby_key(rows: Sequence[Mapping[str, Any]], key: str) -> dict[str, list[Mapping[str, Any]]]:
    out: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        out[safe_text(row.get(key)) or "UNKNOWN"].append(row)
    return dict(out)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="VideoLLaMA2-AV OWP transfer runtime.")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--policy-rows", type=Path, default=DEFAULT_POLICY_ROWS)
    parser.add_argument("--avhbench-dir", type=Path, default=DEFAULT_AVHBENCH_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--model-path", type=str, default="DAMO-NLP-SG/VideoLLaMA2.1-7B-AV"
    )
    parser.add_argument("--benchmark", type=str, default="avhbench")
    parser.add_argument("--selection-mode", choices=["qwen_edit_applied", "avh_unary_full"], default="avh_unary_full")
    parser.add_argument("--runtime-gate", choices=["none", "target_conflict_prior_support"], default="none")
    parser.add_argument("--max-rows", type=int, default=300)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--layers", type=int, nargs="+", default=[24, 26, 27])
    parser.add_argument("--layer-weights", type=str, default="24:0.23,26:0.61,27:0.16")
    parser.add_argument("--beta", type=float, default=0.65)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--dtype", choices=["float16", "bfloat16", "float32"], default="float16")
    parser.add_argument("--load-8bit", action="store_true")
    parser.add_argument("--load-4bit", action="store_true")
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--no-progress", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    layer_weights = parse_layer_weights(args.layer_weights)
    dtype = torch_dtype_from_name(args.dtype)
    layers = [int(layer) for layer in args.layers]
    missing_weights = [layer for layer in layers if layer not in layer_weights]
    if missing_weights:
        raise ValueError(f"Missing layer weights for layers {missing_weights}")
    max_rows = int(args.max_rows) if int(args.max_rows) > 0 else None
    if args.selection_mode == "avh_unary_full":
        selected_all = select_avh_unary_full_rows(avhbench_dir=args.avhbench_dir, max_rows=max_rows)
    else:
        selected_all = select_cross_model_rows(
            manifest_path=args.manifest,
            policy_rows_path=args.policy_rows,
            max_rows=max_rows,
            benchmark=str(args.benchmark),
        )
    selected = [
        row
        for idx, row in enumerate(selected_all)
        if idx % int(args.num_shards) == int(args.shard_index)
    ]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    run_config = {
        "kind": "videollama2_owp",
        "model_path": args.model_path,
        "manifest": str(args.manifest),
        "policy_rows": str(args.policy_rows) if args.policy_rows is not None else None,
        "benchmark": args.benchmark,
        "n_selected_all": len(selected_all),
        "n_selected_shard": len(selected),
        "num_shards": int(args.num_shards),
        "shard_index": int(args.shard_index),
        "selection_mode": args.selection_mode,
        "runtime_gate": args.runtime_gate,
        "runtime_uses_reference_answer": False,
        "runtime_uses_qwen_candidate_answer": False,
        "runtime_uses_task_family": False,
        "prior_direction": "full_minus_target",
        "prior_protection": "raw",
        "alpha": 0.0,
        "beta": float(args.beta),
        "layers": layers,
        "layer_weights": {str(layer): float(layer_weights[layer]) for layer in layers},
        "dtype": str(args.dtype),
        "selection_note": (
            "Rows are all AVHBench unary hallucination samples from QA.json; VideoLLaMA2 runtime "
            "branches decide conflict/gate. No Qwen policy is used."
            if args.selection_mode == "avh_unary_full"
            else "Rows are fixed from Qwen frozen mainline edit_applied conflict set, then filtered to benchmark/media availability. No VideoLLaMA2 outcome is used for selection."
        ),
    }
    write_json(args.output_dir / "run_config.json", run_config)
    with (args.output_dir / "selected_rows.jsonl").open("w", encoding="utf-8") as handle:
        for row in selected:
            keep = {
                key: row.get(key)
                for key in (
                    "sample_id",
                    "benchmark",
                    "mad_protocol_task",
                    "target_modality",
                    "question",
                    "video_path",
                    "audio_path",
                    "reference_answer",
                    "qwen_policy_answer",
                    "qwen_policy_current_answer",
                    "qwen_policy_candidate_answer",
                )
            }
            handle.write(json.dumps(keep, ensure_ascii=False, sort_keys=True) + "\n")
    if args.plan_only:
        print(json.dumps(run_config, ensure_ascii=False, indent=2))
        return

    model, processor, tokenizer = model_init(
        args.model_path,
        device=args.device,
        torch_dtype=dtype,
        attn_implementation="sdpa",
        load_8bit=bool(args.load_8bit),
        load_4bit=bool(args.load_4bit),
    )
    model.eval()
    yes_id, no_id = yes_no_token_ids(tokenizer)
    run_config.update(
        {
            "yes_token_id": int(yes_id),
            "no_token_id": int(no_id),
            "model_type": safe_text(getattr(model.config, "model_type", "")),
            "num_language_layers": int(len(language_layers(model))),
        }
    )
    write_json(args.output_dir / "run_config.json", run_config)
    if max(layers) >= len(language_layers(model)):
        raise ValueError(f"Layer index out of range: layers={layers}, model_layers={len(language_layers(model))}")

    rows_path = args.output_dir / "rows.jsonl"
    rows_path.write_text("", encoding="utf-8")
    output_rows: list[dict[str, Any]] = []
    iterator = tqdm(selected, desc=f"videollama2_owp_s{args.shard_index}", disable=bool(args.no_progress), unit="sample")
    for row in iterator:
        start = time.time()
        sample_id = safe_text(row.get("sample_id"))
        target_branch = target_branch_for(row)
        non_target_branch = non_target_branch_for(row)
        try:
            captures = {
                "full": capture_branch(
                    model=model,
                    tokenizer=tokenizer,
                    processor=processor,
                    row=row,
                    branch="full",
                    layers=layers,
                    yes_id=yes_id,
                    no_id=no_id,
                    dtype=dtype,
                ),
                "target": capture_branch(
                    model=model,
                    tokenizer=tokenizer,
                    processor=processor,
                    row=row,
                    branch=target_branch,
                    layers=layers,
                    yes_id=yes_id,
                    no_id=no_id,
                    dtype=dtype,
                ),
                "non_target": capture_branch(
                    model=model,
                    tokenizer=tokenizer,
                    processor=processor,
                    row=row,
                    branch=non_target_branch,
                    layers=layers,
                    yes_id=yes_id,
                    no_id=no_id,
                    dtype=dtype,
                ),
                "text": capture_branch(
                    model=model,
                    tokenizer=tokenizer,
                    processor=processor,
                    row=row,
                    branch="text",
                    layers=layers,
                    yes_id=yes_id,
                    no_id=no_id,
                    dtype=dtype,
                ),
            }
            prior_directions: dict[int, torch.Tensor] = {}
            carrier_debug: dict[str, dict[str, Any]] = {}
            for layer in layers:
                full_vec = captures["full"]["layer_vectors"][int(layer)]
                target_vec = captures["target"]["layer_vectors"][int(layer)]
                prior = full_vec - target_vec
                prior_directions[int(layer)] = prior
                text_vec = captures["text"]["layer_vectors"][int(layer)]
                non_target_vec = captures["non_target"]["layer_vectors"][int(layer)]
                evidence = target_vec - text_vec
                carrier_debug[str(layer)] = {
                    "prior_norm": float(prior.norm().item()),
                    "evidence_norm": float(evidence.norm().item()),
                    "non_target_minus_text_norm": float((non_target_vec - text_vec).norm().item()),
                    "cos_prior_evidence": float(torch.dot(unit(prior), unit(evidence)).item()),
                    "full_prior_projection": float(torch.dot(full_vec.float(), unit(prior)).item()),
                }
            edited = patched_full_logits(
                model=model,
                tokenizer=tokenizer,
                processor=processor,
                row=row,
                layers=layers,
                layer_weights=layer_weights,
                prior_directions=prior_directions,
                beta=float(args.beta),
                yes_id=yes_id,
                no_id=no_id,
                dtype=dtype,
            )
            baseline_pred = captures["full"]["final_prediction"]
            baseline_margin = float(captures["full"]["final_yes_no_margin"])
            raw_edited_pred = edited["prediction"]
            raw_edited_margin = float(edited["margin_yes_minus_no"])
            gate = {
                "edit_applied": True,
                "gate_reason": "runtime_gate_disabled",
                "gate_target_conflicts_current": None,
                "gate_prior_supporting_branches": [],
            }
            if args.runtime_gate == "target_conflict_prior_support":
                gate = decide_runtime_gate(
                    baseline_prediction=baseline_pred,
                    target_prediction=captures["target"]["final_prediction"],
                    non_target_prediction=captures["non_target"]["final_prediction"],
                    text_prediction=captures["text"]["final_prediction"],
                )
            if gate["edit_applied"]:
                edited_pred = raw_edited_pred
                edited_margin = raw_edited_margin
            else:
                edited_pred = baseline_pred
                edited_margin = baseline_margin
            reference = row.get("reference_answer")
            output = {
                "sample_id": sample_id,
                "benchmark": row.get("benchmark"),
                "mad_protocol_task": row.get("mad_protocol_task"),
                "target_modality": row.get("target_modality"),
                "target_branch": target_branch,
                "non_target_branch": non_target_branch,
                "question": row.get("question"),
                "video_path": row.get("video_path"),
                "audio_path": row.get("audio_path"),
                "reference_answer": reference,
                "baseline_prediction": baseline_pred,
                "baseline_margin": baseline_margin,
                "baseline_correct": answer_correct(baseline_pred, reference),
                "raw_edited_prediction": raw_edited_pred,
                "raw_edited_margin": raw_edited_margin,
                "raw_edited_correct": answer_correct(raw_edited_pred, reference),
                "edited_prediction": edited_pred,
                "edited_margin": edited_margin,
                "edited_correct": answer_correct(edited_pred, reference),
                "delta_margin": edited_margin - baseline_margin,
                "baseline_reference_aligned_margin": signed_margin(baseline_margin, reference),
                "edited_reference_aligned_margin": signed_margin(edited_margin, reference),
                "target_branch_prediction": captures["target"]["final_prediction"],
                "target_branch_margin": captures["target"]["final_yes_no_margin"],
                "non_target_branch_prediction": captures["non_target"]["final_prediction"],
                "non_target_branch_margin": captures["non_target"]["final_yes_no_margin"],
                "text_branch_prediction": captures["text"]["final_prediction"],
                "text_branch_margin": captures["text"]["final_yes_no_margin"],
                "prior_direction": "full_minus_target",
                "prior_protection": "raw",
                "edit_applied": bool(gate["edit_applied"]),
                "gate_reason": gate["gate_reason"],
                "gate_target_conflicts_current": gate["gate_target_conflicts_current"],
                "gate_prior_supporting_branches": gate["gate_prior_supporting_branches"],
                "runtime_gate": args.runtime_gate,
                "alpha": 0.0,
                "beta": float(args.beta),
                "layers": layers,
                "layer_weights": {str(layer): float(layer_weights[layer]) for layer in layers},
                "carrier_debug": carrier_debug,
                "patch_debug_by_layer": edited.get("debug_by_layer"),
                "qwen_policy_answer": row.get("qwen_policy_answer"),
                "qwen_policy_current_answer": row.get("qwen_policy_current_answer"),
                "qwen_policy_candidate_answer": row.get("qwen_policy_candidate_answer"),
                "runtime_seconds": time.time() - start,
                "error": "",
            }
        except Exception as exc:  # Keep shards progressing; failed rows are auditable.
            output = {
                "sample_id": sample_id,
                "benchmark": row.get("benchmark"),
                "mad_protocol_task": row.get("mad_protocol_task"),
                "target_modality": row.get("target_modality"),
                "question": row.get("question"),
                "reference_answer": row.get("reference_answer"),
                "error": repr(exc),
                "runtime_seconds": time.time() - start,
            }
        append_jsonl(rows_path, output)
        output_rows.append(output)

    ok_rows = [row for row in output_rows if not row.get("error")]
    summary = summarize(ok_rows)
    summary.update(
        {
            "n_errors": len(output_rows) - len(ok_rows),
            "errors": [row for row in output_rows if row.get("error")][:20],
            "run_config": run_config,
        }
    )
    write_json(args.output_dir / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2)[:4000])


if __name__ == "__main__":
    main()
