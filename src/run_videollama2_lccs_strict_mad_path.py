#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import json
import os
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
import transformers
from tqdm.auto import tqdm


ROOT = Path(__file__).resolve().parents[1]
VIDEOLLAMA2_ROOT = ROOT / "third_party" / "VideoLLaMA2"
if str(VIDEOLLAMA2_ROOT) not in sys.path:
    sys.path.insert(0, str(VIDEOLLAMA2_ROOT))

DEFAULT_CMM_AUDIO_CACHE_DIR = ROOT / "results" / "cache" / "cmm_videollama2_audio_16k"

from videollama2 import YesNoStoppingCriteria, model_init  # noqa: E402
from videollama2.constants import DEFAULT_AUDIO_TOKEN, DEFAULT_VIDEO_TOKEN  # noqa: E402
from videollama2.mm_utils import KeywordsStoppingCriteria, tokenizer_multimodal_token  # noqa: E402

from run_videollama2_lccs_cross_model import (  # noqa: E402
    normalize_yes_no,
    parse_layer_weights,
    safe_text,
    select_avh_unary_full_rows,
    torch_dtype_from_name,
    unit,
)


def answer_correct_strict(prediction: Any, reference: Any) -> bool:
    pred = normalize_yes_no(prediction)
    ref = normalize_yes_no(reference)
    if ref is None:
        raise ValueError(f"Reference is not yes/no: {reference!r}")
    return pred == ref


def answer_sign(answer: Any) -> int | None:
    pred = normalize_yes_no(answer)
    if pred == "Yes":
        return 1
    if pred == "No":
        return -1
    return None


def margin_from_debug(debug: Mapping[str, Any]) -> float | None:
    if "yes_score" not in debug or "no_score" not in debug:
        return None
    return float(debug["yes_score"]) - float(debug["no_score"])


def aligned_margin(answer: Any, margin_yes_minus_no: float | None) -> float | None:
    sign = answer_sign(answer)
    if sign is None or margin_yes_minus_no is None:
        return None
    return float(sign) * float(margin_yes_minus_no)


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)


def append_jsonl(path: Path, row: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def official_video_id(row: Mapping[str, Any]) -> str:
    video_id = safe_text(row.get("video_id"))
    if video_id:
        return video_id
    video_path = safe_text(row.get("video_path"))
    if video_path:
        return Path(video_path).stem
    sample_id = safe_text(row.get("sample_id"))
    if sample_id:
        return sample_id.split(":")[-1]
    return ""


def official_prompt_question(row: Mapping[str, Any]) -> str:
    question = safe_text(row.get("question"))
    suffix = " Answer only 'Yes' or 'No'."
    if "answer" not in question.lower()[-40:]:
        question = question + suffix
    return question


def official_score_row(
    *,
    row: Mapping[str, Any],
    prediction: str,
    runtime_seconds: float,
    device: str,
) -> dict[str, Any]:
    return {
        "video_id": official_video_id(row),
        "task": row.get("mad_protocol_task"),
        "question": official_prompt_question(row),
        "ground_truth": row.get("reference_answer"),
        "prediction": prediction,
        "inference_time": float(runtime_seconds),
        "device": device,
    }


def official_cmm_score_row(
    *,
    row: Mapping[str, Any],
    prediction: str,
) -> dict[str, Any]:
    return {
        "category": row.get("category", ""),
        "sub_category": row.get("sub_category", ""),
        "modality": row.get("modality", ""),
        "granularity": row.get("granularity", ""),
        "correlation_type": row.get("correlation_type", ""),
        "video_path": row.get("relative_video_path") or row.get("video_path", ""),
        "audio_path": row.get("relative_audio_path") or row.get("audio_path", ""),
        "question": row.get("question", ""),
        "answer": row.get("reference_answer"),
        "pred": prediction,
    }


def select_cmm_unary_full_rows(*, cmm_data_path: Path, cmm_media_base_dir: Path, max_rows: int | None) -> list[dict[str, Any]]:
    rows = json.loads(cmm_data_path.read_text(encoding="utf-8"))
    selected: list[dict[str, Any]] = []
    audio_cache_dir = Path(os.environ.get("VIDEOLLAMA2_CMM_AUDIO_CACHE_DIR", str(DEFAULT_CMM_AUDIO_CACHE_DIR)))
    target_by_sub_category = {
        "overrely_visual_ignore_audio": "audio",
        "overrely_audio_ignore_visual": "visual",
        "overrely_language_ignore_visual": "visual",
    }
    for idx, item in enumerate(rows):
        if safe_text(item.get("category")) != "over-reliance_unimodal_priors":
            continue
        sub_category = safe_text(item.get("sub_category"))
        target = target_by_sub_category.get(sub_category)
        if target is None:
            continue
        rel_video = safe_text(item.get("video_path"))
        rel_audio = safe_text(item.get("audio_path"))
        video_rel_clean = rel_video[2:] if rel_video.startswith("./") else rel_video
        audio_rel_clean = rel_audio[2:] if rel_audio.startswith("./") else rel_audio
        video_path = cmm_media_base_dir / video_rel_clean
        if not video_path.is_file():
            continue
        if audio_rel_clean:
            audio_path = cmm_media_base_dir / audio_rel_clean
            if not audio_path.is_file():
                continue
            resolved_rel_audio = rel_audio
        else:
            audio_path = audio_cache_dir / Path(video_rel_clean).with_suffix(".wav")
            if not audio_path.is_file():
                continue
            resolved_rel_audio = str(audio_path)
        selected.append(
            {
                "sample_id": f"cmm:{idx:05d}",
                "benchmark": "cmm",
                "mad_protocol_task": sub_category,
                "target_modality": target,
                "question": safe_text(item.get("question")),
                "video_path": str(video_path),
                "audio_path": str(audio_path),
                "relative_video_path": rel_video,
                "relative_audio_path": resolved_rel_audio,
                "reference_answer": item.get("answer"),
                "category": item.get("category", ""),
                "sub_category": item.get("sub_category", ""),
                "modality": item.get("modality", ""),
                "granularity": item.get("granularity", ""),
                "correlation_type": item.get("correlation_type", ""),
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
        raise AttributeError("Cannot locate VideoLLaMA2 language layers")
    return layers


def prepare_media(processor: Mapping[str, Any], row: Mapping[str, Any], branch: str) -> tuple[Any, str, str | None]:
    video_path = safe_text(row.get("video_path"))
    audio_path = safe_text(row.get("audio_path"))
    if branch == "full":
        if audio_path and Path(audio_path).is_file():
            return {"video": processor["video"](video_path, va=False), "audio": processor["audio"](audio_path)}, "video", DEFAULT_VIDEO_TOKEN
        return processor["video"](video_path, va=True), "video", DEFAULT_VIDEO_TOKEN
    if branch == "visual":
        return processor["video"](video_path, va=False), "video", DEFAULT_VIDEO_TOKEN
    if branch == "audio":
        return processor["audio"](audio_path), "audio", DEFAULT_AUDIO_TOKEN
    if branch == "text":
        return None, "text", ""
    raise ValueError(f"Unsupported branch={branch!r}")


def build_official_prompt(tokenizer: Any, model: Any, question: str, modal_token: str | None) -> str:
    content = (modal_token or "") + "\n" + question
    message = [{"role": "user", "content": content}]
    if model.config.model_type in ["videollama2", "videollama2_mistral", "videollama2_mixtral"]:
        system_message = [
            {
                "role": "system",
                "content": (
                    "<<SYS>>\nYou are a helpful, respectful and honest assistant. Always answer as helpfully "
                    "as possible, while being safe.  Your answers should not include any harmful, unethical, "
                    "racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses "
                    "are socially unbiased and positive in nature.\nIf a question does not make any sense, "
                    "or is not factually coherent, explain why instead of answering something not correct. "
                    "If you don't know the answer to a question, please don't share false information.\n<</SYS>>"
                ),
            }
        ]
    else:
        system_message = []
    return tokenizer.apply_chat_template(system_message + message, tokenize=False, add_generation_prompt=True)


def candidate_token_ids(tokenizer: Any, answer: str) -> list[int]:
    ids = tokenizer.encode(answer, add_special_tokens=False)
    if not ids:
        raise ValueError(f"empty tokenization for answer={answer!r}")
    return [int(x) for x in ids]


def score_candidate_sequence(
    *,
    model: Any,
    first_logits: torch.Tensor,
    first_past: Any,
    token_ids: Sequence[int],
) -> float:
    score = 0.0
    logits = first_logits
    past = first_past
    for idx, token_id in enumerate(token_ids):
        score += float(torch.log_softmax(logits.float(), dim=-1)[int(token_id)].item())
        if idx == len(token_ids) - 1:
            break
        next_tok = torch.tensor([[int(token_id)]], device=logits.device, dtype=torch.long)
        out = model(
            next_tok,
            attention_mask=None,
            images=None,
            use_cache=True,
            past_key_values=past,
        )
        past = out.past_key_values
        logits = out.logits[0, -1, :]
    return score


def tensor_to_official_cuda(media: Any, modal: str, *, dtype: torch.dtype) -> list[tuple[Any, str]] | None:
    if modal == "text":
        return None
    if isinstance(media, dict):
        tensor = {key: value.to(dtype=dtype).cuda() for key, value in media.items()}
    else:
        tensor = media.to(dtype=dtype).cuda()
    return [(tensor, modal)]


def prepare_inputs(
    *,
    model: Any,
    tokenizer: Any,
    processor: Mapping[str, Any],
    row: Mapping[str, Any],
    branch: str,
    dtype: torch.dtype,
) -> dict[str, Any]:
    question = safe_text(row.get("question"))
    if "answer" not in question.lower()[-40:]:
        question = question + " Answer only 'Yes' or 'No'."
    media, modal, modal_token = prepare_media(processor, row, branch)
    prompt = build_official_prompt(tokenizer, model, question, modal_token)
    input_ids = tokenizer_multimodal_token(prompt, tokenizer, modal_token, return_tensors="pt").unsqueeze(0).long().cuda()
    attention_mask = input_ids.ne(tokenizer.pad_token_id).long().cuda()
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "images": tensor_to_official_cuda(media, modal, dtype=dtype),
        "prompt_question": question,
    }


@torch.inference_mode()
def generate_answer(
    *,
    model: Any,
    tokenizer: Any,
    processor: Mapping[str, Any],
    row: Mapping[str, Any],
    branch: str,
    patch_layers: Sequence[int] | None = None,
    layer_weights: Mapping[int, float] | None = None,
    prior_directions: Mapping[int, torch.Tensor] | None = None,
    beta: float = 0.0,
    max_new_tokens: int = 1,
    decision_mode: str = "generate",
    dtype: torch.dtype,
) -> tuple[str, dict[str, Any]]:
    inputs = prepare_inputs(model=model, tokenizer=tokenizer, processor=processor, row=row, branch=branch, dtype=dtype)
    keywords = [tokenizer.eos_token]
    stopping_criteria = [
        KeywordsStoppingCriteria(keywords, tokenizer, inputs["input_ids"]),
        YesNoStoppingCriteria(tokenizer, inputs["input_ids"].shape[-1]),
    ]
    hooks = []
    debug_by_layer: dict[str, dict[str, Any]] = {}
    if patch_layers:
        modules = language_layers(model)
        assert layer_weights is not None
        assert prior_directions is not None

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
                }
                if isinstance(output, tuple):
                    return (patched,) + tuple(output[1:])
                if isinstance(output, list):
                    return [patched] + list(output[1:])
                return patched

            return hook_fn

        for layer in patch_layers:
            hooks.append(modules[int(layer)].register_forward_hook(make_hook(int(layer))))
    try:
        if decision_mode == "constrained_yesno":
            out = model(
                inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                images=inputs["images"],
                use_cache=True,
                return_dict=True,
                pad_token_id=tokenizer.eos_token_id,
            )
            logits = out.logits[0, -1, :]
            yes_score = score_candidate_sequence(
                model=model,
                first_logits=logits,
                first_past=out.past_key_values,
                token_ids=candidate_token_ids(tokenizer, "Yes"),
            )
            no_score = score_candidate_sequence(
                model=model,
                first_logits=logits,
                first_past=out.past_key_values,
                token_ids=candidate_token_ids(tokenizer, "No"),
            )
            output = "Yes" if yes_score >= no_score else "No"
            return output, {
                "debug_by_layer": debug_by_layer,
                "prompt_question": inputs["prompt_question"],
                "yes_score": yes_score,
                "no_score": no_score,
            }
        output_ids = model.generate(
            inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            images=inputs["images"],
            do_sample=False,
            max_new_tokens=int(max_new_tokens),
            use_cache=True,
            stopping_criteria=stopping_criteria,
            pad_token_id=tokenizer.eos_token_id,
        )
    finally:
        for hook in hooks:
            hook.remove()
    output = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
    return output, {"debug_by_layer": debug_by_layer, "prompt_question": inputs["prompt_question"]}


@torch.no_grad()
def capture_hidden(
    *,
    model: Any,
    tokenizer: Any,
    processor: Mapping[str, Any],
    row: Mapping[str, Any],
    branch: str,
    layers: Sequence[int],
    dtype: torch.dtype,
) -> dict[int, torch.Tensor]:
    inputs = prepare_inputs(model=model, tokenizer=tokenizer, processor=processor, row=row, branch=branch, dtype=dtype)
    modules = language_layers(model)
    captured: dict[int, torch.Tensor] = {}
    hooks = []

    def make_hook(layer_id: int):
        def hook_fn(_module, _inp, output):
            hidden = output[0] if isinstance(output, (tuple, list)) else output
            if isinstance(hidden, torch.Tensor):
                vec = hidden[0, -1, :].detach().float().cpu()
                if not bool(torch.isfinite(vec).all().item()):
                    raise FloatingPointError(f"non-finite hidden state at layer {layer_id}, branch={branch}")
                captured[int(layer_id)] = vec
            return None

        return hook_fn

    try:
        for layer in layers:
            hooks.append(modules[int(layer)].register_forward_hook(make_hook(int(layer))))
        outputs = model(
            inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            images=inputs["images"],
            use_cache=True,
            return_dict=True,
        )
        if not bool(torch.isfinite(outputs.logits[:, -1, :].detach().float()).all().item()):
            raise FloatingPointError(f"non-finite next-token logits, branch={branch}")
    finally:
        for hook in hooks:
            hook.remove()
    missing = [layer for layer in layers if int(layer) not in captured]
    if missing:
        raise RuntimeError(f"Missing hidden captures for layers={missing}, branch={branch}")
    return captured


def target_branch_for(row: Mapping[str, Any]) -> str:
    target = safe_text(row.get("target_modality")).lower()
    if target == "audio":
        return "audio"
    if target in {"visual", "video", "vision"}:
        return "visual"
    raise ValueError(f"Unsupported target_modality={target!r}")


def non_target_branch_for(row: Mapping[str, Any]) -> str:
    target = target_branch_for(row)
    return "visual" if target == "audio" else "audio"


def decide_runtime_gate(baseline: str | None, target: str | None, non_target: str | None, text: str | None) -> dict[str, Any]:
    supporting = []
    if baseline is not None and non_target == baseline:
        supporting.append("non_target")
    if baseline is not None and text == baseline:
        supporting.append("text")
    target_conflicts = baseline is not None and target is not None and target != baseline
    edit = bool(target_conflicts and supporting)
    if edit:
        reason = "target_conflicts_with_current_and_prior_supports_current"
    elif not target_conflicts:
        reason = "target_agrees_with_current_or_invalid"
    else:
        reason = "target_conflicts_but_no_prior_support"
    return {
        "edit_applied": edit,
        "gate_reason": reason,
        "gate_target_conflicts_current": bool(target_conflicts),
        "gate_prior_supporting_branches": supporting,
    }


def summarize(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    valid = [row for row in rows if not row.get("error")]

    def acc(prefix: str, group: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        total = len(group)
        correct = sum(1 for row in group if row.get(f"{prefix}_correct") is True)
        return {"correct": int(correct), "total": int(total), "accuracy": float(correct / total) if total else None}

    baseline = acc("baseline", valid)
    policy = acc("policy", valid)
    raw = acc("raw_edited", valid)
    by_task = {}
    by_target = {}
    for key, out in (("mad_protocol_task", by_task), ("target_modality", by_target)):
        groups: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
        for row in valid:
            groups[safe_text(row.get(key)) or "UNKNOWN"].append(row)
        for name, group in sorted(groups.items()):
            out[name] = {
                "baseline": acc("baseline", group),
                "policy": acc("policy", group),
                "raw_edited": acc("raw_edited", group),
            }
            if out[name]["baseline"]["accuracy"] is not None and out[name]["policy"]["accuracy"] is not None:
                out[name]["delta_pp"] = 100.0 * (
                    out[name]["policy"]["accuracy"] - out[name]["baseline"]["accuracy"]
                )
    delta = None
    if baseline["accuracy"] is not None and policy["accuracy"] is not None:
        delta = 100.0 * (policy["accuracy"] - baseline["accuracy"])
    return {
        "n_rows": len(rows),
        "n_valid": len(valid),
        "n_errors": len(rows) - len(valid),
        "baseline": baseline,
        "policy": policy,
        "raw_edited": raw,
        "delta_pp": delta,
        "edit_applied": sum(1 for row in valid if row.get("edit_applied") is True),
        "gate_reasons": dict(Counter(safe_text(row.get("gate_reason")) for row in valid)),
        "by_task": by_task,
        "by_target_modality": by_target,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Strict MAD-path VideoLLaMA2 hidden edit runner.")
    parser.add_argument("--benchmark", choices=["avh", "cmm"], default="avh")
    parser.add_argument("--avhbench-dir", type=Path, default=ROOT / "data" / "AVHBench")
    parser.add_argument("--cmm-data-path", type=Path, default=ROOT / "data" / "CMM" / "all_data_final_reorg.json")
    parser.add_argument("--cmm-media-base-dir", type=Path, default=ROOT / "data" / "CMM")
    parser.add_argument("--selected-rows-path", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--model-path", type=str, default="DAMO-NLP-SG/VideoLLaMA2.1-7B-AV"
    )
    parser.add_argument("--max-rows", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--layers", type=int, nargs="+", default=[26])
    parser.add_argument("--layer-weights", type=str, default="26:1.0")
    parser.add_argument("--beta", type=float, default=0.65)
    parser.add_argument("--strong-beta", type=float, default=0.95)
    parser.add_argument("--decision-mode", choices=["generate", "constrained_yesno"], default="generate")
    parser.add_argument("--dtype", choices=["float16", "bfloat16"], default="bfloat16")
    parser.add_argument("--max-new-tokens", type=int, default=1)
    parser.add_argument("--no-progress", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dtype = torch_dtype_from_name(args.dtype)
    layers = [int(layer) for layer in args.layers]
    layer_weights = parse_layer_weights(args.layer_weights)
    max_rows = int(args.max_rows) if int(args.max_rows) > 0 else None
    if args.selected_rows_path is not None:
        selected_all = [
            json.loads(line)
            for line in args.selected_rows_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    elif args.benchmark == "avh":
        selected_all = select_avh_unary_full_rows(avhbench_dir=args.avhbench_dir, max_rows=max_rows)
    else:
        selected_all = select_cmm_unary_full_rows(
            cmm_data_path=args.cmm_data_path,
            cmm_media_base_dir=args.cmm_media_base_dir,
            max_rows=max_rows,
        )
    selected = [row for idx, row in enumerate(selected_all) if idx % int(args.num_shards) == int(args.shard_index)]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    run_config = {
        "kind": "videollama2_lccs_strict_mad_path_v1",
        "benchmark": args.benchmark,
        "model_path": args.model_path,
        "python": sys.executable,
        "torch_version": torch.__version__,
        "transformers_version": transformers.__version__,
        "n_selected_all": len(selected_all),
        "n_selected_shard": len(selected),
        "num_shards": int(args.num_shards),
        "shard_index": int(args.shard_index),
        "layers": layers,
        "layer_weights": {str(layer): float(layer_weights[layer]) for layer in layers},
        "beta": float(args.beta),
        "strong_beta": float(args.strong_beta),
        "decision_mode": args.decision_mode,
        "dtype": str(args.dtype),
        "max_new_tokens": int(args.max_new_tokens),
        "official_path_alignment": "prompt/media/model-init/generate path matches MAD VideoLLaMA2 eval_batch.py/mm_infer; hidden edit only adds language-layer hook.",
        "official_score_outputs": [
            "official_baseline_results_av.json",
            "official_policy_results_av.json",
            "official_raw_edited_results_av.json",
            "official_baseline_results_cmm.jsonl",
            "official_policy_results_cmm.jsonl",
            "official_raw_edited_results_cmm.jsonl",
        ],
        "required_env_note": "For fair VideoLLaMA2 comparison, run with .venv_videollama2_mad (transformers==4.53.0), the same environment used for official MAD baseline/MAD rerun.",
        "runtime_uses_reference_answer": False,
        "runtime_uses_task_family": False,
    }
    write_json(args.output_dir / "run_config.json", run_config)
    with (args.output_dir / "selected_rows.jsonl").open("w", encoding="utf-8") as handle:
        for row in selected:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")

    model, processor, tokenizer = model_init(
        args.model_path,
        device_map=torch.device("cuda"),
        use_flash_attn=True,
        torch_dtype=dtype,
    )
    model.eval()
    if max(layers) >= len(language_layers(model)):
        raise ValueError(f"Layer index out of range: layers={layers}, model_layers={len(language_layers(model))}")

    rows_path = args.output_dir / "rows.jsonl"
    rows_path.write_text("", encoding="utf-8")
    outputs: list[dict[str, Any]] = []
    official_baseline_rows: list[dict[str, Any]] = []
    official_policy_rows: list[dict[str, Any]] = []
    official_raw_edited_rows: list[dict[str, Any]] = []
    iterator = tqdm(selected, desc=f"vl2_strict_lccs_s{args.shard_index}", disable=bool(args.no_progress), unit="sample")
    for row in iterator:
        start = time.time()
        try:
            device_text = str(torch.device("cuda", torch.cuda.current_device())) if torch.cuda.is_available() else "cpu"
            target_branch = target_branch_for(row)
            non_target_branch = non_target_branch_for(row)
            baseline_raw, baseline_debug = generate_answer(model=model, tokenizer=tokenizer, processor=processor, row=row, branch="full", max_new_tokens=int(args.max_new_tokens), decision_mode=args.decision_mode, dtype=dtype)
            target_raw, target_debug = generate_answer(model=model, tokenizer=tokenizer, processor=processor, row=row, branch=target_branch, max_new_tokens=int(args.max_new_tokens), decision_mode=args.decision_mode, dtype=dtype)
            non_target_raw, non_target_debug = generate_answer(model=model, tokenizer=tokenizer, processor=processor, row=row, branch=non_target_branch, max_new_tokens=int(args.max_new_tokens), decision_mode=args.decision_mode, dtype=dtype)
            text_raw, text_debug = generate_answer(model=model, tokenizer=tokenizer, processor=processor, row=row, branch="text", max_new_tokens=int(args.max_new_tokens), decision_mode=args.decision_mode, dtype=dtype)
            baseline_pred = normalize_yes_no(baseline_raw)
            target_pred = normalize_yes_no(target_raw)
            non_target_pred = normalize_yes_no(non_target_raw)
            text_pred = normalize_yes_no(text_raw)
            baseline_margin = margin_from_debug(baseline_debug)
            target_margin = margin_from_debug(target_debug)
            non_target_margin = margin_from_debug(non_target_debug)
            text_margin = margin_from_debug(text_debug)
            edit_available = True
            edit_fallback_reason = ""
            try:
                captures = {
                    "full": capture_hidden(model=model, tokenizer=tokenizer, processor=processor, row=row, branch="full", layers=layers, dtype=dtype),
                    "target": capture_hidden(model=model, tokenizer=tokenizer, processor=processor, row=row, branch=target_branch, layers=layers, dtype=dtype),
                }
                prior_directions = {layer: captures["full"][layer] - captures["target"][layer] for layer in layers}
                for layer, direction in prior_directions.items():
                    if not bool(torch.isfinite(direction).all().item()):
                        raise FloatingPointError(f"non-finite prior direction at layer {layer}")
                raw_edited_raw, patch_debug = generate_answer(
                    model=model,
                    tokenizer=tokenizer,
                    processor=processor,
                    row=row,
                    branch="full",
                    patch_layers=layers,
                    layer_weights=layer_weights,
                    prior_directions=prior_directions,
                    beta=float(args.beta),
                    max_new_tokens=int(args.max_new_tokens),
                    decision_mode=args.decision_mode,
                    dtype=dtype,
                )
                raw_edited_pred = normalize_yes_no(raw_edited_raw)
                raw_edited_margin = margin_from_debug(patch_debug)
                strong_edited_raw, strong_patch_debug = generate_answer(
                    model=model,
                    tokenizer=tokenizer,
                    processor=processor,
                    row=row,
                    branch="full",
                    patch_layers=layers,
                    layer_weights=layer_weights,
                    prior_directions=prior_directions,
                    beta=float(args.strong_beta),
                    max_new_tokens=int(args.max_new_tokens),
                    decision_mode=args.decision_mode,
                    dtype=dtype,
                )
                strong_edited_pred = normalize_yes_no(strong_edited_raw)
                strong_edited_margin = margin_from_debug(strong_patch_debug)
            except FloatingPointError as exc:
                edit_available = False
                edit_fallback_reason = repr(exc)
                raw_edited_raw = baseline_raw
                raw_edited_pred = baseline_pred
                raw_edited_margin = baseline_margin
                strong_edited_raw = baseline_raw
                strong_edited_pred = baseline_pred
                strong_edited_margin = baseline_margin
                patch_debug = {"debug_by_layer": {}, "fallback_reason": edit_fallback_reason}
                strong_patch_debug = {"debug_by_layer": {}, "fallback_reason": edit_fallback_reason}
            conflict = bool(baseline_pred is not None and target_pred is not None and baseline_pred != target_pred)
            conflict_state = "target_vs_baseline_conflict" if conflict else "no_target_baseline_conflict"
            policy_raw = raw_edited_raw if conflict and edit_available else baseline_raw
            policy_pred = raw_edited_pred if conflict and edit_available else baseline_pred
            gate_reason = (
                "target_vs_baseline_conflict"
                if conflict and edit_available
                else "hidden_edit_non_finite_fallback"
                if conflict and not edit_available
                else "no_target_baseline_conflict"
            )
            reference = row.get("reference_answer")
            runtime_seconds = time.time() - start
            output = {
                "sample_id": row.get("sample_id"),
                "video_id": official_video_id(row),
                "benchmark": row.get("benchmark"),
                "mad_protocol_task": row.get("mad_protocol_task"),
                "target_modality": row.get("target_modality"),
                "target_branch": target_branch,
                "non_target_branch": non_target_branch,
                "question": row.get("question"),
                "video_path": row.get("video_path"),
                "audio_path": row.get("audio_path"),
                "reference_answer": reference,
                "baseline_raw_output": baseline_raw,
                "baseline_prediction": baseline_pred,
                "baseline_margin": baseline_margin,
                "baseline_correct": answer_correct_strict(baseline_pred, reference),
                "target_branch_raw_output": target_raw,
                "target_branch_prediction": target_pred,
                "target_branch_margin": target_margin,
                "non_target_branch_raw_output": non_target_raw,
                "non_target_branch_prediction": non_target_pred,
                "non_target_branch_margin": non_target_margin,
                "text_branch_raw_output": text_raw,
                "text_branch_prediction": text_pred,
                "text_branch_margin": text_margin,
                "candidate_answer_y": target_pred,
                "conflict_state": conflict_state,
                "raw_edited_raw_output": raw_edited_raw,
                "raw_edited_prediction": raw_edited_pred,
                "raw_edited_margin": raw_edited_margin,
                "edited_candidate_aligned_margin": aligned_margin(target_pred, raw_edited_margin),
                "raw_edited_correct": answer_correct_strict(raw_edited_pred, reference),
                "strong_edited_raw_output": strong_edited_raw,
                "strong_edited_prediction": strong_edited_pred,
                "strong_edited_margin": strong_edited_margin,
                "strong_edited_correct": answer_correct_strict(strong_edited_pred, reference),
                "policy_raw_output": policy_raw,
                "policy_prediction": policy_pred,
                "policy_correct": answer_correct_strict(policy_pred, reference),
                "edit_applied": conflict and edit_available,
                "gate_reason": gate_reason,
                "edit_available": edit_available,
                "edit_fallback_reason": edit_fallback_reason,
                "gate_target_conflicts_current": conflict,
                "gate_prior_supporting_branches": [],
                "patch_debug_by_layer": patch_debug.get("debug_by_layer"),
                "strong_patch_debug_by_layer": strong_patch_debug.get("debug_by_layer"),
                "layers": layers,
                "layer_weights": {str(layer): float(layer_weights[layer]) for layer in layers},
                "beta": float(args.beta),
                "runtime_seconds": runtime_seconds,
                "error": "",
            }
            official_baseline_rows.append(
                official_cmm_score_row(row=row, prediction=baseline_raw)
                if args.benchmark == "cmm"
                else official_score_row(row=row, prediction=baseline_raw, runtime_seconds=runtime_seconds, device=device_text)
            )
            official_policy_rows.append(
                official_cmm_score_row(row=row, prediction=policy_raw)
                if args.benchmark == "cmm"
                else official_score_row(row=row, prediction=policy_raw, runtime_seconds=runtime_seconds, device=device_text)
            )
            official_raw_edited_rows.append(
                official_cmm_score_row(row=row, prediction=raw_edited_raw)
                if args.benchmark == "cmm"
                else official_score_row(row=row, prediction=raw_edited_raw, runtime_seconds=runtime_seconds, device=device_text)
            )
        except Exception as exc:
            runtime_seconds = time.time() - start
            error_prediction = f"ERROR: {repr(exc)}"
            output = {
                "sample_id": row.get("sample_id"),
                "video_id": official_video_id(row),
                "benchmark": row.get("benchmark"),
                "mad_protocol_task": row.get("mad_protocol_task"),
                "target_modality": row.get("target_modality"),
                "question": row.get("question"),
                "reference_answer": row.get("reference_answer"),
                "runtime_seconds": runtime_seconds,
                "error": repr(exc),
            }
            device_text = str(torch.device("cuda", torch.cuda.current_device())) if torch.cuda.is_available() else "cpu"
            official_baseline_rows.append(
                official_cmm_score_row(row=row, prediction=error_prediction)
                if args.benchmark == "cmm"
                else official_score_row(row=row, prediction=error_prediction, runtime_seconds=runtime_seconds, device=device_text)
            )
            official_policy_rows.append(
                official_cmm_score_row(row=row, prediction=error_prediction)
                if args.benchmark == "cmm"
                else official_score_row(row=row, prediction=error_prediction, runtime_seconds=runtime_seconds, device=device_text)
            )
            official_raw_edited_rows.append(
                official_cmm_score_row(row=row, prediction=error_prediction)
                if args.benchmark == "cmm"
                else official_score_row(row=row, prediction=error_prediction, runtime_seconds=runtime_seconds, device=device_text)
            )
        append_jsonl(rows_path, output)
        outputs.append(output)
    summary = summarize(outputs)
    summary["run_config"] = run_config
    write_json(args.output_dir / "summary.json", summary)
    if args.benchmark == "cmm":
        for name, rows in [
            ("official_baseline_results_cmm.jsonl", official_baseline_rows),
            ("official_policy_results_cmm.jsonl", official_policy_rows),
            ("official_raw_edited_results_cmm.jsonl", official_raw_edited_rows),
        ]:
            with (args.output_dir / name).open("w", encoding="utf-8") as handle:
                for item in rows:
                    handle.write(json.dumps(item, ensure_ascii=False) + "\n")
    else:
        write_json(args.output_dir / "official_baseline_results_av.json", official_baseline_rows)
        write_json(args.output_dir / "official_policy_results_av.json", official_policy_rows)
        write_json(args.output_dir / "official_raw_edited_results_av.json", official_raw_edited_rows)
    print(json.dumps(summary, ensure_ascii=False, indent=2)[:4000])


if __name__ == "__main__":
    main()
