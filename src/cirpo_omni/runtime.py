from __future__ import annotations

import importlib.metadata
import math
import os
import random
import re
import types
from typing import Any, Dict, List, Sequence

import numpy as np
import torch
import torch.nn.functional as F

from omni_dpo.media import MediaResolver
from omni_dpo.modeling import (
    build_messages,
    compose_question_text,
    messages_to_inputs,
    prepare_supervised_inputs,
    resolve_model_input_device,
)

DEFAULT_CIRPO_VIDEO_BUDGET = {
    "fps": 2.0,
    "max_pixels": 150528,
    "min_pixels": 50176,
    "max_frames": 8,
}

DEFAULT_CIRPO_LOSS_VIDEO_BUDGET = {
    "fps": 2.0,
    "max_pixels": 100352,
    "min_pixels": 31360,
    "max_frames": 4,
}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def weighted_subset(rows: Sequence[Dict[str, Any]], size: int, seed: int) -> List[Dict[str, Any]]:
    rows = list(rows)
    if size <= 0 or size >= len(rows):
        return list(rows)
    weights = np.array([max(1e-8, float(row.get("sample_weight", 1.0) or 1.0)) for row in rows], dtype=np.float64)
    probs = weights / weights.sum()
    rng = np.random.default_rng(seed)
    indices = rng.choice(len(rows), size=size, replace=False, p=probs)
    return [rows[idx] for idx in indices.tolist()]


def materialize_train_sequence(rows: Sequence[Dict[str, Any]], total_micro_steps: int, seed: int) -> List[Dict[str, Any]]:
    rows = list(rows)
    if not rows:
        return []
    rng = random.Random(seed)
    out: List[Dict[str, Any]] = []
    while len(out) < total_micro_steps:
        chunk = list(rows)
        rng.shuffle(chunk)
        out.extend(chunk)
    return out[:total_micro_steps]


def ensure_qwen2_omni_kbit_compat(model) -> None:
    """
    Qwen2_5OmniForConditionalGeneration in the current transformers build does not
    expose top-level input-embedding accessors, while PEFT's k-bit preparation
    assumes they exist. CIRPO only trains thinker-side LoRA modules, so forwarding
    these accessors to `model.thinker` is the correct compatibility shim.
    """
    if hasattr(model, "get_input_embeddings") and hasattr(model, "set_input_embeddings"):
        try:
            model.get_input_embeddings()
            return
        except NotImplementedError:
            pass

    thinker = getattr(model, "thinker", None)
    if thinker is None or not hasattr(thinker, "get_input_embeddings") or not hasattr(thinker, "set_input_embeddings"):
        raise RuntimeError(
            "Qwen2.5-Omni k-bit compatibility patch failed: thinker input embeddings are unavailable."
        )

    def _get_input_embeddings(self):
        return self.thinker.get_input_embeddings()

    def _set_input_embeddings(self, value):
        return self.thinker.set_input_embeddings(value)

    model.get_input_embeddings = types.MethodType(_get_input_embeddings, model)
    model.set_input_embeddings = types.MethodType(_set_input_embeddings, model)


def get_cirpo_train_model(model):
    thinker = getattr(model, "thinker", None)
    if thinker is not None:
        return thinker
    return model


def resolve_auto_max_memory() -> Dict[int, str] | None:
    if not torch.cuda.is_available():
        return None
    visible_gpu_count = torch.cuda.device_count()
    if visible_gpu_count <= 0:
        return None
    per_gpu_limit = os.environ.get("OMNI_MAX_MEMORY_PER_GPU", "40GB")
    return {idx: per_gpu_limit for idx in range(visible_gpu_count)}


def build_balanced_qwen_omni_device_map(config, num_devices: int) -> Dict[str, int]:
    if num_devices <= 1:
        return {"": 0}

    num_layers = int(config.thinker_config.text_config.num_hidden_layers)
    audio_gpu = 0
    visual_gpu = 1 if num_devices > 1 else 0
    last_gpu = num_devices - 1
    reserve_output_gpu = num_devices >= 4
    layer_devices = list(range(num_devices - 1)) if reserve_output_gpu else list(range(num_devices))

    weights: List[float] = []
    for device_id in layer_devices:
        if device_id == audio_gpu:
            weights.append(0.40)
        elif device_id == visual_gpu:
            weights.append(0.40)
        elif device_id == layer_devices[-1]:
            weights.append(0.70)
        else:
            weights.append(1.00)

    total_weight = sum(weights)
    raw_counts = [(num_layers * weight) / total_weight for weight in weights]
    layer_counts = [int(math.floor(value)) for value in raw_counts]
    remainder = num_layers - sum(layer_counts)
    priority = sorted(
        range(len(layer_devices)),
        key=lambda idx: (raw_counts[idx] - layer_counts[idx], idx),
        reverse=True,
    )
    for idx in priority[:remainder]:
        layer_counts[idx] += 1

    device_map: Dict[str, int] = {
        "thinker.audio_tower": audio_gpu,
        "thinker.visual": visual_gpu,
        "thinker.model.embed_tokens": audio_gpu,
        "thinker.model.rotary_emb": audio_gpu,
        "thinker.model.norm": last_gpu,
        "thinker.lm_head": last_gpu,
    }

    layer_idx = 0
    for device_id, count in zip(layer_devices, layer_counts):
        for _ in range(count):
            device_map[f"thinker.model.layers.{layer_idx}"] = device_id
            layer_idx += 1

    if layer_idx != num_layers:
        raise RuntimeError(f"Balanced device-map builder assigned {layer_idx} layers, expected {num_layers}.")
    return device_map


def load_quantized_model(model_path: str, device: str):
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import BitsAndBytesConfig, Qwen2_5OmniConfig, Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor

    try:
        importlib.metadata.version("bitsandbytes")
    except importlib.metadata.PackageNotFoundError as exc:
        raise RuntimeError(
            "bitsandbytes is required for the fixed 4bit QLoRA CIRPO trainer. "
            "Install bitsandbytes in the active environment before running this script."
        ) from exc

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    attn_implementation = os.environ.get("OMNI_ATTN_IMPL", "flash_attention_2")
    processor = Qwen2_5OmniProcessor.from_pretrained(model_path)
    config = Qwen2_5OmniConfig.from_pretrained(model_path)
    config.enable_audio_output = False
    if device == "auto":
        max_memory = resolve_auto_max_memory()
        strategy = os.environ.get("OMNI_DEVICE_MAP_STRATEGY", "balanced_thinker")
        device_map = (
            build_balanced_qwen_omni_device_map(config, torch.cuda.device_count())
            if strategy == "balanced_thinker"
            else "auto"
        )
        model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
            model_path,
            config=config,
            quantization_config=bnb_config,
            device_map=device_map,
            torch_dtype=torch.bfloat16,
            max_memory=max_memory,
            attn_implementation=attn_implementation,
        )
    else:
        model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
            model_path,
            config=config,
            quantization_config=bnb_config,
            torch_dtype=torch.bfloat16,
            attn_implementation=attn_implementation,
        )
        model = model.to(device)
    # CIRPO v1 is text-only at generation time. The talker/token2wav stack is unused
    # and occupies substantial memory, so drop it before training.
    if getattr(model, "has_talker", False) and hasattr(model, "disable_talker"):
        model.disable_talker()
    model.config.use_cache = False
    for param in model.parameters():
        param.requires_grad = False
    thinker = get_cirpo_train_model(model)
    thinker.config.use_cache = False
    model.thinker = prepare_model_for_kbit_training(thinker)
    return model, processor, LoraConfig, get_peft_model


def collect_lora_targets_last12(model) -> List[str]:
    target_names: List[str] = []
    pattern = re.compile(
        r"^model\.layers\.(\d+)\.(?:self_attn\.(q_proj|k_proj|v_proj|o_proj)|mlp\.(gate_proj|up_proj|down_proj))$"
    )
    for name, _module in model.named_modules():
        match = pattern.match(name)
        if not match:
            continue
        layer_idx = int(match.group(1))
        if 16 <= layer_idx <= 27:
            target_names.append(name)
    if not target_names:
        raise RuntimeError("No LoRA target modules matched thinker.model last 12 layers.")
    return sorted(target_names)


def attach_lora(model, lora_config_cls, get_peft_model_fn, *, r: int, alpha: int, dropout: float):
    thinker = get_cirpo_train_model(model)
    target_modules = collect_lora_targets_last12(thinker)
    lora_config = lora_config_cls(
        r=r,
        lora_alpha=alpha,
        lora_dropout=dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=target_modules,
    )
    model.thinker = get_peft_model_fn(thinker, lora_config)
    return model, [f"thinker.{name}" for name in target_modules]


def resolve_episode_contexts(
    *,
    episode: Dict[str, Any],
    resolver: MediaResolver,
) -> Dict[str, Dict[str, Any]]:
    contexts: Dict[str, Dict[str, Any]] = {}
    for branch in [episode["full_branch"], episode["cf_branch_a"], episode["cf_branch_b"]]:
        branch_id = str(branch.get("branch_id") or "")
        pair_like = {
            "dataset": episode.get("dataset"),
            "sample_id": episode.get("sample_id"),
            "input": branch.get("input") or {},
        }
        prepared = resolver.prepare_pair_inputs(pair_like)
        contexts[branch_id] = {
            "question_text": compose_question_text(((branch.get("input") or {}).get("prompt") or {})),
            "video_path": prepared.get("video_path"),
            "audio_array": prepared.get("audio_array"),
        }
    return contexts


@torch.no_grad()
def generate_branch_response(
    *,
    model,
    processor,
    question_text: str,
    video_path: str | None,
    audio_array,
    max_new_tokens: int,
    do_sample: bool,
    temperature: float,
    top_p: float,
    device: torch.device | None = None,
    video_budget: Dict[str, float] | None = None,
    system_prompt: str | None = None,
) -> str:
    if device is None:
        device = resolve_model_input_device(model)
    messages = build_messages(
        question_text=question_text,
        video_path=video_path,
        audio_array=audio_array,
        answer_text=None,
        system_prompt=system_prompt,
    )
    inputs = messages_to_inputs(
        processor=processor,
        messages=messages,
        audio_array=audio_array,
        device=device,
        video_budget=video_budget or DEFAULT_CIRPO_VIDEO_BUDGET,
    )
    input_len = inputs["input_ids"].shape[-1]
    gen_ids = model.generate(
        **inputs,
        do_sample=do_sample,
        top_p=top_p,
        temperature=temperature,
        return_audio=False,
        thinker_max_new_tokens=max_new_tokens,
    )
    new_ids = gen_ids[:, input_len:]
    return processor.batch_decode(
        new_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0].strip()


def compute_sequence_logp_stats(
    *,
    model,
    model_inputs: Dict[str, torch.Tensor],
) -> Dict[str, torch.Tensor]:
    labels = model_inputs["labels"]
    forward_inputs = {k: v for k, v in model_inputs.items() if k not in {"labels", "prompt_length"}}
    forward_model = get_cirpo_train_model(model)
    outputs = forward_model(**forward_inputs, use_cache=False)
    logits = outputs.logits[:, :-1, :]
    shifted_labels = labels[:, 1:].to(logits.device)
    valid_mask = shifted_labels != -100
    safe_labels = shifted_labels.masked_fill(~valid_mask, 0)
    token_logps = F.log_softmax(logits, dim=-1).gather(-1, safe_labels.unsqueeze(-1)).squeeze(-1)
    seq_logp = (token_logps * valid_mask).sum(dim=-1)
    token_count = valid_mask.sum(dim=-1).clamp_min(1)
    mean_logp = seq_logp / token_count
    return {
        "seq_logp": seq_logp,
        "mean_logp": mean_logp,
        "token_count": token_count,
    }


def prepare_generated_branch_inputs(
    *,
    processor,
    question_text: str,
    answer_text: str,
    video_path: str | None,
    audio_array,
    device: torch.device | None,
    video_budget: Dict[str, float] | None = None,
    system_prompt: str | None = None,
) -> Dict[str, torch.Tensor]:
    return prepare_supervised_inputs(
        processor=processor,
        question_text=question_text,
        answer_text=answer_text,
        video_path=video_path,
        audio_array=audio_array,
        device=device,
        video_budget=video_budget or DEFAULT_CIRPO_VIDEO_BUDGET,
        system_prompt=system_prompt,
    )
