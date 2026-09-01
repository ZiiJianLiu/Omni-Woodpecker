from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F
from transformers.video_utils import load_video


DEFAULT_SYSTEM_PROMPT = (
    "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, "
    "capable of perceiving auditory and visual inputs, as well as generating text and speech."
)


def compose_question_text(prompt: Dict[str, Any]) -> str:
    question = str(prompt.get("question") or "").strip()
    lines = [question]
    choices = prompt.get("choices") or []
    answer_format = str(prompt.get("answer_format") or "")
    if choices:
        lines.append("")
        lines.append("Options:")
        for choice in choices:
            label = str(choice.get("label") or "").strip()
            text = str(choice.get("text") or "").strip()
            if label and text:
                lines.append(f"{label}. {text}")
            elif text:
                lines.append(text)
        lines.append("")
    if answer_format == "choice_label":
        lines.append("Answer with the single best option in the format: Final answer: <LABEL>. <TEXT>")
    else:
        lines.append("Answer briefly in the format: Final answer: <answer>")
    return "\n".join(lines).strip()


def build_messages(
    *,
    question_text: str,
    video_path: Optional[str],
    audio_array: Optional[np.ndarray],
    answer_text: Optional[str] = None,
    system_prompt: Optional[str] = None,
) -> List[Dict[str, Any]]:
    user_content: List[Dict[str, Any]] = []
    if video_path and Path(video_path).exists():
        user_content.append({"type": "video", "video": video_path})
    if audio_array is not None:
        user_content.append({"type": "audio", "audio": "placeholder"})
    user_content.append({"type": "text", "text": question_text})
    resolved_system_prompt = str(system_prompt or DEFAULT_SYSTEM_PROMPT).strip() or DEFAULT_SYSTEM_PROMPT
    messages: List[Dict[str, Any]] = [
        {"role": "system", "content": [{"type": "text", "text": resolved_system_prompt}]},
        {"role": "user", "content": user_content},
    ]
    if answer_text is not None:
        messages.append({"role": "assistant", "content": [{"type": "text", "text": answer_text}]})
    return messages


def _collect_videos(messages: List[Dict[str, Any]]) -> List[str]:
    videos: List[str] = []
    for msg in messages:
        content = msg.get("content", [])
        if isinstance(content, str):
            continue
        for item in content:
            if item.get("type") == "video":
                videos.append(item["video"])
    return videos


def _load_video_inputs(video_paths: List[str], video_budget: Dict[str, float]) -> tuple[List[np.ndarray], List[Any]]:
    loaded: List[np.ndarray] = []
    metadata: List[Any] = []
    max_frames = int(video_budget.get("max_frames", 32))
    for video_path in video_paths:
        try:
            video = load_video(video_path, num_frames=max_frames, backend="pyav")
        except Exception:
            video = load_video(video_path, num_frames=max_frames, backend="opencv")
        if isinstance(video, tuple):
            frames, video_meta = video
        else:
            frames, video_meta = video, None
        loaded.append(frames)
        metadata.append(video_meta)
    return loaded, metadata


def messages_to_inputs(
    *,
    processor,
    messages: List[Dict[str, Any]],
    audio_array: Optional[np.ndarray],
    device: Optional[torch.device] = None,
    video_budget: Optional[Dict[str, float]] = None,
) -> Dict[str, torch.Tensor]:
    text_prompt = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=messages[-1]["role"] != "assistant",
    )
    video_paths = _collect_videos(messages)
    video_budget = video_budget or {
        "fps": 4.0,
        "max_pixels": 602112,
        "min_pixels": 100352,
        "max_frames": 32,
    }
    if video_paths and hasattr(processor, "video_processor"):
        processor.video_processor.do_sample_frames = False
        processor.video_processor.max_pixels = int(video_budget.get("max_pixels", 602112))
        processor.video_processor.min_pixels = int(video_budget.get("min_pixels", 100352))
    videos, video_metadata = _load_video_inputs(video_paths, video_budget) if video_paths else ([], [])

    proc_kwargs: Dict[str, Any] = {
        "text": text_prompt,
        "return_tensors": "pt",
        "padding": True,
    }
    if videos:
        proc_kwargs["videos"] = videos
    if video_metadata and any(meta is not None for meta in video_metadata):
        proc_kwargs["video_metadata"] = [[meta] for meta in video_metadata]
    if audio_array is not None:
        proc_kwargs["audio"] = [audio_array]
    inputs = processor(**proc_kwargs)
    if device is not None:
        inputs = inputs.to(device)
    return dict(inputs)


def prepare_supervised_inputs(
    *,
    processor,
    question_text: str,
    answer_text: str,
    video_path: Optional[str],
    audio_array: Optional[np.ndarray],
    device: Optional[torch.device] = None,
    video_budget: Optional[Dict[str, float]] = None,
    system_prompt: Optional[str] = None,
) -> Dict[str, torch.Tensor]:
    prompt_messages = build_messages(
        question_text=question_text,
        video_path=video_path,
        audio_array=audio_array,
        answer_text=None,
        system_prompt=system_prompt,
    )
    full_messages = build_messages(
        question_text=question_text,
        video_path=video_path,
        audio_array=audio_array,
        answer_text=answer_text,
        system_prompt=system_prompt,
    )
    prompt_inputs = messages_to_inputs(
        processor=processor,
        messages=prompt_messages,
        audio_array=audio_array,
        device=device,
        video_budget=video_budget,
    )
    full_inputs = messages_to_inputs(
        processor=processor,
        messages=full_messages,
        audio_array=audio_array,
        device=device,
        video_budget=video_budget,
    )
    prompt_len = prompt_inputs["input_ids"].shape[-1]
    labels = full_inputs["input_ids"].clone()
    labels[:, :prompt_len] = -100
    if "attention_mask" in full_inputs:
        labels[full_inputs["attention_mask"] == 0] = -100
    full_inputs["labels"] = labels
    full_inputs["prompt_length"] = torch.tensor([prompt_len], device=labels.device)
    return full_inputs


def compute_sequence_logp(
    *,
    model,
    model_inputs: Dict[str, torch.Tensor],
) -> torch.Tensor:
    labels = model_inputs["labels"]
    forward_inputs = {k: v for k, v in model_inputs.items() if k not in {"labels", "prompt_length"}}
    outputs = model(**forward_inputs, use_cache=False)
    logits = outputs.logits[:, :-1, :]
    shifted_labels = labels[:, 1:]
    valid_mask = shifted_labels != -100
    safe_labels = shifted_labels.masked_fill(~valid_mask, 0)
    token_logps = F.log_softmax(logits, dim=-1).gather(-1, safe_labels.unsqueeze(-1)).squeeze(-1)
    seq_logp = (token_logps * valid_mask).sum(dim=-1)
    return seq_logp


def dpo_pair_loss(
    *,
    beta: float,
    policy_chosen_logp: torch.Tensor,
    policy_rejected_logp: torch.Tensor,
    ref_chosen_logp: torch.Tensor,
    ref_rejected_logp: torch.Tensor,
) -> torch.Tensor:
    logits = beta * ((policy_chosen_logp - policy_rejected_logp) - (ref_chosen_logp - ref_rejected_logp))
    return -F.logsigmoid(logits)


def resolve_model_input_device(model) -> torch.device:
    for param in model.parameters():
        if param.device.type != "meta":
            return param.device
    return torch.device("cpu")


@torch.no_grad()
def generate_response(
    *,
    model,
    processor,
    question_text: str,
    video_path: Optional[str],
    audio_array: Optional[np.ndarray],
    max_new_tokens: int = 64,
    device: Optional[torch.device] = None,
    system_prompt: Optional[str] = None,
) -> str:
    messages = build_messages(
        question_text=question_text,
        video_path=video_path,
        audio_array=audio_array,
        answer_text=None,
        system_prompt=system_prompt,
    )
    if device is None:
        device = resolve_model_input_device(model)
    inputs = messages_to_inputs(
        processor=processor,
        messages=messages,
        audio_array=audio_array,
        device=device,
    )
    input_len = inputs["input_ids"].shape[-1]
    gen_ids = model.generate(
        **inputs,
        do_sample=False,
        return_audio=False,
        thinker_max_new_tokens=max_new_tokens,
    )
    new_ids = gen_ids[:, input_len:]
    text = processor.batch_decode(
        new_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0]
    return text.strip()
