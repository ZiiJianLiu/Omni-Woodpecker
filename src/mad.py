#!/usr/bin/env python3
from __future__ import annotations

import gc
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
for path in (SCRIPT_DIR, ROOT, ROOT.parent):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from owp.models.qwen_omni import QwenOmniAdapter, _load_video_inputs  # noqa: E402


SYSTEM_MESSAGE = (
    "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, "
    "capable of perceiving auditory and visual inputs, as well as generating text and speech."
)
MODALITY_QUERY_PROMPT = "To answer this question, which modality is needed (audio, video, or both): "
ANSWER_QUERY_PROMPT = " Answer only 'Yes' or 'No'. Do not include any explanation."

_BRANCH_KEYS = ("av", "v", "a", "t")


def safe_text(value: Any) -> str:
    return "" if value is None else str(value).strip()


def normalize_yes_no(value: Any) -> Optional[str]:
    text = safe_text(value).lower()
    if text.startswith("yes"):
        return "Yes"
    if text.startswith("no"):
        return "No"
    return None


def question_with_answer_prompt(question: str) -> str:
    return f"{safe_text(question)}{ANSWER_QUERY_PROMPT}"


def _path_exists(path: Optional[str]) -> bool:
    return QwenOmniAdapter._path_exists(path)


def _resolve_audio_source_path(*, video_path: Optional[str], audio_path: Optional[str]) -> Optional[str]:
    if QwenOmniAdapter._path_exists(audio_path):
        return audio_path
    if QwenOmniAdapter._path_exists(video_path):
        return video_path
    return None


def _same_media_path(a: Optional[str], b: Optional[str]) -> bool:
    if not a or not b:
        return False
    try:
        return Path(a).resolve() == Path(b).resolve()
    except Exception:
        return str(a) == str(b)


def _build_multimodal_user_content(
    *,
    question: str,
    video_path: Optional[str],
    audio_path: Optional[str],
    add_prompt: Optional[str] = None,
) -> Tuple[List[Dict[str, Any]], bool]:
    has_video = _path_exists(video_path)
    has_audio = _path_exists(audio_path)
    user_content: List[Dict[str, Any]] = []
    use_audio_in_video = False

    if has_video:
        user_content.append({"type": "video", "video": video_path})
        if has_audio and not _same_media_path(video_path, audio_path):
            user_content.append({"type": "audio", "audio": audio_path})
        else:
            use_audio_in_video = True
    elif has_audio:
        user_content.append({"type": "audio", "audio": audio_path})
    else:
        raise ValueError("neither video_path nor audio_path is available for multimodal branch")

    suffix = f"\n{add_prompt}" if safe_text(add_prompt) else ""
    user_content.append({"type": "text", "text": "Question: " + question + suffix})
    return user_content, use_audio_in_video


def _build_head_conversation(
    question: str,
    video_path: Optional[str],
    audio_path: Optional[str],
    add_prompt: str,
) -> Tuple[List[Dict[str, Any]], bool]:
    user_content, use_audio_in_video = _build_multimodal_user_content(
        question=question,
        video_path=video_path,
        audio_path=audio_path,
        add_prompt=add_prompt,
    )
    return [
        {"role": "system", "content": [{"type": "text", "text": SYSTEM_MESSAGE}]},
        {"role": "user", "content": user_content},
    ], use_audio_in_video


def _build_branch_conversation(
    *,
    branch_key: str,
    question: str,
    video_path: Optional[str],
    audio_path: Optional[str],
) -> Tuple[List[Dict[str, Any]], bool]:
    if branch_key == "av":
        user_content, use_audio_in_video = _build_multimodal_user_content(
            question=question,
            video_path=video_path,
            audio_path=audio_path,
        )
    elif branch_key == "v":
        if not _path_exists(video_path):
            raise ValueError("visual branch requested but video_path is unavailable")
        user_content = [
            {"type": "video", "video": video_path},
            {"type": "text", "text": question},
        ]
        use_audio_in_video = False
    elif branch_key == "a":
        audio_source_path = _resolve_audio_source_path(video_path=video_path, audio_path=audio_path)
        if audio_source_path is None:
            raise ValueError("audio branch requested but neither audio_path nor video_path is available")
        user_content = [
            {"type": "audio", "audio": audio_source_path},
            {"type": "text", "text": question},
        ]
        use_audio_in_video = False
    elif branch_key == "t":
        user_content = [{"type": "text", "text": question}]
        use_audio_in_video = False
    else:
        raise ValueError(f"unsupported branch_key: {branch_key}")
    return [
        {"role": "system", "content": [{"type": "text", "text": SYSTEM_MESSAGE}]},
        {"role": "user", "content": user_content},
    ], use_audio_in_video


def _build_media_cache(
    adapter: QwenOmniAdapter,
    *,
    video_path: Optional[str],
    audio_path: Optional[str],
    budget: Dict[str, float],
) -> Dict[str, Any]:
    videos: List[np.ndarray] = []
    video_metadata: List[Any] = []
    if _path_exists(video_path):
        videos, video_metadata = _load_video_inputs([str(video_path)], budget)
    audio_source_path = _resolve_audio_source_path(video_path=video_path, audio_path=audio_path)
    audio_from_video = (
        QwenOmniAdapter._load_audio(str(video_path)) if _path_exists(video_path) else None
    )
    audio_from_audio_path = (
        QwenOmniAdapter._load_audio(audio_source_path) if audio_source_path is not None else None
    )
    return {
        "videos": videos,
        "video_metadata": video_metadata,
        "audio_from_video": audio_from_video,
        "audio_from_audio_path": audio_from_audio_path,
        "audio_source_path": audio_source_path,
    }


def _process_mm_info_equivalent(
    adapter: QwenOmniAdapter,
    *,
    conversation: Sequence[Dict[str, Any]],
    use_audio_in_video: bool,
    media_cache: Dict[str, Any],
) -> Tuple[List[np.ndarray], List[Any], List[np.ndarray]]:
    del adapter
    audios: List[np.ndarray] = []
    images: List[Any] = []
    videos: List[np.ndarray] = []
    for message in conversation:
        content = message.get("content")
        if isinstance(content, str):
            continue
        for item in list(content or []):
            item_type = safe_text(item.get("type"))
            if item_type == "video":
                videos.extend(list(media_cache.get("videos") or []))
                if use_audio_in_video:
                    audio = media_cache.get("audio_from_video")
                    if audio is not None:
                        audios.append(audio)
            elif item_type == "audio":
                audio = media_cache.get("audio_from_audio_path")
                if audio is not None:
                    audios.append(audio)
    return audios, images, videos


def _prepare_processor_inputs(
    adapter: QwenOmniAdapter,
    *,
    conversation: Sequence[Dict[str, Any]],
    use_audio_in_video: bool,
    media_cache: Dict[str, Any],
) -> Dict[str, torch.Tensor]:
    processor = adapter._processor
    text = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
    audios, images, videos = _process_mm_info_equivalent(
        adapter,
        conversation=conversation,
        use_audio_in_video=use_audio_in_video,
        media_cache=media_cache,
    )
    # Some CMM video-only assets have no usable audio stream. In that case we must
    # downgrade to pure-video processing instead of forcing the processor down the
    # audio-in-video path, which can raise StopIteration inside the media stack.
    effective_use_audio_in_video = bool(use_audio_in_video and audios)
    proc_kwargs: Dict[str, Any] = {
        "text": text,
        "return_tensors": "pt",
        "padding": True,
        "use_audio_in_video": effective_use_audio_in_video,
    }
    video_processor = getattr(processor, "video_processor", None)
    restore_do_sample_frames: Optional[bool] = None
    predecoded_videos = bool(videos) and all(
        isinstance(video, np.ndarray) or torch.is_tensor(video) for video in videos
    )
    if audios:
        proc_kwargs["audio"] = audios
    if images:
        proc_kwargs["images"] = images
    if videos:
        proc_kwargs["videos"] = videos
        # `_load_video_inputs()` already returns sampled frame arrays. Passing the
        # original VideoMetadata back into the processor can trigger a second
        # sampling pass inside older HF builds, which indexes the 32-frame array
        # with original-video frame indices and crashes with IndexError.
        if predecoded_videos and video_processor is not None:
            restore_do_sample_frames = bool(getattr(video_processor, "do_sample_frames", False))
            if restore_do_sample_frames:
                video_processor.do_sample_frames = False
        else:
            video_metadata = media_cache.get("video_metadata") or []
            if video_metadata and any(meta is not None for meta in video_metadata):
                proc_kwargs["video_metadata"] = [[meta] for meta in video_metadata]
    try:
        inputs = processor(**proc_kwargs)
    finally:
        if restore_do_sample_frames is not None and video_processor is not None:
            video_processor.do_sample_frames = restore_do_sample_frames
    return {
        key: value.to(adapter._input_device) if hasattr(value, "to") else value
        for key, value in dict(inputs).items()
    }


def _decode_generated_text(adapter: QwenOmniAdapter, generated: Sequence[int]) -> str:
    return adapter._processor.tokenizer.decode(list(generated), skip_special_tokens=True)


def _extract_modality_distribution(
    adapter: QwenOmniAdapter,
    *,
    head_logits: torch.Tensor,
) -> Dict[str, Any]:
    processor = adapter._processor
    audio_token_idx = processor.tokenizer.encode("audio", add_special_tokens=False)[0]
    video_token_idx = processor.tokenizer.encode("video", add_special_tokens=False)[0]
    both_token_idx = processor.tokenizer.encode("both", add_special_tokens=False)[0]
    audio_logit = float(head_logits[0, -1, audio_token_idx].detach().cpu().item())
    video_logit = float(head_logits[0, -1, video_token_idx].detach().cpu().item())
    both_logit = float(head_logits[0, -1, both_token_idx].detach().cpu().item())
    avb_logits = torch.tensor([audio_logit, video_logit, both_logit], dtype=torch.float32)
    av_probs = torch.softmax(avb_logits, dim=0)
    audio_prob, video_prob, both_prob = [float(x) for x in av_probs.tolist()]
    ranked_modalities = sorted(
        [
            ("audio", float(audio_prob)),
            ("video", float(video_prob)),
            ("both", float(both_prob)),
        ],
        key=lambda item: item[1],
        reverse=True,
    )
    return {
        "head_modality_logits": {
            "audio": float(audio_logit),
            "video": float(video_logit),
            "both": float(both_logit),
        },
        "modality_distribution": {
            "audio": float(audio_prob),
            "video": float(video_prob),
            "both": float(both_prob),
        },
        "predicted_modality": str(ranked_modalities[0][0]),
    }


def strict_single_branch_decode(
    adapter: QwenOmniAdapter,
    *,
    question: str,
    video_path: Optional[str],
    audio_path: Optional[str],
    branch_key: str,
    max_new_tokens: int,
    budget: Dict[str, float],
    media_cache: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    if branch_key not in _BRANCH_KEYS:
        raise ValueError(f"unsupported branch_key: {branch_key}")
    media_cache = media_cache or _build_media_cache(
        adapter,
        video_path=video_path,
        audio_path=audio_path,
        budget=budget,
    )
    conversation, use_audio_in_video = _build_branch_conversation(
        branch_key=branch_key,
        question=question,
        video_path=video_path,
        audio_path=audio_path,
    )
    inputs = _prepare_processor_inputs(
        adapter,
        conversation=conversation,
        use_audio_in_video=use_audio_in_video,
        media_cache=media_cache,
    )
    thinker = adapter._model.thinker
    generated: List[int] = []
    outputs = None
    try:
        with adapter.temporary_release_cuda_reserve(f"mad_single_branch_decode:{branch_key}"):
            with torch.inference_mode():
                outputs = thinker(**inputs, use_audio_in_video=use_audio_in_video, use_cache=True)
                step_logits = outputs.logits[:, -1, :].detach()
                branch_state = outputs.past_key_values
                for _ in range(max(0, int(max_new_tokens))):
                    next_token = int(torch.argmax(step_logits, dim=-1).item())
                    generated.append(next_token)
                    if next_token == adapter._processor.tokenizer.eos_token_id:
                        break
                    next_tok_tensor = torch.tensor([[next_token]], device=adapter._input_device, dtype=torch.long)
                    outputs = thinker(
                        next_tok_tensor,
                        use_audio_in_video=use_audio_in_video,
                        use_cache=True,
                        past_key_values=branch_state,
                    )
                    branch_state = outputs.past_key_values
                    step_logits = outputs.logits[:, -1, :].detach()
    finally:
        del inputs, outputs
        thinker.rope_deltas = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    text = _decode_generated_text(adapter, generated).strip()
    return {
        "branch": str(branch_key),
        "text": text,
        "answer": normalize_yes_no(text),
        "generated_token_ids": [int(token_id) for token_id in generated],
        "n_generated_tokens": int(len(generated)),
    }


def strict_mad_head_only_modality_probe(
    adapter: QwenOmniAdapter,
    *,
    question: str,
    video_path: Optional[str],
    audio_path: Optional[str],
    budget: Dict[str, float],
    add_prompt: str = MODALITY_QUERY_PROMPT,
) -> Dict[str, Any]:
    thinker = adapter._model.thinker
    media_cache = _build_media_cache(
        adapter,
        video_path=video_path,
        audio_path=audio_path,
        budget=budget,
    )
    head_conversation, head_use_audio_in_video = _build_head_conversation(
        question,
        video_path,
        audio_path,
        add_prompt,
    )
    head_inputs = _prepare_processor_inputs(
        adapter,
        conversation=head_conversation,
        use_audio_in_video=head_use_audio_in_video,
        media_cache=media_cache,
    )
    head_outputs = None
    try:
        with adapter.temporary_release_cuda_reserve("mad_head_only_modality_probe"):
            with torch.inference_mode():
                head_outputs = thinker.forward(**head_inputs, use_audio_in_video=head_use_audio_in_video)
                head_payload = _extract_modality_distribution(
                    adapter,
                    head_logits=head_outputs.logits,
                )
    finally:
        del head_inputs, head_outputs
        thinker.rope_deltas = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return {
        "mode": "head_only_modality_probe",
        "head_query_prompt": str(add_prompt),
        **head_payload,
    }


def strict_mad_decode(
    adapter: QwenOmniAdapter,
    *,
    question: str,
    video_path: Optional[str],
    audio_path: Optional[str],
    max_new_tokens: int,
    gamma: float,
    budget: Dict[str, float],
    add_prompt: str = MODALITY_QUERY_PROMPT,
    min_new_tokens: int = 0,
    suppress_answer_tokens_before_min: bool = False,
    stop_at_sentence_after_min: bool = False,
    sampling_temperature: Optional[float] = None,
    sampling_seed: Optional[int] = None,
    sample_first_token_only: bool = False,
    sampling_top_k: int = 0,
    sampling_top_p: float = 1.0,
    force_comma_after_first_answer: bool = False,
    allowed_token_ids: Optional[Sequence[int]] = None,
) -> Dict[str, Any]:
    if sampling_temperature is not None and float(sampling_temperature) <= 0:
        raise ValueError("sampling_temperature must be positive when sampling is enabled")
    if int(sampling_top_k) < 0:
        raise ValueError("sampling_top_k must be non-negative")
    if not 0.0 < float(sampling_top_p) <= 1.0:
        raise ValueError("sampling_top_p must be in (0, 1]")
    candidate_ids: Optional[torch.Tensor] = None
    if allowed_token_ids is not None:
        normalized_ids = [int(token_id) for token_id in allowed_token_ids]
        if len(normalized_ids) < 2 or len(set(normalized_ids)) != len(normalized_ids):
            raise ValueError("allowed_token_ids must contain at least two unique token IDs")
        if sampling_temperature is not None or force_comma_after_first_answer:
            raise ValueError("candidate-constrained MAD currently supports greedy decoding only")
        candidate_ids = torch.tensor(normalized_ids, dtype=torch.long)
    processor = adapter._processor
    thinker = adapter._model.thinker
    sampling_generator: Optional[torch.Generator] = None
    if sampling_temperature is not None:
        sampling_generator = torch.Generator(device=adapter._input_device)
        if sampling_seed is not None:
            sampling_generator.manual_seed(int(sampling_seed))
    media_cache = _build_media_cache(
        adapter,
        video_path=video_path,
        audio_path=audio_path,
        budget=budget,
    )
    head_conversation, head_use_audio_in_video = _build_head_conversation(
        question,
        video_path,
        audio_path,
        add_prompt,
    )
    head_inputs = _prepare_processor_inputs(
        adapter,
        conversation=head_conversation,
        use_audio_in_video=head_use_audio_in_video,
        media_cache=media_cache,
    )
    head_outputs = None
    generated: List[int] = []
    first_step_sampling: Dict[str, Any] = {}
    first_step_candidate_logits: List[float] = []
    try:
        with adapter.temporary_release_cuda_reserve("mad_weighted_decode"):
            with torch.inference_mode():
                head_outputs = thinker.forward(**head_inputs, use_audio_in_video=head_use_audio_in_video)
                head_payload = _extract_modality_distribution(
                    adapter,
                    head_logits=head_outputs.logits,
                )
                modality_distribution = dict(head_payload.get("modality_distribution") or {})
                audio_prob = float(modality_distribution.get("audio", 0.0))
                video_prob = float(modality_distribution.get("video", 0.0))
                both_prob = float(modality_distribution.get("both", 0.0))

                branches: Dict[str, Any] = {}
                branch_use_audio_in_video: Dict[str, bool] = {}
                step_logits: Dict[str, torch.Tensor] = {}
                branch_debug: Dict[str, Dict[str, Any]] = {}
                for key in _BRANCH_KEYS:
                    conversation, use_audio_in_video = _build_branch_conversation(
                        branch_key=key,
                        question=question,
                        video_path=video_path,
                        audio_path=audio_path,
                    )
                    branch_use_audio_in_video[key] = bool(use_audio_in_video)
                    branch_inputs = _prepare_processor_inputs(
                        adapter,
                        conversation=conversation,
                        use_audio_in_video=use_audio_in_video,
                        media_cache=media_cache,
                    )
                    outputs = thinker(**branch_inputs, use_audio_in_video=use_audio_in_video, use_cache=True)
                    step_logits[key] = outputs.logits[:, -1, :].detach()
                    branches[key] = outputs.past_key_values
                    branch_debug[key] = {
                        "top_token": processor.tokenizer.decode(
                            [int(torch.argmax(step_logits[key], dim=-1).item())],
                            skip_special_tokens=True,
                        ),
                    }

                alpha_av = 2 + 2 * float(gamma) * both_prob
                alpha_v = 1 - (both_prob - video_prob) * float(gamma)
                alpha_a = 1 - (both_prob - audio_prob) * float(gamma)
                alpha_t = -(video_prob + audio_prob) * float(gamma)
                weights = {
                    "av": float(alpha_av),
                    "v": float(alpha_v),
                    "a": float(alpha_a),
                    "t": float(alpha_t),
                }

                for _ in range(max(0, int(max_new_tokens))):
                    logits_mat = torch.stack(
                        [(weights[key] * step_logits[key]).squeeze(0) for key in _BRANCH_KEYS],
                        dim=0,
                    )
                    avg_logits = logits_mat.sum(dim=0)
                    if len(generated) < max(0, int(min_new_tokens)):
                        avg_logits[processor.tokenizer.eos_token_id] = -torch.inf
                        if suppress_answer_tokens_before_min and generated:
                            avg_logits[generated[0]] = -torch.inf
                            for label in ("Yes", "No"):
                                token_ids = processor.tokenizer.encode(
                                    label,
                                    add_special_tokens=False,
                                )
                                if len(token_ids) == 1:
                                    avg_logits[int(token_ids[0])] = -torch.inf
                    forced_next_token: Optional[int] = None
                    if force_comma_after_first_answer and len(generated) == 1:
                        first_text = _decode_generated_text(adapter, generated).strip()
                        if normalize_yes_no(first_text) is not None:
                            comma_ids = processor.tokenizer.encode(",", add_special_tokens=False)
                            if len(comma_ids) != 1:
                                raise RuntimeError("comma is not a single tokenizer token")
                            forced_next_token = int(comma_ids[0])
                    should_sample = forced_next_token is None and sampling_temperature is not None and (
                        not sample_first_token_only or not generated
                    )
                    if forced_next_token is not None:
                        next_token = forced_next_token
                    elif should_sample:
                        scaled_logits = avg_logits.float() / float(sampling_temperature)
                        sample_logits = scaled_logits.clone()
                        if int(sampling_top_k) > 0:
                            top_k = min(int(sampling_top_k), int(sample_logits.numel()))
                            cutoff = torch.topk(sample_logits, k=top_k).values[-1]
                            sample_logits[sample_logits < cutoff] = -torch.inf
                        if float(sampling_top_p) < 1.0:
                            sorted_logits, sorted_indices = torch.sort(sample_logits, descending=True)
                            sorted_probs = torch.softmax(sorted_logits, dim=-1)
                            cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
                            remove = cumulative_probs > float(sampling_top_p)
                            remove[1:] = remove[:-1].clone()
                            remove[0] = False
                            sample_logits[sorted_indices[remove]] = -torch.inf
                        probs = torch.softmax(sample_logits, dim=-1)
                        if not generated:
                            label_stats: Dict[str, Dict[str, Any]] = {}
                            for label in ("Yes", "No"):
                                token_ids = processor.tokenizer.encode(label, add_special_tokens=False)
                                if len(token_ids) != 1:
                                    continue
                                token_id = int(token_ids[0])
                                token_logit = scaled_logits[token_id]
                                label_stats[label] = {
                                    "token_id": token_id,
                                    "probability": float(probs[token_id].detach().cpu().item()),
                                    "rank": int((scaled_logits > token_logit).sum().detach().cpu().item()) + 1,
                                }
                            first_step_sampling = {
                                "temperature": float(sampling_temperature),
                                "seed": None if sampling_seed is None else int(sampling_seed),
                                "top_k": int(sampling_top_k),
                                "top_p": float(sampling_top_p),
                                "labels": label_stats,
                            }
                        next_token = int(
                            torch.multinomial(probs, num_samples=1, generator=sampling_generator).item()
                        )
                    else:
                        if candidate_ids is None:
                            next_token = int(torch.argmax(avg_logits, dim=-1).item())
                        else:
                            candidate_ids_device = candidate_ids.to(device=avg_logits.device)
                            candidate_logits = avg_logits.index_select(-1, candidate_ids_device)
                            if not generated:
                                first_step_candidate_logits = [
                                    float(value)
                                    for value in candidate_logits.detach().float().cpu().reshape(-1).tolist()
                                ]
                            selected_index = int(torch.argmax(candidate_logits, dim=-1).item())
                            next_token = int(candidate_ids_device[selected_index].item())
                    generated.append(next_token)
                    if next_token == processor.tokenizer.eos_token_id:
                        break
                    if stop_at_sentence_after_min and len(generated) >= max(1, int(min_new_tokens)):
                        partial_text = _decode_generated_text(adapter, generated).strip()
                        if partial_text.endswith((".", "?", "!")):
                            break
                    next_tok_tensor = torch.tensor([[next_token]], device=adapter._input_device, dtype=torch.long)
                    for key, branch_state in list(branches.items()):
                        outputs = thinker(
                            next_tok_tensor,
                            use_audio_in_video=branch_use_audio_in_video[key],
                            use_cache=True,
                            past_key_values=branch_state,
                        )
                        branches[key] = outputs.past_key_values
                        step_logits[key] = outputs.logits[:, -1, :].detach()
    finally:
        del head_inputs, head_outputs
        thinker.rope_deltas = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    text = _decode_generated_text(adapter, generated).strip()
    return {
        "mode": "strict_weighted_decode",
        "text": text,
        "answer": normalize_yes_no(text),
        "generated_token_ids": [int(token_id) for token_id in generated],
        "n_generated_tokens": int(len(generated)),
        "min_new_tokens": int(min_new_tokens),
        "suppress_answer_tokens_before_min": bool(
            suppress_answer_tokens_before_min
        ),
        "stop_at_sentence_after_min": bool(stop_at_sentence_after_min),
        "sampling_enabled": sampling_temperature is not None,
        "sampling_temperature": (
            None if sampling_temperature is None else float(sampling_temperature)
        ),
        "sampling_seed": None if sampling_seed is None else int(sampling_seed),
        "sample_first_token_only": bool(sample_first_token_only),
        "sampling_top_k": int(sampling_top_k),
        "sampling_top_p": float(sampling_top_p),
        "force_comma_after_first_answer": bool(force_comma_after_first_answer),
        "first_step_sampling": first_step_sampling,
        "candidate_token_ids": (
            [] if candidate_ids is None else [int(value) for value in candidate_ids.tolist()]
        ),
        "first_step_candidate_logits": first_step_candidate_logits,
        **head_payload,
        "weights": {key: float(value) for key, value in weights.items()},
        "head_query_prompt": str(add_prompt),
        "branch_top_tokens": branch_debug,
    }
