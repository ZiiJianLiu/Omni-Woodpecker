"""
Qwen2.5-Omni Adapter
=====================
支持模态选择性输入的 Qwen2.5-Omni 适配器（Training-free）

核心能力：
- 视频 + 音频 + 文本 → 文本答案
- 模态屏蔽（mask_audio / mask_visual）
- 情感约束注入
- 分离推理后融合
- CMCD (Cross-Modal Consistency-guided Contrastive Decoding)
  基于 hidden state 层级跨模态一致性监控 + logit 对比的幻觉抑制
"""
import copy
import logging
import os
import re
import subprocess
import tempfile
import warnings
from contextlib import contextmanager
from pathlib import Path
from typing import Optional, Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from transformers.video_utils import load_video

from .cirpo_omni.benchmark_policy import AnswerSpaceSpec, format_question_for_answer_space
from .modules.question_conditioned_evidence import QuestionConditionedEvidenceScorer
from .asset_manager import model_repo, resolve_model_path

# 抑制第三方库的 deprecation warnings
warnings.filterwarnings("ignore", message=".*PySoundFile failed.*")
warnings.filterwarnings("ignore", message=".*librosa.core.audio.__audioread_load.*")
warnings.filterwarnings("ignore", message=".*torchcodec.*")
warnings.filterwarnings("ignore", message=".*torchvision.*video decoding.*")

logger = logging.getLogger(__name__)


_EMPTY_CACHE_SKIP_NOTICE_EMITTED = False


class _ThinkerOnlyOmniWrapper(torch.nn.Module):
    """Expose the text-producing Qwen-Omni thinker through the existing adapter API."""

    def __init__(self, thinker: torch.nn.Module):
        super().__init__()
        self.thinker = thinker
        self.hf_device_map = getattr(thinker, "hf_device_map", None)


def _env_flag(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "y", "on"}


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None:
        return default
    text = raw.strip()
    if not text:
        return default
    try:
        return int(text)
    except ValueError:
        logger.warning("Invalid integer for %s=%r; falling back to %d", name, raw, default)
        return default


def _preserve_cuda_allocator_cache_enabled() -> bool:
    return _env_flag("OMNI_PRESERVE_CUDA_CACHE") or _env_flag("OMNI_INPROCESS_GPU_CACHE_RESERVE")


def _safe_empty_cache(force: bool = False) -> None:
    global _EMPTY_CACHE_SKIP_NOTICE_EMITTED
    if not torch.cuda.is_available():
        return
    if not force and _preserve_cuda_allocator_cache_enabled():
        if not _EMPTY_CACHE_SKIP_NOTICE_EMITTED:
            logger.info(
                "Skipping torch.cuda.empty_cache() because reusable in-process CUDA cache preservation is enabled"
            )
            _EMPTY_CACHE_SKIP_NOTICE_EMITTED = True
        return
    try:
        torch.cuda.empty_cache()
    except Exception as exc:
        logger.warning("Ignoring torch.cuda.empty_cache() failure: %s: %s", type(exc).__name__, exc)


def _load_video_inputs(
    video_paths: List[str],
    video_budget: Dict[str, float],
) -> Tuple[List[np.ndarray], List[Optional[object]]]:
    loaded: List[np.ndarray] = []
    metadata: List[Optional[object]] = []
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

# Qwen2.5-Omni 默认系统提示（保留才能正常工作）
_DEFAULT_SYSTEM_PROMPT = (
    "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, "
    "capable of perceiving auditory and visual inputs, as well as generating "
    "text and speech."
)

_YN_PREFIXES = (
    "is ",
    "are ",
    "was ",
    "were ",
    "do ",
    "does ",
    "did ",
    "can ",
    "could ",
    "will ",
    "would ",
    "has ",
    "have ",
    "had ",
    "should ",
)
_YN_INSTRUCTION_RE = re.compile(
    r"\b(answer|respond)\s+with\s+(only\s+)?yes\s+or\s+no\b|\byes\s+or\s+no\b",
    re.IGNORECASE,
)

_AUDIO_QUERY_CUES = {
    'audio', 'sound', 'sounds', 'hear', 'heard', 'hearing', 'listen', 'listening',
    'voice', 'voices', 'speech', 'speaking', 'music', 'noise', 'noisy', 'loud',
    'quiet', 'audible', 'singing', 'talking', 'saying', 'shouting', 'crying',
    'barking', 'meowing', 'ringing', 'engine', 'siren',
}
_VISUAL_QUERY_CUES = {
    'video', 'image', 'visible', 'visually', 'scene', 'look', 'looks', 'looking',
    'shown', 'showing', 'see', 'seen', 'watch', 'wearing', 'holding', 'face',
    'person', 'people', 'object', 'objects', 'appear', 'appears',
}
_CROSS_MODAL_QUERY_CUES = {
    'match', 'matching', 'matched', 'consistent', 'consistency', 'align', 'aligned',
    'correspond', 'corresponding', 'fit', 'same', 'event', 'context', 'together',
    'relation', 'related', 'describe', 'describes',
}
_AFFECT_CUES = {
    'happy', 'sad', 'angry', 'fear', 'fearful', 'scared', 'surprised', 'surprise',
    'disgust', 'calm', 'neutral', 'excited', 'tense', 'tense', 'joy', 'joyful',
    'anxious', 'anxiety', 'upset', 'emotion', 'mood', 'feeling', 'feelings',
    'tone', 'tones', 'emotional', 'sentiment', 'cheerful', 'gloomy', 'frustrated',
}
_CAUSAL_CUES = {
    'because', 'since', 'due', 'therefore', 'thus', 'so', 'hence', 'implies',
    'imply', 'suggests', 'suggest', 'means', 'mean', 'causes', 'cause', 'caused',
    'indicates', 'indicate', 'shows', 'show',
}
_CSS_STOPWORDS = {
    'the', 'a', 'an', 'this', 'that', 'these', 'those', 'there', 'here', 'then', 'than',
    'with', 'from', 'into', 'onto', 'over', 'under', 'about', 'after', 'before',
    'yes', 'no', 'and', 'or', 'but', 'for', 'nor', 'yet', 'only', 'just', 'very',
    'more', 'less', 'some', 'many', 'much', 'few', 'who', 'what', 'when', 'where',
    'why', 'how', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'do', 'does',
    'did', 'can', 'could', 'will', 'would', 'has', 'have', 'had', 'should',
}


class QwenOmniAdapter:
    """Qwen2.5-Omni 适配器，支持模态控制"""

    def __init__(
        self,
        model_path: Optional[str] = None,
        device: str = "cuda:0",
        torch_dtype=torch.bfloat16,
        max_pixels: int = 602112,
        max_new_tokens: int = 256,
    ):
        # A repository-relative checkpoint is used when available (development
        # and offline validation).  Source-only checkouts transparently resolve
        # the same argument to the canonical Hugging Face snapshot.
        requested_model = model_path or os.environ.get("OWP_QWEN_MODEL_ID") or model_repo("qwen")
        self.model_path = resolve_model_path(requested_model, kind="qwen")
        self.device = device
        self.torch_dtype = torch_dtype
        self._max_pixels = max_pixels
        self._max_new_tokens = max_new_tokens
        self._model = None
        self._processor = None
        self._token_surface_cache: Dict[int, str] = {}
        self._token_style_cache: Dict[int, Dict[str, object]] = {}
        self._live_cuda_reserve_buffers: Dict[int, List[torch.Tensor]] = {}
        self._cuda_reserve_suspend_depth = 0
        self._load_model()

    def _load_model(self):
        from transformers import (
            Qwen2_5OmniForConditionalGeneration,
            Qwen2_5OmniProcessor,
        )
        from transformers.models.qwen2_5_omni.modeling_qwen2_5_omni import (
            Qwen2_5OmniThinkerForConditionalGeneration,
        )

        logger.info("加载 Qwen2.5-Omni: %s → %s", self.model_path, self.device)
        attn_implementation = os.environ.get("OMNI_ATTN_IMPLEMENTATION", "sdpa").strip() or "sdpa"
        use_fast_processor = os.environ.get("OMNI_USE_FAST_PROCESSOR", "0").strip().lower() in {
            "1", "true", "yes", "y", "on",
        }
        logger.info("Qwen2.5-Omni attention backend: %s", attn_implementation)
        logger.info("Qwen2.5-Omni processor use_fast=%s", use_fast_processor)
        thinker_only = _env_flag("OMNI_THINKER_ONLY")
        model_class = Qwen2_5OmniThinkerForConditionalGeneration if thinker_only else Qwen2_5OmniForConditionalGeneration
        logger.info("Qwen2.5-Omni thinker-only load=%s", thinker_only)

        def load_model(**kwargs):
            loaded = model_class.from_pretrained(self.model_path, **kwargs)
            return _ThinkerOnlyOmniWrapper(loaded) if thinker_only else loaded

        self._processor = Qwen2_5OmniProcessor.from_pretrained(
            self.model_path,
            use_fast=use_fast_processor,
        )
        # 视频处理参数在 _enforce_video_limits() 中统一设置
        # 使用当前可见 GPU 的 max_memory，避免把推理固定到硬编码卡号上
        if self.device == "auto":
            visible_gpu_count = torch.cuda.device_count()
            if visible_gpu_count <= 0:
                raise RuntimeError('device=auto 需要至少一张可见 CUDA GPU')
            per_gpu_limit = os.environ.get('OMNI_MAX_MEMORY_PER_GPU', '40GB').strip()
            disable_max_memory = per_gpu_limit.lower() in {"", "none", "off", "disable", "disabled", "unlimited"}
            if disable_max_memory:
                logger.info(
                    "自动分配: 可见 GPU=%s (count=%d), 不设置 max_memory 限制",
                    os.environ.get('CUDA_VISIBLE_DEVICES', 'all'),
                    visible_gpu_count,
                )
                self._model = load_model(
                    torch_dtype=self.torch_dtype,
                    device_map="auto",
                    attn_implementation=attn_implementation,
                )
            else:
                max_memory = {idx: per_gpu_limit for idx in range(visible_gpu_count)}
                logger.info(
                    "自动分配: 可见 GPU=%s (count=%d), 每卡最多 %s",
                    os.environ.get('CUDA_VISIBLE_DEVICES', 'all'),
                    visible_gpu_count,
                    per_gpu_limit,
                )
                self._model = load_model(
                    torch_dtype=self.torch_dtype,
                    device_map="auto",
                    max_memory=max_memory,
                    attn_implementation=attn_implementation,
                )
        else:
            self._model = load_model(
                torch_dtype=self.torch_dtype,
                device_map={"": self.device},
                attn_implementation=attn_implementation,
            )
        self._model.eval()

        # device_map="auto" 时，取第一个参数所在设备作为输入设备
        self._input_device = next(self._model.parameters()).device
        logger.info("Qwen2.5-Omni 加载完成 (input_device=%s)", self._input_device)

        # ★ 强制设置视频处理参数（在模型加载后立即设置，确保全局生效）
        self._enforce_video_limits()
        self._maybe_reserve_reusable_cuda_cache()

    @staticmethod
    def _parse_cuda_device_index(device_ref) -> Optional[int]:
        if device_ref is None:
            return None
        if isinstance(device_ref, int):
            return device_ref
        if isinstance(device_ref, torch.device):
            if device_ref.type != "cuda":
                return None
            return 0 if device_ref.index is None else int(device_ref.index)
        text = str(device_ref).strip().lower()
        if not text.startswith("cuda"):
            return None
        if ":" not in text:
            return 0
        try:
            return int(text.split(":", 1)[1])
        except ValueError:
            return None

    def _active_cuda_device_indices(self) -> List[int]:
        indices = set()
        hf_device_map = getattr(self._model, "hf_device_map", None)
        if isinstance(hf_device_map, dict):
            for device_ref in hf_device_map.values():
                device_index = self._parse_cuda_device_index(device_ref)
                if device_index is not None:
                    indices.add(device_index)
        if not indices:
            for parameter in self._model.parameters():
                if parameter.device.type == "cuda":
                    indices.add(0 if parameter.device.index is None else int(parameter.device.index))
        return sorted(indices)

    def _reserve_reusable_cuda_cache_on_device(
        self,
        *,
        device_index: int,
        keep_free_mib: int,
        only_if_free_mib: int,
        safety_mib: int,
        chunk_mib: int,
    ) -> None:
        if self._live_cuda_reserve_buffers.get(device_index):
            return
        mib = 1024 * 1024
        device = torch.device(f"cuda:{device_index}")
        free_bytes, total_bytes = torch.cuda.mem_get_info(device)
        free_mib = int(free_bytes // mib)
        if free_mib <= only_if_free_mib:
            logger.info(
                "Skip reusable CUDA cache warmup on %s: free_mib=%d <= threshold=%d",
                device,
                free_mib,
                only_if_free_mib,
            )
            return

        target_bytes = int(free_bytes) - ((keep_free_mib + safety_mib) * mib)
        if target_bytes <= 0:
            logger.info(
                "Skip reusable CUDA cache warmup on %s: target_mib=%d keep_free_mib=%d safety_mib=%d",
                device,
                int(target_bytes // mib),
                keep_free_mib,
                safety_mib,
            )
            return

        buffers: List[torch.Tensor] = []
        allocated_bytes = 0
        chunk_bytes = max(16 * mib, int(chunk_mib) * mib)
        logger.info(
            "Acquire live reusable CUDA reserve on %s: free_mib=%d total_mib=%d target_mib=%d keep_free_mib=%d safety_mib=%d chunk_mib=%d",
            device,
            free_mib,
            int(total_bytes // mib),
            int(target_bytes // mib),
            keep_free_mib,
            safety_mib,
            int(chunk_bytes // mib),
        )

        while allocated_bytes < target_bytes:
            step_bytes = min(chunk_bytes, target_bytes - allocated_bytes)
            try:
                tensor = torch.empty(step_bytes, dtype=torch.uint8, device=device)
                buffers.append(tensor)
                allocated_bytes += tensor.numel()
            except torch.OutOfMemoryError:
                if step_bytes <= 16 * mib:
                    break
                chunk_bytes = max(16 * mib, step_bytes // 2)

        self._live_cuda_reserve_buffers[device_index] = buffers
        free_after_hold, _ = torch.cuda.mem_get_info(device)
        logger.info(
            "Live reusable CUDA reserve active on %s: held_mib=%d free_mib_after_hold=%d",
            device,
            int(allocated_bytes // mib),
            int(free_after_hold // mib),
        )

    def _maybe_reserve_reusable_cuda_cache(self) -> None:
        if not torch.cuda.is_available():
            return
        if not _env_flag("OMNI_INPROCESS_GPU_CACHE_RESERVE"):
            return
        keep_free_mib = max(0, _env_int("OMNI_INPROCESS_RESERVE_KEEP_FREE_MB", 16384))
        only_if_free_mib = max(
            keep_free_mib,
            _env_int("OMNI_INPROCESS_RESERVE_ONLY_IF_FREE_MB", keep_free_mib + 4096),
        )
        safety_mib = max(0, _env_int("OMNI_INPROCESS_RESERVE_SAFETY_MIB", 1024))
        chunk_mib = max(16, _env_int("OMNI_INPROCESS_RESERVE_CHUNK_MIB", 256))
        device_indices = self._active_cuda_device_indices()
        if not device_indices:
            logger.info("Skip reusable CUDA cache warmup: no active CUDA devices found")
            return
        for device_index in device_indices:
            try:
                self._reserve_reusable_cuda_cache_on_device(
                    device_index=device_index,
                    keep_free_mib=keep_free_mib,
                    only_if_free_mib=only_if_free_mib,
                    safety_mib=safety_mib,
                    chunk_mib=chunk_mib,
                )
            except Exception as exc:
                logger.warning(
                    "Reusable CUDA cache warmup failed on cuda:%d: %s: %s",
                    device_index,
                    type(exc).__name__,
                    exc,
                )

    def _release_live_cuda_reserve(self, *, reason: str) -> bool:
        if not self._live_cuda_reserve_buffers:
            return False
        mib = 1024 * 1024
        device_ids = sorted(self._live_cuda_reserve_buffers.keys())
        held_mib = {
            device_index: int(sum(tensor.numel() for tensor in buffers) // mib)
            for device_index, buffers in self._live_cuda_reserve_buffers.items()
        }
        logger.info(
            "Release live reusable CUDA reserve before %s: devices=%s held_mib=%s",
            reason,
            device_ids,
            held_mib,
        )
        for device_index, buffers in list(self._live_cuda_reserve_buffers.items()):
            buffers.clear()
            del buffers
            self._live_cuda_reserve_buffers.pop(device_index, None)
        return True

    @contextmanager
    def temporary_release_cuda_reserve(self, reason: str):
        should_manage = _env_flag("OMNI_INPROCESS_GPU_CACHE_RESERVE")
        released = False
        if should_manage and self._cuda_reserve_suspend_depth == 0:
            released = self._release_live_cuda_reserve(reason=reason)
        self._cuda_reserve_suspend_depth += 1
        try:
            yield
        finally:
            self._cuda_reserve_suspend_depth = max(0, self._cuda_reserve_suspend_depth - 1)
            if should_manage and self._cuda_reserve_suspend_depth == 0:
                self._maybe_reserve_reusable_cuda_cache()

    @staticmethod
    def _path_exists(path: Optional[str]) -> bool:
        return bool(path) and Path(path).exists()

    def _resolve_audio_array(
        self,
        *,
        video_path: Optional[str] = None,
        audio_path: Optional[str] = None,
        mask_audio: bool = False,
    ) -> Optional[np.ndarray]:
        if mask_audio:
            return None
        source_path = None
        if self._path_exists(audio_path):
            source_path = audio_path
        elif self._path_exists(video_path):
            source_path = video_path
        if source_path is None:
            return None
        return self._load_audio(source_path)

    def _enforce_video_limits(self):
        """强制设置视频处理参数，充分利用 6 卡显存"""
        vp = getattr(self._processor, 'video_processor', None)
        if vp is not None:
            vp.do_sample_frames = True
            vp.fps = 2.0                 # 2 fps — 10秒视频 = 20帧
            vp.max_frames = 64           # 最多 64 帧（长视频也能覆盖）
            vp.max_pixels = 3211264      # 1792×1792 — 接近原始分辨率
            vp.min_pixels = 602112
            logger.info("视频处理参数: do_sample_frames=True, fps=2.0, "
                        "max_frames=%d, max_pixels=%d", vp.max_frames, vp.max_pixels)

        ip = getattr(self._processor, 'image_processor', None)
        if ip is not None:
            ip.max_pixels = 3211264
            ip.min_pixels = 602112

    # ──────────────────────────────────────────────────────────────────────
    # 公开接口
    # ──────────────────────────────────────────────────────────────────────

    def answer(
        self,
        video_path: Optional[str],
        question: str,
        *,
        mask_audio: bool = False,
        mask_visual: bool = False,
        emotion_constraint: Optional[str] = None,
        max_new_tokens: int = None,
        async_validator=None,
        speculative_config=None,
        return_metadata: bool = False,
        baseline_answer: Optional[str] = None,
        conflict_aware_suppressor=None,
        modality_features=None,
        conflict_report=None,
        decoder_config=None,
        safety_contexts: Optional[List[str]] = None,
        frames: Optional[List[np.ndarray]] = None,
        object_detector=None,
        audio_path: Optional[str] = None,
    ):
        """生成文本答案。

        Parameters
        ----------
        conflict_aware_suppressor : AttentionMaskSuppressor, optional
            冲突感知的注意力掩码抑制器
        modality_features : ModalityFeatures, optional
            多模态特征（用于冲突检测）
        """
        if max_new_tokens is None:
            max_new_tokens = self._max_new_tokens

        # 冲突感知的注意力掩码抑制
        if (conflict_aware_suppressor is not None
            and modality_features is not None
            and not mask_audio
            and not mask_visual):
            from modules.attention_mask_suppressor import infer_query_modality

            # 检测冲突
            conflict_signal = conflict_aware_suppressor.detect_conflict(
                visual_emotion=modality_features.visual_emotion or 'neutral',
                audio_emotion=modality_features.audio_emotion or 'neutral',
                visual_emotion_conf=modality_features.visual_emotion_conf or 0.0,
                audio_emotion_conf=modality_features.audio_emotion_conf or 0.0,
                visual_objects=modality_features.visual_objects or [],
                asr_text=modality_features.asr_text or '',
            )

            # 如果检测到冲突，应用注意力掩码
            if conflict_signal.should_suppress:
                query_modality = infer_query_modality(question)
                logger.info(
                    f"[AttentionMaskSuppressor] Conflict detected (score={conflict_signal.overall_conflict_score:.3f}), "
                    f"query_modality={query_modality}, applying attention masking"
                )

                # 根据查询模态和冲突，选择性屏蔽另一模态
                if query_modality == 'audio':
                    # 查询音频 + 冲突 → 屏蔽视觉
                    mask_visual = True
                    logger.info("[AttentionMaskSuppressor] Masking visual modality for audio query")
                else:  # visual
                    # 查询视觉 + 冲突 → 屏蔽音频
                    mask_audio = True
                    logger.info("[AttentionMaskSuppressor] Masking audio modality for visual query")

        supports_single_video_flow = self._path_exists(video_path) and not self._path_exists(audio_path)

        if async_validator is not None and not mask_audio and not mask_visual and supports_single_video_flow:
            result = self.dynamic_speculative_decoding(
                video_path,
                question,
                async_validator,
                max_new_tokens=max_new_tokens,
                emotion_constraint=emotion_constraint,
                speculative_config=speculative_config,
                baseline_answer=baseline_answer,
                safety_contexts=safety_contexts,
            )
            return result if return_metadata else result.get('text', '')
        if async_validator is not None and not supports_single_video_flow:
            logger.warning(
                "dynamic_speculative_decoding 仅支持单视频输入；当前样本回退到原始 generate 路径。"
            )

        use_dcod = (
            decoder_config is not None
            and getattr(decoder_config, 'enable_dcod', False)
            and not mask_audio
            and not mask_visual
            and supports_single_video_flow
        )
        if use_dcod:
            result = self.dcod_decode(
                video_path,
                question,
                conflict_report=conflict_report,
                modality_features=modality_features,
                decoder_config=decoder_config,
                max_new_tokens=max_new_tokens,
                frames=frames,
                object_detector=object_detector,
                emotion_constraint=emotion_constraint,
                safety_contexts=safety_contexts,
            )
            return result if return_metadata else result.get('text', '')
        if (
            decoder_config is not None
            and getattr(decoder_config, 'enable_dcod', False)
            and not supports_single_video_flow
        ):
            logger.warning(
                "DCOD 仅支持单视频输入；当前样本回退到原始 generate 路径。"
            )

        audio_array = self._resolve_audio_array(
            video_path=video_path,
            audio_path=audio_path,
            mask_audio=mask_audio,
        )

        messages = self._build_messages(
            video_path,
            question,
            audio_array=audio_array,
            mask_visual=mask_visual,
            emotion_constraint=emotion_constraint,
            safety_contexts=safety_contexts,
        )

        # 如果提供了 conflict_aware_suppressor，构建 logits processor
        logits_processor = None
        if conflict_aware_suppressor is not None and modality_features is not None:
            logits_processor = conflict_aware_suppressor.create_logits_processor(
                modality_features=modality_features,
                question=question,
                tokenizer=self._processor.tokenizer,
            )

        answer = self._generate(
            messages,
            max_new_tokens,
            audio_array=audio_array,
            logits_processor=logits_processor,
        )
        if return_metadata:
            return {'text': answer, 'metadata': {}}
        return answer

    def answer_media(
        self,
        question: str,
        *,
        video_path: Optional[str] = None,
        audio_path: Optional[str] = None,
        **kwargs,
    ):
        """通用媒体输入的原始 Qwen2.5-Omni 推理入口。"""
        return self.answer(
            video_path,
            question,
            audio_path=audio_path,
            **kwargs,
        )

    def text_answer(
        self,
        prompt: str,
        *,
        max_new_tokens: Optional[int] = None,
    ) -> str:
        """纯文本生成入口，用于基于结构化证据的答案重写。"""
        if max_new_tokens is None:
            max_new_tokens = self._max_new_tokens

        messages = [
            {
                'role': 'system',
                'content': [{'type': 'text', 'text': _DEFAULT_SYSTEM_PROMPT}],
            },
            {
                'role': 'user',
                'content': [{'type': 'text', 'text': prompt}],
            },
        ]
        return self._generate(messages, max_new_tokens, audio_array=None)


    @staticmethod
    def _has_audio_signal(modality_features) -> bool:
        if modality_features is None:
            return True
        return bool(
            getattr(modality_features, 'audio_type', None)
            or getattr(modality_features, 'audio_events', None)
            or getattr(modality_features, 'audio_event_scores', None)
            or getattr(modality_features, 'asr_text', None)
            or getattr(modality_features, 'audio_emotion', None)
            or getattr(modality_features, 'audio_has_speech', False)
        )

    @staticmethod
    def _has_visual_signal(modality_features) -> bool:
        if modality_features is None:
            return True
        return bool(
            getattr(modality_features, 'visual_objects', None)
            or getattr(modality_features, 'visual_scene', None)
            or getattr(modality_features, 'visual_emotion', None)
            or getattr(modality_features, 'visual_faces_count', 0)
        )

    @staticmethod
    def _normalize_phrase(text: str) -> str:
        text = re.sub(r"[^a-z0-9']+", ' ', (text or '').lower())
        return ' '.join(text.split())

    @staticmethod
    def _alias_match(alias: str, candidate: str) -> bool:
        alias = (alias or '').strip()
        candidate = (candidate or '').strip()
        if not alias or not candidate:
            return False
        if alias == candidate or alias in candidate or candidate in alias:
            return True
        alias_tokens = set(alias.split())
        cand_tokens = set(candidate.split())
        overlap = len(alias_tokens & cand_tokens)
        return overlap > 0 and overlap / max(1, len(alias_tokens)) >= 0.5

    def _mask_modality_features(
        self,
        modality_features,
        *,
        mask_audio: bool = False,
        mask_visual: bool = False,
    ):
        if modality_features is None:
            return None
        masked = copy.deepcopy(modality_features)
        if mask_audio:
            masked.audio_emotion = None
            masked.audio_emotion_conf = 0.0
            masked.audio_type = None
            masked.asr_text = None
            masked.asr_segments = []
            masked.asr_timestamp_index = {}
            masked.audio_events = []
            masked.audio_event_scores = {}
            masked.audio_event_timeline = []
            masked.audio_energy = 0.0
            masked.audio_has_speech = False
        if mask_visual:
            masked.visual_emotion = None
            masked.visual_emotion_conf = 0.0
            masked.visual_objects = []
            masked.visual_scene = None
            masked.visual_faces_count = 0
        return masked

    def _relevance_from_query_spec(
        self,
        query_spec: Optional[Dict[str, object]],
        *,
        modality_features=None,
        conflict_report=None,
        decoder_config=None,
    ) -> Optional[Dict[str, float]]:
        spec = query_spec or {}
        modality = str(spec.get('modality') or '')
        relation = str(spec.get('relation') or 'attribute')
        if modality not in {'audio', 'visual', 'cross_modal'}:
            return None

        if modality == 'audio':
            audio_rel, visual_rel = 0.94, 0.24
        elif modality == 'visual':
            audio_rel, visual_rel = 0.24, 0.94
        elif relation in {'consistency', 'temporal'}:
            audio_rel, visual_rel = 0.82, 0.82
        elif relation == 'emotion':
            audio_rel, visual_rel = 0.78, 0.78
        else:
            audio_rel, visual_rel = 0.74, 0.74

        floor = float(getattr(decoder_config, 'dcod_modality_floor', 0.18) or 0.18)
        if not self._has_audio_signal(modality_features):
            audio_rel = max(floor, audio_rel * 0.72)
        if not self._has_visual_signal(modality_features):
            visual_rel = max(floor, visual_rel * 0.72)

        dominant = getattr(conflict_report, 'dominant_modality', None)
        if dominant == 'audio' and audio_rel >= visual_rel:
            audio_rel = min(0.98, audio_rel + 0.04)
        elif dominant == 'visual' and visual_rel >= audio_rel:
            visual_rel = min(0.98, visual_rel + 0.04)

        return {
            'audio': max(floor, min(0.98, audio_rel)),
            'visual': max(floor, min(0.98, visual_rel)),
        }

    def _score_question_evidence(
        self,
        question: str,
        modality_features=None,
        *,
        conflict_report=None,
        frames: Optional[List[np.ndarray]] = None,
        object_detector=None,
        decoder_config=None,
        max_new_tokens: Optional[int] = None,
    ) -> Dict[str, object]:
        if modality_features is None or decoder_config is None:
            return {'handled': False, 'reason': 'missing_features_or_config'}
        scorer = QuestionConditionedEvidenceScorer(decoder_config)
        try:
            return scorer.score_question(
                question,
                modality_features,
                conflict_report=conflict_report,
                frames=frames,
                object_detector=object_detector,
                max_new_tokens=max_new_tokens,
            )
        except Exception as exc:
            logger.debug('Question evidence scoring failed: %s', exc)
            return {'handled': False, 'reason': f'question_evidence_failed:{type(exc).__name__}'}

    def _score_question_hypotheses(
        self,
        question: str,
        modality_features=None,
        *,
        conflict_report=None,
        frames: Optional[List[np.ndarray]] = None,
        object_detector=None,
        decoder_config=None,
        max_new_tokens: Optional[int] = None,
    ) -> Dict[str, object]:
        if modality_features is None or decoder_config is None:
            return {'handled': False, 'reason': 'missing_features_or_config'}
        scorer = QuestionConditionedEvidenceScorer(decoder_config)
        try:
            return scorer.score_hypotheses(
                question,
                modality_features,
                conflict_report=conflict_report,
                frames=frames,
                object_detector=object_detector,
                max_new_tokens=max_new_tokens,
            )
        except Exception as exc:
            logger.debug('Hypothesis scoring failed: %s', exc)
            return {'handled': False, 'reason': f'hypothesis_scoring_failed:{type(exc).__name__}'}

    @staticmethod
    def _format_choice_answer(candidate: Optional[Dict[str, object]], *, labels_only: bool = False) -> str:
        if not candidate:
            return ''
        label = str(candidate.get('label') or '').strip()
        text = str(candidate.get('text') or '').strip()
        if labels_only and label:
            return label
        if label and text:
            return f'{label}. {text}'
        return text or label

    def _hypothesis_choice_decision(
        self,
        question: str,
        hypothesis_meta: Optional[Dict[str, object]],
        decoder_config=None,
    ) -> Tuple[str, Dict[str, object]]:
        hypothesis_meta = hypothesis_meta or {}
        candidates = list(hypothesis_meta.get('candidates') or [])
        question_form = str(hypothesis_meta.get('question_form') or 'single_choice')
        best = hypothesis_meta.get('best_candidate') or (candidates[0] if candidates else None)
        selected = list(hypothesis_meta.get('selected_candidates') or [])
        support_threshold = float(hypothesis_meta.get('support_threshold', 0.56) or 0.56)
        margin_threshold = float(hypothesis_meta.get('margin_threshold', 0.10) or 0.10)
        top1_gap_threshold = float(hypothesis_meta.get('top1_gap_threshold', 0.06) or 0.06)
        top1_gap = float(hypothesis_meta.get('top1_gap', 0.0) or 0.0)

        if not best:
            return '', {
                'decoder': 'hypothesis_choice',
                'question': question,
                'question_form': question_form,
                'reason': 'no_candidates',
                'candidates': [],
            }

        if question_form == 'multi_select':
            chosen_candidates = selected if selected else [best]
            answer = ', '.join(
                self._format_choice_answer(candidate, labels_only=all(item.get('label') for item in chosen_candidates))
                for candidate in chosen_candidates
            )
            reason = 'supported_subset' if selected else 'fallback_top1'
        else:
            strong_top1 = bool(best.get('allow_commit', False)) and (
                float(best.get('support_score', 0.0) or 0.0) >= support_threshold
                or top1_gap >= top1_gap_threshold
            )
            answer = self._format_choice_answer(best)
            reason = 'supported_top1' if strong_top1 else 'fallback_top1'

        candidate_summary = [
            {
                'label': candidate.get('label'),
                'text': candidate.get('text'),
                'support_score': float(candidate.get('support_score', 0.0) or 0.0),
                'contradiction_score': float(candidate.get('contradiction_score', 0.0) or 0.0),
                'decision_margin': float(candidate.get('decision_margin', 0.0) or 0.0),
                'decision_score': float(candidate.get('decision_score', 0.0) or 0.0),
                'allow_commit': bool(candidate.get('allow_commit', False)),
                'query_spec': candidate.get('query_spec') or {},
            }
            for candidate in candidates
        ]
        meta = {
            'decoder': 'hypothesis_choice',
            'question': question,
            'question_form': question_form,
            'reason': reason,
            'support_threshold': support_threshold,
            'margin_threshold': margin_threshold,
            'top1_gap_threshold': top1_gap_threshold,
            'top1_gap': top1_gap,
            'selected_count': len(selected),
            'best_candidate': candidate_summary[0] if candidate_summary else None,
            'candidates': candidate_summary,
            'question_form_meta': hypothesis_meta.get('question_form_meta') or {},
        }
        return answer, meta

    def _estimate_question_relevance(
        self,
        question: str,
        question_score: Optional[Dict[str, object]],
        *,
        modality_features=None,
        conflict_report=None,
        decoder_config=None,
    ) -> Dict[str, float]:
        if question_score and bool(question_score.get('handled')) and question_score.get('modality') in {'audio', 'visual', 'cross_modal'}:
            modality = question_score.get('modality')
            if modality == 'audio':
                audio_rel, visual_rel = 0.94, 0.24
            elif modality == 'visual':
                audio_rel, visual_rel = 0.24, 0.94
            else:
                audio_rel, visual_rel = 0.80, 0.80

            floor = float(getattr(decoder_config, 'dcod_modality_floor', 0.18) or 0.18)
            if not self._has_audio_signal(modality_features):
                audio_rel = max(floor, audio_rel * 0.72)
            if not self._has_visual_signal(modality_features):
                visual_rel = max(floor, visual_rel * 0.72)
            dominant = getattr(conflict_report, 'dominant_modality', None)
            if dominant == 'audio' and audio_rel >= visual_rel:
                audio_rel = min(0.98, audio_rel + 0.03)
            elif dominant == 'visual' and visual_rel >= audio_rel:
                visual_rel = min(0.98, visual_rel + 0.03)
            return {
                'audio': max(floor, min(0.98, audio_rel)),
                'visual': max(floor, min(0.98, visual_rel)),
            }

        return self._estimate_dcod_relevance(
            question,
            modality_features=modality_features,
            conflict_report=conflict_report,
            decoder_config=decoder_config,
        )

    def _compute_muse_profile(
        self,
        question_score: Optional[Dict[str, object]],
        no_audio_score: Optional[Dict[str, object]],
        no_visual_score: Optional[Dict[str, object]],
        text_score: Optional[Dict[str, object]],
        decoder_config,
    ) -> Dict[str, float]:
        profile = {
            'yes_bias': 0.0,
            'no_bias': 0.0,
            'yes_evidence': 0.0,
            'no_evidence': 0.0,
            'support_full': 0.0,
            'contradiction_full': 0.0,
            'required_drop': 0.0,
            'shadow_support': 0.0,
            'cross_gap': 0.0,
            'text_support': 0.0,
            'handled': False,
            'question_kind': None,
            'modality': None,
        }
        if decoder_config is None or not getattr(decoder_config, 'enable_muse', True):
            return profile
        if not question_score or not bool(question_score.get('handled')):
            return profile
        if question_score.get('modality') not in {'audio', 'visual', 'cross_modal'}:
            return profile

        support_f = float(question_score.get('support_score', 0.0) or 0.0)
        contradiction_f = float(question_score.get('contradiction_score', 0.0) or 0.0)
        evidence_meta = question_score.get('evidence') or {}
        nondecisive_risk = float(evidence_meta.get('nondecisive_risk', 0.0) or 0.0)
        support_na = float((no_audio_score or {}).get('support_score', 0.0) or 0.0)
        support_nv = float((no_visual_score or {}).get('support_score', 0.0) or 0.0)
        support_t = float((text_score or {}).get('support_score', 0.0) or 0.0)
        support_threshold = float(question_score.get('support_threshold', 0.72) or 0.72)
        modality = question_score.get('modality')

        if modality == 'audio':
            required_drop = max(0.0, support_f - support_na)
            shadow_support = max(0.0, support_na - support_t)
            cross_gap = 0.0
        elif modality == 'visual':
            required_drop = max(0.0, support_f - support_nv)
            shadow_support = max(0.0, support_nv - support_t)
            cross_gap = 0.0
        else:
            drop_audio = max(0.0, support_f - support_na)
            drop_visual = max(0.0, support_f - support_nv)
            required_drop = min(drop_audio, drop_visual)
            shadow_support = max(
                max(0.0, support_na - support_t),
                max(0.0, support_nv - support_t),
            )
            cross_gap = max(0.0, 0.20 - min(drop_audio, drop_visual)) + 0.5 * abs(drop_audio - drop_visual)

        insufficiency = max(0.0, support_threshold - support_f)
        yes_evidence = (
            support_f
            + float(getattr(decoder_config, 'muse_counterfactual_weight', 0.95)) * required_drop
            - float(getattr(decoder_config, 'muse_shadow_weight', 1.10)) * shadow_support
            - float(getattr(decoder_config, 'muse_contradiction_weight', 0.85)) * contradiction_f
            - float(getattr(decoder_config, 'muse_insufficiency_weight', 0.70)) * insufficiency
            - float(getattr(decoder_config, 'muse_cross_modal_weight', 0.55)) * cross_gap
        )
        no_evidence = (
            contradiction_f
            + 0.80 * shadow_support
            + float(getattr(decoder_config, 'muse_insufficiency_weight', 0.70)) * insufficiency
            + float(getattr(decoder_config, 'muse_cross_modal_weight', 0.55)) * cross_gap
            - 0.55 * support_f
            - 0.20 * required_drop
        )

        nondecisive_weight = float(getattr(decoder_config, 'muse_nondecisive_weight', 0.75))
        yes_evidence -= nondecisive_weight * nondecisive_risk
        no_evidence += 0.70 * nondecisive_weight * nondecisive_risk

        if not bool(question_score.get('allow_positive_flip', True)):
            yes_evidence -= 0.30
        if not bool(question_score.get('allow_negative_flip', True)):
            no_evidence -= 0.30

        logit_weight = float(getattr(decoder_config, 'muse_logit_weight', 1.35))
        profile.update(
            {
                'yes_bias': float(logit_weight * np.tanh(yes_evidence)),
                'no_bias': float(logit_weight * np.tanh(no_evidence)),
                'yes_evidence': float(yes_evidence),
                'no_evidence': float(no_evidence),
                'support_full': float(support_f),
                'contradiction_full': float(contradiction_f),
                'required_drop': float(required_drop),
                'shadow_support': float(shadow_support),
                'cross_gap': float(cross_gap),
                'text_support': float(support_t),
                'nondecisive_risk': float(nondecisive_risk),
                'handled': True,
                'question_kind': question_score.get('question_kind'),
                'modality': modality,
            }
        )
        return profile

    def _extract_audio_time_anchors(
        self,
        question_score: Optional[Dict[str, object]],
        modality_features,
        decoder_config,
    ) -> List[Dict[str, object]]:
        if not question_score or not bool(question_score.get('handled')) or modality_features is None:
            return []
        scorer = QuestionConditionedEvidenceScorer(decoder_config)
        spec = question_score.get('question_spec') or {}
        aliases: List[str] = []
        entity = spec.get('entity')
        if entity:
            aliases.extend(scorer._aliases_for(entity, 'audio'))
        evidence = question_score.get('evidence') or {}
        alignment = evidence.get('alignment') or {}
        alignment_cue = alignment.get('cue') or evidence.get('alignment_cue')
        if alignment_cue:
            aliases.append(str(alignment_cue))

        normalized_aliases = []
        seen = set()
        for alias in aliases:
            normalized = self._normalize_phrase(alias)
            if normalized and normalized not in seen:
                normalized_aliases.append(normalized)
                seen.add(normalized)

        anchors: List[Dict[str, object]] = []
        for item in getattr(modality_features, 'audio_event_timeline', []) or []:
            label = self._normalize_phrase(str(item.get('label', '')))
            if normalized_aliases and not any(self._alias_match(alias, label) for alias in normalized_aliases):
                continue
            if not label:
                continue
            anchors.append(
                {
                    'start': float(item.get('start', 0.0) or 0.0),
                    'end': float(item.get('end', 0.0) or 0.0),
                    'score': float(item.get('score', 0.0) or 0.0),
                    'label': label,
                    'source': 'ast',
                }
            )

        asr_index = getattr(modality_features, 'asr_timestamp_index', {}) or {}
        for alias in normalized_aliases:
            for span in asr_index.get(alias, [])[:4]:
                anchors.append(
                    {
                        'start': float(span.get('start', 0.0) or 0.0),
                        'end': float(span.get('end', 0.0) or 0.0),
                        'score': 0.70,
                        'label': alias,
                        'source': 'asr',
                    }
                )

        anchors.sort(key=lambda item: (item.get('score', 0.0), item.get('end', 0.0) - item.get('start', 0.0)), reverse=True)
        return anchors

    def _extract_visual_time_anchors(
        self,
        question_score: Optional[Dict[str, object]],
    ) -> Tuple[List[float], int]:
        if not question_score or not bool(question_score.get('handled')):
            return [], 0
        evidence = question_score.get('evidence') or {}
        frame_indices: List[int] = []
        max_frames = 0

        if question_score.get('modality') == 'visual':
            frame_indices = list(evidence.get('support_frame_indices', []) or [])
            max_frames = int(evidence.get('grounding_max_frames', 0) or 0)
            peak_idx = int(evidence.get('peak_frame_index', -1) or -1)
            if not frame_indices and peak_idx >= 0:
                frame_indices = [peak_idx]
        elif question_score.get('question_kind') == 'relation_grounded':
            alignment = evidence.get('alignment') or {}
            cue_results = alignment.get('cue_results') or []
            best_cue = self._normalize_phrase(str(alignment.get('cue') or ''))
            selected = None
            for item in cue_results:
                cue = self._normalize_phrase(str(item.get('cue') or ''))
                if best_cue and cue == best_cue:
                    selected = item
                    break
            if selected is None and cue_results:
                selected = max(cue_results, key=lambda item: float(item.get('score', 0.0) or 0.0))
            presence = (selected or {}).get('presence') or {}
            frame_indices = list(presence.get('grounding_support_frame_indices', []) or [])
            max_frames = int(presence.get('grounding_max_frames', 0) or 0)
            peak_idx = int(presence.get('grounding_peak_frame_index', -1) or -1)
            if not frame_indices and peak_idx >= 0:
                frame_indices = [peak_idx]

        if not frame_indices:
            return [], max_frames
        if max_frames <= 1:
            return [0.5 for _ in frame_indices], max_frames
        positions = sorted({float(idx) / float(max_frames - 1) for idx in frame_indices if idx >= 0})
        return positions, max_frames

    def _compute_tabs_profile(
        self,
        question_score: Optional[Dict[str, object]],
        modality_features,
        decoder_config,
    ) -> Dict[str, float]:
        profile = {
            'risk': 0.0,
            'yes_penalty': 0.0,
            'no_bonus': 0.0,
            'mean_gap': 0.0,
            'support_ratio': 0.0,
            'audio_anchor_count': 0,
            'visual_anchor_count': 0,
            'handled': False,
        }
        if decoder_config is None or not getattr(decoder_config, 'enable_tabs', True):
            return profile
        if not question_score or not bool(question_score.get('handled')):
            return profile
        if question_score.get('modality') not in {'audio', 'cross_modal'}:
            return profile

        audio_anchors = self._extract_audio_time_anchors(question_score, modality_features, decoder_config)
        visual_positions, _ = self._extract_visual_time_anchors(question_score)
        if not audio_anchors or not visual_positions:
            profile.update(
                {
                    'audio_anchor_count': len(audio_anchors),
                    'visual_anchor_count': len(visual_positions),
                }
            )
            return profile

        max_end = max(max(float(anchor.get('end', 0.0) or 0.0), 1e-6) for anchor in audio_anchors)
        tolerance = float(getattr(decoder_config, 'tabs_alignment_tolerance', 0.18) or 0.18)
        gaps: List[float] = []
        matched = 0
        for position in visual_positions:
            min_gap = 1.0
            for anchor in audio_anchors:
                start_norm = float(anchor.get('start', 0.0) or 0.0) / max_end
                end_norm = float(anchor.get('end', 0.0) or 0.0) / max_end
                if start_norm <= position <= end_norm:
                    min_gap = 0.0
                    break
                if position < start_norm:
                    gap = start_norm - position
                else:
                    gap = position - end_norm
                min_gap = min(min_gap, gap)
            gaps.append(float(min_gap))
            if min_gap <= tolerance:
                matched += 1

        mean_gap = float(np.mean(gaps)) if gaps else 0.0
        support_ratio = float(matched / max(1, len(visual_positions)))
        risk = max(0.0, mean_gap - tolerance) / max(1e-6, 1.0 - tolerance)
        risk = risk * float(getattr(decoder_config, 'tabs_max_gap_weight', 0.90) or 0.90)
        if question_score.get('modality') == 'cross_modal':
            risk = min(1.0, risk + 0.15 * (1.0 - support_ratio))
        else:
            risk = min(1.0, risk)

        yes_penalty = float(getattr(decoder_config, 'tabs_yes_penalty_weight', 0.85)) * risk
        no_bonus = float(getattr(decoder_config, 'tabs_no_bonus_weight', 0.40)) * risk
        profile.update(
            {
                'risk': float(risk),
                'yes_penalty': float(yes_penalty),
                'no_bonus': float(no_bonus),
                'mean_gap': float(mean_gap),
                'support_ratio': float(support_ratio),
                'audio_anchor_count': len(audio_anchors),
                'visual_anchor_count': len(visual_positions),
                'handled': True,
            }
        )
        return profile

    def _token_surface(self, token_id: int) -> str:
        token_id = int(token_id)
        surface = self._token_surface_cache.get(token_id)
        if surface is None:
            surface = self._processor.tokenizer.decode([token_id], skip_special_tokens=True)
            surface = re.sub(r'\s+', ' ', surface or '').strip().lower()
            self._token_surface_cache[token_id] = surface
        return surface

    def _token_style(self, token_id: int) -> Dict[str, object]:
        token_id = int(token_id)
        cached = self._token_style_cache.get(token_id)
        if cached is not None:
            return cached

        surface = self._token_surface(token_id)
        parts = [part for part in re.split(r'[^a-z]+', surface) if part]
        detail = any(len(part) >= 3 and part not in _CSS_STOPWORDS for part in parts)
        affect = any(part in _AFFECT_CUES for part in parts)
        causal = any(part in _CAUSAL_CUES for part in parts)
        cached = {
            'surface': surface,
            'detail': detail,
            'affect': affect,
            'causal': causal,
        }
        self._token_style_cache[token_id] = cached
        return cached

    def _estimate_dcod_relevance(
        self,
        question: str,
        modality_features=None,
        conflict_report=None,
        decoder_config=None,
    ) -> Dict[str, float]:
        if decoder_config is not None:
            try:
                query_spec = QuestionConditionedEvidenceScorer(decoder_config).infer_query_spec(question)
            except Exception:
                query_spec = None
            spec_relevance = self._relevance_from_query_spec(
                query_spec,
                modality_features=modality_features,
                conflict_report=conflict_report,
                decoder_config=decoder_config,
            )
            if spec_relevance is not None:
                return spec_relevance

        normalized = re.sub(r'[^a-z0-9\s]+', ' ', (question or '').lower())
        tokens = set(normalized.split())
        audio_hits = len(tokens & _AUDIO_QUERY_CUES)
        visual_hits = len(tokens & _VISUAL_QUERY_CUES)
        cross_hits = len(tokens & _CROSS_MODAL_QUERY_CUES)

        audio_rel = 0.58
        visual_rel = 0.58
        if cross_hits or (audio_hits > 0 and visual_hits > 0):
            audio_rel = 0.72
            visual_rel = 0.72
        elif audio_hits > visual_hits:
            audio_rel = 0.92
            visual_rel = 0.26
        elif visual_hits > audio_hits:
            audio_rel = 0.26
            visual_rel = 0.92

        floor = float(getattr(decoder_config, 'dcod_modality_floor', 0.18) or 0.18)
        if not self._has_audio_signal(modality_features):
            audio_rel = max(floor, audio_rel * 0.72)
        if not self._has_visual_signal(modality_features):
            visual_rel = max(floor, visual_rel * 0.72)

        dominant = getattr(conflict_report, 'dominant_modality', None)
        if dominant == 'audio' and audio_rel >= visual_rel:
            audio_rel = min(0.98, audio_rel + 0.04)
        elif dominant == 'visual' and visual_rel >= audio_rel:
            visual_rel = min(0.98, visual_rel + 0.04)

        return {
            'audio': max(floor, min(0.98, audio_rel)),
            'visual': max(floor, min(0.98, visual_rel)),
        }

    def _build_css_profile(self, conflict_report=None, decoder_config=None) -> Dict[str, float]:
        profile = {'detail': 0.0, 'affect': 0.0, 'causal': 0.0}
        if decoder_config is None or not getattr(decoder_config, 'enable_css', True):
            return profile

        content_conflict = bool(getattr(conflict_report, 'audio_video_content_conflict', False))
        emotion_conflict = bool(getattr(conflict_report, 'audio_video_emotion_conflict', False))
        emotion_distance = float(getattr(conflict_report, 'emotion_distance', 0.0) or 0.0)
        if content_conflict:
            profile['detail'] += float(getattr(decoder_config, 'css_detail_conflict_weight', 0.55))
            profile['causal'] += float(getattr(decoder_config, 'css_causal_conflict_weight', 0.45))
        if emotion_conflict:
            affect_weight = float(getattr(decoder_config, 'css_affect_conflict_weight', 0.80))
            affect_scale = min(1.35, 0.75 + 0.30 * emotion_distance)
            profile['affect'] += affect_weight * affect_scale
            profile['causal'] += 0.20 * affect_scale
        return profile

    @staticmethod
    def _build_state_from_prefill(inputs, out, rope_deltas) -> Dict[str, object]:
        state = {
            'inputs': inputs,
            'cache': out.past_key_values,
            'logits': out.logits[:, -1, :].clone(),
            'position': inputs['input_ids'].shape[1],
            'rope_deltas': rope_deltas,
        }
        if hasattr(out, 'hidden_states') and out.hidden_states is not None:
            del out.hidden_states
        if hasattr(out, 'attentions') and out.attentions is not None:
            del out.attentions
        del out
        return state

    def _build_lwa_profile(
        self,
        probe_result: Optional[Dict[str, torch.Tensor]],
        relevance: Dict[str, float],
        conflict_report=None,
        decoder_config=None,
    ) -> Dict[str, float]:
        if probe_result is None or decoder_config is None or not getattr(decoder_config, 'enable_lwa', True):
            return {
                'agreement': 1.0,
                'relevant_witness': 1.0,
                'counterpart_witness': 0.0,
                'relevance_match': 1.0,
                'risk': 0.0,
                'collapse_layer': -1.0,
                'audio_witness': 0.5,
                'visual_witness': 0.5,
            }

        layer_consistency = probe_result['layer_consistency']
        audio_norms = probe_result['audio_norms']
        video_norms = probe_result['video_norms']
        agreement = float(layer_consistency.mean().item()) if len(layer_consistency) > 0 else 1.0
        late_consistency = float(layer_consistency[len(layer_consistency) // 2:].mean().item()) if len(layer_consistency) > 0 else agreement
        collapse = self._find_collapse_layer(
            layer_consistency,
            threshold=float(getattr(decoder_config, 'lwa_collapse_threshold', 0.30)),
        )
        total_audio = float(audio_norms.mean().item())
        total_visual = float(video_norms.mean().item())
        denom = max(total_audio + total_visual, 1e-6)
        audio_witness = total_audio / denom
        visual_witness = total_visual / denom

        audio_rel = float(relevance.get('audio', 0.5))
        visual_rel = float(relevance.get('visual', 0.5))
        if abs(audio_rel - visual_rel) <= 0.10:
            relevant_witness = min(audio_witness, visual_witness)
            counterpart_witness = max(audio_witness, visual_witness)
        elif audio_rel > visual_rel:
            relevant_witness = audio_witness
            counterpart_witness = visual_witness
        else:
            relevant_witness = visual_witness
            counterpart_witness = audio_witness

        relevance_match = audio_witness * audio_rel + visual_witness * visual_rel
        agreement = max(0.0, min(1.0, 0.55 * agreement + 0.45 * late_consistency))
        if collapse is not None:
            agreement = max(0.0, agreement - 0.08)
        conflict_penalty = 0.0
        if bool(getattr(conflict_report, 'audio_video_content_conflict', False)):
            conflict_penalty += 0.10
        if bool(getattr(conflict_report, 'audio_video_emotion_conflict', False)):
            conflict_penalty += 0.05
        support = 0.5 * agreement + 0.5 * relevance_match
        risk = max(0.0, 0.62 - support + conflict_penalty)

        return {
            'agreement': agreement,
            'relevant_witness': float(relevant_witness),
            'counterpart_witness': float(counterpart_witness),
            'relevance_match': float(relevance_match),
            'risk': float(risk),
            'collapse_layer': float(collapse if collapse is not None else -1),
            'audio_witness': float(audio_witness),
            'visual_witness': float(visual_witness),
        }

    def _compute_dcod_components(
        self,
        logits_f: torch.Tensor,
        logits_na: torch.Tensor,
        logits_nv: torch.Tensor,
        logits_t: torch.Tensor,
        relevance: Dict[str, float],
        decoder_config,
        conflict_report=None,
        lwa_profile: Optional[Dict[str, float]] = None,
    ) -> Dict[str, torch.Tensor]:
        audio_credit = torch.clamp(logits_f - logits_na, min=0.0)
        visual_credit = torch.clamp(logits_f - logits_nv, min=0.0)
        supported_credit = torch.maximum(
            audio_credit * float(relevance.get('audio', 0.5)),
            visual_credit * float(relevance.get('visual', 0.5)),
        )
        prior_pressure = torch.clamp(logits_t - torch.maximum(logits_na, logits_nv), min=0.0)
        irrelevant_pressure = (
            torch.clamp(audio_credit * (1.0 - float(relevance.get('audio', 0.5))), min=0.0)
            + torch.clamp(visual_credit * (1.0 - float(relevance.get('visual', 0.5))), min=0.0)
        )
        overlap_pressure = torch.minimum(audio_credit, visual_credit)
        debt = torch.clamp(
            float(getattr(decoder_config, 'dcod_credit_floor', 0.22)) - supported_credit,
            min=0.0,
        )
        debt = debt * float(getattr(decoder_config, 'dcod_debt_weight', 1.0))
        debt = debt + prior_pressure * float(getattr(decoder_config, 'dcod_prior_weight', 0.85))
        debt = debt + irrelevant_pressure * float(getattr(decoder_config, 'dcod_irrelevant_weight', 0.65))

        content_conflict = bool(getattr(conflict_report, 'audio_video_content_conflict', False))
        emotion_conflict = bool(getattr(conflict_report, 'audio_video_emotion_conflict', False))
        conflict_scale = 0.0
        if content_conflict:
            conflict_scale += 1.0
        if emotion_conflict:
            conflict_scale += 0.35
        if conflict_scale > 0.0:
            debt = debt + overlap_pressure * float(getattr(decoder_config, 'dcod_overlap_weight', 0.18)) * conflict_scale

        lwa_profile = lwa_profile or {}
        lwa_agreement = float(lwa_profile.get('agreement', 1.0))
        lwa_relevant_witness = float(lwa_profile.get('relevant_witness', 1.0))
        lwa_counterpart_witness = float(lwa_profile.get('counterpart_witness', 0.0))
        lwa_relevance_match = float(lwa_profile.get('relevance_match', 1.0))
        lwa_risk = float(lwa_profile.get('risk', 0.0))
        lwa_support_scale = 0.65 + 0.35 * max(lwa_agreement, lwa_relevance_match)
        supported_credit = supported_credit * lwa_support_scale
        witness_gap = max(0.0, float(getattr(decoder_config, 'lwa_min_agreement', 0.24)) - lwa_agreement)
        if witness_gap > 0.0 or lwa_risk > 0.0:
            witness_tensor = torch.full_like(debt, witness_gap + lwa_risk)
            debt = debt + witness_tensor * float(getattr(decoder_config, 'lwa_low_witness_penalty', 0.26))
        if lwa_agreement < 1.0:
            debt = debt + overlap_pressure * (1.0 - lwa_agreement) * float(getattr(decoder_config, 'lwa_agreement_weight', 0.42))
        if lwa_counterpart_witness > 0.0:
            debt = debt + irrelevant_pressure * lwa_counterpart_witness * float(getattr(decoder_config, 'lwa_counterpart_penalty', 0.30))

        return {
            'audio_credit': audio_credit,
            'visual_credit': visual_credit,
            'supported_credit': supported_credit,
            'prior_pressure': prior_pressure,
            'irrelevant_pressure': irrelevant_pressure,
            'overlap_pressure': overlap_pressure,
            'debt': debt,
            'lwa_agreement': torch.full_like(debt, lwa_agreement),
            'lwa_relevant_witness': torch.full_like(debt, lwa_relevant_witness),
            'lwa_counterpart_witness': torch.full_like(debt, lwa_counterpart_witness),
            'lwa_risk': torch.full_like(debt, lwa_risk),
        }

    def _apply_dcod_logits(
        self,
        logits_f: torch.Tensor,
        logits_na: torch.Tensor,
        logits_nv: torch.Tensor,
        logits_t: torch.Tensor,
        *,
        relevance: Dict[str, float],
        css_profile: Dict[str, float],
        decoder_config,
        conflict_report=None,
        lwa_profile: Optional[Dict[str, float]] = None,
        top_k: int = 48,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        vocab_size = logits_f.shape[-1]
        top_k = max(2, min(int(top_k), vocab_size))
        topk_vals, topk_ids = logits_f.topk(top_k, dim=-1)
        threshold = topk_vals[:, -1].unsqueeze(-1)
        mask = logits_f < threshold

        components = self._compute_dcod_components(
            logits_f,
            logits_na,
            logits_nv,
            logits_t,
            relevance,
            decoder_config,
            conflict_report=conflict_report,
            lwa_profile=lwa_profile,
        )
        adjusted = logits_f - components['debt']

        if getattr(decoder_config, 'enable_css', True):
            css_penalty = torch.zeros_like(adjusted)
            css_base_strength = float(getattr(decoder_config, 'css_base_strength', 0.55))
            local_risk = (
                components['prior_pressure']
                + components['irrelevant_pressure']
                + 0.5 * torch.clamp(
                    float(getattr(decoder_config, 'dcod_credit_floor', 0.22)) - components['supported_credit'],
                    min=0.0,
                )
            )
            for token_id in topk_ids[0].tolist():
                style = self._token_style(token_id)
                style_weight = 0.0
                if style.get('detail'):
                    style_weight += float(css_profile.get('detail', 0.0))
                if style.get('affect'):
                    style_weight += float(css_profile.get('affect', 0.0))
                if style.get('causal'):
                    style_weight += float(css_profile.get('causal', 0.0))
                if style_weight <= 0.0:
                    continue
                css_penalty[0, token_id] = css_base_strength * style_weight * local_risk[0, token_id]
            adjusted = adjusted - css_penalty
            components['css_penalty'] = css_penalty
        else:
            components['css_penalty'] = torch.zeros_like(adjusted)

        adjusted = adjusted.masked_fill(mask, float('-inf'))
        return adjusted, components

    def _dcod_yes_no_decision(
        self,
        logits_f: torch.Tensor,
        logits_na: torch.Tensor,
        logits_nv: torch.Tensor,
        logits_t: torch.Tensor,
        *,
        question: str,
        relevance: Dict[str, float],
        decoder_config,
        conflict_report=None,
        lwa_profile: Optional[Dict[str, float]] = None,
        css_profile: Optional[Dict[str, float]] = None,
        question_score: Optional[Dict[str, object]] = None,
        muse_profile: Optional[Dict[str, float]] = None,
        tabs_profile: Optional[Dict[str, float]] = None,
    ) -> Tuple[str, Dict[str, object]]:
        tokenizer = self._processor.tokenizer
        yes_id = tokenizer.encode('Yes', add_special_tokens=False)[0]
        no_id = tokenizer.encode('No', add_special_tokens=False)[0]
        candidate_ids = [yes_id, no_id]

        scored, components = self._apply_dcod_logits(
            logits_f[:, candidate_ids],
            logits_na[:, candidate_ids],
            logits_nv[:, candidate_ids],
            logits_t[:, candidate_ids],
            relevance=relevance,
            css_profile=css_profile or {'detail': 0.0, 'affect': 0.0, 'causal': 0.0},
            decoder_config=decoder_config,
            conflict_report=conflict_report,
            lwa_profile=lwa_profile,
            top_k=2,
        )

        scored = scored.clone()
        muse_profile = muse_profile or {}
        tabs_profile = tabs_profile or {}
        yes_adjustment = float(muse_profile.get('yes_bias', 0.0)) - float(tabs_profile.get('yes_penalty', 0.0))
        no_adjustment = float(muse_profile.get('no_bias', 0.0)) + float(tabs_profile.get('no_bonus', 0.0))
        scored[0, 0] = scored[0, 0] + yes_adjustment
        scored[0, 1] = scored[0, 1] + no_adjustment

        raw_choice = 'Yes' if logits_f[0, yes_id] >= logits_f[0, no_id] else 'No'
        choice_index = int(scored[0].argmax().item())
        chosen = 'Yes' if choice_index == 0 else 'No'

        margin = float((scored[0, 0] - scored[0, 1]).item())
        min_margin = float(getattr(decoder_config, 'dcod_yes_no_margin', 0.02))
        if abs(margin) < min_margin:
            chosen = raw_choice

        evidence_guard = 'none'
        if question_score and bool(question_score.get('handled')):
            support = float(question_score.get('support_score', 0.0) or 0.0)
            contradiction = float(question_score.get('contradiction_score', 0.0) or 0.0)
            support_threshold = float(question_score.get('support_threshold', 0.72) or 0.72)
            contradiction_threshold = float(question_score.get('contradiction_threshold', 0.72) or 0.72)
            allow_positive_flip = bool(question_score.get('allow_positive_flip', True))
            allow_negative_flip = bool(question_score.get('allow_negative_flip', True))

            if chosen == 'Yes' and not allow_positive_flip and support < support_threshold:
                evidence_guard = 'blocked_unjustified_positive'
                chosen = raw_choice if raw_choice != 'Yes' else 'No'
            elif chosen == 'No' and not allow_negative_flip and contradiction < contradiction_threshold:
                evidence_guard = 'blocked_unjustified_negative'
                chosen = raw_choice if raw_choice != 'No' else 'Yes'

        question_evidence_summary = None
        if question_score and bool(question_score.get('handled')):
            evidence_meta = question_score.get('evidence') or {}
            question_evidence_summary = {
                'handled': True,
                'question_kind': question_score.get('question_kind'),
                'modality': question_score.get('modality'),
                'question_spec': question_score.get('question_spec'),
                'support_score': float(question_score.get('support_score', 0.0) or 0.0),
                'contradiction_score': float(question_score.get('contradiction_score', 0.0) or 0.0),
                'margin': float(question_score.get('margin', 0.0) or 0.0),
                'allow_positive_flip': bool(question_score.get('allow_positive_flip', True)),
                'allow_negative_flip': bool(question_score.get('allow_negative_flip', True)),
                'direct_support': bool(evidence_meta.get('direct_support', False)),
                'proxy_support': bool(evidence_meta.get('proxy_support', False)),
                'negative_evidence_ready': bool(evidence_meta.get('negative_evidence_ready', False)),
                'nondecisive_risk': float(evidence_meta.get('nondecisive_risk', 0.0) or 0.0),
                'support_calibration': evidence_meta.get('support_calibration'),
                'shadow_guard': evidence_meta.get('cross_modal_shadow_guard'),
            }

        meta = {
            'decoder': 'dcod_muse_tabs_css_lwa',
            'question': question,
            'margin': margin,
            'raw_choice': raw_choice,
            'evidence_guard': evidence_guard,
            'audio_relevance': float(relevance.get('audio', 0.0)),
            'visual_relevance': float(relevance.get('visual', 0.0)),
            'yes_supported_credit': float(components['supported_credit'][0, 0].item()),
            'no_supported_credit': float(components['supported_credit'][0, 1].item()),
            'yes_debt': float(components['debt'][0, 0].item()),
            'no_debt': float(components['debt'][0, 1].item()),
            'yes_adjustment': float(yes_adjustment),
            'no_adjustment': float(no_adjustment),
            'lwa_agreement': float(components['lwa_agreement'][0, 0].item()),
            'lwa_risk': float(components['lwa_risk'][0, 0].item()),
            'css_profile': css_profile or {'detail': 0.0, 'affect': 0.0, 'causal': 0.0},
            'lwa_profile': lwa_profile or {},
            'muse_profile': muse_profile,
            'tabs_profile': tabs_profile,
            'question_evidence': question_evidence_summary,
        }
        logger.info(
            'DCOD-MUSE-TABS yes/no: answer=%s margin=%.3f yes_adj=%.3f no_adj=%.3f yes_debt=%.3f no_debt=%.3f',
            chosen,
            margin,
            yes_adjustment,
            no_adjustment,
            meta['yes_debt'],
            meta['no_debt'],
        )
        return chosen, meta

    def dcod_decode(
        self,
        video_path: str,
        question: str,
        *,
        conflict_report=None,
        modality_features=None,
        decoder_config=None,
        max_new_tokens: Optional[int] = None,
        frames: Optional[List[np.ndarray]] = None,
        object_detector=None,
        emotion_constraint: Optional[str] = None,
        safety_contexts: Optional[List[str]] = None,
    ) -> Dict[str, object]:
        if max_new_tokens is None:
            max_new_tokens = self._max_new_tokens
        if decoder_config is None:
            raise ValueError('decoder_config is required for DCOD decoding')

        audio_array = self._load_audio(video_path) if Path(video_path).exists() else None
        scorer = QuestionConditionedEvidenceScorer(decoder_config)
        question_form = scorer.classify_question_form(
            question,
            max_new_tokens=max_new_tokens,
        )
        query_spec = scorer.infer_query_spec(question, max_new_tokens=max_new_tokens)
        if question_form.get('form') in {'single_choice', 'multi_select'}:
            hypothesis_meta = self._score_question_hypotheses(
                question,
                modality_features,
                conflict_report=conflict_report,
                frames=frames,
                object_detector=object_detector,
                decoder_config=decoder_config,
                max_new_tokens=max_new_tokens,
            )
            answer, meta = self._hypothesis_choice_decision(
                question,
                hypothesis_meta,
                decoder_config=decoder_config,
            )
            meta['query_spec'] = query_spec
            meta['guidance_context_count'] = len(safety_contexts or [])
            return {'text': answer, 'metadata': meta}

        is_yes_no = question_form.get('form') == 'yes_no'
        question_score = (
            self._score_question_evidence(
                question,
                modality_features,
                conflict_report=conflict_report,
                frames=frames,
                object_detector=object_detector,
                decoder_config=decoder_config,
                max_new_tokens=max_new_tokens,
            )
            if is_yes_no else {'handled': False, 'reason': 'non_yes_no'}
        )
        relevance = self._estimate_question_relevance(
            question,
            question_score,
            modality_features=modality_features,
            conflict_report=conflict_report,
            decoder_config=decoder_config,
        )
        css_profile = self._build_css_profile(conflict_report=conflict_report, decoder_config=decoder_config)

        use_text_branch = bool(getattr(decoder_config, 'dcod_use_text_branch', True))

        if is_yes_no:
            no_audio_features = self._mask_modality_features(modality_features, mask_audio=True)
            no_visual_features = self._mask_modality_features(modality_features, mask_visual=True)
            text_only_features = self._mask_modality_features(modality_features, mask_audio=True, mask_visual=True)
            no_audio_question_score = self._score_question_evidence(
                question,
                no_audio_features,
                conflict_report=conflict_report,
                frames=frames,
                object_detector=object_detector,
                decoder_config=decoder_config,
                max_new_tokens=max_new_tokens,
            )
            no_visual_question_score = self._score_question_evidence(
                question,
                no_visual_features,
                conflict_report=conflict_report,
                frames=None,
                object_detector=object_detector,
                decoder_config=decoder_config,
                max_new_tokens=max_new_tokens,
            )
            text_question_score = self._score_question_evidence(
                question,
                text_only_features,
                conflict_report=conflict_report,
                frames=None,
                object_detector=object_detector,
                decoder_config=decoder_config,
                max_new_tokens=max_new_tokens,
            )
            muse_profile = self._compute_muse_profile(
                question_score,
                no_audio_question_score,
                no_visual_question_score,
                text_question_score,
                decoder_config,
            )
            tabs_profile = self._compute_tabs_profile(question_score, modality_features, decoder_config)

            lwa_profile = {
                'agreement': 1.0,
                'relevant_witness': 1.0,
                'counterpart_witness': 0.0,
                'relevance_match': 1.0,
                'risk': 0.0,
                'collapse_layer': -1.0,
                'audio_witness': 0.5,
                'visual_witness': 0.5,
            }
            if getattr(decoder_config, 'enable_lwa', True):
                inputs_full = self._prepare_probe_inputs(
                    video_path,
                    question,
                    audio_array=audio_array,
                    emotion_constraint=emotion_constraint,
                    safety_contexts=safety_contexts,
                )
                probe_result, out_full, _ = self._probe_cross_modal_consistency(
                    inputs_full,
                    use_cache=False,
                )
                logits_full = out_full.logits[:, -1, :].clone()
                del out_full, inputs_full
                lwa_profile = self._build_lwa_profile(
                    probe_result,
                    relevance,
                    conflict_report=conflict_report,
                    decoder_config=decoder_config,
                )
                _safe_empty_cache()
            else:
                logits_full = self._prefill_last_logits(
                    self._prepare_probe_inputs(
                        video_path,
                        question,
                        audio_array=audio_array,
                        emotion_constraint=emotion_constraint,
                        safety_contexts=safety_contexts,
                    ),
                    use_cache=False,
                )

            logger.info(
                'DCOD-MUSE-TABS-CSS-LWA decode: audio_rel=%.2f visual_rel=%.2f css(detail=%.2f affect=%.2f causal=%.2f) muse(y=%.2f n=%.2f drop=%.2f shadow=%.2f) tabs(risk=%.2f gap=%.2f anchors=%d/%d) lwa(agree=%.2f risk=%.2f aw=%.2f vw=%.2f)',
                relevance['audio'],
                relevance['visual'],
                css_profile['detail'],
                css_profile['affect'],
                css_profile['causal'],
                float(muse_profile.get('yes_evidence', 0.0)),
                float(muse_profile.get('no_evidence', 0.0)),
                float(muse_profile.get('required_drop', 0.0)),
                float(muse_profile.get('shadow_support', 0.0)),
                float(tabs_profile.get('risk', 0.0)),
                float(tabs_profile.get('mean_gap', 0.0)),
                int(tabs_profile.get('audio_anchor_count', 0) or 0),
                int(tabs_profile.get('visual_anchor_count', 0) or 0),
                lwa_profile['agreement'],
                lwa_profile['risk'],
                lwa_profile['audio_witness'],
                lwa_profile['visual_witness'],
            )

            logits_no_audio = self._prefill_last_logits(
                self._prepare_probe_inputs(
                    video_path,
                    question,
                    audio_array=None,
                    emotion_constraint=emotion_constraint,
                    safety_contexts=safety_contexts,
                ),
                use_cache=False,
            )
            logits_no_visual = self._prefill_last_logits(
                self._prepare_probe_inputs(
                    video_path,
                    question,
                    audio_array=audio_array,
                    mask_visual=True,
                    emotion_constraint=emotion_constraint,
                    safety_contexts=safety_contexts,
                ),
                use_cache=False,
            )

            if use_text_branch:
                text_logits = self._prefill_last_logits(
                    self._prepare_probe_inputs(
                        video_path,
                        question,
                        audio_array=None,
                        mask_visual=True,
                        emotion_constraint=emotion_constraint,
                        safety_contexts=safety_contexts,
                    ),
                    use_cache=False,
                )
            else:
                text_logits = torch.maximum(logits_no_audio, logits_no_visual)

            answer, meta = self._dcod_yes_no_decision(
                logits_full,
                logits_no_audio,
                logits_no_visual,
                text_logits,
                question=question,
                relevance=relevance,
                decoder_config=decoder_config,
                conflict_report=conflict_report,
                lwa_profile=lwa_profile,
                css_profile=css_profile,
                question_score=question_score,
                muse_profile=muse_profile,
                tabs_profile=tabs_profile,
            )
            meta['question_form'] = question_form.get('form')
            meta['query_spec'] = query_spec
            meta['guidance_context_count'] = len(safety_contexts or [])
            del logits_full, logits_no_audio, logits_no_visual, text_logits
            _safe_empty_cache()
            return {'text': answer, 'metadata': meta}

        inputs_full = self._prepare_inputs(
            video_path,
            question,
            audio_array=audio_array,
            emotion_constraint=emotion_constraint,
            safety_contexts=safety_contexts,
        )
        lwa_profile = {
            'agreement': 1.0,
            'relevant_witness': 1.0,
            'counterpart_witness': 0.0,
            'relevance_match': 1.0,
            'risk': 0.0,
            'collapse_layer': -1.0,
            'audio_witness': 0.5,
            'visual_witness': 0.5,
        }
        if getattr(decoder_config, 'enable_lwa', True):
            probe_result, out_full, rope_deltas_full = self._probe_cross_modal_consistency(inputs_full)
            state_full = self._build_state_from_prefill(inputs_full, out_full, rope_deltas_full)
            lwa_profile = self._build_lwa_profile(
                probe_result,
                relevance,
                conflict_report=conflict_report,
                decoder_config=decoder_config,
            )
        else:
            state_full = self._prepare_streaming_state(
                video_path,
                question,
                audio_array=audio_array,
                emotion_constraint=emotion_constraint,
                safety_contexts=safety_contexts,
            )

        logger.info(
            'DCOD-CSS-LWA decode: audio_rel=%.2f visual_rel=%.2f css(detail=%.2f affect=%.2f causal=%.2f) lwa(agree=%.2f risk=%.2f aw=%.2f vw=%.2f)',
            relevance['audio'],
            relevance['visual'],
            css_profile['detail'],
            css_profile['affect'],
            css_profile['causal'],
            lwa_profile['agreement'],
            lwa_profile['risk'],
            lwa_profile['audio_witness'],
            lwa_profile['visual_witness'],
        )

        state_no_audio = self._prepare_streaming_state(
            video_path,
            question,
            audio_array=None,
            emotion_constraint=emotion_constraint,
            safety_contexts=safety_contexts,
        )
        state_no_visual = self._prepare_streaming_state(
            video_path,
            question,
            audio_array=audio_array,
            mask_visual=True,
            emotion_constraint=emotion_constraint,
            safety_contexts=safety_contexts,
        )
        state_text = (
            self._prepare_streaming_state(
                video_path,
                question,
                audio_array=None,
                mask_visual=True,
                emotion_constraint=emotion_constraint,
                safety_contexts=safety_contexts,
            )
            if use_text_branch else None
        )

        eos_id = self._processor.tokenizer.eos_token_id
        generated_ids: List[int] = []
        top_k = int(getattr(decoder_config, 'dcod_top_k', 48))
        repetition_penalty = float(getattr(decoder_config, 'dcod_repetition_penalty', 1.12))
        last_components: Optional[Dict[str, torch.Tensor]] = None

        for _ in range(max_new_tokens):
            logits_f = state_full['logits']
            logits_na = state_no_audio['logits']
            logits_nv = state_no_visual['logits']
            logits_t = (
                state_text['logits'] if state_text is not None
                else torch.maximum(logits_na, logits_nv)
            )
            dcod_logits, last_components = self._apply_dcod_logits(
                logits_f,
                logits_na,
                logits_nv,
                logits_t,
                relevance=relevance,
                css_profile=css_profile,
                decoder_config=decoder_config,
                conflict_report=conflict_report,
                lwa_profile=lwa_profile,
                top_k=top_k,
            )
            if generated_ids and repetition_penalty > 1.0:
                dcod_logits = self._apply_repetition_penalty(dcod_logits, generated_ids, repetition_penalty)

            next_token = dcod_logits.argmax(dim=-1)
            if next_token.item() == eos_id:
                break
            generated_ids.append(int(next_token.item()))

            if self._has_ngram_repeat(generated_ids, n=3, max_repeat=3):
                generated_ids = self._truncate_at_repeat(generated_ids, n=3)
                logger.debug('DCOD n-gram 重复检测，截断至 %d tokens', len(generated_ids))
                break

            state_full = self._advance_generation_state(state_full, next_token)
            state_no_audio = self._advance_generation_state(state_no_audio, next_token)
            state_no_visual = self._advance_generation_state(state_no_visual, next_token)
            if state_text is not None:
                state_text = self._advance_generation_state(state_text, next_token)

        answer = self._processor.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
        meta = {
            'decoder': 'dcod_css_lwa',
            'n_tokens': len(generated_ids),
            'audio_relevance': relevance['audio'],
            'visual_relevance': relevance['visual'],
            'css_profile': css_profile,
            'lwa_profile': lwa_profile,
            'question_form': question_form.get('form'),
            'query_spec': query_spec,
            'guidance_context_count': len(safety_contexts or []),
        }
        if last_components is not None and generated_ids:
            meta['last_step_avg_debt'] = float(last_components['debt'].mean().item())
            meta['last_step_avg_supported_credit'] = float(last_components['supported_credit'].mean().item())
        del state_full, state_no_audio, state_no_visual
        if state_text is not None:
            del state_text
        _safe_empty_cache()
        logger.info('DCOD-CSS 生成完成: %d tokens', len(generated_ids))
        return {'text': answer, 'metadata': meta}

    def separate_inference_fusion(
        self,
        video_path: str,
        question: str,
        max_new_tokens: int = None,
    ) -> str:
        """分离推理后融合：分别用视觉和音频推理，再让模型融合"""
        if max_new_tokens is None:
            max_new_tokens = self._max_new_tokens
        visual_answer = self.answer(
            video_path, question, mask_audio=True, max_new_tokens=max_new_tokens,
        )
        audio_answer = self.answer(
            video_path, question, mask_visual=True, max_new_tokens=max_new_tokens,
        )

        fusion_question = (
            f"Based on the video, two analyses were made:\n"
            f"Visual analysis: {visual_answer}\n"
            f"Audio analysis: {audio_answer}\n"
            f"Original question: {question}\n"
            f"Provide a final answer that reconciles both analyses."
        )
        return self.answer(
            video_path, fusion_question, max_new_tokens=max_new_tokens,
        )

    # ──────────────────────────────────────────────────────────────────────
    # Multimodal Contrastive Decoding (MCD)
    # ──────────────────────────────────────────────────────────────────────

    def contrastive_decode(
        self,
        video_path: str,
        question: str,
        conflict_report=None,
        *,
        alpha: float = 0.5,
        beta: float = 0.5,
        max_new_tokens: int = None,
        adaptive_weights: bool = True,
        repetition_penalty: float = 1.2,
        top_k_filter: int = 50,
        use_cmcd: bool = True,
        cmcd_collapse_threshold: float = 0.3,
        use_video_branch: bool = True,
    ) -> str:
        """CMCD — Cross-Modal Consistency-guided Contrastive Decoding

        升级版对比解码：在 prefill 阶段探测各层音视频 hidden state 一致性，
        找到一致性崩塌层，据此动态调整 α/β 权重，聚焦抑制音频/情感幻觉。

        Parameters
        ----------
        use_cmcd : bool
            是否启用 CMCD（hidden state 层级监控）。
            False 时退化为基于 conflict_report 的 MCD。
        cmcd_collapse_threshold : float
            一致性崩塌检测阈值（相邻层余弦相似度降幅）
        use_video_branch : bool
            是否使用 no_video 分支。False 时只用 2-branch（full + no_audio），
            节省显存，适合 GPU 数量较少的情况。
        """
        if max_new_tokens is None:
            max_new_tokens = self._max_new_tokens
        is_yn_question = max_new_tokens is not None and max_new_tokens <= 10
        audio_array = self._load_audio(video_path) if Path(video_path).exists() else None

        inputs_full = self._prepare_inputs(
            video_path, question, audio_array=audio_array,
        )

        # 检查 token 数
        seq_len = inputs_full["input_ids"].shape[1]
        logger.info("CMCD inputs_full: %d tokens, video=%s, has_audio=%s",
                     seq_len, video_path.split('/')[-1], audio_array is not None)
        if seq_len > 32768:
            logger.warning("Token 序列较长 (%d > 32768)，可能影响位置编码精度", seq_len)

        inputs_no_audio = self._prepare_inputs(
            video_path, question, audio_array=None,
        )

        # 根据配置决定是否使用 no_video 分支
        if use_video_branch:
            inputs_no_video = self._prepare_inputs(
                video_path, question, audio_array=audio_array, mask_visual=True,
            )
        else:
            inputs_no_video = None
            beta = 0.0  # 2-branch 模式，不使用视频对比

        thinker = self._model.thinker

        with torch.inference_mode():
            # ── CMCD: 探测跨模态一致性（仅在 full 分支做） ──
            if use_cmcd:
                probe_result, out_full, rope_deltas_full = (
                    self._probe_cross_modal_consistency(inputs_full)
                )
                layer_cons = probe_result["layer_consistency"]
                collapse = self._find_collapse_layer(
                    layer_cons, threshold=cmcd_collapse_threshold,
                )
                if use_video_branch:
                    alpha, beta = self._compute_cmcd_weights(
                        layer_cons,
                        probe_result["audio_norms"],
                        probe_result["video_norms"],
                        collapse,
                        base_alpha=alpha,
                        base_beta=beta,
                    )
                else:
                    # 2-branch 模式：只计算 alpha
                    alpha, _ = self._compute_cmcd_weights(
                        layer_cons,
                        probe_result["audio_norms"],
                        probe_result["video_norms"],
                        collapse,
                        base_alpha=alpha,
                        base_beta=0.0,
                    )
                    beta = 0.0
                # 立即释放探测结果
                del probe_result, layer_cons
            else:
                # 退化为旧版 MCD
                if adaptive_weights and conflict_report is not None:
                    alpha, beta = self._compute_adaptive_weights(
                        conflict_report, alpha, beta,
                    )
                    logger.info("MCD 自适应权重: α=%.3f β=%.3f", alpha, beta)
                out_full = thinker(**inputs_full, use_cache=True)
                rope_deltas_full = (
                    thinker.rope_deltas.clone()
                    if thinker.rope_deltas is not None else None
                )
                if not use_video_branch:
                    beta = 0.0

            # 提取 logits 和 KV cache
            logits_f = out_full.logits[:, -1, :].clone()
            cache_full = out_full.past_key_values
            pos_full = inputs_full["input_ids"].shape[1]
            # 删除大张量
            del out_full.logits
            if hasattr(out_full, 'hidden_states') and out_full.hidden_states is not None:
                del out_full.hidden_states
            if hasattr(out_full, 'attentions') and out_full.attentions is not None:
                del out_full.attentions
            _safe_empty_cache()

            # Prefill no-audio 分支
            out_no_audio = thinker(**inputs_no_audio, use_cache=True)
            rope_deltas_na = thinker.rope_deltas.clone() if thinker.rope_deltas is not None else None
            logits_na = out_no_audio.logits[:, -1, :].clone()
            cache_na = out_no_audio.past_key_values
            pos_na = inputs_no_audio["input_ids"].shape[1]
            del out_no_audio.logits
            if hasattr(out_no_audio, 'hidden_states') and out_no_audio.hidden_states is not None:
                del out_no_audio.hidden_states
            if hasattr(out_no_audio, 'attentions') and out_no_audio.attentions is not None:
                del out_no_audio.attentions
            _safe_empty_cache()

            # Prefill no-video 分支（如果启用）
            if use_video_branch and inputs_no_video is not None:
                out_no_video = thinker(**inputs_no_video, use_cache=True)
                rope_deltas_nv = thinker.rope_deltas.clone() if thinker.rope_deltas is not None else None
                logits_nv = out_no_video.logits[:, -1, :].clone()
                cache_nv = out_no_video.past_key_values
                pos_nv = inputs_no_video["input_ids"].shape[1]
                del out_no_video.logits
                if hasattr(out_no_video, 'hidden_states') and out_no_video.hidden_states is not None:
                    del out_no_video.hidden_states
                if hasattr(out_no_video, 'attentions') and out_no_video.attentions is not None:
                    del out_no_video.attentions
                _safe_empty_cache()
            else:
                # 2-branch 模式：no_video 分支用 dummy
                logits_nv = torch.zeros_like(logits_f)
                cache_nv = None
                rope_deltas_nv = None
                pos_nv = 0

            cd_logits = self._apply_contrastive_logits(
                logits_f, logits_na, logits_nv, alpha, beta, top_k_filter,
            )

            # ── PCD: Probability-level Contrastive Decision ──
            # 对 Yes/No 问题，直接在概率层面做对比决策，不走逐 token 生成
            if is_yn_question:
                answer = self._probability_contrastive_decision(
                    logits_f, logits_na, logits_nv, alpha, beta,
                )
                # 清理
                del cache_full, cache_na, logits_f, logits_na, logits_nv
                if cache_nv is not None:
                    del cache_nv
                del inputs_full, inputs_no_audio
                if inputs_no_video is not None:
                    del inputs_no_video
                del rope_deltas_full, rope_deltas_na
                if rope_deltas_nv is not None:
                    del rope_deltas_nv
                thinker.rope_deltas = None
                _safe_empty_cache()
                mode = "CPCD-3branch" if use_video_branch else "CPCD-2branch"
                logger.info("CPCD 决策完成: answer=%s (α=%.3f, β=%.3f, %s)",
                            answer, alpha, beta, mode)
                return answer

            eos_id = self._processor.tokenizer.eos_token_id
            generated_ids = []

            for step in range(max_new_tokens):
                # Repetition penalty
                if generated_ids and repetition_penalty > 1.0:
                    cd_logits = self._apply_repetition_penalty(
                        cd_logits, generated_ids, repetition_penalty,
                    )

                next_token = cd_logits.argmax(dim=-1)

                if next_token.item() == eos_id:
                    break

                generated_ids.append(next_token.item())

                # N-gram blocking: 检测到 3-gram 重复 3 次则终止
                if self._has_ngram_repeat(generated_ids, n=3, max_repeat=3):
                    # 回退到重复开始前的位置
                    generated_ids = self._truncate_at_repeat(generated_ids, n=3)
                    logger.debug("MCD n-gram 重复检测，截断至 %d tokens", len(generated_ids))
                    break

                next_input = next_token.unsqueeze(0)

                # Full forward
                thinker.rope_deltas = rope_deltas_full
                cp_full = torch.tensor([pos_full], device=self._input_device)
                out_full = thinker(
                    input_ids=next_input,
                    past_key_values=cache_full,
                    use_cache=True,
                    cache_position=cp_full,
                )
                pos_full += 1

                # No-audio forward
                thinker.rope_deltas = rope_deltas_na
                cp_na = torch.tensor([pos_na], device=self._input_device)
                out_no_audio = thinker(
                    input_ids=next_input,
                    past_key_values=cache_na,
                    use_cache=True,
                    cache_position=cp_na,
                )
                pos_na += 1

                logits_f = out_full.logits[:, -1, :]
                logits_na = out_no_audio.logits[:, -1, :]

                # No-video forward（如果启用）
                if use_video_branch and cache_nv is not None:
                    thinker.rope_deltas = rope_deltas_nv
                    cp_nv = torch.tensor([pos_nv], device=self._input_device)
                    out_no_video = thinker(
                        input_ids=next_input,
                        past_key_values=cache_nv,
                        use_cache=True,
                        cache_position=cp_nv,
                    )
                    pos_nv += 1
                    logits_nv = out_no_video.logits[:, -1, :]
                    cache_nv = out_no_video.past_key_values
                else:
                    logits_nv = torch.zeros_like(logits_f)

                cd_logits = self._apply_contrastive_logits(
                    logits_f, logits_na, logits_nv, alpha, beta, top_k_filter,
                )

                cache_full = out_full.past_key_values
                cache_na = out_no_audio.past_key_values

        answer = self._processor.tokenizer.decode(
            generated_ids, skip_special_tokens=True,
        ).strip()

        del cache_full, cache_na
        if cache_nv is not None:
            del cache_nv
        del out_full, out_no_audio
        if use_video_branch and 'out_no_video' in locals():
            del out_no_video
        del inputs_full, inputs_no_audio
        if inputs_no_video is not None:
            del inputs_no_video
        del rope_deltas_full, rope_deltas_na
        if rope_deltas_nv is not None:
            del rope_deltas_nv
        thinker.rope_deltas = None
        _safe_empty_cache()

        mode = "3-branch" if use_video_branch else "2-branch"
        logger.info("CMCD 生成完成: %d tokens (α=%.3f, β=%.3f, %s)",
                    len(generated_ids), alpha, beta, mode)
        return answer

    def _probability_contrastive_decision(
        self,
        logits_f: torch.Tensor,
        logits_na: torch.Tensor,
        logits_nv: torch.Tensor,
        alpha: float,
        beta: float,
    ) -> str:
        """CPCD — Cross-modal Probability Contrastive Decoding

        核心思想：通过比较 full / no-audio / no-video 三个分支的概率分布，
        定位幻觉源模态，然后选择最可靠的分支作为最终答案。

        策略：
        1. 如果三分支一致 → 直接用 full（无跨模态冲突）
        2. 如果某个缺失分支与 full 不同 → 该模态是决策关键因素
           - 计算该模态的"幻觉嫌疑度" = 偏移幅度
           - 如果嫌疑度超过阈值，信任缺失分支（去掉幻觉源后的判断）
        3. 如果两个缺失分支都与 full 不同 → 选择偏移更大的模态作为主要幻觉源
        """
        tokenizer = self._processor.tokenizer
        yes_id = tokenizer.encode("Yes", add_special_tokens=False)[0]
        no_id = tokenizer.encode("No", add_special_tokens=False)[0]

        def get_yn_probs(logits):
            yn_logits = logits[0, [yes_id, no_id]]
            probs = torch.softmax(yn_logits, dim=0)
            return probs[0].item(), probs[1].item()

        p_yes_f, p_no_f = get_yn_probs(logits_f)
        p_yes_na, p_no_na = get_yn_probs(logits_na)
        p_yes_nv, p_no_nv = get_yn_probs(logits_nv)

        vote_f = "Yes" if p_yes_f > 0.5 else "No"
        vote_na = "Yes" if p_yes_na > 0.5 else "No"
        vote_nv = "Yes" if p_yes_nv > 0.5 else "No"

        # 偏移幅度（绝对值）
        shift_a = abs(p_yes_f - p_yes_na)  # 音频对决策的影响力
        shift_v = abs(p_yes_f - p_yes_nv)  # 视频对决策的影响力

        # 翻转检测
        audio_flips = (vote_na != vote_f)   # 去掉音频后答案翻转
        video_flips = (vote_nv != vote_f)   # 去掉视频后答案翻转

        # 决策阈值：偏移超过此值才认为是显著的幻觉信号
        flip_threshold = 0.05

        if not audio_flips and not video_flips:
            # 三分支一致 → 无跨模态冲突，信任 full
            answer = vote_f
            reason = "unanimous"
        elif audio_flips and not video_flips:
            # 只有音频导致翻转 → 音频是幻觉源
            if shift_a > flip_threshold:
                answer = vote_na  # 信任去掉音频后的判断
                reason = f"audio_halluc(shift={shift_a:.3f})→trust_no_audio"
            else:
                answer = vote_f
                reason = f"audio_flip_weak(shift={shift_a:.3f})"
        elif video_flips and not audio_flips:
            # 只有视频导致翻转 → 视频是幻觉源
            if shift_v > flip_threshold:
                answer = vote_nv  # 信任去掉视频后的判断
                reason = f"video_halluc(shift={shift_v:.3f})→trust_no_video"
            else:
                answer = vote_f
                reason = f"video_flip_weak(shift={shift_v:.3f})"
        else:
            # 两个模态都导致翻转 → 选择偏移更大的作为主要幻觉源
            if shift_a >= shift_v:
                answer = vote_na
                reason = f"both_flip→audio_dominant(a={shift_a:.3f},v={shift_v:.3f})"
            else:
                answer = vote_nv
                reason = f"both_flip→video_dominant(a={shift_a:.3f},v={shift_v:.3f})"

        logger.info(
            "CPCD: full=%s(Y=%.3f) na=%s(Y=%.3f) nv=%s(Y=%.3f) "
            "shift_a=%.3f shift_v=%.3f flip_a=%s flip_v=%s → %s (%s)",
            vote_f, p_yes_f, vote_na, p_yes_na, vote_nv, p_yes_nv,
            shift_a, shift_v, audio_flips, video_flips, answer, reason,
        )
        return answer

    @staticmethod
    def _apply_contrastive_logits(
        logits_f: torch.Tensor,
        logits_na: torch.Tensor,
        logits_nv: torch.Tensor,
        alpha: float,
        beta: float,
        top_k: int = 50,
    ) -> torch.Tensor:
        """Token-level Adaptive Contrastive Decoding

        核心改进：不是简单地减去 no_audio/no_video 的 logits，
        而是只抑制那些"因为某个模态而异常升高"的 token。

        策略：
        1. 找到 logits_full 的 top-k 候选
        2. 对每个候选 token，计算它在 full vs no_audio/no_video 中的差异
        3. 如果某个 token 在 full 中远高于 no_audio → 它可能是音频幻觉
        4. 如果某个 token 在 full 中远高于 no_video → 它可能是视频幻觉
        5. 只对这些"异常升高"的 token 施加惩罚，其他 token 保持不变
        """
        # 找到 logits_full 的 top-k 候选
        topk_vals, _ = logits_f.topk(top_k, dim=-1)
        threshold = topk_vals[:, -1].unsqueeze(-1)
        mask = logits_f < threshold

        # 计算每个 token 因音频/视频引入的 logit 增量
        audio_boost = logits_f - logits_na  # 正值 = 音频让这个 token 更可能
        video_boost = logits_f - logits_nv  # 正值 = 视频让这个 token 更可能

        # 只惩罚正向增量（即模态引入的幻觉），不惩罚负向（模态抑制的）
        audio_penalty = torch.clamp(audio_boost, min=0) * alpha
        video_penalty = torch.clamp(video_boost, min=0) * beta

        # 对比解码：从 full logits 中减去模态引入的幻觉部分
        cd = logits_f - audio_penalty - video_penalty

        # 将不在 top-k 中的位置设为 -inf
        cd[mask] = float("-inf")
        return cd

    @staticmethod
    def _apply_repetition_penalty(
        logits: torch.Tensor,
        generated_ids: list,
        penalty: float,
    ) -> torch.Tensor:
        """对已生成的 token 施加 repetition penalty"""
        logits = logits.clone()
        unique_ids = set(generated_ids)
        for token_id in unique_ids:
            if logits[0, token_id] > 0:
                logits[0, token_id] /= penalty
            else:
                logits[0, token_id] *= penalty
        return logits

    @staticmethod
    def _has_ngram_repeat(ids: list, n: int = 3, max_repeat: int = 3) -> bool:
        """检测是否存在 n-gram 连续重复 max_repeat 次（检测 1/2/3-gram）"""
        for gram_size in range(1, n + 1):
            if len(ids) < gram_size * max_repeat:
                continue
            tail = ids[-(gram_size * max_repeat):]
            ngram = tuple(tail[:gram_size])
            all_same = True
            for i in range(max_repeat):
                if tuple(tail[i * gram_size:(i + 1) * gram_size]) != ngram:
                    all_same = False
                    break
            if all_same:
                return True
        return False

    @staticmethod
    def _truncate_at_repeat(ids: list, n: int = 3) -> list:
        """截断到第一次重复出现的位置（检测 1/2/3-gram）"""
        if len(ids) < 2:
            return ids
        # 检测最短的重复 gram
        for gram_size in range(1, n + 1):
            if len(ids) < gram_size * 2:
                continue
            last_ngram = tuple(ids[-gram_size:])
            pos = len(ids) - gram_size
            while pos >= gram_size:
                prev_ngram = tuple(ids[pos - gram_size:pos])
                if prev_ngram != last_ngram:
                    break
                pos -= gram_size
            if pos < len(ids) - gram_size:
                # 找到了重复，保留第一次出现
                return ids[:pos + gram_size]
        return ids

    @staticmethod
    def _compute_adaptive_weights(
        conflict_report, default_alpha: float, default_beta: float,
    ) -> tuple:
        """根据冲突报告自适应调整 α/β

        策略：
        - 音频不可信 → 增大 α（更强地抑制音频幻觉）
        - 视觉不可信 → 增大 β（更强地抑制视觉幻觉）
        - 情感冲突严重 → 同时增大 α 和 β
        - 无冲突 → 使用较小的默认值做轻微对比
        """
        alpha = default_alpha
        beta = default_beta

        if conflict_report is None:
            return alpha, beta

        # 音频不可信 → 增大 α
        if "audio" in getattr(conflict_report, "unreliable_modalities", []):
            alpha = min(alpha * 1.5, 1.0)

        # 视觉不可信 → 增大 β
        if "visual" in getattr(conflict_report, "unreliable_modalities", []):
            beta = min(beta * 1.5, 1.0)

        # 情感冲突严重 → 根据距离调整
        emo_dist = getattr(conflict_report, "emotion_distance", 0.0)
        if emo_dist > 1.0:
            scale = min(emo_dist / 2.0, 1.5)
            alpha *= scale
            beta *= scale
            alpha = min(alpha, 1.0)
            beta = min(beta, 1.0)

        return alpha, beta

    # ──────────────────────────────────────────────────────────────────────
    # CMCD: Cross-Modal Consistency-guided Contrastive Decoding
    # 基于 hidden state 层级跨模态一致性监控，聚焦音频/情感幻觉抑制
    # ──────────────────────────────────────────────────────────────────────

    # Qwen2.5-Omni 特殊 token ID
    _AUDIO_TOKEN_ID = 151646
    _VIDEO_TOKEN_ID = 151656

    def _probe_cross_modal_consistency(
        self,
        inputs: Dict[str, torch.Tensor],
        probe_layers: Optional[List[int]] = None,
        *,
        use_cache: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """在 prefill 阶段探测各层 audio/video token 的 hidden state 一致性

        通过 register_forward_hook 捕获指定层的输出，计算音频 token
        与视频 token 的 hidden state 余弦相似度。

        Returns
        -------
        dict with:
          - "layer_consistency": Tensor[n_layers] 每层的音视频一致性
          - "audio_norms": Tensor[n_layers] 音频 token 的 hidden state 范数
          - "video_norms": Tensor[n_layers] 视频 token 的 hidden state 范数
        """
        thinker = self._model.thinker
        n_layers = len(thinker.model.layers)

        if probe_layers is None:
            # 默认探测 4 层（每 8 层采样一次），降低开销
            # 32 层模型 → 采样 [0, 8, 16, 24]
            probe_layers = list(range(0, n_layers, max(1, n_layers // 4)))

        input_ids = inputs["input_ids"]
        audio_mask = (input_ids == self._AUDIO_TOKEN_ID)
        video_mask = (input_ids == self._VIDEO_TOKEN_ID)

        has_audio = audio_mask.any().item()
        has_video = video_mask.any().item()

        if not has_audio or not has_video:
            # 缺少某个模态 — 仍需做 prefill，但返回全 1（无冲突信号）
            out = thinker(**inputs, use_cache=use_cache)
            rope_deltas = (
                thinker.rope_deltas.clone()
                if use_cache and thinker.rope_deltas is not None else None
            )
            ones = torch.ones(len(probe_layers), device=self._input_device)
            return {
                "layer_consistency": ones,
                "audio_norms": ones,
                "video_norms": ones,
            }, out, rope_deltas

        # 注册 hooks 捕获 hidden states
        captured = {}
        hooks = []

        def _make_hook(layer_idx):
            def hook_fn(module, input, output):
                # output[0] = hidden_states: [batch, seq_len, hidden_dim]
                hs = output[0].detach()
                # device_map="auto" 时各层可能在不同 GPU，mask 需跟随
                dev = hs.device
                a_hs = hs[audio_mask.to(dev)].mean(dim=0)  # [hidden_dim]
                v_hs = hs[video_mask.to(dev)].mean(dim=0)  # [hidden_dim]
                captured[layer_idx] = (a_hs, v_hs)
            return hook_fn

        for idx in probe_layers:
            h = thinker.model.layers[idx].register_forward_hook(
                _make_hook(idx),
            )
            hooks.append(h)

        try:
            # 执行 prefill forward（带 hidden state 捕获）
            out = thinker(**inputs, use_cache=use_cache)
            rope_deltas = (
                thinker.rope_deltas.clone()
                if use_cache and thinker.rope_deltas is not None else None
            )
        finally:
            for h in hooks:
                h.remove()

        # 计算每层一致性（转为标量，释放 GPU tensor）
        consistency = []
        audio_norms = []
        video_norms = []

        for idx in probe_layers:
            a_hs, v_hs = captured[idx]
            cos_sim = F.cosine_similarity(
                a_hs.unsqueeze(0), v_hs.unsqueeze(0),
            ).item()
            consistency.append(cos_sim)
            audio_norms.append(a_hs.norm().item())
            video_norms.append(v_hs.norm().item())
            # 立即释放该层的 hidden state
            del a_hs, v_hs

        # 清理 captured 字典
        captured.clear()
        del captured

        result = {
            "layer_consistency": torch.tensor(
                consistency, device=self._input_device,
            ),
            "audio_norms": torch.tensor(
                audio_norms, device=self._input_device,
            ),
            "video_norms": torch.tensor(
                video_norms, device=self._input_device,
            ),
        }

        return result, out, rope_deltas

    def _find_collapse_layer(
        self,
        layer_consistency: torch.Tensor,
        threshold: float = 0.3,
    ) -> Optional[int]:
        """检测跨模态一致性崩塌层

        寻找一致性急剧下降的层 — 即相邻层一致性差值最大的位置。
        如果最大降幅超过 threshold，返回崩塌层索引；否则返回 None。

        这个崩塌层表示模型开始"混淆"音频和视频信息的位置，
        是音频/情感幻觉产生的关键层。
        """
        if len(layer_consistency) < 2:
            return None

        # 计算相邻层一致性差值（负值表示下降）
        diffs = layer_consistency[1:] - layer_consistency[:-1]
        min_diff = diffs.min().item()

        if min_diff < -threshold:
            # 崩塌层 = 差值最大下降处的后一层
            collapse_idx = diffs.argmin().item() + 1
            logger.info(
                "CMCD 检测到一致性崩塌: layer %d, "
                "drop=%.3f (%.3f → %.3f)",
                collapse_idx, -min_diff,
                layer_consistency[collapse_idx - 1].item(),
                layer_consistency[collapse_idx].item(),
            )
            return collapse_idx

        # 备选：检查绝对一致性是否在后半段持续低于阈值
        n = len(layer_consistency)
        late_layers = layer_consistency[n // 2:]
        if late_layers.mean().item() < 0.5:
            # 后半段整体一致性低，取最低点
            collapse_idx = (
                n // 2 + late_layers.argmin().item()
            )
            logger.info(
                "CMCD 后半段一致性低: layer %d, "
                "consistency=%.3f",
                collapse_idx,
                layer_consistency[collapse_idx].item(),
            )
            return collapse_idx

        return None

    def _compute_cmcd_weights(
        self,
        layer_consistency: torch.Tensor,
        audio_norms: torch.Tensor,
        video_norms: torch.Tensor,
        collapse_layer: Optional[int],
        base_alpha: float = 0.5,
        base_beta: float = 0.5,
    ) -> Tuple[float, float]:
        """基于层级一致性动态计算 α/β

        核心逻辑（聚焦音频/情感幻觉）：
        1. 崩塌层存在 → 说明音视频信息在深层发生混淆
           - 音频 norm 在崩塌层后异常增大 → 音频主导幻觉 → 增大 α
           - 视频 norm 在崩塌层后异常增大 → 视频主导幻觉 → 增大 β
        2. 整体一致性低 → 两个模态都不可信，同时增大 α 和 β
        3. 一致性高 → 模态融合正常，使用较小权重做轻微对比
        """
        n = len(layer_consistency)
        overall_consistency = layer_consistency.mean().item()

        if collapse_layer is not None and collapse_layer < n:
            # 崩塌前后的 norm 变化
            pre = max(0, collapse_layer - 3)
            post = min(n, collapse_layer + 3)

            audio_norm_pre = audio_norms[pre:collapse_layer].mean().item()
            audio_norm_post = audio_norms[collapse_layer:post].mean().item()
            video_norm_pre = video_norms[pre:collapse_layer].mean().item()
            video_norm_post = video_norms[collapse_layer:post].mean().item()

            # norm 增长比 — 增长越大说明该模态在崩塌层后"膨胀"
            audio_growth = (
                (audio_norm_post / max(audio_norm_pre, 1e-6)) - 1.0
            )
            video_growth = (
                (video_norm_post / max(video_norm_pre, 1e-6)) - 1.0
            )

            # 音频膨胀 → 增大 α（抑制音频幻觉）
            alpha = base_alpha * (1.0 + max(audio_growth, 0.0) * 2.0)
            # 视频膨胀 → 增大 β（抑制视频幻觉）
            beta = base_beta * (1.0 + max(video_growth, 0.0) * 2.0)

            # 崩塌层越早，幻觉越严重 — 但根据哪个模态膨胀更多来决定权重
            early_factor = 1.0 + (1.0 - collapse_layer / n) * 0.5
            if audio_growth > video_growth:
                # 音频幻觉更严重
                alpha *= early_factor
                beta *= (early_factor * 0.5)  # 视频权重略微增加
            else:
                # 视频幻觉更严重
                beta *= early_factor
                alpha *= (early_factor * 0.5)  # 音频权重略微增加

        elif overall_consistency < 0.6:
            # 无明显崩塌但整体一致性低 — 根据整体 norm 差异调整
            audio_norm_mean = audio_norms.mean().item()
            video_norm_mean = video_norms.mean().item()

            if audio_norm_mean > video_norm_mean:
                # 音频 norm 更大，可能音频幻觉更多
                boost_audio = (0.6 - overall_consistency) * 2.5
                boost_video = (0.6 - overall_consistency) * 1.5
                alpha = base_alpha * (1.0 + boost_audio)
                beta = base_beta * (1.0 + boost_video)
            else:
                # 视频 norm 更大，可能视频幻觉更多
                boost_audio = (0.6 - overall_consistency) * 1.5
                boost_video = (0.6 - overall_consistency) * 2.5
                alpha = base_alpha * (1.0 + boost_audio)
                beta = base_beta * (1.0 + boost_video)

        else:
            # 一致性良好，轻微对比即可
            alpha = base_alpha * 0.5
            beta = base_beta * 0.5

        # Clamp
        alpha = min(max(alpha, 0.1), 1.5)
        beta = min(max(beta, 0.1), 1.5)

        logger.info(
            "CMCD 动态权重: α=%.3f β=%.3f "
            "(overall_consistency=%.3f, collapse=%s)",
            alpha, beta, overall_consistency,
            collapse_layer if collapse_layer is not None else "none",
        )
        return alpha, beta

    # ──────────────────────────────────────────────────────────────────────
    # 内部方法
    # ──────────────────────────────────────────────────────────────────────

    def _compose_question_text(
        self,
        question: str,
        emotion_constraint: Optional[str] = None,
        safety_contexts: Optional[List[str]] = None,
    ) -> str:
        pieces: List[str] = []
        if safety_contexts:
            pieces.extend(ctx for ctx in safety_contexts if ctx)
        if emotion_constraint:
            pieces.append(
                f"[The dominant emotion in this content is {emotion_constraint}. "
                'Consider this when answering.]'
            )
        pieces.append(question)
        return "\n".join(pieces)

    @staticmethod
    def _is_yes_no_question(
        question: str,
        max_new_tokens: Optional[int] = None,
    ) -> bool:
        q = (question or '').strip().lower()
        if _YN_INSTRUCTION_RE.search(q):
            return True
        if max_new_tokens is not None and max_new_tokens <= 10:
            return True
        return q.startswith(_YN_PREFIXES)

    @staticmethod
    def _strip_yes_no_suffix(question: str) -> str:
        return re.sub(
            r'\s*answer with only yes or no\.?\s*$',
            '',
            question or '',
            flags=re.IGNORECASE,
        ).strip()

    def _build_dsav_draft_question(self, question: str) -> str:
        base_question = self._strip_yes_no_suffix(question)
        return (
            f"{base_question}\n"
            'Briefly describe the concrete audiovisual evidence relevant to this question in one short sentence. '
            'Mention only directly visible or audible evidence. Do not answer with Yes or No.'
        )

    @staticmethod
    def _build_yes_no_evidence_context(validated_text: str) -> str:
        validated_text = (validated_text or '').strip()
        if not validated_text:
            return ''
        return (
            '[DSAV evidence summary] '
            f'Verified audiovisual evidence: {validated_text}. '
            'Base the Yes or No answer only on this evidence.'
        )

    def _score_yes_no_inputs(
        self,
        inputs: Dict[str, torch.Tensor],
    ) -> Dict[str, float]:
        tokenizer = self._processor.tokenizer
        yes_id = tokenizer.encode('Yes', add_special_tokens=False)[0]
        no_id = tokenizer.encode('No', add_special_tokens=False)[0]
        thinker = self._model.thinker
        with self.temporary_release_cuda_reserve("score_yes_no_inputs"):
            with torch.inference_mode():
                out = thinker(**inputs, use_cache=False)
                logits = out.logits[:, -1, :].float()
        yn_logits = logits[0, [yes_id, no_id]]
        probs = torch.softmax(yn_logits, dim=0)
        p_yes = float(probs[0].item())
        p_no = float(probs[1].item())
        answer = 'Yes' if p_yes >= p_no else 'No'
        margin = abs(p_yes - p_no)
        del out, logits, yn_logits, probs, inputs
        thinker.rope_deltas = None
        _safe_empty_cache()
        return {
            'answer': answer,
            'p_yes': p_yes,
            'p_no': p_no,
            'margin': margin,
        }

    def _score_single_choice_inputs(
        self,
        inputs: Dict[str, torch.Tensor],
        *,
        answer_space_spec: AnswerSpaceSpec,
    ) -> Dict[str, object]:
        tokenizer = self._processor.tokenizer
        candidate_options = []
        candidate_token_ids: List[int] = []
        for option in answer_space_spec.options:
            label = str(option.label or '').strip()
            if not label:
                continue
            token_ids = tokenizer.encode(label, add_special_tokens=False)
            if len(token_ids) != 1:
                prefixed = tokenizer.encode(f' {label}', add_special_tokens=False)
                if len(prefixed) != 1:
                    continue
                token_ids = prefixed
            candidate_options.append(option)
            candidate_token_ids.append(int(token_ids[0]))
        if len(candidate_token_ids) < 2:
            raise RuntimeError('single-choice branch scoring requires at least two single-token option labels')

        logits = self._prefill_last_logits(inputs, use_cache=False)[0].float()
        choice_logits = logits[candidate_token_ids]
        probs = torch.softmax(choice_logits, dim=0)
        top_index = int(probs.argmax().item())
        top_prob = float(probs[top_index].item())
        sorted_probs = torch.sort(probs, descending=True).values
        second_prob = float(sorted_probs[1].item()) if sorted_probs.numel() > 1 else 0.0
        selected = candidate_options[top_index]
        label = str(selected.label or '').strip()
        text = str(selected.text or '').strip()
        scores = {
            str(option.label or '').strip(): float(probs[idx].item())
            for idx, option in enumerate(candidate_options)
            if str(option.label or '').strip()
        }
        del logits, choice_logits, probs, sorted_probs
        _safe_empty_cache()
        return {
            'answer': label,
            'answer_label': label,
            'answer_text': text,
            'choice_probs': scores,
            'margin': max(0.0, top_prob - second_prob),
            'top1_prob': top_prob,
            'top2_prob': second_prob,
        }

    def _prefill_last_logits(
        self,
        inputs: Dict[str, torch.Tensor],
        *,
        use_cache: bool = False,
    ) -> torch.Tensor:
        thinker = self._model.thinker
        with self.temporary_release_cuda_reserve("prefill_last_logits"):
            with torch.inference_mode():
                out = thinker(**inputs, use_cache=use_cache)
                logits = out.logits[:, -1, :].clone()
        del out, inputs
        thinker.rope_deltas = None
        _safe_empty_cache()
        return logits

    def score_yes_no_media_branches(
        self,
        question: str,
        *,
        video_path: Optional[str] = None,
        audio_path: Optional[str] = None,
        emotion_constraint: Optional[str] = None,
        include_no_visual: bool = True,
        video_budget: Optional[Dict[str, float]] = None,
    ) -> Dict[str, Dict[str, float]]:
        audio_array = self._resolve_audio_array(
            video_path=video_path,
            audio_path=audio_path,
            mask_audio=False,
        )
        branch_specs: List[Tuple[str, Optional[np.ndarray], bool]] = [
            ('full', audio_array, False),
            ('no_audio', None, False),
        ]
        if include_no_visual:
            branch_specs.append(('no_visual', audio_array, True))

        scores: Dict[str, Dict[str, float]] = {}
        for branch_name, branch_audio, mask_visual in branch_specs:
            inputs: Optional[Dict[str, torch.Tensor]] = None
            try:
                inputs = self._prepare_inputs(
                    video_path,
                    question,
                    audio_array=branch_audio,
                    mask_visual=mask_visual,
                    emotion_constraint=emotion_constraint,
                    video_budget=video_budget,
                )
                scores[branch_name] = self._score_yes_no_inputs(inputs)
            finally:
                if inputs is not None:
                    del inputs
                _safe_empty_cache()
        return scores

    def score_choice_media_branches(
        self,
        question: str,
        *,
        answer_space_spec: AnswerSpaceSpec,
        video_path: Optional[str] = None,
        audio_path: Optional[str] = None,
        emotion_constraint: Optional[str] = None,
        include_no_visual: bool = True,
        video_budget: Optional[Dict[str, float]] = None,
    ) -> Dict[str, Dict[str, object]]:
        if answer_space_spec.kind != 'single_choice':
            raise NotImplementedError(f'choice branch scoring only supports single_choice, got {answer_space_spec.kind}')
        audio_array = self._resolve_audio_array(
            video_path=video_path,
            audio_path=audio_path,
            mask_audio=False,
        )
        branch_specs: List[Tuple[str, Optional[np.ndarray], bool]] = [
            ('full', audio_array, False),
            ('no_audio', None, False),
        ]
        if include_no_visual:
            branch_specs.append(('no_visual', audio_array, True))

        scores: Dict[str, Dict[str, object]] = {}
        for branch_name, branch_audio, mask_visual in branch_specs:
            inputs: Optional[Dict[str, torch.Tensor]] = None
            try:
                inputs = self._prepare_inputs(
                    video_path,
                    question,
                    audio_array=branch_audio,
                    mask_visual=mask_visual,
                    emotion_constraint=emotion_constraint,
                    video_budget=video_budget,
                )
                scores[branch_name] = self._score_single_choice_inputs(
                    inputs,
                    answer_space_spec=answer_space_spec,
                )
            finally:
                if inputs is not None:
                    del inputs
                _safe_empty_cache()
        return scores

    def score_answer_space_media_branches(
        self,
        question: str,
        *,
        answer_space_spec: AnswerSpaceSpec,
        video_path: Optional[str] = None,
        audio_path: Optional[str] = None,
        emotion_constraint: Optional[str] = None,
        include_no_visual: bool = True,
        video_budget: Optional[Dict[str, float]] = None,
    ) -> Dict[str, Dict[str, object]]:
        with self.temporary_release_cuda_reserve("score_answer_space_media_branches"):
            formatted_question = format_question_for_answer_space(question, answer_space_spec)
            if answer_space_spec.kind == 'yes_no':
                return self.score_yes_no_media_branches(
                    formatted_question,
                    video_path=video_path,
                    audio_path=audio_path,
                    emotion_constraint=emotion_constraint,
                    include_no_visual=include_no_visual,
                    video_budget=video_budget,
                )
            if answer_space_spec.kind == 'single_choice':
                return self.score_choice_media_branches(
                    formatted_question,
                    answer_space_spec=answer_space_spec,
                    video_path=video_path,
                    audio_path=audio_path,
                    emotion_constraint=emotion_constraint,
                    include_no_visual=include_no_visual,
                    video_budget=video_budget,
                )
            raise NotImplementedError(
                f'answer-space branch scoring is not implemented for kind={answer_space_spec.kind}'
            )

    def score_yes_no_branches(
        self,
        video_path: str,
        question: str,
        *,
        emotion_constraint: Optional[str] = None,
        use_video_branch: bool = True,
    ) -> Dict[str, Dict[str, float]]:
        return self.score_yes_no_media_branches(
            question,
            video_path=video_path,
            emotion_constraint=emotion_constraint,
            include_no_visual=use_video_branch,
        )

    def _build_messages(
        self,
        video_path: Optional[str],
        question: str,
        *,
        audio_array: Optional[np.ndarray] = None,
        mask_visual: bool = False,
        emotion_constraint: Optional[str] = None,
        safety_contexts: Optional[List[str]] = None,
    ) -> List[Dict[str, List[Dict[str, str]]]]:
        user_content = []
        if not mask_visual and self._path_exists(video_path):
            user_content.append({'type': 'video', 'video': video_path})
        if audio_array is not None:
            user_content.append({'type': 'audio', 'audio': 'placeholder'})
        user_content.append(
            {
                'type': 'text',
                'text': self._compose_question_text(
                    question,
                    emotion_constraint=emotion_constraint,
                    safety_contexts=safety_contexts,
                ),
            }
        )
        return [
            {
                'role': 'system',
                'content': [{'type': 'text', 'text': _DEFAULT_SYSTEM_PROMPT}],
            },
            {'role': 'user', 'content': user_content},
        ]

    def _messages_to_inputs(
        self,
        messages: list,
        audio_array: Optional[np.ndarray] = None,
        video_budget: Optional[Dict[str, float]] = None,
    ) -> Dict[str, torch.Tensor]:
        text_prompt = self._processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        video_paths = []
        for msg in messages:
            content = msg.get('content', [])
            if isinstance(content, str):
                continue
            for item in content:
                if item.get('type') == 'video':
                    video_paths.append(item['video'])

        video_budget = video_budget or {
            'fps': 4.0,
            'max_pixels': 602112,
            'min_pixels': 100352,
            'max_frames': 32,
        }
        if video_paths and hasattr(self._processor, 'video_processor'):
            self._processor.video_processor.do_sample_frames = False
            self._processor.video_processor.max_pixels = int(video_budget.get('max_pixels', 602112))
            self._processor.video_processor.min_pixels = int(video_budget.get('min_pixels', 100352))
        videos, video_metadata = _load_video_inputs(video_paths, video_budget) if video_paths else ([], [])

        proc_kwargs = {
            'text': text_prompt,
            'return_tensors': 'pt',
            'padding': True,
        }
        if videos:
            proc_kwargs['videos'] = videos
        if video_metadata and any(meta is not None for meta in video_metadata):
            proc_kwargs['video_metadata'] = [[meta] for meta in video_metadata]
        if audio_array is not None:
            proc_kwargs['audio'] = [audio_array]

        inputs = self._processor(**proc_kwargs)
        inputs = inputs.to(self._input_device)
        return dict(inputs)

    def _prepare_inputs(
        self,
        video_path: Optional[str],
        question: str,
        audio_array: Optional[np.ndarray] = None,
        mask_visual: bool = False,
        emotion_constraint: Optional[str] = None,
        safety_contexts: Optional[List[str]] = None,
        video_budget: Optional[Dict[str, float]] = None,
    ) -> Dict[str, torch.Tensor]:
        """准备模型输入（不调用 generate，返回 processor 输出）。"""
        messages = self._build_messages(
            video_path,
            question,
            audio_array=audio_array,
            mask_visual=mask_visual,
            emotion_constraint=emotion_constraint,
            safety_contexts=safety_contexts,
        )
        return self._messages_to_inputs(
            messages,
            audio_array=audio_array,
            video_budget=video_budget,
        )

    def _prepare_probe_inputs(
        self,
        video_path: Optional[str],
        question: str,
        audio_array: Optional[np.ndarray] = None,
        mask_visual: bool = False,
        emotion_constraint: Optional[str] = None,
        safety_contexts: Optional[List[str]] = None,
    ) -> Dict[str, torch.Tensor]:
        return self._prepare_inputs(
            video_path,
            question,
            audio_array=audio_array,
            mask_visual=mask_visual,
            emotion_constraint=emotion_constraint,
            safety_contexts=safety_contexts,
            video_budget={
                'fps': 1.0,
                'max_pixels': 200704,
                'min_pixels': 50176,
                'max_frames': 8,
            },
        )

    def _prepare_state_from_inputs(
        self,
        inputs: Dict[str, torch.Tensor],
    ) -> Dict[str, object]:
        thinker = self._model.thinker
        with torch.inference_mode():
            out = thinker(**inputs, use_cache=True)
            rope_deltas = thinker.rope_deltas.clone() if thinker.rope_deltas is not None else None
        return self._build_state_from_prefill(inputs, out, rope_deltas)

    def _prepare_probe_streaming_state(
        self,
        video_path: Optional[str],
        question: str,
        *,
        audio_array: Optional[np.ndarray] = None,
        mask_visual: bool = False,
        emotion_constraint: Optional[str] = None,
        safety_contexts: Optional[List[str]] = None,
    ) -> Dict[str, object]:
        inputs = self._prepare_probe_inputs(
            video_path,
            question,
            audio_array=audio_array,
            mask_visual=mask_visual,
            emotion_constraint=emotion_constraint,
            safety_contexts=safety_contexts,
        )
        return self._prepare_state_from_inputs(inputs)

    def _score_text_continuation_from_state(
        self,
        state: Dict[str, object],
        continuation: str,
    ) -> Dict[str, object]:
        token_ids = self._processor.tokenizer.encode(
            continuation or '',
            add_special_tokens=False,
        )
        if not token_ids:
            return {
                'token_count': 0,
                'token_ids': [],
                'total_logprob': 0.0,
                'mean_logprob': 0.0,
            }

        total_logprob = 0.0
        token_logprobs: List[float] = []
        for token_id in token_ids:
            logits = state['logits'].float()
            log_probs = F.log_softmax(logits, dim=-1)
            logprob = float(log_probs[0, token_id].item())
            token_logprobs.append(logprob)
            total_logprob += logprob
            next_token = torch.tensor([token_id], device=self._input_device, dtype=torch.long)
            state = self._advance_generation_state(state, next_token)

        token_count = len(token_ids)
        return {
            'token_count': token_count,
            'token_ids': token_ids,
            'token_logprobs': token_logprobs,
            'total_logprob': float(total_logprob),
            'mean_logprob': float(total_logprob / max(1, token_count)),
        }

    def score_text_continuation_branches(
        self,
        video_path: str,
        prompt: str,
        continuation: str,
        *,
        emotion_constraint: Optional[str] = None,
        safety_contexts: Optional[List[str]] = None,
        include_no_visual: bool = True,
        use_probe_budget: bool = True,
    ) -> Dict[str, Dict[str, object]]:
        return self.score_text_continuation_media_branches(
            prompt,
            continuation,
            video_path=video_path,
            emotion_constraint=emotion_constraint,
            safety_contexts=safety_contexts,
            include_no_visual=include_no_visual,
            use_probe_budget=use_probe_budget,
        )

    def score_text_continuation_media_branches(
        self,
        prompt: str,
        continuation: str,
        *,
        video_path: Optional[str] = None,
        audio_path: Optional[str] = None,
        emotion_constraint: Optional[str] = None,
        safety_contexts: Optional[List[str]] = None,
        include_no_visual: bool = True,
        use_probe_budget: bool = True,
    ) -> Dict[str, Dict[str, object]]:
        audio_array = self._resolve_audio_array(
            video_path=video_path,
            audio_path=audio_path,
            mask_audio=False,
        )
        branches = {
            'full': {
                'audio_array': audio_array,
                'mask_visual': False,
            },
            'no_audio': {
                'audio_array': None,
                'mask_visual': False,
            },
        }
        if include_no_visual:
            branches['no_visual'] = {
                'audio_array': audio_array,
                'mask_visual': True,
            }

        results: Dict[str, Dict[str, object]] = {}
        for branch_name, branch_kwargs in branches.items():
            state = None
            try:
                if use_probe_budget:
                    state = self._prepare_probe_streaming_state(
                        video_path,
                        prompt,
                        audio_array=branch_kwargs['audio_array'],
                        mask_visual=branch_kwargs['mask_visual'],
                        emotion_constraint=emotion_constraint,
                        safety_contexts=safety_contexts,
                    )
                else:
                    state = self._prepare_streaming_state(
                        video_path,
                        prompt,
                        audio_array=branch_kwargs['audio_array'],
                        mask_visual=branch_kwargs['mask_visual'],
                        emotion_constraint=emotion_constraint,
                        safety_contexts=safety_contexts,
                    )
                results[branch_name] = self._score_text_continuation_from_state(
                    state,
                    continuation,
                )
            finally:
                self._release_generation_state(state)
        return results

    def _generate(
        self,
        messages: list,
        max_new_tokens: int,
        audio_array: Optional[np.ndarray] = None,
        logits_processor=None,
    ) -> str:
        """调用模型生成文本答案（标准 generate）。

        Parameters
        ----------
        logits_processor : LogitsProcessorList, optional
            自定义 logits 处理器（用于冲突感知抑制）
        """
        with self.temporary_release_cuda_reserve("generate"):
            inputs = self._messages_to_inputs(messages, audio_array=audio_array)

            gen_kwargs = {
                'return_audio': False,
                'thinker_max_new_tokens': max_new_tokens,
            }

            # 如果提供了 logits processor，添加到生成参数中
            if logits_processor is not None:
                from transformers import LogitsProcessorList
                if not isinstance(logits_processor, LogitsProcessorList):
                    logits_processor = LogitsProcessorList([logits_processor])
                gen_kwargs['logits_processor'] = logits_processor

            with torch.inference_mode():
                output_ids = self._model.generate(
                    **inputs,
                    **gen_kwargs,
                )

        input_len = inputs['input_ids'].shape[1]
        generated = output_ids[0, input_len:]
        answer = self._processor.tokenizer.decode(
            generated,
            skip_special_tokens=True,
        ).strip()

        del inputs, output_ids
        _safe_empty_cache()
        return answer

    def _prepare_streaming_state(
        self,
        video_path: Optional[str],
        question: str,
        *,
        audio_array: Optional[np.ndarray] = None,
        mask_visual: bool = False,
        emotion_constraint: Optional[str] = None,
        safety_contexts: Optional[List[str]] = None,
    ) -> Dict[str, object]:
        inputs = self._prepare_inputs(
            video_path,
            question,
            audio_array=audio_array,
            mask_visual=mask_visual,
            emotion_constraint=emotion_constraint,
            safety_contexts=safety_contexts,
        )
        thinker = self._model.thinker
        with torch.inference_mode():
            out = thinker(**inputs, use_cache=True)
            rope_deltas = thinker.rope_deltas.clone() if thinker.rope_deltas is not None else None
            logits = out.logits[:, -1, :].clone()
            cache = out.past_key_values

        return {
            'inputs': inputs,
            'cache': cache,
            'logits': logits,
            'position': inputs['input_ids'].shape[1],
            'rope_deltas': rope_deltas,
        }

    def _advance_generation_state(
        self,
        state: Dict[str, object],
        next_token: torch.Tensor,
    ) -> Dict[str, object]:
        thinker = self._model.thinker
        thinker.rope_deltas = state['rope_deltas']
        cache_position = torch.tensor([state['position']], device=self._input_device)
        next_input = next_token.unsqueeze(0)
        with torch.inference_mode():
            out = thinker(
                input_ids=next_input,
                past_key_values=state['cache'],
                use_cache=True,
                cache_position=cache_position,
            )
        state['position'] += 1
        state['cache'] = out.past_key_values
        state['logits'] = out.logits[:, -1, :].clone()
        return state

    @staticmethod
    def _mask_blocked_tokens(
        logits: torch.Tensor,
        blocked_tokens: Dict[int, int],
    ) -> torch.Tensor:
        if not blocked_tokens:
            return logits
        masked = logits.clone()
        for token_id, ttl in blocked_tokens.items():
            if ttl > 0 and 0 <= token_id < masked.shape[-1]:
                masked[:, token_id] = float('-inf')
        return masked

    def _release_generation_state(self, state: Optional[Dict[str, object]]) -> None:
        if not state:
            return
        thinker = self._model.thinker
        thinker.rope_deltas = None
        state.clear()
        if torch.cuda.is_available():
            _safe_empty_cache()

    @staticmethod
    def _load_audio(video_path: str) -> Optional[np.ndarray]:
        """从视频提取音频为 16kHz mono numpy 数组。"""
        try:
            import librosa
            audio, _ = librosa.load(video_path, sr=16000, mono=True)
            if len(audio) > 0:
                return audio
            return None
        except Exception:
            pass

        try:
            tmp = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
            tmp.close()
            cmd = [
                'ffmpeg', '-y', '-i', video_path,
                '-vn', '-acodec', 'pcm_s16le',
                '-ar', '16000', '-ac', '1',
                tmp.name, '-loglevel', 'error',
            ]
            ret = subprocess.run(cmd, capture_output=True, timeout=30)
            if ret.returncode == 0 and Path(tmp.name).stat().st_size > 4096:
                import soundfile as sf
                audio, _ = sf.read(tmp.name, dtype='float32')
                Path(tmp.name).unlink(missing_ok=True)
                return audio
            Path(tmp.name).unlink(missing_ok=True)
            return None
        except Exception as exc:
            logger.debug('音频提取失败: %s', exc)
            return None

    def dynamic_speculative_decoding(
        self,
        video_path: str,
        question: str,
        async_validator,
        max_new_tokens: int = None,
        emotion_constraint: Optional[str] = None,
        speculative_config=None,
        baseline_answer: Optional[str] = None,
        safety_contexts: Optional[List[str]] = None,
    ) -> Dict[str, object]:
        """DSAV: Yes/No 任务走隐藏证据草稿，描述任务走直接文本扫描。"""
        if max_new_tokens is None:
            max_new_tokens = self._max_new_tokens

        audio_array = self._load_audio(video_path) if Path(video_path).exists() else None
        tokenizer = self._processor.tokenizer
        is_yes_no = self._is_yes_no_question(question, max_new_tokens)
        use_hidden_yn_draft = bool(
            getattr(speculative_config, 'enable_hidden_yn_draft', True) if speculative_config else True
        ) and is_yes_no
        max_contexts = max(
            1,
            getattr(speculative_config, 'max_validation_contexts', 3) if speculative_config else 3,
        )
        max_passes = max_contexts + 1
        draft_max_new_tokens = (
            max(
                max_new_tokens,
                getattr(speculative_config, 'yn_draft_max_new_tokens', 48) if speculative_config else 48,
            )
            if use_hidden_yn_draft else max_new_tokens
        )

        safety_contexts = [
            context
            for context in dict.fromkeys(context for context in (safety_contexts or []) if context)
        ]
        safety_contexts = safety_contexts[-max_contexts:]
        seen_event_keys = set()
        final_answer = ''
        last_accepted_answer = ''
        last_validation_draft = ''
        completed = False
        metadata = {
            'trigger_count': 0,
            'rollback_count': 0,
            'validation_latency_s': 0.0,
            'rebuild_count': 0,
            'events': [],
            'validation_mode': 'hidden_yes_no_draft' if use_hidden_yn_draft else 'direct_answer_scan',
            'preserved_baseline_without_intervention': False,
        }

        for attempt in range(max_passes):
            prompt_question = (
                self._build_dsav_draft_question(question)
                if use_hidden_yn_draft else question
            )
            messages = self._build_messages(
                video_path,
                prompt_question,
                audio_array=audio_array,
                emotion_constraint=emotion_constraint,
                safety_contexts=safety_contexts,
            )
            candidate_answer = self._generate(
                messages,
                draft_max_new_tokens,
                audio_array=audio_array,
            ).strip()
            if candidate_answer:
                final_answer = candidate_answer
            if use_hidden_yn_draft:
                last_validation_draft = candidate_answer

            scan_summary = async_validator.scan_text_sync(
                candidate_answer,
                video_path,
                audio_array=audio_array,
            )
            metadata['trigger_count'] += int(scan_summary.get('trigger_count', 0))
            metadata['rollback_count'] += int(scan_summary.get('rollback_count', 0))
            metadata['validation_latency_s'] += float(
                scan_summary.get('validation_latency_s', 0.0)
            )

            for event in scan_summary.get('events', []):
                event_key = (
                    event.get('term'),
                    event.get('canonical_term'),
                    event.get('reason'),
                )
                if event_key not in seen_event_keys:
                    seen_event_keys.add(event_key)
                    metadata['events'].append(event)

            accepted_text = (scan_summary.get('accepted_text') or '').strip()
            if accepted_text:
                last_accepted_answer = accepted_text

            rollback_count = int(scan_summary.get('rollback_count', 0))
            new_contexts: List[str] = []
            for context in scan_summary.get('safe_contexts', []):
                if context and context not in safety_contexts:
                    new_contexts.append(context)

            if rollback_count > 0 and new_contexts:
                safety_contexts.extend(new_contexts)
                safety_contexts = safety_contexts[-max_contexts:]
                metadata['rebuild_count'] += 1
                logger.info(
                    'DSAV 第 %d 轮发现 %d 个可疑片段，注入 %d 条安全上下文后重生成',
                    attempt + 1,
                    rollback_count,
                    len(safety_contexts),
                )
                continue

            if use_hidden_yn_draft:
                had_intervention = bool(
                    metadata.get('rollback_count', 0)
                    or metadata.get('rebuild_count', 0)
                    or safety_contexts
                )
                if not had_intervention and baseline_answer:
                    final_answer = baseline_answer.strip() or final_answer
                    metadata['preserved_baseline_without_intervention'] = True
                    completed = True
                    break
                final_contexts = list(safety_contexts)
                evidence_context = self._build_yes_no_evidence_context(
                    last_accepted_answer or candidate_answer
                )
                if evidence_context and evidence_context not in final_contexts:
                    final_contexts.append(evidence_context)
                final_messages = self._build_messages(
                    video_path,
                    question,
                    audio_array=audio_array,
                    emotion_constraint=emotion_constraint,
                    safety_contexts=final_contexts,
                )
                regenerated_answer = self._generate(
                    final_messages,
                    max_new_tokens,
                    audio_array=audio_array,
                ).strip()
                final_answer = regenerated_answer or final_answer
            else:
                final_answer = last_accepted_answer or candidate_answer

            completed = True
            break

        if not completed and use_hidden_yn_draft:
            had_intervention = bool(
                metadata.get('rollback_count', 0)
                or metadata.get('rebuild_count', 0)
                or safety_contexts
            )
            if not had_intervention and baseline_answer:
                final_answer = baseline_answer.strip() or final_answer
                metadata['preserved_baseline_without_intervention'] = True
            else:
                final_contexts = list(safety_contexts)
                evidence_context = self._build_yes_no_evidence_context(last_accepted_answer)
                if evidence_context and evidence_context not in final_contexts:
                    final_contexts.append(evidence_context)
                final_messages = self._build_messages(
                    video_path,
                    question,
                    audio_array=audio_array,
                    emotion_constraint=emotion_constraint,
                    safety_contexts=final_contexts,
                )
                regenerated_answer = self._generate(
                    final_messages,
                    max_new_tokens,
                    audio_array=audio_array,
                ).strip()
                final_answer = regenerated_answer or final_answer

        final_answer = final_answer or last_accepted_answer
        metadata['generated_tokens'] = len(
            tokenizer.encode(final_answer, add_special_tokens=False)
        ) if final_answer else 0
        metadata['safe_contexts'] = safety_contexts
        metadata['last_pass_accepted_text'] = last_accepted_answer
        if use_hidden_yn_draft:
            metadata['validation_draft'] = last_validation_draft
        return {
            'text': final_answer,
            'metadata': metadata,
        }

    def _rollback_kv_cache(
        self,
        video_path: str,
        question: str,
        *,
        audio_array: Optional[np.ndarray] = None,
        retained_ids: Optional[List[int]] = None,
        safety_contexts: Optional[List[str]] = None,
        emotion_constraint: Optional[str] = None,
    ) -> Dict[str, object]:
        """通过重新 prefill 重建到安全前缀，实现可控回退。"""
        retained_ids = retained_ids or []
        logger.info(
            '回滚 KV Cache 状态: 保留 %d tokens, 注入 %d 条安全上下文',
            len(retained_ids),
            len(safety_contexts or []),
        )
        state = self._prepare_streaming_state(
            video_path,
            question,
            audio_array=audio_array,
            emotion_constraint=emotion_constraint,
            safety_contexts=safety_contexts,
        )
        for token_id in retained_ids:
            next_token = torch.tensor([token_id], device=self._input_device, dtype=torch.long)
            state = self._advance_generation_state(state, next_token)
        return state
