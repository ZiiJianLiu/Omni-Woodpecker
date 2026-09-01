"""
Audio Text Grounding Detector
=============================
Use a stronger audio-text model to localize query-relevant audio evidence
over sliding windows. This complements the legacy AST event detector by
directly scoring question-conditioned text prompts against raw audio chunks.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class AudioTextGroundingDetector:
    """Sliding-window audio-text grounding with CLAP-like models."""

    PROMPT_TEMPLATES: Tuple[str, ...] = (
        "{}",
        "sound of {}",
        "audio of {}",
        "{} sound",
        "{} making sound",
    )

    def __init__(
        self,
        model_name: str = "laion/clap-htsat-unfused",
        device: str = "cpu",
        top_k: int = 5,
        batch_size: int = 8,
    ):
        self.model_name = model_name
        self.device = self._resolve_device(device)
        self.top_k = max(int(top_k), 1)
        self.batch_size = max(int(batch_size), 1)
        self._model = None
        self._processor = None
        self._sample_rate = 48000
        self._load_model()

    @staticmethod
    def _resolve_device(device: str) -> str:
        if device.startswith("cuda"):
            if not torch.cuda.is_available():
                logger.warning("未检测到 CUDA，AudioTextGroundingDetector 回退到 CPU")
                return "cpu"
            if ":" in device:
                req_idx = int(device.split(":", 1)[1])
                n_gpus = torch.cuda.device_count()
                if req_idx >= n_gpus:
                    fallback = f"cuda:{n_gpus - 1}"
                    logger.warning("请求设备 %s 不存在，回退到 %s", device, fallback)
                    return fallback
        return device

    def _load_model(self) -> None:
        from transformers import AutoProcessor, ClapModel

        logger.info("加载音频 grounding 模型: %s -> %s", self.model_name, self.device)
        self._processor = AutoProcessor.from_pretrained(self.model_name)
        self._model = ClapModel.from_pretrained(self.model_name)
        self._model.to(self.device)
        self._model.eval()

        sample_rate = getattr(getattr(self._processor, "feature_extractor", None), "sampling_rate", None)
        if sample_rate is not None:
            self._sample_rate = int(sample_rate)
        logger.info("音频 grounding 模型加载完成: sample_rate=%d", self._sample_rate)

    @staticmethod
    def _normalize_label(label: str) -> str:
        label = re.sub(r"[^a-z0-9']+", " ", str(label or "").lower())
        return " ".join(label.split())

    def _resample(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        if int(sample_rate) == int(self._sample_rate):
            return np.asarray(audio, dtype=np.float32)
        import librosa

        return librosa.resample(
            np.asarray(audio, dtype=np.float32),
            orig_sr=int(sample_rate),
            target_sr=int(self._sample_rate),
        ).astype(np.float32)

    def _build_prompt_specs(self, aliases: Sequence[str]) -> List[Dict[str, str]]:
        specs: List[Dict[str, str]] = []
        seen = set()
        for alias in aliases:
            canonical = self._normalize_label(alias)
            if not canonical:
                continue
            for template in self.PROMPT_TEMPLATES:
                prompt = template.format(canonical).strip()
                key = (canonical, prompt)
                if key in seen:
                    continue
                seen.add(key)
                specs.append({"alias": canonical, "prompt": prompt})
        return specs

    @torch.inference_mode()
    def _encode_text(self, prompt_specs: Sequence[Dict[str, str]]) -> torch.Tensor:
        if not prompt_specs:
            return torch.empty((0, 0), dtype=torch.float32)
        prompts = [spec["prompt"] for spec in prompt_specs]
        inputs = self._processor(text=prompts, return_tensors="pt", padding=True)
        inputs = {
            key: value.to(self.device) if isinstance(value, torch.Tensor) else value
            for key, value in inputs.items()
        }
        text_features = self._model.get_text_features(**inputs)
        return F.normalize(text_features.detach().float().cpu(), dim=-1)

    @torch.inference_mode()
    def _encode_audio_batch(self, audio_batch: Sequence[np.ndarray]) -> torch.Tensor:
        if not audio_batch:
            return torch.empty((0, 0), dtype=torch.float32)
        inputs = self._processor(
            audios=[np.asarray(chunk, dtype=np.float32) for chunk in audio_batch],
            sampling_rate=int(self._sample_rate),
            return_tensors="pt",
            padding=True,
        )
        inputs = {
            key: value.to(self.device) if isinstance(value, torch.Tensor) else value
            for key, value in inputs.items()
        }
        audio_features = self._model.get_audio_features(**inputs)
        return F.normalize(audio_features.detach().float().cpu(), dim=-1)

    def detect_timeline(
        self,
        audio: np.ndarray,
        *,
        sample_rate: int = 16000,
        aliases: Sequence[str],
        chunk_duration_s: float = 2.0,
        hop_duration_s: float = 1.0,
        top_k: Optional[int] = None,
        threshold: float = 0.28,
    ) -> List[Dict[str, Any]]:
        """Return query-conditioned localization windows over raw audio."""
        if audio is None:
            return []
        audio = np.asarray(audio).reshape(-1)
        if audio.size == 0:
            return []

        prompt_specs = self._build_prompt_specs(aliases)
        if not prompt_specs:
            return []

        audio = self._resample(audio, int(sample_rate))
        sample_rate = int(self._sample_rate)
        text_features = self._encode_text(prompt_specs)
        if text_features.numel() <= 0:
            return []

        top_k = max(int(top_k or self.top_k), 1)
        chunk_samples = max(int(float(chunk_duration_s) * sample_rate), 1)
        hop_samples = max(int(float(hop_duration_s) * sample_rate), 1)

        windows: List[Tuple[int, int, np.ndarray]] = []
        total = int(audio.shape[0])
        start = 0
        while start < total:
            end = min(start + chunk_samples, total)
            chunk = np.asarray(audio[start:end], dtype=np.float32)
            if chunk.size < sample_rate // 4:
                break
            if chunk.size < chunk_samples:
                chunk = np.pad(chunk, (0, chunk_samples - chunk.size))
            windows.append((start, end, chunk))
            if end >= total:
                break
            start += hop_samples

        if not windows:
            return []

        timeline: List[Dict[str, Any]] = []
        for batch_start in range(0, len(windows), int(self.batch_size)):
            batch = windows[batch_start: batch_start + int(self.batch_size)]
            audio_features = self._encode_audio_batch([item[2] for item in batch])
            if audio_features.numel() <= 0:
                continue
            sims = torch.matmul(audio_features, text_features.T).clamp(-1.0, 1.0)
            probs = (sims + 1.0) / 2.0

            for row_idx, (start_samp, end_samp, _chunk) in enumerate(batch):
                alias_best: Dict[str, Tuple[float, str]] = {}
                row = probs[row_idx]
                for spec_idx, spec in enumerate(prompt_specs):
                    alias = spec["alias"]
                    prompt = spec["prompt"]
                    score = float(row[spec_idx].item())
                    prev = alias_best.get(alias)
                    if prev is None or score > prev[0]:
                        alias_best[alias] = (score, prompt)
                ranked = sorted(alias_best.items(), key=lambda kv: kv[1][0], reverse=True)
                for alias, (score, prompt) in ranked[: min(int(top_k), len(ranked))]:
                    if float(score) < float(threshold):
                        continue
                    timeline.append(
                        {
                            "label": alias,
                            "prompt": prompt,
                            "score": float(score),
                            "raw_similarity": float((score * 2.0) - 1.0),
                            "start": float(start_samp / sample_rate),
                            "end": float(end_samp / sample_rate),
                            "source": "audio_query_grounding",
                        }
                    )
        return timeline

    def summarize_timeline(
        self,
        timeline: Iterable[Dict[str, Any]],
        top_k: Optional[int] = None,
    ) -> Dict[str, float]:
        scores: Dict[str, float] = {}
        for item in timeline:
            label = self._normalize_label(str(item.get("label", "")))
            if not label:
                continue
            score = float(item.get("score", 0.0) or 0.0)
            scores[label] = max(score, scores.get(label, 0.0))
        ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
        if top_k is not None:
            ranked = ranked[: int(top_k)]
        return dict(ranked)
