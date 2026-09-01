"""
Audio Event Detector (Training-free, AST-based)
================================================
使用 Audio Spectrogram Transformer 解析音频事件，并生成时间线证据。
"""
import logging
import re
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch

logger = logging.getLogger(__name__)


class AudioEventDetector:
    """基于 AST 的音频事件检测器。"""

    def __init__(
        self,
        model_name: str = 'MIT/ast-finetuned-audioset-10-10-0.4593',
        device: str = 'cpu',
        top_k: int = 5,
    ):
        self.model_name = model_name
        self.device = self._resolve_device(device)
        self.top_k = top_k
        self._model = None
        self._processor = None
        self._load_model()

    @staticmethod
    def _resolve_device(device: str) -> str:
        if device.startswith('cuda'):
            if not torch.cuda.is_available():
                logger.warning('未检测到 CUDA，AudioEventDetector 回退到 CPU')
                return 'cpu'
            if ':' in device:
                req_idx = int(device.split(':', 1)[1])
                n_gpus = torch.cuda.device_count()
                if req_idx >= n_gpus:
                    fallback = f"cuda:{n_gpus - 1}"
                    logger.warning('请求设备 %s 不存在，回退到 %s', device, fallback)
                    return fallback
        return device

    def _load_model(self) -> None:
        from transformers import ASTFeatureExtractor, ASTForAudioClassification

        logger.info('加载 AST 音频事件模型: %s → %s', self.model_name, self.device)
        self._processor = ASTFeatureExtractor.from_pretrained(self.model_name)
        self._model = ASTForAudioClassification.from_pretrained(self.model_name)
        self._model.to(self.device)
        self._model.eval()

    @staticmethod
    def _normalize_label(label: str) -> str:
        label = re.sub(r'[^a-z0-9]+', ' ', label.lower())
        return ' '.join(label.split())

    @staticmethod
    def _resample(audio: np.ndarray, sample_rate: int) -> np.ndarray:
        if sample_rate == 16000:
            return audio.astype(np.float32)
        import librosa
        return librosa.resample(audio.astype(np.float32), orig_sr=sample_rate, target_sr=16000)

    def detect(
        self,
        audio: np.ndarray,
        sample_rate: int = 16000,
        top_k: Optional[int] = None,
        threshold: float = 0.01,
    ) -> List[Tuple[str, float]]:
        """检测完整音频或片段中的高置信音频事件。"""
        if audio is None:
            return []
        audio = np.asarray(audio).reshape(-1)
        if audio.size == 0:
            return []

        if top_k is None:
            top_k = self.top_k
        audio = self._resample(audio, sample_rate)

        inputs = self._processor(audio, sampling_rate=16000, return_tensors='pt')
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.inference_mode():
            logits = self._model(**inputs).logits
            probs = torch.sigmoid(logits)[0]

        topk_vals, topk_ids = probs.topk(top_k)
        events: List[Tuple[str, float]] = []
        for val, idx in zip(topk_vals, topk_ids):
            score = float(val.item())
            if score < threshold:
                continue
            raw_label = self._model.config.id2label[idx.item()]
            events.append((self._normalize_label(raw_label), score))
        return events

    def detect_timeline(
        self,
        audio: np.ndarray,
        sample_rate: int = 16000,
        chunk_duration_s: float = 2.0,
        hop_duration_s: float = 1.0,
        top_k: Optional[int] = None,
        threshold: float = 0.15,
    ) -> List[Dict[str, Any]]:
        """基于滑动窗口建立 AST 时间线索引。"""
        if audio is None:
            return []
        audio = np.asarray(audio).reshape(-1)
        if audio.size == 0:
            return []

        audio = self._resample(audio, sample_rate)
        sample_rate = 16000
        chunk_samples = max(int(chunk_duration_s * sample_rate), 1)
        hop_samples = max(int(hop_duration_s * sample_rate), 1)
        timeline: List[Dict[str, Any]] = []

        total = len(audio)
        start = 0
        while start < total:
            end = min(start + chunk_samples, total)
            chunk = audio[start:end]
            if chunk.size < sample_rate // 4:
                break
            if chunk.size < chunk_samples:
                pad = np.zeros(chunk_samples - chunk.size, dtype=np.float32)
                chunk = np.concatenate([chunk.astype(np.float32), pad], axis=0)

            detections = self.detect(
                chunk,
                sample_rate=sample_rate,
                top_k=top_k,
                threshold=threshold,
            )
            for label, score in detections:
                timeline.append(
                    {
                        'label': label,
                        'score': score,
                        'start': start / sample_rate,
                        'end': end / sample_rate,
                    }
                )

            if end >= total:
                break
            start += hop_samples

        return timeline

    def summarize_timeline(
        self,
        timeline: Iterable[Dict[str, Any]],
        top_k: Optional[int] = None,
    ) -> Dict[str, float]:
        """聚合 AST 时间线，得到事件级别的最高置信度摘要。"""
        scores: Dict[str, float] = {}
        for item in timeline:
            label = self._normalize_label(str(item.get('label', '')))
            if not label:
                continue
            score = float(item.get('score', 0.0))
            scores[label] = max(score, scores.get(label, 0.0))

        ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
        if top_k is not None:
            ranked = ranked[:top_k]
        return dict(ranked)

    def listen_and_detect(
        self,
        audio_stream: Iterable[np.ndarray],
        sample_rate: int = 16000,
        threshold: float = 0.15,
    ) -> Iterable[Dict[str, Any]]:
        """流式监听音频片段，持续产出事件特征。"""
        for chunk in audio_stream:
            chunk = np.asarray(chunk).reshape(-1)
            events = self.detect(chunk, sample_rate=sample_rate, threshold=threshold)
            if not events:
                continue
            yield {
                'events': [{'label': label, 'score': score} for label, score in events],
                'energy': float(np.sqrt(np.mean(chunk ** 2))) if chunk.size else 0.0,
                'duration_s': float(chunk.size / sample_rate) if sample_rate else 0.0,
            }
