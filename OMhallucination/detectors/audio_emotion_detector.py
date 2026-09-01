"""
Audio Emotion Detector (Training-free)
=======================================
使用预训练的 wav2vec2 模型检测音频情感
"""
import logging
from typing import Tuple, Optional
import numpy as np
import torch

logger = logging.getLogger(__name__)


class AudioEmotionDetector:
    """音频情感检测器（Training-free）"""

    def __init__(
        self,
        model_name: str = "ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition",
        device: str = "cuda",
    ):
        self.device = device
        self.model_name = model_name
        self._model = None
        self._processor = None
        self._load_model()

    def _load_model(self):
        """加载预训练模型"""
        try:
            from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2ForSequenceClassification

            logger.info(f"加载音频情感模型: {self.model_name}")
            self._processor = Wav2Vec2FeatureExtractor.from_pretrained(self.model_name)
            self._model = Wav2Vec2ForSequenceClassification.from_pretrained(self.model_name)
            self._model.to(self.device)
            self._model.eval()
            logger.info("音频情感模型加载完成")

        except Exception as e:
            logger.error(f"音频情感模型加载失败: {e}")
            raise

    def detect(
        self,
        audio: np.ndarray,
        sample_rate: int = 16000,
    ) -> Tuple[Optional[str], float]:
        """检测音频情感

        Parameters
        ----------
        audio : np.ndarray
            音频波形，shape (n_samples,)，单声道
        sample_rate : int
            采样率（模型要求 16kHz）

        Returns
        -------
        emotion : str
            检测到的情感标签
        confidence : float
            置信度 [0, 1]
        """
        if audio is None or len(audio) == 0:
            logger.warning("音频为空，返回 neutral")
            return "neutral", 0.0

        try:
            # 重采样到 16kHz（如果需要）
            if sample_rate != 16000:
                import librosa
                audio = librosa.resample(audio, orig_sr=sample_rate, target_sr=16000)

            # 预处理
            inputs = self._processor(
                audio,
                sampling_rate=16000,
                return_tensors="pt",
                padding=True,
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            # 推理
            with torch.no_grad():
                outputs = self._model(**inputs)
                logits = outputs.logits
                probs = torch.softmax(logits, dim=-1)
                pred_idx = probs.argmax(dim=-1).item()
                confidence = probs[0, pred_idx].item()

            # 获取标签
            emotion_label = self._model.config.id2label[pred_idx]

            # 标准化标签
            emotion_label = self._normalize_label(emotion_label)

            logger.debug(f"音频情感检测: {emotion_label} (conf={confidence:.3f})")

            return emotion_label, float(confidence)

        except Exception as e:
            logger.error(f"音频情感检测失败: {e}")
            return "neutral", 0.0

    def _normalize_label(self, label: str) -> str:
        """标准化情感标签"""
        from ..config import EMOTION_LABEL_MAPPING

        label_lower = label.lower()
        return EMOTION_LABEL_MAPPING.get(label_lower, label_lower)

    def classify_audio_type(
        self,
        audio: np.ndarray,
        sample_rate: int = 16000,
    ) -> str:
        """分类音频类型：speech/music/noise/silence

        Parameters
        ----------
        audio : np.ndarray
            音频波形
        sample_rate : int
            采样率

        Returns
        -------
        audio_type : str
            speech/music/noise/silence
        """
        try:
            import librosa

            # 1. 检查是否静音
            rms = np.sqrt(np.mean(audio ** 2))
            if rms < 0.01:
                return "silence"

            # 2. 计算 spectral flatness（频谱平坦度）
            flatness = librosa.feature.spectral_flatness(y=audio)
            flatness_mean = float(flatness.mean())

            # 3. 计算 zero crossing rate
            zcr = librosa.feature.zero_crossing_rate(audio)
            zcr_mean = float(zcr.mean())

            # 4. 简单规则判断
            if flatness_mean > 0.3:
                # 高频谱平坦度 → 噪声
                return "noise"
            elif zcr_mean < 0.1 and flatness_mean < 0.15:
                # 低 ZCR + 低平坦度 → 音乐
                return "music"
            else:
                # 默认为语音
                return "speech"

        except Exception as e:
            logger.error(f"音频类型分类失败: {e}")
            return "speech"  # 默认
