"""
Visual Emotion Detector (Training-free)
========================================
使用预训练的 vit-face-expression 模型检测视频中的情感
"""
import logging
from typing import Tuple, Optional, List
import numpy as np
import torch
from PIL import Image

logger = logging.getLogger(__name__)


class VisualEmotionDetector:
    """视觉情感检测器（Training-free）"""

    def __init__(self, model_name: str = "trpakov/vit-face-expression", device: str = "cuda"):
        self.device = device
        self.model_name = model_name
        self._model = None
        self._processor = None
        self._load_model()

    def _load_model(self):
        """加载预训练模型"""
        try:
            from transformers import ViTImageProcessor, ViTForImageClassification

            logger.info(f"加载视觉情感模型: {self.model_name}")
            self._processor = ViTImageProcessor.from_pretrained(self.model_name)
            self._model = ViTForImageClassification.from_pretrained(self.model_name)
            self._model.to(self.device)
            self._model.eval()
            logger.info("视觉情感模型加载完成")

        except Exception as e:
            logger.error(f"视觉情感模型加载失败: {e}")
            raise

    def detect_from_frames(
        self,
        frames: List[np.ndarray],
    ) -> Tuple[Optional[str], float]:
        """从视频帧列表检测情感

        Parameters
        ----------
        frames : List[np.ndarray]
            视频帧列表，每帧 shape 为 (H, W, 3)，RGB 格式

        Returns
        -------
        emotion : str
            检测到的情感标签（happy/sad/angry/fear/surprise/disgust/neutral）
        confidence : float
            置信度 [0, 1]
        """
        if not frames:
            logger.warning("未提供视频帧，返回 neutral")
            return "neutral", 0.0

        try:
            # 对每一帧进行情感检测
            frame_emotions = []
            frame_confidences = []

            for frame in frames:
                # 转为 PIL Image
                if isinstance(frame, np.ndarray):
                    frame = Image.fromarray(frame.astype(np.uint8))

                # 预处理
                inputs = self._processor(images=frame, return_tensors="pt")
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
                frame_emotions.append(emotion_label)
                frame_confidences.append(confidence)

            # 投票：选择出现最多的情感
            from collections import Counter
            emotion_counts = Counter(frame_emotions)
            dominant_emotion = emotion_counts.most_common(1)[0][0]

            # 平均置信度
            avg_confidence = np.mean(frame_confidences)

            # 标准化标签
            dominant_emotion = self._normalize_label(dominant_emotion)

            logger.debug(
                f"视觉情感检测: {dominant_emotion} (conf={avg_confidence:.3f}), "
                f"帧情感分布: {dict(emotion_counts)}"
            )

            return dominant_emotion, float(avg_confidence)

        except Exception as e:
            logger.error(f"视觉情感检测失败: {e}")
            return "neutral", 0.0

    def _normalize_label(self, label: str) -> str:
        """标准化情感标签"""
        from ..config import EMOTION_LABEL_MAPPING

        label_lower = label.lower()
        return EMOTION_LABEL_MAPPING.get(label_lower, label_lower)

    def detect_faces_count(self, frames: List[np.ndarray]) -> int:
        """检测视频中的人脸数量（简单实现：检查是否有高置信度的情感）"""
        if not frames:
            return 0

        # 简单策略：如果能检测到情感且置信度高，说明有人脸
        _, confidence = self.detect_from_frames(frames)
        return 1 if confidence > 0.5 else 0
