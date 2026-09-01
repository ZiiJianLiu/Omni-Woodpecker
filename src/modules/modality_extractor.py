"""
Modality Extractor
==================
提取视频/音频/文本的多模态特征（Training-free）
"""
import logging
import tempfile
import subprocess
from pathlib import Path
from typing import Tuple, List, Optional
import numpy as np
import librosa
from PIL import Image

from ..data_types import ModalityFeatures
from ..detectors import (
    VisualEmotionDetector,
    AudioEmotionDetector,
    ASRDetector,
    ObjectDetector,
)

logger = logging.getLogger(__name__)


class ModalityExtractor:
    """多模态特征提取器（Training-free）"""

    def __init__(
        self,
        visual_emotion_model: str,
        audio_emotion_model: str,
        asr_model_size: str,
        clip_model: str,
        grounding_model: Optional[str] = None,
        grounding_fallback_model: Optional[str] = None,
        device: str = "cuda",
    ):
        self.device = self._resolve_runtime_device(device)

        # 初始化各个检测器
        logger.info("初始化多模态特征提取器 (device=%s)...", self.device)
        self.visual_detector = VisualEmotionDetector(visual_emotion_model, self.device)
        self.audio_detector = AudioEmotionDetector(audio_emotion_model, self.device)
        self.asr_detector = ASRDetector(asr_model_size, self.device)
        self.object_detector = ObjectDetector(
            clip_model,
            self.device,
            grounding_model_name=grounding_model,
            grounding_fallback_model_name=grounding_fallback_model,
        )

        # 文本情感分类器（缓存，避免每次重新加载）
        self._text_emotion_classifier = None

        logger.info("多模态特征提取器初始化完成")

    @staticmethod
    def _path_exists(path: Optional[str]) -> bool:
        return bool(path) and Path(path).exists()

    @staticmethod
    def _resolve_runtime_device(device: Optional[str]) -> str:
        requested = str(device or '').strip()
        try:
            import torch
        except Exception:
            return requested or 'cpu'

        if requested == 'auto' or not requested:
            if not torch.cuda.is_available():
                logger.warning("device=auto 但 CUDA 不可用，特征提取器回退到 cpu")
                return 'cpu'

            n_gpus = torch.cuda.device_count()
            if n_gpus <= 1:
                return 'cuda:0'

            best_idx = 0
            best_free = -1
            for idx in range(n_gpus):
                try:
                    free_bytes = int(torch.cuda.mem_get_info(idx)[0])
                except Exception:
                    free_bytes = 0
                if free_bytes > best_free:
                    best_free = free_bytes
                    best_idx = idx
            chosen = f'cuda:{best_idx}'
            logger.info(
                "特征提取器自动选择设备: %s (free=%.1f GiB)",
                chosen,
                max(best_free, 0) / float(1024 ** 3),
            )
            return chosen

        if requested.startswith('cuda') and not torch.cuda.is_available():
            logger.warning("请求设备 %s 但 CUDA 不可用，特征提取器回退到 cpu", requested)
            return 'cpu'
        return requested

    def extract(
        self,
        video_path: Optional[str],
        answer_text: Optional[str] = None,
        *,
        audio_path: Optional[str] = None,
    ) -> ModalityFeatures:
        return self.extract_media(
            video_path=video_path,
            audio_path=audio_path,
            answer_text=answer_text,
        )

    def extract_media(
        self,
        *,
        video_path: Optional[str] = None,
        audio_path: Optional[str] = None,
        answer_text: Optional[str] = None,
    ) -> ModalityFeatures:
        """提取视频的多模态特征

        Parameters
        ----------
        video_path : Optional[str]
            视频文件路径（若存在视觉模态）
        audio_path : Optional[str]
            外置音频文件路径（若显式提供则优先于视频内嵌音轨）
        answer_text : Optional[str]
            模型生成的答案文本（用于提取文本情感）

        Returns
        -------
        features : ModalityFeatures
            提取的特征
        """
        features = ModalityFeatures()

        # 1. 提取视频帧
        frames = self._extract_frames(video_path, n_frames=8) if self._path_exists(video_path) else []

        # 2. 提取音频
        audio_source = None
        if self._path_exists(audio_path):
            audio_source = audio_path
        elif self._path_exists(video_path):
            audio_source = video_path
        audio, sr = self._extract_audio(audio_source)

        # 3. 视觉特征
        if frames:
            features.visual_emotion, features.visual_emotion_conf = \
                self.visual_detector.detect_from_frames(frames)
            features.visual_objects = self.object_detector.detect_objects(frames)
            features.visual_scene = self._classify_scene(frames)
            features.visual_faces_count = self.visual_detector.detect_faces_count(frames)

        # 4. 音频特征
        if audio is not None and len(audio) > 0:
            features.audio_emotion, features.audio_emotion_conf = \
                self.audio_detector.detect(audio, sr)
            features.audio_type = self.audio_detector.classify_audio_type(audio, sr)
            features.audio_energy = float(np.sqrt(np.mean(audio ** 2)))
            features.audio_has_speech = (features.audio_type == "speech")

            # ASR（仅当音频类型为 speech 时）
            if features.audio_has_speech:
                asr_payload = self.asr_detector.transcribe_with_timestamps(audio, sr)
                features.asr_text = asr_payload.get('text')
                features.asr_segments = asr_payload.get('segments', [])
                features.asr_timestamp_index = asr_payload.get('index', {})

        # 5. 文本特征（如果提供了答案）
        if answer_text:
            features.text_emotion, features.text_emotion_conf = \
                self._classify_text_emotion(answer_text)
            features.text_content = answer_text

        return features

    def _extract_frames(self, video_path: Optional[str], n_frames: int = 8) -> List[np.ndarray]:
        """从视频提取均匀分布的帧

        Returns
        -------
        frames : List[np.ndarray]
            帧列表，每帧 shape 为 (H, W, 3)，RGB 格式
        """
        if not self._path_exists(video_path):
            return []
        try:
            import cv2

            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                logger.error(f"无法打开视频: {video_path}")
                return []

            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if total_frames == 0:
                logger.error(f"视频帧数为 0: {video_path}")
                return []

            # 计算采样间隔
            indices = np.linspace(0, total_frames - 1, n_frames, dtype=int)

            frames = []
            for idx in indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ret, frame = cap.read()
                if ret:
                    # BGR → RGB
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frames.append(frame_rgb)

            cap.release()
            logger.debug(f"提取了 {len(frames)} 帧")
            return frames

        except Exception as e:
            logger.error(f"视频帧提取失败: {e}")
            return []

    def _extract_audio(self, media_path: Optional[str]) -> Tuple[Optional[np.ndarray], int]:
        """从视频或音频文件提取音频

        Returns
        -------
        audio : np.ndarray
            音频波形，单声道
        sr : int
            采样率
        """
        if not self._path_exists(media_path):
            return None, 16000

        try:
            audio, sr = librosa.load(media_path, sr=16000, mono=True)
            if len(audio) > 0:
                logger.debug(f"直接加载音频: {len(audio)} 采样点, {sr} Hz")
                return audio, sr
        except Exception:
            pass

        try:
            # 使用 ffmpeg 提取音频到临时文件
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                tmp_path = tmp.name

            cmd = [
                "ffmpeg", "-y", "-i", media_path,
                "-vn",  # 不处理视频
                "-acodec", "pcm_s16le",
                "-ar", "16000",  # 16kHz
                "-ac", "1",  # 单声道
                tmp_path,
                "-loglevel", "error",
            ]

            ret = subprocess.run(cmd, capture_output=True)
            if ret.returncode != 0 or not Path(tmp_path).exists():
                logger.warning(f"音频提取失败: {media_path}")
                return None, 16000

            # 加载音频
            audio, sr = librosa.load(tmp_path, sr=16000, mono=True)

            # 删除临时文件
            Path(tmp_path).unlink()

            logger.debug(f"提取音频: {len(audio)} 采样点, {sr} Hz")
            return audio, sr

        except Exception as e:
            logger.error(f"音频提取失败: {e}")
            return None, 16000

    def _classify_scene(self, frames: List[np.ndarray]) -> Optional[str]:
        """分类场景类型（简单实现）

        Returns
        -------
        scene : str
            indoor/outdoor/nature/urban
        """
        if not frames:
            return None

        try:
            # 使用 CLIP 进行零样本分类
            scene_categories = ["indoor", "outdoor", "nature", "urban"]
            frame = frames[len(frames) // 2]  # 取中间帧

            # 转为 PIL Image
            if isinstance(frame, np.ndarray):
                frame = Image.fromarray(frame.astype(np.uint8))

            # 构造文本提示
            text_prompts = [f"a photo of {scene}" for scene in scene_categories]

            # 预处理
            inputs = self.object_detector._processor(
                text=text_prompts,
                images=frame,
                return_tensors="pt",
                padding=True,
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            # 推理
            import torch
            with torch.no_grad():
                outputs = self.object_detector._model(**inputs)
                logits_per_image = outputs.logits_per_image
                probs = logits_per_image.softmax(dim=1)[0]
                pred_idx = probs.argmax().item()

            scene = scene_categories[pred_idx]
            logger.debug(f"场景分类: {scene}")
            return scene

        except Exception as e:
            logger.error(f"场景分类失败: {e}")
            return None

    def _classify_text_emotion(self, text: str) -> Tuple[Optional[str], float]:
        """分类文本情感

        Returns
        -------
        emotion : str
            情感标签
        confidence : float
            置信度
        """
        if not text:
            return None, 0.0

        try:
            # 懒加载并缓存分类器
            if self._text_emotion_classifier is None:
                from transformers import pipeline as hf_pipeline

                # 解析 device index: "cuda:0" → 0, "cuda:1" → 1, "cpu" → -1
                if self.device.startswith("cuda"):
                    dev = int(self.device.split(":")[1]) if ":" in self.device else 0
                else:
                    dev = -1

                self._text_emotion_classifier = hf_pipeline(
                    "text-classification",
                    model="j-hartmann/emotion-english-distilroberta-base",
                    device=dev,
                )

            result = self._text_emotion_classifier(text[:512])[0]  # 限制长度
            emotion = result["label"].lower()
            confidence = result["score"]

            # 标准化标签
            from ..config import EMOTION_LABEL_MAPPING
            emotion = EMOTION_LABEL_MAPPING.get(emotion, emotion)

            logger.debug(f"文本情感: {emotion} (conf={confidence:.3f})")
            return emotion, float(confidence)

        except Exception as e:
            logger.error(f"文本情感分类失败: {e}")
            return None, 0.0
