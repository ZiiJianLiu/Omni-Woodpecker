"""
改进的实体冲突检测 - 不依赖视觉物体检测器

核心思路：
1. 从ASR文本中提取实体
2. 使用CLIP直接验证这些实体是否在视频中出现
3. 如果ASR提到的实体在视频中CLIP相似度很低，则判定为冲突
"""

import torch
import logging
from typing import List, Tuple
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)


class ImprovedEntityConflictDetector:
    """
    改进的实体冲突检测器

    不依赖物体检测器的输出，而是：
    1. 从ASR中提取实体
    2. 用CLIP直接验证实体是否在视频中
    """

    def __init__(self, clip_model, clip_processor, device='cuda'):
        self.clip_model = clip_model
        self.clip_processor = clip_processor
        self.device = device

        # 常见物体类别（扩展列表）
        self.common_entities = [
            'person', 'people', 'man', 'woman', 'child', 'baby',
            'car', 'vehicle', 'truck', 'bus', 'motorcycle', 'bicycle',
            'dog', 'cat', 'bird', 'horse', 'animal',
            'tree', 'flower', 'plant', 'grass',
            'building', 'house', 'room', 'street', 'road',
            'phone', 'computer', 'laptop', 'screen', 'keyboard',
            'book', 'paper', 'pen', 'pencil',
            'food', 'drink', 'water', 'coffee', 'cake',
            'chair', 'table', 'desk', 'bed', 'sofa',
            'door', 'window', 'wall', 'floor',
            'hand', 'face', 'eye', 'mouth',
            'music', 'sound', 'voice', 'song',
        ]

    def extract_entities_from_asr(self, asr_text: str) -> List[str]:
        """从ASR文本中提取可能的实体"""
        if not asr_text:
            return []

        asr_lower = asr_text.lower()
        found_entities = []

        # 检查常见实体
        for entity in self.common_entities:
            if entity in asr_lower:
                found_entities.append(entity)

        return found_entities

    def verify_entity_in_frames(
        self,
        entity: str,
        frames: List[np.ndarray],
        threshold: float = 0.25,
    ) -> Tuple[bool, float]:
        """
        使用CLIP验证实体是否在视频帧中出现

        Returns:
            (is_present, max_similarity)
        """
        if not frames:
            return False, 0.0

        try:
            # 准备文本提示
            text_prompts = [
                f"a photo of a {entity}",
                f"a video showing a {entity}",
                f"{entity}",
            ]

            # 计算每帧的相似度
            max_similarity = 0.0

            for frame in frames[:8]:  # 最多检查8帧
                # 转换为PIL Image
                if isinstance(frame, np.ndarray):
                    frame_pil = Image.fromarray(frame)
                else:
                    frame_pil = frame

                # CLIP编码
                inputs = self.clip_processor(
                    text=text_prompts,
                    images=frame_pil,
                    return_tensors="pt",
                    padding=True
                ).to(self.device)

                with torch.no_grad():
                    outputs = self.clip_model(**inputs)
                    logits_per_image = outputs.logits_per_image
                    probs = logits_per_image.softmax(dim=1)

                    # 取最大相似度
                    frame_max_sim = probs.max().item()
                    max_similarity = max(max_similarity, frame_max_sim)

            is_present = max_similarity >= threshold
            return is_present, max_similarity

        except Exception as e:
            logger.warning(f"CLIP验证实体失败: {e}")
            return False, 0.0

    def detect_entity_conflicts(
        self,
        asr_text: str,
        frames: List[np.ndarray],
        threshold: float = 0.25,
    ) -> Tuple[List[str], float]:
        """
        检测实体冲突

        Returns:
            (conflicting_entities, conflict_score)
        """
        # 1. 从ASR中提取实体
        entities = self.extract_entities_from_asr(asr_text)

        if not entities:
            return [], 0.0

        # 2. 验证每个实体是否在视频中
        conflicting_entities = []
        conflict_scores = []

        for entity in entities:
            is_present, similarity = self.verify_entity_in_frames(
                entity, frames, threshold
            )

            if not is_present:
                # ASR提到但视频中不存在 → 冲突
                conflicting_entities.append(entity)
                # 相似度越低，冲突越强
                conflict_scores.append(1.0 - similarity)
                logger.info(
                    f"实体冲突: ASR提到'{entity}'但视频中未出现 "
                    f"(CLIP相似度={similarity:.3f})"
                )

        # 3. 计算综合冲突分数
        if conflict_scores:
            avg_conflict_score = sum(conflict_scores) / len(conflict_scores)
        else:
            avg_conflict_score = 0.0

        return conflicting_entities, avg_conflict_score
