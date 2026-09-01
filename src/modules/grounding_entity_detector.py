"""
基于Grounding DINO的强大实体检测器

使用Grounding DINO进行开放词汇物体检测，支持文本提示
"""

import torch
import logging
from typing import List, Tuple, Dict, Optional
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)


class GroundingEntityDetector:
    """
    基于Grounding DINO的实体检测器

    特点：
    1. 开放词汇检测 - 可以检测任意文本描述的物体
    2. 文本提示驱动 - 根据ASR文本中的实体进行检测
    3. 高准确率 - 比CLIP更准确的物体定位
    """

    def __init__(
        self,
        model_name: str = "IDEA-Research/grounding-dino-tiny",
        device: str = "cuda",
        box_threshold: float = 0.25,
        text_threshold: float = 0.20,
    ):
        self.device = device
        self.model_name = model_name
        self.box_threshold = box_threshold
        self.text_threshold = text_threshold

        self.model = None
        self.processor = None

        self._load_model()

    def _load_model(self):
        """加载Grounding DINO模型"""
        try:
            from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection

            logger.info(f"加载Grounding DINO模型: {self.model_name}")
            self.processor = AutoProcessor.from_pretrained(self.model_name)
            self.model = AutoModelForZeroShotObjectDetection.from_pretrained(self.model_name)
            self.model.to(self.device)
            self.model.eval()
            logger.info("Grounding DINO模型加载完成")

        except Exception as e:
            logger.error(f"加载Grounding DINO失败: {e}")
            logger.info("尝试使用OWL-ViT作为备选...")
            self._load_owlvit_fallback()

    def _load_owlvit_fallback(self):
        """备选方案：使用OWL-ViT"""
        try:
            from transformers import OwlViTProcessor, OwlViTForObjectDetection

            logger.info("加载OWL-ViT模型作为备选")
            self.processor = OwlViTProcessor.from_pretrained("google/owlvit-base-patch32")
            self.model = OwlViTForObjectDetection.from_pretrained("google/owlvit-base-patch32")
            self.model.to(self.device)
            self.model.eval()
            self.model_name = "owlvit"
            logger.info("OWL-ViT模型加载完成")

        except Exception as e:
            logger.error(f"加载OWL-ViT也失败: {e}")
            raise RuntimeError("无法加载任何物体检测模型")

    def extract_entities_from_asr(self, asr_text: str) -> List[str]:
        """从ASR文本中提取可能的实体"""
        if not asr_text:
            return []

        asr_lower = asr_text.lower()

        # 扩展的实体列表（按类别组织）
        entity_categories = {
            'people': ['person', 'people', 'man', 'woman', 'child', 'baby', 'boy', 'girl'],
            'vehicles': ['car', 'vehicle', 'truck', 'bus', 'motorcycle', 'bicycle', 'bike', 'train', 'airplane', 'boat'],
            'animals': ['dog', 'cat', 'bird', 'horse', 'cow', 'sheep', 'fish', 'animal'],
            'nature': ['tree', 'flower', 'plant', 'grass', 'mountain', 'river', 'ocean', 'sky', 'cloud'],
            'buildings': ['building', 'house', 'room', 'street', 'road', 'bridge', 'door', 'window'],
            'electronics': ['phone', 'computer', 'laptop', 'screen', 'keyboard', 'camera', 'tv', 'television'],
            'objects': ['book', 'paper', 'pen', 'pencil', 'bag', 'bottle', 'cup', 'glass'],
            'food': ['food', 'drink', 'water', 'coffee', 'cake', 'bread', 'fruit', 'vegetable'],
            'furniture': ['chair', 'table', 'desk', 'bed', 'sofa', 'couch'],
            'body_parts': ['hand', 'face', 'eye', 'mouth', 'head', 'arm', 'leg'],
        }

        found_entities = []

        # 检查每个类别的实体
        for category, entities in entity_categories.items():
            for entity in entities:
                if entity in asr_lower:
                    found_entities.append(entity)

        # 去重并保持顺序
        seen = set()
        unique_entities = []
        for entity in found_entities:
            if entity not in seen:
                seen.add(entity)
                unique_entities.append(entity)

        return unique_entities

    def detect_entities_in_frame(
        self,
        frame: np.ndarray,
        entities: List[str],
    ) -> Dict[str, float]:
        """
        在单帧中检测指定的实体

        Returns:
            {entity: confidence} 字典
        """
        if not entities:
            return {}

        try:
            # 转换为PIL Image
            if isinstance(frame, np.ndarray):
                frame_pil = Image.fromarray(frame)
            else:
                frame_pil = frame

            # 准备文本提示
            text_prompts = [f"{entity}" for entity in entities]

            # 处理输入
            inputs = self.processor(
                images=frame_pil,
                text=text_prompts,
                return_tensors="pt"
            ).to(self.device)

            # 推理
            with torch.no_grad():
                outputs = self.model(**inputs)

            # 后处理
            results = self.processor.post_process_grounded_object_detection(
                outputs,
                inputs.input_ids,
                box_threshold=self.box_threshold,
                text_threshold=self.text_threshold,
                target_sizes=[frame_pil.size[::-1]]
            )[0]

            # 提取每个实体的最高置信度
            entity_confidences = {}

            if 'labels' in results and 'scores' in results:
                labels = results['labels']
                scores = results['scores']

                for label, score in zip(labels, scores):
                    label_text = label.lower().strip()
                    score_val = score.item()

                    # 匹配到对应的实体
                    for entity in entities:
                        if entity in label_text or label_text in entity:
                            if entity not in entity_confidences:
                                entity_confidences[entity] = score_val
                            else:
                                entity_confidences[entity] = max(
                                    entity_confidences[entity], score_val
                                )

            return entity_confidences

        except Exception as e:
            logger.warning(f"Grounding DINO检测失败: {e}")
            return {}

    def detect_entity_conflicts(
        self,
        asr_text: str,
        frames: List[np.ndarray],
        confidence_threshold: float = 0.25,
    ) -> Tuple[List[str], float, Dict[str, float]]:
        """
        检测实体冲突

        Returns:
            (conflicting_entities, conflict_score, entity_confidences)
        """
        # 1. 从ASR中提取实体
        entities = self.extract_entities_from_asr(asr_text)

        if not entities:
            return [], 0.0, {}

        logger.info(f"从ASR中提取到 {len(entities)} 个实体: {entities}")

        # 2. 在视频帧中检测这些实体
        entity_max_confidences = {entity: 0.0 for entity in entities}

        # 检查多帧（最多8帧）
        frames_to_check = frames[:8] if len(frames) > 8 else frames

        for i, frame in enumerate(frames_to_check):
            frame_results = self.detect_entities_in_frame(frame, entities)

            # 更新最大置信度
            for entity, conf in frame_results.items():
                entity_max_confidences[entity] = max(
                    entity_max_confidences[entity], conf
                )

        # 3. 判定冲突：ASR提到但视频中未检测到（置信度低）
        conflicting_entities = []
        conflict_scores = []

        for entity, max_conf in entity_max_confidences.items():
            if max_conf < confidence_threshold:
                # ASR提到但视频中未检测到 → 冲突
                conflicting_entities.append(entity)
                # 置信度越低，冲突越强
                conflict_score = 1.0 - max_conf
                conflict_scores.append(conflict_score)

                logger.info(
                    f"✗ 实体冲突: ASR提到'{entity}'但视频中未检测到 "
                    f"(最高置信度={max_conf:.3f} < {confidence_threshold})"
                )
            else:
                logger.info(
                    f"✓ 实体匹配: '{entity}' 在视频中检测到 "
                    f"(置信度={max_conf:.3f})"
                )

        # 4. 计算综合冲突分数
        if conflict_scores:
            avg_conflict_score = sum(conflict_scores) / len(conflict_scores)
        else:
            avg_conflict_score = 0.0

        return conflicting_entities, avg_conflict_score, entity_max_confidences


# ============================================================================
# 集成到V2抑制器
# ============================================================================

def integrate_grounding_detector_to_v2():
    """
    将Grounding DINO集成到V2抑制器的示例代码
    """
    example_code = '''
# 在V2抑制器初始化时添加：
from modules.grounding_entity_detector import GroundingEntityDetector

self.entity_detector = GroundingEntityDetector(
    device='cuda',
    box_threshold=0.25,
    text_threshold=0.20,
)

# 在detect_conflict方法中使用：
if asr_text and frames:
    conflicting_entities, content_conflict_score, entity_confs = \\
        self.entity_detector.detect_entity_conflicts(
            asr_text=asr_text,
            frames=frames,
            confidence_threshold=0.25,
        )
    has_content_conflict = content_conflict_score >= self.content_conflict_threshold
'''
    return example_code
