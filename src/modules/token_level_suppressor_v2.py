"""
Token-level 选择性注意力抑制器 - V2版本（对抗No偏向）

核心改进：
1. 识别模型的"No"偏向问题
2. 当检测到冲突时，增强"Yes" token的logits来对抗偏向
3. 更激进的冲突检测阈值
4. 简化的实现，专注于解决实际问题
"""

import torch
import logging
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class ConflictSignal:
    """冲突信号"""
    has_emotion_conflict: bool
    emotion_distance: float
    has_content_conflict: bool
    content_conflict_score: float
    conflicting_entities: List[str]

    @property
    def overall_conflict_score(self) -> float:
        """综合冲突分数"""
        return 0.6 * self.emotion_distance + 0.4 * self.content_conflict_score

    @property
    def should_suppress(self) -> bool:
        """是否应该启动抑制（更激进的阈值）"""
        return self.overall_conflict_score >= 0.2  # 降低阈值，更容易触发


class ImprovedTokenLevelSuppressor:
    """
    改进的Token级别抑制器 - 对抗No偏向

    策略：
    1. 检测到冲突时，增强"Yes" token的logits
    2. 同时适度降低"No" token的logits
    3. 根据冲突强度动态调整增强/抑制力度
    """

    def __init__(
        self,
        emotion_conflict_threshold: float = 0.25,  # 降低阈值
        content_conflict_threshold: float = 0.20,  # 降低阈值
        yes_boost_strength: float = 2.0,  # Yes增强强度
        no_penalty_strength: float = 1.0,  # No惩罚强度
    ):
        self.emotion_conflict_threshold = emotion_conflict_threshold
        self.content_conflict_threshold = content_conflict_threshold
        self.yes_boost_strength = yes_boost_strength
        self.no_penalty_strength = no_penalty_strength

        # 情感距离映射
        self.emotion_distances = {
            ('happy', 'sad'): 0.9,
            ('happy', 'angry'): 0.7,
            ('happy', 'fear'): 0.6,
            ('sad', 'happy'): 0.9,
            ('sad', 'angry'): 0.5,
            ('angry', 'happy'): 0.7,
            ('angry', 'sad'): 0.5,
            ('fear', 'happy'): 0.6,
            ('neutral', 'happy'): 0.3,
            ('neutral', 'sad'): 0.3,
        }

    def detect_conflict(
        self,
        visual_emotion: str,
        audio_emotion: str,
        visual_emotion_conf: float,
        audio_emotion_conf: float,
        visual_objects: List[str],
        asr_text: str,
        frames: Optional[List] = None,
        clip_model=None,
        clip_processor=None,
    ) -> ConflictSignal:
        """检测跨模态冲突"""

        # 1. 情感冲突检测
        emotion_distance = self._compute_emotion_distance(
            visual_emotion, audio_emotion,
            visual_emotion_conf, audio_emotion_conf
        )
        has_emotion_conflict = emotion_distance >= self.emotion_conflict_threshold

        # 2. 内容冲突检测（改进版 - 使用CLIP直接验证）
        content_conflict_score = 0.0
        conflicting_entities = []

        if asr_text:
            # 从ASR中提取实体
            asr_lower = asr_text.lower()

            # 扩展的实体列表
            common_entities = [
                'person', 'people', 'man', 'woman', 'child',
                'car', 'vehicle', 'truck', 'bus',
                'dog', 'cat', 'bird', 'animal',
                'tree', 'flower', 'plant',
                'building', 'house', 'room', 'street',
                'phone', 'computer', 'screen',
                'book', 'paper',
                'food', 'drink', 'cake',
                'chair', 'table', 'bed',
                'music', 'sound', 'voice',
            ]

            # 检测ASR中提到的实体
            mentioned_entities = [e for e in common_entities if e in asr_lower]

            if mentioned_entities:
                # 如果有CLIP模型和帧，使用CLIP验证
                if frames and clip_model and clip_processor:
                    conflicting_entities = self._verify_entities_with_clip(
                        mentioned_entities, frames, clip_model, clip_processor
                    )
                else:
                    # 降级方案：检查visual_objects列表
                    if visual_objects:
                        visual_lower = [v.lower() for v in visual_objects]
                        conflicting_entities = [
                            e for e in mentioned_entities
                            if e not in visual_lower
                        ]
                    else:
                        # 如果没有visual_objects，假设所有提到的实体都可能冲突
                        conflicting_entities = mentioned_entities[:3]  # 最多3个

                # 计算冲突分数
                if conflicting_entities:
                    content_conflict_score = min(len(conflicting_entities) * 0.3, 1.0)

        has_content_conflict = content_conflict_score >= self.content_conflict_threshold

        return ConflictSignal(
            has_emotion_conflict=has_emotion_conflict,
            emotion_distance=emotion_distance,
            has_content_conflict=has_content_conflict,
            content_conflict_score=content_conflict_score,
            conflicting_entities=conflicting_entities,
        )

    def _verify_entities_with_clip(
        self,
        entities: List[str],
        frames: List,
        clip_model,
        clip_processor,
        threshold: float = 0.25,
    ) -> List[str]:
        """使用CLIP验证实体是否在视频中"""
        from PIL import Image

        conflicting = []

        for entity in entities[:5]:  # 最多检查5个实体
            try:
                # 准备文本提示
                text_prompts = [f"a photo of a {entity}", f"{entity}"]

                # 检查前4帧
                max_sim = 0.0
                for frame in frames[:4]:
                    if isinstance(frame, np.ndarray):
                        frame_pil = Image.fromarray(frame)
                    else:
                        frame_pil = frame

                    inputs = clip_processor(
                        text=text_prompts,
                        images=frame_pil,
                        return_tensors="pt",
                        padding=True
                    ).to(clip_model.device)

                    with torch.no_grad():
                        outputs = clip_model(**inputs)
                        sim = outputs.logits_per_image.softmax(dim=1).max().item()
                        max_sim = max(max_sim, sim)

                # 如果相似度低于阈值，判定为冲突
                if max_sim < threshold:
                    conflicting.append(entity)
                    logger.info(f"实体冲突: '{entity}' CLIP相似度={max_sim:.3f} < {threshold}")

            except Exception as e:
                logger.warning(f"CLIP验证实体'{entity}'失败: {e}")
                # 失败时保守处理，不判定为冲突
                continue

        return conflicting

    def _compute_emotion_distance(
        self,
        visual_emotion: str,
        audio_emotion: str,
        visual_conf: float,
        audio_conf: float,
    ) -> float:
        """计算情感距离"""
        if not visual_emotion or not audio_emotion:
            return 0.0

        key = (visual_emotion.lower(), audio_emotion.lower())
        base_distance = self.emotion_distances.get(key, 0.0)

        # 考虑置信度
        conf_weight = (visual_conf + audio_conf) / 2.0
        return base_distance * conf_weight

    def create_logits_processor(
        self,
        tokenizer,
        conflict_signal: ConflictSignal,
    ):
        """创建logits处理器"""
        if not conflict_signal.should_suppress:
            return None

        return AntiNoBiasLogitsProcessor(
            tokenizer=tokenizer,
            conflict_score=conflict_signal.overall_conflict_score,
            yes_boost_strength=self.yes_boost_strength,
            no_penalty_strength=self.no_penalty_strength,
        )


class AntiNoBiasLogitsProcessor:
    """
    对抗No偏向的Logits处理器

    策略：增强Yes，惩罚No
    """

    def __init__(
        self,
        tokenizer,
        conflict_score: float,
        yes_boost_strength: float,
        no_penalty_strength: float,
    ):
        self.tokenizer = tokenizer
        self.conflict_score = conflict_score
        self.yes_boost_strength = yes_boost_strength
        self.no_penalty_strength = no_penalty_strength

        # 获取Yes/No token IDs
        self.yes_token_ids = self._get_token_ids(['Yes', 'yes', 'YES', 'Y'])
        self.no_token_ids = self._get_token_ids(['No', 'no', 'NO', 'N'])

        logger.info(
            f"AntiNoBiasLogitsProcessor initialized: "
            f"conflict_score={conflict_score:.3f}, "
            f"yes_boost={yes_boost_strength:.2f}, "
            f"no_penalty={no_penalty_strength:.2f}"
        )

    def _get_token_ids(self, tokens: List[str]) -> List[int]:
        """获取token IDs"""
        token_ids = []
        for token in tokens:
            ids = self.tokenizer.encode(token, add_special_tokens=False)
            if ids:
                token_ids.extend(ids)
        return list(set(token_ids))

    def __call__(
        self,
        input_ids: torch.LongTensor,
        scores: torch.FloatTensor,
    ) -> torch.FloatTensor:
        """
        处理logits：增强Yes，惩罚No
        """
        # 根据冲突强度动态调整
        dynamic_yes_boost = self.yes_boost_strength * self.conflict_score
        dynamic_no_penalty = self.no_penalty_strength * self.conflict_score

        # 增强Yes token
        for token_id in self.yes_token_ids:
            if token_id < scores.shape[-1]:
                scores[:, token_id] += dynamic_yes_boost

        # 惩罚No token
        for token_id in self.no_token_ids:
            if token_id < scores.shape[-1]:
                scores[:, token_id] -= dynamic_no_penalty

        return scores
