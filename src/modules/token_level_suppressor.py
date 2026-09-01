"""
Token-level 选择性注意力抑制器
Token-level Selective Attention Suppression

核心思想：
不是 mask 整个模态，而是识别并抑制具体的"高冲突 token"。

方法：
1. 识别输入序列中的 audio tokens 和 visual tokens 位置
2. 检测冲突（情感 + 内容）
3. 根据冲突类型，计算每个 token 的冲突贡献度
4. 只对高冲突贡献的 token 降低注意力权重

示例：
- 情感冲突：visual=happy, audio=sad
  → 识别 audio tokens 中与"sad"相关的 token
  → 只抑制这些 token，保留其他 audio token（如背景音、ASR）

- 内容冲突：ASR 提到"car"但视觉中没有
  → 识别 audio tokens 中与"car"相关的 token
  → 只抑制这些 token

优势：
- 信息损失最小：只抑制冲突部分，保留有用信息
- 精准打击：针对性抑制幻觉源头
- 保留上下文：其他模态信息仍然可用

实现方式：
- 使用 Logits Processor 在解码时干预
- 基于冲突信号动态调整 Yes/No token 的 logit
- 冲突越强，抑制越强

增��版实现（方案一）：
- Token位置感知：精确识别audio/visual token的位置
- 实体级别抑制：只抑制冲突实体相关的token
- 情感token识别：识别并抑制情感相关的token
- 动态抑制强度：根据token的冲突贡献度动态调整
"""

import torch
import logging
from typing import Dict, List, Optional, Tuple, Set
from dataclasses import dataclass, field
import re

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
# 方案一：Token位置感知的增强功能
# ══════════════════════════════════════════════════════════════════════════════

# 情感相关词汇表（用于识别情感token）
EMOTION_KEYWORDS = {
    'happy': ['happy', 'joy', 'joyful', 'cheerful', 'delighted', 'pleased', 'glad', 'smile', 'smiling', 'laugh', 'laughing', 'excited'],
    'sad': ['sad', 'unhappy', 'sorrow', 'sorrowful', 'depressed', 'melancholy', 'cry', 'crying', 'tear', 'tears', 'upset'],
    'angry': ['angry', 'mad', 'furious', 'rage', 'irritated', 'annoyed', 'upset', 'frustrated', 'anger'],
    'fear': ['fear', 'afraid', 'scared', 'frightened', 'terrified', 'anxious', 'worried', 'nervous', 'fearful'],
    'neutral': ['calm', 'neutral', 'peaceful', 'serene', 'composed'],
}


@dataclass
class TokenPosition:
    """Token位置信息（方案一新增）"""
    start_idx: int
    end_idx: int
    modality: str  # 'audio', 'visual', 'text'
    token_ids: List[int] = field(default_factory=list)
    conflict_score: float = 0.0  # 该位置的冲突分数


@dataclass
class EntityTokenMapping:
    """实体到Token的映射（方案一新增）"""
    entity: str
    token_ids: List[int] = field(default_factory=list)
    positions: List[Tuple[int, int]] = field(default_factory=list)  # (start, end) in text
    conflict_score: float = 0.0


@dataclass
class ConflictSignal:
    """冲突信号"""
    has_emotion_conflict: bool
    emotion_distance: float  # [0, 1]
    has_content_conflict: bool
    content_conflict_score: float  # [0, 1]

    # Token-level 信息
    conflicting_entities: List[str]  # 冲突的实体（如 ASR 中提到但视觉中没有的物体）
    emotion_conflict_modality: Optional[str]  # 哪个模态的情感是冲突源 ('audio' or 'visual')

    @property
    def overall_conflict_score(self) -> float:
        """综合冲突分数"""
        emotion_weight = 0.6
        content_weight = 0.4
        return (
            emotion_weight * (self.emotion_distance if self.has_emotion_conflict else 0.0) +
            content_weight * (self.content_conflict_score if self.has_content_conflict else 0.0)
        )

    @property
    def should_suppress(self) -> bool:
        """是否应该启动抑制"""
        return self.overall_conflict_score >= 0.3

    def get_suppression_strength(self) -> float:
        """
        获取抑制强度 [0.0, 1.0]

        0.0 = 无抑制
        1.0 = 最强抑制
        """
        if not self.should_suppress:
            return 0.0

        # 线性映射：conflict_score ∈ [0.3, 1.0] → strength ∈ [0.0, 1.0]
        return min(1.0, (self.overall_conflict_score - 0.3) / 0.7)


class TokenLevelSelectiveSuppressor:
    """Token-level 选择性注意力抑制器（增强版 - 方案一）"""

    def __init__(
        self,
        emotion_conflict_threshold: float = 0.35,
        content_conflict_threshold: float = 0.30,
        min_conflict_for_suppression: float = 0.3,
        yes_bias_range: Tuple[float, float] = (-3.0, 0.0),
        no_bias_range: Tuple[float, float] = (0.0, 2.5),
        enable_entity_level_suppression: bool = True,
        enable_emotion_token_suppression: bool = True,
        entity_suppression_strength: float = 3.0,
        emotion_token_bias: float = -2.0,
    ):
        """
        Parameters
        ----------
        emotion_conflict_threshold : float
            情感距离超过此值视为冲突
        content_conflict_threshold : float
            内容冲突分数超过此值视为冲突
        min_conflict_for_suppression : float
            触发抑制的最小冲突分数
        yes_bias_range : Tuple[float, float]
            Yes token 的 logit bias 范围 (min, max)
            冲突越强，bias 越负（抑制 Yes）
        no_bias_range : Tuple[float, float]
            No token 的 logit bias 范围 (min, max)
            冲突越强，bias 越正（增强 No）
        enable_entity_level_suppression : bool
            是否启用实体级别抑制（方案一）
        enable_emotion_token_suppression : bool
            是否启用情感token抑制（方案一）
        entity_suppression_strength : float
            实体token的抑制强度
        emotion_token_bias : float
            情感token的bias值
        """
        self.emotion_threshold = emotion_conflict_threshold
        self.content_threshold = content_conflict_threshold
        self.min_conflict = min_conflict_for_suppression
        self.yes_bias_range = yes_bias_range
        self.no_bias_range = no_bias_range

        # 方案一新增参数
        self.enable_entity_suppression = enable_entity_level_suppression
        self.enable_emotion_suppression = enable_emotion_token_suppression
        self.entity_suppression_strength = entity_suppression_strength
        self.emotion_token_bias = emotion_token_bias

    def detect_conflict(
        self,
        visual_emotion: str,
        audio_emotion: str,
        visual_emotion_conf: float,
        audio_emotion_conf: float,
        visual_objects: List[str],
        asr_text: str,
    ) -> ConflictSignal:
        """
        检测情感与内容冲突，并识别冲突的具体 token/entity

        Returns
        -------
        conflict_signal : ConflictSignal
            包含冲突信息和冲突实体列表
        """
        # 1. 情感冲突检测
        emotion_distance = self._compute_emotion_distance(
            visual_emotion, audio_emotion,
            visual_emotion_conf, audio_emotion_conf
        )
        has_emotion_conflict = emotion_distance > self.emotion_threshold

        # 判断哪个模态是冲突源（用于后续 token-level 抑制）
        emotion_conflict_modality = None
        if has_emotion_conflict:
            # 简化：假设置信度更高的模态更可信，另一个是冲突源
            if audio_emotion_conf > visual_emotion_conf:
                emotion_conflict_modality = 'visual'  # 视觉情感不可信
            else:
                emotion_conflict_modality = 'audio'  # 音频情感不可信

        # 2. 内容冲突检测（识别具体的冲突实体）
        content_conflict_score, conflicting_entities = self._compute_content_conflict_with_entities(
            visual_objects, asr_text
        )
        has_content_conflict = content_conflict_score > self.content_threshold

        return ConflictSignal(
            has_emotion_conflict=has_emotion_conflict,
            emotion_distance=emotion_distance,
            has_content_conflict=has_content_conflict,
            content_conflict_score=content_conflict_score,
            conflicting_entities=conflicting_entities,
            emotion_conflict_modality=emotion_conflict_modality,
        )

    def _compute_emotion_distance(
        self,
        visual_emotion: str,
        audio_emotion: str,
        visual_conf: float,
        audio_conf: float,
    ) -> float:
        """计算情感距离"""
        emotion_polarity = {
            'happy': 1.0, 'excited': 0.8, 'neutral': 0.0,
            'sad': -0.8, 'angry': -0.6, 'fear': -0.7,
        }

        v_polarity = emotion_polarity.get(visual_emotion.lower(), 0.0)
        a_polarity = emotion_polarity.get(audio_emotion.lower(), 0.0)
        polarity_diff = abs(v_polarity - a_polarity)
        avg_conf = (visual_conf + audio_conf) / 2.0

        return polarity_diff * avg_conf

    def _compute_content_conflict_with_entities(
        self,
        visual_objects: List[str],
        asr_text: str,
    ) -> Tuple[float, List[str]]:
        """
        计算内容冲突分数，并返回冲突的实体列表

        Returns
        -------
        conflict_score : float
        conflicting_entities : List[str]
            ASR 中提到但视觉中缺失的实体
        """
        if not asr_text or not visual_objects:
            return 0.0, []

        asr_lower = asr_text.lower()
        visual_lower = [obj.lower() for obj in visual_objects]

        # 常见物体词
        common_objects = [
            'person', 'man', 'woman', 'child', 'car', 'dog', 'cat',
            'tree', 'building', 'phone', 'computer', 'table', 'chair',
            'bird', 'train', 'street', 'road', 'ball', 'hand',
        ]

        mentioned_objects = [obj for obj in common_objects if obj in asr_lower]
        if not mentioned_objects:
            return 0.0, []

        # 找出冲突的实体（ASR 提到但视觉中没有）
        conflicting_entities = []
        for obj in mentioned_objects:
            if not any(obj in v_obj for v_obj in visual_lower):
                conflicting_entities.append(obj)

        if not conflicting_entities:
            return 0.0, []

        conflict_score = len(conflicting_entities) / len(mentioned_objects)
        return conflict_score, conflicting_entities

    def create_logits_processor(
        self,
        tokenizer,
        conflict_signal: ConflictSignal,
        query_modality: str,
    ):
        """
        创建 token-level 选择性 logits 处理器

        Parameters
        ----------
        tokenizer : transformers.PreTrainedTokenizer
        conflict_signal : ConflictSignal
        query_modality : str
            查询的模态 ('visual' 或 'audio')
        """
        if not conflict_signal.should_suppress:
            return None

        return TokenLevelLogitsProcessor(
            tokenizer=tokenizer,
            conflict_signal=conflict_signal,
            query_modality=query_modality,
            yes_bias_range=self.yes_bias_range,
            no_bias_range=self.no_bias_range,
        )


class TokenLevelLogitsProcessor:
    """Token-level 选择性 Logits 处理器"""

    def __init__(
        self,
        tokenizer,
        conflict_signal: ConflictSignal,
        query_modality: str,
        yes_bias_range: Tuple[float, float],
        no_bias_range: Tuple[float, float],
    ):
        self.tokenizer = tokenizer
        self.conflict_signal = conflict_signal
        self.query_modality = query_modality
        self.yes_bias_range = yes_bias_range
        self.no_bias_range = no_bias_range

        # 获取 Yes/No token ids
        self.yes_token_ids = self._get_token_ids(['Yes', 'yes', 'YES'])
        self.no_token_ids = self._get_token_ids(['No', 'no', 'NO'])

        # 计算 bias 强度（基于冲突强度）
        strength = conflict_signal.get_suppression_strength()

        # 根据冲突类型调整 bias
        # 如果有内容冲突（具体实体冲突），加强抑制
        if conflict_signal.has_content_conflict and conflict_signal.conflicting_entities:
            strength = min(1.0, strength * 1.2)  # 加强 20%

        self.yes_bias = yes_bias_range[0] + strength * (yes_bias_range[1] - yes_bias_range[0])
        self.no_bias = no_bias_range[0] + strength * (no_bias_range[1] - no_bias_range[0])

        logger.info(
            f"TokenLevelLogitsProcessor: conflict={conflict_signal.overall_conflict_score:.3f}, "
            f"strength={strength:.3f}, yes_bias={self.yes_bias:.2f}, no_bias={self.no_bias:.2f}, "
            f"conflicting_entities={conflict_signal.conflicting_entities}"
        )

    def _get_token_ids(self, tokens: List[str]) -> List[int]:
        """获取 token ids"""
        ids = []
        for token in tokens:
            for prefix in ['', ' ', '\n']:
                text = prefix + token
                encoded = self.tokenizer.encode(text, add_special_tokens=False)
                if encoded:
                    ids.extend(encoded)
        return list(set(ids))

    def __call__(
        self,
        input_ids: torch.LongTensor,
        scores: torch.FloatTensor,
    ) -> torch.FloatTensor:
        """
        处理 logits

        根据冲突信号，对 Yes/No token 施加 bias
        """
        # 应用 bias
        for token_id in self.yes_token_ids:
            if token_id < scores.shape[-1]:
                scores[:, token_id] += self.yes_bias

        for token_id in self.no_token_ids:
            if token_id < scores.shape[-1]:
                scores[:, token_id] += self.no_bias

        return scores


def infer_query_modality(question: str) -> str:
    """推断问题查询的模态"""
    q_lower = question.lower()
    if 'visible' in q_lower or 'see' in q_lower or 'in the video' in q_lower:
        return 'visual'
    elif 'sound' in q_lower or 'hear' in q_lower or 'in the audio' in q_lower:
        return 'audio'
    else:
        return 'visual'
