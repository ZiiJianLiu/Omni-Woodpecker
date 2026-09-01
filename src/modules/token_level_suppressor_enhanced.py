"""
Token-level 选择性注意力抑制器 - 增强版（方案一完整实现）
Enhanced Token-level Selective Attention Suppression

方案一：Token位置感知的注意力掩码 (Token-Position-Aware Attention Masking)

核心功能：
1. Token位置识别：精确定位audio/visual token在序列中的位置
2. 实体级别抑制：识别并抑制冲突实体相关的token
3. 情感token识别：识别情感相关的token并动态抑制
4. 冲突贡献度计算：为每个token计算冲突贡献分数
5. 动态抑制强度：根据冲突分数自适应调整抑制力度

改进点：
- 不再简单地mask整个模态，而是精确识别并抑制冲突token
- 保留非冲突的有用信息（如背景音、非冲突物体等）
- 支持渐进式抑制（soft masking）
- 可追踪和解释哪些token被抑制了
"""

import torch
import logging
import numpy as np
from typing import Dict, List, Optional, Tuple, Set
from dataclasses import dataclass, field
import re

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
# 情感相关词汇表（用于识别情感token）
# ══════════════════════════════════════════════════════════════════════════════
EMOTION_KEYWORDS = {
    'happy': ['happy', 'joy', 'joyful', 'cheerful', 'delighted', 'pleased', 'glad',
              'smile', 'smiling', 'laugh', 'laughing', 'excited', 'happiness'],
    'sad': ['sad', 'unhappy', 'sorrow', 'sorrowful', 'depressed', 'melancholy',
            'cry', 'crying', 'tear', 'tears', 'upset', 'sadness'],
    'angry': ['angry', 'mad', 'furious', 'rage', 'irritated', 'annoyed',
              'upset', 'frustrated', 'anger', 'angered'],
    'fear': ['fear', 'afraid', 'scared', 'frightened', 'terrified', 'anxious',
             'worried', 'nervous', 'fearful', 'panic'],
    'neutral': ['calm', 'neutral', 'peaceful', 'serene', 'composed', 'relaxed'],
}


# ══════════════════════════════════════════════════════════════════════════════
# 数据结构定义
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class TokenPosition:
    """Token位置信息"""
    start_idx: int
    end_idx: int
    modality: str  # 'audio', 'visual', 'text'
    token_ids: List[int] = field(default_factory=list)
    conflict_score: float = 0.0  # 该位置的冲突分数


@dataclass
class EntityTokenMapping:
    """实体到Token的映射"""
    entity: str
    token_ids: List[int] = field(default_factory=list)
    positions: List[Tuple[int, int]] = field(default_factory=list)  # (start, end) in text
    conflict_score: float = 0.0


@dataclass
class ConflictSignal:
    """冲突信号（增强版）"""
    has_emotion_conflict: bool
    emotion_distance: float  # [0, 1]
    has_content_conflict: bool
    content_conflict_score: float  # [0, 1]

    # Token-level 信息
    conflicting_entities: List[str]  # 冲突的实体
    emotion_conflict_modality: Optional[str]  # 哪个模态的情感是冲突源

    # 方案一新增：详细的token级别信息
    entity_token_mappings: List[EntityTokenMapping] = field(default_factory=list)
    emotion_keywords_in_conflict: List[str] = field(default_factory=list)
    asr_text: Optional[str] = None  # 保存ASR文本用于token识别

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
        """获取抑制强度 [0.0, 1.0]"""
        if not self.should_suppress:
            return 0.0
        return min(1.0, (self.overall_conflict_score - 0.3) / 0.7)


# ══════════════════════════════════════════════════════════════════════════════
# 方案一核心类：增强版Token级别抑制器
# ══════════════════════════════════════════════════════════════════════════════

class EnhancedTokenLevelSuppressor:
    """
    增强版Token级别抑制器（方案一完整实现）

    核心改进：
    1. 精确识别冲突实体的token位置
    2. 识别情感相关的token
    3. 计算每个token的冲突贡献度
    4. 动态调整抑制强度
    """

    def __init__(
        self,
        emotion_conflict_threshold: float = 0.35,
        content_conflict_threshold: float = 0.30,
        min_conflict_for_suppression: float = 0.3,
        yes_bias_range: Tuple[float, float] = (-3.0, 0.0),
        no_bias_range: Tuple[float, float] = (0.0, 2.5),
        # 方案一新增参数
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
            Yes token 的 logit bias 范围
        no_bias_range : Tuple[float, float]
            No token 的 logit bias 范围
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
        # 方案一新增：直接传入 Grounding DINO 检测结果
        grounding_entity_scores: Optional[Dict[str, float]] = None,
    ) -> ConflictSignal:
        """
        检测情感与内容冲突，并识别冲突的具体 token/entity

        方案一增强：
        - 返回详细的实体-token映射
        - 识别情感相关的关键词
        """
        # 1. 情感冲突检测
        emotion_distance = self._compute_emotion_distance(
            visual_emotion, audio_emotion,
            visual_emotion_conf, audio_emotion_conf
        )
        has_emotion_conflict = emotion_distance > self.emotion_threshold

        # 判断哪个模态是冲突源
        emotion_conflict_modality = None
        emotion_keywords = []
        if has_emotion_conflict:
            if audio_emotion_conf > visual_emotion_conf:
                emotion_conflict_modality = 'visual'
                # 识别visual情感相关的关键词
                emotion_keywords = EMOTION_KEYWORDS.get(visual_emotion.lower(), [])
            else:
                emotion_conflict_modality = 'audio'
                # 识别audio情感相关的关键词
                emotion_keywords = EMOTION_KEYWORDS.get(audio_emotion.lower(), [])

        # 2. 内容冲突检测
        # 优先使用 Grounding DINO 结果，fallback 到 visual_objects 关键词匹配
        if grounding_entity_scores is not None:
            content_conflict_score, conflicting_entities = \
                self._compute_content_conflict_from_grounding(grounding_entity_scores)
        else:
            content_conflict_score, conflicting_entities = \
                self._compute_content_conflict_with_entities(visual_objects, asr_text)
        has_content_conflict = content_conflict_score > self.content_threshold

        # 3. 方案一新增：构建实体-token映射
        entity_token_mappings = []
        if has_content_conflict and asr_text:
            entity_token_mappings = self._build_entity_token_mappings(
                asr_text, conflicting_entities, content_conflict_score
            )

        return ConflictSignal(
            has_emotion_conflict=has_emotion_conflict,
            emotion_distance=emotion_distance,
            has_content_conflict=has_content_conflict,
            content_conflict_score=content_conflict_score,
            conflicting_entities=conflicting_entities,
            emotion_conflict_modality=emotion_conflict_modality,
            entity_token_mappings=entity_token_mappings,
            emotion_keywords_in_conflict=emotion_keywords,
            asr_text=asr_text,
        )

    def _compute_emotion_distance(
        self,
        visual_emotion: str,
        audio_emotion: str,
        visual_conf: float,
        audio_conf: float,
    ) -> float:
        """
        计算情感距离。

        不再乘以置信度：情感检测器输出的类别标签已经是最高置信分类结果，
        标签不同就说明冲突，置信度低只代表检测器不确定，不代表冲突不存在。
        用置信度做软权重：取 max(conf) 作为可信度下限，避免双低置信度时误报。
        """
        emotion_polarity = {
            'happy': 1.0, 'excited': 0.8, 'neutral': 0.0,
            'sad': -0.8, 'angry': -0.6, 'fear': -0.7,
        }

        if visual_emotion.lower() == audio_emotion.lower():
            return 0.0

        v_polarity = emotion_polarity.get(visual_emotion.lower(), 0.0)
        a_polarity = emotion_polarity.get(audio_emotion.lower(), 0.0)
        polarity_diff = abs(v_polarity - a_polarity)  # [0, 2.0]

        # 至少有一个模态的置信度达到阈值才认为可信
        max_conf = max(visual_conf, audio_conf)
        if max_conf < 0.25:
            return 0.0

        # 归一化到 [0, 1]，用较高置信度软加权（而非乘以平均置信度）
        normalized_diff = min(1.0, polarity_diff / 2.0)
        credibility = min(1.0, max_conf / 0.6)   # conf=0.6 时满分，低于 0.25 截断
        return normalized_diff * credibility

    def _compute_content_conflict_from_grounding(
        self,
        grounding_entity_scores: Dict[str, float],
        presence_threshold: float = 0.22,
    ) -> Tuple[float, List[str]]:
        """
        基于 Grounding DINO 检测结果计算内容冲突。

        逻辑：ASR 提及的实体在视频帧中置信度 < presence_threshold → 冲突。
        比 CLIP softmax 准确得多。
        """
        if not grounding_entity_scores:
            return 0.0, []

        conflicting_entities = []
        absent_scores = []
        for entity, max_conf in grounding_entity_scores.items():
            if max_conf < presence_threshold:
                conflicting_entities.append(entity)
                # 置信度越低，该实体的冲突越强
                absent_scores.append(1.0 - max_conf)

        if not conflicting_entities:
            return 0.0, []

        # 综合分：冲突实体占比 × 平均缺失强度
        conflict_ratio = len(conflicting_entities) / len(grounding_entity_scores)
        avg_absence = float(np.mean(absent_scores))
        score = min(1.0, conflict_ratio * avg_absence)

        logger.debug(
            "Grounding 内容冲突: %d/%d 实体缺失, score=%.3f, entities=%s",
            len(conflicting_entities), len(grounding_entity_scores),
            score, conflicting_entities,
        )
        return score, conflicting_entities

    def _compute_content_conflict_with_entities(
        self,
        visual_objects: List[str],
        asr_text: str,
    ) -> Tuple[float, List[str]]:
        """计算内容冲突分数，并返回冲突的实体列表"""
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

    def _build_entity_token_mappings(
        self,
        asr_text: str,
        conflicting_entities: List[str],
        conflict_score: float,
    ) -> List[EntityTokenMapping]:
        """
        方案一核心方法：构建实体到token的映射

        识别冲突实体在ASR文本中的位置
        """
        mappings = []
        asr_lower = asr_text.lower()

        for entity in conflicting_entities:
            positions = []
            # 使用正则表达式找到实体的所有出现位置
            pattern = r'\b' + re.escape(entity) + r'\b'
            for match in re.finditer(pattern, asr_lower):
                positions.append((match.start(), match.end()))

            if positions:
                mapping = EntityTokenMapping(
                    entity=entity,
                    positions=positions,
                    conflict_score=conflict_score,
                )
                mappings.append(mapping)

        return mappings

    def create_logits_processor(
        self,
        tokenizer,
        conflict_signal: ConflictSignal,
        query_modality: str,
    ):
        """创建增强版 logits 处理器"""
        if not conflict_signal.should_suppress:
            return None

        return EnhancedTokenLevelLogitsProcessor(
            tokenizer=tokenizer,
            conflict_signal=conflict_signal,
            query_modality=query_modality,
            yes_bias_range=self.yes_bias_range,
            no_bias_range=self.no_bias_range,
            enable_entity_suppression=self.enable_entity_suppression,
            enable_emotion_suppression=self.enable_emotion_suppression,
            entity_suppression_strength=self.entity_suppression_strength,
            emotion_token_bias=self.emotion_token_bias,
        )


# ══════════════════════════════════════════════════════════════════════════════
# 方案一核心类：增强版Logits处理器
# ══════════════════════════════════════════════════════════════════════════════

class EnhancedTokenLevelLogitsProcessor:
    """
    增强版Token-level Logits处理器（方案一完整实现）

    核心功能：
    1. 基础Yes/No token bias（保留原有功能）
    2. 实体级别token抑制（方案一新增）
    3. 情感token抑制（方案一新增）
    """

    def __init__(
        self,
        tokenizer,
        conflict_signal: ConflictSignal,
        query_modality: str,
        yes_bias_range: Tuple[float, float],   # 保留签名兼容，但不再使用方向 bias
        no_bias_range: Tuple[float, float],
        enable_entity_suppression: bool = True,
        enable_emotion_suppression: bool = True,
        entity_suppression_strength: float = 3.0,
        emotion_token_bias: float = -2.0,
    ):
        self.tokenizer = tokenizer
        self.conflict_signal = conflict_signal
        self.query_modality = query_modality

        self.enable_entity_suppression = enable_entity_suppression
        self.enable_emotion_suppression = enable_emotion_suppression
        self.entity_suppression_strength = entity_suppression_strength
        self.emotion_token_bias = emotion_token_bias

        strength = conflict_signal.get_suppression_strength()
        if conflict_signal.has_content_conflict and conflict_signal.conflicting_entities:
            strength = min(1.0, strength * 1.3)
        self.strength = strength

        # ── 不再做 Yes/No 方向 bias ──────────────────────────────────────────
        # 改用 top-logit 平坦化：冲突越强，对高置信 token 的压制越强，
        # 让模型自己重新分配概率，而不是人为指定 Yes/No 方向。
        # flatten_scale ∈ [1.0, 2.5]：strength=0 → 不变，strength=1 → logits/2.5
        # 提升上限，让抑制在中等冲突强度时也能有效干预
        self.flatten_scale = 1.0 + strength * 1.5

        # 实体 token ids（ASR 提及但视频中不存在的实体）
        self.entity_token_ids = set()
        if enable_entity_suppression and conflict_signal.entity_token_mappings:
            self.entity_token_ids = self._identify_entity_token_ids(
                conflict_signal.entity_token_mappings
            )

        # 情感 token ids（冲突模态的情感关键词）
        self.emotion_token_ids = set()
        if enable_emotion_suppression and conflict_signal.emotion_keywords_in_conflict:
            self.emotion_token_ids = self._identify_emotion_token_ids(
                conflict_signal.emotion_keywords_in_conflict
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

    def _identify_entity_token_ids(
        self,
        entity_mappings: List[EntityTokenMapping],
    ) -> Set[int]:
        """
        方案一核心方法：识别冲突实体对应的token ids

        将实体在文本中的位置转换为token ids
        """
        entity_token_ids = set()

        for mapping in entity_mappings:
            entity = mapping.entity
            # 编码实体本身
            entity_tokens = self.tokenizer.encode(entity, add_special_tokens=False)
            entity_token_ids.update(entity_tokens)

            # 也尝试带空格的版本
            entity_tokens_with_space = self.tokenizer.encode(' ' + entity, add_special_tokens=False)
            entity_token_ids.update(entity_tokens_with_space)

        return entity_token_ids

    def _identify_emotion_token_ids(
        self,
        emotion_keywords: List[str],
    ) -> Set[int]:
        """
        方案一核心方法：识别情感相关的token ids
        """
        emotion_token_ids = set()

        for keyword in emotion_keywords:
            # 编码关键词
            keyword_tokens = self.tokenizer.encode(keyword, add_special_tokens=False)
            emotion_token_ids.update(keyword_tokens)

            # 也尝试带空格的版本
            keyword_tokens_with_space = self.tokenizer.encode(' ' + keyword, add_special_tokens=False)
            emotion_token_ids.update(keyword_tokens_with_space)

        return emotion_token_ids

    def __call__(
        self,
        input_ids: torch.LongTensor,
        scores: torch.FloatTensor,
    ) -> torch.FloatTensor:
        """
        Token-level 抑制（无方向 bias 版本）

        三步：
        1. Top-logit 平坦化：冲突越强对高置信 token 压制越强，
           让模型重新分配概率，而非强行指定方向。
        2. 实体 token 抑制：直接降低幻觉实体词的 logit。
        3. 情感 token 抑制：降低冲突模态情感关键词的 logit。
        """
        # 1. 平坦化（temperature scaling 等效）
        if self.flatten_scale > 1.0:
            scores = scores / self.flatten_scale

        # 2. 实体级别 token 抑制
        if self.enable_entity_suppression and self.entity_token_ids:
            for token_id in self.entity_token_ids:
                if token_id < scores.shape[-1]:
                    scores[:, token_id] -= self.entity_suppression_strength

        # 3. 情感 token 抑制
        if self.enable_emotion_suppression and self.emotion_token_ids:
            for token_id in self.emotion_token_ids:
                if token_id < scores.shape[-1]:
                    scores[:, token_id] += self.emotion_token_bias

        return scores


# ══════════════════════════════════════════════════════════════════════════════
# 辅助函数
# ══════════════════════════════════════════════════════════════════════════════

def infer_query_modality(question: str) -> str:
    """推断问题查询的模态"""
    q_lower = question.lower()
    if 'visible' in q_lower or 'see' in q_lower or 'in the video' in q_lower:
        return 'visual'
    elif 'sound' in q_lower or 'hear' in q_lower or 'in the audio' in q_lower:
        return 'audio'
    else:
        return 'visual'
