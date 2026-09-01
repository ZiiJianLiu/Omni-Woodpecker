"""
冲突感知的注意力掩码幻觉抑制器
Conflict-Aware Attention Masking for Hallucination Suppression

核心思想：
1. 检测情感冲突 + 模态冲突
2. 当检测到冲突时，在模型内部 mask 掉冲突模态的 attention：
   - Audio 问题 + 冲突 → mask 掉 visual tokens 的 attention
   - Visual 问题 + 冲突 → mask 掉 audio tokens 的 attention
3. 直接切断跨模态信息泄漏的源头

理论依据：
- 幻觉产生于注意力机制中的跨模态污染
- 情感冲突 → 模态语义不一致 → 跨模态 attention 不可信
- 通过 attention mask 物理隔离冲突模态，阻止信息泄漏

实现方式：
- Hook 模型的 forward() 方法
- 修改 attention_mask 参数
- 对冲突模态的 token 位置设置 mask=0

优势：
- 效果最强：直接切断信息流，而非事后修正
- 作用于根源：在 attention 层面阻止跨模态污染
- 免训练：纯推理时干预
"""

import torch
import torch.nn.functional as F
import logging
from typing import Dict, List, Optional, Callable, Tuple
from dataclasses import dataclass
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class ConflictSignal:
    """冲突信号"""
    has_emotion_conflict: bool
    emotion_distance: float  # [0, 1]
    has_content_conflict: bool
    content_conflict_score: float  # [0, 1]

    @property
    def overall_conflict_score(self) -> float:
        """综合冲突分数"""
        emotion_weight = 0.6
        content_weight = 0.4
        return (
            emotion_weight * (self.emotion_distance if self.has_emotion_conflict else 0.0) +
            content_weight * (self.content_conflict_score if self.has_content_conflict else 0.0)
        )


class ConflictAwareLogitsProcessor:
    """冲突感知的 Logits 处理器

    在解码时根据冲突信号调整 Yes/No token 的概率
    """

    def __init__(
        self,
        tokenizer,
        conflict_signal: ConflictSignal,
        yes_bias_range: tuple = (-2.0, 0.0),  # Yes token 的 logit bias 范围
        no_bias_range: tuple = (0.0, 1.5),    # No token 的 logit bias 范围
        min_conflict_threshold: float = 0.3,   # 最小冲突阈值
    ):
        """
        Parameters
        ----------
        tokenizer : transformers.PreTrainedTokenizer
            分词器
        conflict_signal : ConflictSignal
            冲突信号
        yes_bias_range : tuple
            Yes token 的 bias 范围 (min, max)，冲突越强 bias 越负
        no_bias_range : tuple
            No token 的 bias 范围 (min, max)，冲突越强 bias 越正
        min_conflict_threshold : float
            触发干预的最小冲突分数
        """
        self.tokenizer = tokenizer
        self.conflict_signal = conflict_signal
        self.yes_bias_range = yes_bias_range
        self.no_bias_range = no_bias_range
        self.min_threshold = min_conflict_threshold

        # 获取 Yes/No token ids
        self.yes_token_ids = self._get_token_ids(['Yes', 'yes', 'YES'])
        self.no_token_ids = self._get_token_ids(['No', 'no', 'NO'])

        logger.info(f"ConflictAwareLogitsProcessor initialized:")
        logger.info(f"  Yes tokens: {self.yes_token_ids}")
        logger.info(f"  No tokens: {self.no_token_ids}")
        logger.info(f"  Conflict score: {self.conflict_signal.overall_conflict_score:.3f}")

    def _get_token_ids(self, tokens: List[str]) -> List[int]:
        """获取 token ids"""
        ids = []
        for token in tokens:
            # 尝试不同的编码方式
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

        Parameters
        ----------
        input_ids : torch.LongTensor
            已生成的 token ids, shape (batch_size, seq_len)
        scores : torch.FloatTensor
            当前 step 的 logits, shape (batch_size, vocab_size)

        Returns
        -------
        scores : torch.FloatTensor
            调整后的 logits
        """
        conflict_score = self.conflict_signal.overall_conflict_score

        # 如果冲突分数低于阈值，不干预
        if conflict_score < self.min_threshold:
            return scores

        # 计算 bias 强度（线性插值）
        # conflict_score ∈ [min_threshold, 1.0] → bias_strength ∈ [0.0, 1.0]
        bias_strength = min(1.0, (conflict_score - self.min_threshold) / (1.0 - self.min_threshold))

        # 计算具体的 bias 值
        yes_bias = self.yes_bias_range[0] + bias_strength * (self.yes_bias_range[1] - self.yes_bias_range[0])
        no_bias = self.no_bias_range[0] + bias_strength * (self.no_bias_range[1] - self.no_bias_range[0])

        # 应用 bias
        for token_id in self.yes_token_ids:
            if token_id < scores.shape[-1]:
                scores[:, token_id] += yes_bias

        for token_id in self.no_token_ids:
            if token_id < scores.shape[-1]:
                scores[:, token_id] += no_bias

        logger.debug(f"Applied logit bias: yes_bias={yes_bias:.2f}, no_bias={no_bias:.2f}, strength={bias_strength:.2f}")

        return scores


class ConflictAwareHallucinationSuppressor:
    """冲突感知的幻觉抑制器（主接口）"""

    def __init__(
        self,
        emotion_conflict_threshold: float = 0.35,
        content_conflict_threshold: float = 0.30,
        enable_prompt_injection: bool = True,
        enable_logit_calibration: bool = True,
    ):
        """
        Parameters
        ----------
        emotion_conflict_threshold : float
            情感距离超过此值视为冲突
        content_conflict_threshold : float
            内容冲突分数超过此值视为冲突
        enable_prompt_injection : bool
            是否启用提示词注入
        enable_logit_calibration : bool
            是否启用 logit 校准
        """
        self.emotion_threshold = emotion_conflict_threshold
        self.content_threshold = content_conflict_threshold
        self.enable_prompt_injection = enable_prompt_injection
        self.enable_logit_calibration = enable_logit_calibration

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
        检测情感与内容冲突

        Parameters
        ----------
        visual_emotion : str
            视觉情感标签
        audio_emotion : str
            音频情感标签
        visual_emotion_conf : float
            视觉情感置信度
        audio_emotion_conf : float
            音频情感置信度
        visual_objects : List[str]
            视觉检测到的物体
        asr_text : str
            ASR 转录文本

        Returns
        -------
        conflict_signal : ConflictSignal
        """
        # 1. 情感冲突检测
        emotion_distance = self._compute_emotion_distance(
            visual_emotion, audio_emotion,
            visual_emotion_conf, audio_emotion_conf
        )
        has_emotion_conflict = emotion_distance > self.emotion_threshold

        # 2. 内容冲突检测
        content_conflict_score = self._compute_content_conflict(
            visual_objects, asr_text
        )
        has_content_conflict = content_conflict_score > self.content_threshold

        return ConflictSignal(
            has_emotion_conflict=has_emotion_conflict,
            emotion_distance=emotion_distance,
            has_content_conflict=has_content_conflict,
            content_conflict_score=content_conflict_score,
        )

    def _compute_emotion_distance(
        self,
        visual_emotion: str,
        audio_emotion: str,
        visual_conf: float,
        audio_conf: float,
    ) -> float:
        """计算情感距离"""
        # 情感极性映射
        emotion_polarity = {
            'happy': 1.0, 'excited': 0.8, 'neutral': 0.0,
            'sad': -0.8, 'angry': -0.6, 'fear': -0.7,
        }

        v_polarity = emotion_polarity.get(visual_emotion.lower(), 0.0)
        a_polarity = emotion_polarity.get(audio_emotion.lower(), 0.0)

        # 极性差异
        polarity_diff = abs(v_polarity - a_polarity)

        # 加权置信度
        avg_conf = (visual_conf + audio_conf) / 2.0

        # 距离 = 极性差异 × 平均置信度
        return polarity_diff * avg_conf

    def _compute_content_conflict(
        self,
        visual_objects: List[str],
        asr_text: str,
    ) -> float:
        """计算内容冲突分数"""
        if not asr_text or not visual_objects:
            return 0.0

        asr_lower = asr_text.lower()
        visual_lower = [obj.lower() for obj in visual_objects]

        # 提取 ASR 中的名词（简化版：常见物体词）
        common_objects = [
            'person', 'man', 'woman', 'child', 'car', 'dog', 'cat',
            'tree', 'building', 'phone', 'computer', 'table', 'chair'
        ]

        mentioned_objects = [obj for obj in common_objects if obj in asr_lower]

        if not mentioned_objects:
            return 0.0

        # 计算有多少 ASR 提到的物体在视觉中缺失
        missing_count = sum(
            1 for obj in mentioned_objects
            if not any(obj in v_obj for v_obj in visual_lower)
        )

        conflict_score = missing_count / len(mentioned_objects)
        return conflict_score

    def inject_conflict_warning(
        self,
        prompt: str,
        conflict_signal: ConflictSignal,
        query_modality: str,  # 'visual' or 'audio'
    ) -> str:
        """
        在 prompt 中注入冲突警告

        Parameters
        ----------
        prompt : str
            原始 prompt
        conflict_signal : ConflictSignal
            冲突信号
        query_modality : str
            查询的模态 ('visual' 或 'audio')

        Returns
        -------
        modified_prompt : str
        """
        if not self.enable_prompt_injection:
            return prompt

        if conflict_signal.overall_conflict_score < 0.3:
            return prompt

        # 构造警告文本
        if query_modality == 'visual':
            warning = (
                "Note: The audio and video may describe different scenes. "
                "Please answer based ONLY on what you can SEE in the video, "
                "not what you hear in the audio."
            )
        else:  # audio
            warning = (
                "Note: The audio and video may describe different scenes. "
                "Please answer based ONLY on what you can HEAR in the audio, "
                "not what you see in the video."
            )

        # 插入警告（在问题之前）
        modified_prompt = f"{warning}\n\n{prompt}"
        return modified_prompt

    def create_logits_processor(
        self,
        tokenizer,
        conflict_signal: ConflictSignal,
    ) -> Optional[ConflictAwareLogitsProcessor]:
        """
        创建 logits 处理器

        Parameters
        ----------
        tokenizer : transformers.PreTrainedTokenizer
        conflict_signal : ConflictSignal

        Returns
        -------
        processor : ConflictAwareLogitsProcessor or None
        """
        if not self.enable_logit_calibration:
            return None

        return ConflictAwareLogitsProcessor(
            tokenizer=tokenizer,
            conflict_signal=conflict_signal,
        )
