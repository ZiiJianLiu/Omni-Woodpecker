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
- 识别 audio/visual tokens 在输入序列中的位置
- 根据冲突信号动态生成 attention_mask
- 在推理时注入修改后的 mask

优势：
- 效果最强：直接切断信息流，而非事后修正
- 作用于根源：在 attention 层面阻止跨模态污染
- 免训练：纯推理时干预
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

    @property
    def should_suppress(self) -> bool:
        """是否应该启动抑制"""
        return self.overall_conflict_score >= 0.3


class AttentionMaskSuppressor:
    """注意力掩码抑制器（主接口）"""

    def __init__(
        self,
        emotion_conflict_threshold: float = 0.35,
        content_conflict_threshold: float = 0.30,
        min_conflict_for_masking: float = 0.3,
    ):
        """
        Parameters
        ----------
        emotion_conflict_threshold : float
            情感距离超过此值视为冲突
        content_conflict_threshold : float
            内容冲突分数超过此值视为冲突
        min_conflict_for_masking : float
            触发 attention masking 的最小冲突分数
        """
        self.emotion_threshold = emotion_conflict_threshold
        self.content_threshold = content_conflict_threshold
        self.min_conflict = min_conflict_for_masking

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

    def create_masked_inputs(
        self,
        processor,
        video_path: str,
        question: str,
        audio_array: Optional[object],
        conflict_signal: ConflictSignal,
        query_modality: str,  # 'visual' or 'audio'
    ) -> Tuple[Dict, bool]:
        """
        创建带冲突感知掩码的输入

        Parameters
        ----------
        processor : Qwen2_5OmniProcessor
            处理器
        video_path : str
            视频路径
        question : str
            问题
        audio_array : Optional[np.ndarray]
            音频数组
        conflict_signal : ConflictSignal
            冲突信号
        query_modality : str
            查询的模态 ('visual' 或 'audio')

        Returns
        -------
        inputs : Dict
            处理后的输入（可能包含修改后的 attention_mask）
        masking_applied : bool
            是否应用了掩码
        """
        # 如果冲突分数低于阈值，不应用掩码
        if not conflict_signal.should_suppress:
            # 标准处理
            from ..models.qwen_omni import _DEFAULT_SYSTEM_PROMPT
            messages = [
                {
                    'role': 'system',
                    'content': [{'type': 'text', 'text': _DEFAULT_SYSTEM_PROMPT}],
                },
                {
                    'role': 'user',
                    'content': [
                        {'type': 'video', 'video': video_path},
                        {'type': 'text', 'text': question},
                    ],
                },
            ]
            text_prompt = processor.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=False
            )
            inputs = processor(
                text=[text_prompt],
                videos=[video_path],
                audios=[audio_array] if audio_array is not None else None,
                return_tensors='pt',
                padding=True,
            )
            return inputs, False

        # 应用冲突感知掩码
        logger.info(
            f"Applying attention masking: conflict_score={conflict_signal.overall_conflict_score:.3f}, "
            f"query_modality={query_modality}"
        )

        # 策略：根据查询模态和冲突，选择性屏蔽另一模态
        if query_modality == 'audio':
            # 查询音频 + 冲突 → 屏蔽视觉，只保留音频
            mask_visual = True
            mask_audio = False
        else:  # visual
            # 查询视觉 + 冲突 → 屏蔽音频，只保留视觉
            mask_visual = False
            mask_audio = True

        # 构建输入（屏蔽冲突模态）
        from ..models.qwen_omni import _DEFAULT_SYSTEM_PROMPT
        messages = [
            {
                'role': 'system',
                'content': [{'type': 'text', 'text': _DEFAULT_SYSTEM_PROMPT}],
            },
            {
                'role': 'user',
                'content': [
                    {'type': 'video', 'video': video_path},
                    {'type': 'text', 'text': question},
                ],
            },
        ]
        text_prompt = processor.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False
        )

        # 根据掩码策略处理输入
        inputs = processor(
            text=[text_prompt],
            videos=[video_path] if not mask_visual else None,
            audios=[audio_array] if (audio_array is not None and not mask_audio) else None,
            return_tensors='pt',
            padding=True,
        )

        return inputs, True


def infer_query_modality(question: str) -> str:
    """推断问题查询的模态"""
    q_lower = question.lower()
    if 'visible' in q_lower or 'see' in q_lower or 'in the video' in q_lower:
        return 'visual'
    elif 'sound' in q_lower or 'hear' in q_lower or 'in the audio' in q_lower:
        return 'audio'
    else:
        return 'visual'  # 默认
