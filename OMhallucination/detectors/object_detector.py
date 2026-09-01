"""
Object Detector (Conservative inventory + query-conditioned grounding)
=====================================================================
仅将 grounding 用于问题条件化的实体存在性验证。
通用 visual object inventory 保持保守的 CLIP 多帧一致性策略，
避免 open-vocabulary grounding 在全类别扫图时产生系统性假阳性。
"""
import logging
import os
import re
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

logger = logging.getLogger(__name__)


class ObjectDetector:
    """物体检测器。"""

    def __init__(
        self,
        model_name: str = "openai/clip-vit-large-patch14",
        device: str = "cuda",
        grounding_model_name: Optional[str] = "IDEA-Research/grounding-dino-tiny",
        grounding_fallback_model_name: Optional[str] = "google/owlv2-base-patch16-ensemble",
        local_files_only: bool = False,
    ):
        self.device = device
        self.model_name = model_name
        self.grounding_model_name = grounding_model_name
        self.grounding_fallback_model_name = grounding_fallback_model_name
        self.local_files_only = bool(
            local_files_only
            or os.environ.get("HF_HUB_OFFLINE") == "1"
            or os.environ.get("TRANSFORMERS_OFFLINE") == "1"
        )

        # Backward-compatible CLIP handles used by scene classification and verifier.
        self._model = None
        self._processor = None

        # Stronger grounding backend used for object presence and text-conditioned detection.
        self._grounding = None
        self._grounding_backend = 'none'
        self._grounding_model = None

        self._load_clip_model()
        self._load_grounding_model()

        self.object_categories = [
            "person", "car", "dog", "cat", "tree", "building", "chair", "table",
            "phone", "computer", "book", "bottle", "cup", "food", "flower",
            "bird", "bicycle", "motorcycle", "airplane", "boat", "traffic light",
            "fire hydrant", "stop sign", "parking meter", "bench", "elephant",
            "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag",
            "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
            "kite", "baseball bat", "skateboard", "surfboard", "tennis racket",
        ]
        self._object_prompts = [f"a photo of a {obj}" for obj in self.object_categories]

    def _pipeline_device(self) -> int:
        if not self.device or self.device == 'cpu':
            return -1
        if self.device.startswith('cuda:'):
            try:
                return int(self.device.split(':', 1)[1])
            except (IndexError, ValueError):
                return 0
        if self.device.startswith('cuda'):
            return 0
        return -1

    def _load_clip_model(self):
        """加载 CLIP 模型，供全局相似度与场景分类 fallback 使用。"""
        try:
            from transformers import CLIPModel, CLIPProcessor

            logger.info("加载 CLIP 模型: %s", self.model_name)
            self._processor = CLIPProcessor.from_pretrained(
                self.model_name,
                local_files_only=self.local_files_only,
            )
            self._model = CLIPModel.from_pretrained(
                self.model_name,
                local_files_only=self.local_files_only,
            )
            self._model.to(self.device)
            self._model.eval()
            logger.info("CLIP 模型加载完成")

        except Exception as e:
            logger.error("CLIP 模型加载失败: %s", e)
            raise

    def _load_grounding_model(self):
        """加载强视觉 grounding 模型；若失败则保留 CLIP fallback。"""
        candidates: List[str] = []
        for model_name in [self.grounding_model_name, self.grounding_fallback_model_name]:
            if model_name and model_name not in candidates:
                candidates.append(model_name)

        if not candidates:
            logger.warning("未配置 grounding 模型，将仅使用 CLIP fallback")
            return

        last_error = None
        try:
            from transformers import pipeline as hf_pipeline
        except Exception as exc:
            logger.warning("无法导入 zero-shot-object-detection pipeline，将仅使用 CLIP fallback: %s", exc)
            return

        device_index = self._pipeline_device()
        for model_name in candidates:
            try:
                logger.info("加载视觉 grounding 模型: %s", model_name)
                self._grounding = hf_pipeline(
                    task='zero-shot-object-detection',
                    model=model_name,
                    device=device_index,
                    model_kwargs={'local_files_only': self.local_files_only},
                )
                self._grounding_backend = 'zero-shot-object-detection'
                self._grounding_model = model_name
                logger.info("视觉 grounding 模型加载完成: %s", model_name)
                return
            except Exception as exc:
                last_error = exc
                logger.warning("视觉 grounding 模型加载失败 %s: %s", model_name, exc)

        logger.warning("所有 grounding 模型均不可用，将回退到 CLIP 全局相似度: %s", last_error)

    @staticmethod
    def _to_pil(frame: np.ndarray) -> Image.Image:
        if isinstance(frame, Image.Image):
            return frame
        return Image.fromarray(frame.astype(np.uint8))

    @staticmethod
    def _normalize_phrase(text: Optional[str]) -> str:
        text = re.sub(r"[^a-z0-9']+", ' ', (text or '').lower())
        return ' '.join(text.split())

    def _canonical_grounding_label(self, text: Optional[str]) -> str:
        label = self._normalize_phrase(text)
        if not label:
            return ''

        prefixes = (
            'a photo of ',
            'a photo of a ',
            'a video frame containing ',
            'a video frame of ',
            'a scene with ',
            'there is ',
            'there are ',
        )
        suffixes = (
            ' in the scene',
            ' in the video',
            ' visible in the video',
            ' visible in the scene',
        )
        for prefix in prefixes:
            if label.startswith(prefix):
                label = label[len(prefix):].strip()
                break
        for suffix in suffixes:
            if label.endswith(suffix):
                label = label[:-len(suffix)].strip()
                break
        label = re.sub(r'^(?:a|an|the)\s+', '', label)
        return label.strip()

    def _compute_clip_frame_text_similarity(
        self,
        frame: np.ndarray,
        prompts: List[str],
    ) -> torch.Tensor:
        """使用 CLIP 计算单帧与文本提示词的余弦相似度。"""
        image = self._to_pil(frame)
        inputs = self._processor(
            text=prompts,
            images=image,
            return_tensors="pt",
            padding=True,
            truncation=True,
        )
        inputs = {key: value.to(self.device) for key, value in inputs.items()}

        with torch.no_grad():
            outputs = self._model(**inputs)

        image_embeds = getattr(outputs, 'image_embeds', None)
        text_embeds = getattr(outputs, 'text_embeds', None)
        if isinstance(image_embeds, torch.Tensor) and isinstance(text_embeds, torch.Tensor):
            image_embeds = F.normalize(image_embeds.float(), dim=-1)
            text_embeds = F.normalize(text_embeds.float(), dim=-1)
            return image_embeds @ text_embeds.T

        logits_per_image = getattr(outputs, 'logits_per_image', None)
        if isinstance(logits_per_image, torch.Tensor):
            logit_scale = getattr(self._model, 'logit_scale', None)
            if isinstance(logit_scale, torch.Tensor):
                scale = logit_scale.exp().to(logits_per_image.device)
                if torch.isfinite(scale).all() and torch.all(scale > 0):
                    return logits_per_image / scale
            return logits_per_image

        raise TypeError(f"Unsupported CLIP output type: {type(outputs).__name__}")

    def _run_grounding(
        self,
        frame: np.ndarray,
        candidate_labels: List[str],
        *,
        threshold: float,
        top_k: int,
    ) -> List[Dict[str, object]]:
        if self._grounding is None or not candidate_labels:
            return []

        labels = []
        seen = set()
        for label in candidate_labels:
            normalized = self._canonical_grounding_label(label)
            if not normalized or normalized in seen:
                continue
            labels.append(normalized)
            seen.add(normalized)
        if not labels:
            return []

        image = self._to_pil(frame)
        try:
            try:
                outputs = self._grounding(
                    image,
                    candidate_labels=labels,
                    threshold=threshold,
                )
            except TypeError:
                outputs = self._grounding(image, candidate_labels=labels)
        except Exception as exc:
            logger.error("视觉 grounding 失败: %s", exc)
            return []

        if isinstance(outputs, dict):
            outputs = outputs.get('detections') or outputs.get('outputs') or [outputs]

        detections: List[Dict[str, object]] = []
        for item in outputs or []:
            label = self._canonical_grounding_label(
                item.get('label') or item.get('text') or item.get('phrase')
            )
            score = float(item.get('score', 0.0) or 0.0)
            if not label or score < threshold:
                continue
            detections.append(
                {
                    'label': label,
                    'score': score,
                    'box': item.get('box') or item.get('boxes'),
                }
            )

        detections.sort(key=lambda item: item['score'], reverse=True)
        return detections[:top_k]

    def _aggregate_grounding_scores(
        self,
        frames: List[np.ndarray],
        prompts: List[str],
        *,
        threshold: float,
        aggregate: str,
        max_frames: int,
    ) -> Dict[str, float]:
        if self._grounding is None or not frames or not prompts:
            return {}

        prompt_to_label: Dict[str, str] = {}
        labels: List[str] = []
        seen = set()
        for prompt in prompts:
            label = self._canonical_grounding_label(prompt)
            if not label:
                continue
            prompt_to_label[prompt] = label
            if label not in seen:
                labels.append(label)
                seen.add(label)

        if not labels:
            return {}

        frame_scores: Dict[str, List[float]] = {prompt: [] for prompt in prompts}
        for frame in frames[:max_frames]:
            detections = self._run_grounding(
                frame,
                labels,
                threshold=threshold,
                top_k=max(len(labels) * 2, 8),
            )
            best_by_label: Dict[str, float] = {}
            for detection in detections:
                best_by_label[detection['label']] = max(
                    best_by_label.get(detection['label'], 0.0),
                    float(detection['score']),
                )
            for prompt in prompts:
                frame_scores[prompt].append(best_by_label.get(prompt_to_label.get(prompt, ''), 0.0))

        result: Dict[str, float] = {}
        for prompt, scores in frame_scores.items():
            if not scores:
                continue
            if aggregate == 'mean':
                result[prompt] = float(np.mean(scores))
            else:
                result[prompt] = float(np.max(scores))
        return result

    def _score_entity_presence_with_grounding(
        self,
        frames: List[np.ndarray],
        variants: List[str],
    ) -> Dict[str, object]:
        """仅对小候选集实体做保守 grounding，避免全局扫图式假阳性。"""
        if self._grounding is None or not frames or not variants:
            return {
                'scores': {},
                'support_counts': {},
                'peak_scores': {},
                'support_frame_indices': {},
                'peak_frame_indices': {},
                'max_frames': 0,
            }

        detection_threshold = 0.28
        max_frames = min(len(frames), 6)
        min_support = 1 if max_frames <= 2 else 2
        support_scores: Dict[str, List[float]] = {variant: [] for variant in variants}
        support_frame_indices: Dict[str, List[int]] = {variant: [] for variant in variants}
        peak_scores: Dict[str, float] = {variant: 0.0 for variant in variants}
        peak_frame_indices: Dict[str, int] = {variant: -1 for variant in variants}

        for frame_idx, frame in enumerate(frames[:max_frames]):
            detections = self._run_grounding(
                frame,
                variants,
                threshold=detection_threshold,
                top_k=max(len(variants) * 2, 6),
            )
            per_frame_best: Dict[str, float] = {}
            for detection in detections:
                label = detection['label']
                per_frame_best[label] = max(
                    per_frame_best.get(label, 0.0),
                    float(detection['score']),
                )

            for variant in variants:
                score = float(per_frame_best.get(variant, 0.0))
                if score > peak_scores[variant]:
                    peak_scores[variant] = score
                    peak_frame_indices[variant] = frame_idx
                if score >= detection_threshold:
                    support_scores[variant].append(score)
                    support_frame_indices[variant].append(frame_idx)

        aggregated_scores: Dict[str, float] = {}
        support_counts: Dict[str, int] = {}
        accepted_frame_indices: Dict[str, List[int]] = {}
        for variant, scores in support_scores.items():
            if len(scores) >= min_support:
                aggregated_scores[variant] = float(np.mean(scores))
                support_counts[variant] = len(scores)
                accepted_frame_indices[variant] = list(support_frame_indices.get(variant, []))
            elif len(scores) == 1 and max_frames <= 2 and scores[0] >= detection_threshold + 0.12:
                aggregated_scores[variant] = float(scores[0])
                support_counts[variant] = 1
                accepted_frame_indices[variant] = list(support_frame_indices.get(variant, []))

        return {
            'scores': aggregated_scores,
            'support_counts': support_counts,
            'peak_scores': peak_scores,
            'support_frame_indices': accepted_frame_indices,
            'peak_frame_indices': peak_frame_indices,
            'max_frames': max_frames,
        }

    def _detect_objects_with_clip(
        self,
        frames: List[np.ndarray],
        threshold: float = 0.32,
        min_votes: int = 2,
        top_k: int = 5,
        frame_top_k: int = 2,
    ) -> List[str]:
        if not frames:
            return []

        max_frames = min(len(frames), 8)
        support_scores: Dict[int, List[float]] = {}

        for frame in frames[:max_frames]:
            sims = self._compute_clip_frame_text_similarity(frame, self._object_prompts)[0]
            probs = torch.softmax(sims, dim=0)
            probs_np = probs.detach().float().cpu().numpy()
            top_indices = probs_np.argsort()[-frame_top_k:][::-1]
            for idx in top_indices:
                score = float(probs_np[idx])
                if score >= threshold:
                    support_scores.setdefault(int(idx), []).append(score)

        min_support = 1 if max_frames <= 3 else min_votes
        candidates = []
        for idx, scores in support_scores.items():
            mean_score = float(np.mean(scores))
            if len(scores) < min_support or mean_score < threshold:
                continue
            candidates.append(
                (
                    self.object_categories[idx],
                    len(scores),
                    mean_score,
                )
            )

        candidates.sort(key=lambda item: (item[1], item[2]), reverse=True)

        return [name for name, _, _ in candidates[:top_k]]

    def detect_objects(
        self,
        frames: List[np.ndarray],
        threshold: float = 0.28,
        min_votes: int = 2,
        top_k: int = 5,
        frame_top_k: int = 3,
    ) -> List[str]:
        """生成保守的通用 visual object inventory。"""
        if not frames:
            return []

        try:
            detected_objects = self._detect_objects_with_clip(
                frames,
                threshold=max(float(threshold), 0.32),
                min_votes=max(int(min_votes), 2),
                top_k=min(int(top_k), 4),
                frame_top_k=min(int(frame_top_k), 2),
            )
            logger.debug("保守 CLIP inventory 检测到的物体: %s", detected_objects)
            return detected_objects

        except Exception as e:
            logger.error("物体检测失败: %s", e)
            return []

    def score_text_prompts(
        self,
        frames: List[np.ndarray],
        prompts: List[str],
        *,
        aggregate: str = 'max',
    ) -> Dict[str, float]:
        """对任意文本提示词做保守视觉打分。这里固定使用 CLIP。"""
        if not frames or not prompts:
            return {}

        try:
            frame_scores: Dict[str, List[float]] = {prompt: [] for prompt in prompts}
            for frame in frames[:8]:
                sims = self._compute_clip_frame_text_similarity(frame, prompts)[0]
                sims = ((sims + 1.0) / 2.0).clamp(0.0, 1.0)
                sims_np = sims.detach().float().cpu().numpy()
                for prompt, score in zip(prompts, sims_np):
                    frame_scores[prompt].append(float(score))

            result: Dict[str, float] = {}
            for prompt, scores in frame_scores.items():
                if not scores:
                    continue
                if aggregate == 'mean':
                    result[prompt] = float(np.mean(scores))
                else:
                    result[prompt] = float(np.max(scores))
            return result
        except Exception as e:
            logger.error("视觉 prompt 打分失败: %s", e)
            return {}

    def detect_asr_entities_in_frames(
        self,
        frames: List[np.ndarray],
        asr_text: str,
        detection_threshold: float = 0.22,
        max_frames: int = 6,
    ) -> Dict[str, float]:
        """
        用 Grounding DINO 检测 ASR 文本中提及的实体在视频帧中是否存在。

        与 detect_objects 不同，这里是 ASR 驱动：
        先从 ASR 文本中提取名词实体，再用 Grounding DINO 逐帧检测，
        返回每个实体的最高置信度（0.0 = 未检测到）。

        Parameters
        ----------
        frames       : 视频帧列表
        asr_text     : ASR 转录文本
        detection_threshold : 单帧检测置信度阈值
        max_frames   : 最多检测的帧数

        Returns
        -------
        {entity: max_confidence}  —— 若 Grounding DINO 不可用则返回 {}
        """
        if self._grounding is None or not frames or not asr_text:
            return {}

        entities = self._extract_entities_from_text(asr_text)
        if not entities:
            return {}

        entity_max: Dict[str, float] = {e: 0.0 for e in entities}
        for frame in frames[:max_frames]:
            detections = self._run_grounding(
                frame,
                entities,
                threshold=detection_threshold,
                top_k=len(entities) * 2,
            )
            for det in detections:
                label = det['label']
                score = float(det['score'])
                # 匹配回原始实体（grounding 可能会截断词尾）
                for entity in entities:
                    if entity in label or label in entity:
                        if score > entity_max[entity]:
                            entity_max[entity] = score

        logger.debug(
            "ASR 实体 Grounding DINO 检测结果: %s",
            {k: f"{v:.3f}" for k, v in entity_max.items() if v > 0},
        )
        return entity_max

    @staticmethod
    def _extract_entities_from_text(text: str) -> List[str]:
        """
        从文本中提取可检测的名词实体（不依赖 NLP 库，基于词表匹配）。
        只保留 Grounding DINO 能可靠检测的具体可视物体，
        排除抽象概念、自然现象、场景词（fire/water/sky/room 等难以定位）。
        """
        text_lower = text.lower()
        # 只保留 Grounding DINO 擅长的具体有界物体
        ENTITY_VOCAB = [
            # 人物
            "person", "man", "woman", "child", "baby", "boy", "girl",
            # 交通工具（有明确外形）
            "car", "truck", "bus", "motorcycle", "bicycle", "bike",
            "train", "airplane", "plane", "boat", "ship",
            # 动物
            "dog", "cat", "bird", "horse", "cow", "sheep", "elephant",
            "bear", "rabbit", "lion", "tiger",
            # 电子产品
            "phone", "computer", "laptop", "keyboard", "camera", "television", "tv",
            # 食物（有明确外形）
            "cake", "bread", "pizza", "sandwich", "donut",
            # 家具/物品
            "chair", "table", "desk", "bed", "sofa",
            "bottle", "cup", "bag", "ball", "book",
            # 服饰
            "hat", "umbrella",
        ]
        found = []
        seen = set()
        for entity in ENTITY_VOCAB:
            # 词边界匹配
            pattern = r'\b' + re.escape(entity) + r'\b'
            if re.search(pattern, text_lower):
                canonical = entity
                if canonical not in seen:
                    found.append(canonical)
                    seen.add(canonical)
        return found

    def score_entity_presence(
        self,
        frames: List[np.ndarray],
        entity: str,
        *,
        prompt_templates: Optional[List[str]] = None,
        aliases: Optional[List[str]] = None,
    ) -> Dict[str, object]:
        """对任意查询实体执行问题条件化视觉存在性打分。"""
        entity = (entity or '').strip()
        if not entity:
            return {
                'score': 0.0,
                'best_prompt': None,
                'best_alias': None,
                'prompt_scores': {},
                'clip_peak_prompt_scores': {},
                'grounding_scores': {},
                'grounding_score': 0.0,
                'grounding_support_count': 0,
                'grounding_peak_score': 0.0,
                'clip_prompt_scores': {},
                'clip_score': 0.0,
                'clip_peak_score': 0.0,
            }

        variants = []
        for variant in [entity] + list(aliases or []):
            variant = self._canonical_grounding_label(variant)
            if not variant or variant in variants:
                continue
            variants.append(variant)

        templates = prompt_templates or [
            'a photo of {entity}',
            'a video frame containing {entity}',
            'there is {entity} in the scene',
            'a scene with {entity}',
            '{entity}',
        ]
        prompts = [template.format(entity=variant) for variant in variants[:8] for template in templates]

        clip_prompt_scores = self.score_text_prompts(frames, prompts, aggregate='mean')
        clip_peak_prompt_scores = self.score_text_prompts(frames, prompts, aggregate='max')
        clip_best_prompt = None
        clip_best_score = 0.0
        clip_best_alias = entity
        clip_peak_prompt = None
        clip_peak_score = 0.0
        clip_peak_alias = entity
        if clip_prompt_scores:
            clip_best_prompt, clip_best_score = max(clip_prompt_scores.items(), key=lambda item: item[1])
            for variant in variants:
                if variant and variant in clip_best_prompt:
                    clip_best_alias = variant
                    break
        if clip_peak_prompt_scores:
            clip_peak_prompt, clip_peak_score = max(
                clip_peak_prompt_scores.items(),
                key=lambda item: item[1],
            )
            for variant in variants:
                if variant and variant in clip_peak_prompt:
                    clip_peak_alias = variant
                    break

        grounding_scores: Dict[str, float] = {}
        support_counts: Dict[str, int] = {}
        peak_scores: Dict[str, float] = {}
        support_frame_indices: Dict[str, List[int]] = {}
        peak_frame_indices: Dict[str, int] = {}
        grounding_max_frames = 0
        grounding_best_alias = None
        grounding_best_score = 0.0
        grounding_peak_alias = None
        grounding_peak_score = 0.0

        if self._grounding is not None and frames and variants:
            grounding_info = self._score_entity_presence_with_grounding(frames, variants)
            grounding_scores = grounding_info.get('scores', {})
            support_counts = grounding_info.get('support_counts', {})
            peak_scores = grounding_info.get('peak_scores', {})
            support_frame_indices = grounding_info.get('support_frame_indices', {})
            peak_frame_indices = grounding_info.get('peak_frame_indices', {})
            grounding_max_frames = int(grounding_info.get('max_frames', 0) or 0)
            if grounding_scores and any(score > 0.0 for score in grounding_scores.values()):
                grounding_best_alias, grounding_best_score = max(
                    grounding_scores.items(),
                    key=lambda item: item[1],
                )
            if peak_scores and any(score > 0.0 for score in peak_scores.values()):
                grounding_peak_alias, grounding_peak_score = max(
                    peak_scores.items(),
                    key=lambda item: item[1],
                )

        best_alias = grounding_best_alias or clip_best_alias
        best_prompt = grounding_best_alias or clip_best_prompt
        composite_score = 0.0
        backend = 'none'
        clip_hybrid_score = max(
            float(clip_best_score),
            min(0.86, 0.58 * float(clip_best_score) + 0.42 * float(clip_peak_score)),
        )

        if grounding_best_score > 0.0:
            composite_score = grounding_best_score
            backend = 'grounding'
            if clip_hybrid_score > 0.0 and (
                not grounding_best_alias
                or clip_best_alias == grounding_best_alias
                or clip_peak_alias == grounding_best_alias
            ):
                composite_score = max(
                    composite_score,
                    min(0.95, 0.70 * grounding_best_score + 0.30 * clip_hybrid_score),
                )
                backend = 'hybrid'
        elif grounding_peak_score > 0.0 and clip_hybrid_score > 0.0:
            composite_score = min(0.86, 0.50 * grounding_peak_score + 0.50 * clip_hybrid_score)
            best_alias = grounding_peak_alias or clip_peak_alias or clip_best_alias
            best_prompt = grounding_peak_alias or clip_peak_prompt or clip_best_prompt
            backend = 'hybrid_peak'
        elif clip_hybrid_score > 0.0:
            composite_score = clip_hybrid_score
            best_alias = clip_peak_alias if clip_peak_score >= clip_best_score else clip_best_alias
            best_prompt = clip_peak_prompt if clip_peak_score >= clip_best_score else clip_best_prompt
            backend = 'clip_fallback_peak' if clip_peak_score > clip_best_score else 'clip_fallback'

        selected_grounding_alias = grounding_best_alias or grounding_peak_alias or ''
        return {
            'score': float(composite_score),
            'best_prompt': best_prompt,
            'best_alias': best_alias,
            'prompt_scores': grounding_scores or clip_prompt_scores,
            'clip_peak_prompt_scores': clip_peak_prompt_scores,
            'grounding_scores': grounding_scores,
            'grounding_score': float(grounding_best_score),
            'grounding_support_count': int(
                support_counts.get(selected_grounding_alias, 0)
            ),
            'grounding_peak_score': float(grounding_peak_score),
            'grounding_peak_alias': grounding_peak_alias,
            'grounding_support_frame_indices': list(
                support_frame_indices.get(selected_grounding_alias, [])
            ),
            'grounding_peak_frame_index': int(
                peak_frame_indices.get(selected_grounding_alias, -1)
            ),
            'grounding_max_frames': int(grounding_max_frames),
            'support_counts': support_counts,
            'peak_scores': peak_scores,
            'clip_prompt_scores': clip_prompt_scores,
            'clip_score': float(clip_best_score),
            'clip_peak_score': float(clip_peak_score),
            'clip_best_prompt': clip_best_prompt,
            'clip_peak_prompt': clip_peak_prompt,
            'backend': backend,
        }

    def compute_video_text_similarity(
        self,
        frames: List[np.ndarray],
        text: str,
    ) -> float:
        """计算视频帧与文本的 CLIP 相似度。"""
        if not frames or not text:
            return 0.0
        scores: List[float] = []
        for frame in frames[:8]:
            sims = self._compute_clip_frame_text_similarity(frame, [text])[0]
            sim = float(((sims + 1.0) / 2.0).clamp(0.0, 1.0)[0].item())
            scores.append(sim)
        similarity = float(np.mean(scores)) if scores else 0.0
        logger.debug("视频-文本相似度: %.3f", similarity)
        return similarity
