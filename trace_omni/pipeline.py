"""
Main orchestration pipeline for TRACe-Omni.
"""

from __future__ import annotations

import logging
import time
from typing import List, Optional

from ..config import PipelineConfig
from ..modules.modality_extractor import ModalityExtractor
from ..qwen_omni_adapter import QwenOmniAdapter
from .claim_extractor import ClaimDecomposer
from .config import TraceOmniConfig
from .parser import ResponsibilityParser
from .probing import build_dependency_probe
from .realizer import CommitmentControlledRealizer
from .scoring import ClaimResponsibilityScorer
from .types import Claim, TraceOmniResult
from .witnesses import TemporalWitnessGraphBuilder

logger = logging.getLogger(__name__)


class TraceOmniPipeline:
    """Standalone claim-level pipeline for Omni hallucination control."""

    def __init__(
        self,
        config: Optional[TraceOmniConfig] = None,
        *,
        adapter: Optional[QwenOmniAdapter] = None,
        extractor: Optional[ModalityExtractor] = None,
    ):
        self.config = config or TraceOmniConfig()
        self.adapter = adapter or QwenOmniAdapter(
            model_path=self.config.model_paths.qwen_omni_path,
            device=self.config.device,
        )
        self.extractor = extractor or ModalityExtractor(
            visual_emotion_model=self.config.model_paths.visual_emotion_model,
            audio_emotion_model=self.config.model_paths.audio_emotion_model,
            asr_model_size=self.config.model_paths.asr_model_size,
            clip_model=self.config.model_paths.clip_model,
            grounding_model=self.config.model_paths.grounding_model,
            grounding_fallback_model=self.config.model_paths.grounding_fallback_model,
            device=self.config.device,
        )
        self.parser = ResponsibilityParser()
        self.decomposer = ClaimDecomposer()
        self.graph_builder = TemporalWitnessGraphBuilder()
        self.dependency_probe = build_dependency_probe(self.config)
        self.scorer = ClaimResponsibilityScorer(self.config)
        self.realizer = CommitmentControlledRealizer(self.config)

    @classmethod
    def from_legacy_pipeline_config(cls, pipeline_config: Optional[PipelineConfig] = None) -> 'TraceOmniPipeline':
        return cls(config=TraceOmniConfig.from_pipeline_config(pipeline_config))

    def _collect_claims(self, question_spec, draft_answer: str) -> List[Claim]:
        if question_spec.form == 'yes_no':
            return [self.parser.build_binary_claim(question_spec)]
        if question_spec.form in {'single_choice', 'multi_select'}:
            return self.decomposer.option_claims(question_spec)
        return self.decomposer.decompose(
            draft_answer,
            question_spec,
            max_claims=self.config.realizer.open_ended_max_claims,
        )

    def run(
        self,
        video_path: str,
        question: str,
        *,
        max_new_tokens: Optional[int] = None,
    ) -> TraceOmniResult:
        return self.run_media(
            question,
            video_path=video_path,
            max_new_tokens=max_new_tokens,
        )

    def run_media(
        self,
        question: str,
        *,
        video_path: Optional[str] = None,
        audio_path: Optional[str] = None,
        max_new_tokens: Optional[int] = None,
    ) -> TraceOmniResult:
        t0 = time.time()
        question_spec = self.parser.parse(question, max_new_tokens=max_new_tokens)
        answer_tokens = max_new_tokens
        if answer_tokens is None:
            answer_tokens = 8 if question_spec.form == 'yes_no' else self.config.generation_max_new_tokens

        logger.info('TRACe-Omni: extract multimodal witnesses')
        feature_t0 = time.time()
        features = self.extractor.extract_media(
            video_path=video_path,
            audio_path=audio_path,
        )
        feature_time = time.time() - feature_t0
        witness_graph = self.graph_builder.build(features)

        logger.info('TRACe-Omni: generate draft answer')
        draft_t0 = time.time()
        draft_answer = self.adapter.answer_media(
            question,
            video_path=video_path,
            audio_path=audio_path,
            max_new_tokens=answer_tokens,
        )
        draft_time = time.time() - draft_t0

        claims = self._collect_claims(question_spec, draft_answer)
        logger.info('TRACe-Omni: score %d claims', len(claims))
        score_t0 = time.time()
        claim_scores = []
        for index, claim in enumerate(claims):
            dependency_evidence = None
            if self.config.probe.mode == 'branch_contrastive' and index < self.config.probe.branch_probe_max_claims:
                dependency_evidence = self.dependency_probe.probe(
                    video_path=video_path,
                    audio_path=audio_path,
                    question_spec=question_spec,
                    claim=claim,
                    witness_graph=witness_graph,
                    adapter=self.adapter,
                    config=self.config,
                )
            else:
                dependency_evidence = self.dependency_probe.probe(
                    video_path=video_path,
                    audio_path=audio_path,
                    question_spec=question_spec,
                    claim=claim,
                    witness_graph=witness_graph,
                    adapter=None,
                    config=self.config,
                )
            claim_scores.append(
                self.scorer.score_claim(
                    claim,
                    question_spec,
                    witness_graph,
                    dependency_evidence=dependency_evidence,
                )
            )
        score_time = time.time() - score_t0

        realized = self.realizer.realize(question_spec, draft_answer, claims, claim_scores)
        total_time = time.time() - t0
        canonical_path = video_path or audio_path or ''
        return TraceOmniResult(
            video_path=canonical_path,
            question=question,
            question_spec=question_spec,
            draft_answer=draft_answer,
            realized_answer=realized,
            claims=claims,
            claim_scores=claim_scores,
            witness_graph=witness_graph,
            feature_metadata={
                'feature_time_s': feature_time,
                'video_path': video_path,
                'audio_path': audio_path,
                'n_visual_witnesses': witness_graph.metadata.get('n_visual_nodes', 0),
                'n_audio_witnesses': witness_graph.metadata.get('n_audio_nodes', 0),
                'audio_type': getattr(features, 'audio_type', None),
                'visual_objects': list(getattr(features, 'visual_objects', []) or []),
                'audio_events': list(getattr(features, 'audio_events', []) or []),
            },
            runtime_metadata={
                'draft_time_s': draft_time,
                'score_time_s': score_time,
                'total_time_s': total_time,
                'probe_mode': self.config.probe.mode,
            },
        )
