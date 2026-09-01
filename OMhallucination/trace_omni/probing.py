"""
Dependency probes for claim-level responsibility estimation.
"""

from __future__ import annotations

import math
import re
from typing import Dict, Optional

from .config import TraceOmniConfig
from .types import Claim, DependencyEvidence, QuerySpec, WitnessGraph


class BaseDependencyProbe:
    def probe(
        self,
        *,
        video_path: Optional[str],
        audio_path: Optional[str] = None,
        question_spec: QuerySpec,
        claim: Claim,
        witness_graph: WitnessGraph,
        adapter=None,
        config: Optional[TraceOmniConfig] = None,
    ) -> DependencyEvidence:
        raise NotImplementedError


class HeuristicDependencyProbe(BaseDependencyProbe):
    """A lightweight default probe that does not require extra model calls."""

    def probe(
        self,
        *,
        video_path: Optional[str],
        audio_path: Optional[str] = None,
        question_spec: QuerySpec,
        claim: Claim,
        witness_graph: WitnessGraph,
        adapter=None,
        config: Optional[TraceOmniConfig] = None,
    ) -> DependencyEvidence:
        del video_path, audio_path, adapter, question_spec
        audio_signal = 1.0 if any(node.modality == 'audio' for node in witness_graph.nodes) else 0.0
        visual_signal = 1.0 if any(node.modality == 'visual' for node in witness_graph.nodes) else 0.0

        if claim.primary_modality == 'audio':
            audio_dependency = 0.82 * audio_signal
            visual_dependency = 0.28 * visual_signal
        elif claim.primary_modality == 'visual':
            audio_dependency = 0.28 * audio_signal
            visual_dependency = 0.82 * visual_signal
        else:
            audio_dependency = 0.62 * audio_signal
            visual_dependency = 0.62 * visual_signal

        if claim.predicate == 'temporal_order':
            audio_dependency = min(1.0, audio_dependency + 0.10 * audio_signal)
            visual_dependency = min(1.0, visual_dependency + 0.10 * visual_signal)

        full_support = min(1.0, max(audio_dependency, visual_dependency))
        return DependencyEvidence(
            full_support=full_support,
            no_audio_support=max(0.0, full_support - audio_dependency),
            no_visual_support=max(0.0, full_support - visual_dependency),
            audio_dependency=audio_dependency,
            visual_dependency=visual_dependency,
            method='heuristic',
            notes=[f'predicate={claim.predicate}', f'primary={claim.primary_modality}'],
        )


class BranchContrastiveDependencyProbe(BaseDependencyProbe):
    """Branch contrastive clause probe using teacher-forced continuation scoring."""

    @staticmethod
    def _clip01(value: float) -> float:
        return max(0.0, min(1.0, float(value)))

    @staticmethod
    def _build_probe_prompt(question_spec: QuerySpec, claim: Claim, prompt_style: str = 'responsibility_clause') -> str:
        if prompt_style != 'responsibility_clause':
            return (
                f'Question: {question_spec.raw_question}\n'
                'Answer with one short clause only.'
            )

        if claim.predicate == 'visibility':
            task = 'Output one short clause that is directly visible in the scene. Do not infer sounds or causes.'
        elif claim.predicate == 'sound_source':
            task = 'Output one short clause that is directly audible. Do not infer visibility from audio alone.'
        elif claim.predicate == 'speech_content':
            task = 'Output one short clause that is directly supported by spoken audio. Do not infer unseen content.'
        elif claim.predicate == 'temporal_order':
            task = 'Output one short clause about the timing relation only. Do not force simultaneity unless it is truly supported.'
        elif claim.predicate == 'cross_modal_consistency':
            task = 'Output one short clause about whether audio and video match in content.'
        elif claim.predicate == 'emotion':
            task = 'Output one short clause about tone or emotion only.'
        else:
            task = 'Output one short clause that is directly supported by the multimodal content.'
        return (
            'You are producing a single evidence-grounded clause.\n'
            f'Original question: {question_spec.raw_question}\n'
            f'Task: {task}\n'
            'Return only the clause, with no explanation.'
        )

    @staticmethod
    def _softmax(scores: Dict[str, float], temperature: float) -> Dict[str, float]:
        if not scores:
            return {}
        temperature = max(1e-6, float(temperature))
        peak = max(scores.values())
        exp_scores = {
            key: math.exp((float(value) - peak) / temperature)
            for key, value in scores.items()
        }
        total = sum(exp_scores.values()) or 1.0
        return {
            key: float(value / total)
            for key, value in exp_scores.items()
        }

    def probe(
        self,
        *,
        video_path: Optional[str],
        audio_path: Optional[str] = None,
        question_spec: QuerySpec,
        claim: Claim,
        witness_graph: WitnessGraph,
        adapter=None,
        config: Optional[TraceOmniConfig] = None,
    ) -> DependencyEvidence:
        if adapter is None:
            return HeuristicDependencyProbe().probe(
                video_path=video_path,
                audio_path=audio_path,
                question_spec=question_spec,
                claim=claim,
                witness_graph=witness_graph,
                adapter=None,
                config=config,
            )

        probe_cfg = (config.probe if config is not None else None)
        prompt = self._build_probe_prompt(
            question_spec,
            claim,
            prompt_style=str(getattr(probe_cfg, 'branch_probe_prompt_style', 'responsibility_clause') or 'responsibility_clause'),
        )
        branch_scores = adapter.score_text_continuation_media_branches(
            prompt,
            continuation=claim.text,
            video_path=video_path,
            audio_path=audio_path,
            include_no_visual=bool(getattr(probe_cfg, 'branch_probe_use_masks', True)),
            use_probe_budget=True,
        )
        mean_logprobs = {
            branch: float(payload.get('mean_logprob', 0.0) or 0.0)
            for branch, payload in branch_scores.items()
        }
        relative_support = self._softmax(
            mean_logprobs,
            temperature=float(getattr(probe_cfg, 'branch_probe_softmax_temperature', 0.35) or 0.35),
        )
        delta_scale = float(getattr(probe_cfg, 'branch_probe_delta_scale', 0.25) or 0.25)
        full_mean = mean_logprobs.get('full', 0.0)
        no_audio_mean = mean_logprobs.get('no_audio', full_mean)
        no_visual_mean = mean_logprobs.get('no_visual', full_mean)
        audio_dependency = self._clip01(max(0.0, full_mean - no_audio_mean) / max(1e-6, delta_scale))
        visual_dependency = self._clip01(max(0.0, full_mean - no_visual_mean) / max(1e-6, delta_scale))
        token_count = int((branch_scores.get('full') or {}).get('token_count', 0) or 0)
        return DependencyEvidence(
            full_support=float(relative_support.get('full', 0.0)),
            no_audio_support=float(relative_support.get('no_audio', 0.0)),
            no_visual_support=float(relative_support.get('no_visual', 0.0)),
            audio_dependency=audio_dependency,
            visual_dependency=visual_dependency,
            method='branch_contrastive_clause_logprob',
            branch_mean_logprobs=mean_logprobs,
            branch_relative_support=relative_support,
            token_count=token_count,
            notes=[
                f'predicate={claim.predicate}',
                f'primary={question_spec.primary_modality}',
                f'prompt={prompt}',
            ],
        )


def build_dependency_probe(config: TraceOmniConfig) -> BaseDependencyProbe:
    if not config.probe.enabled:
        return HeuristicDependencyProbe()
    if config.probe.mode == 'branch_contrastive':
        return BranchContrastiveDependencyProbe()
    return HeuristicDependencyProbe()
