"""DSAV async validator.

根据敏感词、音频事件别名与 ASR 证据索引触发验证，并决定是否回溯。
"""
import asyncio
import logging
import re
import subprocess
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

from ..detectors.asr_detector import ASRDetector
from ..detectors.audio_event_detector import AudioEventDetector

logger = logging.getLogger(__name__)

_DEFAULT_SENSITIVE_ROWS = [
    'dog,dogs,bark,barking,whimper,whimpering',
    'cat,cats,meow,meowing',
    'bird,birds,chirp,chirping',
    'frog,frogs,croak,croaking',
    'music,song,songs,singing,melody',
    'laugh,laughter,clap,applause,cheer,cheering',
    'cry,crying,baby,babies,scream,screaming',
    'siren,alarm,fire alarm,horn',
    'engine,motor,vehicle,car,cars,helicopter,rotor,rotors,boat,train,train engine',
    'radio,intercom,communication,walkie talkie',
    'accelerate,accelerating,engine idle,idle,idling,tire squeal,tires squealing,squeal,squealing',
    'rain,thunder,wind',
    'phone,ring,ringing',
    'toilet,flush,flushing',
    'gunshot,gunshots,gunfire,fire,explosion,explosions',
    'drum,drums,piano,guitar',
]
_LOW_RECALL_AUDIO_CANONICALS = {
    'rain',
    'thunder',
    'wind',
    'engine',
    'motor',
    'vehicle',
    'car',
    'boat',
    'train',
    'airplane',
    'helicopter',
    'accelerate',
    'water',
}
_STOPWORDS = {
    'a', 'an', 'the', 'and', 'or', 'of', 'to', 'in', 'on', 'at', 'for', 'with',
    'by', 'from', 'up', 'down', 'out', 'into', 'over', 'under', 'off', 'through',
    'is', 'are', 'was', 'were', 'be', 'been', 'being', 'it', 'this', 'that', 'these',
    'those', 'you', 'your', 'i', 'we', 'they', 'he', 'she', 'them', 'his', 'her',
}


@dataclass
class ValidationDecision:
    passed: bool = True
    triggered: bool = False
    hallucinated: bool = False
    trigger_term: Optional[str] = None
    canonical_term: Optional[str] = None
    rollback_steps: int = 1
    safe_context: str = ''
    reason: str = ''
    validation_latency_s: float = 0.0
    evidence: Dict[str, Any] = field(default_factory=dict)


class AsyncValidator:
    """DSAV 异步验证器。"""

    def __init__(
        self,
        sensitive_words_path: Optional[str],
        *,
        audio_event_model_name: str = 'MIT/ast-finetuned-audioset-10-10-0.4593',
        asr_model_size: str = 'large-v3',
        device: str = 'cuda:2',
        max_rollback_steps: int = 10,
        validation_window_tokens: int = 6,
        chunk_duration_s: float = 2.0,
        hop_duration_s: float = 1.0,
        audio_support_threshold: float = 0.18,
        audio_event_detector: Optional[AudioEventDetector] = None,
        asr_detector: Optional[ASRDetector] = None,
    ):
        self.device = device
        self.max_rollback_steps = max_rollback_steps
        self.validation_window_tokens = validation_window_tokens
        self.chunk_duration_s = chunk_duration_s
        self.hop_duration_s = hop_duration_s
        self.audio_support_threshold = audio_support_threshold
        self.audio_event_detector = audio_event_detector or AudioEventDetector(
            model_name=audio_event_model_name,
            device=device,
        )
        self.asr_detector = asr_detector or ASRDetector(
            model_size=asr_model_size,
            device=device,
        )
        self.alias_to_canonical, self.canonical_to_aliases = self._load_sensitive_words(
            sensitive_words_path,
        )
        self._evidence_cache: Dict[str, Dict[str, Any]] = {}

    @staticmethod
    def _normalize_phrase(text: str) -> str:
        text = re.sub(r"[^a-z0-9']+", ' ', text.lower())
        return ' '.join(text.split())

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        normalized = AsyncValidator._normalize_phrase(text)
        return normalized.split() if normalized else []

    @staticmethod
    def _singularize(term: str) -> str:
        if term.endswith('ies') and len(term) > 3:
            return term[:-3] + 'y'
        if term.endswith('s') and not term.endswith('ss') and len(term) > 3:
            return term[:-1]
        return term

    def _load_sensitive_words(
        self,
        path: Optional[str],
    ) -> Tuple[Dict[str, str], Dict[str, Set[str]]]:
        rows: List[str]
        if path and Path(path).exists():
            rows = Path(path).read_text(encoding='utf-8').splitlines()
            logger.info('加载 DSAV 敏感词表: %s', path)
        else:
            rows = list(_DEFAULT_SENSITIVE_ROWS)
            if path:
                logger.warning('未找到敏感词表 %s，使用内置默认词表', path)

        alias_to_canonical: Dict[str, str] = {}
        canonical_to_aliases: Dict[str, Set[str]] = {}
        for row in rows:
            row = row.split('#', 1)[0].strip()
            if not row:
                continue
            items = [row.split(':', 1)[0]] + row.split(':', 1)[1].split(',') if ':' in row else row.split(',')
            normalized = [self._normalize_phrase(item) for item in items]
            normalized = [item for item in normalized if item]
            if not normalized:
                continue
            canonical = normalized[0]
            aliases = set(normalized)
            aliases.update(self._singularize(alias) for alias in list(aliases))
            aliases = {alias for alias in aliases if alias}
            canonical_to_aliases.setdefault(canonical, set()).update(aliases)
            for alias in aliases:
                alias_to_canonical[alias] = canonical
        return alias_to_canonical, canonical_to_aliases

    def _register_alias(
        self,
        alias_to_canonical: Dict[str, str],
        canonical_to_aliases: Dict[str, Set[str]],
        canonical: str,
        alias: str,
    ) -> None:
        alias = self._normalize_phrase(alias)
        canonical = self._normalize_phrase(canonical)
        if not alias or not canonical:
            return
        if alias in _STOPWORDS:
            return
        alias_to_canonical[alias] = canonical
        canonical_to_aliases.setdefault(canonical, set()).add(alias)

    def _build_dynamic_aliases(self, evidence: Dict[str, Any]) -> Tuple[Dict[str, str], Dict[str, Set[str]]]:
        alias_to_canonical: Dict[str, str] = {}
        canonical_to_aliases: Dict[str, Set[str]] = {}
        for label in evidence.get('top_audio_events', [])[:8]:
            canonical = self._normalize_phrase(label)
            if not canonical:
                continue
            self._register_alias(alias_to_canonical, canonical_to_aliases, canonical, canonical)
            tokens = canonical.split()
            for token in tokens:
                self._register_alias(alias_to_canonical, canonical_to_aliases, canonical, token)
                self._register_alias(alias_to_canonical, canonical_to_aliases, canonical, self._singularize(token))
            for ngram in (2, 3):
                for idx in range(0, max(0, len(tokens) - ngram + 1)):
                    phrase = ' '.join(tokens[idx:idx + ngram])
                    self._register_alias(alias_to_canonical, canonical_to_aliases, canonical, phrase)
        return alias_to_canonical, canonical_to_aliases

    @staticmethod
    def _load_audio(video_path: str) -> Optional[np.ndarray]:
        try:
            import librosa
            audio, _ = librosa.load(video_path, sr=16000, mono=True)
            if len(audio) > 0:
                return audio.astype(np.float32)
        except Exception:
            pass

        try:
            tmp = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
            tmp.close()
            cmd = [
                'ffmpeg', '-y', '-i', video_path,
                '-vn', '-acodec', 'pcm_s16le',
                '-ar', '16000', '-ac', '1',
                tmp.name, '-loglevel', 'error',
            ]
            ret = subprocess.run(cmd, capture_output=True, timeout=30)
            if ret.returncode != 0:
                Path(tmp.name).unlink(missing_ok=True)
                return None
            import soundfile as sf
            audio, _ = sf.read(tmp.name, dtype='float32')
            Path(tmp.name).unlink(missing_ok=True)
            audio = np.asarray(audio).reshape(-1)
            return audio if audio.size else None
        except Exception as exc:
            logger.debug('DSAV 音频提取失败: %s', exc)
            return None

    async def prepare(
        self,
        video_path: str,
        audio_array: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        if video_path in self._evidence_cache:
            return self._evidence_cache[video_path]

        if audio_array is None:
            audio_array = await asyncio.to_thread(self._load_audio, video_path)

        if audio_array is None:
            evidence = {
                'video_path': video_path,
                'audio_event_timeline': [],
                'audio_event_scores': {},
                'top_audio_events': [],
                'asr_text': None,
                'asr_segments': [],
                'asr_index': {},
            }
            evidence['dynamic_alias_to_canonical'], evidence['dynamic_canonical_to_aliases'] = self._build_dynamic_aliases(evidence)
            self._evidence_cache[video_path] = evidence
            return evidence

        timeline = await asyncio.to_thread(
            self.audio_event_detector.detect_timeline,
            audio_array,
            16000,
            self.chunk_duration_s,
            self.hop_duration_s,
            None,
            self.audio_support_threshold,
        )
        summary = self.audio_event_detector.summarize_timeline(timeline, top_k=8)
        asr_payload = await asyncio.to_thread(
            self.asr_detector.transcribe_with_timestamps,
            audio_array,
            16000,
            'en',
        )
        evidence = {
            'video_path': video_path,
            'audio_event_timeline': timeline,
            'audio_event_scores': summary,
            'top_audio_events': list(summary.keys()),
            'asr_text': asr_payload.get('text'),
            'asr_segments': asr_payload.get('segments', []),
            'asr_index': asr_payload.get('index', {}),
        }
        evidence['dynamic_alias_to_canonical'], evidence['dynamic_canonical_to_aliases'] = self._build_dynamic_aliases(evidence)
        self._evidence_cache[video_path] = evidence
        return evidence

    async def start_session(
        self,
        video_path: str,
        audio_array: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        evidence = await self.prepare(video_path, audio_array=audio_array)
        return {
            'video_path': video_path,
            'evidence': evidence,
            'validated_phrases': set(),
            'last_token_count': 0,
            'trigger_count': 0,
            'rollback_count': 0,
            'validation_latency_s': 0.0,
        }

    def prepare_sync(self, video_path: str, audio_array: Optional[np.ndarray] = None) -> Dict[str, Any]:
        return self._run(self.prepare(video_path, audio_array=audio_array))

    def start_session_sync(self, video_path: str, audio_array: Optional[np.ndarray] = None) -> Dict[str, Any]:
        return self._run(self.start_session(video_path, audio_array=audio_array))

    def validate_incremental_sync(
        self,
        generated_text: str,
        video_path: str,
        *,
        session: Optional[Dict[str, Any]] = None,
        audio_array: Optional[np.ndarray] = None,
    ) -> ValidationDecision:
        return self._run(
            self.validate_incremental(
                generated_text,
                video_path,
                session=session,
                audio_array=audio_array,
            )
        )

    def scan_text_sync(
        self,
        generated_text: str,
        video_path: str,
        *,
        audio_array: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        session = self.start_session_sync(video_path, audio_array=audio_array)
        words = generated_text.split()
        accepted_words: List[str] = []
        safe_contexts: List[str] = []
        events: List[Dict[str, Any]] = []

        for word in words:
            accepted_words.append(word)
            prefix_text = ' '.join(accepted_words)
            decision = self.validate_incremental_sync(
                prefix_text,
                video_path,
                session=session,
                audio_array=audio_array,
            )
            if decision.hallucinated:
                if decision.safe_context and decision.safe_context not in safe_contexts:
                    safe_contexts.append(decision.safe_context)
                events.append(
                    {
                        'term': decision.trigger_term,
                        'canonical_term': decision.canonical_term,
                        'rollback_steps': decision.rollback_steps,
                        'reason': decision.reason,
                        'evidence': decision.evidence,
                    }
                )
                rollback = min(decision.rollback_steps, len(accepted_words))
                accepted_words = accepted_words[:-rollback]

        summary = self.summarize_session(session)
        summary.update(
            {
                'events': events,
                'safe_contexts': safe_contexts,
                'accepted_text': ' '.join(accepted_words).strip(),
            }
        )
        return summary

    @staticmethod
    def _run(coro):
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(coro)
        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(coro)
        finally:
            loop.close()

    def _resolve_canonical(
        self,
        phrase: str,
        evidence: Dict[str, Any],
    ) -> Optional[str]:
        canonical = self.alias_to_canonical.get(phrase)
        if canonical is None:
            canonical = self.alias_to_canonical.get(self._singularize(phrase))
        if canonical is None:
            dynamic_aliases = evidence.get('dynamic_alias_to_canonical', {})
            canonical = dynamic_aliases.get(phrase)
        if canonical is None:
            dynamic_aliases = evidence.get('dynamic_alias_to_canonical', {})
            canonical = dynamic_aliases.get(self._singularize(phrase))
        return canonical

    def _extract_candidates(
        self,
        tokens: List[str],
        last_token_count: int,
        validated_phrases: Set[str],
        evidence: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        if not tokens:
            return []
        start_idx = max(0, last_token_count - 2)
        candidates: List[Dict[str, Any]] = []
        seen: Set[str] = set()
        max_width = min(3, self.validation_window_tokens)

        for end in range(start_idx + 1, len(tokens) + 1):
            for width in range(max_width, 0, -1):
                begin = end - width
                if begin < 0:
                    continue
                phrase = ' '.join(tokens[begin:end])
                canonical = self._resolve_canonical(phrase, evidence)
                if canonical is None:
                    continue
                if phrase in validated_phrases or phrase in seen:
                    continue
                seen.add(phrase)
                candidates.append(
                    {
                        'phrase': phrase,
                        'canonical': canonical,
                        'rollback_steps': min(width, self.max_rollback_steps),
                    }
                )
        return candidates[-self.validation_window_tokens:]

    def _candidate_aliases(self, canonical: str, evidence: Dict[str, Any]) -> Set[str]:
        aliases = set(self.canonical_to_aliases.get(canonical, set()))
        aliases.update(evidence.get('dynamic_canonical_to_aliases', {}).get(canonical, set()))
        aliases.add(canonical)
        aliases.add(self._singularize(canonical))
        return {alias for alias in aliases if alias}

    def _match_audio_support(self, aliases: Set[str], event_scores: Dict[str, float]) -> Dict[str, Any]:
        best = {'supported': False, 'label': None, 'score': 0.0, 'overlap': 0.0}
        for label, score in event_scores.items():
            label_tokens = set(label.split())
            for alias in aliases:
                alias_tokens = set(alias.split())
                if not alias_tokens:
                    continue
                overlap = len(alias_tokens & label_tokens) / max(1, len(alias_tokens))
                if alias == label or alias in label or label in alias:
                    overlap = max(overlap, 1.0 if len(alias_tokens) == 1 else 0.8)
                weighted = float(score) * overlap
                if weighted > best['score'] * max(best['overlap'], 1e-6):
                    best = {
                        'supported': overlap > 0 and float(score) >= self.audio_support_threshold,
                        'label': label,
                        'score': float(score),
                        'overlap': overlap,
                    }
        return best

    def _match_asr_support(self, aliases: Set[str], asr_index: Dict[str, List[Dict[str, float]]]) -> Dict[str, Any]:
        for alias in aliases:
            spans = asr_index.get(alias)
            if spans:
                return {'supported': True, 'match': alias, 'spans': spans[:3]}
        return {'supported': False, 'match': None, 'spans': []}

    def _should_soft_accept_unsupported(self, canonical: str, details: Dict[str, Any]) -> bool:
        if canonical not in _LOW_RECALL_AUDIO_CANONICALS:
            return False
        if details.get('asr_support', {}).get('supported'):
            return False
        return True

    def _build_safe_context(self, canonical: str, evidence: Dict[str, Any]) -> str:
        top_events = ', '.join(evidence.get('top_audio_events', [])[:3]) or 'no strong audio event'
        asr_text = (evidence.get('asr_text') or 'no speech recognized').strip()
        if len(asr_text) > 160:
            asr_text = asr_text[:157] + '...'
        return (
            '[DSAV verified context] '
            f'Supported audio evidence: {top_events}. '
            f'ASR transcript: {asr_text}. '
            f'Do not mention unsupported audio entity "{canonical}" unless the audio clearly supports it.'
        )

    def _validate_candidate(self, candidate: Dict[str, Any], evidence: Dict[str, Any]) -> ValidationDecision:
        canonical = candidate['canonical']
        aliases = self._candidate_aliases(canonical, evidence)
        audio_support = self._match_audio_support(aliases, evidence.get('audio_event_scores', {}))
        asr_support = self._match_asr_support(aliases, evidence.get('asr_index', {}))
        supported = audio_support['supported'] or asr_support['supported']
        details = {'audio_support': audio_support, 'asr_support': asr_support}
        if supported:
            return ValidationDecision(
                passed=True,
                triggered=True,
                hallucinated=False,
                trigger_term=candidate['phrase'],
                canonical_term=canonical,
                rollback_steps=candidate['rollback_steps'],
                evidence=details,
                reason='supported_by_audio_or_asr',
            )
        if self._should_soft_accept_unsupported(canonical, details):
            details['policy'] = 'low_recall_audio_soft_accept'
            return ValidationDecision(
                passed=True,
                triggered=True,
                hallucinated=False,
                trigger_term=candidate['phrase'],
                canonical_term=canonical,
                rollback_steps=candidate['rollback_steps'],
                evidence=details,
                reason='unsupported_low_recall_audio_event',
            )
        return ValidationDecision(
            passed=False,
            triggered=True,
            hallucinated=True,
            trigger_term=candidate['phrase'],
            canonical_term=canonical,
            rollback_steps=candidate['rollback_steps'],
            safe_context=self._build_safe_context(canonical, evidence),
            evidence=details,
            reason='unsupported_entity',
        )

    async def validate_incremental(
        self,
        generated_text: str,
        video_path: str,
        *,
        session: Optional[Dict[str, Any]] = None,
        audio_array: Optional[np.ndarray] = None,
    ) -> ValidationDecision:
        t0 = time.perf_counter()
        if session is None:
            session = await self.start_session(video_path, audio_array=audio_array)
        evidence = session['evidence']
        tokens = self._tokenize(generated_text)
        candidates = self._extract_candidates(
            tokens,
            session['last_token_count'],
            session['validated_phrases'],
            evidence,
        )

        if not candidates:
            latency = time.perf_counter() - t0
            session['last_token_count'] = len(tokens)
            session['validation_latency_s'] += latency
            return ValidationDecision(validation_latency_s=latency)

        last_safe = ValidationDecision(triggered=True)
        for candidate in candidates:
            session['trigger_count'] += 1
            decision = self._validate_candidate(candidate, evidence)
            decision.validation_latency_s = time.perf_counter() - t0
            session['validation_latency_s'] += decision.validation_latency_s
            session['validated_phrases'].add(candidate['phrase'])
            if decision.hallucinated:
                session['rollback_count'] += 1
                session['last_token_count'] = max(0, len(tokens) - decision.rollback_steps)
                return decision
            last_safe = decision

        session['last_token_count'] = len(tokens)
        last_safe.validation_latency_s = time.perf_counter() - t0
        session['validation_latency_s'] += last_safe.validation_latency_s
        return last_safe

    def summarize_session(self, session: Dict[str, Any]) -> Dict[str, Any]:
        evidence = session.get('evidence', {})
        return {
            'trigger_count': session.get('trigger_count', 0),
            'rollback_count': session.get('rollback_count', 0),
            'validation_latency_s': session.get('validation_latency_s', 0.0),
            'top_audio_events': evidence.get('top_audio_events', []),
            'asr_text': evidence.get('asr_text'),
        }
