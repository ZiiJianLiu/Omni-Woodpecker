"""
ASR Detector (Training-free)
=============================
使用 faster-whisper 做转录，并建立时间戳索引。
"""
import logging
import re
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


class ASRDetector:
    """语音识别检测器（Training-free）"""

    def __init__(
        self,
        model_size: str = 'large-v3',
        device: str = 'cuda',
        compute_type: str = 'float16',
    ):
        self.model_size = model_size
        if device.startswith('cuda'):
            self.device = 'cuda'
            self._device_index = int(device.split(':')[1]) if ':' in device else 0
        else:
            self.device = 'cpu'
            self._device_index = 0
            if compute_type == 'float16':
                compute_type = 'int8'
                logger.info('CPU 模式：compute_type 自动切换为 int8')
        self.compute_type = compute_type
        self._model = None
        self._load_model()

    def _load_model(self) -> None:
        try:
            from faster_whisper import WhisperModel

            logger.info('加载 ASR 模型: faster-whisper %s', self.model_size)
            self._model = WhisperModel(
                self.model_size,
                device=self.device,
                device_index=self._device_index,
                compute_type=self.compute_type,
            )
            logger.info('ASR 模型加载完成')
        except ImportError:
            logger.error('faster-whisper 未安装，请运行: pip install faster-whisper')
            raise
        except Exception as exc:
            logger.error('ASR 模型加载失败: %s', exc)
            raise

    @staticmethod
    def _normalize_token(token: str) -> str:
        token = re.sub(r"[^a-z0-9']+", ' ', token.lower())
        return ' '.join(token.split())

    @staticmethod
    def build_timestamp_index(
        segments: List[Dict[str, Any]],
        max_ngram: int = 3,
    ) -> Dict[str, List[Dict[str, float]]]:
        """根据词级时间戳建立 unigram / bigram / trigram 索引。"""
        index: Dict[str, List[Dict[str, float]]] = {}

        def add_entry(key: str, start: float, end: float) -> None:
            key = ASRDetector._normalize_token(key)
            if not key:
                return
            index.setdefault(key, []).append({'start': float(start), 'end': float(end)})

        for segment in segments:
            text = ASRDetector._normalize_token(segment.get('text', ''))
            if text:
                add_entry(text, segment.get('start', 0.0), segment.get('end', 0.0))

            words = segment.get('words', []) or []
            normalized_words = []
            for word in words:
                token = ASRDetector._normalize_token(word.get('word', ''))
                if not token:
                    continue
                normalized_words.append(
                    {
                        'word': token,
                        'start': float(word.get('start', segment.get('start', 0.0))),
                        'end': float(word.get('end', segment.get('end', 0.0))),
                    }
                )
                add_entry(token, normalized_words[-1]['start'], normalized_words[-1]['end'])

            for ngram in range(2, max_ngram + 1):
                for idx in range(0, max(0, len(normalized_words) - ngram + 1)):
                    chunk = normalized_words[idx:idx + ngram]
                    phrase = ' '.join(item['word'] for item in chunk)
                    add_entry(phrase, chunk[0]['start'], chunk[-1]['end'])

        return index

    def transcribe(
        self,
        audio: np.ndarray,
        sample_rate: int = 16000,
        language: str = 'en',
    ) -> Optional[str]:
        payload = self.transcribe_with_timestamps(audio, sample_rate=sample_rate, language=language)
        return payload.get('text')

    def transcribe_with_timestamps(
        self,
        audio: np.ndarray,
        sample_rate: int = 16000,
        language: str = 'en',
    ) -> Dict[str, Any]:
        """转录并输出带时间戳的片段和词索引。"""
        if audio is None:
            logger.warning('音频为空，无法转录')
            return {'text': None, 'segments': [], 'index': {}}

        audio = np.asarray(audio).reshape(-1)
        if audio.size == 0:
            logger.warning('音频为空，无法转录')
            return {'text': None, 'segments': [], 'index': {}}

        try:
            if sample_rate != 16000:
                import librosa
                audio = librosa.resample(audio.astype(np.float32), orig_sr=sample_rate, target_sr=16000)

            segments, _ = self._model.transcribe(
                audio,
                language=language,
                beam_size=5,
                vad_filter=True,
                word_timestamps=True,
            )

            rows: List[Dict[str, Any]] = []
            texts: List[str] = []
            for segment in segments:
                seg_text = (segment.text or '').strip()
                word_rows: List[Dict[str, Any]] = []
                for word in getattr(segment, 'words', []) or []:
                    token = (word.word or '').strip()
                    if not token:
                        continue
                    word_rows.append(
                        {
                            'word': token,
                            'start': float(word.start or segment.start or 0.0),
                            'end': float(word.end or segment.end or 0.0),
                            'probability': float(getattr(word, 'probability', 0.0) or 0.0),
                        }
                    )
                rows.append(
                    {
                        'start': float(segment.start or 0.0),
                        'end': float(segment.end or 0.0),
                        'text': seg_text,
                        'words': word_rows,
                    }
                )
                if seg_text:
                    texts.append(seg_text)

            text = ' '.join(texts).strip() or None
            index = self.build_timestamp_index(rows)
            logger.debug('ASR 转录完成: %s', (text or '')[:120])
            return {'text': text, 'segments': rows, 'index': index}
        except Exception as exc:
            logger.error('ASR 转录失败: %s', exc)
            return {'text': None, 'segments': [], 'index': {}}

    def transcribe_from_file(
        self,
        audio_path: str,
        language: str = 'en',
    ) -> Optional[str]:
        try:
            segments, _ = self._model.transcribe(
                audio_path,
                language=language,
                beam_size=5,
                vad_filter=True,
                word_timestamps=True,
            )
            text = ' '.join((segment.text or '').strip() for segment in segments).strip()
            return text or None
        except Exception as exc:
            logger.error('ASR 转录失败: %s', exc)
            return None
