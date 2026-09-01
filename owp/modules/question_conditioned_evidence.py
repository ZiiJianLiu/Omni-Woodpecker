"""
Question-Conditioned Evidence Scorer
====================================
通用问题条件化证据打分器：直接围绕问题实体与查询模态验证支撑证据，
不依赖 benchmark task 名称或固定样例规则。
"""
import logging
import re
from typing import Dict, List, Optional, Tuple

from ..data_types import ModalityFeatures

logger = logging.getLogger(__name__)

_YN_PREFIXES = (
    'is ',
    'are ',
    'was ',
    'were ',
    'do ',
    'does ',
    'did ',
    'can ',
    'could ',
    'will ',
    'would ',
    'has ',
    'have ',
    'had ',
    'should ',
)
_YN_INSTRUCTION_RE = re.compile(
    r"\b(answer|respond)\s+with\s+(only\s+)?yes\s+or\s+no\b|\byes\s+or\s+no\b",
    re.IGNORECASE,
)
_OPTION_PATTERN_RE = re.compile(
    r'([A-H]|\d{1,2})[\.\):]\s*(.+?)(?=(?:\s+(?:[A-H]|\d{1,2})[\.\):]\s)|$)',
    re.IGNORECASE | re.DOTALL,
)
_MULTI_SELECT_PATTERNS = (
    r'\bselect all that apply\b',
    r'\bchoose all that apply\b',
    r'\bmultiple answers?\b',
    r'\bmulti[\s-]?select\b',
    r'\bwhich of the following are\b',
    r'\bwhich options are\b',
    r'\ball correct\b',
    r'\bone or more\b',
)
_TEMPORAL_QUERY_PATTERNS = (
    r'\bwhen\b',
    r'\bbefore\b',
    r'\bafter\b',
    r'\bwhile\b',
    r'\bduring\b',
    r'\bfirst\b',
    r'\bthen\b',
    r'\blater\b',
    r'\bearlier\b',
    r'\bstart(?:s|ed|ing)?\b',
    r'\bstop(?:s|ped|ping)?\b',
    r'\bend(?:s|ed|ing)?\b',
    r'\bsimultaneous(?:ly)?\b',
    r'\bsame time\b',
    r'\btiming\b',
    r'\border\b',
    r'\bsequence\b',
)
_EMOTION_CANONICAL = {
    'happy': 'happy',
    'happiness': 'happy',
    'joy': 'happy',
    'joyful': 'happy',
    'cheerful': 'happy',
    'sad': 'sad',
    'sadness': 'sad',
    'gloomy': 'sad',
    'angry': 'angry',
    'anger': 'angry',
    'mad': 'angry',
    'fear': 'fear',
    'fearful': 'fear',
    'scared': 'fear',
    'afraid': 'fear',
    'surprise': 'surprise',
    'surprised': 'surprise',
    'disgust': 'disgust',
    'disgusted': 'disgust',
    'calm': 'neutral',
    'neutral': 'neutral',
}
_CHOICE_POSITIVE_RELATION_TERMS = {
    'match', 'matches', 'matched', 'matching', 'consistent', 'aligned',
    'corresponding', 'fit', 'fits', 'related', 'coherent',
}
_CHOICE_NEGATIVE_RELATION_TERMS = {
    'mismatch', 'mismatched', 'inconsistent', 'misaligned', 'contradictory',
    'unrelated', 'conflicting', 'conflict',
}
_STOPWORDS = {
    'a', 'an', 'the', 'in', 'on', 'at', 'to', 'of', 'for', 'with', 'and', 'or',
    'visible', 'video', 'scene', 'audio', 'sound', 'sounds', 'noise', 'noises',
    'hear', 'heard', 'see', 'seen', 'making', 'make', 'made', 'is', 'are', 'was',
    'were', 'be', 'been', 'there', 'any',
}
_RELATION_AUDIO_STOPWORDS = {
    'speech', 'music', 'animal', 'animals', 'vehicle', 'vehicles', 'liquid',
    'mechanisms', 'noise', 'sound', 'sounds', 'audio', 'silence', 'door',
    'wood', 'ping', 'slam', 'beep bleep', 'hiss', 'water', 'voice', 'voices',
    'crowd', 'laughter', 'laughing', 'narration', 'narration monologue',
    'engine', 'wind', 'singing',
}
_HUMAN_TERMS = {
    'person', 'people', 'human', 'man', 'woman', 'boy', 'girl', 'child', 'children',
    'kid', 'kids', 'baby', 'babies', 'speaker', 'singer', 'driver', 'player',
}
_GENERIC_AUDIO_PROXY_TERMS = {
    'speech', 'voice', 'talking', 'speaking', 'animal', 'vehicle', 'music',
}
_GROUP_HUMAN_AUDIO_TERMS = {
    'crowd', 'cheering', 'laughter', 'laughing', 'voices', 'applause',
    'clapping', 'snicker', 'giggle',
}
_VEHICLE_AUDIO_TERMS = {
    'vehicle', 'car', 'truck', 'motorcycle', 'boat', 'train', 'airplane', 'helicopter',
    'engine', 'motor', 'horn', 'siren', 'accelerating revving vroom',
}
_ANIMAL_AUDIO_TERMS = {
    'animal', 'dog', 'cat', 'bird', 'horse', 'cow', 'sheep', 'owl', 'bleat',
    'moo', 'meow', 'chirp tweet', 'bird vocalization bird call bird song',
    'neigh whinny', 'quack', 'bee wasp etc',
}
_WATER_AUDIO_TERMS = {
    'water', 'water tap faucet', 'sink filling or washing', 'liquid', 'slosh', 'spray',
}
_MUSIC_AUDIO_TERMS = {
    'music', 'musical instrument', 'singing', 'song', 'melody', 'steelpan',
    'guitar', 'piano', 'drum',
}
_RELATION_GENERIC_CUES = {
    'speech', 'music', 'sound', 'sounds', 'audio', 'noise', 'animal', 'animals',
    'vehicle', 'vehicles', 'water', 'liquid', 'crowd', 'laughter', 'laughing',
    'voice', 'voices', 'narration', 'narration monologue', 'singing', 'engine',
    'wind', 'door',
}
_ENTITY_HINTS = {
    'person': {
        'visual': {'person', 'people', 'man', 'woman', 'boy', 'girl', 'child', 'baby', 'human', 'face'},
        'audio': {'speech', 'voice', 'talking', 'speaking', 'male speech man speaking', 'female speech woman speaking'},
    },
    'woman': {
        'visual': {'woman', 'person', 'people', 'human', 'face'},
        'audio': {'female speech woman speaking'},
    },
    'man': {
        'visual': {'man', 'person', 'people', 'human', 'face'},
        'audio': {'male speech man speaking'},
    },
    'boy': {
        'visual': {'boy', 'person', 'people', 'child', 'human', 'face'},
        'audio': {'child speech kid speaking'},
    },
    'girl': {
        'visual': {'girl', 'person', 'people', 'child', 'human', 'face'},
        'audio': {'child speech kid speaking'},
    },
    'kid': {
        'visual': {'kid', 'child', 'person', 'people', 'human', 'face'},
        'audio': {'child speech kid speaking', 'baby cry infant cry'},
    },
    'baby': {
        'visual': {'baby', 'person', 'people', 'child', 'human', 'face'},
        'audio': {'baby cry infant cry', 'crying sobbing'},
    },
    'car': {
        'visual': {'car', 'vehicle'},
        'audio': {'car', 'vehicle', 'engine', 'motor', 'vehicle horn car horn honking', 'accelerating revving vroom'},
    },
    'vehicle': {
        'visual': {'vehicle', 'car', 'truck', 'bus', 'motorcycle', 'bike', 'bicycle', 'boat', 'train', 'airplane', 'helicopter'},
        'audio': {'vehicle', 'car', 'truck', 'bus', 'motorcycle', 'boat', 'train', 'airplane', 'helicopter', 'engine', 'motor', 'vehicle horn car horn honking', 'accelerating revving vroom', 'train horn'},
    },
    'engine': {
        'visual': {'engine', 'vehicle', 'car', 'truck', 'motorcycle', 'boat', 'train', 'airplane', 'helicopter'},
        'audio': {'engine', 'motor', 'vehicle', 'car', 'truck', 'motorcycle', 'boat', 'train', 'airplane', 'helicopter', 'accelerating revving vroom', 'vehicle horn car horn honking'},
    },
    'motorcycle': {
        'visual': {'motorcycle', 'vehicle'},
        'audio': {'motorcycle', 'vehicle', 'engine', 'motor', 'accelerating revving vroom'},
    },
    'train': {
        'visual': {'train', 'vehicle'},
        'audio': {'train', 'vehicle', 'engine', 'horn', 'train horn'},
    },
    'airplane': {
        'visual': {'airplane', 'plane', 'aircraft'},
        'audio': {'airplane', 'plane', 'aircraft', 'engine', 'vehicle'},
    },
    'helicopter': {
        'visual': {'helicopter', 'aircraft'},
        'audio': {'helicopter', 'rotor', 'engine', 'vehicle'},
    },
    'boat': {
        'visual': {'boat', 'ship', 'canoe', 'kayak', 'rowboat'},
        'audio': {'boat', 'water vehicle', 'rowboat canoe kayak', 'vehicle', 'liquid'},
    },
    'mower': {
        'visual': {'mower', 'lawn mower', 'machine'},
        'audio': {'engine', 'motor', 'vehicle', 'buzz', 'lawn mower'},
    },
    'rain': {
        'visual': {'rain', 'water', 'umbrella'},
        'audio': {'rain', 'water', 'thunder', 'wind', 'slosh', 'spray'},
    },
    'water': {
        'visual': {'water', 'river', 'ocean', 'lake', 'pool', 'sea'},
        'audio': {'water', 'slosh', 'spray', 'sink filling or washing', 'water tap faucet'},
    },
    'fireworks': {
        'visual': {'fireworks', 'firework', 'spark', 'burst'},
        'audio': {'fireworks', 'firecracker', 'explosion', 'burst pop', 'gunshot gunfire'},
    },
    'dog': {
        'visual': {'dog', 'animal'},
        'audio': {'dog', 'bark', 'barking', 'animal'},
    },
    'animal': {
        'visual': {'animal', 'dog', 'cat', 'bird', 'horse', 'cow', 'sheep'},
        'audio': {'animal', 'dog', 'cat', 'bird', 'horse', 'cow', 'sheep', 'bark', 'barking', 'meow', 'chirp tweet', 'bird vocalization bird call bird song', 'neigh whinny', 'moo', 'bleat'},
    },
    'cat': {
        'visual': {'cat', 'animal'},
        'audio': {'cat', 'meow', 'animal'},
    },
    'bird': {
        'visual': {'bird', 'animal'},
        'audio': {'bird', 'chirp tweet', 'bird vocalization bird call bird song', 'animal'},
    },
    'horse': {
        'visual': {'horse', 'animal'},
        'audio': {'horse', 'clip clop', 'neigh whinny', 'animal'},
    },
    'cow': {
        'visual': {'cow', 'animal'},
        'audio': {'cow', 'moo', 'animal', 'livestock farm animals working animals'},
    },
    'sheep': {
        'visual': {'sheep', 'animal'},
        'audio': {'sheep', 'bleat', 'animal', 'livestock farm animals working animals'},
    },
    'instrument': {
        'visual': {'guitar', 'piano', 'drum', 'violin', 'instrument'},
        'audio': {'music', 'musical instrument', 'guitar', 'piano', 'drum'},
    },
    'guitar': {
        'visual': {'guitar', 'instrument'},
        'audio': {'guitar', 'music', 'musical instrument'},
    },
    'piano': {
        'visual': {'piano', 'instrument'},
        'audio': {'piano', 'music', 'musical instrument'},
    },
    'drum': {
        'visual': {'drum', 'instrument'},
        'audio': {'drum', 'music', 'musical instrument'},
    },
    'music': {
        'visual': {'instrument', 'speaker'},
        'audio': {'music', 'song', 'singing', 'melody'},
    },
    'sponge': {
        'visual': {'sponge', 'cleaning sponge', 'makeup sponge', 'scrubber', 'pad'},
        'audio': {'water', 'water tap faucet', 'sink filling or washing', 'spray'},
    },
    'voice': {
        'visual': {'person', 'people', 'man', 'woman', 'boy', 'girl', 'face'},
        'audio': {'speech', 'voice', 'speaking', 'talking'},
    },
    'child': {
        'visual': {'child', 'kid', 'person', 'people', 'human', 'face'},
        'audio': {'child speech kid speaking'},
    },
    'children': {
        'visual': {'child', 'children', 'kid', 'person', 'people', 'human', 'face'},
        'audio': {'child speech kid speaking'},
    },
    'girls': {
        'visual': {'girl', 'child', 'person', 'people', 'human', 'face'},
        'audio': {'child speech kid speaking'},
    },
    'truck': {
        'visual': {'truck', 'vehicle'},
        'audio': {'truck', 'vehicle', 'engine', 'motor', 'vehicle horn car horn honking', 'accelerating revving vroom'},
    },
    'crowd': {
        'visual': {'crowd', 'people', 'person'},
        'audio': {'crowd', 'cheering', 'laughter', 'laughing', 'voices', 'applause', 'clapping'},
    },
    'bell': {
        'visual': {'bell'},
        'audio': {'bell', 'ding', 'chime'},
    },
    'lightning': {
        'visual': {'lightning', 'storm'},
        'audio': {'thunder'},
    },
    'thunder': {
        'visual': {'thunder', 'storm', 'lightning'},
        'audio': {'thunder'},
    },
    'harmonica': {
        'visual': {'harmonica', 'instrument'},
        'audio': {'harmonica', 'music', 'musical instrument'},
    },
    'trombone': {
        'visual': {'trombone', 'instrument'},
        'audio': {'trombone', 'music', 'musical instrument'},
    },
    'cellist': {
        'visual': {'cellist', 'person', 'instrument'},
        'audio': {'music', 'musical instrument', 'cello'},
    },
    'machine': {
        'visual': {'machine', 'equipment'},
        'audio': {'engine', 'motor', 'machinery'},
    },
    'engine rev': {
        'visual': {'engine', 'vehicle'},
        'audio': {'engine', 'motor', 'accelerating revving vroom'},
    },
    'car rev': {
        'visual': {'car', 'vehicle'},
        'audio': {'car', 'engine', 'motor', 'accelerating revving vroom'},
    },
    'car engine rev': {
        'visual': {'car', 'vehicle', 'engine'},
        'audio': {'car', 'engine', 'motor', 'accelerating revving vroom'},
    },
    'car honk': {
        'visual': {'car', 'vehicle'},
        'audio': {'car', 'vehicle', 'vehicle horn car horn honking'},
    },
    'dog bark': {
        'visual': {'dog', 'animal'},
        'audio': {'dog', 'bark', 'barking'},
    },
    'dog barking': {
        'visual': {'dog', 'animal'},
        'audio': {'dog', 'bark', 'barking'},
    },
    'water splash': {
        'visual': {'water', 'pool', 'spray'},
        'audio': {'water', 'slosh', 'spray'},
    },
    'spray': {
        'visual': {'spray', 'water'},
        'audio': {'spray', 'water', 'slosh'},
    },
    'bees': {
        'visual': {'bee', 'bees', 'insect'},
        'audio': {'bee wasp etc', 'buzz'},
    },
    'fire': {
        'visual': {'fire', 'flame'},
        'audio': {'fire', 'crackle'},
    },
    'lion': {
        'visual': {'lion', 'animal'},
        'audio': {'animal', 'roar'},
    },
}

_AUDIO_PROXY_HINTS = {
    'street': {'engine', 'car passing by', 'vehicle horn car horn honking'},
    'road': {'engine', 'car passing by', 'vehicle horn car horn honking'},
    'highway': {'engine', 'car passing by', 'vehicle horn car horn honking'},
    'wood': {'knock', 'tap', 'creak', 'squeak'},
    'grass': {'footsteps', 'rustling leaves', 'wind'},
    'tree': {'rustling leaves', 'wind'},
    'leaf': {'rustling leaves', 'wind'},
    'leaves': {'rustling leaves', 'wind'},
    'car key': {'key jangling', 'jingle', 'metallic click'},
    'key': {'key jangling', 'jingle', 'metallic click'},
    'microphone': {'speech', 'voice', 'speaking', 'microphone handling noise', 'microphone tap'},
    'mic': {'speech', 'voice', 'speaking', 'microphone handling noise', 'microphone tap'},
    'judge': {'speech', 'voice', 'speaking'},
    'glass': {'glass clink', 'glass shatter', 'tap'},
    'origami': {'paper rustling', 'paper folding', 'paper crumpling'},
    'paper': {'paper rustling', 'paper crumpling', 'page turning'},
    'snow': {'footsteps in snow', 'snow crunch', 'wind'},
    'fence': {'rattle', 'metal clink'},
    'cage': {'rattle', 'metal clink'},
    'tire': {'rolling', 'skid', 'engine'},
    'knife': {'metallic clink', 'chop', 'cutting'},
    'stage': {'applause', 'crowd', 'speech', 'music'},
}

_AUDIO_PROXY_HUMAN_ROLE_TERMS = {
    'judge', 'speaker', 'announcer', 'host', 'reporter', 'teacher', 'singer',
    'commentator', 'interviewer', 'narrator', 'preacher', 'performer',
}


class QuestionConditionedEvidenceScorer:
    """围绕问题实体做模态证据验证。"""

    def __init__(self, config):
        self.config = config

    @staticmethod
    def _normalize(text: str) -> str:
        text = re.sub(r"[^a-z0-9']+", ' ', (text or '').lower())
        return ' '.join(text.split())

    @staticmethod
    def _clip01(value: float) -> float:
        try:
            return max(0.0, min(1.0, float(value)))
        except (TypeError, ValueError):
            return 0.0

    @staticmethod
    def _singularize(term: str) -> str:
        if term.endswith('ies') and len(term) > 3:
            return term[:-3] + 'y'
        if term.endswith('s') and not term.endswith('ss') and len(term) > 3:
            return term[:-1]
        return term

    @staticmethod
    def _strip_yn_instruction(question: str) -> str:
        return re.sub(
            r'\s*(answer|respond)\s+with\s+(only\s+)?yes\s+or\s+no\.?\s*$',
            '',
            question or '',
            flags=re.IGNORECASE,
        ).strip()

    @staticmethod
    def _extract_yes_no(text: str) -> str:
        normalized = (text or '').strip().lower()
        if normalized.startswith('yes'):
            return 'Yes'
        if normalized.startswith('no'):
            return 'No'
        return 'Unknown'

    @staticmethod
    def is_yes_no_question(question: str, max_new_tokens: Optional[int] = None) -> bool:
        q = (question or '').strip().lower()
        if _YN_INSTRUCTION_RE.search(q):
            return True
        if max_new_tokens is not None and max_new_tokens <= 10:
            return True
        return q.startswith(_YN_PREFIXES)

    def _clean_entity_span(self, entity: str) -> str:
        entity = self._normalize(entity)
        entity = re.sub(r'^(?:a|an|the|any)\s+', '', entity)
        entity = re.sub(r'^(?:sound|noise)s?\s+of\s+', '', entity)
        entity = re.sub(r'\s+(?:in|on)\s+(?:the\s+)?(?:video|scene|audio)$', '', entity)
        return entity.strip()

    def _extract_choice_spec(self, question: str) -> Optional[Dict[str, object]]:
        raw = re.sub(r'\s+', ' ', question or '').strip()
        if not raw:
            return None
        matches = list(_OPTION_PATTERN_RE.finditer(raw))
        if len(matches) < 2:
            return None

        stem = raw[:matches[0].start()].strip(' :-')
        if len(stem.split()) < 2:
            return None

        options: List[Dict[str, str]] = []
        for match in matches:
            label = str(match.group(1)).strip().upper()
            text = re.sub(r'\s+', ' ', match.group(2) or '').strip(' ;,')
            if not label or not text:
                continue
            options.append({'label': label, 'text': text})
        if len(options) < 2:
            return None

        normalized_stem = self._normalize(stem)
        is_multi_select = any(
            re.search(pattern, normalized_stem)
            for pattern in _MULTI_SELECT_PATTERNS
        )
        query_spec = self._infer_choice_query_spec(stem)
        return {
            'question_kind': 'choice_grounded',
            'form': 'multi_select' if is_multi_select else 'single_choice',
            'stem': stem,
            'normalized_stem': normalized_stem,
            'normalized_question': self._normalize(raw),
            'options': options,
            'query_spec': query_spec,
        }

    def classify_question_form(
        self,
        question: str,
        max_new_tokens: Optional[int] = None,
    ) -> Dict[str, object]:
        if self.is_yes_no_question(question, max_new_tokens):
            return {
                'form': 'yes_no',
                'supported': True,
                'normalized_question': self._normalize(self._strip_yn_instruction(question)),
            }
        choice_spec = self._extract_choice_spec(question)
        if choice_spec is not None:
            return {
                'form': choice_spec['form'],
                'supported': True,
                **choice_spec,
            }
        return {
            'form': 'open_ended',
            'supported': True,
            'normalized_question': self._normalize(question),
        }

    def _infer_choice_query_spec(self, stem: str) -> Dict[str, str]:
        normalized = self._normalize(stem)
        tokens = set(normalized.split())
        has_audio_side = bool(tokens & {'audio', 'sound', 'sounds', 'hear', 'heard', 'audible'})
        has_visual_side = bool(tokens & {'video', 'visual', 'scene', 'frame', 'frames', 'visible', 'see', 'seen'})
        has_emotion = bool(tokens & {'emotion', 'mood', 'feeling', 'feelings', 'tone', 'sentiment'})

        if has_emotion:
            modality = 'cross_modal'
            if has_audio_side and not has_visual_side:
                modality = 'audio'
            elif has_visual_side and not has_audio_side:
                modality = 'visual'
            return {'modality': modality, 'relation': 'emotion'}

        if (
            'visible in the video' in normalized
            or 'visible in the scene' in normalized
            or 'seen in the video' in normalized
            or normalized.startswith('can you see ')
        ):
            return {'modality': 'visual', 'relation': 'presence'}

        if (
            'heard in the audio' in normalized
            or 'making sound in the audio' in normalized
            or 'making noise in the audio' in normalized
            or 'in the audio' in normalized
            or 'audible' in normalized
            or normalized.startswith('can you hear ')
        ):
            return {'modality': 'audio', 'relation': 'sound'}

        relation_patterns = (
            r'\bmatch(?:es|ed|ing)?\b',
            r'\bconsistent\b',
            r'\bconsistency\b',
            r'\balign(?:ed|ing)?\b',
            r'\bcorrespond(?:s|ed|ing)?\b',
            r'\bfit(?:s|ted|ting)?\b',
            r'\bgo(?:es)?\s+with\b',
        )
        if any(re.search(pattern, normalized) for pattern in relation_patterns):
            return {'modality': 'cross_modal', 'relation': 'consistency'}

        if has_audio_side and not has_visual_side:
            return {'modality': 'audio', 'relation': 'attribute'}
        if has_visual_side and not has_audio_side:
            return {'modality': 'visual', 'relation': 'attribute'}
        return {'modality': 'cross_modal', 'relation': 'attribute'}

    def _infer_freeform_query_spec(self, question: str) -> Dict[str, str]:
        normalized = self._normalize(self._strip_yn_instruction(question))
        tokens = set(normalized.split())
        has_audio_side = bool(
            tokens & {
                'audio',
                'sound',
                'sounds',
                'hear',
                'heard',
                'hearing',
                'listen',
                'listening',
                'voice',
                'voices',
                'speech',
                'music',
                'noise',
                'noises',
                'audible',
                'speaker',
                'speaking',
                'saying',
            }
        )
        has_visual_side = bool(
            tokens & {
                'video',
                'visual',
                'scene',
                'frame',
                'frames',
                'visible',
                'see',
                'seen',
                'look',
                'looks',
                'show',
                'shown',
                'showing',
                'appear',
                'appears',
                'image',
            }
        )
        has_emotion = bool(tokens & {'emotion', 'mood', 'feeling', 'feelings', 'tone', 'sentiment'})
        has_temporal = any(re.search(pattern, normalized) for pattern in _TEMPORAL_QUERY_PATTERNS)

        if has_emotion:
            modality = 'cross_modal'
            if has_audio_side and not has_visual_side:
                modality = 'audio'
            elif has_visual_side and not has_audio_side:
                modality = 'visual'
            return {'modality': modality, 'relation': 'emotion'}

        if has_temporal:
            modality = 'cross_modal'
            if has_audio_side and not has_visual_side:
                modality = 'audio'
            elif has_visual_side and not has_audio_side:
                modality = 'visual'
            return {'modality': modality, 'relation': 'temporal'}

        if (
            re.search(r'\bwhat (?:is|are) (?:visible|shown|seen)\b', normalized)
            or re.search(r'\bwho is (?:visible|shown|seen)\b', normalized)
            or re.search(r'\bwhat can you see\b', normalized)
            or re.search(r'\bwhat do you see\b', normalized)
            or re.search(r'\bwhat is in the video\b', normalized)
            or re.search(r'\bwhat happens in the video\b', normalized)
        ):
            return {'modality': 'visual', 'relation': 'presence'}

        if (
            re.search(r'\bwhat (?:sound|sounds) (?:is|are)\b', normalized)
            or re.search(r'\bwhat can you hear\b', normalized)
            or re.search(r'\bwhat do you hear\b', normalized)
            or re.search(r'\bwho is speaking\b', normalized)
            or re.search(r'\bwhat is being said\b', normalized)
            or re.search(r'\bwhat is heard\b', normalized)
        ):
            return {'modality': 'audio', 'relation': 'sound'}

        relation_patterns = (
            r'\bmatch(?:es|ed|ing)?\b',
            r'\bconsistent\b',
            r'\bconsistency\b',
            r'\balign(?:ed|ing)?\b',
            r'\bcorrespond(?:s|ed|ing)?\b',
            r'\bfit(?:s|ted|ting)?\b',
            r'\bgo(?:es)?\s+with\b',
            r'\bdescribe(?:s|d)?\s+the\s+same\b',
        )
        if any(re.search(pattern, normalized) for pattern in relation_patterns):
            return {'modality': 'cross_modal', 'relation': 'consistency'}

        if has_audio_side and not has_visual_side:
            return {'modality': 'audio', 'relation': 'attribute'}
        if has_visual_side and not has_audio_side:
            return {'modality': 'visual', 'relation': 'attribute'}
        return {'modality': 'cross_modal', 'relation': 'attribute'}

    def infer_query_spec(
        self,
        question: str,
        max_new_tokens: Optional[int] = None,
    ) -> Dict[str, str]:
        form_meta = self.classify_question_form(question, max_new_tokens)
        form = str(form_meta.get('form') or 'open_ended')

        spec: Dict[str, str]
        if form in {'single_choice', 'multi_select'}:
            spec = dict(form_meta.get('query_spec') or {})
        elif form == 'yes_no':
            entity_spec = self._extract_entity_spec(question)
            if entity_spec is not None:
                spec = {
                    'modality': str(entity_spec.get('modality') or 'cross_modal'),
                    'relation': str(entity_spec.get('relation') or 'attribute'),
                    'question_kind': str(entity_spec.get('question_kind') or 'entity_grounded'),
                }
            else:
                relation_spec = self._extract_relation_spec(question)
                if relation_spec is not None:
                    spec = {
                        'modality': str(relation_spec.get('modality') or 'cross_modal'),
                        'relation': str(relation_spec.get('relation') or 'consistency'),
                        'question_kind': str(relation_spec.get('question_kind') or 'relation_grounded'),
                    }
                else:
                    spec = self._infer_freeform_query_spec(question)
        else:
            spec = self._infer_freeform_query_spec(question)

        spec = dict(spec or {})
        spec['question_form'] = form
        spec['normalized_question'] = str(
            form_meta.get('normalized_question') or self._normalize(question)
        )
        return spec

    def _canonical_emotion_label(self, text: Optional[str]) -> str:
        normalized = self._normalize(text or '')
        if not normalized:
            return ''
        if normalized in _EMOTION_CANONICAL:
            return _EMOTION_CANONICAL[normalized]
        for token in normalized.split():
            if token in _EMOTION_CANONICAL:
                return _EMOTION_CANONICAL[token]
        return normalized

    def _extract_entity_spec(self, question: str) -> Optional[Dict[str, str]]:
        base = self._strip_yn_instruction(question)
        q = self._normalize(base)
        patterns: List[Tuple[str, str, str]] = [
            (r'^(?:is|are|was|were) the (.+?) visible in the video$', 'visual', 'presence'),
            (r'^(?:is|are|was|were) the (.+?) visible in the scene$', 'visual', 'presence'),
            (r'^(?:is|are|was|were) the (.+?) seen in the video$', 'visual', 'presence'),
            (r'^(?:is|are|was|were) the (.+?) in the video$', 'visual', 'presence'),
            (r'^(?:is|are|was|were) there (.+?) in the video$', 'visual', 'presence'),
            (r'^(?:is|are|was|were) there (.+?) in the scene$', 'visual', 'presence'),
            (r'^can you see (.+?) in the video$', 'visual', 'presence'),
            (r'^can you see (.+?) in the scene$', 'visual', 'presence'),
            (r'^(?:do|does|did) you hear (.+?)$', 'audio', 'sound'),
            (r'^(?:do|does|did) you hear (.+?) in the audio$', 'audio', 'sound'),
            (r'^can you hear (.+?)$', 'audio', 'sound'),
            (r'^can you hear (.+?) in the audio$', 'audio', 'sound'),
            (r'^(?:is|are|was|were) the (.+?) making sound in the audio$', 'audio', 'sound'),
            (r'^(?:is|are|was|were) the (.+?) making noise in the audio$', 'audio', 'sound'),
            (r'^(?:does|do|did) the (.+?) make sound in the audio$', 'audio', 'sound'),
            (r'^(?:does|do|did) the (.+?) make noise in the audio$', 'audio', 'sound'),
            (r'^(?:is|are|was|were) the sound of (.+?) in the audio$', 'audio', 'sound'),
            (r'^(?:is|are|was|were) the noise of (.+?) in the audio$', 'audio', 'sound'),
            (r'^(?:is|are|was|were) the (.+?) audible$', 'audio', 'sound'),
            (r'^(?:is|are|was|were) the (.+?) heard in the audio$', 'audio', 'sound'),
        ]
        for pattern, modality, relation in patterns:
            match = re.match(pattern, q)
            if not match:
                continue
            entity = self._clean_entity_span(match.group(1))
            if not entity:
                continue
            return {
                'question_kind': 'entity_grounded',
                'entity': entity,
                'modality': modality,
                'relation': relation,
                'normalized_question': q,
            }
        return None

    def _extract_relation_spec(self, question: str) -> Optional[Dict[str, str]]:
        base = self._strip_yn_instruction(question)
        q = self._normalize(base)
        tokens = set(q.split())
        has_audio_side = bool(tokens & {'audio', 'sound', 'sounds'})
        has_visual_side = bool(tokens & {'video', 'visual', 'scene', 'scenes', 'frame', 'frames'})
        if not (has_audio_side and has_visual_side):
            return None

        relation_patterns = (
            r'\bmatch(?:es|ed|ing)?\b',
            r'\bconsistent\b',
            r'\bconsistency\b',
            r'\balign(?:ed|ing)?\b',
            r'\bcorrespond(?:s|ed|ing)?\b',
            r'\bfit(?:s|ted|ting)?\b',
            r'\bgo(?:es)?\s+with\b',
        )
        if not any(re.search(pattern, q) for pattern in relation_patterns):
            return None

        return {
            'question_kind': 'relation_grounded',
            'entity': None,
            'modality': 'cross_modal',
            'relation': 'consistency',
            'normalized_question': q,
        }

    def _entity_tokens(self, entity: str) -> List[str]:
        normalized = self._normalize(entity)
        return [tok for tok in normalized.split() if tok and tok not in _STOPWORDS]

    def _head_term(self, entity: str) -> str:
        tokens = self._entity_tokens(entity)
        if not tokens:
            return ''
        return self._singularize(tokens[-1])

    def _is_human_entity(self, entity: str) -> bool:
        tokens = {self._singularize(tok) for tok in self._entity_tokens(entity)}
        return bool(tokens & _HUMAN_TERMS)

    def _is_specific_human_entity(self, entity: str) -> bool:
        return self._head_term(entity) in {'man', 'woman', 'boy', 'girl', 'kid', 'child', 'baby'}

    def _is_generic_human_entity(self, entity: str) -> bool:
        return self._is_human_entity(entity) and not self._is_specific_human_entity(entity)

    def _aliases_for(self, entity: str, modality: str) -> List[str]:
        normalized = self._normalize(entity)
        tokens = self._entity_tokens(entity)
        aliases = {normalized, self._singularize(normalized)} if normalized else set()
        aliases.update(tokens)
        aliases.update(self._singularize(tok) for tok in tokens)
        head = self._head_term(entity)
        if head:
            aliases.add(head)
            hints = _ENTITY_HINTS.get(head, {})
            aliases.update(hints.get(modality, set()))
        if self._is_generic_human_entity(entity):
            aliases.update(_ENTITY_HINTS['person'][modality])
        elif modality == 'visual' and self._is_human_entity(entity):
            aliases.update({'person', 'people', 'human'})
        return [alias for alias in aliases if alias]

    def _audio_only_hints_for(self, entity: str) -> List[str]:
        normalized = self._normalize(entity)
        head = self._head_term(entity)
        candidate_keys = []
        if normalized:
            candidate_keys.append(normalized)
            singular_normalized = self._singularize(normalized)
            if singular_normalized and singular_normalized != normalized:
                candidate_keys.append(singular_normalized)
        if head and head not in candidate_keys:
            candidate_keys.append(head)
        if not candidate_keys:
            return []
        hints = set()
        for key in candidate_keys:
            hints.update(_ENTITY_HINTS.get(key, {}).get('audio', set()))
        aliases = set(hints)
        for key in candidate_keys:
            if key in hints:
                aliases.add(key)
        normalized_aliases: List[str] = []
        seen = set()
        for alias in sorted(aliases):
            canonical = self._singularize(self._normalize(alias))
            if canonical and canonical not in seen:
                normalized_aliases.append(canonical)
                seen.add(canonical)
        return normalized_aliases

    def _audio_proxy_hints_for(self, entity: str) -> List[str]:
        normalized = self._normalize(entity)
        head = self._head_term(entity)
        tokens = self._entity_tokens(entity)
        candidate_keys: List[str] = []
        if normalized:
            candidate_keys.append(normalized)
            singular_normalized = self._singularize(normalized)
            if singular_normalized and singular_normalized != normalized:
                candidate_keys.append(singular_normalized)
        if head and head not in candidate_keys:
            candidate_keys.append(head)
        for token in tokens:
            canonical = self._singularize(self._normalize(token))
            if canonical and canonical not in candidate_keys:
                candidate_keys.append(canonical)

        aliases = set()
        for key in candidate_keys:
            aliases.update(_AUDIO_PROXY_HINTS.get(key, set()))

        if self._is_specific_human_entity(entity) or head in _AUDIO_PROXY_HUMAN_ROLE_TERMS:
            aliases.update({'speech', 'voice', 'speaking'})

        normalized_aliases: List[str] = []
        seen = set()
        for alias in sorted(aliases):
            canonical = self._singularize(self._normalize(alias))
            if canonical and canonical not in seen:
                normalized_aliases.append(canonical)
                seen.add(canonical)
        return normalized_aliases

    def audio_grounding_alias_profile(self, entity: str) -> Dict[str, List[str]]:
        strict_aliases = self._audio_only_hints_for(entity)
        proxy_aliases = []
        if not strict_aliases:
            proxy_aliases = self._audio_proxy_hints_for(entity)
        return {
            'strict': list(strict_aliases),
            'proxy': list(proxy_aliases),
            'combined': list(strict_aliases or proxy_aliases),
        }

    @staticmethod
    def _match_alias(alias: str, candidate: str) -> bool:
        alias = alias.strip()
        candidate = candidate.strip()
        if not alias or not candidate:
            return False
        if alias == candidate:
            return True
        if alias in candidate:
            return True
        alias_tokens = set(alias.split())
        cand_tokens = set(candidate.split())
        overlap = len(alias_tokens & cand_tokens)
        return overlap > 0 and overlap / max(1, len(alias_tokens)) >= 0.5

    def _canonical_prompt_entity(self, text: Optional[str]) -> str:
        label = self._normalize(text or '')
        if not label:
            return ''

        prefixes = (
            'a photo of ',
            'a photo of a ',
            'a close up of ',
            'a video frame containing ',
            'a video frame of ',
            'a scene with ',
            'there is ',
            'there are ',
        )
        suffixes = (
            ' in the scene',
            ' in the video',
            ' visible in the scene',
            ' visible in the video',
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

    def _specific_human_visual_aliases(self, entity: str) -> List[str]:
        normalized = self._normalize(entity)
        head = self._head_term(entity)
        aliases = {normalized, head}

        if head == 'baby':
            aliases.update({'baby', 'infant', 'newborn'})
        elif head in {'child', 'kid'}:
            aliases.update({'child', 'kid'})
        elif head == 'boy':
            aliases.update({'boy'})
        elif head == 'girl':
            aliases.update({'girl'})
        elif head == 'man':
            aliases.update({'man', 'male'})
        elif head == 'woman':
            aliases.update({'woman', 'female'})

        normalized_aliases = []
        seen = set()
        for alias in aliases:
            canonical = self._singularize(self._canonical_prompt_entity(alias))
            if canonical and canonical not in seen:
                normalized_aliases.append(canonical)
                seen.add(canonical)
        return normalized_aliases

    def _presence_alias_profile(
        self,
        presence: Optional[Dict[str, object]],
        aliases: List[str],
    ) -> Dict[str, object]:
        normalized_aliases = {
            self._singularize(self._canonical_prompt_entity(alias))
            for alias in aliases
            if alias
        }
        normalized_aliases.discard('')
        if not normalized_aliases:
            return {
                'grounding_score': 0.0,
                'grounding_peak_score': 0.0,
                'grounding_support_count': 0,
                'clip_score': 0.0,
                'clip_peak_score': 0.0,
                'alias_hit': False,
                'hit_sources': [],
            }

        presence = presence or {}

        def is_target_alias(value: Optional[str]) -> bool:
            canonical = self._singularize(self._canonical_prompt_entity(value))
            return canonical in normalized_aliases

        grounding_score = 0.0
        for alias, score in (presence.get('grounding_scores') or {}).items():
            if is_target_alias(alias):
                grounding_score = max(grounding_score, float(score or 0.0))

        grounding_peak_score = 0.0
        for alias, score in (presence.get('peak_scores') or {}).items():
            if is_target_alias(alias):
                grounding_peak_score = max(grounding_peak_score, float(score or 0.0))

        grounding_support_count = 0
        for alias, count in (presence.get('support_counts') or {}).items():
            if is_target_alias(alias):
                grounding_support_count = max(grounding_support_count, int(count or 0))

        clip_score = 0.0
        for prompt, score in (presence.get('clip_prompt_scores') or {}).items():
            if is_target_alias(prompt):
                clip_score = max(clip_score, float(score or 0.0))

        clip_peak_score = 0.0
        for prompt, score in (presence.get('clip_peak_prompt_scores') or {}).items():
            if is_target_alias(prompt):
                clip_peak_score = max(clip_peak_score, float(score or 0.0))

        hit_sources: List[str] = []
        for key in ('best_alias', 'grounding_peak_alias', 'best_prompt', 'clip_best_prompt', 'clip_peak_prompt'):
            if is_target_alias(presence.get(key)):
                hit_sources.append(key)

        return {
            'grounding_score': float(grounding_score),
            'grounding_peak_score': float(grounding_peak_score),
            'grounding_support_count': int(grounding_support_count),
            'clip_score': float(clip_score),
            'clip_peak_score': float(clip_peak_score),
            'alias_hit': bool(hit_sources),
            'hit_sources': hit_sources,
        }

    def _presence_signature(self, presence: Optional[Dict[str, object]]) -> Dict[str, object]:
        presence = presence or {}
        grounding_score = float(presence.get('grounding_score', 0.0) or 0.0)
        grounding_peak_score = float(presence.get('grounding_peak_score', 0.0) or 0.0)
        grounding_support_count = int(presence.get('grounding_support_count', 0) or 0)
        clip_mean_score = float(presence.get('clip_score', 0.0) or 0.0)
        clip_peak_score = float(presence.get('clip_peak_score', clip_mean_score) or 0.0)
        clip_score = max(
            clip_mean_score,
            min(0.88, 0.60 * clip_mean_score + 0.40 * clip_peak_score),
            float(presence.get('score', 0.0) or 0.0),
        )

        persistent_grounding = grounding_support_count >= 2 and grounding_score >= 0.35
        corroborated_grounding = grounding_peak_score >= 0.34 and max(clip_score, clip_peak_score) >= 0.34 and grounding_peak_score >= 0.25
        transient_clip_support = clip_peak_score >= 0.42 and clip_score >= 0.30
        composite_score = max(grounding_score, clip_score)
        if persistent_grounding and grounding_score > 0.0:
            composite_score = max(
                composite_score,
                min(
                    0.92,
                    0.48
                    + 0.24 * grounding_score
                    + 0.05 * min(grounding_support_count, 3)
                    + 0.10 * clip_mean_score
                    + 0.10 * clip_peak_score,
                ),
            )
        elif corroborated_grounding:
            composite_score = max(
                composite_score,
                min(0.86, 0.28 + 0.24 * grounding_peak_score + 0.12 * clip_mean_score + 0.16 * clip_peak_score),
            )
        elif transient_clip_support:
            composite_score = max(
                composite_score,
                min(0.74, 0.24 + 0.26 * clip_peak_score + 0.22 * clip_score),
            )

        return {
            'grounding_score': grounding_score,
            'grounding_peak_score': grounding_peak_score,
            'grounding_support_count': grounding_support_count,
            'clip_score': clip_score,
            'clip_peak_score': clip_peak_score,
            'persistent_grounding': persistent_grounding,
            'corroborated_grounding': corroborated_grounding,
            'transient_clip_support': transient_clip_support,
            'composite_score': float(min(composite_score, 1.0)),
        }

    def _normalized_audio_evidence(
        self,
        features: ModalityFeatures,
    ) -> List[Tuple[str, float]]:
        if features.audio_event_scores:
            return sorted(
                (
                    (self._normalize(label), float(score))
                    for label, score in features.audio_event_scores.items()
                    if self._normalize(label)
                ),
                key=lambda item: item[1],
                reverse=True,
            )
        if features.audio_events:
            return [
                (self._normalize(label), 0.5)
                for label in features.audio_events
                if self._normalize(label)
            ]
        return []

    def _audio_context_flags(self, features: ModalityFeatures) -> Dict[str, bool]:
        event_pairs = self._normalized_audio_evidence(features)
        event_labels = [label for label, _ in event_pairs]
        asr = self._normalize(features.asr_text or '')

        def has_any(keywords) -> bool:
            for keyword in keywords:
                if keyword in asr:
                    return True
            for label in event_labels:
                for keyword in keywords:
                    if keyword in label or label in keyword:
                        return True
            return False

        speech = bool(
            getattr(features, 'audio_has_speech', False)
            or (features.audio_type or '').lower() == 'speech'
            or has_any({'speech', 'voice', 'talking', 'speaking'})
            or bool(asr)
        )
        water = has_any(_WATER_AUDIO_TERMS | {'wash', 'washing', 'bathroom', 'faucet', 'sink'})
        vehicle = has_any(_VEHICLE_AUDIO_TERMS)
        animal = has_any(_ANIMAL_AUDIO_TERMS)
        music = bool((features.audio_type or '').lower() == 'music' or has_any(_MUSIC_AUDIO_TERMS))
        manipulation = has_any({'holding', 'using', 'hand', 'clean', 'cleaning', 'dry', 'wipe', 'scrub', 'regimen'})

        return {
            'speech': speech,
            'water': water,
            'vehicle': vehicle,
            'animal': animal,
            'music': music,
            'manipulation': manipulation or water,
        }

    def _is_small_object_entity(self, entity: str) -> bool:
        head = self._head_term(entity)
        if not head or self._is_human_entity(entity):
            return False
        large_terms = {
            'person', 'vehicle', 'car', 'truck', 'motorcycle', 'train', 'airplane',
            'helicopter', 'boat', 'dog', 'cat', 'bird', 'horse', 'cow', 'sheep',
            'animal', 'tree', 'building', 'fireworks', 'water',
            # Animals not previously covered
            'goat', 'rooster', 'chicken', 'pigeon', 'dove', 'duck', 'goose',
            'owl', 'pig', 'deer', 'lion', 'tiger', 'bear', 'elephant',
            'monkey', 'frog', 'snake', 'whale', 'dolphin', 'fish',
            # Water bodies & natural phenomena
            'stream', 'river', 'waterfall', 'ocean', 'lake', 'sea', 'rain',
            'thunder', 'lightning', 'flood',
            # Mechanical / large sound sources
            'engine', 'horn', 'siren', 'bus', 'tractor', 'jet',
        }
        return head not in large_terms

    def _allows_contextual_visual_recovery(self, entity: str) -> bool:
        """Whether audio context may rescue borderline visual evidence.

        This is intentionally conservative and benchmark-agnostic: contextual
        audio may help recover manipulable or context-bound objects whose visual
        grounding is weak, but it must not stand in for direct visual evidence
        on large scene entities, animals, vehicles, people, or natural phenomena.
        """
        head = self._head_term(entity)
        if not head or self._is_human_entity(entity):
            return False
        if self._is_small_object_entity(entity):
            return True
        return head in {
            'instrument',
            'guitar',
            'piano',
            'drum',
            'toilet',
            'microwave',
            'oven',
            'kettle',
            'speaker',
            'blender',
            'vacuum',
        }

    def _visual_presence_prompt_templates(
        self,
        entity: str,
        features: ModalityFeatures,
    ) -> List[str]:
        templates = [
            'a photo of {entity}',
            'a close-up of {entity}',
            'a video frame containing {entity}',
            'there is {entity} in the scene',
            'a scene with {entity}',
            '{entity}',
        ]
        if not self._is_human_entity(entity):
            templates.extend(
                [
                    'a small {entity}',
                    'a hand holding {entity}',
                    'a person using {entity}',
                ]
            )

        flags = self._audio_context_flags(features)
        if not self._is_human_entity(entity) and flags['water']:
            templates.extend(
                [
                    'a wet {entity}',
                    '{entity} near water',
                    '{entity} in a sink',
                    '{entity} in a bathroom',
                ]
            )
        if not self._is_human_entity(entity) and flags['music']:
            templates.extend(
                [
                    'a person playing {entity}',
                    '{entity} on stage',
                ]
            )
        if not self._is_human_entity(entity) and flags['vehicle']:
            templates.extend(
                [
                    'a moving {entity}',
                    '{entity} outdoors',
                ]
            )

        deduped = []
        seen = set()
        for template in templates:
            if template not in seen:
                deduped.append(template)
                seen.add(template)
        return deduped

    def _relation_family_candidates(
        self,
        features: ModalityFeatures,
    ) -> List[Tuple[str, float, str]]:
        event_pairs = self._normalized_audio_evidence(features)
        flags = self._audio_context_flags(features)
        candidates: List[Tuple[str, float, str]] = []
        event_labels = [label for label, _ in event_pairs]

        def best_score(keywords, floor: float) -> float:
            best = 0.0
            for label, score in event_pairs:
                if any(keyword in label or label in keyword for keyword in keywords):
                    best = max(best, float(score))
            return max(floor, best) if best > 0.0 or floor > 0.0 else 0.0

        human_voice_specific = any(
            label in {
                'male speech man speaking',
                'female speech woman speaking',
                'child speech kid speaking',
                'baby cry infant cry',
            }
            or label in _GROUP_HUMAN_AUDIO_TERMS
            for label in event_labels
        )

        if human_voice_specific:
            candidates.append(
                ('person', best_score(_GROUP_HUMAN_AUDIO_TERMS | {
                    'male speech man speaking',
                    'female speech woman speaking',
                    'child speech kid speaking',
                    'baby cry infant cry',
                    'voice',
                    'voices',
                }, 0.60), 'family')
            )
        if flags['vehicle']:
            candidates.append(('vehicle', best_score(_VEHICLE_AUDIO_TERMS, 0.56), 'family'))
        if flags['animal']:
            candidates.append(('animal', best_score(_ANIMAL_AUDIO_TERMS, 0.56), 'family'))
        if flags['water']:
            candidates.append(('water', best_score(_WATER_AUDIO_TERMS, 0.54), 'family'))
        if flags['music']:
            candidates.append(('instrument', best_score(_MUSIC_AUDIO_TERMS, 0.56), 'family'))
            if human_voice_specific and not any(cue == 'person' for cue, _, _ in candidates):
                candidates.append(('person', 0.52, 'family'))
        return candidates

    def _score_visual(
        self,
        entity: str,
        features: ModalityFeatures,
        frames: Optional[List],
        object_detector,
    ) -> Dict[str, object]:
        aliases = self._aliases_for(entity, 'visual')
        visual_objects = [self._normalize(obj) for obj in (features.visual_objects or [])]
        matched_objects = sorted({obj for obj in visual_objects for alias in aliases if self._match_alias(alias, obj)})

        face_count = int(getattr(features, 'visual_faces_count', 0) or 0)
        is_human = self._is_human_entity(entity)
        is_specific_human = self._is_specific_human_entity(entity)
        is_generic_human = self._is_generic_human_entity(entity)

        support_score = 0.0
        support_sources: List[str] = []
        object_direct_anchor = bool(matched_objects) and not is_specific_human
        direct_support = object_direct_anchor
        allow_positive_flip = object_direct_anchor
        allow_negative_flip = bool(
            getattr(self.config, 'question_evidence_allow_visual_negative_flip', False)
        )
        weak_visual_cue = False
        proxy_support = False

        if matched_objects:
            if is_specific_human:
                support_score = max(support_score, 0.28)
                support_sources.append(f'generic_person_like_objects={matched_objects}')
            else:
                support_score = max(support_score, 0.9)
                support_sources.append(f'objects={matched_objects}')
            weak_visual_cue = True

        if is_human and face_count > 0:
            if is_specific_human:
                face_score = min(0.20 + 0.04 * float(face_count), 0.32)
            else:
                face_score = min(0.68 + 0.05 * float(face_count), 0.88)
            support_score = max(support_score, face_score)
            support_sources.append(f'faces={face_count}')
            weak_visual_cue = True

        clip_score = 0.0
        clip_info = {'score': 0.0, 'best_prompt': None, 'prompt_scores': {}}
        presence_signature = self._presence_signature(None)
        if frames and object_detector is not None:
            clip_info = object_detector.score_entity_presence(
                frames,
                entity,
                prompt_templates=self._visual_presence_prompt_templates(entity, features),
                aliases=aliases,
            )
            presence_signature = self._presence_signature(clip_info)
            clip_score = float(presence_signature.get('clip_score', 0.0))
            clip_peak_score = float(presence_signature.get('clip_peak_score', 0.0))
            if clip_score >= 0.30:
                clip_cap = 0.42 if is_specific_human else (0.64 if is_human else 0.60)
                support_score = max(support_score, min(clip_score, clip_cap))
                support_sources.append(
                    f"clip={clip_score:.3f}:{clip_info.get('best_alias') or entity}"
                )
            if clip_peak_score >= 0.42:
                support_sources.append(
                    f"clip_peak={clip_peak_score:.3f}:{clip_info.get('best_alias') or entity}"
                )
            grounding_score = float(presence_signature.get('grounding_score', 0.0))
            grounding_peak_score = float(presence_signature.get('grounding_peak_score', 0.0))
            grounding_support_count = int(presence_signature.get('grounding_support_count', 0))
            if grounding_score > 0.0:
                support_sources.append(
                    f'grounding={grounding_score:.3f}:frames={grounding_support_count}'
                )
            elif grounding_peak_score > 0.0:
                support_sources.append(f'grounding_peak={grounding_peak_score:.3f}')
            if clip_score >= 0.28:
                weak_visual_cue = True

        positive_clip_threshold = float(
            getattr(self.config, 'question_evidence_visual_positive_clip_threshold', 0.45)
        )
        generic_human_clip_threshold = float(
            getattr(self.config, 'question_evidence_visual_generic_human_flip_threshold', 0.34)
        )
        specific_human_face_clip_threshold = float(
            getattr(self.config, 'question_evidence_visual_specific_human_face_clip_threshold', 0.32)
        )
        specific_human_clip_threshold = float(
            getattr(self.config, 'question_evidence_visual_specific_human_clip_threshold', 0.46)
        )
        specific_human_proxy_cap = float(
            getattr(self.config, 'question_evidence_visual_specific_human_proxy_cap', 0.40)
        )
        specific_human_typed_grounding_threshold = float(
            getattr(self.config, 'question_evidence_visual_specific_human_typed_grounding_threshold', 0.36)
        )
        specific_human_typed_peak_threshold = float(
            getattr(self.config, 'question_evidence_visual_specific_human_typed_peak_threshold', 0.50)
        )
        contextual_peak_threshold = float(
            getattr(self.config, 'question_evidence_visual_contextual_peak_threshold', 0.28)
        )
        contextual_clip_threshold = float(
            getattr(self.config, 'question_evidence_visual_contextual_clip_threshold', 0.30)
        )
        contextual_clip_peak_threshold = float(
            getattr(self.config, 'question_evidence_visual_contextual_clip_peak_threshold', 0.42)
        )
        context_flags = self._audio_context_flags(features)
        contextual_prompt = (
            (
                clip_info.get('clip_peak_prompt')
                or clip_info.get('clip_best_prompt')
                or clip_info.get('best_prompt')
                or ''
            ).lower()
        )
        contextual_prompt_hit = any(
            phrase in contextual_prompt
            for phrase in ('close-up', 'small ', 'holding ', 'using ', 'bathroom', 'sink', 'water', 'wet ')
        )
        if is_specific_human:
            typed_profile = self._presence_alias_profile(
                clip_info,
                self._specific_human_visual_aliases(entity),
            )
            typed_grounding_score = float(typed_profile.get('grounding_score', 0.0))
            typed_grounding_peak_score = float(typed_profile.get('grounding_peak_score', 0.0))
            typed_grounding_support_count = int(typed_profile.get('grounding_support_count', 0))
            typed_clip_score = float(typed_profile.get('clip_score', 0.0))
            typed_clip_peak_score = float(typed_profile.get('clip_peak_score', 0.0))
            typed_alias_hit = bool(typed_profile.get('alias_hit', False))
            typed_persistent_grounding = (
                typed_grounding_support_count >= 2
                and typed_grounding_score >= specific_human_typed_grounding_threshold
            )
            typed_corroborated_grounding = (
                typed_grounding_peak_score >= 0.30
                and max(typed_clip_score, typed_clip_peak_score) >= specific_human_face_clip_threshold
            )

            if typed_persistent_grounding:
                support_score = max(
                    support_score,
                    min(
                        0.86,
                        0.66
                        + 0.10 * typed_grounding_score
                        + 0.04 * min(typed_grounding_support_count, 3)
                        + 0.03 * min(face_count, 2),
                    ),
                )
                support_sources.append(
                    'specific_human_typed_grounding='
                    f'clip{typed_clip_score:.3f}:frames{typed_grounding_support_count}'
                )
                direct_support = True
                allow_positive_flip = True
                weak_visual_cue = True
            elif typed_corroborated_grounding and (
                face_count > 0 or typed_clip_peak_score >= specific_human_typed_peak_threshold
            ):
                support_score = max(
                    support_score,
                    min(
                        0.82,
                        0.58
                        + 0.12 * typed_grounding_peak_score
                        + 0.08 * typed_clip_score
                        + 0.08 * typed_clip_peak_score,
                    ),
                )
                support_sources.append(
                    'specific_human_typed_grounding_clip='
                    f'peak{typed_grounding_peak_score:.3f}:clip{typed_clip_score:.3f}:peakclip{typed_clip_peak_score:.3f}'
                )
                direct_support = True
                allow_positive_flip = True
                weak_visual_cue = True
            elif (
                face_count > 0
                and typed_alias_hit
                and typed_clip_score >= specific_human_face_clip_threshold
                and typed_clip_peak_score >= specific_human_typed_peak_threshold
            ):
                support_score = max(
                    support_score,
                    min(
                        0.80,
                        0.56
                        + 0.04 * min(face_count, 3)
                        + 0.12 * typed_clip_score
                        + 0.08 * typed_clip_peak_score,
                    ),
                )
                support_sources.append(
                    f'specific_human_face_typed_clip=faces{face_count}:clip{typed_clip_score:.3f}:peak{typed_clip_peak_score:.3f}'
                )
                direct_support = True
                allow_positive_flip = True
                weak_visual_cue = True
            elif (
                typed_alias_hit
                and typed_clip_score >= specific_human_clip_threshold
                and typed_clip_peak_score >= specific_human_typed_peak_threshold
            ):
                support_sources.append(
                    f'specific_human_typed_clip_weak={typed_clip_score:.3f}:peak{typed_clip_peak_score:.3f}'
                )
                weak_visual_cue = True
            elif clip_score >= specific_human_clip_threshold:
                support_sources.append(f'specific_human_generic_clip_weak={clip_score:.3f}')
                weak_visual_cue = True

            if not direct_support and weak_visual_cue:
                proxy_support = True
                uncapped_support = support_score
                support_score = min(support_score, specific_human_proxy_cap)
                if support_score + 1e-6 < uncapped_support:
                    support_sources.append(
                        f'specific_human_proxy_cap={uncapped_support:.3f}->{support_score:.3f}'
                    )
        elif is_generic_human and face_count > 0:
            if presence_signature['persistent_grounding']:
                support_score = max(
                    support_score,
                    min(
                        0.84,
                        0.62
                        + 0.10 * presence_signature['grounding_score']
                        + 0.04 * min(face_count, 3),
                    ),
                )
                support_sources.append(
                    'generic_human_face_grounding='
                    f'faces{face_count}:frames{presence_signature["grounding_support_count"]}'
                )
                direct_support = True
                allow_positive_flip = True
            elif (
                presence_signature['corroborated_grounding']
                or clip_score >= generic_human_clip_threshold
            ):
                support_score = max(
                    support_score,
                    min(
                        0.80,
                        0.58
                        + 0.10 * presence_signature['grounding_peak_score']
                        + 0.12 * clip_score,
                    ),
                )
                support_sources.append(
                    f'generic_human_face_clip=faces{face_count}:clip{clip_score:.3f}'
                )
                direct_support = True
                allow_positive_flip = True
        else:
            if matched_objects and (
                presence_signature['persistent_grounding'] or clip_score >= positive_clip_threshold
            ):
                support_score = max(
                    support_score,
                    min(
                        0.90,
                        0.68
                        + 0.08 * presence_signature['grounding_peak_score']
                        + 0.10 * clip_score,
                    ),
                )
                support_sources.append(
                    'inventory_grounding_agreement='
                    f'objects={matched_objects}:clip{clip_score:.3f}'
                )
                direct_support = True
                allow_positive_flip = True
                weak_visual_cue = True
            elif presence_signature['persistent_grounding']:
                support_score = max(
                    support_score,
                    min(
                        0.86,
                        0.60
                        + 0.16 * presence_signature['grounding_score']
                        + 0.04 * min(presence_signature['grounding_support_count'], 3),
                    ),
                )
                support_sources.append(
                    'grounding_persistent='
                    f'frames{presence_signature["grounding_support_count"]}:score{presence_signature["grounding_score"]:.3f}'
                )
                direct_support = True
                allow_positive_flip = True
                weak_visual_cue = True
            elif (
                presence_signature['corroborated_grounding']
                and presence_signature['composite_score'] >= 0.66
            ):
                support_score = max(
                    support_score,
                    min(
                        0.80,
                        0.52
                        + 0.18 * presence_signature['grounding_peak_score']
                        + 0.12 * clip_score,
                    ),
                )
                support_sources.append(
                    'grounding_clip_corroboration='
                    f'peak{presence_signature["grounding_peak_score"]:.3f}:clip{clip_score:.3f}'
                )
                direct_support = True
                allow_positive_flip = True
                weak_visual_cue = True
            elif (
                self._is_small_object_entity(entity)
                and context_flags['manipulation']
                and contextual_prompt_hit
                and clip_score >= contextual_clip_threshold
                and clip_peak_score >= contextual_clip_peak_threshold
                and (
                    presence_signature['grounding_peak_score'] >= contextual_peak_threshold
                    or presence_signature['transient_clip_support']
                )
            ):
                support_score = max(
                    support_score,
                    min(
                        0.84,
                        0.46
                        + 0.12 * presence_signature['grounding_peak_score']
                        + 0.12 * clip_score
                        + 0.16 * clip_peak_score,
                    ),
                )
                support_sources.append(
                    'contextual_small_object='
                    f'peak{presence_signature["grounding_peak_score"]:.3f}:clip{clip_score:.3f}:peakclip{clip_peak_score:.3f}:{contextual_prompt[:48]}'
                )
                direct_support = True
                allow_positive_flip = True
                weak_visual_cue = True
            elif clip_score >= positive_clip_threshold:
                support_sources.append(f'clip_only_weak={clip_score:.3f}')
                weak_visual_cue = True

        contradiction_score = 0.0
        contradiction_reasons: List[str] = []
        if is_human:
            if (
                face_count <= 0
                and presence_signature['composite_score'] < 0.24
                and not matched_objects
            ):
                contradiction_score = max(contradiction_score, 0.75)
                contradiction_reasons.append('no_faces_or_person_like_visual_support')
        elif (
            presence_signature['grounding_peak_score'] < 0.20
            and clip_score < 0.22
            and visual_objects
            and not matched_objects
        ):
            contradiction_score = max(contradiction_score, 0.62)
            contradiction_reasons.append('low_clip_and_no_object_overlap')

        return {
            'support_score': float(min(support_score, 1.0)),
            'contradiction_score': float(min(contradiction_score, 1.0)),
            'matched_objects': matched_objects,
            'support_sources': support_sources,
            'contradiction_reasons': contradiction_reasons,
            'clip_score': clip_score,
            'clip_peak_score': float(presence_signature.get('clip_peak_score', 0.0)),
            'clip_prompt': clip_info.get('best_prompt'),
            'grounding_score': presence_signature['grounding_score'],
            'grounding_peak_score': presence_signature['grounding_peak_score'],
            'grounding_support_count': presence_signature['grounding_support_count'],
            'support_frame_indices': list(clip_info.get('grounding_support_frame_indices', [])),
            'peak_frame_index': int(clip_info.get('grounding_peak_frame_index', -1) or -1),
            'grounding_max_frames': int(clip_info.get('grounding_max_frames', 0) or 0),
            'presence_composite_score': presence_signature['composite_score'],
            'direct_support': direct_support,
            'proxy_support': proxy_support and not direct_support,
            'weak_visual_cue': weak_visual_cue,
            'face_count': face_count,
            'allow_positive_flip': allow_positive_flip,
            'allow_negative_flip': allow_negative_flip,
        }

    def _score_audio(
        self,
        entity: str,
        features: ModalityFeatures,
    ) -> Dict[str, object]:
        strict_aliases = self._aliases_for(entity, 'audio')
        proxy_aliases = self._audio_proxy_hints_for(entity)
        strict_alias_set = {self._normalize(alias) for alias in strict_aliases if self._normalize(alias)}
        proxy_aliases = [
            alias for alias in proxy_aliases
            if self._normalize(alias) and self._normalize(alias) not in strict_alias_set
        ]
        normalized_events = {
            self._normalize(label): float(score)
            for label, score in (features.audio_event_scores or {}).items()
        }
        if not normalized_events and features.audio_events:
            normalized_events = {self._normalize(label): 0.5 for label in features.audio_events}
        event_sources: Dict[str, set] = {}
        for item in getattr(features, 'audio_event_timeline', []) or []:
            label = self._normalize(str(item.get('label', '')))
            if not label:
                continue
            source = str(item.get('source', '') or 'unknown').strip().lower()
            event_sources.setdefault(label, set()).add(source)

        support_score = 0.0
        support_sources: List[str] = []
        direct_support = False
        proxy_support = False
        asr_direct_support = False

        normalized_asr = self._normalize(features.asr_text or '')
        if normalized_asr:
            for alias in strict_aliases:
                if alias and alias in normalized_asr:
                    support_score = max(support_score, 0.92)
                    support_sources.append(f'asr={alias}')
                    direct_support = True
                    asr_direct_support = True

        matched_events = []
        proxy_matched_events = []
        for label, score in normalized_events.items():
            for alias in strict_aliases:
                if self._match_alias(alias, label):
                    event_score = max(0.55, min(0.95, 0.45 + float(score)))
                    support_score = max(support_score, event_score)
                    matched_events.append((label, score))
                    break
        if matched_events:
            qg_only_labels = [
                label for label, _ in matched_events
                if event_sources.get(label) and event_sources.get(label, set()) <= {'audio_query_grounding'}
            ]
            non_qg_labels = [
                label for label, _ in matched_events
                if not event_sources.get(label) or not event_sources.get(label, set()) <= {'audio_query_grounding'}
            ]
            if non_qg_labels:
                support_sources.append('audio_events=' + ','.join(label for label in non_qg_labels[:3]))
            if qg_only_labels:
                support_sources.append('query_grounding_proposal=' + ','.join(label for label in qg_only_labels[:3]))
                if not non_qg_labels and not asr_direct_support:
                    # CLAP-style audio-text similarity proposes where relevant
                    # evidence may be, but alone it is not a source-attribution
                    # verifier.  Keep the signal for localization while blocking
                    # positive answer commitment.
                    support_score = min(support_score, 0.44)
            proxy_support = True

        if proxy_aliases:
            for label, score in normalized_events.items():
                for alias in proxy_aliases:
                    if self._match_alias(alias, label):
                        proxy_matched_events.append((label, score))
                        break
        if proxy_matched_events and not matched_events:
            # Proxy/context sounds (e.g. wind for trees, engine for road) are
            # useful evidence that the audio contains related context, but they
            # do not by themselves verify that the queried entity is the sound
            # source.  Keep them non-decisive for positive flips.
            proxy_support = True
            proxy_score = max(float(score) for _, score in proxy_matched_events)
            support_score = max(support_score, min(0.34, 0.16 + 0.22 * proxy_score))
            support_sources.append('proxy_audio_events=' + ','.join(label for label, _ in proxy_matched_events[:3]))
        elif proxy_matched_events:
            support_sources.append('proxy_audio_events=' + ','.join(label for label, _ in proxy_matched_events[:3]))

        audio_type = (features.audio_type or '').lower()
        is_human = self._is_human_entity(entity)
        is_generic_human = self._is_generic_human_entity(entity)
        is_specific_human = self._is_specific_human_entity(entity)
        matched_labels = {label for label, _ in matched_events}
        head = self._head_term(entity)

        if is_human:
            if matched_labels & _GROUP_HUMAN_AUDIO_TERMS:
                if asr_direct_support or any(
                    not event_sources.get(label) or not event_sources.get(label, set()) <= {'audio_query_grounding'}
                    for label in matched_labels & _GROUP_HUMAN_AUDIO_TERMS
                ):
                    direct_support = True
            elif is_specific_human and matched_labels & {
                'male speech man speaking',
                'female speech woman speaking',
                'child speech kid speaking',
                'baby cry infant cry',
            }:
                if asr_direct_support or any(
                    not event_sources.get(label) or not event_sources.get(label, set()) <= {'audio_query_grounding'}
                    for label in matched_labels
                ):
                    direct_support = True
        elif matched_labels:
            has_non_qg_source = any(
                not event_sources.get(label) or not event_sources.get(label, set()) <= {'audio_query_grounding'}
                for label in matched_labels
            )
            if has_non_qg_source and any(label not in _GENERIC_AUDIO_PROXY_TERMS for label in matched_labels):
                direct_support = True

        if is_generic_human and getattr(features, 'audio_has_speech', False):
            support_score = max(support_score, 0.62)
            support_sources.append('audio_has_speech')
            proxy_support = True

        if head == 'music' and audio_type == 'music':
            support_score = max(support_score, 0.88)
            support_sources.append('audio_type=music')
            direct_support = True

        if proxy_support and not direct_support:
            proxy_cap = 0.68 if is_human else 0.72
            support_score = min(support_score, proxy_cap)

        contradiction_score = 0.0
        contradiction_reasons: List[str] = []
        negative_evidence_ready = False
        if audio_type == 'silence':
            contradiction_score = max(contradiction_score, 0.95)
            contradiction_reasons.append('audio_is_silence')
            negative_evidence_ready = True
        elif proxy_matched_events and not matched_events:
            proxy_score = max(float(score) for _, score in proxy_matched_events)
            contradiction_score = max(contradiction_score, min(0.72, 0.42 + 0.32 * proxy_score))
            contradiction_reasons.append(
                'proxy_audio_without_source_attribution='
                + ','.join(label for label, _ in proxy_matched_events[:3])
            )
            negative_evidence_ready = True
        elif is_human and not getattr(features, 'audio_has_speech', False) and support_score < 0.5:
            contradiction_score = max(contradiction_score, 0.82)
            contradiction_reasons.append('no_speech_evidence_for_human_sound_query')
            negative_evidence_ready = True
        elif is_specific_human and not direct_support:
            if normalized_asr:
                # ASR text is present → someone is audibly speaking.  Treat
                # this as proxy support rather than contradiction; the lack
                # of gender-specific AST labels does not disprove the human
                # is making sound.
                support_score = max(support_score, 0.58)
                support_sources.append('asr_speech_proxy_for_specific_human')
                proxy_support = True
            elif getattr(features, 'audio_has_speech', False):
                # Speech flag without ASR — weaker signal. Use a low
                # contradiction that cannot trigger risk_negative on its own.
                contradiction_score = max(contradiction_score, 0.42)
                contradiction_reasons.append('generic_speech_without_specific_human_cues')
            elif normalized_events and not matched_events and support_score < 0.45:
                contradiction_score = max(contradiction_score, 0.80)
                contradiction_reasons.append('generic_audio_without_specific_human_cues')
                negative_evidence_ready = True
        elif head in {'car', 'motorcycle', 'train', 'airplane', 'helicopter', 'boat', 'mower'}:
            if normalized_events and not matched_events and support_score < 0.45:
                contradiction_score = max(contradiction_score, 0.80)
                contradiction_reasons.append('mechanical_entity_without_matching_audio_cues')
                negative_evidence_ready = True
            elif audio_type == 'speech' and support_score < 0.45 and normalized_events:
                contradiction_score = max(contradiction_score, 0.70)
                contradiction_reasons.append('speech_dominant_without_mechanical_audio_cues')
                negative_evidence_ready = True
        elif head in {'dog', 'cat', 'bird', 'horse', 'cow', 'sheep'}:
            if audio_type == 'music' and support_score < 0.45:
                contradiction_score = max(contradiction_score, 0.60)
                contradiction_reasons.append('music_dominant_without_animal_audio_cues')
                negative_evidence_ready = True

        allow_positive_flip = direct_support
        allow_negative_flip = bool(negative_evidence_ready)

        return {
            'support_score': float(min(support_score, 1.0)),
            'contradiction_score': float(min(contradiction_score, 1.0)),
            'support_sources': support_sources,
            'contradiction_reasons': contradiction_reasons,
            'matched_events': matched_events[:5],
            'proxy_matched_events': proxy_matched_events[:5],
            'strict_audio_aliases': strict_aliases[:12],
            'proxy_audio_aliases': proxy_aliases[:12],
            'direct_support': direct_support,
            'proxy_support': proxy_support and not direct_support,
            'allow_positive_flip': allow_positive_flip,
            'allow_negative_flip': allow_negative_flip,
            'negative_evidence_ready': negative_evidence_ready,
        }

    def _apply_specific_human_shadow_guard(
        self,
        entity: Optional[str],
        modality: str,
        evidence: Dict[str, object],
        features: ModalityFeatures,
        *,
        frames: Optional[List] = None,
        object_detector=None,
    ) -> Dict[str, object]:
        if modality not in {'visual', 'audio'}:
            return evidence
        if not entity or not self._is_specific_human_entity(entity):
            return evidence
        if bool(evidence.get('direct_support', False)):
            return evidence
        if not bool(evidence.get('proxy_support', False) or evidence.get('weak_visual_cue', False)):
            return evidence

        shadow_support_threshold = float(
            getattr(self.config, 'question_evidence_specific_human_shadow_support_threshold', 0.72)
        )
        shadow_proxy_cap = float(
            getattr(self.config, 'question_evidence_specific_human_shadow_proxy_cap', 0.30)
        )
        shadow_contradiction_base = float(
            getattr(self.config, 'question_evidence_specific_human_shadow_contradiction_base', 0.44)
        )
        shadow_contradiction_gain = float(
            getattr(self.config, 'question_evidence_specific_human_shadow_contradiction_gain', 0.34)
        )

        if modality == 'visual':
            counterpart_modality = 'audio'
            counterpart = self._score_audio(entity, features)
        else:
            counterpart_modality = 'visual'
            counterpart = self._score_visual(entity, features, frames, object_detector)

        counterpart_support = float(counterpart.get('support_score', 0.0) or 0.0)
        counterpart_contradiction = float(counterpart.get('contradiction_score', 0.0) or 0.0)
        counterpart_direct = bool(counterpart.get('direct_support', False))
        counterpart_ready = counterpart_direct or counterpart_support >= shadow_support_threshold
        if not counterpart_ready:
            return evidence

        updated = dict(evidence)
        support_sources = list(updated.get('support_sources', []) or [])
        contradiction_reasons = list(updated.get('contradiction_reasons', []) or [])
        uncapped_support = float(updated.get('support_score', 0.0) or 0.0)
        shadow_strength = counterpart_support if counterpart_support > 0.0 else shadow_support_threshold
        shadow_contradiction = min(
            0.90,
            shadow_contradiction_base + shadow_contradiction_gain * max(shadow_support_threshold, shadow_strength),
        )

        updated['support_score'] = float(min(uncapped_support, shadow_proxy_cap))
        updated['contradiction_score'] = float(
            max(float(updated.get('contradiction_score', 0.0) or 0.0), shadow_contradiction)
        )
        updated['allow_positive_flip'] = False
        updated['cross_modal_shadow_guard'] = {
            'applied': True,
            'queried_modality': modality,
            'counterpart_modality': counterpart_modality,
            'counterpart_support_score': counterpart_support,
            'counterpart_contradiction_score': counterpart_contradiction,
            'counterpart_direct_support': counterpart_direct,
        }
        if updated['support_score'] + 1e-6 < uncapped_support:
            support_sources.append(
                f'specific_human_shadow_cap={uncapped_support:.3f}->{updated["support_score"]:.3f}'
            )
        support_sources.append(
            f'specific_human_shadow_from_{counterpart_modality}={counterpart_support:.3f}:direct={int(counterpart_direct)}'
        )
        contradiction_reasons.append(
            f'specific_human_cross_modal_shadow_from_{counterpart_modality}'
        )
        updated['support_sources'] = support_sources
        updated['contradiction_reasons'] = contradiction_reasons
        return updated

    def _apply_nondecisive_support_calibration(
        self,
        modality: str,
        evidence: Dict[str, object],
    ) -> Dict[str, object]:
        if modality not in {'visual', 'audio'}:
            return evidence
        if bool(evidence.get('allow_positive_flip', True)) or bool(evidence.get('direct_support', False)):
            return evidence

        support_score = float(evidence.get('support_score', 0.0) or 0.0)
        if support_score <= 0.0:
            return evidence

        proxy_support = bool(evidence.get('proxy_support', False))
        if modality == 'visual':
            support_cap = float(
                getattr(
                    self.config,
                    'question_evidence_visual_nondecisive_proxy_cap' if proxy_support else 'question_evidence_visual_nondecisive_support_cap',
                    0.30 if proxy_support else 0.36,
                )
            )
            risk = float(getattr(self.config, 'question_evidence_visual_nondecisive_risk', 0.18))
        else:
            support_cap = float(
                getattr(
                    self.config,
                    'question_evidence_audio_nondecisive_proxy_cap' if proxy_support else 'question_evidence_audio_nondecisive_support_cap',
                    0.36 if proxy_support else 0.44,
                )
            )
            risk = float(getattr(self.config, 'question_evidence_audio_nondecisive_risk', 0.12))

        updated = dict(evidence)
        support_sources = list(updated.get('support_sources', []) or [])
        calibrated_support = min(support_score, support_cap)
        if calibrated_support + 1e-6 < support_score:
            support_sources.append(
                f'nondecisive_support_cap={support_score:.3f}->{calibrated_support:.3f}'
            )
        updated['support_score'] = float(calibrated_support)
        updated['nondecisive_risk'] = float(max(float(updated.get('nondecisive_risk', 0.0) or 0.0), risk))
        updated['support_calibration'] = {
            'applied': True,
            'modality': modality,
            'proxy_support': proxy_support,
            'original_support_score': support_score,
            'calibrated_support_score': float(calibrated_support),
            'nondecisive_risk': float(updated['nondecisive_risk']),
        }
        updated['support_sources'] = support_sources
        return updated

    def _relation_audio_visual_alignment(
        self,
        features: ModalityFeatures,
        frames: Optional[List],
        object_detector,
    ) -> Dict[str, object]:
        if not frames or object_detector is None:
            return {'score': 0.0, 'cue': None, 'prompt': None, 'cues': [], 'cue_results': []}

        ranked_events = []
        if features.audio_event_scores:
            ranked_events = sorted(
                ((self._normalize(label), float(score)) for label, score in features.audio_event_scores.items()),
                key=lambda item: item[1],
                reverse=True,
            )
        elif features.audio_events:
            ranked_events = [(self._normalize(label), 0.5) for label in features.audio_events]

        cues = []
        for label, _ in ranked_events:
            if not label or label in _RELATION_AUDIO_STOPWORDS:
                continue
            if len(label) <= 2 or label in cues:
                continue
            cues.append(label)
            if len(cues) >= 3:
                break

        if not cues:
            cues = []

        cue_entries: List[Tuple[str, float, str]] = [(cue, score, 'specific') for cue, score in ranked_events if cue in cues]
        family_entries = self._relation_family_candidates(features)
        seen_cues = {cue for cue, _, _ in cue_entries}
        for cue, score, cue_type in family_entries:
            if cue not in seen_cues:
                cue_entries.append((cue, score, cue_type))
                seen_cues.add(cue)

        if not cue_entries:
            return {'score': 0.0, 'cue': None, 'prompt': None, 'cues': [], 'cue_results': []}

        visual_objects = [self._normalize(obj) for obj in (features.visual_objects or [])]
        cue_results = []
        face_count = int(getattr(features, 'visual_faces_count', 0) or 0)
        for cue, audio_score, cue_type in cue_entries:
            aliases = self._aliases_for(cue, 'visual')
            presence = object_detector.score_entity_presence(frames, cue, aliases=aliases)
            presence_signature = self._presence_signature(presence)
            matched_visual_cues = sorted(
                {
                    obj
                    for obj in visual_objects
                    for alias in aliases
                    if self._match_alias(alias, obj)
                }
            )
            specific_cue = cue_type == 'specific' and bool(cue and cue not in _RELATION_GENERIC_CUES and len(cue) > 3)
            inventory_overlap = bool(matched_visual_cues)
            if cue_type == 'family' and cue == 'person':
                direct_anchor = bool(
                    face_count > 0
                    and (
                        presence_signature['persistent_grounding']
                        or presence_signature['corroborated_grounding']
                        or presence_signature['clip_score'] >= 0.28
                    )
                )
                typed_proxy_anchor = bool(
                    not direct_anchor
                    and (
                        face_count > 0
                        or presence_signature['composite_score'] >= 0.50
                    )
                )
                if direct_anchor:
                    support_score = max(
                        0.66,
                        min(
                            0.82,
                            0.56 + 0.05 * min(face_count, 3) + 0.12 * presence_signature['clip_score'],
                        ),
                    )
                elif typed_proxy_anchor:
                    support_score = max(
                        0.56,
                        min(
                            0.72,
                            0.48 + 0.04 * min(face_count, 3) + 0.10 * presence_signature['clip_score'],
                        ),
                    )
                else:
                    support_score = float(min(presence_signature['composite_score'], 0.54))
            elif cue_type == 'family':
                direct_anchor = bool(
                    presence_signature['persistent_grounding'] or inventory_overlap
                )
                typed_proxy_anchor = bool(
                    not direct_anchor
                    and (
                        presence_signature['corroborated_grounding']
                        or presence_signature['composite_score'] >= 0.54
                    )
                )
                if direct_anchor:
                    support_score = max(
                        0.68 if inventory_overlap else 0.64,
                        min(0.84, presence_signature['composite_score']),
                    )
                elif typed_proxy_anchor:
                    support_score = max(
                        0.56,
                        min(
                            0.74,
                            0.46
                            + 0.16 * presence_signature['grounding_peak_score']
                            + 0.12 * presence_signature['clip_score'],
                        ),
                    )
                else:
                    support_score = float(min(presence_signature['composite_score'], 0.54))
            else:
                direct_anchor = bool(
                    specific_cue
                    and (
                        presence_signature['persistent_grounding']
                        or (inventory_overlap and presence_signature['clip_score'] >= 0.32)
                    )
                )
                typed_proxy_anchor = bool(
                    specific_cue
                    and not direct_anchor
                    and (
                        inventory_overlap
                        or presence_signature['corroborated_grounding']
                        or presence_signature['composite_score'] >= 0.62
                    )
                )
                if direct_anchor:
                    support_score = max(
                        0.82 if inventory_overlap else 0.78,
                        min(0.92, presence_signature['composite_score']),
                    )
                elif typed_proxy_anchor:
                    support_score = max(
                        0.68,
                        min(
                            0.80,
                            0.50
                            + 0.18 * presence_signature['grounding_peak_score']
                            + 0.10 * presence_signature['clip_score'],
                        ),
                    )
                else:
                    support_score = float(min(presence_signature['composite_score'], 0.58))

            cue_results.append(
                {
                    'cue': cue,
                    'audio_score': float(audio_score),
                    'score': float(support_score),
                    'prompt': presence.get('best_prompt'),
                    'cue_type': cue_type,
                    'presence': presence,
                    'presence_signature': presence_signature,
                    'matched_visual_cues': matched_visual_cues,
                    'direct_anchor': direct_anchor,
                    'typed_proxy_anchor': typed_proxy_anchor,
                    'specific_cue': specific_cue,
                }
            )

        if not cue_results:
            return {'score': 0.0, 'cue': None, 'prompt': None, 'cues': cues, 'cue_results': []}

        best = max(
            cue_results,
            key=lambda item: (item['score'], item['presence_signature']['composite_score'], item['audio_score']),
        )
        multi_cue_threshold = float(
            getattr(self.config, 'relation_evidence_multi_cue_threshold', 0.45)
        )
        multi_cue_bonus = float(
            getattr(self.config, 'relation_evidence_multi_cue_bonus', 0.08)
        )
        moderate_results = [item for item in cue_results if item['score'] >= multi_cue_threshold]
        specific_results = [item for item in moderate_results if item.get('cue_type') == 'specific']
        family_results = [item for item in moderate_results if item.get('cue_type') == 'family']
        aggregate_score = float(best['score'])
        if len(moderate_results) >= 2 and specific_results:
            aggregate_score = min(
                0.90,
                aggregate_score + multi_cue_bonus * min(len(moderate_results) - 1, 2),
            )
        if specific_results and family_results:
            aggregate_score = min(0.90, aggregate_score + 0.05)
        specific_direct_results = [
            item for item in cue_results
            if item.get('cue_type') == 'specific' and item.get('direct_anchor')
        ]
        aggregate_direct_anchor = bool(specific_direct_results)
        aggregate_proxy_anchor = bool(
            any(item.get('typed_proxy_anchor') for item in cue_results)
            or (len(moderate_results) >= 2 and bool(specific_results))
        )
        return {
            'score': float(aggregate_score),
            'cue': best.get('cue'),
            'prompt': best.get('prompt'),
            'cues': cues,
            'cue_results': cue_results,
            'matched_visual_cues': best.get('matched_visual_cues', []),
            'direct_anchor': aggregate_direct_anchor,
            'typed_proxy_anchor': aggregate_proxy_anchor,
            'specific_cue': bool(best.get('specific_cue', False)),
            'presence_signature': best.get('presence_signature') or {},
            'multi_cue_support': len(moderate_results) >= 2,
            'moderate_cue_count': len(moderate_results),
            'specific_direct_count': len(specific_direct_results),
            'specific_support_count': len(specific_results),
            'family_support_count': len(family_results),
        }

    def _score_relation_consistency(
        self,
        features: ModalityFeatures,
        conflict_report,
        frames: Optional[List],
        object_detector,
    ) -> Dict[str, object]:
        support_score = 0.0
        contradiction_score = 0.0
        support_sources: List[str] = []
        contradiction_reasons: List[str] = []
        allow_positive_flip = False
        negative_evidence_ready = False
        allow_negative_flip = False
        affect_risk = 0.0

        if conflict_report is None:
            return {
                'support_score': 0.0,
                'contradiction_score': 0.0,
                'support_sources': [],
                'contradiction_reasons': ['missing_conflict_report'],
                'allow_positive_flip': False,
                'allow_negative_flip': False,
                'affect_risk': 0.0,
                'alignment': {'score': 0.0, 'cue': None, 'prompt': None, 'cues': [], 'cue_results': []},
            }

        if conflict_report.audio_video_content_conflict:
            contradiction_score = max(contradiction_score, 0.9)
            contradiction_reasons.append('audio_video_content_conflict')
            negative_evidence_ready = True

        strong_emotion_distance = 1.1
        emotion_distance = float(getattr(conflict_report, 'emotion_distance', 0.0) or 0.0)
        emotion_consistency = float(getattr(conflict_report, 'emotion_consistency_score', 1.0) or 0.0)
        content_consistency = float(getattr(conflict_report, 'content_consistency_score', 0.0) or 0.0)

        if conflict_report.audio_video_emotion_conflict:
            affect_risk = max(
                affect_risk,
                0.75 if emotion_distance >= strong_emotion_distance else 0.60,
            )

        if emotion_consistency <= 0.34:
            affect_risk = max(affect_risk, 0.58)

        if content_consistency > 0.0 and content_consistency <= 0.18:
            contradiction_score = max(contradiction_score, 0.62)
            contradiction_reasons.append(f'low_content_consistency={content_consistency:.2f}')
            negative_evidence_ready = True

        alignment = self._relation_audio_visual_alignment(features, frames, object_detector)
        alignment_score = float(alignment.get('score', 0.0))
        alignment_cue = self._normalize(alignment.get('cue') or '')
        direct_anchor = bool(alignment.get('direct_anchor', False))
        typed_proxy_anchor = bool(alignment.get('typed_proxy_anchor', False))
        specific_support_count = int(alignment.get('specific_support_count', 0) or 0)
        family_support_count = int(alignment.get('family_support_count', 0) or 0)
        if direct_anchor:
            support_score = max(
                support_score,
                max(
                    alignment_score,
                    float(getattr(self.config, 'relation_evidence_audio_prompt_support_score', 0.84)),
                ),
            )
            support_sources.append(
                'direct_audio_visual_anchor='
                f'{alignment.get("cue")}:{alignment_score:.2f}:objects={alignment.get("matched_visual_cues", [])}'
            )
            allow_positive_flip = True
        elif (
            typed_proxy_anchor
            and specific_support_count > 0
            and not conflict_report.audio_video_content_conflict
        ):
            support_score = max(
                support_score,
                min(
                    float(getattr(self.config, 'relation_evidence_no_object_support_score', 0.82)),
                    alignment_score,
                ),
            )
            support_sources.append(
                'typed_proxy_alignment='
                f'{alignment.get("cue")}:{alignment_score:.2f}:objects={alignment.get("matched_visual_cues", [])}'
            )
            allow_positive_flip = True
        elif alignment_cue:
            support_sources.append(
                f'weak_alignment={alignment.get("cue")}:{alignment_score:.2f}'
            )
        if alignment.get('multi_cue_support'):
            support_sources.append(
                f'multi_cue_alignment=count{alignment.get("moderate_cue_count", 0)}:{alignment_score:.2f}'
            )
        if family_support_count > 0 and specific_support_count <= 0:
            support_sources.append('family_only_alignment')

        # Affect conflict is a risk prior for AV matching, not standalone
        # contradiction evidence. A Yes -> No flip requires explicit content
        # mismatch evidence rather than emotion-only disagreement.
        allow_negative_flip = bool(negative_evidence_ready)

        return {
            'support_score': float(min(support_score, 1.0)),
            'contradiction_score': float(min(contradiction_score, 1.0)),
            'support_sources': support_sources,
            'contradiction_reasons': contradiction_reasons,
            'allow_positive_flip': allow_positive_flip,
            'allow_negative_flip': allow_negative_flip,
            'negative_evidence_ready': negative_evidence_ready,
            'direct_anchor': direct_anchor,
            'typed_proxy_anchor': typed_proxy_anchor,
            'alignment_score': alignment_score,
            'alignment_cue': alignment.get('cue'),
            'alignment': alignment,
            'affect_risk': affect_risk,
        }

    def _score_emotion_option(
        self,
        option_text: str,
        modality: str,
        features: ModalityFeatures,
    ) -> Dict[str, object]:
        target = self._canonical_emotion_label(option_text)
        support_score = 0.0
        contradiction_score = 0.0
        support_sources: List[str] = []
        contradiction_reasons: List[str] = []
        direct_support = False

        def update_from_prediction(prefix: str, predicted: Optional[str], confidence: float, weight: float = 1.0):
            nonlocal support_score, contradiction_score, direct_support
            canonical = self._canonical_emotion_label(predicted)
            if not canonical or not target:
                return
            conf = self._clip01(float(confidence or 0.0))
            if canonical == target:
                support_score = max(support_score, min(0.94, 0.56 + weight * 0.34 * conf))
                support_sources.append(f'{prefix}_emotion={canonical}')
                direct_support = True
            else:
                contradiction_score = max(contradiction_score, min(0.88, 0.42 + weight * 0.32 * conf))
                contradiction_reasons.append(f'{prefix}_emotion={canonical}')

        if modality == 'audio':
            update_from_prediction('audio', features.audio_emotion, features.audio_emotion_conf)
        elif modality == 'visual':
            update_from_prediction('visual', features.visual_emotion, features.visual_emotion_conf)
        else:
            update_from_prediction('audio', features.audio_emotion, features.audio_emotion_conf, weight=0.75)
            update_from_prediction('visual', features.visual_emotion, features.visual_emotion_conf, weight=0.75)

        return {
            'support_score': float(min(support_score, 1.0)),
            'contradiction_score': float(min(contradiction_score, 1.0)),
            'support_sources': support_sources,
            'contradiction_reasons': contradiction_reasons,
            'direct_support': direct_support,
            'proxy_support': False,
            'allow_positive_flip': direct_support,
            'allow_negative_flip': bool(contradiction_score >= 0.5),
        }

    def _score_relation_option(
        self,
        option_text: str,
        features: ModalityFeatures,
        conflict_report,
        frames: Optional[List],
        object_detector,
    ) -> Dict[str, object]:
        base = self._score_relation_consistency(features, conflict_report, frames, object_detector)
        normalized = self._normalize(option_text)
        option_tokens = set(normalized.split())
        is_positive = bool(option_tokens & _CHOICE_POSITIVE_RELATION_TERMS)
        is_negative = bool(option_tokens & _CHOICE_NEGATIVE_RELATION_TERMS)

        support_score = 0.0
        contradiction_score = 0.0
        support_sources: List[str] = []
        contradiction_reasons: List[str] = []

        if is_positive:
            support_score = float(base.get('support_score', 0.0) or 0.0)
            contradiction_score = float(base.get('contradiction_score', 0.0) or 0.0)
            support_sources.extend(base.get('support_sources') or [])
            contradiction_reasons.extend(base.get('contradiction_reasons') or [])
        elif is_negative:
            support_score = float(base.get('contradiction_score', 0.0) or 0.0)
            contradiction_score = float(base.get('support_score', 0.0) or 0.0)
            support_sources.extend(base.get('contradiction_reasons') or [])
            contradiction_reasons.extend(base.get('support_sources') or [])
        else:
            support_score = max(
                float(base.get('support_score', 0.0) or 0.0),
                float(base.get('contradiction_score', 0.0) or 0.0) * 0.82,
            )
            contradiction_score = min(
                0.8,
                abs(float(base.get('support_score', 0.0) or 0.0) - float(base.get('contradiction_score', 0.0) or 0.0)),
            )
            support_sources.append('generic_relation_option')

        return {
            'support_score': float(min(support_score, 1.0)),
            'contradiction_score': float(min(contradiction_score, 1.0)),
            'support_sources': support_sources,
            'contradiction_reasons': contradiction_reasons,
            'direct_support': bool(base.get('direct_anchor', False)),
            'proxy_support': bool(base.get('typed_proxy_anchor', False)),
            'allow_positive_flip': bool(support_score >= contradiction_score),
            'allow_negative_flip': bool(contradiction_score >= 0.5),
            'base_relation_evidence': base,
        }

    def _score_choice_option(
        self,
        stem: str,
        option: Dict[str, str],
        features: ModalityFeatures,
        *,
        conflict_report=None,
        frames: Optional[List] = None,
        object_detector=None,
    ) -> Dict[str, object]:
        query_spec = self._infer_choice_query_spec(stem)
        modality = str(query_spec.get('modality') or 'cross_modal')
        relation = str(query_spec.get('relation') or 'attribute')
        option_text = str(option.get('text') or '').strip()

        if relation == 'emotion':
            evidence = self._score_emotion_option(option_text, modality, features)
        elif relation == 'consistency':
            evidence = self._score_relation_option(option_text, features, conflict_report, frames, object_detector)
        elif modality == 'visual' or relation == 'presence':
            evidence = self._score_visual(option_text, features, frames, object_detector)
        elif modality == 'audio' or relation == 'sound':
            evidence = self._score_audio(option_text, features)
        else:
            visual_evidence = self._score_visual(option_text, features, frames, object_detector)
            audio_evidence = self._score_audio(option_text, features)
            evidence = {
                'support_score': float(max(
                    float(visual_evidence.get('support_score', 0.0) or 0.0),
                    float(audio_evidence.get('support_score', 0.0) or 0.0),
                )),
                'contradiction_score': float(max(
                    float(visual_evidence.get('contradiction_score', 0.0) or 0.0),
                    float(audio_evidence.get('contradiction_score', 0.0) or 0.0),
                ) * 0.75),
                'support_sources': list((visual_evidence.get('support_sources') or [])[:2]) + list((audio_evidence.get('support_sources') or [])[:2]),
                'contradiction_reasons': list((visual_evidence.get('contradiction_reasons') or [])[:2]) + list((audio_evidence.get('contradiction_reasons') or [])[:2]),
                'direct_support': bool(visual_evidence.get('direct_support', False) or audio_evidence.get('direct_support', False)),
                'proxy_support': bool(visual_evidence.get('proxy_support', False) or audio_evidence.get('proxy_support', False)),
                'allow_positive_flip': bool(visual_evidence.get('allow_positive_flip', False) or audio_evidence.get('allow_positive_flip', False)),
                'allow_negative_flip': bool(visual_evidence.get('allow_negative_flip', False) or audio_evidence.get('allow_negative_flip', False)),
                'visual_evidence': visual_evidence,
                'audio_evidence': audio_evidence,
            }

        if modality in {'visual', 'audio'} and relation in {'presence', 'sound', 'attribute'}:
            evidence = self._apply_specific_human_shadow_guard(
                option_text,
                modality,
                evidence,
                features,
                frames=frames,
                object_detector=object_detector,
            )
            evidence = self._apply_nondecisive_support_calibration(modality, evidence)

        support = float(evidence.get('support_score', 0.0) or 0.0)
        contradiction = float(evidence.get('contradiction_score', 0.0) or 0.0)
        nondecisive_risk = float(evidence.get('nondecisive_risk', 0.0) or 0.0)
        decision_margin = support - contradiction
        decision_score = decision_margin - 0.35 * nondecisive_risk
        allow_commit = bool(evidence.get('allow_positive_flip', True))
        if support < contradiction:
            allow_commit = False

        return {
            'label': option.get('label'),
            'text': option_text,
            'query_spec': query_spec,
            'support_score': support,
            'contradiction_score': contradiction,
            'decision_margin': float(decision_margin),
            'decision_score': float(decision_score),
            'allow_commit': allow_commit,
            'direct_support': bool(evidence.get('direct_support', False)),
            'proxy_support': bool(evidence.get('proxy_support', False)),
            'nondecisive_risk': nondecisive_risk,
            'evidence': evidence,
        }

    def score_hypotheses(
        self,
        question: str,
        features: ModalityFeatures,
        *,
        conflict_report=None,
        frames: Optional[List] = None,
        object_detector=None,
        max_new_tokens: Optional[int] = None,
    ) -> Dict[str, object]:
        form_meta = self.classify_question_form(question, max_new_tokens)
        meta: Dict[str, object] = {
            'handled': False,
            'question_form': form_meta.get('form', 'open_ended'),
            'reason': 'not_hypothesis_question',
        }
        if form_meta.get('form') not in {'single_choice', 'multi_select'}:
            meta['question_form_meta'] = form_meta
            return meta

        candidates = [
            self._score_choice_option(
                form_meta.get('stem', question),
                option,
                features,
                conflict_report=conflict_report,
                frames=frames,
                object_detector=object_detector,
            )
            for option in (form_meta.get('options') or [])
        ]
        candidates = [
            candidate for candidate in candidates
            if candidate.get('label') or candidate.get('text')
        ]
        candidates.sort(
            key=lambda item: (
                float(item.get('decision_score', 0.0) or 0.0),
                float(item.get('support_score', 0.0) or 0.0),
                -float(item.get('contradiction_score', 0.0) or 0.0),
            ),
            reverse=True,
        )
        if not candidates:
            meta.update(
                {
                    'handled': True,
                    'reason': 'no_choice_candidates',
                    'question_form_meta': form_meta,
                    'candidates': [],
                }
            )
            return meta

        support_threshold = float(getattr(self.config, 'choice_evidence_support_threshold', 0.56) or 0.56)
        margin_threshold = float(getattr(self.config, 'choice_evidence_margin_threshold', 0.10) or 0.10)
        top1_gap_threshold = float(getattr(self.config, 'choice_evidence_top1_gap_threshold', 0.06) or 0.06)

        best = candidates[0]
        runner_up_score = float(candidates[1].get('decision_score', -1.0) or -1.0) if len(candidates) > 1 else -1.0
        selected = [
            candidate
            for candidate in candidates
            if bool(candidate.get('allow_commit', False))
            and float(candidate.get('support_score', 0.0) or 0.0) >= support_threshold
            and float(candidate.get('decision_margin', 0.0) or 0.0) >= margin_threshold
        ]

        meta.update(
            {
                'handled': True,
                'reason': 'choice_hypotheses_scored',
                'question_form': form_meta.get('form'),
                'question_form_meta': form_meta,
                'query_spec': form_meta.get('query_spec') or {},
                'candidates': candidates,
                'best_candidate': best,
                'selected_candidates': selected,
                'support_threshold': support_threshold,
                'margin_threshold': margin_threshold,
                'top1_gap_threshold': top1_gap_threshold,
                'top1_gap': float(float(best.get('decision_score', 0.0) or 0.0) - runner_up_score),
            }
        )
        return meta

    def score_question(
        self,
        question: str,
        features: ModalityFeatures,
        *,
        conflict_report=None,
        frames: Optional[List] = None,
        object_detector=None,
        max_new_tokens: Optional[int] = None,
    ) -> Dict[str, object]:
        """Return raw, baseline-independent evidence for a yes/no question."""
        meta: Dict[str, object] = {'handled': False, 'reason': 'not_applicable'}
        if not self.is_yes_no_question(question, max_new_tokens):
            return meta

        spec = self._extract_entity_spec(question)
        question_kind = 'entity_grounded'
        if spec is None and getattr(self.config, 'enable_relation_consistency_evidence', True):
            spec = self._extract_relation_spec(question)
            question_kind = 'relation_grounded'
        if spec is None:
            meta.update({'handled': True, 'question_kind': 'unknown', 'reason': 'unsupported_question_shape'})
            return meta

        meta['handled'] = True
        meta['question_spec'] = spec
        meta['question_kind'] = spec.get('question_kind', question_kind)
        meta['modality'] = spec.get('modality')
        entity = spec.get('entity')
        modality = spec['modality']

        if modality == 'visual':
            evidence = self._score_visual(entity, features, frames, object_detector)
            support_threshold = float(getattr(self.config, 'question_evidence_support_threshold', 0.72))
            contradiction_threshold = float(getattr(self.config, 'question_evidence_contradiction_threshold', 0.72))
            flip_guard = float(getattr(self.config, 'question_evidence_flip_guard', 0.14))
            method_name = 'question_evidence'
        elif modality == 'audio':
            evidence = self._score_audio(entity, features)
            support_threshold = float(getattr(self.config, 'question_evidence_support_threshold', 0.72))
            contradiction_threshold = float(getattr(self.config, 'question_evidence_contradiction_threshold', 0.72))
            flip_guard = float(getattr(self.config, 'question_evidence_flip_guard', 0.14))
            method_name = 'question_evidence'
        else:
            evidence = self._score_relation_consistency(features, conflict_report, frames, object_detector)
            support_threshold = float(getattr(self.config, 'relation_evidence_support_threshold', 0.78))
            contradiction_threshold = float(getattr(self.config, 'relation_evidence_contradiction_threshold', 0.72))
            flip_guard = float(getattr(self.config, 'relation_evidence_flip_guard', 0.14))
            method_name = 'relation_evidence'

        if modality in {'visual', 'audio'}:
            evidence = self._apply_specific_human_shadow_guard(
                entity,
                modality,
                evidence,
                features,
                frames=frames,
                object_detector=object_detector,
            )
            evidence = self._apply_nondecisive_support_calibration(modality, evidence)

        support = float(evidence.get('support_score', 0.0))
        contradiction = float(evidence.get('contradiction_score', 0.0))
        margin = support - contradiction
        reverse_margin = contradiction - support
        allow_positive_flip = bool(evidence.get('allow_positive_flip', True))
        allow_negative_flip = bool(evidence.get('allow_negative_flip', True))

        meta.update(
            {
                'evidence': evidence,
                'support_score': support,
                'contradiction_score': contradiction,
                'margin': margin,
                'reverse_margin': reverse_margin,
                'allow_positive_flip': allow_positive_flip,
                'allow_negative_flip': allow_negative_flip,
                'support_threshold': support_threshold,
                'contradiction_threshold': contradiction_threshold,
                'flip_guard': flip_guard,
                'method_name': method_name,
            }
        )
        return meta

    def evaluate(
        self,
        question: str,
        baseline_answer: str,
        features: ModalityFeatures,
        *,
        conflict_report=None,
        frames: Optional[List] = None,
        object_detector=None,
        max_new_tokens: Optional[int] = None,
    ) -> Tuple[str, str, Dict[str, object]]:
        meta = self.score_question(
            question,
            features,
            conflict_report=conflict_report,
            frames=frames,
            object_detector=object_detector,
            max_new_tokens=max_new_tokens,
        )
        if not bool(meta.get('handled', False)):
            return baseline_answer, 'none', meta
        if meta.get('reason') == 'unsupported_question_shape':
            return baseline_answer, 'none', meta

        baseline_label = self._extract_yes_no(baseline_answer)
        meta['baseline_label'] = baseline_label
        support_threshold = float(meta.get('support_threshold', 0.72) or 0.72)
        contradiction_threshold = float(meta.get('contradiction_threshold', 0.72) or 0.72)
        flip_guard = float(meta.get('flip_guard', 0.14) or 0.14)
        method_name = str(meta.get('method_name', 'question_evidence'))
        support = float(meta.get('support_score', 0.0) or 0.0)
        contradiction = float(meta.get('contradiction_score', 0.0) or 0.0)
        margin = float(meta.get('margin', support - contradiction) or 0.0)
        reverse_margin = float(meta.get('reverse_margin', contradiction - support) or 0.0)
        allow_positive_flip = bool(meta.get('allow_positive_flip', True))
        allow_negative_flip = bool(meta.get('allow_negative_flip', True))

        if support >= support_threshold and margin >= flip_guard:
            if baseline_label == 'Yes':
                meta['reason'] = 'evidence_matches_baseline_positive'
                return baseline_answer, 'none', meta
            if not allow_positive_flip:
                meta['reason'] = 'positive_flip_guarded'
                return baseline_answer, 'none', meta
            meta['reason'] = 'strong_positive_evidence'
            return 'Yes', method_name, meta

        if contradiction >= contradiction_threshold and reverse_margin >= flip_guard:
            if baseline_label == 'No':
                meta['reason'] = 'evidence_matches_baseline_negative'
                return baseline_answer, 'none', meta
            if not allow_negative_flip:
                meta['reason'] = 'negative_flip_guarded'
                return baseline_answer, 'none', meta
            meta['reason'] = 'strong_negative_evidence'
            return 'No', method_name, meta

        meta['reason'] = 'evidence_abstained'
        return baseline_answer, 'none', meta
