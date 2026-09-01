"""
Task Router
============
根据问题文本分类任务类型，用于任务感知的修正策略选择
"""
import re

_AV_MATCH_RE = re.compile(
    r"(audio and visual|audio.*visual|visual.*audio).*(match|matching|correspond)",
    re.IGNORECASE,
)
_AUDIO_HALL_RE = re.compile(
    r"making sound.*(in the audio|audio)|hear.*(in the audio)",
    re.IGNORECASE,
)
_VISUAL_HALL_RE = re.compile(
    r"visible.*(in the video|video)|see.*(in the video)",
    re.IGNORECASE,
)
_CAPTION_RE = re.compile(
    r"describe what you see and hear|describe.*single sentence",
    re.IGNORECASE,
)


def classify_task(question: str) -> str:
    """根据问题文本分类任务类型

    Returns
    -------
    task_type : str
        av_matching / video_audio_hallucination /
        audio_video_hallucination / av_captioning / unknown
    """
    if _AV_MATCH_RE.search(question):
        return "av_matching"
    if _AUDIO_HALL_RE.search(question):
        return "video_audio_hallucination"
    if _VISUAL_HALL_RE.search(question):
        return "audio_video_hallucination"
    if _CAPTION_RE.search(question):
        return "av_captioning"
    return "unknown"
