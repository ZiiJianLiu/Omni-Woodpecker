"""Core package for the training-free Omni-Woodpecker implementation."""

__version__ = "0.1.0"

from .data_types import ModalityFeatures
from .models.qwen_omni import QwenOmniAdapter
from .api import correct

__all__ = ["ModalityFeatures", "QwenOmniAdapter", "correct"]
