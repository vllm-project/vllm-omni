"""MammothModa2 entrypoints for vLLM-Omni."""

from .mammoth_moda2 import MammothModa2ForConditionalGeneration
from .mammoth_moda2_ar import MammothModa2ARForConditionalGeneration
from .mammoth_moda2_dit import MammothModa2DiTForConditionalGeneration

__all__ = [
    "MammothModa2ForConditionalGeneration",
    "MammothModa2ARForConditionalGeneration",
    "MammothModa2DiTForConditionalGeneration",
]
