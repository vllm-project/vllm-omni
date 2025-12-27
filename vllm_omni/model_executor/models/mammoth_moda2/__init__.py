"""MammothModa2 entrypoints for vLLM-Omni."""

from .configuration_mammothmoda2 import Mammothmoda2Config  # noqa: F401 registers AutoConfig
from .mammoth_moda2_ar import MammothModa2ARForConditionalGeneration

__all__ = [
    "MammothModa2ARForConditionalGeneration",
]
