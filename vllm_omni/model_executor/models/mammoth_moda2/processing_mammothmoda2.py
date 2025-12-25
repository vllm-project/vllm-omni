"""Transformers Processor for MammothModa2.

MammothModa2 reuses Qwen2.5-VL's image/video processing logic but uses the custom 
MammothUTokenizer (tiktoken-based) for text. Since the upstream `Qwen2_5_VLProcessor` 
hardcodes `Qwen2Tokenizer`/Fast, this derived Processor relaxes tokenizer type 
checks and loading.
"""

from __future__ import annotations

from transformers.models.qwen2_5_vl.processing_qwen2_5_vl import Qwen2_5_VLProcessor

# Trigger AutoConfig/AutoTokenizer registration so ProcessorMixin can find 
# MammothUTokenizer in `TOKENIZER_MAPPING._extra_content`.
from .configuration_mammothmoda2 import (  # noqa: F401
    Mammothmoda2Config,
    Mammothmoda2Qwen2_5_VLConfig,
)
from .tokenization_mammothmoda2_qwen2_5_vl import MammothUTokenizer  # noqa: F401


class Mammothmoda2Processor(Qwen2_5_VLProcessor):
    """Qwen2.5-VL Processor with MammothU tokenizer."""

    tokenizer_class = ("MammothUTokenizer", None)
