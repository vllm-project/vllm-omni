"""Transformers Processor for MammothModa2.

MammothModa2 复用 Qwen2.5-VL 的图像/视频处理逻辑，但文本侧使用自定义的
MammothUTokenizer（tiktoken 词表）。Transformers 上游的
`Qwen2_5_VLProcessor` 会硬编码 `Qwen2Tokenizer`/Fast，因此需要一个派生
Processor 来放宽 tokenizer 类型检查与加载。
"""

from __future__ import annotations

from transformers.models.qwen2_5_vl.processing_qwen2_5_vl import Qwen2_5_VLProcessor

# 触发 AutoConfig/AutoTokenizer 注册，使 ProcessorMixin 能在
# `TOKENIZER_MAPPING._extra_content` 中找到 MammothUTokenizer。
from .configuration_mammothmoda2 import (  # noqa: F401
    Mammothmoda2Config,
    Mammothmoda2Qwen2_5_VLConfig,
)
from .tokenization_mammothmoda2_qwen2_5_vl import MammothUTokenizer  # noqa: F401


class Mammothmoda2Processor(Qwen2_5_VLProcessor):
    """Qwen2.5-VL Processor with MammothU tokenizer."""

    tokenizer_class = ("MammothUTokenizer", None)

