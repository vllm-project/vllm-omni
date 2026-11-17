# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Spark-TTS model integration."""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .bicodec import SparkBiCodecDecoder
    from .llm import SparkQwen2ForCausalLM

__all__ = ["SparkBiCodecDecoder", "SparkQwen2ForCausalLM"]


def __getattr__(name: str) -> type:
    # The pipeline registry imports this package even when Spark-TTS isn't selected.
    if name == "SparkBiCodecDecoder":
        from .bicodec import SparkBiCodecDecoder

        return SparkBiCodecDecoder
    if name == "SparkQwen2ForCausalLM":
        from .llm import SparkQwen2ForCausalLM

        return SparkQwen2ForCausalLM
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
