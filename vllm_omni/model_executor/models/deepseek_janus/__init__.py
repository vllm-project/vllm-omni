# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project


def __getattr__(name: str):
    """Lazy import so environments without vLLM can still import pipeline modules."""
    if name == "OmniDeepSeekJanusForConditionalGeneration":
        from .deepseek_janus_ar import OmniDeepSeekJanusForConditionalGeneration

        return OmniDeepSeekJanusForConditionalGeneration
    if name == "JanusForImageGeneration":
        from .deepseek_janus_ar import JanusForImageGeneration

        return JanusForImageGeneration
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["OmniDeepSeekJanusForConditionalGeneration", "JanusForImageGeneration"]
