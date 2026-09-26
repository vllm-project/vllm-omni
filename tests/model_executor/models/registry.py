# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""
Test metadata for Omni models.

Mirrors vllm's ``tests/models/registry.py`` (_HfExamplesInfo) but adapted for
the Omni multi-stage architecture.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import pytest
from packaging.version import Version
from transformers import __version__ as TRANSFORMERS_VERSION


@dataclass(frozen=True)
class _OmniExamplesInfo:
    """Metadata used by processing-correctness tests for Omni models."""

    default: str
    """The default HF model ID to use for testing this architecture."""

    model_stage: str = "thinker"
    """Stage type identifier, e.g. ``"thinker"`` or ``"talker"``."""

    hf_config_name: str | None = None
    """
    Name of the sub-config inside the top-level HF config
    (e.g. ``"thinker_config"``).  ``None`` means use the root config.
    """

    has_multimodal_processor: bool = False
    """Whether the architecture registers a multimodal processor."""

    min_transformers_version: str | None = None
    """Minimum ``transformers`` version required to load this model."""

    max_transformers_version: str | None = None
    """Maximum ``transformers`` version that this model supports."""

    is_available_online: bool = True
    """Set to ``False`` if the model is no longer hosted on HF Hub."""

    trust_remote_code: bool = False
    """Whether ``trust_remote_code=True`` is needed to load the model."""

    max_model_len: int | None = None
    """
    An explicit ``max_model_len`` override.  Useful when the default value
    from the HF config is too large for CI.
    """

    # ----- helper methods (same API as _HfExamplesInfo) --------------------

    def check_transformers_version(
        self,
        *,
        on_fail: Literal["error", "skip", "return"],
    ) -> str | None:
        """Skip / error / return a message when the installed
        ``transformers`` does not satisfy the version constraints."""
        if self.min_transformers_version is None and self.max_transformers_version is None:
            return None

        current = TRANSFORMERS_VERSION
        cur_base = Version(current).base_version
        min_ver = self.min_transformers_version
        max_ver = self.max_transformers_version

        msg = f"`transformers=={current}` installed, but `transformers"
        if min_ver and Version(cur_base) < Version(min_ver):
            msg += f">={min_ver}` is required to run this model."
        elif max_ver and Version(cur_base) > Version(max_ver):
            msg += f"<={max_ver}` is required to run this model."
        else:
            return None

        if on_fail == "error":
            raise RuntimeError(msg)
        elif on_fail == "skip":
            pytest.skip(msg)
        return msg

    def check_available_online(
        self,
        *,
        on_fail: Literal["error", "skip"],
    ) -> None:
        """Skip / error when the model is not available online."""
        if not self.is_available_online:
            msg = "Model is not available online"
            if on_fail == "error":
                raise RuntimeError(msg)
            else:
                pytest.skip(msg)


# ---------------------------------------------------------------------------
# Example model registry for tests
# ---------------------------------------------------------------------------

_OMNI_EXAMPLE_MODELS: dict[str, _OmniExamplesInfo] = {
    # ---- Qwen2.5-Omni ----
    "Qwen2_5OmniForConditionalGeneration": _OmniExamplesInfo(
        default="Qwen/Qwen2.5-Omni-7B",
        model_stage="thinker",
        has_multimodal_processor=True,
    ),
    "Qwen2_5OmniThinkerModel": _OmniExamplesInfo(
        default="Qwen/Qwen2.5-Omni-7B",
        model_stage="thinker",
        hf_config_name="thinker_config",
        has_multimodal_processor=True,
    ),
    "Qwen2_5OmniTalkerModel": _OmniExamplesInfo(
        default="Qwen/Qwen2.5-Omni-7B",
        model_stage="talker",
        hf_config_name="talker_config",
    ),
    "Qwen2_5OmniToken2WavModel": _OmniExamplesInfo(
        default="Qwen/Qwen2.5-Omni-7B",
        model_stage="token2wav",
    ),
    "Qwen2_5OmniToken2WavDiTModel": _OmniExamplesInfo(
        default="Qwen/Qwen2.5-Omni-7B",
        model_stage="token2wav",
    ),
    # Internal sub-model key hardcoded by Qwen2_5OmniTalkerForConditionalGeneration
    # to load its language model component (see qwen2_5_omni_talker.py).
    "Qwen2ForCausalLM_old": _OmniExamplesInfo(
        default="Qwen/Qwen2.5-Omni-7B",
        model_stage="thinker",
    ),
    # ---- Qwen3-Omni MoE  ----
    "Qwen3OmniMoeForConditionalGeneration": _OmniExamplesInfo(
        default="Qwen/Qwen3-Omni-30B-A3B-Instruct",
        model_stage="thinker",
        has_multimodal_processor=True,
    ),
    "Qwen3OmniMoeThinkerForConditionalGeneration": _OmniExamplesInfo(
        default="Qwen/Qwen3-Omni-30B-A3B-Instruct",
        model_stage="thinker",
        hf_config_name="thinker_config",
        has_multimodal_processor=True,
    ),
    "Qwen3OmniMoeTalkerForConditionalGeneration": _OmniExamplesInfo(
        default="Qwen/Qwen3-Omni-30B-A3B-Instruct",
        model_stage="talker",
        hf_config_name="talker_config",
    ),
    "Qwen3OmniMoeCode2Wav": _OmniExamplesInfo(
        default="Qwen/Qwen3-Omni-30B-A3B-Instruct",
        model_stage="code2wav",
    ),
    # ---- Qwen3-TTS ----
    "Qwen3TTSForConditionalGeneration": _OmniExamplesInfo(
        default="Qwen/Qwen3-TTS",
        model_stage="talker",
    ),
    "Qwen3TTSTalkerForConditionalGeneration": _OmniExamplesInfo(
        default="Qwen/Qwen3-TTS",
        model_stage="talker",
    ),
    "Qwen3TTSCode2Wav": _OmniExamplesInfo(
        default="Qwen/Qwen3-TTS",
        model_stage="code2wav",
    ),
}

# Only architectures that register a multimodal processor are relevant for
# processing-correctness tests.
_MULTIMODAL_OMNI_EXAMPLE_MODELS: dict[str, _OmniExamplesInfo] = {
    arch: info for arch, info in _OMNI_EXAMPLE_MODELS.items() if info.has_multimodal_processor
}
