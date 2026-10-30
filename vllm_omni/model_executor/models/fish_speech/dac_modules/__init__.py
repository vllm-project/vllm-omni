# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Vendored DAC codec modules for Fish Speech S2 Pro.

Adopted from the ``fish-speech`` 0.1.0 PyPI release
(https://pypi.org/project/fish-speech/0.1.0/, Apache-2.0) and
descript-audio-codec (https://github.com/descriptinc/descript-audio-codec,
MIT), trimmed to inference-only so the codec loads without external model
packages. Module attribute paths and weight-norm styles are kept identical
to upstream so ``codec.pth`` checkpoints load unchanged.
"""

from vllm_omni.model_executor.models.fish_speech.dac_modules.codec import DAC
from vllm_omni.model_executor.models.fish_speech.dac_modules.rvq import (
    DownsampleResidualVectorQuantize,
)
from vllm_omni.model_executor.models.fish_speech.dac_modules.transformer import (
    ModelArgs,
    WindowLimitedTransformer,
)

__all__ = [
    "DAC",
    "DownsampleResidualVectorQuantize",
    "ModelArgs",
    "WindowLimitedTransformer",
]
