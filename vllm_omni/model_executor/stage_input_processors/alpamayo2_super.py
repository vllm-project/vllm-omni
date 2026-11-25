# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Stage input processor for Alpamayo 2 Super."""

from __future__ import annotations

from vllm.inputs import TextPrompt

from vllm_omni.inputs.data import OmniTokensPrompt
from vllm_omni.outputs import OmniRequestOutput


def reasoning2expert(
    source_outputs: list[OmniRequestOutput],
    prompt: OmniTokensPrompt | TextPrompt | list | None = None,
    requires_multimodal_data: bool = False,
) -> TextPrompt:
    """Build the expert stage's prompt.

    The expert reads only the transferred KV prefix and its own sampling
    params. Without this processor the request's prompt, with all camera
    frames, would be sent to the expert stage.
    """
    del source_outputs, prompt, requires_multimodal_data
    return {"prompt": ""}
