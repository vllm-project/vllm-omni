# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Qwen3-Omni turn-based full-duplex integration (experimental)."""

from vllm_omni.experimental.fullduplex.qwen3omni.policy import (
    INTERRUPTION_NOTE,
    SYSTEM_PROMPT,
)

__all__ = ["INTERRUPTION_NOTE", "SYSTEM_PROMPT"]
