# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compatibility exports for DeepSeek Janus stage bridges."""

from vllm_omni.model_executor.models.deepseek_janus.stage_input_processors import (
    ar2generation,
    ar_tokens_to_vq,
)

__all__ = ["ar2generation", "ar_tokens_to_vq"]
