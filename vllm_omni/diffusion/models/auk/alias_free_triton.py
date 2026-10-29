# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Deprecated import path: the kernel moved to ``common.ops.alias_free_triton``."""

from vllm_omni.model_executor.models.common.ops.alias_free_triton import (  # noqa: F401
    alias_free_snake,
)

__all__ = ["alias_free_snake"]
