# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Apply process-wide Torch settings configured through environment variables."""

from __future__ import annotations

import os

_TORCH_DYNAMO_RECOMPILE_LIMIT = "VLLM_OMNI_TORCH_DYNAMO_RECOMPILE_LIMIT"


def configure_torch_dynamo() -> None:
    """Set Torch Dynamo's recompile limit when explicitly requested."""
    value = os.environ.get(_TORCH_DYNAMO_RECOMPILE_LIMIT)
    if value is None:
        return

    try:
        recompile_limit = int(value)
    except ValueError as exc:
        raise ValueError(f"{_TORCH_DYNAMO_RECOMPILE_LIMIT} must be a positive integer, got {value!r}") from exc
    if recompile_limit <= 0:
        raise ValueError(f"{_TORCH_DYNAMO_RECOMPILE_LIMIT} must be positive, got {value!r}")

    import torch._dynamo.config as dynamo_config

    dynamo_config.recompile_limit = recompile_limit


configure_torch_dynamo()
