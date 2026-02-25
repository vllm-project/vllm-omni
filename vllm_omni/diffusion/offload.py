# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compatibility shim for diffusion offload hooks.

Some call sites import `vllm_omni.diffusion.offload.apply_offload_hooks`,
but the offloading implementation now lives under `vllm_omni.diffusion.offloader`.
Keeping this no-op hook avoids import errors without changing behavior.
"""

from __future__ import annotations

from typing import Any


def apply_offload_hooks(*_args: Any, **_kwargs: Any) -> None:
    """No-op compatibility hook."""
    return None
