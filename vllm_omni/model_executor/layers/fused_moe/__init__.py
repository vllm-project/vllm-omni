# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Omni-shipped fused MoE Triton tile configs (consumed via vLLM loader)."""

from __future__ import annotations

import os
from pathlib import Path

# Same env as upstream vLLM fused_moe / MoT GEMM.
_TUNED_CONFIG_FOLDER_ENV = "VLLM_TUNED_CONFIG_FOLDER"


def get_fused_moe_configs_dir() -> Path:
    """Return the package directory that stores fused_moe tile JSON files."""
    return Path(__file__).resolve().parent / "configs"


def maybe_set_vllm_tuned_config_folder() -> bool:
    """Point ``VLLM_TUNED_CONFIG_FOLDER`` at omni configs if unset.

    Upstream ``get_moe_configs`` checks this env first, then falls back to
    vLLM's built-in ``fused_moe/configs/``. Setting the env here does **not**
    disable that fallback: missing shapes still use vLLM's packaged JSONs.

    Returns True if this call set the environment variable.
    """
    if os.environ.get(_TUNED_CONFIG_FOLDER_ENV):
        return False

    configs_dir = get_fused_moe_configs_dir()
    if not configs_dir.is_dir():
        return False

    os.environ[_TUNED_CONFIG_FOLDER_ENV] = str(configs_dir)
    return True


__all__ = [
    "get_fused_moe_configs_dir",
    "maybe_set_vllm_tuned_config_folder",
]
