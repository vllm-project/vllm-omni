# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Global diffusion config — mirrors vLLM's set_current_vllm_config pattern.

Set once during model initialisation so that every ``Attention`` layer can
read the ``OmniDiffusionConfig`` without coupling ``__init__`` to the
runtime ``ForwardContext``.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from vllm_omni.diffusion.data import OmniDiffusionConfig

_current_diffusion_config: OmniDiffusionConfig | None = None


@contextmanager
def set_current_diffusion_config(config: OmniDiffusionConfig):
    """Context manager that sets the global diffusion config for model init."""
    global _current_diffusion_config
    old = _current_diffusion_config
    _current_diffusion_config = config
    try:
        yield
    finally:
        _current_diffusion_config = old


def get_current_diffusion_config() -> OmniDiffusionConfig:
    """Return the current diffusion config or raise."""
    assert _current_diffusion_config is not None, (
        "Diffusion config is not set. Wrap model construction with set_current_diffusion_config()."
    )
    return _current_diffusion_config


def get_current_diffusion_config_or_none() -> OmniDiffusionConfig | None:
    """Return the current diffusion config, or ``None`` if not set."""
    return _current_diffusion_config
