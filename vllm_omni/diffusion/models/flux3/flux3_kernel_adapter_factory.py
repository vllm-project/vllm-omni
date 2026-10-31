# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Select the Flux3 kernel adapter from runtime configuration and support."""

from __future__ import annotations

from typing import TYPE_CHECKING

from vllm_omni.diffusion.models.flux3.flux3_kernel_adapter import Flux3KernelAdapter, QKNormRopeFn

if TYPE_CHECKING:
    from vllm_omni.diffusion.data import OmniDiffusionConfig


def build_flux3_kernel_adapter(
    *,
    qk_norm_rope: QKNormRopeFn,
    od_config: OmniDiffusionConfig | None,
    quant_config: object | None,
) -> Flux3KernelAdapter:
    """Return the default Flux3 kernel adapter backed by vLLM kernels."""
    return Flux3KernelAdapter(qk_norm_rope)


__all__ = ["build_flux3_kernel_adapter"]
