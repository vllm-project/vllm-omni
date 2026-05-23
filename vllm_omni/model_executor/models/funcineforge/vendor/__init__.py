# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Vendored FunCineForge modules (DiT, CausalHiFiGAN).

These are thin shims that re-export the original FunCineForge model code
with minimal adaptations for vLLM-Omni compatibility (e.g. removing
``funcineforge.register`` / ``x_transformers`` dependencies).
"""
