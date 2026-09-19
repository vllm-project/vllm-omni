# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""NVIDIA (CUDA + Triton) implementations of the H3 VAE operators.

Both kernels pin exact FP16/FP32 rounding with inline NVIDIA PTX
(``mul.rn.f16x2`` / ``add.rn.f16x2`` / ``mul.rn.f32``) and exclude ROCm in
their input guards, so they only run on NVIDIA hardware. Selection between
this package and the remote-model reference path is handled by the capability
table in ``..dispatch``, not here.
"""

from .qk_norm_rope import try_qk_norm_rope_exact
from .scaled_residual import try_scaled_residual_exact

__all__ = ["try_qk_norm_rope_exact", "try_scaled_residual_exact"]
