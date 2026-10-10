# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Released BF16 audio activation arithmetic with portable Torch fallback."""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

import torch
import torch.nn.functional as F
from vllm.triton_utils import tl, triton
from vllm.triton_utils import tldevice as libdevice

from vllm_omni.platforms import current_omni_platform


@lru_cache(maxsize=1)
def released_libdevice_path() -> str | None:
    cuda_home = Path(os.environ.get("CUDA_HOME", "/usr/local/cuda-12.9"))
    path = Path(os.environ.get("LYCHEE_RELEASED_LIBDEVICE_PATH", cuda_home / "nvvm/libdevice/libdevice.10.bc"))
    if not path.is_file():
        if "LYCHEE_RELEASED_LIBDEVICE_PATH" in os.environ:
            raise RuntimeError(f"Lychee explicit released libdevice path does not exist: {path}")
        return None
    return str(path)


@triton.jit
def _released_gelu_kernel(x_ptr, y_ptr, n: tl.constexpr, block: tl.constexpr):
    indices = tl.program_id(0) * block + tl.arange(0, block)
    values = tl.load(x_ptr + indices, indices < n, 0).to(tl.float32)
    # Match Torch 2.7's float arithmetic. The fixed toolchain's erf preserves
    # the released negative-tail rounding before the final BF16 conversion.
    outputs = (values * 0.5) * (1.0 + libdevice.erf(values * 0.70710678118654752440))
    tl.store(y_ptr + indices, outputs, indices < n)


def released_gelu(inputs: torch.Tensor) -> torch.Tensor:
    if (
        inputs.device.type != "cuda"
        or inputs.dtype != torch.bfloat16
        or not current_omni_platform.is_cuda()
        or str(torch.__version__) != "2.13.0+cu132"
        or torch.version.cuda != "13.2"
        or torch.backends.cudnn.version() != 92000
        or current_omni_platform.get_device_capability(inputs.device.index) != (8, 0)
    ):
        return F.gelu(inputs)
    libdevice_path = released_libdevice_path()
    if libdevice_path is None:
        return F.gelu(inputs)
    contiguous = inputs.contiguous()
    output = torch.empty_like(contiguous)
    if contiguous.numel():
        _released_gelu_kernel[(triton.cdiv(contiguous.numel(), 1024),)](
            contiguous,
            output,
            contiguous.numel(),
            block=1024,
            enable_fp_fusion=False,
            extern_libs={"libdevice": libdevice_path},
        )
    return output


class LycheeAudioGELU(torch.nn.Module):
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return released_gelu(inputs)
