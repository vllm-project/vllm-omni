# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""310P platform helpers."""

from __future__ import annotations

import torch


def is_310p(device: torch.device | None = None) -> bool:
    """Return True on Ascend 310P devices, False otherwise."""
    if device is not None and device.type != "npu":
        return False
    try:
        from vllm_ascend.device.device_config import is_310p as ascend_is_310p

        return bool(ascend_is_310p())
    except (ImportError, AttributeError):
        return False


def disable_jit_compile() -> None:
    import torch_npu

    torch_npu.npu.set_compile_mode(jit_compile=False)


def disable_internal_format() -> None:
    """Keep tensors in the plain layout the 310P op library expects.

    Conv2d/Conv3d outputs are handed to the next op in the private NCDHW/NZ
    "internal format". Several 310P ops do not accept it and fail at runtime
    (for example ``reflection_pad3d`` in image VAEs). Disabling the option
    makes every op see ordinary strided tensors; the cost is limited to
    convolution-heavy modules, while the DiT is matmul-bound.
    """
    import torch_npu

    torch_npu.npu.config.allow_internal_format = False


def apply_patches() -> None:
    if not is_310p():
        return

    from vllm_omni.platforms.npu._310p.patch import apply_patches as apply_310p_patches

    apply_310p_patches()
