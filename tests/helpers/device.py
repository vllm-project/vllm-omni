# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Accelerator handles for device tests, resolved from the active platform.

A GPU test that writes ``torch.device("cuda", 0)``, ``torch.cuda.stream(...)``
or ``backend="nccl"`` only runs on CUDA, even when the kernel under test is
plain PyTorch and the platform offers the same API. Import these instead:

    from tests.helpers.device import DEVICE_MODULE, DEVICE_TYPE

    x = torch.randn(8, device=DEVICE_TYPE)
    with DEVICE_MODULE.stream(DEVICE_MODULE.Stream()):
        ...

``DEVICE_MODULE`` is ``torch.cuda`` on CUDA/ROCm and ``torch.xpu`` on XPU, so
``is_available()``, ``stream()``, ``current_stream()``, ``get_rng_state()``,
``set_rng_state()``, ``graph_pool_handle()`` and ``_sleep()`` all resolve to the
platform's own implementation. The graph class differs by name
(``CUDAGraph`` vs ``XPUGraph``), so take it from ``DEVICE_GRAPH``.

Resolution happens at import time: these are constants, not fixtures, so they
can appear in ``pytest.mark.skipif`` conditions and ``parametrize`` lists.
"""

from __future__ import annotations

from typing import Any

import torch

from vllm_omni.platforms import current_omni_platform

#: ``"cuda"``, ``"xpu"``, ``"npu"``, ... - use in place of a ``"cuda"`` literal.
DEVICE_TYPE: str = current_omni_platform.device_type

#: The platform's torch submodule (``torch.cuda`` / ``torch.xpu`` / ...).
DEVICE_MODULE = torch.get_device_module(DEVICE_TYPE)

#: Device graph class, or ``None`` where the platform has none. Typed ``Any``
#: because the tests that use it are gated to platforms that have one, and an
#: ``Optional`` here would make every ``DEVICE_GRAPH()`` call a type error.
DEVICE_GRAPH: Any = getattr(DEVICE_MODULE, "CUDAGraph", None) or getattr(DEVICE_MODULE, "XPUGraph", None)

#: Collective backend that matches the device (``nccl`` / ``xccl`` / ``gloo``).
DIST_BACKEND: str = {"cuda": "nccl", "rocm": "nccl", "xpu": "xccl"}.get(DEVICE_TYPE, "gloo")

#: NVIDIA CUDA or XPU. Not a GPU-presence check: ``DEVICE_MODULE.is_available()``
#: is also true on ROCm. Use it where a test used to require
#: ``current_omni_platform.is_cuda()`` and its subject (plain PyTorch, or a
#: Triton kernel the code launches without a platform gate) runs the same path
#: on XPU. Tests that assert a CUDA-only fused kernel keep ``is_cuda()``.
NVIDIA_CUDA_OR_XPU: bool = current_omni_platform.is_cuda() or current_omni_platform.is_xpu()
