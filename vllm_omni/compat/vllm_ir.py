# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from contextlib import contextmanager, nullcontext
from typing import Any

try:
    import vllm.ir as _vllm_ir
except ImportError:
    _vllm_ir = None


def enable_torch_wrap(enabled: bool):
    if _vllm_ir is None:
        return nullcontext()
    return _vllm_ir.enable_torch_wrap(enabled)


@contextmanager
def maybe_set_ir_op_priority(vllm_config: Any):
    priority = getattr(getattr(vllm_config, "kernel_config", None), "ir_op_priority", None)
    set_priority = getattr(priority, "set_priority", None)
    if set_priority is None:
        yield
        return
    with set_priority():
        yield


__all__ = ["enable_torch_wrap", "maybe_set_ir_op_priority"]
