# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Shared weight-attribute helper for vLLM-omni quantization configs.

``copy_missing_attrs`` used to live in upstream
``vllm.model_executor.layers.quantization.fp8`` but was removed from vLLM
(commit b5bcb3ce88, "[Refactor] Remove dead code in multiple files").
vLLM-omni's diffusion quant methods still rely on it, so it is re-homed here.
"""

from typing import Any

import torch
from vllm.model_executor.utils import set_weight_attrs


def copy_missing_attrs(old: torch.Tensor, new: torch.Tensor) -> None:
    """Copies any attrs present in `old` but not in `new` to `new`."""
    new_attrs = set(dir(new))
    attrs_to_set: dict[str, Any] = {}
    for attr in dir(old):
        if attr not in new_attrs:
            attrs_to_set[attr] = getattr(old, attr)
    set_weight_attrs(new, attrs_to_set)
