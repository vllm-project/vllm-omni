# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tiny utilities the vendored V1 data model needs, removed from their live
upstream homes between the rebase baseline 4ba787ec99 and the 0.34 pin:

* ``swap_dict_values`` — was vllm/utils/collection_utils.py:128.
* ``copy_slice`` — was vllm/v1/utils.py:675.

Vendored verbatim (bodies identical); only the home module changed.
"""

from __future__ import annotations

from typing import TypeVar

import torch
from vllm.utils.torch_utils import TORCH_DTYPE_TO_NUMPY_DTYPE, make_ndarray_with_pad

_K = TypeVar("_K")
_V = TypeVar("_V")
_T = TypeVar("_T")


def swap_dict_values(obj: dict[_K, _V], key1: _K, key2: _K) -> None:
    """Swap values between two keys."""
    v1 = obj.get(key1)
    v2 = obj.get(key2)
    if v1 is not None:
        obj[key2] = v1
    else:
        obj.pop(key2, None)
    if v2 is not None:
        obj[key1] = v2
    else:
        obj.pop(key1, None)


def copy_slice(from_tensor: torch.Tensor, to_tensor: torch.Tensor, length: int) -> torch.Tensor:
    """Copy the first length elements of a tensor into another tensor in a
    non-blocking manner.

    Used to copy pinned CPU tensor data to pre-allocated GPU tensors.

    Returns the sliced target tensor.
    """
    return to_tensor[:length].copy_(from_tensor[:length], non_blocking=True)


def make_tensor_with_pad(
    x: list[list[_T]],
    pad: _T,
    dtype: torch.dtype,
    *,
    max_len: int | None = None,
    device: str | torch.device | None = None,
    pin_memory: bool = False,
) -> torch.Tensor:
    """Make a padded tensor from 2D inputs. (Vendored from
    4ba787ec99 vllm/utils/torch_utils.py:760; removed at the pin.)"""
    np_dtype = TORCH_DTYPE_TO_NUMPY_DTYPE[dtype]
    padded_x = make_ndarray_with_pad(x, pad, np_dtype, max_len=max_len)

    tensor = torch.from_numpy(padded_x).to(device)
    if pin_memory:
        tensor = tensor.pin_memory()
    return tensor
