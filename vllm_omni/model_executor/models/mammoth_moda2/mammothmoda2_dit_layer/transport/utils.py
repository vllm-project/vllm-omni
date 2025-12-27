# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0
#
# This file has been modified by ByteDance Ltd. on 2025-09-30.
#
# Original file was released under Apache-2.0, with the full license text
# available at https://github.com/Alpha-VLLM/Lumina-Image-2.0/blob/main/LICENSE
#
# This modified file is released under the same license.

import math

import torch as th


def mean_flat(x):
    """
    Take the mean over all non-batch dimensions.
    """
    return th.mean(x, dim=list(range(1, len(x.size()))))


def log_state(state):
    result = []

    sorted_state = dict(sorted(state.items()))
    for key, value in sorted_state.items():
        # Check if the value is an instance of a class
        if "<object" in str(value) or "object at" in str(value):
            result.append(f"{key}: [{value.__class__.__name__}]")
        else:
            result.append(f"{key}: {value}")

    return "\n".join(result)


def time_shift(mu: float, sigma: float, t: th.Tensor):
    # the following implementation was original for t=0: clean / t=1: noise
    # Since we adopt the reverse, the 1-t operations are needed
    t = 1 - t
    t = math.exp(mu) / (math.exp(mu) + (1 / t - 1) ** sigma)
    t = 1 - t
    return t


def get_lin_function(x1: float = 256, y1: float = 0.5, x2: float = 4096, y2: float = 1.15):
    m = (y2 - y1) / (x2 - x1)
    b = y1 - m * x1
    return lambda x: m * x + b


def expand_dims(v, dims):
    """
    Expand the tensor `v` to the dim `dims`.

    Args:
        `v`: a PyTorch tensor with shape [N].
        `dim`: a `int`.
    Returns:
        a PyTorch tensor with shape [N, 1, 1, ..., 1] and the total dimension is `dims`.
    """
    return v[(...,) + (None,) * (dims - 1)]
