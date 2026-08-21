# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Small platform-agnostic helpers for UNet-style blocks (adapted from diffusers).

``conv_nd`` / ``normalization`` / ``linear`` / ``zero_module`` are used by the
``ResBlock`` / ``UNetDown`` / ``UNetUp`` implementations across HunyuanImage3's
platform-specific layer packages. They are plain ``nn.ConvNd`` / ``nn.GroupNorm``
/ ``nn.Linear`` wrappers with no hardware-specific behaviour, so they live here
instead of being duplicated per backend.
"""

from torch import nn


def conv_nd(dims, *args, **kwargs):  # noqa: N802
    """
    Create a 1D, 2D, or 3D convolution module.
    """
    if dims == 1:
        return nn.Conv1d(*args, **kwargs)
    elif dims == 2:
        return nn.Conv2d(*args, **kwargs)
    elif dims == 3:
        return nn.Conv3d(*args, **kwargs)
    raise ValueError(f"unsupported dimensions: {dims}")


def normalization(channels, **kwargs):
    """
    Make a standard normalization layer.
    :param channels: number of input channels.
    :return: a nn.Module for normalization.
    """
    return nn.GroupNorm(32, channels, **kwargs)


def linear(*args, **kwargs):
    """
    Create a linear module.
    """
    return nn.Linear(*args, **kwargs)


def zero_module(module):
    """
    Zero out the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module
