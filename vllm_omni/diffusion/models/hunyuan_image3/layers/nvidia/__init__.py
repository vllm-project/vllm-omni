# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""NVIDIA (CUDA) implementations."""

from .autoencoder_blocks import ResnetBlock  # noqa: E402
from .transformer_blocks import ResBlock  # noqa: E402

__all__ = ["ResnetBlock", "ResBlock"]
