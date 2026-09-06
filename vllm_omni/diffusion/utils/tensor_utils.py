# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch


def expand_scalar_to_batch(
    scalar: torch.Tensor,
    batch_size: int,
    *,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Expand a device scalar without synchronizing it through Python."""
    return scalar.expand(batch_size).to(dtype=dtype)
