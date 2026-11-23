# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Host-owned segment boundaries qualified for H3's gfx950 grouped FMHA."""

from collections.abc import Sequence

import torch

# Each supported [document, padding] pair fills 37,760 rows.
_PACKING_OWNER = object()
GROUPED_CUTS = frozenset((37746, 37747, 37748, 37749, 37750, 37751, 37753, 37755, 37757))


def packed_cu_seqlens(bounds: torch.Tensor | Sequence[int], device: torch.device) -> torch.Tensor:
    """Publish known CPU bounds once; externally produced device bounds keep their normal semantics."""
    if (
        torch.version.hip is None
        or len(bounds) != 3
        or (isinstance(bounds, torch.Tensor) and bounds.device.type != "cpu")
    ):
        return torch.as_tensor(bounds, dtype=torch.int32, device=device)

    values = tuple(bounds.tolist()) if isinstance(bounds, torch.Tensor) else tuple(bounds)
    if values[0] != 0 or values[1] not in GROUPED_CUTS or values[2] != 37760:
        return torch.as_tensor(bounds, dtype=torch.int32, device=device)

    # Inference tensors have no version counter. A fresh, versioned allocation
    # gives subsequent layers a cheap identity/version check without a GPU read.
    with torch.inference_mode(False):
        tensor = torch.tensor(values, dtype=torch.int32, device=device)
        tensor._minimax_h3_fmha_bounds = (_PACKING_OWNER, id(tensor), tensor._version, values[1])
    return tensor


def has_packed_fmha_bounds(tensor: torch.Tensor, max_seqlen: int) -> bool:
    mark = getattr(tensor, "_minimax_h3_fmha_bounds", None)
    return mark is not None and mark[:2] == (_PACKING_OWNER, id(tensor)) and mark[2:] == (tensor._version, max_seqlen)
