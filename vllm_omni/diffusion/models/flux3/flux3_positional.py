# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Position-id packers and scatter helpers for Flux3.

Each image/text latent is flattened to a token sequence with a 4-axis ``(t, h, w, l)``
position id per token:

- image: ``(c h w) -> (h w) c``, ids ``cartesian_prod(t, arange(h), arange(w), l)``
- text:  ids with ``t = h = w = 0`` and ``l = arange(length)`` (sequence index)

``scatter_ids`` inverts the packing: it scatters denoise-output tokens back into a
dense ``(1, C, h, w)`` image latent grid for the VAE, using the spatial position ids.
"""

from collections.abc import Callable

import torch
from einops import rearrange
from torch import Tensor


def prc_img(x: Tensor, t_coord: Tensor | None = None, l_coord: Tensor | None = None) -> tuple[Tensor, Tensor]:
    _, h, w = x.shape
    dev = x.device
    coords = {
        "t": torch.arange(1, device=dev) if t_coord is None else t_coord.to(dev),
        "h": torch.arange(h, device=dev),
        "w": torch.arange(w, device=dev),
        "l": torch.arange(1, device=dev) if l_coord is None else l_coord.to(dev),
    }
    x_ids = torch.cartesian_prod(coords["t"], coords["h"], coords["w"], coords["l"])
    x = rearrange(x, "c h w -> (h w) c")
    return x, x_ids


def prc_txt(x: Tensor, t_coord: Tensor | None = None, l_coord: Tensor | None = None) -> tuple[Tensor, Tensor]:
    assert l_coord is None, "l_coord not supported for txt"
    length, _ = x.shape
    dev = x.device
    coords = {
        "t": torch.arange(1, device=dev) if t_coord is None else t_coord.to(dev),
        "h": torch.arange(1, device=dev),
        "w": torch.arange(1, device=dev),
        "l": torch.arange(length, device=dev),
    }
    x_ids = torch.cartesian_prod(coords["t"], coords["h"], coords["w"], coords["l"])
    return x, x_ids


def _batched(fn: Callable) -> Callable:
    def wrapped(x: Tensor, t_coord: Tensor | None = None, l_coord: Tensor | None = None) -> tuple[Tensor, Tensor]:
        pairs = [
            fn(
                x[i],
                t_coord[i] if t_coord is not None else None,
                l_coord[i] if l_coord is not None else None,
            )
            for i in range(len(x))
        ]
        xs, ids = zip(*pairs, strict=True)
        return torch.stack(xs), torch.stack(ids)

    return wrapped


batched_prc_img = _batched(prc_img)
batched_prc_txt = _batched(prc_txt)


def scatter_ids(x: Tensor, x_ids: Tensor) -> list[Tensor]:
    """Scatter image token sequences back to dense ``(1, C, h, w)`` latent grids."""
    out_list = []
    for data, pos in zip(x, x_ids, strict=True):
        _, ch = data.shape
        h_ids = pos[:, 1].to(torch.int64)
        w_ids = pos[:, 2].to(torch.int64)

        h = int(torch.max(h_ids)) + 1
        w = int(torch.max(w_ids)) + 1

        flat_ids = h_ids * w + w_ids
        out = torch.zeros((h * w, ch), device=data.device, dtype=data.dtype)
        out.scatter_(0, flat_ids.unsqueeze(1).expand(-1, ch), data)
        out_list.append(rearrange(out, "(h w) c -> 1 c h w", h=h, w=w))
    return out_list
