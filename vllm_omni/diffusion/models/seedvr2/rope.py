# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
# Vendored from the Apache-2.0 SeedVR2 reference implementation
# (`src/models/dit_3b/rope.py`) and the MIT-licensed `rotary_embedding_torch`
# primitives it builds on, so the port does not add a runtime dependency.
"""SeedVR2 rotary position embeddings (``mmrope3d``).

The reference builds its frequencies with
``rotary_embedding_torch.RotaryEmbedding(dim=head_dim // 3, freqs_for="lang")``
and applies them axially.  Two properties matter for a faithful port:

* **Video positions are window-local.**  For a window of shape ``(f, h, w)`` the
  token at window-local ``(dt, dh, dw)`` uses axial indices
  ``(text_len + dt, dh, dw)`` -- the temporal axis is offset by the text length,
  the spatial axes are not, and every window restarts at 0.
* **The rotary dimension is ``3 * (head_dim // 3)``** (126 for ``head_dim=128``),
  so the last ``head_dim % 3`` channels pass through unrotated.
"""

from __future__ import annotations

import torch
from torch import nn

#: The reference always builds the axial table for the *first* ``dim // 2``
#: frequency slots of a ``lang`` frequency bank.
LANG_THETA = 10000.0


def lang_freqs(axis_dim: int, theta: float = LANG_THETA) -> torch.Tensor:
    """``1 / theta ** (arange(0, axis_dim, 2) / axis_dim)`` (``rotary_embedding_torch``)."""
    return 1.0 / (theta ** (torch.arange(0, axis_dim, 2)[: axis_dim // 2].float() / axis_dim))


def _axis_table(positions: torch.Tensor, freqs: torch.Tensor) -> torch.Tensor:
    """Outer product of positions with the frequency bank, each freq duplicated."""
    angles = positions.float().unsqueeze(-1) * freqs.float().unsqueeze(0)
    return torch.repeat_interleave(angles, 2, dim=-1)


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x = x.unflatten(-1, (-1, 2))
    x1, x2 = x.unbind(dim=-1)
    return torch.stack((-x2, x1), dim=-1).flatten(-2)


def apply_rotary_emb(freqs: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """Rotate ``t`` with precomputed ``freqs`` (fp32 math, original dtype out).

    Mirrors ``rotary_embedding_torch.apply_rotary_emb`` for the 2-D frequency
    tables and 3-D ``(heads, seq, dim)`` inputs used here: the rotated span is
    ``freqs.shape[-1]`` channels and any remaining channels are untouched.
    """
    dtype = t.dtype
    rot_dim = freqs.shape[-1]
    if rot_dim > t.shape[-1]:
        raise ValueError(f"rotary dim {rot_dim} exceeds the feature dim {t.shape[-1]}")
    if freqs.shape[0] < t.shape[-2]:
        raise ValueError(f"freqs cover {freqs.shape[0]} positions but the sequence has {t.shape[-2]}")
    freqs = freqs[-t.shape[-2] :].to(torch.float32)

    t_middle = t[..., :rot_dim].float()
    rotated = t_middle * freqs.cos() + rotate_half(t_middle) * freqs.sin()
    if rot_dim == t.shape[-1]:
        return rotated.to(dtype)
    return torch.cat((rotated, t[..., rot_dim:]), dim=-1).to(dtype)


class NaMMRotaryEmbedding3d(nn.Module):
    """Reference ``mmrope3d``: axial 3-D RoPE for windows plus 1-D RoPE for text."""

    mm = True

    def __init__(self, rotary_dim: int, num_axes: int = 3, theta: float = LANG_THETA) -> None:
        super().__init__()
        if num_axes <= 0 or rotary_dim < num_axes:
            raise ValueError(f"invalid rotary_dim={rotary_dim} for num_axes={num_axes}")
        self.rotary_dim = int(rotary_dim)
        self.num_axes = int(num_axes)
        # Reference: ``RotaryEmbedding(dim=rope_dim // 3, freqs_for="lang")`` and
        # the axial table concatenates one such bank per axis.
        self.axis_dim = self.rotary_dim // self.num_axes
        self.rot_dim = self.axis_dim * self.num_axes
        if self.rot_dim % 2:
            raise ValueError(
                f"rotary_dim={rotary_dim} with num_axes={num_axes} gives an odd rotary span "
                f"({self.rot_dim}); the reference pairs channels two at a time"
            )
        # Registered as `freqs`.  The released checkpoint stores this table one
        # module deeper (`...attn.rope.rope.freqs`) than this port registers it, so
        # the validation loader normalizes exactly that suffix; every other key
        # must already match the port's parameter layout.
        self.register_buffer("freqs", lang_freqs(self.axis_dim, theta), persistent=True)
        self._axis_cache: dict[tuple, torch.Tensor] = {}

    # -- frequency construction -------------------------------------------
    def _axis_angles(self, length: int, *, device, dtype) -> torch.Tensor:
        """Cache angles for zero-based, unit-stride positions."""
        key = (length, str(device), str(dtype))
        cached = self._axis_cache.get(key)
        if cached is None:
            positions = torch.arange(length, device=self.freqs.device, dtype=torch.float32)
            cached = _axis_table(positions, self.freqs.float()).to(device=device, dtype=dtype)
            if len(self._axis_cache) > 256:
                self._axis_cache.clear()
            self._axis_cache[key] = cached
        return cached

    def _table(self, dims: tuple[int, ...], *, device, dtype) -> torch.Tensor:
        """Axial table of shape ``(*dims, rot_dim)`` (``get_axial_freqs``)."""
        axes = []
        for axis, dim in enumerate(dims):
            angles = self._axis_angles(dim, device=device, dtype=dtype)
            shape = [1] * len(dims) + [angles.shape[-1]]
            shape[axis] = dim
            axes.append(angles.reshape(shape))
        return torch.cat(torch.broadcast_tensors(*axes), dim=-1)

    # -- per-window / per-text positions ----------------------------------
    def window_freqs(self, window_shape: tuple[int, int, int], text_len: int, *, device, dtype) -> torch.Tensor:
        """``[f * h * w, rot_dim]`` frequencies of one window, window-local."""
        f, h, w = (int(v) for v in window_shape)
        table = self._table((text_len + f, h, w), device=device, dtype=dtype)
        return table[text_len : text_len + f].reshape(-1, table.shape[-1])

    def window_freqs_batch(
        self,
        window_shapes: torch.Tensor | list[tuple[int, int, int]],
        text_len: int,
        *,
        device,
        dtype,
    ) -> torch.Tensor:
        """Concatenated per-window video frequencies for a window batch."""
        if isinstance(window_shapes, torch.Tensor):
            shapes = [tuple(int(v) for v in row) for row in window_shapes.tolist()]
        else:
            shapes = [(int(a), int(b), int(c)) for a, b, c in window_shapes]
        if not shapes:
            return torch.empty((0, self.rot_dim), device=device, dtype=dtype)
        return torch.cat([self.window_freqs(s, text_len, device=device, dtype=dtype) for s in shapes], dim=0)

    def text_freqs(self, text_len: int, *, device, dtype) -> torch.Tensor:
        """``[text_len, rot_dim]`` text frequencies (one axis repeated ``rope_dim`` times)."""
        if text_len <= 0:
            return torch.empty((0, self.rot_dim), device=device, dtype=dtype)
        angles = self._axis_angles(text_len, device=device, dtype=dtype)
        return angles.repeat(1, self.num_axes)

    # -- forward ----------------------------------------------------------
    def forward(
        self,
        vid_q: torch.Tensor,  # (L h d)
        vid_k: torch.Tensor,
        vid_freqs: torch.Tensor,  # (L rot_dim)
        txt_q: torch.Tensor,  # (l h d)
        txt_k: torch.Tensor,
        txt_freqs: torch.Tensor,  # (l rot_dim)
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        vid_q = apply_rotary_emb(vid_freqs, vid_q.transpose(0, 1)).transpose(0, 1)
        vid_k = apply_rotary_emb(vid_freqs, vid_k.transpose(0, 1)).transpose(0, 1)
        txt_q = apply_rotary_emb(txt_freqs, txt_q.transpose(0, 1)).transpose(0, 1)
        txt_k = apply_rotary_emb(txt_freqs, txt_k.transpose(0, 1)).transpose(0, 1)
        return vid_q, vid_k, txt_q, txt_k
