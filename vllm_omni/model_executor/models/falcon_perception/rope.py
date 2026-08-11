# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Falcon Perception split rotary embeddings.

``head_dim`` is split in half:

* the first half gets a standard 1-D RoPE indexed by the temporal position
  ``pos_t`` (the whole image block shares a single temporal position);
* the second half gets a learned 2-D "golden gate" rotation driven by a
  continuous, aspect-normalised ``(h, w)`` grid. Text tokens sit at ``(0, 0)``,
  which gives ``theta = 0`` and therefore an identity rotation, so the spatial
  half is a no-op outside images and can be skipped entirely during decode.

The frequencies for the spatial half are **learned per attention head** and
loaded from the checkpoint buffer ``freqs_cis_golden`` of shape
``(n_heads, head_dim // 4, 2)``. Because those rows differ across heads within
a GQA group, K must be expanded to ``n_heads`` before the rotation is applied —
see the note in ``falcon_perception_thinker.FalconPerceptionAttention``.

Ported from the reference implementation's ``falcon_perception/rope.py``, with
tensors in vLLM's flat ``(num_tokens, num_heads, head_dim)`` layout instead of
``(batch, seq, heads, dim)``.
"""

from __future__ import annotations

import torch


def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0) -> torch.Tensor:
    """Precompute complex 1-D RoPE frequencies. Returns ``(end, dim // 2)``."""
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device)
    freqs = torch.outer(t, freqs).float()
    return torch.polar(torch.ones_like(freqs), freqs)


def apply_rotary_emb_1d(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Standard interleaved-pair RoPE over the temporal half.

    ``xq``/``xk`` are ``(num_tokens, num_heads, head_dim // 2)`` and
    ``freqs_cis`` is ``(num_tokens, head_dim // 4)`` complex.
    """
    # (N, f) -> (N, 1, f) so it broadcasts across heads.
    freqs_cis = freqs_cis.unsqueeze(1)
    cos, sin = freqs_cis.real, freqs_cis.imag

    def rotate(x_in: torch.Tensor) -> torch.Tensor:
        # Spell out complex multiplication in real arithmetic. This remains
        # bit-exact with the reference operation, while avoiding an AOT-load
        # failure in Inductor's symbolic handling of ``view_as_complex``.
        x = x_in.float()
        x_even, x_odd = x[..., 0::2], x[..., 1::2]
        out = torch.empty_like(x)
        out[..., 0::2] = x_even * cos - x_odd * sin
        out[..., 1::2] = x_even * sin + x_odd * cos
        return out.type_as(x_in)

    return rotate(xq), rotate(xk)


def apply_golden_freqs_cis_to_visual_pos(
    golden_freqs: torch.Tensor,
    pos_hw: torch.Tensor,
) -> torch.Tensor:
    """Build per-token, per-head complex 2-D frequencies.

    ``golden_freqs`` is the learned ``(num_heads, head_dim // 4, 2)`` buffer and
    ``pos_hw`` is ``(num_tokens, 2)`` holding ``(h, w)``. Text tokens are at
    ``(0, 0)`` so they come back as ``1 + 0j`` (identity).

    Returns ``(num_tokens, num_heads, head_dim // 4)`` complex.
    """
    theta = torch.einsum("np,hfp->nhf", pos_hw.float(), golden_freqs.float())
    return torch.polar(torch.ones_like(theta), theta)


def apply_golden_rotary_emb(x_in: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
    """Apply the learned 2-D rotation to the spatial half.

    Note this uses the **even/odd interleaved** convention (``x[..., 0::2]`` /
    ``x[..., 1::2]``), not the split-half convention used elsewhere in vLLM.
    """
    x = x_in.float()
    x_even = x[..., 0::2]
    x_odd = x[..., 1::2]

    cos = freqs_cis.real
    sin = freqs_cis.imag

    out = torch.empty_like(x)
    out[..., 0::2] = x_even * cos - x_odd * sin
    out[..., 1::2] = x_even * sin + x_odd * cos
    return out.type_as(x_in)


def apply_3d_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor,
    freqs_cis_2d: torch.Tensor | None,
    spatial_enabled: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Temporal RoPE on the first half of ``head_dim``, golden 2-D on the second."""
    xq_t, xq_hw = xq.chunk(chunks=2, dim=-1)
    xk_t, xk_hw = xk.chunk(chunks=2, dim=-1)

    xq_t, xk_t = apply_rotary_emb_1d(xq_t, xk_t, freqs_cis)
    if freqs_cis_2d is not None:
        xq_hw_rotated = apply_golden_rotary_emb(xq_hw, freqs_cis_2d)
        xk_hw_rotated = apply_golden_rotary_emb(xk_hw, freqs_cis_2d)
        if spatial_enabled is None:
            xq_hw, xk_hw = xq_hw_rotated, xk_hw_rotated
        else:
            # A tensor gate keeps the compiled backbone's signature stable
            # across prefill and decode. Selecting the original values when
            # disabled also preserves the reference decode path bit-for-bit;
            # rotating by an identity complex value still introduces a float
            # round trip for bf16 inputs.
            enabled = spatial_enabled.reshape(1, 1, 1)
            xq_hw = torch.where(enabled, xq_hw_rotated, xq_hw)
            xk_hw = torch.where(enabled, xk_hw_rotated, xk_hw)

    xq_out = torch.cat([xq_t, xq_hw], dim=-1).type_as(xq)
    xk_out = torch.cat([xk_t, xk_hw], dim=-1).type_as(xk)
    return xq_out, xk_out


def compute_image_spatial_positions(
    grid_h: int,
    grid_w: int,
    *,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Continuous aspect-normalised ``(h, w)`` grid for one image.

    Mirrors ``data._compute_image_spatial_positions``: the grid is normalised so
    that the longer side spans a wider range, keeping the aspect ratio in the
    rotation rather than in the token count.

    Returns ``(grid_h * grid_w, 2)`` in row-major (h-major) patch order.
    """
    xlim = (grid_w / grid_h) ** 0.5
    ylim = (grid_h / grid_w) ** 0.5
    xpos = torch.linspace(-xlim, xlim, grid_w, device=device, dtype=dtype)
    ypos = torch.linspace(-ylim, ylim, grid_h, device=device, dtype=dtype)
    # meshgrid(..., indexing="ij") over (y, x) gives h-major row order, matching
    # numpy's meshgrid(xpos, ypos, indexing="xy") in the reference.
    hpos, wpos = torch.meshgrid(ypos, xpos, indexing="ij")
    return torch.stack([hpos.flatten(), wpos.flatten()], dim=-1)


# Rows 1 and 2 of the ``[3, N]`` M-RoPE positions buffer are unused by this
# model (``mrope_section`` is ``[head_dim // 2, 0, 0]``), so they transport the
# patch grid instead: each image token packs its grid dimension and its index
# along that axis as ``dim * _GRID_STRIDE + index``. Any patch grid is bounded
# by ``max_image_size // spatial_patch_size`` (64 for the released configs), so
# this stride leaves three orders of magnitude of headroom.
_GRID_STRIDE = 4096


def pack_image_grid_positions(
    grid_h: int,
    grid_w: int,
    *,
    device: torch.device | None = None,
) -> torch.Tensor:
    """Pack one image's patch grid into two int64 columns.

    Returns ``(grid_h * grid_w, 2)`` in the same h-major patch order as
    :func:`compute_image_spatial_positions`; column 0 packs ``(grid_h, row)``
    and column 1 packs ``(grid_w, col)``.

    Repeating the grid *dimensions* on every token, rather than only the
    indices, is what makes the unpacking exact when a forward pass sees only
    part of an image — a prefix-cache hit resumes mid-span, and the runner
    slices these rows by ``num_computed_tokens`` — or several images of
    different sizes at once.
    """
    if not 0 < grid_h < _GRID_STRIDE or not 0 < grid_w < _GRID_STRIDE:
        raise ValueError(f"Falcon Perception: patch grid {(grid_h, grid_w)} is outside 1..{_GRID_STRIDE - 1}")
    rows = torch.arange(grid_h, device=device).repeat_interleave(grid_w)
    cols = torch.arange(grid_w, device=device).repeat(grid_h)
    return torch.stack([grid_h * _GRID_STRIDE + rows, grid_w * _GRID_STRIDE + cols], dim=-1)


def unpack_image_grid_positions(packed: torch.Tensor) -> torch.Tensor:
    """Inverse of :func:`pack_image_grid_positions`, giving continuous ``(h, w)``.

    ``packed`` is ``(n, 2)`` int64. Rows are grouped by their ``(grid_h,
    grid_w)`` and each group is rebuilt with the very same ``linspace`` call
    :func:`compute_image_spatial_positions` makes, so the result is bit-exact
    rather than merely close.
    """
    grid = torch.div(packed, _GRID_STRIDE, rounding_mode="floor")
    index = packed - grid * _GRID_STRIDE
    out = torch.zeros((packed.shape[0], 2), device=packed.device, dtype=torch.float32)
    # One iteration per distinct image size in the batch, not per image.
    for grid_h, grid_w in torch.unique(grid, dim=0).tolist():
        rows = (grid[:, 0] == grid_h) & (grid[:, 1] == grid_w)
        flat = index[rows, 0] * grid_w + index[rows, 1]
        out[rows] = compute_image_spatial_positions(grid_h, grid_w, device=packed.device)[flat]
    return out


def compute_pos_hw(
    input_ids: torch.Tensor,
    positions: torch.Tensor,
    *,
    image_token_id: int,
) -> torch.Tensor | None:
    """Rebuild the continuous ``(h, w)`` grid from the M-RoPE positions buffer.

    The grid is read back per token from rows 1/2 of ``positions`` (see
    :func:`pack_image_grid_positions`) rather than from the multimodal kwargs.
    That keeps it immune to the multimodal encoder cache, which serves the
    stored *output* of ``embed_multimodal`` on a hit and so never runs the call
    — nor its kwargs — for a repeated image.

    Non-image tokens keep ``(0, 0)`` so their golden rotation is the identity.
    Returns ``(num_tokens, 2)``, or ``None`` when there is nothing to rotate:
    no image patch tokens in the batch (a pure decode step), or unpacked rows
    (the profile run feeds a zeroed positions buffer).
    """
    if positions.dim() != 2 or positions.shape[0] < 3:
        return None
    image_positions = (input_ids == image_token_id).nonzero(as_tuple=True)[0]
    if image_positions.numel() == 0:
        return None

    packed = torch.stack([positions[1][image_positions], positions[2][image_positions]], dim=-1)
    # Every packed entry carries a grid dimension of at least 1, so anything
    # below one stride means these rows were never filled in.
    if bool((packed < _GRID_STRIDE).any()):
        return None

    pos_hw = torch.zeros((input_ids.shape[0], 2), device=input_ids.device, dtype=torch.float32)
    pos_hw[image_positions] = unpack_image_grid_positions(packed)
    return pos_hw
