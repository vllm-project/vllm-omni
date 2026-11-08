# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""SANA-WM UCPE (Unified Camera Pose Embedding) per-block attention transforms.

Ported from NVlabs/Sana ``sana_camctrl_blocks.py`` for the inference-only
SANA-WM bidirectional 1600M release. Scope is intentionally narrow:

* Pinhole camera only (xi=0); the UCM ``xi`` parameter is fixed.
* Inference path only — no training-time camera-branch dropout.
* Online computation only — the precomputed ``cam_pos_embeds`` shortcut
  used by NVlabs' fused kernels is omitted because vLLM-Omni builds the
  matrices once per transformer forward and shares them across all blocks.
* ``apply_vo=True`` only — the SANA-WM checkpoint always uses the
  inverse-output transform.

Public surface:
    ``prepare_cam_geometry(...)`` -> :class:`SanaWmCamGeometry`, built once per
    transformer forward and threaded through every block. It carries both the
    ``(apply_q, apply_kv, apply_output)`` closures used by the softmax camera
    branch and the projection/RoPE tensors used by ``cam_prep_func(...)``, the
    NVlabs fused camera-prep contract on the GDN camera branch.

Each closure transforms a tensor of shape ``(B, num_heads, N, D)`` where
``N = T * H * W`` (matching the GDN token layout) and ``D == head_dim``,
returning a tensor of the same shape.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from functools import partial

import torch
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Pinhole ray-grid construction
# ---------------------------------------------------------------------------


def _make_pixel_grid(
    height: int,
    width: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Return ``(H, W, 3)`` grid ``(x, y, 1)`` of pixel coordinates."""
    xs = torch.arange(width, device=device, dtype=dtype)
    ys = torch.arange(height, device=device, dtype=dtype)
    y_grid, x_grid = torch.meshgrid(ys, xs, indexing="ij")
    return torch.stack((x_grid, y_grid, torch.ones_like(x_grid)), dim=-1)


def _unproject_pinhole(
    fx: torch.Tensor,
    fy: torch.Tensor,
    cx: torch.Tensor,
    cy: torch.Tensor,
    height: int,
    width: int,
) -> torch.Tensor:
    """Unproject a pixel grid into camera-frame direction vectors.

    Args:
        fx, fy, cx, cy: shape ``(B, T)`` intrinsics in token-grid units — the
            caller divides all four by patch size before calling.
        height, width: token-grid size.

    Returns:
        ``(B, T, H, W, 3)`` direction tensor (NOT normalised).
    """
    B, T = fx.shape
    device = fx.device
    dtype = fx.dtype
    grid = _make_pixel_grid(height, width, device, dtype)  # (H, W, 3)
    u = grid[..., 0]  # (H, W)
    v = grid[..., 1]
    fx_e = fx.view(B, T, 1, 1)
    fy_e = fy.view(B, T, 1, 1)
    cx_e = cx.view(B, T, 1, 1)
    cy_e = cy.view(B, T, 1, 1)
    x = (u - cx_e) / fx_e
    y = (v - cy_e) / fy_e
    z = torch.ones_like(x)
    # Match NVlabs UCM normalisation at xi=0: alpha = 1, gamma = 1/(1+r^2),
    # X = gamma*x, Y = gamma*y, Z = gamma. The resulting vector has unit
    # norm. Computed compactly via direct stack + normalise.
    d_cam = torch.stack((x, y, z), dim=-1)  # (B, T, H, W, 3)
    d_cam = d_cam / d_cam.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    return d_cam


def _world_to_ray_mats(
    d_cam: torch.Tensor,  # (B, T, H, W, 3) unit rays in camera frame
    c2w: torch.Tensor,  # (B, T, 4, 4) camera-to-world SE(3)
) -> torch.Tensor:
    """Build per-pixel ``ray <- world`` SE(3) transforms ``(B, T, H, W, 4, 4)``.

    The ray frame is anchored at the camera origin with the +Z axis along
    the per-pixel viewing direction and a +Y axis defined by the camera-Y
    cross product. This matches NVlabs ``world_to_ray_mats``.
    """
    B, T, H, W, _ = d_cam.shape
    device = d_cam.device
    dtype = d_cam.dtype
    rot_cam = c2w[..., :3, :3]
    trans_cam = c2w[..., :3, 3]
    d_world = torch.einsum("btij,bthwj->bthwi", rot_cam, d_cam)
    cam_y = rot_cam[..., :, 1].unsqueeze(2).unsqueeze(3).expand(B, T, H, W, 3)
    z_ray = F.normalize(d_world, dim=-1, eps=1e-6)
    x_ray = F.normalize(torch.cross(cam_y, z_ray, dim=-1), dim=-1, eps=1e-6)
    y_ray = F.normalize(torch.cross(z_ray, x_ray, dim=-1), dim=-1, eps=1e-6)
    rot_local_to_world = torch.stack((x_ray, y_ray, z_ray), dim=-1)  # (B,T,H,W,3,3)
    rot_world_to_local = rot_local_to_world.transpose(-1, -2)
    t_world = trans_cam.unsqueeze(2).unsqueeze(3).expand(B, T, H, W, 3)
    t_w2l = -torch.einsum("bthwij,bthwj->bthwi", rot_world_to_local, t_world)
    raymats = torch.zeros(B, T, H, W, 4, 4, device=device, dtype=dtype)
    raymats[..., :3, :3] = rot_world_to_local
    raymats[..., :3, 3] = t_w2l
    raymats[..., 3, 3] = 1.0
    # Guard NaNs from degenerate pixels (e.g. nan direction at infinity).
    nan_mask = torch.isnan(d_world).any(-1)
    if nan_mask.any():
        eye = torch.eye(4, device=device, dtype=dtype)
        raymats[nan_mask] = eye
    return raymats


def _process_camera_conditions(
    camera_conditions: torch.Tensor,  # (B, T, 20)
    hw: tuple[int, int, int],
    patch_size: tuple[int, int, int] = (1, 1, 1),
) -> torch.Tensor:
    """Convert ``(B, T, 20)`` SANA-WM camera conditions to ``(B, T, H, W, 4, 4)``
    ``ray <- world`` transforms.

    Layout of the last axis is ``[c2w_flat (16), fx, fy, cx, cy]``.
    Intrinsics follow the NVlabs UCPE contract: the packed camera payload is
    expressed on the pre-patch latent image grid, while ``HW`` is the
    transformer token grid. ``patch_size`` maps those units to the token grid
    before unprojection. The released SANA-WM P1 config uses ``(1, 1, 1)``.

    Args:
        camera_conditions: ``(B, T, 20)`` from ``_pack_camera_conditions``.
        hw: ``(T, H, W)`` token-grid shape. Equal to the latent grid only
            because the released config patches by ``(1, 1, 1)``.
    """
    B, T_cond, last = camera_conditions.shape
    if last != 20:
        raise ValueError(f"SANA-WM camera_conditions last dim must be 20, got {last}.")
    T_latent, H_latent, W_latent = hw
    if T_cond != T_latent:
        raise ValueError(f"SANA-WM camera_conditions frames {T_cond} != latent frames {T_latent}.")
    c2w = camera_conditions[..., :16].reshape(B, T_latent, 4, 4)
    patch_t, patch_h, patch_w = patch_size
    del patch_t
    fx = camera_conditions[..., 16] / float(patch_w)
    fy = camera_conditions[..., 17] / float(patch_h)
    cx = camera_conditions[..., 18] / float(patch_w)
    cy = camera_conditions[..., 19] / float(patch_h)
    d_cam = _unproject_pinhole(fx, fy, cx, cy, H_latent, W_latent)
    raymats = _world_to_ray_mats(d_cam, c2w)
    return raymats


def _invert_se3(transforms: torch.Tensor) -> torch.Tensor:
    """Closed-form inverse of a stack of 4x4 SE(3) matrices."""
    if transforms.shape[-2:] != (4, 4):
        raise ValueError(f"SE3 inverse expects (..., 4, 4), got {tuple(transforms.shape)}.")
    rot_inv = transforms[..., :3, :3].transpose(-1, -2)
    out = torch.zeros_like(transforms)
    out[..., :3, :3] = rot_inv
    out[..., :3, 3] = -torch.einsum("...ij,...j->...i", rot_inv, transforms[..., :3, 3])
    out[..., 3, 3] = 1.0
    return out


# ---------------------------------------------------------------------------
# Per-token transform primitives
# ---------------------------------------------------------------------------


def _apply_ray_projmat(
    feats: torch.Tensor,  # (B, num_heads, N, D)
    matrix: torch.Tensor,  # (B, N, 4, 4)
) -> torch.Tensor:
    """Apply a per-token 4x4 projection to features grouped by 4 channels.

    ``D`` must be a multiple of 4. Out-of-place; preserves ``feats.shape``.
    """
    B, num_heads, N, D = feats.shape
    if D % 4 != 0:
        raise ValueError(f"UCPE projmat expects D divisible by 4, got D={D}.")
    feats_grouped = feats.reshape(B, num_heads, N, D // 4, 4)
    # matrix: (B, N, 4, 4); feats_grouped: (B, num_heads, N, K, 4)
    out = torch.einsum("bnij,bhnkj->bhnki", matrix, feats_grouped)
    return out.reshape(B, num_heads, N, D)


def _apply_complex_rope(
    hidden_states: torch.Tensor,
    freqs: torch.Tensor,
    *,
    inverse: bool = False,
) -> torch.Tensor:
    """Apply complex RoPE: treat last-dim pairs as ``(real, imag)`` and multiply
    by ``freqs``. Computation runs in fp64 to match the NVlabs reference.

    Args:
        hidden_states: ``(..., D)`` with ``D % 2 == 0``.
        freqs: complex tensor broadcastable to ``(..., D // 2)``.
        inverse: when ``True`` apply ``freqs.conj()`` (used for ``apply_o``).
    """
    x_real = hidden_states.to(torch.float64).contiguous()
    x_complex = torch.view_as_complex(x_real.unflatten(-1, (-1, 2)))
    if inverse:
        freqs = freqs.conj()
    x_out = torch.view_as_real(x_complex * freqs).flatten(-2, -1)
    return x_out.to(hidden_states.dtype)


def _apply_block_diagonal(
    feats: torch.Tensor,
    func_size_pairs: list[tuple[Callable[[torch.Tensor], torch.Tensor], int]],
) -> torch.Tensor:
    """Split ``feats`` along the last axis by ``block_sizes`` and apply one
    callable per block, then concatenate. Preserves ``feats.shape``.
    """
    funcs, block_sizes = zip(*func_size_pairs)
    if feats.shape[-1] != sum(block_sizes):
        raise ValueError(f"UCPE block-diagonal block sum {sum(block_sizes)} != feats D {feats.shape[-1]}.")
    blocks = torch.split(feats, list(block_sizes), dim=-1)
    return torch.cat([func(block) for func, block in zip(funcs, blocks)], dim=-1)


def _slice_rope_for_cam(
    rotary_emb: torch.Tensor | None,
    rope_dim: int,
) -> torch.Tensor | None:
    """Re-slice WAN-style RoPE frequencies for a smaller ``rope_dim`` using the
    same ``(T, H, W)`` split as the main branch. Mirrors NVlabs slicing logic.

    The T/H/W band layout is read off the input shape
    (``rotary_emb.shape[-1] * 2``) rather than a separately supplied head_dim,
    so the split stays correct without the caller passing it. ``cam_head_dim``
    still has to equal the main ``head_dim`` — ``SanaWmSelfAttention.__init__``
    rejects anything else.
    """
    if rotary_emb is None:
        return None
    # Derive WAN RoPE T/H/W band offsets from the actual input tensor size,
    # not from a head_dim argument. This is correct regardless of whether
    # cam_head_dim == main_head_dim.
    in_head_dim = rotary_emb.shape[-1] * 2
    orig_t = in_head_dim // 2 - 2 * (in_head_dim // 6)
    orig_h = in_head_dim // 6
    new_t = rope_dim // 2 - 2 * (rope_dim // 6)
    new_h = rope_dim // 6
    new_w = rope_dim // 6
    t_part = rotary_emb[..., :new_t]
    h_part = rotary_emb[..., orig_t : orig_t + new_h]
    w_part = rotary_emb[..., orig_t + orig_h : orig_t + orig_h + new_w]
    return torch.cat([t_part, h_part, w_part], dim=-1)


# ---------------------------------------------------------------------------
# Apply-fn preparation
# ---------------------------------------------------------------------------


def _identity(x: torch.Tensor) -> torch.Tensor:
    return x


def _prepare_ray_apply_fns(
    head_dim: int,
    proj: torch.Tensor,  # (B, N, 4, 4) ray <- world
    proj_t: torch.Tensor,  # (B, N, 4, 4) ray <- world transpose
    proj_inv: torch.Tensor,  # (B, N, 4, 4) world <- ray
    rotary_emb: torch.Tensor | None = None,
) -> tuple[Callable, Callable, Callable]:
    """Build ``(apply_q, apply_kv, apply_o)`` block-diagonal callables.

    Each callable transforms a ``(B, num_heads, N, head_dim)`` tensor:
    the first ``head_dim // 2`` channels go through the per-token 4x4
    projection, the remaining ``head_dim // 2`` channels go through complex
    RoPE (or identity when ``rotary_emb`` is None).
    """
    if rotary_emb is not None:
        rope_fwd = partial(_apply_complex_rope, freqs=rotary_emb, inverse=False)
        rope_inv = partial(_apply_complex_rope, freqs=rotary_emb, inverse=True)
    else:
        rope_fwd = _identity
        rope_inv = _identity

    half = head_dim // 2

    transforms_q = [
        (partial(_apply_ray_projmat, matrix=proj_t), half),
        (rope_fwd, half),
    ]
    transforms_kv = [
        (partial(_apply_ray_projmat, matrix=proj_inv), half),
        (rope_fwd, half),
    ]
    transforms_o = [
        (partial(_apply_ray_projmat, matrix=proj), half),
        (rope_inv, half),
    ]

    apply_q = partial(_apply_block_diagonal, func_size_pairs=transforms_q)
    apply_kv = partial(_apply_block_diagonal, func_size_pairs=transforms_kv)
    apply_o = partial(_apply_block_diagonal, func_size_pairs=transforms_o)
    return apply_q, apply_kv, apply_o


@dataclass(frozen=True)
class SanaWmCamGeometry:
    """UCPE geometry shared by every block of one transformer forward.

    The ray grid, its SE(3) inverse and the camera RoPE tables depend only on
    the camera payload, the latent grid, the patch size and the head dim — all
    fixed for the whole forward — yet each block's camera branch used to
    rebuild them. At 20 blocks x 60 steps x 2 CFG branches that is ~2400
    rebuilds of an 18480-token ray grid and its 4x4 inverses per request.

    ``apply_q``/``apply_kv``/``apply_output`` serve the softmax camera branch;
    ``proj_q``/``proj_kv``/``rope_cos``/``rope_sin`` serve the fused
    ``cam_prep_func`` contract on the GDN branch. Both are derived from the
    same projection matrices, so they are built together.
    """

    proj_q: torch.Tensor
    proj_kv: torch.Tensor
    rope_cos: torch.Tensor
    rope_sin: torch.Tensor
    apply_q: Callable[[torch.Tensor], torch.Tensor]
    apply_kv: Callable[[torch.Tensor], torch.Tensor]
    apply_output: Callable[[torch.Tensor], torch.Tensor]


def _prepare_ucpe_rope_tables(
    rotary_emb_cam: torch.Tensor,
    token_count: int,
    half_dim: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert complex RoPE freqs to NVlabs' interleaved cos/sin tables.

    The output convention is ``out[d] = x[d] * cos[d] + x[d ^ 1] * sin[d]``.
    For every complex pair, ``sin[2i] = -imag(freq[i])`` and
    ``sin[2i + 1] = imag(freq[i])``.
    """
    freqs = rotary_emb_cam
    while freqs.ndim > 2 and freqs.shape[0] == 1:
        freqs = freqs.squeeze(0)
    if freqs.ndim != 2:
        raise ValueError(f"Sana-WM cam RoPE freqs must be 2D after squeeze, got {tuple(freqs.shape)}.")
    if freqs.shape[0] != token_count:
        raise ValueError(f"Sana-WM cam RoPE token count {freqs.shape[0]} != {token_count}.")
    if freqs.shape[1] * 2 != half_dim:
        raise ValueError(f"Sana-WM cam RoPE dim {freqs.shape[1] * 2} != half_dim {half_dim}.")
    cos_half = freqs.real.float()
    sin_half = freqs.imag.float()
    rope_cos = cos_half.repeat_interleave(2, dim=-1).contiguous()
    rope_sin = torch.stack((-sin_half, sin_half), dim=-1).reshape(token_count, half_dim).contiguous()
    return rope_cos, rope_sin


def prepare_cam_geometry(
    *,
    camera_conditions: torch.Tensor,
    spatial_shape: tuple[int, int, int],
    patch_size: tuple[int, int, int],
    head_dim: int,
    rotary_emb: torch.Tensor | None,
) -> SanaWmCamGeometry:
    """Build the per-forward UCPE geometry: ray projections, RoPE tables, apply fns.

    Call once per transformer forward and share the result across every block —
    see :class:`SanaWmCamGeometry` for why.
    """
    if head_dim % 4 != 0:
        raise ValueError(f"UCPE head_dim must be divisible by 4, got {head_dim}.")
    if head_dim % 2 != 0 or (head_dim // 2) % 4 != 0:
        raise ValueError(f"Sana-WM cam prep head_dim={head_dim} must satisfy D % 2 == 0 and (D/2) % 4 == 0.")

    batch = camera_conditions.shape[0]
    frames, height, width = spatial_shape
    token_count = frames * height * width
    half_dim = head_dim // 2

    raymats = _process_camera_conditions(camera_conditions, spatial_shape, patch_size=patch_size)
    proj = raymats.reshape(batch, token_count, 4, 4).contiguous()
    proj_q = proj.transpose(-1, -2).contiguous()
    proj_kv = _invert_se3(proj).contiguous()

    rotary_emb_cam = _slice_rope_for_cam(rotary_emb, half_dim)
    if rotary_emb_cam is None:
        rope_cos = torch.ones(token_count, half_dim, device=camera_conditions.device, dtype=torch.float32)
        rope_sin = torch.zeros(token_count, half_dim, device=camera_conditions.device, dtype=torch.float32)
    else:
        rope_cos, rope_sin = _prepare_ucpe_rope_tables(rotary_emb_cam, token_count, half_dim)

    apply_q, apply_kv, apply_output = _prepare_ray_apply_fns(head_dim, proj, proj_q, proj_kv, rotary_emb=rotary_emb_cam)
    return SanaWmCamGeometry(
        proj_q=proj_q,
        proj_kv=proj_kv,
        rope_cos=rope_cos,
        rope_sin=rope_sin,
        apply_q=apply_q,
        apply_kv=apply_kv,
        apply_output=apply_output,
    )


def _validate_cam_prep_inputs(
    q_raw: torch.Tensor,
    k_raw: torch.Tensor,
    v_raw: torch.Tensor,
    proj_q: torch.Tensor,
    proj_kv: torch.Tensor,
    rope_cos: torch.Tensor,
    rope_sin: torch.Tensor,
) -> int:
    """Validate the fused cam-prep contract and return ``head_dim // 2``."""
    if q_raw.shape != k_raw.shape or q_raw.shape != v_raw.shape:
        raise ValueError(f"Sana-WM cam prep q/k/v shapes must match, got {q_raw.shape}, {k_raw.shape}, {v_raw.shape}.")
    if q_raw.ndim != 4:
        raise ValueError(f"Sana-WM cam prep q/k/v must be (B,N,H,D), got {tuple(q_raw.shape)}.")
    batch, token_count, _num_heads, head_dim = q_raw.shape
    if head_dim % 2 != 0 or (head_dim // 2) % 4 != 0:
        raise ValueError(f"Sana-WM cam prep head_dim={head_dim} must satisfy D % 2 == 0 and (D/2) % 4 == 0.")
    half_dim = head_dim // 2
    if proj_q.shape != (batch, token_count, 4, 4):
        raise ValueError(f"Sana-WM cam prep proj_q shape {tuple(proj_q.shape)} != {(batch, token_count, 4, 4)}.")
    if proj_kv.shape != (batch, token_count, 4, 4):
        raise ValueError(f"Sana-WM cam prep proj_kv shape {tuple(proj_kv.shape)} != {(batch, token_count, 4, 4)}.")
    if rope_cos.shape != (token_count, half_dim) or rope_sin.shape != (token_count, half_dim):
        raise ValueError(
            "Sana-WM cam prep rope table shapes must be "
            f"{(token_count, half_dim)}, got {tuple(rope_cos.shape)} and {tuple(rope_sin.shape)}."
        )
    return half_dim


def _apply_ray_projection(feats: torch.Tensor, matrix: torch.Tensor) -> torch.Tensor:
    batch, token_count, num_heads, half_dim = feats.shape
    grouped = feats.reshape(batch, token_count, num_heads, half_dim // 4, 4)
    out = torch.einsum("bnij,bnhgj->bnhgi", matrix.float(), grouped)
    return out.reshape(batch, token_count, num_heads, half_dim)


def _apply_interleaved_rope(feats: torch.Tensor, rope_cos: torch.Tensor, rope_sin: torch.Tensor) -> torch.Tensor:
    half_dim = feats.shape[-1]
    pair_index = torch.arange(half_dim, device=feats.device) ^ 1
    paired = feats.index_select(-1, pair_index)
    return feats * rope_cos.reshape(1, rope_cos.shape[0], 1, half_dim) + paired * rope_sin.reshape(
        1, rope_sin.shape[0], 1, half_dim
    )


def cam_prep_func(
    q_normed: torch.Tensor,
    k_normed: torch.Tensor,
    v_raw: torch.Tensor,
    *,
    proj_q: torch.Tensor,
    proj_kv: torch.Tensor,
    rope_cos: torch.Tensor,
    rope_sin: torch.Tensor,
    k_scale: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Native equivalent of NVlabs ``cam_prep_func``.

    The caller is responsible for the q/k RMSNorm, which must run through the
    model's norm modules so a tensor-parallel shard still normalises over the
    global channel width. This function only applies ReLU, the K scale, the
    UCPE ray projection and RoPE.

    Args:
        q_normed: ``(B, N, H, D)`` camera Q, already RMSNorm'd.
        k_normed: ``(B, N, H, D)`` camera K, already RMSNorm'd. K must also
            include the temporal short convolution.
        v_raw: ``(B, N, H, D)`` raw camera V.
        proj_q: ``(B, N, 4, 4)`` Q ray projection matrices.
        proj_kv: ``(B, N, 4, 4)`` K/V ray projection matrices.
        rope_cos: ``(N, D//2)`` interleaved-pair RoPE cosine table.
        rope_sin: ``(N, D//2)`` interleaved-pair RoPE sine table.
        k_scale: ``D^-0.5 * spatial_tokens^-0.5``.

    Returns:
        ``q_trans``, ``k_trans``, ``v_trans`` in ``(B, H, D, N)`` layout, and
        ``inflation_sq`` in ``(B, H, N)``.
    """
    half_dim = _validate_cam_prep_inputs(
        q_normed,
        k_normed,
        v_raw,
        proj_q,
        proj_kv,
        rope_cos,
        rope_sin,
    )

    q_norm = torch.relu(q_normed.float())
    k_norm = torch.relu(k_normed.float()) * float(k_scale)
    value = v_raw.float()

    q_half = _apply_ray_projection(q_norm[..., :half_dim], proj_q)
    k_half = _apply_ray_projection(k_norm[..., :half_dim], proj_kv)
    v_half = _apply_ray_projection(value[..., :half_dim], proj_kv)

    q_rope = _apply_interleaved_rope(q_norm[..., half_dim:], rope_cos.float(), rope_sin.float())
    k_rope = _apply_interleaved_rope(k_norm[..., half_dim:], rope_cos.float(), rope_sin.float())
    v_rope = _apply_interleaved_rope(value[..., half_dim:], rope_cos.float(), rope_sin.float())

    k_pre_sq = k_norm.square().sum(dim=-1).permute(0, 2, 1).contiguous()
    k_post = torch.cat((k_half, k_rope), dim=-1)
    k_post_sq = k_post.square().sum(dim=-1).permute(0, 2, 1).contiguous()
    inflation_sq = k_post_sq.clamp_min(1e-12) / k_pre_sq.clamp_min(1e-12)

    def to_bhdn(tensor: torch.Tensor) -> torch.Tensor:
        return tensor.permute(0, 2, 3, 1).contiguous().to(v_raw.dtype)

    q_out = to_bhdn(torch.cat((q_half, q_rope), dim=-1))
    k_out = to_bhdn(k_post)
    v_out = to_bhdn(torch.cat((v_half, v_rope), dim=-1))
    return q_out, k_out, v_out, inflation_sq


__all__ = [
    "SanaWmCamGeometry",
    "cam_prep_func",
    "prepare_cam_geometry",
]
