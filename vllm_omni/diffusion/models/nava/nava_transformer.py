# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Native vLLM-Omni adapter for the NAVA MMDiT backbone.

Most backbone components below are ported from upstream NAVA:
https://github.com/ernie-research/NAVA/blob/main/nava_src/models/nava/modules/model_mm.py

The local changes remove upstream Diffusers mixins and NAVA-specific distributed
helpers, keep module names close to the checkpoint layout, and add the
``NAVATransformer`` wrapper/load path used by vLLM-Omni.
"""

import math
import warnings
from collections.abc import Iterable
from contextlib import nullcontext

import torch
import torch.amp as amp
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel
from vllm.logger import init_logger
from vllm.model_executor.model_loader.weight_utils import default_weight_loader

from vllm_omni.diffusion.models.nava.config import NAVAConfig

logger = init_logger(__name__)

_NAVA_CUDA_SDPA_BACKENDS = [SDPBackend.CUDNN_ATTENTION]


class ChannelLastConv1d(nn.Conv1d):
    """NAVA-local channel-last Conv1d wrapper used by upstream audio patch embedding."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 2, 1)
        x = super().forward(x)
        x = x.permute(0, 2, 1)
        return x


class ConvMLP(nn.Module):
    """NAVA-local ConvMLP matching upstream model_mm.py."""

    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        multiple_of: int = 256,
        kernel_size: int = 3,
        padding: int = 1,
    ) -> None:
        super().__init__()
        hidden_dim = int(2 * hidden_dim / 3)
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)

        self.w1 = ChannelLastConv1d(dim, hidden_dim, bias=False, kernel_size=kernel_size, padding=padding)
        self.w2 = ChannelLastConv1d(hidden_dim, dim, bias=False, kernel_size=kernel_size, padding=padding)
        self.w3 = ChannelLastConv1d(dim, hidden_dim, bias=False, kernel_size=kernel_size, padding=padding)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class WanRMSNorm(nn.Module):
    """NAVA/Wan query-key RMSNorm with upstream BF16 arithmetic semantics."""

    def __init__(self, dim: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._norm(x.bfloat16()).type_as(x) * self.weight.bfloat16()

    def _norm(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)


class WanLayerNorm(nn.LayerNorm):
    """NAVA/Wan LayerNorm with upstream BF16 arithmetic semantics."""

    def __init__(self, dim: int, eps: float = 1e-6, elementwise_affine: bool = False) -> None:
        super().__init__(dim, eps=eps, elementwise_affine=elementwise_affine)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return super().forward(x.bfloat16()).type_as(x)


# TODO: Wire NAVA through vLLM-Omni's Attention layer when enabling native
# TP/SP/CFG support. This local SDPA fallback preserves upstream NAVA behavior
# but intentionally bypasses framework attention sharding hooks.
def _nava_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    q_lens: torch.Tensor | None = None,
    k_lens: torch.Tensor | None = None,
    window_size: tuple[int, int] | list[int] | None = None,
    dropout_p: float = 0.0,
    softmax_scale: float | None = None,
    q_scale: float | torch.Tensor | None = None,
    causal: bool = False,
    dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    if window_size is None:
        window_size = (-1, -1)
    if isinstance(window_size, list):
        window_size = tuple(window_size)
    half_dtypes = (torch.float16, torch.bfloat16)
    assert dtype in half_dtypes
    out_dtype = q.dtype
    batch, query_len, key_len = q.size(0), q.size(1), k.size(1)

    if q_lens is None:
        q_lens = torch.full((batch,), query_len, dtype=torch.int32, device=q.device)
    else:
        q_lens = q_lens.to(device=q.device, dtype=torch.int32)
    if k_lens is None:
        k_lens = torch.full((batch,), key_len, dtype=torch.int32, device=k.device)
    else:
        k_lens = k_lens.to(device=k.device, dtype=torch.int32)

    if window_size != (-1, -1):
        warnings.warn("Sliding-window attention is ignored by the scaled_dot_product_attention fallback.")
    compute_dtype = v.dtype if v.dtype in half_dtypes else dtype
    q_sdpa = q.to(compute_dtype)
    k_sdpa = k.to(compute_dtype)
    v_sdpa = v.to(compute_dtype)
    if q_scale is not None:
        q_sdpa = (q_sdpa * q_scale).to(compute_dtype)

    chunks = []
    for index in range(batch):
        cur_query_len = int(q_lens[index].item())
        cur_key_len = int(k_lens[index].item())
        query_item = q_sdpa[index : index + 1, :cur_query_len].transpose(1, 2)
        key_item = k_sdpa[index : index + 1, :cur_key_len].transpose(1, 2)
        value_item = v_sdpa[index : index + 1, :cur_key_len].transpose(1, 2)
        if query_item.size(1) != key_item.size(1):
            if query_item.size(1) % key_item.size(1) != 0:
                raise ValueError(
                    "NAVA attention requires query heads to be divisible by key heads: "
                    f"got {query_item.size(1)} and {key_item.size(1)}."
                )
            repeat = query_item.size(1) // key_item.size(1)
            key_item = key_item.repeat_interleave(repeat, dim=1)
            value_item = value_item.repeat_interleave(repeat, dim=1)
        sdpa_context = sdpa_kernel(_NAVA_CUDA_SDPA_BACKENDS) if query_item.is_cuda else nullcontext()
        try:
            with sdpa_context:
                output = F.scaled_dot_product_attention(
                    query_item,
                    key_item,
                    value_item,
                    attn_mask=None,
                    dropout_p=dropout_p,
                    is_causal=causal,
                    scale=softmax_scale,
                )
        except RuntimeError as e:
            if not query_item.is_cuda or "No available kernel" not in str(e):
                raise
            logger.warning_once(
                "cuDNN SDPA rejected this NAVA attention shape; falling back to default SDPA dispatcher."
            )
            output = F.scaled_dot_product_attention(
                query_item,
                key_item,
                value_item,
                attn_mask=None,
                dropout_p=dropout_p,
                is_causal=causal,
                scale=softmax_scale,
            )
        output = output.transpose(1, 2)
        if cur_query_len < query_len:
            output = F.pad(output, (0, 0, 0, 0, 0, query_len - cur_query_len))
        chunks.append(output)
    return torch.cat(chunks, dim=0).to(out_dtype)


def _sinusoidal_embedding_1d(dim, position):
    # preprocess
    assert dim % 2 == 0
    half = dim // 2
    position = position.type(torch.float64)

    # calculation
    sinusoid = torch.outer(position, torch.pow(10000, -torch.arange(half).to(position).div(half)))
    x = torch.cat([torch.cos(sinusoid), torch.sin(sinusoid)], dim=1)
    return x


@amp.autocast("cuda", enabled=False)
def _rope_params(max_seq_len, dim, theta=10000, freqs_scaling=1.0):
    assert dim % 2 == 0
    pos = torch.arange(max_seq_len)
    freqs = 1.0 / torch.pow(theta, torch.arange(0, dim, 2).to(torch.float64).div(dim))
    freqs = freqs_scaling * freqs
    freqs = torch.outer(pos, freqs)
    freqs = torch.polar(torch.ones_like(freqs), freqs)
    return freqs


@amp.autocast("cuda", enabled=False)
def _rope_apply_joint(x, grid_sizes_vid, grid_sizes_audio, freqs_vid, freqs_audio, vid_seq_len):
    x_vid = x[:, :vid_seq_len, :, :]
    x_audio = x[:, vid_seq_len:, :, :]
    x_video_rope = _rope_apply_3d(x_vid, grid_sizes_vid, freqs_vid)
    x_audio_rope = _rope_apply_1d(x_audio, grid_sizes_audio, freqs_audio)
    x_rope = torch.cat([x_video_rope, x_audio_rope], dim=1)
    return x_rope


@amp.autocast("cuda", enabled=False)
def _rope_apply_1d(x, grid_sizes, freqs):
    n, c = x.size(2), x.size(3) // 2  ## b l h d
    c_rope = freqs.shape[1]  # number of complex dims to rotate
    assert c_rope <= c, "RoPE dimensions cannot exceed half of hidden size"

    output = []
    for i, (length,) in enumerate(grid_sizes.tolist()):
        seq_len = length
        x_i = torch.view_as_complex(x[i, :seq_len].to(torch.float64).reshape(seq_len, n, -1, 2))
        x_i_rope = x_i[:, :, :c_rope] * freqs[:seq_len, None, :]
        x_i_passthrough = x_i[:, :, c_rope:]
        x_i = torch.cat([x_i_rope, x_i_passthrough], dim=2)
        x_i = torch.view_as_real(x_i).flatten(2)
        x_i = torch.cat([x_i, x[i, seq_len:]])
        output.append(x_i)
    return torch.stack(output).bfloat16()


@amp.autocast("cuda", enabled=False)
def _rope_apply_3d(x, grid_sizes, freqs):
    n, c = x.size(2), x.size(3) // 2

    # split freqs
    freqs = freqs.split([c - 2 * (c // 3), c // 3, c // 3], dim=1)

    output = []
    for i, (f, h, w) in enumerate(grid_sizes.tolist()):
        seq_len = f * h * w

        x_i = torch.view_as_complex(x[i, :seq_len].to(torch.float64).reshape(seq_len, n, -1, 2))
        freqs_i = torch.cat(
            [
                freqs[0][:f].view(f, 1, 1, -1).expand(f, h, w, -1),
                freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
                freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1),
            ],
            dim=-1,
        ).reshape(seq_len, 1, -1)

        x_i = torch.view_as_real(x_i * freqs_i).flatten(2)
        x_i = torch.cat([x_i, x[i, seq_len:]])
        output.append(x_i)
    return torch.stack(output).bfloat16()


@amp.autocast("cuda", enabled=False)
def _rope_apply_3d_to_1d(x, grid_sizes, freqs):
    n, c = x.size(2), x.size(3) // 2
    c_rope = freqs.shape[1]  # number of complex dims to rotate
    assert c_rope <= c, "RoPE dimensions cannot exceed half of hidden size"

    output = []
    for i, (f, h, w) in enumerate(grid_sizes.tolist()):
        seq_len = f * h * w

        x_i = torch.view_as_complex(x[i, :seq_len].to(torch.float64).reshape(seq_len, n, -1, 2))
        freqs_i = torch.cat(
            [
                freqs[:f].view(f, 1, 1, -1).expand(f, h, w, -1),
            ],
            dim=-1,
        ).reshape(seq_len, 1, -1)

        x_i_rope = x_i[:, :, :c_rope] * freqs_i
        x_i_passthrough = x_i[:, :, c_rope:]
        x_i = torch.cat([x_i_rope, x_i_passthrough], dim=2)
        x_i = torch.view_as_real(x_i).flatten(2)
        x_i = torch.cat([x_i, x[i, seq_len:]])
        output.append(x_i)
    return torch.stack(output).bfloat16()


@amp.autocast("cuda", enabled=False)
def _rope_apply(x, grid_sizes, freqs, cross_1d_rope=False):
    x_ndim = grid_sizes.shape[-1]
    if x_ndim == 3:
        return _rope_apply_3d(x, grid_sizes, freqs) if not cross_1d_rope else _rope_apply_3d_to_1d(x, grid_sizes, freqs)
    else:
        return _rope_apply_1d(x, grid_sizes, freqs)


class WanDoubleStreamSelfAttention(nn.Module):
    def __init__(self, dim, num_heads, window_size=(-1, -1), qk_norm=True, eps=1e-6, joint_attention=False):
        assert dim % num_heads == 0
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.window_size = window_size
        self.qk_norm = qk_norm
        self.eps = eps
        self.joint_attention = joint_attention

        # layers
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)
        self.norm_q = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()
        self.norm_k = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()
        self.q_audio = nn.Linear(dim, dim)
        self.k_audio = nn.Linear(dim, dim)
        self.v_audio = nn.Linear(dim, dim)
        self.o_audio = nn.Linear(dim, dim)
        self.norm_q_audio = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()
        self.norm_k_audio = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()

    # query, key, value function
    def _qkv_fn(self, x):
        b, s, n, d = *x.shape[:2], self.num_heads, self.head_dim

        q = self.norm_q(self.q(x)).view(b, s, n, d)
        k = self.norm_k(self.k(x)).view(b, s, n, d)
        v = self.v(x).view(b, s, n, d)
        return q, k, v

    def _qkv_fn_audio(self, x):
        b, s, n, d = *x.shape[:2], self.num_heads, self.head_dim

        q = self.norm_q_audio(self.q_audio(x)).view(b, s, n, d)
        k = self.norm_k_audio(self.k_audio(x)).view(b, s, n, d)
        v = self.v_audio(x).view(b, s, n, d)
        return q, k, v

    def _single_forward(self, x, seq_lens, grid_sizes, freqs, is_audio=False):
        r"""
        Args:
            x(Tensor): Shape [B, L, C]
            seq_lens(Tensor): Shape [B]
            grid_sizes(Tensor): Shape [B, 3], the second dimension contains (F, H, W)
            freqs(Tensor): Rope freqs, shape [1024, C / num_heads / 2]
            is_audio(bool): Whether to use the audio attention projections.
        """
        q, k, v = self._qkv_fn(x) if not is_audio else self._qkv_fn_audio(x)

        x = _nava_attention(
            q=_rope_apply(q, grid_sizes, freqs),
            k=_rope_apply(k, grid_sizes, freqs),
            v=v,
            k_lens=seq_lens,
            window_size=self.window_size,
        )
        # output
        x = x.flatten(2)
        x = self.o(x) if not is_audio else self.o_audio(x)
        return x

    def forward(
        self,
        x_vid,
        x_audio,
        seq_lens_vid,
        seq_lens_audio,
        grid_sizes_vid,
        grid_sizes_audio=None,
        freqs_vid=None,
        freqs_audio=None,
        max_seq_len_vid=None,
        max_seq_len_audio=None,
        use_joint_attention=True,
    ):
        r"""
        Args:
            x_vid(Tensor): Video hidden states with shape [B, L, C].
            x_audio(Tensor): Audio hidden states with shape [B, L, C].
            seq_lens_vid(Tensor): Video sequence lengths with shape [B].
            seq_lens_audio(Tensor): Audio sequence lengths with shape [B].
            grid_sizes_vid(Tensor): Video grid sizes with shape [B, 3], containing (F, H, W).
            grid_sizes_audio(Tensor): Audio grid sizes.
            freqs_vid(Tensor): Video RoPE frequencies.
            freqs_audio(Tensor): Audio RoPE frequencies.
            max_seq_len_vid(int): Maximum video sequence length.
            max_seq_len_audio(int): Maximum audio sequence length.
            use_joint_attention(bool): Whether to run joint audio-video attention.
        """
        if x_vid is not None and x_audio is None:
            return self._single_forward(x_vid, seq_lens_vid, grid_sizes_vid, freqs_vid, is_audio=False), None
        elif x_audio is not None and x_vid is None:
            return None, self._single_forward(x_audio, seq_lens_audio, grid_sizes_audio, freqs_audio, is_audio=True)
        else:
            B = x_vid.shape[0]
            L = x_vid.shape[1] + x_audio.shape[1]
            q_vid, k_vid, v_vid = self._qkv_fn(x_vid)
            q_audio, k_audio, v_audio = self._qkv_fn_audio(x_audio)
            # concat for joint pre-processing
            q = torch.cat([q_vid, q_audio], dim=1)
            k = torch.cat([k_vid, k_audio], dim=1)
            v = torch.cat([v_vid, v_audio], dim=1)

            pos = torch.arange(L).unsqueeze(0).expand(B, L)

            if use_joint_attention:
                # Mark valid video/audio tokens before compacting padded positions.
                is_vid_valid = (pos < max_seq_len_vid) & (pos < seq_lens_vid.unsqueeze(1))
                is_aud_valid = (pos >= max_seq_len_vid) & ((pos - max_seq_len_vid) < seq_lens_audio.unsqueeze(1))

                # Joint validity mask used to move real tokens before padding.
                is_valid = is_vid_valid | is_aud_valid
                sort_keys = (~is_valid).int()
                gather_indices = torch.argsort(sort_keys, dim=1, stable=True).to(x_vid.device)  # shape: [B, L]

                q_rope = _rope_apply_joint(q, grid_sizes_vid, grid_sizes_audio, freqs_vid, freqs_audio, max_seq_len_vid)
                k_rope = _rope_apply_joint(k, grid_sizes_vid, grid_sizes_audio, freqs_vid, freqs_audio, max_seq_len_vid)

                # Expand indices to 4D [B, L, H, D] to match QKV tensors.
                gather_indices_expanded = (
                    gather_indices.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, q_rope.size(2), q_rope.size(3))
                )

                q_shifted = torch.gather(q_rope, dim=1, index=gather_indices_expanded)
                k_shifted = torch.gather(k_rope, dim=1, index=gather_indices_expanded)
                v_shifted = torch.gather(v, dim=1, index=gather_indices_expanded)
                x_shifted = _nava_attention(
                    q=q_shifted,
                    k=k_shifted,
                    v=v_shifted,
                    k_lens=(seq_lens_vid + seq_lens_audio),
                    window_size=self.window_size,
                )
                scatter_indices = torch.argsort(gather_indices, dim=1)
                scatter_indices_expanded = (
                    scatter_indices.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, x_shifted.size(2), x_shifted.size(3))
                )

                # Scatter attention results back to the original padded layout.
                x = torch.gather(x_shifted, dim=1, index=scatter_indices_expanded)
            else:
                x_vid = _nava_attention(
                    q=_rope_apply(q_vid, grid_sizes_vid, freqs_vid),
                    k=_rope_apply(k_vid, grid_sizes_vid, freqs_vid),
                    v=v_vid,
                    k_lens=seq_lens_vid,
                    window_size=self.window_size,
                )
                x_audio = _nava_attention(
                    q=_rope_apply(q_audio, grid_sizes_audio, freqs_audio),
                    k=_rope_apply(k_audio, grid_sizes_audio, freqs_audio),
                    v=v_audio,
                    k_lens=seq_lens_audio,
                    window_size=self.window_size,
                )
                x = torch.cat([x_vid, x_audio], dim=1)
            # output
            x = x.flatten(2)
            x_vid = self.o(x[:, :max_seq_len_vid, :])
            x_audio = self.o_audio(x[:, max_seq_len_vid:, :])
            return x_vid, x_audio


class WanSelfAttention(nn.Module):
    def __init__(self, dim, num_heads, window_size=(-1, -1), qk_norm=True, eps=1e-6):
        assert dim % num_heads == 0
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.window_size = window_size
        self.qk_norm = qk_norm
        self.eps = eps

        # layers
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)
        self.norm_q = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()
        self.norm_k = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()

    # query, key, value function
    def _qkv_fn(self, x):
        b, s, n, d = *x.shape[:2], self.num_heads, self.head_dim

        q = self.norm_q(self.q(x)).view(b, s, n, d)
        k = self.norm_k(self.k(x)).view(b, s, n, d)
        v = self.v(x).view(b, s, n, d)
        return q, k, v

    def _single_forward(self, x, seq_lens, grid_sizes, freqs):
        r"""
        Args:
            x(Tensor): Shape [B, L, C]
            seq_lens(Tensor): Shape [B]
            grid_sizes(Tensor): Shape [B, 3], the second dimension contains (F, H, W)
            freqs(Tensor): Rope freqs, shape [1024, C / num_heads / 2]
        """
        q, k, v = self._qkv_fn(x)
        x = _nava_attention(
            q=_rope_apply(q, grid_sizes, freqs),
            k=_rope_apply(k, grid_sizes, freqs),
            v=v,
            k_lens=seq_lens,
            window_size=self.window_size,
        )
        # output
        x = x.flatten(2)
        x = self.o(x)
        return x

    def forward(
        self,
        x,
        seq_lens_vid,
        seq_lens_audio,
        grid_sizes_vid,
        grid_sizes_audio=None,
        freqs_vid=None,
        freqs_audio=None,
        max_seq_len_vid=None,
        max_seq_len_audio=None,
        use_joint_attention=True,
    ):
        r"""
        Args:
            x(Tensor): Shape [B, L, C]
            seq_lens_vid(Tensor): Video sequence lengths with shape [B].
            seq_lens_audio(Tensor): Audio sequence lengths with shape [B].
            grid_sizes_vid(Tensor): Video grid sizes with shape [B, 3], containing (F, H, W).
            grid_sizes_audio(Tensor): Audio grid sizes.
            freqs_vid(Tensor): Video RoPE frequencies.
            freqs_audio(Tensor): Audio RoPE frequencies.
            max_seq_len_vid(int): Maximum video sequence length.
            max_seq_len_audio(int): Maximum audio sequence length.
            use_joint_attention(bool): Whether to run joint audio-video attention.
        """
        if max_seq_len_vid > 0 and max_seq_len_audio == 0:
            return self._single_forward(x, seq_lens_vid, grid_sizes_vid, freqs_vid)
        elif max_seq_len_vid == 0 and max_seq_len_audio > 0:
            return self._single_forward(x, seq_lens_audio, grid_sizes_audio, freqs_audio)
        else:
            B, L = x.shape[0], x.shape[1]
            pos = torch.arange(L).unsqueeze(0).expand(B, L)
            q, k, v = self._qkv_fn(x)
            if use_joint_attention:
                is_vid_valid = (pos < max_seq_len_vid) & (pos < seq_lens_vid.unsqueeze(1))
                is_aud_valid = (pos >= max_seq_len_vid) & ((pos - max_seq_len_vid) < seq_lens_audio.unsqueeze(1))

                # Joint validity mask used to move real tokens before padding.
                is_valid = is_vid_valid | is_aud_valid
                sort_keys = (~is_valid).int()
                gather_indices = torch.argsort(sort_keys, dim=1, stable=True).to(x.device)  # shape: [B, L]

                q_rope = _rope_apply_joint(q, grid_sizes_vid, grid_sizes_audio, freqs_vid, freqs_audio, max_seq_len_vid)
                k_rope = _rope_apply_joint(k, grid_sizes_vid, grid_sizes_audio, freqs_vid, freqs_audio, max_seq_len_vid)

                # Expand indices to 4D [B, L, H, D] to match QKV tensors.
                gather_indices_expanded = (
                    gather_indices.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, q_rope.size(2), q_rope.size(3))
                )

                q_shifted = torch.gather(q_rope, dim=1, index=gather_indices_expanded)
                k_shifted = torch.gather(k_rope, dim=1, index=gather_indices_expanded)
                v_shifted = torch.gather(v, dim=1, index=gather_indices_expanded)

                x_shifted = _nava_attention(
                    q=q_shifted,
                    k=k_shifted,
                    v=v_shifted,
                    k_lens=(seq_lens_vid + seq_lens_audio),
                    window_size=self.window_size,
                )
                scatter_indices = torch.argsort(gather_indices, dim=1)
                scatter_indices_expanded = (
                    scatter_indices.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, x_shifted.size(2), x_shifted.size(3))
                )

                # Scatter attention results back to the original padded layout.
                x = torch.gather(x_shifted, dim=1, index=scatter_indices_expanded)
            else:
                q_vid, k_vid, v_vid = q[:, :max_seq_len_vid, :], k[:, :max_seq_len_vid, :], v[:, :max_seq_len_vid, :]
                q_audio, k_audio, v_audio = (
                    q[:, max_seq_len_vid:, :],
                    k[:, max_seq_len_vid:, :],
                    v[:, max_seq_len_vid:, :],
                )
                x_vid = _nava_attention(
                    q=_rope_apply(q_vid, grid_sizes_vid, freqs_vid),
                    k=_rope_apply(k_vid, grid_sizes_vid, freqs_vid),
                    v=v_vid,
                    k_lens=seq_lens_vid,
                    window_size=self.window_size,
                )
                x_audio = _nava_attention(
                    q=_rope_apply(q_audio, grid_sizes_audio, freqs_audio),
                    k=_rope_apply(k_audio, grid_sizes_audio, freqs_audio),
                    v=v_audio,
                    k_lens=seq_lens_audio,
                    window_size=self.window_size,
                )
                x = torch.cat([x_vid, x_audio], dim=1)
            # output
            x = x.flatten(2)
            x = self.o(x)
            return x


class WanT2VCrossAttention(WanSelfAttention):
    def _qkv_fn(self, x, context):
        b, n, d = x.size(0), self.num_heads, self.head_dim

        # compute query, key, value
        q = self.norm_q(self.q(x)).view(b, -1, n, d)
        k = self.norm_k(self.k(context)).view(b, -1, n, d)
        v = self.v(context).view(b, -1, n, d)

        return q, k, v

    def forward(self, x, context, context_lens):
        r"""
        Args:
            x(Tensor): Shape [B, L1, C]
            context(Tensor): Shape [B, L2, C]
            context_lens(Tensor): Shape [B]
        """
        q, k, v = self._qkv_fn(x, context)

        # compute attention
        x = _nava_attention(q, k, v, k_lens=context_lens)

        # output
        x = x.flatten(2)
        x = self.o(x)
        return x


class WanT2VDoubleStreamCrossAttention(WanDoubleStreamSelfAttention):
    def _qkv_fn_audio(self, x, context):
        b, n, d = x.size(0), self.num_heads, self.head_dim

        # compute query, key, value
        q = self.norm_q_audio(self.q_audio(x)).view(b, -1, n, d)
        k = self.norm_k_audio(self.k_audio(context)).view(b, -1, n, d)
        v = self.v_audio(context).view(b, -1, n, d)

        return q, k, v

    def _qkv_fn(self, x, context):
        b, n, d = x.size(0), self.num_heads, self.head_dim

        # compute query, key, value
        q = self.norm_q(self.q(x)).view(b, -1, n, d)
        k = self.norm_k(self.k(context)).view(b, -1, n, d)
        v = self.v(context).view(b, -1, n, d)

        return q, k, v

    def _single_forward(self, x, context, context_lens, is_audio=False):
        r"""
        Args:
            x(Tensor): Shape [B, L1, C]
            context(Tensor): Shape [B, L2, C]
            context_lens(Tensor): Shape [B]
            is_audio(bool): Whether to use the audio attention projections.
        """
        q, k, v = self._qkv_fn(x, context) if not is_audio else self._qkv_fn_audio(x, context)

        # compute attention
        x = _nava_attention(q, k, v, k_lens=context_lens)

        # output
        x = x.flatten(2)
        x = self.o(x) if not is_audio else self.o_audio(x)
        return x

    def forward(self, x_vid, x_audio, context, context_lens, vid_seq_len=None):
        r"""
        Args:
            x_vid(Tensor): Video hidden states with shape [B, L1, C].
            x_audio(Tensor): Audio hidden states with shape [B, L1, C].
            context(Tensor): Shape [B, L2, C]
            context_lens(Tensor): Shape [B]
            vid_seq_len(int): Video sequence length when splitting concatenated output.
        """
        if x_vid is not None and x_audio is not None:
            q, k, v = self._qkv_fn(x_vid, context)
            q_audio, k_audio, v_audio = self._qkv_fn_audio(x_audio, context)

            # compute attention
            x_vid = _nava_attention(q, k, v, k_lens=context_lens)
            x_audio = _nava_attention(q_audio, k_audio, v_audio, k_lens=context_lens)

            # output
            x_vid = x_vid.flatten(2)
            x_audio = x_audio.flatten(2)
            x_vid = self.o(x_vid)
            x_audio = self.o_audio(x_audio)
            return x_vid, x_audio
        elif x_vid is not None:
            return self._single_forward(x_vid, context, context_lens, is_audio=False), None
        else:
            return None, self._single_forward(x_audio, context, context_lens, is_audio=True)


class ModulationAdd(nn.Module):
    def __init__(self, dim, num):
        super().__init__()
        self.modulation = nn.Parameter(torch.randn(1, num, dim) / dim**0.5)

    def forward(self, e):
        return self.modulation.bfloat16() + e.bfloat16()


class WanDoubleStreamAttentionBlock(nn.Module):
    def __init__(
        self,
        dim,
        ffn_dim,
        num_heads,
        window_size=(-1, -1),
        qk_norm=True,
        cross_attn_norm=False,
        eps=1e-6,
        no_split_norm_ffn=False,
    ):
        """Initialize the dual-stream cross-modal attention block."""
        super().__init__()
        # Basic parameters.
        self.dim = dim  # input dimension
        self.ffn_dim = ffn_dim  # hidden dimension of the FFN
        self.num_heads = num_heads  # number of attention heads
        self.window_size = window_size  # attention window size; -1 means global attention
        self.qk_norm = qk_norm  # whether to normalize Q/K
        self.cross_attn_norm = cross_attn_norm  # whether to normalize before cross-attention
        self.eps = eps  # normalization epsilon
        self.no_split_norm_ffn = no_split_norm_ffn  # whether to share norm/FFN across audio and video

        # Network layers.
        self.norm1 = WanLayerNorm(dim, eps, elementwise_affine=False)  # pre-self-attention normalization
        if not no_split_norm_ffn:
            self.norm1_audio = WanLayerNorm(dim, eps, elementwise_affine=False)  # pre-self-attention normalization
        self.self_attn = WanDoubleStreamSelfAttention(dim, num_heads, window_size, qk_norm, eps)  # self-attention
        self.norm3 = (
            WanLayerNorm(dim, eps, elementwise_affine=True).float() if cross_attn_norm else nn.Identity()
        )  # optional pre-cross-attention normalization
        if not no_split_norm_ffn:
            self.norm3_audio = (
                WanLayerNorm(dim, eps, elementwise_affine=True).float() if cross_attn_norm else nn.Identity()
            )  # optional pre-cross-attention normalization

        self.cross_attn = WanT2VDoubleStreamCrossAttention(
            dim,
            num_heads,
            (-1, -1),
            qk_norm,
            eps,
        )

        self.norm2 = WanLayerNorm(dim, eps, elementwise_affine=False)  # pre-FFN normalization
        if not no_split_norm_ffn:
            self.norm2_audio = WanLayerNorm(dim, eps, elementwise_affine=False)  # pre-FFN normalization
        self.ffn = nn.Sequential(  # feed-forward network
            nn.Linear(dim, ffn_dim), nn.GELU(approximate="tanh"), nn.Linear(ffn_dim, dim)
        )
        if not no_split_norm_ffn:
            self.ffn_audio = nn.Sequential(  # feed-forward network
                nn.Linear(dim, ffn_dim), nn.GELU(approximate="tanh"), nn.Linear(ffn_dim, dim)
            )

        # Modulation parameters.
        self.modulation = ModulationAdd(dim, 6)  # six-channel modulation layer
        self.modulation_audio = ModulationAdd(dim, 6)  # six-channel modulation layer

    def forward(
        self,
        x,
        e_vid,
        e_audio,
        freqs_vid,
        freqs_audio,
        context,
        context_lens,
        seq_lens_vid=None,
        seq_lens_audio=None,
        grid_sizes_vid=None,
        grid_sizes_audio=None,
        max_seq_len_vid=None,
        max_seq_len_audio=None,
        masking_modality=False,
        **kwargs,
    ):
        r"""
        Args:
            x(Tensor): Shape [B, L, C]
            e_vid(Tensor): Video modulation tensor with shape [B, L1, 6, C].
            e_audio(Tensor): Audio modulation tensor with shape [B, L1, 6, C].
            freqs_vid(Tensor): Video RoPE frequencies.
            freqs_audio(Tensor): Audio RoPE frequencies.
            context(Tensor): Text context tensor with shape [B, L2, C].
            context_lens(Tensor): Text context lengths with shape [B].
            seq_lens_vid(Tensor): Video sequence lengths with shape [B].
            seq_lens_audio(Tensor): Audio sequence lengths with shape [B].
            grid_sizes_vid(Tensor): Video grid sizes with shape [B, 3], containing (F, H, W).
            grid_sizes_audio(Tensor): Audio grid sizes.
            max_seq_len_vid(int): Maximum video sequence length.
            max_seq_len_audio(int): Maximum audio sequence length.
            masking_modality(bool): Whether modality masking is enabled.
        """
        # has video input
        x_vid, x_audio = None, None

        if max_seq_len_vid > 0:
            x_vid = x[:, :max_seq_len_vid]
            assert e_vid.dtype == torch.bfloat16
            assert len(e_vid.shape) == 4 and e_vid.size(2) == 6 and e_vid.shape[1] == x_vid.shape[1], (
                f"{e_vid.shape}, {x_vid.shape}"
            )
            with amp.autocast("cuda", dtype=torch.bfloat16):
                e_vid = self.modulation(e_vid).chunk(6, dim=2)
            assert e_vid[0].dtype == torch.bfloat16
        if max_seq_len_audio > 0:
            x_audio = x[:, max_seq_len_vid:]
            assert e_audio.dtype == torch.bfloat16
            assert len(e_audio.shape) == 4 and e_audio.size(2) == 6 and e_audio.shape[1] == x_audio.shape[1], (
                f"{e_audio.shape}, {x_audio.shape}"
            )
            with amp.autocast("cuda", dtype=torch.bfloat16):
                e_audio = self.modulation_audio(e_audio).chunk(6, dim=2)
            assert e_audio[0].dtype == torch.bfloat16

        # joint attention begin
        x_vid_norm, x_audio_norm = None, None
        if x_vid is not None:
            x_vid_norm = self.norm1(x_vid).bfloat16() * (1 + e_vid[1].squeeze(2)) + e_vid[0].squeeze(2)
        if x_audio is not None:
            x_audio_norm = (self.norm1 if self.no_split_norm_ffn else self.norm1_audio)(x_audio).bfloat16() * (
                1 + e_audio[1].squeeze(2)
            ) + e_audio[0].squeeze(2)

        y_vid_attn, y_audio_attn = self.self_attn(
            x_vid_norm,
            x_audio_norm,
            seq_lens_vid,
            seq_lens_audio,
            grid_sizes_vid,
            grid_sizes_audio,
            freqs_vid,
            freqs_audio,
            max_seq_len_vid,
            max_seq_len_audio,
            use_joint_attention=(not masking_modality),
        )
        with amp.autocast("cuda", dtype=torch.bfloat16):
            if x_vid is not None:
                x_vid = x_vid + y_vid_attn * e_vid[2].squeeze(2)
            if x_audio is not None:
                x_audio = x_audio + y_audio_attn * e_audio[2].squeeze(2)

        def _cross_attn_ffn_doublestream(x_vid, x_audio, context, context_lens, e_vid, e_audio):
            x_vid_norm, x_audio_norm = None, None
            if x_vid is not None:
                x_vid_norm = self.norm3(x_vid)
            if x_audio is not None:
                x_audio_norm = (self.norm3 if self.no_split_norm_ffn else self.norm3_audio)(x_audio)
            x_vid_attn, x_audio_attn = self.cross_attn(x_vid_norm, x_audio_norm, context, context_lens, max_seq_len_vid)

            if x_vid is not None:
                x_vid = x_vid + x_vid_attn
                y_vid = self.ffn(self.norm2(x_vid).bfloat16() * (1 + e_vid[4].squeeze(2)) + e_vid[3].squeeze(2))
                with amp.autocast("cuda", dtype=torch.bfloat16):
                    x_vid = x_vid + y_vid * e_vid[5].squeeze(2)
            if x_audio is not None:
                x_audio = x_audio + x_audio_attn
                _norm2 = self.norm2 if self.no_split_norm_ffn else self.norm2_audio
                _ffn = self.ffn if self.no_split_norm_ffn else self.ffn_audio
                y_audio = _ffn(_norm2(x_audio).bfloat16() * (1 + e_audio[4].squeeze(2)) + e_audio[3].squeeze(2))
                with amp.autocast("cuda", dtype=torch.bfloat16):
                    x_audio = x_audio + y_audio * e_audio[5].squeeze(2)
            return x_vid, x_audio

        x_vid, x_audio = _cross_attn_ffn_doublestream(x_vid, x_audio, context, context_lens, e_vid, e_audio)
        if x_vid is not None and x_audio is not None:
            x = torch.cat([x_vid, x_audio], dim=1)
        elif x_vid is not None:
            x = x_vid
        elif x_audio is not None:
            x = x_audio

        return x


class WanAttentionBlock(nn.Module):
    def __init__(
        self,
        dim,
        ffn_dim,
        num_heads,
        window_size=(-1, -1),
        qk_norm=True,
        cross_attn_norm=False,
        eps=1e-6,
        split_av_qk_norm_modulation=False,
    ):
        """Initialize the single-stream cross-modal attention block."""
        super().__init__()
        # Basic parameters.
        self.dim = dim  # input dimension
        self.ffn_dim = ffn_dim  # hidden dimension of the FFN
        self.num_heads = num_heads  # number of attention heads
        self.window_size = window_size  # attention window size; -1 means global attention
        self.qk_norm = qk_norm  # whether to normalize Q/K
        self.cross_attn_norm = cross_attn_norm  # whether to normalize before cross-attention
        self.eps = eps  # normalization epsilon
        self.split_av_qk_norm_modulation = split_av_qk_norm_modulation

        # Network layers.
        self.norm1 = WanLayerNorm(dim, eps, elementwise_affine=False)  # pre-self-attention normalization
        self.self_attn = WanSelfAttention(dim, num_heads, window_size, qk_norm, eps)  # self-attention
        self.norm3 = (
            WanLayerNorm(dim, eps, elementwise_affine=True) if cross_attn_norm else nn.Identity()
        )  # optional pre-cross-attention normalization

        self.cross_attn = WanT2VCrossAttention(
            dim,
            num_heads,
            (-1, -1),
            qk_norm,
            eps,
        )

        self.norm2 = WanLayerNorm(dim, eps, elementwise_affine=False)  # pre-FFN normalization
        self.ffn = nn.Sequential(  # feed-forward network
            nn.Linear(dim, ffn_dim), nn.GELU(approximate="tanh"), nn.Linear(ffn_dim, dim)
        )

        # Modulation parameters.
        self.modulation = ModulationAdd(dim, 6)  # six-channel modulation layer
        if split_av_qk_norm_modulation:
            self.modulation_audio = ModulationAdd(dim, 6)

    def forward(
        self,
        x,
        e_vid,
        e_audio,
        freqs_vid,
        freqs_audio,
        context,
        context_lens,
        seq_lens_vid=None,
        seq_lens_audio=None,
        grid_sizes_vid=None,
        grid_sizes_audio=None,
        max_seq_len_vid=None,
        max_seq_len_audio=None,
        masking_modality=False,
        **kwargs,
    ):
        r"""
        Args:
            x(Tensor): Shape [B, L, C]
            e_vid(Tensor): Video modulation tensor with shape [B, L1, 6, C].
            e_audio(Tensor): Audio modulation tensor with shape [B, L1, 6, C].
            freqs_vid(Tensor): Video RoPE frequencies.
            freqs_audio(Tensor): Audio RoPE frequencies.
            context(Tensor): Text context tensor with shape [B, L2, C].
            context_lens(Tensor): Text context lengths with shape [B].
            seq_lens_vid(Tensor): Video sequence lengths with shape [B].
            seq_lens_audio(Tensor): Audio sequence lengths with shape [B].
            grid_sizes_vid(Tensor): Video grid sizes with shape [B, 3], containing (F, H, W).
            grid_sizes_audio(Tensor): Audio grid sizes.
            max_seq_len_vid(int): Maximum video sequence length.
            max_seq_len_audio(int): Maximum audio sequence length.
            masking_modality(bool): Whether modality masking is enabled.
        """
        if not self.split_av_qk_norm_modulation:
            if max_seq_len_vid > 0 and max_seq_len_audio > 0:
                e = torch.cat([e_vid, e_audio], dim=1)
            elif max_seq_len_vid > 0:
                e = e_vid
            elif max_seq_len_audio > 0:
                e = e_audio

            assert e.dtype == torch.bfloat16
            assert len(e.shape) == 4 and e.size(2) == 6 and e.shape[1] == x.shape[1], f"{e.shape}, {x.shape}"
            with amp.autocast("cuda", dtype=torch.bfloat16):
                e = self.modulation(e).chunk(6, dim=2)
            assert e[0].dtype == torch.bfloat16
        else:
            if max_seq_len_vid > 0:
                x_vid = x[:, :max_seq_len_vid]
                assert e_vid.dtype == torch.bfloat16
                assert len(e_vid.shape) == 4 and e_vid.size(2) == 6 and e_vid.shape[1] == x_vid.shape[1], (
                    f"{e_vid.shape}, {x_vid.shape}"
                )
                with amp.autocast("cuda", dtype=torch.bfloat16):
                    e_vid = self.modulation(e_vid).chunk(6, dim=2)
                assert e_vid[0].dtype == torch.bfloat16
            if max_seq_len_audio > 0:
                x_audio = x[:, max_seq_len_vid:]
                assert e_audio.dtype == torch.bfloat16
                assert len(e_audio.shape) == 4 and e_audio.size(2) == 6 and e_audio.shape[1] == x_audio.shape[1], (
                    f"{e_audio.shape}, {x_audio.shape}"
                )
                with amp.autocast("cuda", dtype=torch.bfloat16):
                    e_audio = self.modulation_audio(e_audio).chunk(6, dim=2)
                assert e_audio[0].dtype == torch.bfloat16

            if max_seq_len_vid > 0 and max_seq_len_audio > 0:
                # e = tuple(torch.cat([e_v, e_a] for e_v, e_a in zip(e_vid, e_audio)))
                e = tuple(torch.cat([e_v, e_a], dim=1) for e_v, e_a in zip(e_vid, e_audio))
            elif max_seq_len_vid > 0:
                e = e_vid
            elif max_seq_len_audio > 0:
                e = e_audio

        # self-attention
        y = self.self_attn(
            self.norm1(x).bfloat16() * (1 + e[1].squeeze(2)) + e[0].squeeze(2),
            seq_lens_vid,
            seq_lens_audio,
            grid_sizes_vid,
            grid_sizes_audio,
            freqs_vid,
            freqs_audio,
            max_seq_len_vid,
            max_seq_len_audio,
            use_joint_attention=(not masking_modality),
        )
        with amp.autocast("cuda", dtype=torch.bfloat16):
            x = x + y * e[2].squeeze(2)

        # cross-attention & ffn function
        def _cross_attn_ffn(x, context, context_lens, e):
            x = x + self.cross_attn(self.norm3(x), context, context_lens)
            y = self.ffn(self.norm2(x).bfloat16() * (1 + e[4].squeeze(2)) + e[3].squeeze(2))
            with amp.autocast("cuda", dtype=torch.bfloat16):
                x = x + y * e[5].squeeze(2)
            return x

        x = _cross_attn_ffn(x, context, context_lens, e)
        return x


class Head(nn.Module):
    def __init__(self, dim, out_dim, patch_size, eps=1e-6):
        super().__init__()
        self.dim = dim
        self.out_dim = out_dim
        self.patch_size = patch_size
        self.eps = eps

        # layers
        out_dim = math.prod(patch_size) * out_dim
        self.norm = WanLayerNorm(dim, eps, elementwise_affine=False)
        self.head = nn.Linear(dim, out_dim)

        # modulation
        self.modulation = nn.Parameter(torch.randn(1, 2, dim) / dim**0.5)

    def forward(self, x, e):
        r"""
        Args:
            x(Tensor): Shape [B, L1, C]
            e(Tensor): Shape [B, L, C]
        """
        assert e.dtype == torch.bfloat16
        with amp.autocast("cuda", dtype=torch.bfloat16):
            e = (self.modulation.bfloat16().unsqueeze(0) + e.unsqueeze(2)).chunk(
                2, dim=2
            )  # 1 1 2 D, B L 1 D -> B L 2 D -> 2 * (B L 1 D)
            x = self.head(self.norm(x) * (1 + e[1].squeeze(2)) + e[0].squeeze(2))
        return x


class SpkToken(nn.Module):
    def __init__(self, spk_dim=192, dim=1024, eps=1e-6):
        super().__init__()
        self.spk_dim = spk_dim
        self.eps = eps
        self.net = nn.Sequential(
            nn.LayerNorm(spk_dim),
            nn.Linear(spk_dim, dim),
            nn.SiLU(),
            nn.Linear(dim, dim),
        )
        self.out_norm = nn.LayerNorm(dim)
        # learnable global speaker embedding
        self.null_token = nn.Parameter(torch.zeros(1, dim))

    def forward(self, spk_emb):  # spk_emb: [B, 192], fake spk_emb contains all zeros
        assert spk_emb.shape[-1] == self.spk_dim, f"{spk_emb.shape}"
        B = spk_emb.shape[0]
        fake_pos = spk_emb.float().pow(2).sum(dim=-1) <= self.eps  # [B] bool
        spk_embeds = self.out_norm(self.net(spk_emb))  # [B,dim]
        null = self.null_token.expand(B, -1)  # [B,dim]
        m = (~fake_pos).to(spk_embeds.dtype).view(B, 1)  # valid speaker=1, null speaker=0
        spk_embeds = null * (1 - m) + spk_embeds * m
        return spk_embeds


class WanAVModel(nn.Module):
    r"""
    NAVA ti2v diffusion backbone for joint audio-video generation.
    """

    ignore_for_config = ["patch_size", "cross_attn_norm", "qk_norm", "text_dim", "window_size"]
    _no_split_modules = ["WanAttentionBlock"]

    def __init__(
        self,
        patch_size=(1, 2, 2),
        text_len=512,
        vid_in_dim=16,
        audio_in_dim=16,
        dim=2048,
        ffn_dim=8192,
        freq_dim=256,
        text_dim=4096,
        vid_out_dim=16,
        audio_out_dim=16,
        num_heads=16,
        num_layers=32,
        num_double_layers=8,
        num_single_layers=24,
        num_double_final_layers=0,
        window_size=(-1, -1),
        qk_norm=True,
        cross_attn_norm=True,
        temporal_rope_scaling_factor=1.0,
        eps=1e-6,
        add_spk_emb=False,
        cross_1d_rope=False,
        no_split_norm_ffn=False,
    ):
        r"""
        Initialize the diffusion model backbone.

        Args:
            patch_size (`tuple`, *optional*, defaults to (1, 2, 2)):
                3D patch dimensions for video embedding (t_patch, h_patch, w_patch)
            text_len (`int`, *optional*, defaults to 512):
                Fixed length for text embeddings
            vid_in_dim (`int`, *optional*, defaults to 16):
                Input video channels (C_in)
            audio_in_dim (`int`, *optional*, defaults to 16):
                Input audio channels (C_in)
            dim (`int`, *optional*, defaults to 2048):
                Hidden dimension of the transformer
            ffn_dim (`int`, *optional*, defaults to 8192):
                Intermediate dimension in feed-forward network
            freq_dim (`int`, *optional*, defaults to 256):
                Dimension for sinusoidal time embeddings
            text_dim (`int`, *optional*, defaults to 4096):
                Input dimension for text embeddings
            vid_out_dim (`int`, *optional*, defaults to 16):
                Output video channels (C_out)
            audio_out_dim (`int`, *optional*, defaults to 16):
                Output audio channels (C_out)
            num_heads (`int`, *optional*, defaults to 16):
                Number of attention heads
            num_layers (`int`, *optional*, defaults to 32):
                Number of transformer blocks
            window_size (`tuple`, *optional*, defaults to (-1, -1)):
                Window size for local attention (-1 indicates global attention)
            qk_norm (`bool`, *optional*, defaults to True):
                Enable query/key normalization
            cross_attn_norm (`bool`, *optional*, defaults to False):
                Enable cross-attention normalization
            eps (`float`, *optional*, defaults to 1e-6):
                Epsilon value for normalization layers
        """

        super().__init__()

        self.patch_size = patch_size
        self.text_len = text_len
        self.vid_in_dim = vid_in_dim
        self.audio_in_dim = audio_in_dim
        self.dim = dim
        self.ffn_dim = ffn_dim
        self.freq_dim = freq_dim
        self.text_dim = text_dim
        self.vid_out_dim = vid_out_dim
        self.audio_out_dim = audio_out_dim
        self.num_heads = num_heads
        # self.num_layers = num_layers
        assert num_double_layers + num_single_layers + num_double_final_layers == num_layers, (
            num_double_layers,
            num_single_layers,
            num_double_final_layers,
            num_layers,
        )
        self.num_double_layers = num_double_layers
        self.num_single_layers = num_single_layers
        self.num_double_final_layers = num_double_final_layers
        self.window_size = window_size
        self.qk_norm = qk_norm
        self.cross_attn_norm = cross_attn_norm
        self.eps = eps
        self.temporal_rope_scaling_factor = temporal_rope_scaling_factor
        logger.debug("NAVA temporal RoPE scaling factor: %s", temporal_rope_scaling_factor)
        self.add_spk_emb = add_spk_emb
        self.cross_1d_rope = cross_1d_rope

        self.patch_embedding = nn.Conv3d(vid_in_dim, dim, kernel_size=patch_size, stride=patch_size)
        self.patch_embedding_audio = nn.Sequential(
            ChannelLastConv1d(audio_in_dim, dim, kernel_size=7, padding=3),
            nn.SiLU(),
            ConvMLP(dim, dim * 4, kernel_size=7, padding=3),
        )
        if add_spk_emb:
            self.speaker_embedding = SpkToken(spk_dim=192, dim=dim)

        self.text_embedding = nn.Sequential(nn.Linear(text_dim, dim), nn.GELU(approximate="tanh"), nn.Linear(dim, dim))

        self.time_embedding = nn.Sequential(nn.Linear(freq_dim, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.time_projection = nn.Sequential(nn.SiLU(), nn.Linear(dim, dim * 6))

        self.double_blocks = nn.ModuleList(
            [
                WanDoubleStreamAttentionBlock(
                    dim,
                    ffn_dim,
                    num_heads,
                    window_size,
                    qk_norm,
                    cross_attn_norm,
                    eps,
                    no_split_norm_ffn=no_split_norm_ffn,
                )
                for _ in range(num_double_layers)
            ]
        )

        self.single_blocks = nn.ModuleList(
            [
                WanAttentionBlock(
                    dim,
                    ffn_dim,
                    num_heads,
                    window_size,
                    qk_norm,
                    cross_attn_norm,
                    eps,
                )
                for _ in range(num_single_layers)
            ]
        )

        self.double_final_blocks = nn.ModuleList(
            [
                WanDoubleStreamAttentionBlock(
                    dim,
                    ffn_dim,
                    num_heads,
                    window_size,
                    qk_norm,
                    cross_attn_norm,
                    eps,
                    no_split_norm_ffn=no_split_norm_ffn,
                )
                for _ in range(num_double_final_layers)
            ]
        )

        # head
        self.head = Head(dim, vid_out_dim, patch_size, eps)
        self.head_audio = Head(dim, audio_out_dim, patch_size=[1], eps=eps)

        self._set_rope_params()

        # initialize weights
        self._init_weights()

    def _merge_kwargs(self, vid_kwargs, audio_kwargs):
        """
        keys in each kwarg:
        e
        seq_lens
        grid_sizes
        freqs
        context
        context_lens
        """
        if vid_kwargs is None:
            vid_kwargs = dict(
                e=None,
                seq_lens=0,
                max_seq_len=0,
                grid_sizes=None,
                freqs=self.freqs,
                context=None,
                context_lens=None,
            )
        if audio_kwargs is None:
            audio_kwargs = dict(
                e=None,
                seq_lens=0,
                max_seq_len=0,
                grid_sizes=None,
                freqs=self.freqs_audio,
                context=None,
                context_lens=None,
            )
        merged_kwargs = {}
        for key in vid_kwargs:
            merged_kwargs[f"{key}_vid"] = vid_kwargs[key]
        for key in audio_kwargs:
            merged_kwargs[f"{key}_audio"] = audio_kwargs[key]
        return merged_kwargs

    def _set_rope_params(self):
        # buffers (don't use register_buffer otherwise dtype will be changed in to())
        dim = self.dim
        num_heads = self.num_heads
        assert (dim % num_heads) == 0 and (dim // num_heads) % 2 == 0
        d = dim // num_heads

        ## to be determined
        # self.freqs = _rope_params(1024, d, freqs_scaling=temporal_rope_scaling_factor)
        self.freqs_audio = _rope_params(1024, d - 4 * (d // 6), freqs_scaling=self.temporal_rope_scaling_factor)

        self.freqs = torch.cat(
            [_rope_params(1024, d - 4 * (d // 6)), _rope_params(1024, 2 * (d // 6)), _rope_params(1024, 2 * (d // 6))],
            dim=1,
        )

    def _prepare_transformer_block_kwargs(
        self,
        x,
        t,
        context,
        seq_len,
        first_frame_is_clean=False,
        spk_embed=None,
        spk_pos=None,
        is_audio_type=False,
    ):
        # params
        ## need to change!
        device = next(self.patch_embedding.parameters()).device

        if self.freqs.device != device:
            self.freqs = self.freqs.to(device)
            self.freqs_audio = self.freqs_audio.to(device)

        # embeddings
        if not is_audio_type:
            x = [self.patch_embedding(u.unsqueeze(0)) for u in x]  ## x is list of [B L D] or [B C F H W]
        else:
            x = [self.patch_embedding_audio(u.unsqueeze(0)) for u in x]  ## x is list of [B L D] or [B C F H W]
        if is_audio_type:
            # [B, 1]
            grid_sizes = torch.stack([torch.tensor(u.shape[1:2], dtype=torch.long) for u in x])
        else:
            # [B, 3]
            grid_sizes = torch.stack([torch.tensor(u.shape[2:], dtype=torch.long) for u in x])
            x = [u.flatten(2).transpose(1, 2) for u in x]  # [B C F H W] -> [B (F H W) C] -> [B L C]

        seq_lens = torch.tensor([u.size(1) for u in x], dtype=torch.long)
        assert seq_lens.max() <= seq_len, f"Sequence length {seq_lens.max()} exceeds maximum {seq_len}."
        x = torch.cat(
            [torch.cat([u, u.new_zeros(1, seq_len - u.size(1), u.size(2))], dim=1) for u in x]
        )  # single [B, L, C]

        # time embeddings
        if t.dim() == 1:
            if first_frame_is_clean:
                t = torch.ones((t.size(0), seq_len), device=t.device, dtype=t.dtype) * t.unsqueeze(1)
                _first_images_seq_len = grid_sizes[:, 1:].prod(-1)
                for i in range(t.size(0)):
                    t[i, : _first_images_seq_len[i]] = 0
            else:
                t = t.unsqueeze(1).expand(t.size(0), seq_len)
        with amp.autocast("cuda", dtype=torch.bfloat16):
            bt = t.size(0)
            t = t.flatten()
            e = self.time_embedding(_sinusoidal_embedding_1d(self.freq_dim, t).unflatten(0, (bt, seq_len)).bfloat16())
            e0 = self.time_projection(e).unflatten(2, (6, self.dim))  # [1, 26784, 6, 3072] - B, seq_len, 6, dim
            assert e.dtype == torch.bfloat16 and e0.dtype == torch.bfloat16

        # context
        context_lens = None
        context = self.text_embedding(
            torch.stack([torch.cat([u, u.new_zeros(self.text_len - u.size(0), u.size(1))]) for u in context])
        )

        if self.add_spk_emb and spk_embed is not None:
            spk_embeds = self.speaker_embedding(spk_embed)  # [total_spk, dim]
            B, L, D = context.shape

            if spk_pos is not None:
                indices = [b * L + pos for b, pos_list in enumerate(spk_pos) for pos in pos_list]
                if indices:  # ensure at least one speaker token exists
                    indices = torch.tensor(indices, device=context.device)
                    if spk_embeds.shape[0] != len(indices):
                        logger.warning(
                            "NAVA speaker positions do not match embeddings: %s vs %s", spk_embeds.shape, indices
                        )
                        context.view(-1, D)[indices] = spk_embeds[: len(indices)].to(context.dtype)
                    else:
                        context.view(-1, D)[indices] = spk_embeds.to(context.dtype)
        # arguments
        kwargs = dict(
            e=e0,
            seq_lens=seq_lens,
            max_seq_len=seq_len,
            grid_sizes=grid_sizes,
            freqs=self.freqs if not is_audio_type else self.freqs_audio,
            context=context,
            context_lens=context_lens,
        )

        return x, e, kwargs

    def _post_transformer_block_out_doublestream(self, x_vid, x_audio, grid_sizes, grid_sizes_audio, e_vid, e_audio):
        # head
        return self._post_transformer_block_out(
            x_vid, grid_sizes, e_vid, is_audio=False
        ), self._post_transformer_block_out(x_audio, grid_sizes_audio, e_audio, is_audio=True)

    def _post_transformer_block_out(self, x, grid_sizes, e, is_audio=False):
        # head
        if x is None:
            return None
        if not is_audio:
            x = self.head(x, e)
        else:
            x = self.head_audio(x, e)
        # unpatchify
        if is_audio:
            ## grid_sizes is [B 1] where 1 is L,
            # converting grid_sizes from [B 1] -> [B]
            grid_sizes = [gs[0] for gs in grid_sizes]
            assert len(x) == len(grid_sizes)
            x = [u[:gs] for u, gs in zip(x, grid_sizes)]
        else:
            ## grid_sizes is [B 3] where 3 is F H w
            x = self._unpatchify(x, grid_sizes)

        return [u.bfloat16() for u in x]

    def forward(
        self,
        vid,
        audio,
        t,
        vid_context,
        audio_context,
        vid_seq_len,
        audio_seq_len,
        spk_embed=None,
        spk_pos=None,
        masking_modality=False,
        first_frame_is_clean=False,
        **kwargs,
    ):
        r"""
        Forward pass through the diffusion model

        Args:
            vid (List[Tensor]):
                List of input video tensors, each with shape [C_in, F, H, W].
            audio (List[Tensor]):
                List of input audio tensors, each with shape [L, C_in].
            t (Tensor):
                Diffusion timesteps tensor of shape [B]
            vid_context (List[Tensor]):
                List of video text embeddings, each with shape [L, C].
            audio_context (List[Tensor]):
                List of audio text embeddings, each with shape [L, C].
            vid_seq_len (int):
                Maximum video sequence length for positional encoding.
            audio_seq_len (int):
                Maximum audio sequence length for positional encoding.
            spk_embed (Tensor):
                Optional speaker embedding.
            spk_pos (Tensor):
                Optional speaker embedding position.
            masking_modality (bool):
                Whether modality masking is enabled.
            first_frame_is_clean (bool):
                Whether first-frame conditioning should stay clean.
        Returns:
            List[Tensor]:
                List of denoised video tensors with original input shapes [C_out, F, H / 8, W / 8]
                OR
                List of denoised audio tensors with original input shapes [L, C_in]
        """
        x_vid, e_vid, kwargs_vid = None, None, None
        x_audio, e_audio, kwargs_audio = None, None, None
        if vid is not None:
            x_vid, e_vid, kwargs_vid = self._prepare_transformer_block_kwargs(
                x=vid,
                t=t,
                context=vid_context,
                seq_len=vid_seq_len,
                first_frame_is_clean=first_frame_is_clean,
            )
        if audio is not None:
            x_audio, e_audio, kwargs_audio = self._prepare_transformer_block_kwargs(
                x=audio,
                t=t,
                context=audio_context,
                seq_len=audio_seq_len,
                first_frame_is_clean=False,
                spk_embed=spk_embed,
                spk_pos=spk_pos,
                is_audio_type=True,
            )
        kwargs = self._merge_kwargs(kwargs_vid, kwargs_audio)
        # kwargs["context"] = kwargs["context_vid"]
        # kwargs["context_lens"] = kwargs["context_lens_vid"]
        kwargs["context"] = (
            kwargs["context_vid"]
            if (kwargs["context_vid"] is not None and spk_embed is None)
            else kwargs["context_audio"]
        )
        kwargs["context_lens"] = (
            kwargs["context_lens_vid"] if kwargs["context_lens_vid"] is not None else kwargs["context_lens_audio"]
        )
        kwargs["masking_modality"] = masking_modality

        if x_vid is not None and x_audio is not None:
            x = torch.cat([x_vid, x_audio], dim=1)
        elif x_vid is not None:
            x = x_vid
        elif x_audio is not None:
            x = x_audio

        for block in self.double_blocks:
            x = block(x=x, **kwargs)

        for block in self.single_blocks:
            x = block(x=x, **kwargs)
        for block in self.double_final_blocks:
            x = block(x=x, **kwargs)
        if vid is not None:
            x_vid = x[:, : kwargs["max_seq_len_vid"]]
        if audio is not None:
            x_audio = x[:, kwargs["max_seq_len_vid"] :]

        return self._post_transformer_block_out_doublestream(
            x_vid, x_audio, kwargs["grid_sizes_vid"], kwargs["grid_sizes_audio"], e_vid, e_audio
        )

    def _unpatchify(self, x, grid_sizes, is_audio=False):
        r"""
        Reconstruct video tensors from patch embeddings.

        Args:
            x (List[Tensor]):
                List of patchified features, each with shape [L, C_out * prod(patch_size)]
            grid_sizes (Tensor):
                Original spatial-temporal grid dimensions before patching,
                    shape [B, 3] (3 dimensions correspond to F_patches, H_patches, W_patches)

        Returns:
            List[Tensor]:
                Reconstructed video tensors with shape [C_out, F, H / 8, W / 8]
        """

        c = self.vid_out_dim if not is_audio else self.audio_out_dim
        patch_size = self.patch_size if not is_audio else [1]
        out = []
        for u, v in zip(x, grid_sizes.tolist()):
            # v is [F H w] F * H * 80, 100, it was right padded by 20.
            u = u[: math.prod(v)].view(*v, *patch_size, c)
            u = torch.einsum("fhwpqrc->cfphqwr", u)
            u = u.reshape(c, *[i * j for i, j in zip(v, self.patch_size)])
            out.append(u)
        # out is list of [C F H W]
        return out

    def _init_weights(self):
        r"""
        Initialize model parameters using Xavier initialization.
        """

        # basic init
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        # init embeddings
        assert isinstance(self.patch_embedding, nn.Conv3d), (
            f"Patch embedding for video should be a Conv3d layer, got {type(self.patch_embedding)}"
        )
        nn.init.xavier_uniform_(self.patch_embedding.weight.flatten(1))
        for m in self.text_embedding.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)
        for m in self.time_embedding.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)

        # init output layer
        nn.init.zeros_(self.head.head.weight)


class NAVATransformer(nn.Module):
    _repeated_blocks = ["backbone.double_blocks", "backbone.single_blocks", "backbone.double_final_blocks"]
    _layerwise_offload_blocks_attr = "backbone.double_blocks"

    def __init__(self, config: NAVAConfig) -> None:
        super().__init__()
        self.config = config
        self.patch_size = 2
        self.backbone = WanAVModel(
            patch_size=config.patch_size,
            text_len=config.text_len,
            vid_in_dim=config.video_latent_ch,
            audio_in_dim=config.audio_latent_ch,
            vid_out_dim=config.video_latent_ch,
            audio_out_dim=config.audio_latent_ch,
            dim=config.hidden_size,
            ffn_dim=config.ffn_dim,
            freq_dim=config.freq_dim,
            text_dim=config.text_embed_dim,
            num_heads=config.num_heads,
            num_layers=config.num_layers,
            num_double_layers=config.num_double_layers,
            num_single_layers=config.num_single_layers,
            temporal_rope_scaling_factor=0.24,
            add_spk_emb=True,
            no_split_norm_ffn=True,
        )
        self.audio_latent_ch = config.audio_latent_ch
        self.video_latent_ch = config.video_latent_ch

    def forward(
        self,
        *,
        video_latents: torch.Tensor,
        audio_latents: torch.Tensor,
        timestep: torch.Tensor,
        text_embeds: torch.Tensor,
        audio_text_embeds: torch.Tensor | None = None,
        image_embeds: torch.Tensor | None = None,
        speaker_embeds: torch.Tensor | None = None,
        speaker_positions: list[list[int]] | None = None,
        video_grid: tuple[int, int, int] | None = None,
        masking_modality: bool = False,
        **_: object,
    ) -> dict[str, torch.Tensor]:
        video_was_batched = video_latents.ndim == 3
        audio_was_batched = audio_latents.ndim == 3
        batch_size = video_latents.shape[0] if video_was_batched else 1
        if batch_size != 1:
            raise ValueError("NAVATransformer currently expects one request per forward call.")
        if video_grid is None:
            latent_h, latent_w = self.config.video_latent_hw()
            latent_frames = video_latents.shape[-2] // (latent_h * latent_w)
        else:
            latent_frames, latent_h, latent_w = [int(value) for value in video_grid]
        expected_video_tokens = latent_frames * latent_h * latent_w
        video_token_count = video_latents.shape[-2]
        if video_token_count != expected_video_tokens:
            raise ValueError(
                "NAVA video latent token count does not match video_grid: "
                f"expected {expected_video_tokens}, got {video_token_count}."
            )
        t_h_w_list = torch.tensor(
            [(latent_frames, latent_h, latent_w)],
            dtype=torch.long,
            device=video_latents.device,
        )
        audio_len_list = torch.tensor([[audio_latents.shape[-2]]], dtype=torch.long, device=audio_latents.device)
        first_frames = None
        if image_embeds is not None:
            expected_first_frame_tokens = latent_h * latent_w
            if image_embeds.shape[1] != expected_first_frame_tokens:
                raise ValueError(
                    "NAVA image latent token count does not match video_grid first frame: "
                    f"expected {expected_first_frame_tokens}, got {image_embeds.shape[1]}."
                )
            # Image conditioning: replace the first latent video frame with
            # the encoded input image before denoising.
            first_frames = [image_embeds[0].reshape(1, latent_h, latent_w, self.video_latent_ch)]
        latents_vid = video_latents[0] if video_was_batched else video_latents
        latents_audio = audio_latents[0] if audio_was_batched else audio_latents
        video_pred, audio_pred = self.predict_eps(
            vid_context=[text_embeds[0]],
            audio_context=[(audio_text_embeds if audio_text_embeds is not None else text_embeds)[0]],
            latents_vid=latents_vid,
            latents_audio=latents_audio,
            timesteps=timestep.to(video_latents.device),
            spk_embs=speaker_embeds,
            spk_pos=speaker_positions,
            t_h_w_list=t_h_w_list,
            audio_len_list=audio_len_list,
            masking_modality=masking_modality,
            is_i2v=image_embeds is not None,
            first_frames=first_frames,
        )
        video_out = video_pred.unsqueeze(0) if video_was_batched else video_pred
        audio_out = audio_pred.unsqueeze(0) if audio_was_batched else audio_pred
        return {
            "video": video_out,
            "audio": audio_out,
        }

    @torch.no_grad()
    def predict_eps(
        self,
        *,
        vid_context: list[torch.Tensor] | None,
        audio_context: list[torch.Tensor] | None,
        latents_vid: torch.Tensor | None,
        latents_audio: torch.Tensor | None,
        timesteps: torch.Tensor,
        spk_embs: torch.Tensor | None = None,
        spk_pos: list[list[int]] | None = None,
        t_h_w_list: torch.Tensor | None = None,
        audio_len_list: torch.Tensor | None = None,
        masking_modality: bool = False,
        is_i2v: bool = False,
        first_frames: list[torch.Tensor] | None = None,
        **_: object,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        device = None
        if latents_vid is not None:
            device = latents_vid.device
        elif latents_audio is not None:
            device = latents_audio.device
        autocast_enabled = device is not None and device.type == "cuda"
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=autocast_enabled):
            return self._predict_eps_impl(
                vid_context=vid_context,
                audio_context=audio_context,
                latents_vid=latents_vid,
                latents_audio=latents_audio,
                timesteps=timesteps,
                spk_embs=spk_embs,
                spk_pos=spk_pos,
                t_h_w_list=t_h_w_list,
                audio_len_list=audio_len_list,
                masking_modality=masking_modality,
                is_i2v=is_i2v,
                first_frames=first_frames,
            )

    def _predict_eps_impl(
        self,
        *,
        vid_context: list[torch.Tensor] | None,
        audio_context: list[torch.Tensor] | None,
        latents_vid: torch.Tensor | None,
        latents_audio: torch.Tensor | None,
        timesteps: torch.Tensor,
        spk_embs: torch.Tensor | None = None,
        spk_pos: list[list[int]] | None = None,
        t_h_w_list: torch.Tensor | None = None,
        audio_len_list: torch.Tensor | None = None,
        masking_modality: bool = False,
        is_i2v: bool = False,
        first_frames: list[torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        has_video = latents_vid is not None
        has_audio = latents_audio is not None
        batch_size = len(vid_context) if has_video and vid_context is not None else len(audio_context or [])
        max_seq_len_audio = int(audio_len_list.max().item()) if has_audio and audio_len_list is not None else 0
        if has_video:
            assert t_h_w_list is not None
            max_seq_len_video = max(
                int(
                    (
                        int(t)
                        * math.ceil(int(h) / self.patch_size)
                        * self.patch_size
                        * math.ceil(int(w) / self.patch_size)
                        * self.patch_size
                    )
                    // (self.backbone.patch_size[1] * self.backbone.patch_size[2])
                )
                for t, h, w in t_h_w_list
            )
        else:
            max_seq_len_video = 0

        video_inputs: list[torch.Tensor] | None = [] if has_video else None
        audio_inputs: list[torch.Tensor] | None = [] if has_audio else None
        offset_vid = 0
        offset_audio = 0
        for i in range(batch_size):
            if has_video and video_inputs is not None and latents_vid is not None and t_h_w_list is not None:
                t, h, w = [int(x) for x in t_h_w_list[i]]
                valid_len = t * h * w
                z_item = latents_vid[offset_vid : offset_vid + valid_len]
                offset_vid += valid_len
                z_item = z_item.transpose(0, 1).reshape(self.video_latent_ch, t, h, w)
                if is_i2v and first_frames is not None:
                    z_item[:, :1] = first_frames[i].permute(3, 0, 1, 2)
                pad_h = (self.patch_size - h % self.patch_size) % self.patch_size
                pad_w = (self.patch_size - w % self.patch_size) % self.patch_size
                if pad_h or pad_w:
                    z_item = F.pad(z_item, (0, pad_w, 0, pad_h), mode="constant", value=0)
                video_inputs.append(z_item)

            if has_audio and audio_inputs is not None and latents_audio is not None and audio_len_list is not None:
                audio_len = int(audio_len_list[i])
                audio_inputs.append(latents_audio[offset_audio : offset_audio + audio_len])
                offset_audio += audio_len

        pred_video_list, pred_audio_list = self.backbone(
            vid=video_inputs,
            audio=audio_inputs,
            t=timesteps,
            vid_context=vid_context,
            audio_context=audio_context,
            vid_seq_len=max_seq_len_video,
            audio_seq_len=max_seq_len_audio,
            spk_embed=spk_embs,
            spk_pos=spk_pos,
            masking_modality=masking_modality,
            first_frame_is_clean=is_i2v and first_frames is not None,
        )

        velocity_pred_vid = torch.zeros_like(latents_vid) if has_video and latents_vid is not None else None
        velocity_pred_audio = torch.zeros_like(latents_audio) if has_audio and latents_audio is not None else None
        offset_vid = 0
        offset_audio = 0
        if has_video and velocity_pred_vid is not None and pred_video_list is not None and t_h_w_list is not None:
            for i, pred in enumerate(pred_video_list):
                t, h, w = [int(x) for x in t_h_w_list[i]]
                flat_pred = pred[:, :t, :h, :w].permute(1, 2, 3, 0).flatten(0, 2)
                velocity_pred_vid[offset_vid : offset_vid + flat_pred.shape[0]] = flat_pred
                offset_vid += flat_pred.shape[0]
        if has_audio and velocity_pred_audio is not None and pred_audio_list is not None and audio_len_list is not None:
            for i, pred in enumerate(pred_audio_list):
                audio_len = int(audio_len_list[i])
                flat_pred = pred[:audio_len]
                velocity_pred_audio[offset_audio : offset_audio + flat_pred.shape[0]] = flat_pred
                offset_audio += flat_pred.shape[0]
        return velocity_pred_vid, velocity_pred_audio

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        params = dict(self.named_parameters())
        loaded: set[str] = set()
        unmatched: list[str] = []
        seen = 0
        for name, tensor in weights:
            seen += 1
            name = name.removeprefix("transformer.")
            if not name.startswith("backbone.") and f"backbone.{name}" in params:
                name = f"backbone.{name}"
            if name in params:
                default_weight_loader(params[name], tensor)
                loaded.add(name)
            else:
                unmatched.append(name)
        if seen and not loaded:
            raise ValueError(
                "No NAVA transformer weights matched the native adapter parameters. "
                f"First unmatched keys: {unmatched[:5]}"
            )
        if unmatched:
            logger.warning(
                "Ignored %d unmatched NAVA transformer weight(s), first keys: %s", len(unmatched), unmatched[:5]
            )
        missing = sorted(set(params) - loaded)
        if missing:
            logger.warning("NAVA transformer left %d parameter(s) unloaded, first keys: %s", len(missing), missing[:5])
        return loaded
