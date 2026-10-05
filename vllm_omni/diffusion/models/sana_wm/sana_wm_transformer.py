# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""SANA-WM Stage-1 transformer.

Native vLLM-Omni port of the NVlabs SANA-WM DiT. Modules are built eagerly at
construction (under the loader's target-device context) and checkpoint tensors
are streamed in via ``load_weights``. The Bidirectional Gated DeltaNet
recurrence is implemented in pure PyTorch in this module.
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from typing import Any, ClassVar

import torch
import torch.nn.functional as F
from diffusers.models.embeddings import Timesteps
from torch import nn
from vllm.distributed import (
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
    tensor_model_parallel_all_reduce,
)
from vllm.distributed.parallel_state import get_tp_group
from vllm.model_executor.layers.linear import (
    ColumnParallelLinear,
    MergedColumnParallelLinear,
    QKVParallelLinear,
    RowParallelLinear,
)
from vllm.model_executor.layers.quantization.base_config import QuantizationConfig
from vllm.model_executor.model_loader.weight_utils import sharded_weight_loader

from vllm_omni.diffusion.attention.backends.abstract import AttentionMetadata
from vllm_omni.diffusion.attention.layer import Attention
from vllm_omni.diffusion.layers.adalayernorm import AdaLayerNorm
from vllm_omni.diffusion.layers.norm import RMSNorm
from vllm_omni.diffusion.models.sana_wm.config import SanaWmConfig
from vllm_omni.diffusion.models.sana_wm.ucpe import (
    SanaWmCamGeometry,
    cam_prep_func,
    prepare_cam_geometry,
)

SANA_WM_STAGE1_LATENT_CHANNELS = 128
SANA_WM_STAGE1_PROMPT_CHANNELS = 2304
SANA_WM_STAGE1_TIMESTEP_CHANNELS = 256


def _shard_param_across_tp(param: torch.Tensor | None, dim: int = 0) -> None:
    """Attach vLLM's TP shard loader to a plain (non-parallel-layer) parameter.

    vLLM parallel layers narrow full checkpoint tensors to the local shard via
    their own ``weight_loader``. SANA-WM also has plain parameters that become
    TP-local - the GDN vectors (``A_log``, ``dt_bias``), the depthwise temporal
    convs and the q/k norms - so they need the same behavior at load time.
    Attached only under TP > 1: at TP = 1 the checkpoint tensor already matches
    the parameter and ``sharded_weight_loader`` would require an initialized TP
    group that standalone/unit-test builds do not have.
    """
    if param is None or get_tensor_model_parallel_world_size() <= 1:
        return
    param.weight_loader = sharded_weight_loader(dim)


def _is_sana_wm_transformer_block(name: str, module: Any) -> bool:
    del module
    parts = name.split(".")
    return len(parts) == 2 and parts[0] == "blocks" and parts[1].isdigit()


def _to_3tuple(value: int | tuple[int, int] | tuple[int, int, int]) -> tuple[int, int, int]:
    if isinstance(value, int):
        return (value, value, value)
    if len(value) == 2:
        return (1, int(value[0]), int(value[1]))
    return (int(value[0]), int(value[1]), int(value[2]))


# ---------------------------------------------------------------------------
# Bidirectional Gated DeltaNet recurrence (pure PyTorch)
#
# Ported from the Apache-2.0 NVlabs/Sana reference. SANA-WM's bidirectional
# video-latent recurrence is a different contract than vLLM's autoregressive
# Qwen3-Next GDN cache path, so it stays model-local.
# ---------------------------------------------------------------------------


def _validate_gdn_inputs(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    beta: torch.Tensor,
    decay: torch.Tensor,
    spatial_tokens: int,
) -> tuple[int, int, int, int, int]:
    if query.ndim != 4 or key.ndim != 4 or value.ndim != 4:
        raise ValueError("Sana-WM GDN query/key/value must be shaped [B, H, D, N].")
    if query.shape != key.shape or query.shape != value.shape:
        raise ValueError(
            "Sana-WM GDN query/key/value shapes must match, got "
            f"{tuple(query.shape)}, {tuple(key.shape)}, {tuple(value.shape)}."
        )
    if beta.ndim != 4:
        raise ValueError("Sana-WM GDN beta must be shaped [B, H, T, S].")
    if decay.ndim != 3:
        raise ValueError("Sana-WM GDN decay must be shaped [B, H, T].")
    if spatial_tokens <= 0:
        raise ValueError("Sana-WM GDN spatial_tokens must be positive.")

    batch_size, num_heads, head_dim, token_count = query.shape
    if token_count % spatial_tokens != 0:
        raise ValueError(f"Sana-WM GDN token count {token_count} is not divisible by spatial_tokens={spatial_tokens}.")
    frames = token_count // spatial_tokens
    if beta.shape != (batch_size, num_heads, frames, spatial_tokens):
        raise ValueError(
            "Sana-WM GDN beta shape mismatch: expected "
            f"{(batch_size, num_heads, frames, spatial_tokens)}, got {tuple(beta.shape)}."
        )
    if decay.shape != (batch_size, num_heads, frames):
        raise ValueError(
            f"Sana-WM GDN decay shape mismatch: expected {(batch_size, num_heads, frames)}, got {tuple(decay.shape)}."
        )
    return batch_size, num_heads, head_dim, token_count, frames


def _delta_scan(
    query_rot: torch.Tensor,
    key_rot: torch.Tensor,
    value: torch.Tensor,
    beta: torch.Tensor,
    decay: torch.Tensor,
    *,
    spatial_tokens: int,
    query: torch.Tensor | None = None,
    key: torch.Tensor | None = None,
    skip_z: bool = False,
    flip_output: bool = False,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """One-directional gated delta-rule recurrence over frames.

    The KV stream consumes only the rotary-embedded Q/K; the Z (denominator)
    stream consumes only the non-rotary Q/K. ``skip_z=True`` drops the Z stream
    entirely and returns ``(numerator, None)`` — the numerator-only variant the
    camera branch needs (NVlabs ``torch_recurrent_cam_single_path_delta_rule``),
    so ``query``/``key`` may be omitted there.

    ``flip_output=True`` emits the frames in reverse of the order they were
    computed, which is what a caller running on flipped inputs wants; it costs a
    list reversal instead of a flip over the assembled result.
    """
    if not skip_z and (query is None or key is None):
        raise ValueError("Sana-WM delta scan needs non-rotary query/key unless skip_z is set.")

    batch_size, num_heads, head_dim, token_count = query_rot.shape
    frames = beta.shape[2]
    if token_count != frames * spatial_tokens:
        raise ValueError(f"Sana-WM delta scan token_count={token_count} != frames*S={frames * spatial_tokens}.")

    def to_frames(tensor: torch.Tensor) -> torch.Tensor:
        return tensor.view(batch_size, num_heads, head_dim, frames, spatial_tokens).permute(0, 1, 3, 2, 4)

    query_rot_f = to_frames(query_rot)
    key_rot_f = to_frames(key_rot)
    value_f = to_frames(value)
    state_kv = torch.zeros(batch_size, num_heads, head_dim, head_dim, device=query_rot.device, dtype=query_rot.dtype)
    numerators: list[torch.Tensor] = []

    if skip_z:
        query_f = key_f = None
        state_z = None
        denominators = None
    else:
        query_f = to_frames(query)
        key_f = to_frames(key)
        state_z = torch.zeros(batch_size, num_heads, head_dim, 1, device=query_rot.device, dtype=query_rot.dtype)
        denominators = []

    for frame_idx in range(frames):
        query_rot_t = query_rot_f[:, :, frame_idx]
        key_rot_t = key_rot_f[:, :, frame_idx]
        value_t = value_f[:, :, frame_idx]
        beta_t = beta[:, :, frame_idx].unsqueeze(2)
        decay_t = decay[:, :, frame_idx].view(batch_size, num_heads, 1, 1)

        state_kv = state_kv * decay_t
        value_pred = torch.matmul(state_kv, key_rot_t)
        delta_value = (value_t - value_pred) * beta_t
        state_kv = state_kv + torch.matmul(delta_value, key_rot_t.transpose(-1, -2))
        numerators.append(torch.matmul(state_kv, query_rot_t))

        if skip_z:
            continue
        query_t = query_f[:, :, frame_idx]
        key_t = key_f[:, :, frame_idx]
        state_z = state_z * decay_t
        z_pred = torch.matmul(state_z.transpose(-1, -2), key_t)
        delta_z = (1.0 - z_pred) * beta_t
        state_z = state_z + torch.matmul(key_t, delta_z.transpose(-1, -2))
        denominators.append(torch.matmul(state_z.transpose(-1, -2), query_t))

    def restore(tensors: list[torch.Tensor], dim: int) -> torch.Tensor:
        stacked = torch.stack(tensors[::-1] if flip_output else tensors, dim=2)
        return stacked.permute(0, 1, 3, 2, 4).reshape(batch_size, num_heads, dim, token_count)

    return restore(numerators, head_dim), (None if skip_z else restore(denominators, 1))


def _bidirectional_delta_scan(
    query_rot: torch.Tensor,
    key_rot: torch.Tensor,
    value: torch.Tensor,
    beta: torch.Tensor,
    decay: torch.Tensor,
    *,
    spatial_tokens: int,
    query: torch.Tensor | None = None,
    key: torch.Tensor | None = None,
    skip_z: bool = False,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Forward + backward delta scan, summed.

    The backward direction shifts K/V/beta by one frame (zero pad) and the decay
    by one frame (neutral 1.0 pad), matching the NVlabs ``flip_and_shift``
    convention. The backward scan also emits its frames in forward order, so the
    two directions add directly instead of flipping the assembled result back.
    """
    frames = beta.shape[2]

    def reverse(tensor: torch.Tensor, *, shift_value: float | None = None) -> torch.Tensor:
        return _reverse_frames(tensor, frames=frames, spatial_tokens=spatial_tokens, shift_value=shift_value)

    num_fwd, den_fwd = _delta_scan(
        query_rot,
        key_rot,
        value,
        beta,
        decay,
        spatial_tokens=spatial_tokens,
        query=query,
        key=key,
        skip_z=skip_z,
    )

    bwd_kwargs: dict[str, torch.Tensor] = {}
    if not skip_z:
        bwd_kwargs["query"] = reverse(query)
        bwd_kwargs["key"] = reverse(key, shift_value=0.0)

    num_bwd, den_bwd = _delta_scan(
        reverse(query_rot),
        reverse(key_rot, shift_value=0.0),
        reverse(value, shift_value=0.0),
        _flip_and_shift(beta, dim=2, shift_value=0.0),
        _flip_and_shift(decay, dim=2, shift_value=1.0),
        spatial_tokens=spatial_tokens,
        skip_z=skip_z,
        flip_output=True,
        **bwd_kwargs,
    )

    return num_fwd + num_bwd, (None if skip_z else den_fwd + den_bwd)


def _reverse_frames(
    tensor: torch.Tensor,
    *,
    frames: int,
    spatial_tokens: int,
    shift_value: float | None = None,
) -> torch.Tensor:
    """Reverse frame order on a ``[..., T*S]`` tensor, optionally shifting by one frame.

    The frame axis is the outer half of the flat token axis, so unfolding it in
    place keeps the tensor's own layout: the flip writes a contiguous result and
    the reshape back is a view. Routing this through a ``[..., T, D, S]``
    permutation instead would cost a second full materialization.
    """
    lead = tensor.shape[:-1]
    reversed_ = torch.flip(tensor.reshape(*lead, frames, spatial_tokens), dims=[-2])
    if shift_value is not None:
        padding = torch.full((*lead, 1, spatial_tokens), shift_value, device=tensor.device, dtype=tensor.dtype)
        reversed_ = torch.cat([padding, reversed_.narrow(-2, 0, frames - 1)], dim=-2)
    return reversed_.reshape(*lead, frames * spatial_tokens)


def _flip_and_shift(tensor: torch.Tensor, *, dim: int, shift_value: float) -> torch.Tensor:
    flipped = torch.flip(tensor, dims=[dim])
    shifted = flipped.narrow(dim, 0, tensor.shape[dim] - 1)
    pad_shape = list(tensor.shape)
    pad_shape[dim] = 1
    padding = torch.full(pad_shape, shift_value, device=tensor.device, dtype=tensor.dtype)
    return torch.cat([padding, shifted], dim=dim)


def reference_bidirectional_gated_delta_net(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    beta: torch.Tensor,
    decay: torch.Tensor,
    spatial_tokens: int,
    query_rot: torch.Tensor | None = None,
    key_rot: torch.Tensor | None = None,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Run the SANA-WM bidirectional gated delta recurrence in PyTorch.

    Inputs follow the layout used by the official Stage-1 operator:
    ``query/key/value/query_rot/key_rot`` are ``[B, H, D, T*S]``, ``beta`` is
    ``[B, H, T, S]``, and ``decay`` is ``[B, H, T]``.
    """

    _validate_gdn_inputs(query, key, value, beta, decay, spatial_tokens)
    if query_rot is None:
        query_rot = query
    if key_rot is None:
        key_rot = key
    if query_rot.shape != query.shape or key_rot.shape != key.shape:
        raise ValueError("Sana-WM GDN rotary query/key shapes must match query/key.")

    dtype_orig = query.dtype
    numerator, denominator = _bidirectional_delta_scan(
        query_rot.float(),
        key_rot.float(),
        value.float(),
        beta.float(),
        decay.float(),
        spatial_tokens=spatial_tokens,
        query=query.float(),
        key=key.float(),
    )
    return (numerator / (denominator + eps)).to(dtype_orig)


class TensorParallelRMSNorm(nn.Module):
    """RMSNorm whose statistics span a tensor-parallel sharded last dimension.

    Models that normalize q/k across *all* heads (``qk_norm="rms_norm_across_heads"``)
    cannot use a plain RMSNorm once the q/k projections are column-parallel: each
    rank would then compute the RMS over its local head shard only. This
    all-reduces the squared sum so the denominator matches the global width,
    while the affine weight stays local.

    Model-local on purpose: LTX-2 and DreamZero carry their own near-identical
    copies, and converging all three belongs in its own change rather than in
    the SANA-WM port.
    """

    def __init__(
        self,
        hidden_size: int,
        eps: float = 1e-6,
        elementwise_affine: bool = True,
        tp_size: int = 1,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.tp_size = max(int(tp_size), 1)
        self.global_hidden_size = hidden_size * self.tp_size
        self.eps = eps
        if elementwise_affine:
            self.weight = nn.Parameter(torch.ones(hidden_size))
        else:
            self.register_parameter("weight", None)

    def _local_sum_sq(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x_float = x.float()
        return x_float, x_float.pow(2).sum(dim=-1, keepdim=True)

    def _scale(self, x_float: torch.Tensor, global_sum_sq: torch.Tensor, input_dtype: torch.dtype) -> torch.Tensor:
        out = x_float * torch.rsqrt(global_sum_sq / self.global_hidden_size + self.eps)
        if self.weight is not None:
            out = out * self.weight.float()
        return out.to(dtype=input_dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_float, sum_sq = self._local_sum_sq(x)
        if self.tp_size > 1:
            sum_sq = tensor_model_parallel_all_reduce(sum_sq)
        return self._scale(x_float, sum_sq, x.dtype)


def fused_qk_rms_norm(
    norm_q: nn.Module,
    norm_k: nn.Module,
    q: torch.Tensor,
    k: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply q/k :class:`TensorParallelRMSNorm` with a SINGLE fused all-reduce.

    Self-attention normalizes q and k every step, and under TP each norm issues
    its own tiny all-reduce of the per-token sum-of-squares. These collectives
    are latency-bound, so packing both sums into one tensor halves the count
    (2 collectives -> 1) for free.

    Numerically identical to ``norm_q(q), norm_k(k)``: all-reduce is
    elementwise, so packing along the last dim reduces each slice independently
    with the same fp32 accumulation. Falls back to independent application when
    either norm is not a :class:`TensorParallelRMSNorm` (e.g. ``nn.Identity``
    when qk_norm is off, or plain ``RMSNorm`` at TP=1).
    """
    if not (isinstance(norm_q, TensorParallelRMSNorm) and isinstance(norm_k, TensorParallelRMSNorm)):
        return norm_q(q), norm_k(k)
    if norm_q.tp_size <= 1:
        return norm_q(q), norm_k(k)

    q_float, q_sum_sq = norm_q._local_sum_sq(q)
    k_float, k_sum_sq = norm_k._local_sum_sq(k)
    packed = tensor_model_parallel_all_reduce(torch.cat([q_sum_sq, k_sum_sq], dim=-1))
    q_sum_sq, k_sum_sq = packed[..., 0:1], packed[..., 1:2]
    return norm_q._scale(q_float, q_sum_sq, q.dtype), norm_k._scale(k_float, k_sum_sq, k.dtype)


def _make_sharded_qk_rms_norm(hidden_size: int, eps: float = 1e-6) -> nn.Module:
    """Build the q/k norm, TP-aware.

    SANA-WM normalizes q/k across all heads, so once the projections are
    column-parallel the RMS denominator has to span the TP ranks.
    """
    tp_size = get_tensor_model_parallel_world_size()
    if tp_size > 1:
        return TensorParallelRMSNorm(hidden_size, eps=eps, tp_size=tp_size)
    return RMSNorm(hidden_size, eps=eps)


class SanaWmTextProjection(nn.Module):
    def __init__(
        self,
        prompt_channels: int,
        hidden_size: int,
        *,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.fc1 = ColumnParallelLinear(
            prompt_channels,
            hidden_size,
            bias=True,
            gather_output=False,
            return_bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.fc1" if prefix else "fc1",
        )
        self.fc2 = RowParallelLinear(
            hidden_size,
            hidden_size,
            bias=True,
            input_is_parallel=True,
            return_bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.fc2" if prefix else "fc2",
        )
        self.act = nn.SiLU()

    def forward(self, prompt_embeds: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.act(self.fc1(prompt_embeds)))


class SanaWmTextEmbedder(nn.Module):
    def __init__(
        self,
        prompt_channels: int,
        hidden_size: int,
        max_length: int,
        *,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.y_embedding = nn.Parameter(torch.zeros(max_length, prompt_channels))
        self.y_proj = SanaWmTextProjection(
            prompt_channels,
            hidden_size,
            quant_config=quant_config,
            prefix=f"{prefix}.y_proj" if prefix else "y_proj",
        )

    def forward(
        self,
        prompt_embeds: torch.Tensor | None,
        *,
        batch_size: int,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        if prompt_embeds is None:
            prompt_embeds = self.y_embedding.unsqueeze(0).expand(batch_size, -1, -1)
        return self.y_proj(prompt_embeds.to(dtype=dtype))


class SanaWmTimestepEmbedder(nn.Module):
    def __init__(
        self,
        in_features: int,
        hidden_size: int,
        *,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        linear_1 = ColumnParallelLinear(
            in_features,
            hidden_size,
            bias=True,
            gather_output=False,
            return_bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.mlp.0" if prefix else "mlp.0",
        )
        linear_2 = RowParallelLinear(
            hidden_size,
            hidden_size,
            bias=True,
            input_is_parallel=True,
            return_bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.mlp.2" if prefix else "mlp.2",
        )
        self.act = nn.SiLU()
        self.mlp = nn.ModuleList([linear_1, self.act, linear_2])
        # diffusers' Timesteps reproduces this model's sinusoid exactly:
        # downscale_freq_shift=0 makes the exponent denominator half_dim, and
        # flip_sin_to_cos=True emits [cos, sin] in that order. The MLP stays
        # model-local because its checkpoint keys are mlp.0/mlp.2 and it is
        # tensor-parallel.
        self.timesteps_proj = Timesteps(num_channels=in_features, flip_sin_to_cos=True, downscale_freq_shift=0)

    def forward(self, timestep: torch.Tensor) -> torch.Tensor:
        emb = self.timesteps_proj(timestep.flatten())
        hidden_states = self.mlp[0](emb.to(dtype=self.mlp[0].weight.dtype))
        hidden_states = self.act(hidden_states)
        return self.mlp[2](hidden_states)


class SanaWmPatchEmbedMS3D(nn.Module):
    """Official-style 3D patch embedder used by SANA-WM Stage-1."""

    def __init__(
        self,
        patch_size: tuple[int, int, int],
        in_channels: int,
        hidden_size: int,
        *,
        kernel_size: tuple[int, int, int] | None = None,
        bias: bool = True,
    ) -> None:
        super().__init__()
        self.patch_size = _to_3tuple(patch_size)
        self.kernel_size = _to_3tuple(kernel_size or patch_size)
        self.proj = nn.Conv3d(
            in_channels,
            hidden_size,
            kernel_size=self.kernel_size,
            stride=self.patch_size,
            bias=bias,
        )
        self.norm = nn.Identity()

    def project_with_shape(self, latents: torch.Tensor) -> tuple[torch.Tensor, tuple[int, int, int]]:
        hidden_states = self.norm(self.proj(latents))
        _, _, frames, height, width = hidden_states.shape
        return hidden_states.flatten(2).transpose(1, 2), (frames, height, width)

    def forward(self, latents: torch.Tensor) -> torch.Tensor:
        return self.project_with_shape(latents)[0]


class SanaWmWanRotaryPosEmbed(nn.Module):
    """Wan-style 3D RoPE table used by the official SANA-WM GDN blocks."""

    def __init__(
        self,
        attention_head_dim: int,
        *,
        max_seq_len: int = 1024,
        theta: float = 10000.0,
    ) -> None:
        super().__init__()
        self.theta = theta
        h_dim = w_dim = 2 * (attention_head_dim // 6)
        t_dim = attention_head_dim - h_dim - w_dim
        self._split_sizes = (
            t_dim // 2,
            h_dim // 2,
            w_dim // 2,
        )
        self.freqs = self._build_freqs(max_seq_len)

    def _build_1d_freq(self, dim: int, positions: torch.Tensor) -> torch.Tensor:
        if dim <= 0:
            return torch.empty(positions.shape[0], 0, dtype=torch.complex128)
        freqs = 1.0 / (
            self.theta ** (torch.arange(0, dim, 2, dtype=torch.float64, device=positions.device)[: dim // 2] / dim)
        )
        phase = torch.outer(positions.to(torch.float64), freqs)
        return torch.polar(torch.ones_like(phase), phase)

    def _build_freqs(self, max_seq_len: int) -> torch.Tensor:
        positions = torch.arange(max_seq_len, dtype=torch.float64)
        dims = (
            self._split_sizes[0] * 2,
            self._split_sizes[1] * 2,
            self._split_sizes[2] * 2,
        )
        return torch.cat([self._build_1d_freq(dim, positions) for dim in dims], dim=1)

    def forward(self, spatial_shape: tuple[int, int, int], device: torch.device) -> torch.Tensor:
        frames, height, width = spatial_shape
        if max(spatial_shape) > self.freqs.shape[0]:
            self.freqs = self._build_freqs(max(spatial_shape)).to(self.freqs.device)
        freqs = self.freqs.to(device=device)
        freqs_f, freqs_h, freqs_w = freqs.split(self._split_sizes, dim=1)
        f_dim, h_dim, w_dim = self._split_sizes
        parts = [
            freqs_f[:frames].view(frames, 1, 1, f_dim).expand(frames, height, width, f_dim),
            freqs_h[:height].view(1, height, 1, h_dim).expand(frames, height, width, h_dim),
            freqs_w[:width].view(1, 1, width, w_dim).expand(frames, height, width, w_dim),
        ]
        return torch.cat(parts, dim=-1).reshape(1, 1, frames * height * width, -1)


class SanaWmSelfAttention(nn.Module):
    def __init__(
        self,
        config: SanaWmConfig,
        *,
        use_gdn: bool = True,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        hidden_size = config.hidden_size
        self.total_num_heads = max(hidden_size // max(config.linear_head_dim, 1), 1)
        self.head_dim = hidden_size // self.total_num_heads
        self.num_heads = self.total_num_heads
        self.num_kv_heads = self.total_num_heads
        self.eps = 1e-8
        self.use_gdn = use_gdn and "GDN" in config.attn_type
        self.patch_size = _to_3tuple(config.patch_size)
        # Total camera branch width from the checkpoint. TP ColumnParallel
        # layers expose a per-rank local slice at runtime; keep both sizes so
        # layer construction uses the global contract while local conv/norm
        # tensors match q/k/v outputs.
        self.total_cam_dim = hidden_size // max(config.cam_attn_compress, 1)
        self.cam_dim = self.total_cam_dim
        self.qkv = QKVParallelLinear(
            hidden_size=hidden_size,
            head_size=self.head_dim,
            total_num_heads=self.total_num_heads,
            bias=False,
            return_bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.qkv" if prefix else "qkv",
        )
        self.num_heads = self.qkv.num_heads
        self.num_kv_heads = self.qkv.num_kv_heads
        self.proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            hidden_size,
            bias=True,
            input_is_parallel=True,
            return_bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.proj" if prefix else "proj",
        )
        self.beta_proj = ColumnParallelLinear(
            hidden_size,
            self.total_num_heads,
            bias=True,
            gather_output=False,
            return_bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.beta_proj" if prefix else "beta_proj",
        )
        self.gate_proj = ColumnParallelLinear(
            hidden_size,
            self.total_num_heads,
            bias=True,
            gather_output=False,
            return_bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.gate_proj" if prefix else "gate_proj",
        )
        self.output_gate = ColumnParallelLinear(
            hidden_size,
            self.total_num_heads * self.head_dim,
            bias=True,
            gather_output=False,
            return_bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.output_gate" if prefix else "output_gate",
        )
        self.q_proj_cam = ColumnParallelLinear(
            hidden_size,
            self.total_cam_dim,
            bias=True,
            gather_output=False,
            return_bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.q_proj_cam" if prefix else "q_proj_cam",
        )
        self.k_proj_cam = ColumnParallelLinear(
            hidden_size,
            self.total_cam_dim,
            bias=True,
            gather_output=False,
            return_bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.k_proj_cam" if prefix else "k_proj_cam",
        )
        self.v_proj_cam = ColumnParallelLinear(
            hidden_size,
            self.total_cam_dim,
            bias=True,
            gather_output=False,
            return_bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.v_proj_cam" if prefix else "v_proj_cam",
        )
        self.out_proj_cam = RowParallelLinear(
            self.total_cam_dim,
            hidden_size,
            bias=True,
            input_is_parallel=True,
            # The camera contribution is summed into the TP-local GDN stream, so
            # the cross-rank sum and the slice down to the local width collapse
            # into a single reduce-scatter (see _reduce_scatter_cam_contrib).
            # RowParallelLinear must therefore not all-reduce, and its bias is
            # applied after the scatter rather than inside the matmul.
            reduce_results=False,
            skip_bias_add=True,
            return_bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.out_proj_cam" if prefix else "out_proj_cam",
        )
        local_inner_dim = self.num_heads * self.head_dim
        norm_cls = _make_sharded_qk_rms_norm if config.qk_norm else (lambda *_args, **_kwargs: nn.Identity())
        self.q_norm = norm_cls(local_inner_dim)
        self.k_norm = norm_cls(local_inner_dim)

        # These names mirror the official GDN checkpoint.
        self.A_log = nn.Parameter(torch.zeros(self.num_heads))
        self.dt_bias = nn.Parameter(torch.zeros(self.num_heads))
        self.register_buffer("recall_gate", torch.zeros(1))
        self.conv_k = (
            nn.Conv1d(
                local_inner_dim,
                local_inner_dim,
                kernel_size=config.conv_kernel_size,
                groups=local_inner_dim,
                bias=False,
            )
            if self.use_gdn and config.conv_kernel_size > 0
            else None
        )
        self.conv_q = None
        self.conv_v = None

        self.cam_dim = int(getattr(self.q_proj_cam, "output_size_per_partition", self.total_cam_dim))

        cam_compress = max(config.cam_attn_compress, 1)
        self.cam_heads = max(self.num_heads // cam_compress, 1)
        if self.cam_dim % self.cam_heads != 0:
            raise ValueError(f"Sana-WM local cam_dim={self.cam_dim} must be divisible by cam_heads={self.cam_heads}.")
        self.cam_head_dim = self.cam_dim // self.cam_heads
        # Under TP, cam_head_dim must equal main head_dim so the WAN RoPE
        # (built for main head_dim) can be correctly sliced for the cam branch.
        # A mismatch indicates the cam ColumnParallelLinear layers were not
        # partitioned consistently with the main QKVParallelLinear.
        if self.cam_head_dim != self.head_dim:
            raise ValueError(
                f"Sana-WM cam_head_dim={self.cam_head_dim} != head_dim={self.head_dim}. "
                f"Under TP the cam layers must be partitioned by the same TP degree as the "
                f"main QKV (cam_dim={self.cam_dim}, cam_heads={self.cam_heads}, "
                f"total_cam_dim={self.total_cam_dim})."
            )
        self.q_norm_cam = norm_cls(self.cam_dim)
        self.k_norm_cam = norm_cls(self.cam_dim)
        self.conv_k_cam = (
            nn.Conv1d(
                self.cam_dim,
                self.cam_dim,
                kernel_size=config.conv_kernel_size,
                groups=self.cam_dim,
                bias=False,
            )
            if self.use_gdn and config.conv_kernel_size > 0
            else None
        )
        self.conv_q_cam = None
        self.conv_v_cam = None
        # Softmax hybrid blocks (every ``softmax_every_n``-th) and the camera
        # branch both run plain non-causal self-attention, so both go through
        # the shared Attention for platform backend selection and the fp32
        # fallback. The camera branch gets its own module because
        # ``cam_attn_compress`` may give it fewer heads.
        self.softmax_attn = Attention(
            num_heads=self.num_heads,
            head_size=self.head_dim,
            num_kv_heads=self.num_kv_heads,
            softmax_scale=self.head_dim**-0.5,
            causal=False,
            role="self",
            qkv_layout="BSND",
            skip_sequence_parallel=True,
            prefix=prefix,
        )
        self.softmax_attn_cam = Attention(
            num_heads=self.cam_heads,
            head_size=self.cam_head_dim,
            num_kv_heads=self.cam_heads,
            softmax_scale=self.cam_head_dim**-0.5,
            causal=False,
            role="self",
            qkv_layout="BSND",
            skip_sequence_parallel=True,
            prefix=prefix,
        )
        self._init_short_convs()
        self._mark_tp_sharded_params()

    def _mark_tp_sharded_params(self) -> None:
        """Declare which plain parameters are TP-local (see ``_shard_param_across_tp``)."""
        # Per-head GDN vectors: one entry per local head.
        _shard_param_across_tp(self.A_log)
        _shard_param_across_tp(self.dt_bias)
        # Depthwise temporal convs and q/k norms: one entry per local channel.
        for module in (self.conv_q, self.conv_k, self.conv_v, self.conv_q_cam, self.conv_k_cam, self.conv_v_cam):
            if module is not None:
                _shard_param_across_tp(module.weight)
        for norm in (self.q_norm, self.k_norm, self.q_norm_cam, self.k_norm_cam):
            _shard_param_across_tp(getattr(norm, "weight", None))

    @staticmethod
    def _init_short_conv(conv: nn.Conv1d | None) -> None:
        if conv is None:
            return
        with torch.no_grad():
            conv.weight.zero_()
            conv.weight[:, 0, -1] = 1.0

    def _init_short_convs(self) -> None:
        for conv in (self.conv_q, self.conv_k, self.conv_v, self.conv_q_cam, self.conv_k_cam, self.conv_v_cam):
            self._init_short_conv(conv)

    @staticmethod
    def _apply_rotary_emb(hidden_states: torch.Tensor, freqs: torch.Tensor) -> torch.Tensor:
        rotated = torch.view_as_complex(hidden_states.permute(0, 1, 3, 2).to(torch.float64).unflatten(3, (-1, 2)))
        output = torch.view_as_real(rotated * freqs).flatten(3, 4).permute(0, 1, 3, 2)
        return output.type_as(hidden_states)

    @staticmethod
    def _reshape_to_temporal(
        hidden_states: torch.Tensor,
        spatial_shape: tuple[int, int, int],
    ) -> tuple[torch.Tensor, int, int, int]:
        batch_size, token_count, hidden_size = hidden_states.shape
        frames, height, width = spatial_shape
        spatial_tokens = height * width
        if token_count != frames * spatial_tokens:
            raise ValueError(f"Sana-WM temporal conv expects N=T*H*W, got N={token_count}, THW={spatial_shape}.")
        hidden_states = hidden_states.reshape(batch_size, frames, spatial_tokens, hidden_size)
        hidden_states = hidden_states.permute(0, 2, 1, 3).reshape(batch_size * spatial_tokens, frames, hidden_size)
        return hidden_states, batch_size, spatial_tokens, frames

    @staticmethod
    def _reshape_from_temporal(
        hidden_states: torch.Tensor,
        batch_size: int,
        spatial_tokens: int,
        frames: int,
    ) -> torch.Tensor:
        hidden_size = hidden_states.shape[-1]
        return (
            hidden_states.reshape(batch_size, spatial_tokens, frames, hidden_size)
            .permute(0, 2, 1, 3)
            .reshape(batch_size, frames * spatial_tokens, hidden_size)
        )

    @staticmethod
    def _causal_conv_1d(hidden_states: torch.Tensor, conv: nn.Conv1d) -> torch.Tensor:
        dtype = hidden_states.dtype
        conv_input = hidden_states.transpose(1, 2).to(conv.weight.dtype)
        conv_input = F.pad(conv_input, (conv.kernel_size[0] - 1, 0))
        output = conv(conv_input).transpose(1, 2)
        return output.to(dtype)

    def _bidirectional_temporal_short_conv(
        self,
        hidden_states: torch.Tensor,
        conv: nn.Conv1d,
        spatial_shape: tuple[int, int, int],
    ) -> torch.Tensor:
        hidden_states, batch_size, spatial_tokens, frames = self._reshape_to_temporal(hidden_states, spatial_shape)
        forward_states = self._causal_conv_1d(hidden_states, conv)
        backward_states = self._causal_conv_1d(hidden_states.flip(1), conv).flip(1)
        center_weight = conv.weight[:, 0, -1].to(hidden_states.dtype)
        center_states = hidden_states * center_weight.view(1, 1, -1)
        output = forward_states + backward_states - center_states
        return self._reshape_from_temporal(output, batch_size, spatial_tokens, frames)

    def _compute_frame_gates(
        self,
        hidden_states: torch.Tensor,
        spatial_shape: tuple[int, int, int],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, token_count, hidden_size = hidden_states.shape
        frames, height, width = spatial_shape
        spatial_tokens = height * width
        if token_count != frames * spatial_tokens:
            raise ValueError(f"Sana-WM GDN token layout mismatch: N={token_count}, expected {frames * spatial_tokens}.")
        beta = torch.sigmoid(self.beta_proj(hidden_states))
        beta = beta.reshape(batch_size, frames, spatial_tokens, self.num_heads).permute(0, 3, 1, 2)
        frame_states = hidden_states.reshape(batch_size, frames, spatial_tokens, hidden_size).mean(dim=2)
        gate = self.gate_proj(frame_states).float()
        decay = torch.exp(
            -self.A_log.float().exp().view(1, 1, -1) * F.softplus(gate + self.dt_bias.float().view(1, 1, -1))
        )
        return beta, decay.transpose(1, 2)

    def _forward_gdn_raw(
        self,
        hidden_states: torch.Tensor,
        spatial_shape: tuple[int, int, int],
        rotary_emb: torch.Tensor | None,
        *,
        precomputed_gates: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """Run main-branch bidirectional GDN and return the (B, N, q_size)
        raw output BEFORE ``output_gate`` and ``proj`` are applied.

        Returns ``(raw_output, (beta, decay))`` so the caller can share the
        beta/decay gates with the camera branch (matches NVlabs' shared
        ``precomputed_gates`` plumbing).
        """
        batch_size, token_count, _hidden_size = hidden_states.shape
        frames, height, width = spatial_shape
        spatial_tokens = height * width
        if token_count != frames * spatial_tokens:
            raise ValueError(f"Sana-WM GDN expects N=T*H*W, got N={token_count}, THW={spatial_shape}.")

        qkv = self.qkv(hidden_states)
        q_size = self.num_heads * self.head_dim
        kv_size = self.num_kv_heads * self.head_dim
        query, key, value = qkv.split([q_size, kv_size, kv_size], dim=-1)
        if self.conv_k is not None:
            key = self._bidirectional_temporal_short_conv(key, self.conv_k, spatial_shape)
        if precomputed_gates is None:
            beta, decay = self._compute_frame_gates(hidden_states, spatial_shape)
        else:
            beta, decay = precomputed_gates

        query, key = fused_qk_rms_norm(self.q_norm, self.k_norm, query, key)
        query = query.reshape(batch_size, token_count, self.num_heads, self.head_dim)
        key = key.reshape(batch_size, token_count, self.num_heads, self.head_dim)
        value = value.reshape(batch_size, token_count, self.num_heads, self.head_dim)

        query = F.relu(query).permute(0, 2, 3, 1)
        key = F.relu(key).permute(0, 2, 3, 1)
        value = value.permute(0, 2, 3, 1)
        key = key * ((self.head_dim**-0.5) * (spatial_tokens**-0.5))
        if rotary_emb is not None:
            query_rot = self._apply_rotary_emb(query, rotary_emb)
            key_rot = self._apply_rotary_emb(key, rotary_emb)
        else:
            query_rot = query
            key_rot = key

        output = reference_bidirectional_gated_delta_net(
            query,
            key,
            value,
            beta=beta,
            decay=decay,
            spatial_tokens=spatial_tokens,
            query_rot=query_rot,
            key_rot=key_rot,
            eps=self.eps,
        )
        output = output.permute(0, 3, 1, 2).reshape(batch_size, token_count, q_size)
        return output, (beta, decay)

    def _apply_output_gate_and_proj(
        self,
        combined: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        """Apply SiLU output gate (driven by hidden_states) + final projection.

        The gate reads ``hidden_states``, not ``combined``, and NVlabs applies
        it once to the main+camera sum rather than per branch.
        """
        gate = F.silu(self.output_gate(hidden_states).float())
        gated = combined * gate
        return self.proj(gated.to(self.proj.weight.dtype))

    def _reduce_scatter_cam_contrib(self, contrib: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
        """Reduce the partial camera projection straight down to the local shard.

        ``out_proj_cam`` is row-parallel over the camera features, so each rank
        produces a partial sum spanning the full hidden width. The GDN stream it
        is added to is TP-local (``output_gate`` and ``proj`` both expect the
        local width), so an all-reduce would hand every rank the full result
        only for all but one slice of it to be discarded. Reduce-scatter does
        both steps in one collective and moves 1/tp of the bytes.

        ``skip_bias_add`` keeps the bias out of the matmul so it can be applied
        to the scattered shard; that slice is the only remaining use of the TP
        rank here, and it is a 1-D view rather than a copy of the hidden tensor.

        ``tensor_model_parallel_reduce_scatter`` is not usable here: it calls
        ``get_tp_group().reduce_scatter()``, and vLLM-Omni's diffusion
        ``GroupCoordinator`` deliberately does not implement that method (see
        the note in ``diffusion/distributed/parallel_state.py``). Drive
        ``torch.distributed`` on the group's process group instead, and fall
        back to all-reduce plus a slice if the group is not exposed.
        """
        bias = self.out_proj_cam.bias
        tp_size = get_tensor_model_parallel_world_size()
        if tp_size == 1:
            return contrib if bias is None else contrib + bias

        local_width = reference.shape[-1]
        if local_width <= 0 or contrib.shape[-1] != local_width * tp_size:
            raise ValueError(
                "Sana-WM TP camera contribution width mismatch: "
                f"contrib={contrib.shape[-1]} expected={local_width * tp_size} (local={local_width}, tp={tp_size})."
            )

        device_group = getattr(get_tp_group(), "device_group", None)
        if device_group is None:
            out = tensor_model_parallel_all_reduce(contrib.contiguous())
            start = get_tensor_model_parallel_rank() * local_width
            out = out[..., start : start + local_width].contiguous()
        else:
            # reduce_scatter_tensor splits along dim 0, so move the rank axis to
            # the front first. Rank r then receives the summed columns
            # [r * local_width, (r + 1) * local_width), matching what the
            # all-reduce-and-slice fallback produces.
            batch, tokens, _ = contrib.shape
            source = (
                contrib.reshape(batch, tokens, tp_size, local_width)
                .permute(2, 0, 1, 3)
                .contiguous()
                .reshape(tp_size * batch, tokens, local_width)
            )
            out = torch.empty((batch, tokens, local_width), device=contrib.device, dtype=contrib.dtype)
            torch.distributed.reduce_scatter_tensor(out, source, group=device_group)

        if bias is not None:
            start = get_tensor_model_parallel_rank() * local_width
            out = out + bias[start : start + local_width]
        return out

    def _forward_gdn(
        self,
        hidden_states: torch.Tensor,
        spatial_shape: tuple[int, int, int],
        rotary_emb: torch.Tensor | None,
    ) -> torch.Tensor:
        """Main-branch only GDN forward with output_gate + proj applied."""
        raw, _ = self._forward_gdn_raw(hidden_states, spatial_shape, rotary_emb)
        return self._apply_output_gate_and_proj(raw, hidden_states)

    @staticmethod
    def _downscale_to_reference_rms(
        ref: torch.Tensor,
        transformed: torch.Tensor,
        eps: float = 1e-6,
    ) -> torch.Tensor:
        """Downscale ``transformed`` so its per-token channel RMS does not
        exceed ``ref``'s. Mirrors NVlabs ``_downscale_to_reference_rms``.

        Inputs are ``(B, H, N, D)`` (channel dim last). The RMS-scale is
        clamped at 1.0 — the function only ever shrinks, never amplifies.
        """
        ref_rms = ref.square().mean(dim=-1, keepdim=True).add(eps).sqrt()
        tr_rms = transformed.square().mean(dim=-1, keepdim=True).add(eps).sqrt()
        scale = (ref_rms / tr_rms.clamp_min(eps)).clamp(max=1.0)
        return transformed * scale

    @staticmethod
    def _ucpe_rotary_freqs(rotary_emb: torch.Tensor | None) -> torch.Tensor | None:
        # SanaWmRope emits (1, 1, N, D//2) complex; the cam path wants (N, D//2).
        if rotary_emb is None:
            return None
        return rotary_emb.squeeze(0).squeeze(0)

    def _forward_softmax_raw(
        self,
        hidden_states: torch.Tensor,
        spatial_shape: tuple[int, int, int],
        rotary_emb: torch.Tensor | None,
    ) -> torch.Tensor:
        """Softmax self-attention raw output before output_gate/proj.

        Mirrors NVlabs ``_forward_softmax_attn(..., apply_output_gate=False)``
        for every-N-th hybrid blocks. GDN-specific gates are intentionally not
        used here.
        """
        batch_size, token_count, hidden_size = hidden_states.shape
        frames, height, width = spatial_shape
        if token_count != frames * height * width:
            raise ValueError(f"Sana-WM softmax attention expects N=T*H*W, got N={token_count}, THW={spatial_shape}.")

        qkv = self.qkv(hidden_states)
        q_size = self.num_heads * self.head_dim
        kv_size = self.num_kv_heads * self.head_dim
        query, key, value = qkv.split([q_size, kv_size, kv_size], dim=-1)

        query, key = fused_qk_rms_norm(self.q_norm, self.k_norm, query, key)
        query = query.reshape(batch_size, token_count, self.num_heads, self.head_dim)
        key = key.reshape(batch_size, token_count, self.num_kv_heads, self.head_dim)
        value = value.reshape(batch_size, token_count, self.num_kv_heads, self.head_dim)

        if rotary_emb is not None:
            query = self._apply_rotary_emb(query.permute(0, 2, 3, 1), rotary_emb).permute(0, 3, 1, 2)
            key = self._apply_rotary_emb(key.permute(0, 2, 3, 1), rotary_emb).permute(0, 3, 1, 2)

        # NVlabs runs the attention in bf16 even under ``fp32_attention=True``,
        # since FlashAttention only supports bf16/fp16 and fp32 falls back to
        # the math backend. Keep Q/K/V at the module dtype so an upstream step
        # that promoted to fp32 does not silently change the kernel. See
        # ``sana_gdn_camctrl_blocks.py::_forward_softmax_attn_sdpa``.
        dtype_orig = hidden_states.dtype
        attn = self.softmax_attn(query.to(dtype_orig), key.to(dtype_orig), value.to(dtype_orig))
        return attn.reshape(batch_size, token_count, q_size)

    def _forward_softmax_cam_branch(
        self,
        hidden_states: torch.Tensor,
        spatial_shape: tuple[int, int, int],
        cam_geometry: SanaWmCamGeometry,
        rotary_emb: torch.Tensor | None,
    ) -> torch.Tensor:
        """UCPE camera branch for softmax hybrid blocks.

        Matches NVlabs ``_forward_cam_branch_softmax``: camera Q/K/V are
        projected from the same hidden stream, transformed by UCPE, run through
        SDPA, inverse-transformed, then returned as raw ``cam_dim`` features.
        """
        batch_size, token_count, _ = hidden_states.shape
        frames, height, width = spatial_shape
        if token_count != frames * height * width:
            raise ValueError(f"Sana-WM softmax cam branch expects N=T*H*W, got N={token_count}, THW={spatial_shape}.")

        q_cam = self.q_proj_cam(hidden_states)
        k_cam = self.k_proj_cam(hidden_states)
        v_cam = self.v_proj_cam(hidden_states)

        if self.conv_q_cam is not None:
            q_cam = self._bidirectional_temporal_short_conv(q_cam, self.conv_q_cam, spatial_shape)
        if self.conv_k_cam is not None:
            k_cam = self._bidirectional_temporal_short_conv(k_cam, self.conv_k_cam, spatial_shape)
        if self.conv_v_cam is not None:
            v_cam = self._bidirectional_temporal_short_conv(v_cam, self.conv_v_cam, spatial_shape)

        q_cam, k_cam = fused_qk_rms_norm(self.q_norm_cam, self.k_norm_cam, q_cam, k_cam)
        q_cam = q_cam.reshape(batch_size, token_count, self.cam_heads, self.cam_head_dim)
        k_cam = k_cam.reshape(batch_size, token_count, self.cam_heads, self.cam_head_dim)
        v_cam = v_cam.reshape(batch_size, token_count, self.cam_heads, self.cam_head_dim)

        q_cam = q_cam.transpose(1, 2).contiguous()  # (B, H_cam, N, D)
        k_cam = k_cam.transpose(1, 2).contiguous()
        v_cam = v_cam.transpose(1, 2).contiguous()

        q_cam_trans = cam_geometry.apply_q(q_cam)
        kv_cam_trans = cam_geometry.apply_kv(torch.cat([k_cam, v_cam], dim=1))
        k_cam_trans, v_cam_trans = torch.chunk(kv_cam_trans, chunks=2, dim=1)

        q_cam_trans = self._downscale_to_reference_rms(q_cam, q_cam_trans)
        k_cam_trans = self._downscale_to_reference_rms(k_cam, k_cam_trans)
        v_cam_trans = self._downscale_to_reference_rms(v_cam, v_cam_trans)

        # The shared Attention takes BSND and selects a backend that supports
        # ``cam_head_dim`` directly, so odd head dims need no padding up to an
        # SDPA-friendly size. Q/K/V stay at the module dtype for the same
        # reason as the main softmax path above.
        dtype_orig = hidden_states.dtype
        out = self.softmax_attn_cam(
            q_cam_trans.transpose(1, 2).to(dtype_orig),
            k_cam_trans.transpose(1, 2).to(dtype_orig),
            v_cam_trans.transpose(1, 2).to(dtype_orig),
        )
        out = cam_geometry.apply_output(out.transpose(1, 2))
        return out.transpose(1, 2).reshape(batch_size, token_count, self.cam_dim)

    def _forward_cam_branch(
        self,
        hidden_states: torch.Tensor,
        spatial_shape: tuple[int, int, int],
        cam_geometry: SanaWmCamGeometry,
        rotary_emb: torch.Tensor | None,
        *,
        precomputed_gates: tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        """UCPE camera branch — matches the NVlabs production
        ``BidirectionalGDNUCPESinglePathLiteLABothTriton`` variant declared
        in the SANA-WM 1600M release config.

        Two things differ from the main GDN branch and are easy to get wrong:
        Q/K/V all come from ``hidden_states`` (UCPE is not cross-attention —
        camera information enters through the per-pixel projection matrices),
        and the recurrence is single-path, i.e. numerator only, with no Z
        denominator and no final divide.

        Returns ``(B, N, cam_dim)`` raw — caller applies ``out_proj_cam`` and
        the shared ``output_gate`` + ``proj``.
        """
        batch_size, token_count, _ = hidden_states.shape
        frames, height, width = spatial_shape
        spatial_tokens = height * width
        if token_count != frames * spatial_tokens:
            raise ValueError(f"Sana-WM cam branch expects N=T*H*W, got N={token_count}, THW={spatial_shape}.")

        q_cam = self.q_proj_cam(hidden_states)
        k_cam = self.k_proj_cam(hidden_states)
        v_cam = self.v_proj_cam(hidden_states)

        if self.conv_k_cam is not None:
            k_cam = self._bidirectional_temporal_short_conv(k_cam, self.conv_k_cam, spatial_shape)
        if self.conv_q_cam is not None:
            q_cam = self._bidirectional_temporal_short_conv(q_cam, self.conv_q_cam, spatial_shape)
        if self.conv_v_cam is not None:
            v_cam = self._bidirectional_temporal_short_conv(v_cam, self.conv_v_cam, spatial_shape)

        # Normalise through the model's own norm modules: under TP these are
        # TensorParallelRMSNorm, which all-reduces the squared sum so the
        # denominator spans the global channel width. Doing the RMS inside
        # cam_prep_func would normalise over this rank's head shard only. q and
        # k share a shape here, so the two reductions fuse into one.
        q_cam, k_cam = fused_qk_rms_norm(self.q_norm_cam, self.k_norm_cam, q_cam, k_cam)
        q_normed = q_cam.reshape(batch_size, token_count, self.cam_heads, self.cam_head_dim)
        k_normed = k_cam.reshape(batch_size, token_count, self.cam_heads, self.cam_head_dim)

        # cam_prep_func returns NVlabs' ``(B, H, D, N)`` scan layout directly,
        # so nothing needs permuting before the recurrence below.
        v_raw = v_cam.reshape(batch_size, token_count, self.cam_heads, self.cam_head_dim).contiguous()
        k_scale = (self.cam_head_dim**-0.5) * (spatial_tokens**-0.5)
        q_rot_bhdn, k_rot_bhdn, v_bhdn, inflation_sq = cam_prep_func(
            q_normed.contiguous(),
            k_normed.contiguous(),
            v_raw,
            proj_q=cam_geometry.proj_q,
            proj_kv=cam_geometry.proj_kv,
            rope_cos=cam_geometry.rope_cos,
            rope_sin=cam_geometry.rope_sin,
            k_scale=k_scale,
        )

        # Discount beta by the raw K inflation instead of shrinking Q/K/V first:
        # the released NVlabs ``BothTriton`` cam path skips the Python
        # ``PostUCPERenorm`` step and feeds the scan the raw UCPE-transformed
        # tensors.
        beta, decay = precomputed_gates
        inflation_per_token = inflation_sq.reshape(batch_size, self.cam_heads, frames, spatial_tokens)
        frame_inflation_sq = inflation_per_token.mean(dim=-1)  # (B, H_cam, T)
        # β/decay come from the main branch, so the cam branch can only reuse
        # them head-for-head. cam_attn_compress > 1 would give the cam branch
        # fewer heads and needs a grouping rule this port does not implement;
        # __init__ already rejects it via the cam_head_dim check.
        if beta.shape[1] != self.cam_heads:
            raise ValueError(
                "Sana-WM cam branch requires cam_heads == main heads (cam_attn_compress=1),"
                f" got cam_heads={self.cam_heads} main_heads={beta.shape[1]}."
            )
        beta_cam = beta / frame_inflation_sq.unsqueeze(-1).clamp_min(1.0)
        decay_cam = decay

        cam_out_bhdn, _ = _bidirectional_delta_scan(
            q_rot_bhdn.float(),
            k_rot_bhdn.float(),
            v_bhdn.float(),
            beta_cam.float(),
            decay_cam.float(),
            spatial_tokens=spatial_tokens,
            skip_z=True,
        )
        cam_out_bhdn = cam_out_bhdn.to(q_rot_bhdn.dtype)

        # ``apply_output`` expects (B, H, N, D); the scan emits (B, H, D, N).
        cam_out_bhnd = cam_out_bhdn.transpose(-1, -2).contiguous()
        cam_out_bhnd = cam_geometry.apply_output(cam_out_bhnd)
        cam_out_bhdn = cam_out_bhnd.transpose(-1, -2).contiguous()

        # (B, H_cam, D_cam, N) → (B, N, cam_dim)
        return cam_out_bhdn.permute(0, 3, 1, 2).reshape(batch_size, token_count, self.cam_dim)

    # The GDN/UCPE branch runs a sequential recurrence whose trip count is the
    # latent frame count, i.e. data-dependent at trace time. Run the whole
    # attention eagerly (graph break) rather than letting torch.compile unroll
    # it per shape; the rest of the block (MLP / norms / cross-attention) still
    # compiles.
    @torch.compiler.disable
    def forward(
        self,
        hidden_states: torch.Tensor,
        spatial_shape: tuple[int, int, int],
        rotary_emb: torch.Tensor | None = None,
        cam_geometry: SanaWmCamGeometry | None = None,
    ) -> torch.Tensor:
        """Forward with dual-branch GDN+UCPE when ``cam_geometry`` is
        provided, otherwise main-branch-only GDN or softmax fallback.

        ``cam_geometry`` holds the per-pixel 4x4 transforms built once per
        transformer forward from the raw ``(B, T_latent, 20)`` raymap. The cam
        branch consumes ``hidden_states`` as its sole Q/K/V source and applies
        those transforms to the Q/K/V channels via UCPE.
        """
        if self.use_gdn:
            if cam_geometry is None:
                # No camera conditioning on this request — main branch only.
                return self._forward_gdn(hidden_states, spatial_shape, rotary_emb)
            # β/decay are computed here rather than inside each branch: NVlabs
            # drives both recurrences from one set of gates, and the camera
            # branch then discounts beta by its own K inflation.
            beta, decay = self._compute_frame_gates(hidden_states, spatial_shape)
            main_raw, _ = self._forward_gdn_raw(
                hidden_states,
                spatial_shape,
                rotary_emb,
                precomputed_gates=(beta, decay),
            )
            cam_raw = self._forward_cam_branch(
                hidden_states,
                spatial_shape,
                cam_geometry,
                rotary_emb,
                precomputed_gates=(beta, decay),
            )
            cam_contrib = self.out_proj_cam(cam_raw)
            cam_contrib = self._reduce_scatter_cam_contrib(cam_contrib, main_raw)
            combined = main_raw + cam_contrib.to(main_raw.dtype)
            attn_out = self._apply_output_gate_and_proj(combined, hidden_states)
            return attn_out

        main_raw = self._forward_softmax_raw(hidden_states, spatial_shape, rotary_emb)
        if cam_geometry is not None:
            cam_raw = self._forward_softmax_cam_branch(
                hidden_states,
                spatial_shape,
                cam_geometry,
                rotary_emb,
            )
            cam_contrib = self.out_proj_cam(cam_raw)
            cam_contrib = self._reduce_scatter_cam_contrib(cam_contrib, main_raw)
            main_raw = main_raw + cam_contrib.to(main_raw.dtype)
        return self._apply_output_gate_and_proj(main_raw, hidden_states)


class SanaWmCrossAttention(nn.Module):
    def __init__(
        self,
        config: SanaWmConfig,
        *,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        hidden_size = config.hidden_size
        self.total_num_heads = max(hidden_size // max(config.linear_head_dim, 1), 1)
        self.head_dim = hidden_size // self.total_num_heads
        self.num_heads = self.total_num_heads
        inner = self.total_num_heads * self.head_dim
        self.q_linear = ColumnParallelLinear(
            hidden_size,
            inner,
            bias=True,
            gather_output=False,
            return_bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.q_linear" if prefix else "q_linear",
        )
        # The checkpoint stores K and V fused as one (2 * inner, hidden)
        # tensor. A plain ColumnParallelLinear would shard that by the
        # concatenated output dim, handing all of K to one rank and all of V
        # to another; MergedColumnParallelLinear knows the block structure
        # and shards K and V by heads independently, so `chunk(2)` below
        # yields this rank's K and V directly - no gather, no manual slice.
        self.kv_linear = MergedColumnParallelLinear(
            hidden_size,
            [inner, inner],
            bias=True,
            gather_output=False,
            return_bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.kv_linear" if prefix else "kv_linear",
        )
        self.num_heads = self.q_linear.output_size_per_partition // self.head_dim
        self.proj = RowParallelLinear(
            inner,
            hidden_size,
            bias=True,
            input_is_parallel=True,
            return_bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.proj" if prefix else "proj",
        )
        local_inner = self.num_heads * self.head_dim
        norm_cls = _make_sharded_qk_rms_norm if config.cross_norm else (lambda *_args, **_kwargs: nn.Identity())
        self.q_norm = norm_cls(local_inner)
        self.k_norm = norm_cls(local_inner)
        # q is column-parallel and k is sliced to the local width before k_norm,
        # so both norms hold one weight entry per local channel.
        _shard_param_across_tp(getattr(self.q_norm, "weight", None))
        _shard_param_across_tp(getattr(self.k_norm, "weight", None))
        self.softmax_attn = Attention(
            num_heads=self.num_heads,
            head_size=self.head_dim,
            num_kv_heads=self.num_heads,
            softmax_scale=1.0 / (self.head_dim**0.5),
            causal=False,
            role="cross",
            qkv_layout="BSND",
            skip_sequence_parallel=True,
            disable_kv_quant=True,
            prefix=prefix,
        )

    def _reshape_to_seq_heads(self, tensor: torch.Tensor) -> torch.Tensor:
        batch, seq_len, hidden_size = tensor.shape
        return tensor.reshape(batch, seq_len, self.num_heads, hidden_size // self.num_heads)

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        query = self.q_norm(self.q_linear(hidden_states))
        # MergedColumnParallelLinear emits [K_local; V_local], so the chunk is
        # already this rank's head shard.
        key, value = self.kv_linear(encoder_hidden_states).chunk(2, dim=-1)
        key = self.k_norm(key)
        if attention_mask is not None and attention_mask.ndim != 2:
            raise ValueError(
                f"Sana-WM cross-attention mask must be shaped [B, text_len], got {tuple(attention_mask.shape)}."
            )
        query = self._reshape_to_seq_heads(query)
        key = self._reshape_to_seq_heads(key)
        value = self._reshape_to_seq_heads(value)
        # The prompt padding mask goes to the shared Attention through
        # ``AttentionMetadata``: backends take the [B, text_len] form directly
        # and convert/reshape it themselves (``_maybe_reshape_attn_mask``), so
        # cross-attention keeps backend selection. ``None`` keeps the mask-free
        # fast path - callers normalise an all-ones mask to ``None`` rather than
        # paying for a no-op mask.
        attn_metadata = AttentionMetadata(attn_mask=attention_mask) if attention_mask is not None else None
        attn = self.softmax_attn(query, key, value, attn_metadata)
        return self.proj(attn.flatten(2, 3))


class _ConvWrapper(nn.Module):
    def __init__(self, conv: nn.Module) -> None:
        super().__init__()
        self.conv = conv


class SanaWmMbConvFfn(nn.Module):
    def __init__(self, config: SanaWmConfig) -> None:
        super().__init__()
        hidden_size = config.hidden_size
        expanded = int(hidden_size * config.mlp_ratio) * 2
        t_padding = config.t_kernel_size // 2
        self.glu_act = nn.SiLU()
        self.inverted_conv = _ConvWrapper(nn.Conv2d(hidden_size, expanded, kernel_size=1))
        self.depth_conv = _ConvWrapper(nn.Conv2d(expanded, expanded, kernel_size=3, padding=1, groups=expanded))
        self.point_conv = _ConvWrapper(nn.Conv2d(expanded // 2, hidden_size, kernel_size=1, bias=False))
        self.t_conv = nn.Conv2d(
            hidden_size,
            hidden_size,
            kernel_size=(config.t_kernel_size, 1),
            padding=(t_padding, 0),
            bias=False,
        )

    def forward(self, hidden_states: torch.Tensor, spatial_shape: tuple[int, int, int]) -> torch.Tensor:
        """Match NVlabs ``GLUMBConvTemp.forward`` exactly:

        1. Reshape ``(B, N=F*H*W, C) → (B*F, H, W, C) → (B*F, C, H, W)`` so the
           spatial MBConv runs PER FRAME with proper 2D (H, W) geometry — the
           3×3 ``depth_conv`` must not span an (F, H*W) plane.
        2. Apply expand → depthwise 3×3 → GLU → contract on per-frame
           (B*F, C, H, W).
        3. Temporal aggregation: reshape back to ``(B, C, F, H*W)``
           and add ``t_conv`` (kernel ``(3, 1)``) along the F axis.
        4. Reshape to ``(B, N, C)``.
        """
        batch, _, hidden_size = hidden_states.shape
        frames, height, width = spatial_shape
        spatial_tokens = height * width
        # (B, F*H*W, C) → (B*F, H, W, C) → (B*F, C, H, W)
        x = (
            hidden_states.reshape(batch * frames, spatial_tokens, hidden_size)
            .reshape(batch * frames, height, width, hidden_size)
            .permute(0, 3, 1, 2)
        )
        # Keep this NHWC-derived NCHW view non-contiguous like NVlabs: on bf16
        # CUDA convs the stride selects a different kernel, and a contiguous
        # copy breaks late-step parity.
        # NVlabs ``ConvLayer`` is `Conv → norm → act` with
        # `act=("silu", "silu", None)`, so inverted_conv activates and the other
        # two do not. ``_ConvWrapper`` stores only the raw Conv2d (to match the
        # `inverted_conv.conv.weight` checkpoint key), hence the explicit SiLU.
        x_inv = self.glu_act(self.inverted_conv.conv(x))
        x_dep = self.depth_conv.conv(x_inv)
        value, gate = x_dep.chunk(2, dim=1)
        x_glu = value * self.glu_act(gate)
        x_pt = self.point_conv.conv(x_glu)  # (B*F, hidden, H, W)
        x = x_pt
        # Temporal aggregation: (B*F, C, H, W) → (B, F, C, H*W) → (B, C, F, H*W)
        x_temporal = x.reshape(batch, frames, hidden_size, spatial_tokens).permute(0, 2, 1, 3)
        t_conv_out = self.t_conv(x_temporal)
        x_temporal = x_temporal + t_conv_out
        # → (B, F, H*W, C) → (B, N, C)
        final = x_temporal.permute(0, 2, 3, 1).reshape(batch, frames * spatial_tokens, hidden_size).contiguous()
        return final


class SanaWmBlock(nn.Module):
    def __init__(
        self,
        config: SanaWmConfig,
        *,
        block_idx: int = 0,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        hidden_size = config.hidden_size
        use_gdn = config.softmax_every_n <= 0 or (block_idx + 1) % config.softmax_every_n != 0
        use_plucker_proj = config.use_chunk_plucker_post_attn and (
            config.chunk_plucker_post_attn_blocks < 0 or block_idx < config.chunk_plucker_post_attn_blocks
        )
        self.norm1 = AdaLayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = SanaWmSelfAttention(
            config,
            use_gdn=use_gdn,
            quant_config=quant_config,
            prefix=f"{prefix}.attn" if prefix else "attn",
        )
        self.cross_attn = SanaWmCrossAttention(
            config,
            quant_config=quant_config,
            prefix=f"{prefix}.cross_attn" if prefix else "cross_attn",
        )
        self.norm2 = AdaLayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.mlp = SanaWmMbConvFfn(config)
        if use_plucker_proj:
            self.plucker_proj: nn.Module | None = ColumnParallelLinear(
                hidden_size,
                hidden_size,
                bias=True,
                gather_output=True,
                return_bias=False,
                quant_config=quant_config,
                prefix=f"{prefix}.plucker_proj" if prefix else "plucker_proj",
            )
        else:
            self.plucker_proj = None
        self.scale_shift_table = nn.Parameter(torch.zeros(6, hidden_size))

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        timestep_modulation: torch.Tensor,
        spatial_shape: tuple[int, int, int],
        rotary_emb: torch.Tensor | None = None,
        camera_hidden_states: torch.Tensor | None = None,
        cam_geometry: SanaWmCamGeometry | None = None,
        encoder_attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # Per-frame timestep contract (see SanaWmTransformer3DModel.forward).
        if timestep_modulation.ndim > 2:
            return self._forward_frame_aware(
                hidden_states,
                encoder_hidden_states,
                timestep_modulation,
                spatial_shape,
                rotary_emb,
                camera_hidden_states,
                cam_geometry,
                encoder_attention_mask,
            )
        batch_size = hidden_states.shape[0]
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.scale_shift_table[None] + timestep_modulation.reshape(batch_size, 6, -1)
        ).chunk(6, dim=1)
        attn_input = self.norm1(hidden_states, scale_msa, shift_msa)
        attn_output = self.attn(attn_input, spatial_shape, rotary_emb, cam_geometry)
        if camera_hidden_states is not None and self.plucker_proj is not None:
            attn_output = attn_output + self.plucker_proj(camera_hidden_states)
        hidden_states = hidden_states + gate_msa * attn_output
        hidden_states = hidden_states + self.cross_attn(
            hidden_states,
            encoder_hidden_states,
            encoder_attention_mask,
        )
        mlp_input = self.norm2(hidden_states, scale_mlp, shift_mlp)
        hidden_states = hidden_states + gate_mlp * self.mlp(mlp_input, spatial_shape)
        return hidden_states

    def _forward_frame_aware(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        timestep_modulation: torch.Tensor,
        spatial_shape: tuple[int, int, int],
        rotary_emb: torch.Tensor | None,
        camera_hidden_states: torch.Tensor | None,
        cam_geometry: SanaWmCamGeometry | None,
        encoder_attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Per-frame timestep modulation path matching NVlabs
        ``SanaVideoMSCamCtrlBlock.forward_frame_aware``.

        ``timestep_modulation`` is ``(B, 1, F, 6*D)``. We split into the
        six per-frame ``(B, F, 1, D)`` chunks via ``scale_shift_table``
        and apply ``shift/scale/gate`` per frame, broadcasting over the
        spatial tokens within each frame.
        """
        batch_size, token_count, hidden_size = hidden_states.shape
        frames, height, width = spatial_shape
        spatial_tokens = height * width
        if token_count != frames * spatial_tokens:
            raise ValueError(f"Sana-WM frame-aware block expects N=T*H*W, got N={token_count}, THW={spatial_shape}.")
        if timestep_modulation.shape[2] != frames:
            raise ValueError(
                "Sana-WM frame-aware block: timestep frame axis must match spatial frames, "
                f"got modulation frames={timestep_modulation.shape[2]} vs spatial frames={frames}."
            )

        # (B, 1, F, 6*D) -> (B, F, 6, D), then add the (1, 1, 6, D) table.
        t_per_frame = timestep_modulation.reshape(batch_size, frames, 6, -1)
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.scale_shift_table[None, None, :, :] + t_per_frame
        ).chunk(6, dim=-2)  # each: (B, F, 1, D)

        # Apply per-frame modulation by reshaping x to (B, F, S, D), broadcasting
        # scale/shift over the spatial axis, then flattening back. The reshape
        # comes first because the norm only reduces over the last axis, so it is
        # unaffected, and the modulation then broadcasts as written.
        x_4d = hidden_states.reshape(batch_size, frames, spatial_tokens, hidden_size)
        x_msa_in = self.norm1(x_4d, scale_msa, shift_msa).reshape(batch_size, token_count, hidden_size)
        attn_output = self.attn(x_msa_in, spatial_shape, rotary_emb, cam_geometry)
        if camera_hidden_states is not None and self.plucker_proj is not None:
            attn_output = attn_output + self.plucker_proj(camera_hidden_states)
        attn_output_4d = attn_output.reshape(batch_size, frames, spatial_tokens, hidden_size)
        hidden_states = hidden_states + (gate_msa * attn_output_4d).reshape(batch_size, token_count, hidden_size)

        hidden_states = hidden_states + self.cross_attn(
            hidden_states,
            encoder_hidden_states,
            encoder_attention_mask,
        )

        x_4d = hidden_states.reshape(batch_size, frames, spatial_tokens, hidden_size)
        x_mlp_in = self.norm2(x_4d, scale_mlp, shift_mlp).reshape(batch_size, token_count, hidden_size)
        mlp_out = self.mlp(x_mlp_in, spatial_shape)
        mlp_out_4d = mlp_out.reshape(batch_size, frames, spatial_tokens, hidden_size)
        hidden_states = hidden_states + (gate_mlp * mlp_out_4d).reshape(batch_size, token_count, hidden_size)
        return hidden_states


class SanaWmFinalLayer(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        patch_size: tuple[int, int, int],
        out_channels: int,
        *,
        quant_config: Any = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.norm_final = AdaLayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.scale_shift_table = nn.Parameter(torch.zeros(2, hidden_size))
        self.out_channels = out_channels
        out_features = math.prod(patch_size) * out_channels
        self.linear: nn.Module = ColumnParallelLinear(
            hidden_size,
            out_features,
            bias=True,
            gather_output=True,
            return_bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.linear" if prefix else "linear",
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        timestep_embed: torch.Tensor,
        spatial_shape: tuple[int, int, int] | None = None,
    ) -> torch.Tensor:
        # Per-frame timestep contract (see SanaWmTransformer3DModel.forward).
        if timestep_embed.ndim > 2:
            return self._forward_frame_aware(hidden_states, timestep_embed, spatial_shape)
        shift, scale = (self.scale_shift_table[None] + timestep_embed[:, None]).chunk(2, dim=1)
        return self.linear(self.norm_final(hidden_states, scale, shift))

    def _forward_frame_aware(
        self,
        hidden_states: torch.Tensor,
        timestep_embed: torch.Tensor,
        spatial_shape: tuple[int, int, int] | None,
    ) -> torch.Tensor:
        """Per-frame final-layer modulation matching NVlabs
        ``T2IFinalLayer.forward_frame_aware``.

        ``timestep_embed`` is ``(B, 1, F, D)``. We transpose to
        ``(B, F, 1, D)`` so it adds correctly to the ``(1, 1, 2, D)``
        ``scale_shift_table`` and produces per-frame ``(B, F, 1, D)``
        ``shift`` / ``scale`` that broadcast over the spatial tokens
        within each frame.
        """
        batch_size, token_count, hidden_size = hidden_states.shape
        frames = timestep_embed.shape[2]
        if spatial_shape is not None:
            spatial_tokens = spatial_shape[1] * spatial_shape[2]
        else:
            spatial_tokens = token_count // frames
        if frames * spatial_tokens != token_count:
            raise ValueError(
                "Sana-WM frame-aware final layer: token count mismatch "
                f"(N={token_count}, F={frames}, S={spatial_tokens})."
            )
        # (B, 1, F, D) -> (B, F, 1, D); add (1, 1, 2, D); chunk into shift/scale: each (B, F, 1, D).
        t_per_frame = timestep_embed.transpose(1, 2)
        shift, scale = (self.scale_shift_table[None, None, :, :] + t_per_frame).chunk(2, dim=-2)
        x_4d = hidden_states.reshape(batch_size, frames, spatial_tokens, hidden_size)
        x_mod = self.norm_final(x_4d, scale, shift).reshape(batch_size, token_count, hidden_size)
        return self.linear(x_mod)


class SanaWmTransformer3DModel(nn.Module):
    """SANA-WM Stage-1 DiT: bidirectional Gated DeltaNet blocks with a softmax
    attention block every ``softmax_every_n``."""

    _repeated_blocks: ClassVar[list[str]] = ["SanaWmBlock"]
    _layerwise_offload_blocks_attrs: ClassVar[list[str]] = ["blocks"]
    _hsdp_shard_conditions: ClassVar[list[Any]] = [_is_sana_wm_transformer_block]

    def __init__(
        self,
        config: SanaWmConfig | None = None,
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
        latent_channels: int = SANA_WM_STAGE1_LATENT_CHANNELS,
        prompt_channels: int = SANA_WM_STAGE1_PROMPT_CHANNELS,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.config = config or SanaWmConfig()
        self.quant_config = quant_config
        self.prefix = prefix
        self._latent_channels = latent_channels
        self._prompt_channels = prompt_channels
        self.patch_size = _to_3tuple(self.config.patch_size)
        self.x_embedder = SanaWmPatchEmbedMS3D(self.patch_size, self._latent_channels, self.config.hidden_size)
        self.y_embedder = SanaWmTextEmbedder(
            self._prompt_channels,
            self.config.hidden_size,
            self.config.model_max_length,
            quant_config=self.quant_config,
            prefix=f"{self.prefix}.y_embedder" if self.prefix else "y_embedder",
        )
        self.t_embedder = SanaWmTimestepEmbedder(
            SANA_WM_STAGE1_TIMESTEP_CHANNELS,
            self.config.hidden_size,
            quant_config=self.quant_config,
            prefix=f"{self.prefix}.t_embedder" if self.prefix else "t_embedder",
        )
        # t_block: SiLU → Linear(hidden, 6*hidden).  Weight key is t_block.1.{weight,bias}
        # to match the Stage-1 checkpoint layout (nn.Sequential index 1).
        _t_block_linear: nn.Module = ColumnParallelLinear(
            self.config.hidden_size,
            6 * self.config.hidden_size,
            bias=True,
            gather_output=True,
            return_bias=False,
            quant_config=self.quant_config,
            prefix=f"{self.prefix}.t_block.1" if self.prefix else "t_block.1",
        )
        self.t_block = nn.Sequential(nn.SiLU(), _t_block_linear)
        self.plucker_embedder = SanaWmPatchEmbedMS3D(
            self.patch_size,
            self.config.chunk_plucker_channels,
            self.config.hidden_size,
        )
        self.raymap_embedder = SanaWmPatchEmbedMS3D(self.patch_size, 3, self.config.hidden_size)
        self.blocks = nn.ModuleList(
            [
                SanaWmBlock(
                    self.config,
                    block_idx=i,
                    quant_config=self.quant_config,
                    prefix=f"{self.prefix}.blocks.{i}" if self.prefix else f"blocks.{i}",
                )
                for i in range(self.config.num_blocks)
            ]
        )
        self.final_layer = SanaWmFinalLayer(
            self.config.hidden_size,
            self.patch_size,
            self._latent_channels,
            quant_config=self.quant_config,
            prefix=f"{self.prefix}.final_layer" if self.prefix else "final_layer",
        )
        self.pos_embed = nn.Parameter(torch.zeros(1, 484, self.config.hidden_size))
        self.rope = SanaWmWanRotaryPosEmbed(self.config.linear_head_dim)
        self.attention_y_norm = RMSNorm(self.config.hidden_size)
        if device is not None or dtype is not None:
            self.to(device=device, dtype=dtype)

    def _positional_embedding(self, token_count: int, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
        pos_embed = self.pos_embed.to(device=device, dtype=dtype)
        if pos_embed.shape[1] == token_count:
            return pos_embed
        pos_embed = pos_embed.transpose(1, 2)
        pos_embed = F.interpolate(pos_embed, size=token_count, mode="linear", align_corners=False)
        return pos_embed.transpose(1, 2)

    @staticmethod
    def _match_tokens(hidden_states: torch.Tensor, expected_tokens: int) -> torch.Tensor:
        if hidden_states.shape[1] == expected_tokens:
            return hidden_states
        if hidden_states.shape[1] > expected_tokens:
            return hidden_states[:, :expected_tokens]
        pad = hidden_states[:, -1:].expand(-1, expected_tokens - hidden_states.shape[1], -1)
        return torch.cat([hidden_states, pad], dim=1)

    def _camera_hidden_states_from_conditions(
        self,
        *,
        plucker: torch.Tensor | None,
        spatial_raymap: torch.Tensor | None,
        spatial_shape: tuple[int, int, int],
        batch_size: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> torch.Tensor | None:
        if plucker is None:
            return None
        if plucker.ndim == 4:
            plucker = plucker.unsqueeze(0)
        plucker = plucker.to(device=device, dtype=dtype)
        if plucker.shape[0] == 1 and batch_size > 1:
            plucker = plucker.expand(batch_size, -1, -1, -1, -1)
        camera_hidden_states = self.plucker_embedder(plucker)
        expected_tokens = spatial_shape[0] * spatial_shape[1] * spatial_shape[2]
        camera_hidden_states = self._match_tokens(camera_hidden_states, expected_tokens)

        # Fuse per-pixel ray-direction map via raymap_embedder only on the
        # non-plucker path. NVlabs skips the absmap/raymap embedder whenever
        # chunk-plucker post-attention injection is enabled.
        if spatial_raymap is not None and not self.config.use_chunk_plucker_post_attn:
            if spatial_raymap.ndim == 4:  # [C, F, H, W] → [1, C, F, H, W]
                spatial_raymap = spatial_raymap.unsqueeze(0)
            spatial_raymap = spatial_raymap.to(device=device, dtype=dtype)
            if spatial_raymap.shape[0] == 1 and batch_size > 1:
                spatial_raymap = spatial_raymap.expand(batch_size, -1, -1, -1, -1)
            ray_hidden = self._match_tokens(self.raymap_embedder(spatial_raymap), expected_tokens)
            camera_hidden_states = camera_hidden_states + ray_hidden

        return camera_hidden_states

    def _unpatchify(self, hidden_states: torch.Tensor, spatial_shape: tuple[int, int, int]) -> torch.Tensor:
        batch_size = hidden_states.shape[0]
        frames, height, width = spatial_shape
        patch_frames, patch_height, patch_width = self.patch_size
        hidden_states = hidden_states.reshape(
            batch_size,
            frames,
            height,
            width,
            patch_frames,
            patch_height,
            patch_width,
            self._latent_channels,
        )
        hidden_states = torch.einsum("nfhwopqc->ncfohpwq", hidden_states)
        return hidden_states.reshape(
            batch_size,
            self._latent_channels,
            frames * patch_frames,
            height * patch_height,
            width * patch_width,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        timestep: torch.Tensor | float | int,
        *,
        encoder_hidden_states: torch.Tensor | None = None,
        encoder_attention_mask: torch.Tensor | None = None,
        camera_hidden_states: torch.Tensor | None = None,
        plucker: torch.Tensor | None = None,
        raymap: torch.Tensor | None = None,
        spatial_raymap: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if hidden_states.ndim != 5:
            raise ValueError("Sana-WM transformer expects latent input shaped [B, C, F, H, W].")
        batch_size = hidden_states.shape[0]
        latent_shape = hidden_states.shape[2:]
        hidden_states, spatial_shape = self.x_embedder.project_with_shape(hidden_states)
        rotary_emb = None
        if self.config.pos_embed_type == "wan_rope":
            rotary_emb = self.rope(spatial_shape, hidden_states.device)
        else:
            hidden_states = hidden_states + self._positional_embedding(
                hidden_states.shape[1], hidden_states.dtype, hidden_states.device
            )

        encoder_hidden_states = self.y_embedder(
            encoder_hidden_states.to(device=hidden_states.device) if encoder_hidden_states is not None else None,
            batch_size=batch_size,
            dtype=hidden_states.dtype,
        )
        encoder_hidden_states = self.attention_y_norm(encoder_hidden_states)
        # No mask means "attend to every prompt token"; keep it ``None`` so
        # cross-attention takes the mask-free attention path rather than
        # building an all-ones mask the backend would have to reshape per block.
        if encoder_attention_mask is not None:
            encoder_attention_mask = encoder_attention_mask.to(device=hidden_states.device)
            if encoder_attention_mask.shape != encoder_hidden_states.shape[:2]:
                raise ValueError(
                    "Sana-WM encoder_attention_mask must match text tokens "
                    f"{tuple(encoder_hidden_states.shape[:2])}, got {tuple(encoder_attention_mask.shape)}."
                )

        if not torch.is_tensor(timestep):
            timestep = torch.tensor([timestep], device=hidden_states.device, dtype=torch.float32)
        timestep = timestep.to(device=hidden_states.device)
        if timestep.ndim == 0:
            timestep = timestep.expand(batch_size)
        elif timestep.ndim == 1 and timestep.shape[0] == 1 and batch_size > 1:
            timestep = timestep.expand(batch_size)
        # Per-frame timestep contract: a rank > 1 ``timestep`` (``(B, 1, F)``
        # from the LTX flow-matching sampler) embeds on the flattened token axis
        # and unflattens back, giving ``(B, 1, F, D)`` / ``(B, 1, F, 6*D)``.
        # SanaWmBlock and SanaWmFinalLayer dispatch on ``ndim > 2`` to the
        # frame-aware paths that broadcast modulation over spatial tokens.
        timestep_shape = tuple(timestep.shape)
        time_embed = self.t_embedder(timestep)  # (numel, D)
        _t_silu = self.t_block[0](time_embed)
        timestep_modulation = self.t_block[1](_t_silu).to(hidden_states.dtype)
        if len(timestep_shape) > 1:
            time_embed = time_embed.unflatten(0, timestep_shape)
            timestep_modulation = timestep_modulation.unflatten(0, timestep_shape)

        if camera_hidden_states is None:
            camera_hidden_states = self._camera_hidden_states_from_conditions(
                plucker=plucker,
                spatial_raymap=spatial_raymap,
                spatial_shape=spatial_shape,
                batch_size=batch_size,
                dtype=hidden_states.dtype,
                device=hidden_states.device,
            )

        # Build (B, T_latent, 20) camera_conditions from the raw raymap, then
        # derive the UCPE geometry ONCE for the whole forward. The ray grid,
        # its SE(3) inverse and the camera RoPE tables are identical in every
        # block, so building them per block cost ~2400 rebuilds per request
        # (20 blocks x 60 steps x 2 CFG branches). When the caller passes
        # pre-embedded camera_hidden_states without raymap, the cam branch
        # becomes a no-op (main-branch only).
        cam_geometry: SanaWmCamGeometry | None = None
        if raymap is not None:
            # ``_pack_camera_conditions`` emits an unbatched (T_latent, 20).
            camera_conditions = raymap.unsqueeze(0)
            if batch_size > 1:
                camera_conditions = camera_conditions.expand(batch_size, -1, -1)
            camera_conditions = camera_conditions.to(device=hidden_states.device, dtype=hidden_states.dtype)
            # cam_head_dim comes from the attention module rather than being
            # re-derived here: __init__ already asserts it equals the main
            # head_dim and is consistent across blocks.
            attn = self.blocks[0].attn
            cam_geometry = prepare_cam_geometry(
                camera_conditions=camera_conditions,
                spatial_shape=spatial_shape,
                patch_size=self.patch_size,
                head_dim=attn.cam_head_dim,
                rotary_emb=attn._ucpe_rotary_freqs(rotary_emb),
            )

        for block in self.blocks:
            hidden_states = block(
                hidden_states,
                encoder_hidden_states,
                timestep_modulation,
                spatial_shape,
                rotary_emb,
                camera_hidden_states,
                cam_geometry,
                encoder_attention_mask,
            )

        hidden_states = self.final_layer(hidden_states, time_embed.to(hidden_states.dtype), spatial_shape)
        hidden_states = self._unpatchify(hidden_states, spatial_shape)
        if hidden_states.shape[2:] != latent_shape:
            hidden_states = F.interpolate(hidden_states, size=latent_shape, mode="trilinear", align_corners=False)
        return hidden_states

    def load_weights(self, weights: Iterable[tuple[str, Any]]) -> set[str]:
        """Stream checkpoint tensors into the eagerly-built modules.

        Follows the wan2_2 idiom: one ``named_parameters`` lookup table plus the
        per-tensor ``weight_loader``, which narrows full checkpoint tensors to
        the TP-local shard at copy time for both vLLM parallel layers and the
        plain parameters marked by ``_shard_param_across_tp``. The source tensor
        is dropped each iteration — no copy of the checkpoint is retained.
        """
        params_dict = dict(self.named_parameters())
        buffers_dict = dict(self.named_buffers())
        loaded: set[str] = set()
        unmapped: list[str] = []
        duplicates: list[str] = []

        for source_name, tensor in weights:
            # Weights ship in the standard diffusers ``transformer/`` layout, so
            # the checkpoint key is already the module-local parameter name.
            if source_name in loaded:
                duplicates.append(source_name)
                continue
            if not isinstance(tensor, torch.Tensor):
                raise TypeError(f"Sana-WM weight {source_name!r} must be a torch.Tensor, got {type(tensor).__name__}.")
            target = params_dict.get(source_name)
            if target is None:
                target = buffers_dict.get(source_name)
            if target is None:
                unmapped.append(source_name)
                continue
            weight_loader = getattr(target, "weight_loader", None)
            if callable(weight_loader):
                # vLLM parallel layers and the plain params marked by
                # `_shard_param_across_tp` narrow to the local shard here.
                weight_loader(target, tensor)
            else:
                if tuple(target.shape) != tuple(tensor.shape):
                    raise ValueError(
                        f"Sana-WM weight shape mismatch for {source_name}: "
                        f"expected {tuple(target.shape)}, got {tuple(tensor.shape)}."
                    )
                with torch.no_grad():
                    target.copy_(tensor.to(device=target.device, dtype=target.dtype))
            loaded.add(source_name)

        if unmapped or duplicates:
            details = []
            if unmapped:
                details.append(f"unmapped={unmapped[:10]}")
            if duplicates:
                details.append(f"duplicates={duplicates[:10]}")
            raise ValueError("Invalid SANA-WM Stage-1 checkpoint keys: " + "; ".join(details))
        return loaded
