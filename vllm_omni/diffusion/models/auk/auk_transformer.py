# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""The AuK audio DiT as a self-contained inference module.

Two phases. ``num_layers`` double-stream blocks attend jointly over text and
audio while keeping each stream's residual separate, then
``num_single_layers`` single-stream blocks run over the concatenated
``[text | ref audio | target audio]`` sequence. Only the target-audio slice is
projected back to latent space, so the module predicts a flow-matching
velocity for the target frames and :func:`sample_latents` integrates it.

Parameter names and shapes match the AuK reference backbone, so a released
checkpoint loads with ``load_state_dict(strict=True)`` once the
``transformer.`` prefix is stripped (see :func:`dit_state_dict`).

Deliberate differences from the reference, all inference-only:

* ``x_transformers`` and ``torchdiffeq`` are gone. Rotary embeddings are
  computed here (interleaved GPT-J pairs, no xpos scaling) and the ODE is an
  explicit Euler loop.
* Dropout, activation checkpointing, the ``flash_attn`` backend switch and the
  zero-init of the modulation layers are training concerns and are dropped.
* The timestep sinusoid is cast to the time MLP's weight dtype rather than to
  the timestep's own dtype, so an fp32 timestep works against half-precision
  weights outside autocast.
* ``attn_mask_enabled`` defaults to ``True``, the value the released config
  sets, rather than the reference's ``False``.
"""

import math
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.attention import SDPBackend, sdpa_kernel

__all__ = ["AuKStepContext", "AuKTransformer", "build_time_grid", "dit_state_dict", "sample_latents"]


def _sdpa(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, bias: torch.Tensor | None) -> torch.Tensor:
    """The single attention call site, so the backend can be swapped in one place.

    ``q``/``k``/``v`` are ``[B, H, N, D]``. ``bias`` is an additive key-padding
    bias ``[B, 1, 1, K]`` (see :func:`_key_padding_bias`) or ``None``. Passing
    it unexpanded lets SDPA broadcast it inside a fused kernel (cuDNN on
    Hopper) instead of materializing a ``[B, H, Q, K]`` mask per layer.
    """
    if bias is None or not q.is_cuda:
        return F.scaled_dot_product_attention(q, k, v, attn_mask=bias, dropout_p=0.0, is_causal=False)
    # With a bias, SDPA's default choice here is the memory-efficient kernel,
    # about 3x slower than cuDNN's at these lengths on Hopper.
    with sdpa_kernel(_MASKED_SDPA_PRIORITY, set_priority=True):
        return F.scaled_dot_product_attention(q, k, v, attn_mask=bias, dropout_p=0.0, is_causal=False)


_MASKED_SDPA_PRIORITY = [
    SDPBackend.CUDNN_ATTENTION,
    SDPBackend.FLASH_ATTENTION,
    SDPBackend.EFFICIENT_ATTENTION,
    SDPBackend.MATH,
]


def _key_padding_bias(mask: torch.Tensor | None, dtype: torch.dtype) -> torch.Tensor | None:
    """Additive attention bias ``[B, 1, 1, K]`` from a boolean key mask ``[B, K]``.

    Valid keys add 0 and padded keys add ``-inf``, which is exactly what SDPA
    does with a boolean mask internally.
    """
    if mask is None:
        return None
    bias = torch.zeros(mask.shape, dtype=dtype, device=mask.device).masked_fill(~mask, float("-inf"))
    return bias[:, None, None, :]


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotate interleaved GPT-J pairs: ``(x1, x2) -> (-x2, x1)``."""
    pairs = x.reshape(*x.shape[:-1], -1, 2)
    x1, x2 = pairs.unbind(dim=-1)
    return torch.stack((-x2, x1), dim=-1).flatten(-2)


def _apply_rope_cos_sin(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Rotate ``x`` ``[B, H, N, D]`` in fp32 with precomputed ``cos``/``sin`` of the frequencies."""
    out_dtype = x.dtype
    x32 = x.float()
    out = x32 * cos + _rotate_half(x32) * sin
    return out.to(out_dtype)


def _apply_rope(x: torch.Tensor, freqs: torch.Tensor) -> torch.Tensor:
    """Apply rotary frequencies ``[1, 1, N, D]`` to ``x`` ``[B, H, N, D]`` in fp32."""
    return _apply_rope_cos_sin(x, freqs.cos(), freqs.sin())


RopeCache = tuple[torch.Tensor, torch.Tensor]


def _rope_cos_sin(freqs: torch.Tensor) -> RopeCache:
    return freqs.cos(), freqs.sin()


class Rotary(nn.Module):
    """Rotary position frequencies for one stream, evaluated in fp32.

    The frequencies live in a buffer so that a released checkpoint's own copy
    of them loads with ``strict=True``. They are recomputed whenever the
    module has been cast below fp32: rounding either the frequencies or the
    integer positions to bf16 wrecks the phase at long positions.
    """

    def __init__(self, dim: int, base: float = 10000.0) -> None:
        super().__init__()
        if dim % 2 != 0:
            raise ValueError(f"rotary dim must be even, got {dim}")
        self.dim = dim
        self.base = base
        self.register_buffer("inv_freq", self._frequencies(torch.device("cpu")))

    def _frequencies(self, device: torch.device) -> torch.Tensor:
        exponents = torch.arange(0, self.dim, 2, device=device, dtype=torch.float32) / self.dim
        return 1.0 / (self.base**exponents)

    def forward(self, seq_len: int, mask: torch.Tensor | None = None) -> torch.Tensor:
        """Return rotary frequencies, compressing positions across padding."""
        inv_freq = self.inv_freq
        if inv_freq.dtype != torch.float32:
            inv_freq = self._frequencies(inv_freq.device)
        if mask is None:
            pos = torch.arange(seq_len, device=inv_freq.device, dtype=torch.float32)[None]
        else:
            pos = (mask.to(torch.int32).cumsum(dim=1) - 1).clamp_min(0).to(torch.float32)
        freqs = pos.unsqueeze(-1) * inv_freq
        return torch.stack((freqs, freqs), dim=-1).flatten(-2).unsqueeze(1)


class TimeEmbedding(nn.Module):
    """Sinusoidal timestep features followed by a two-layer MLP."""

    def __init__(self, dim: int, freq_embed_dim: int = 256) -> None:
        super().__init__()
        self.freq_embed_dim = freq_embed_dim
        self.time_mlp = nn.Sequential(nn.Linear(freq_embed_dim, dim), nn.SiLU(), nn.Linear(dim, dim))

    def forward(self, timestep: torch.Tensor, scale: float = 1000.0) -> torch.Tensor:
        half = self.freq_embed_dim // 2
        decay = math.log(10000) / (half - 1)
        freqs = torch.exp(torch.arange(half, device=timestep.device, dtype=torch.float32) * -decay)
        angles = scale * timestep.unsqueeze(1) * freqs.unsqueeze(0)
        hidden = torch.cat((angles.sin(), angles.cos()), dim=-1)
        return self.time_mlp(hidden.to(self.time_mlp[0].weight.dtype))


class ConvPosEmbedding(nn.Module):
    """Depthwise-grouped conv stack that adds local position information."""

    def __init__(self, dim: int, kernel_size: int = 31, groups: int = 16) -> None:
        super().__init__()
        if kernel_size % 2 == 0:
            raise ValueError(f"kernel_size must be odd, got {kernel_size}")
        padding = kernel_size // 2
        self.conv1d = nn.Sequential(
            nn.Conv1d(dim, dim, kernel_size, groups=groups, padding=padding),
            nn.Mish(),
            nn.Conv1d(dim, dim, kernel_size, groups=groups, padding=padding),
            nn.Mish(),
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        keep = None if mask is None else mask.unsqueeze(1)
        x = x.transpose(1, 2)
        if keep is not None:
            x = x.masked_fill(~keep, 0.0)
        for layer in self.conv1d:
            x = layer(x)
            # Re-zero padding after each convolution so it cannot bleed into
            # valid frames through the kernel window.
            if keep is not None and isinstance(layer, nn.Conv1d):
                x = x.masked_fill(~keep, 0.0)
        return x.transpose(1, 2)


class AudioEmbedding(nn.Module):
    """Project audio latents to model width and add conv position information."""

    def __init__(self, latent_dim: int, dim: int) -> None:
        super().__init__()
        self.linear = nn.Linear(latent_dim, dim)
        self.conv_pos_embed = ConvPosEmbedding(dim)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        x = self.linear(x)
        return self.conv_pos_embed(x, mask=mask) + x


class AdaLayerNorm(nn.Module):
    """Timestep-conditioned modulation for a block: six chunks from one projection."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.silu = nn.SiLU()
        self.linear = nn.Linear(dim, dim * 6)
        self.norm = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)

    def forward(
        self, x: torch.Tensor, emb: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.linear(self.silu(emb)).chunk(6, dim=1)
        x = self.norm(x) * (1 + scale_msa[:, None]) + shift_msa[:, None]
        return x, gate_msa, shift_mlp, scale_mlp, gate_mlp


class AdaLayerNormFinal(nn.Module):
    """Timestep-conditioned modulation before the output projection."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.silu = nn.SiLU()
        self.linear = nn.Linear(dim, dim * 2)
        self.norm = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)

    def forward(self, x: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        scale, shift = self.linear(self.silu(emb)).chunk(2, dim=1)
        return self.norm(x) * (1 + scale)[:, None, :] + shift[:, None, :]


class FeedForward(nn.Module):
    """SwiGLU feed-forward with the gate projection fused into ``linear_in``."""

    def __init__(self, dim: int, mult: float = 4.0) -> None:
        super().__init__()
        inner_dim = int(dim * mult)
        self.linear_in = nn.Linear(dim, inner_dim * 2, bias=False)
        self.linear_out = nn.Linear(inner_dim, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate, up = self.linear_in(x).chunk(2, dim=-1)
        return self.linear_out(F.silu(gate) * up)


class Attention(nn.Module):
    """Self-attention with per-head QK RMSNorm and rotary positions.

    ``to_out`` stays a ``ModuleList`` because the released checkpoints name the
    output projection ``to_out.0``; the reference's trailing dropout carries no
    parameters and is dropped.
    """

    def __init__(self, dim: int, heads: int, dim_head: int, attn_mask_enabled: bool = True) -> None:
        super().__init__()
        self.heads = heads
        self.dim_head = dim_head
        self.attn_mask_enabled = attn_mask_enabled
        inner_dim = heads * dim_head
        self.to_qkv = nn.Linear(dim, 3 * inner_dim)
        self.q_norm = nn.RMSNorm(dim_head, elementwise_affine=True)
        self.k_norm = nn.RMSNorm(dim_head, elementwise_affine=True)
        self.to_out = nn.ModuleList([nn.Linear(inner_dim, dim)])

    def _merge_heads(self, x: torch.Tensor) -> torch.Tensor:
        return x.transpose(1, 2).reshape(x.shape[0], -1, self.heads * self.dim_head)

    def _qkv(
        self,
        packed: torch.Tensor,
        q_norm: nn.Module,
        k_norm: nn.Module,
        rope: RopeCache | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Split a packed QKV projection into heads, RMSNorm Q/K, then rotate.

        ``rope`` is the ``(cos, sin)`` pair of the rotary frequencies.
        """
        q, k, v = (t.view(t.shape[0], -1, self.heads, self.dim_head).transpose(1, 2) for t in packed.chunk(3, dim=-1))
        q = q_norm(q)
        k = k_norm(k)
        if rope is not None:
            q = _apply_rope_cos_sin(q, *rope)
            k = _apply_rope_cos_sin(k, *rope)
        return q, k, v

    def _project(self, x: torch.Tensor, rope: RopeCache | None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self._qkv(self.to_qkv(x), self.q_norm, self.k_norm, rope)

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None,
        rope: RopeCache | None = None,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """``bias`` is the precomputed key-padding bias for ``mask``; built here when absent."""
        q, k, v = self._project(x, rope)
        if bias is None and self.attn_mask_enabled:
            bias = _key_padding_bias(mask, q.dtype)
        out = self._merge_heads(_sdpa(q, k, v, bias).to(q.dtype))
        out = self.to_out[0](out)
        if mask is not None:
            out = out.masked_fill(~mask.unsqueeze(-1), 0.0)
        return out


def joint_key_mask(mask: torch.Tensor | None, c_mask: torch.Tensor | None, text_len: int) -> torch.Tensor | None:
    """Key mask of joint attention over ``[audio | text]``; ``None`` when the audio has no mask."""
    if mask is None:
        return None
    if c_mask is not None:
        return torch.cat([mask, c_mask], dim=1)
    return F.pad(mask, (0, text_len), value=True)


class JointAttention(Attention):
    """Attention over the audio and text streams jointly, with separate projections."""

    def __init__(self, dim: int, heads: int, dim_head: int, context_dim: int, attn_mask_enabled: bool = True) -> None:
        super().__init__(dim, heads, dim_head, attn_mask_enabled)
        inner_dim = heads * dim_head
        self.to_qkv_c = nn.Linear(context_dim, 3 * inner_dim)
        self.c_q_norm = nn.RMSNorm(dim_head, elementwise_affine=True)
        self.c_k_norm = nn.RMSNorm(dim_head, elementwise_affine=True)
        self.to_out_c = nn.Linear(inner_dim, context_dim)

    def _project_context(self, c: torch.Tensor, c_rope: RopeCache | None) -> tuple[torch.Tensor, ...]:
        return self._qkv(self.to_qkv_c(c), self.c_q_norm, self.c_k_norm, c_rope)

    def forward(  # type: ignore[override]  # joint attention takes the context stream too
        self,
        x: torch.Tensor,
        c: torch.Tensor,
        mask: torch.Tensor | None = None,
        rope: RopeCache | None = None,
        c_rope: RopeCache | None = None,
        c_mask: torch.Tensor | None = None,
        bias: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """``bias`` is the precomputed :func:`joint_key_mask` bias; built here when absent."""
        audio_len = x.shape[1]
        q, k, v = self._project(x, rope)
        c_q, c_k, c_v = self._project_context(c, c_rope)

        q = torch.cat([q, c_q], dim=2)
        k = torch.cat([k, c_k], dim=2)
        v = torch.cat([v, c_v], dim=2)

        if bias is None and self.attn_mask_enabled:
            bias = _key_padding_bias(joint_key_mask(mask, c_mask, c.shape[1]), q.dtype)

        out = self._merge_heads(_sdpa(q, k, v, bias).to(q.dtype))
        x_out = self.to_out[0](out[:, :audio_len])
        c_out = self.to_out_c(out[:, audio_len:])

        if mask is not None:
            x_out = x_out.masked_fill(~mask.unsqueeze(-1), 0.0)
        if c_mask is not None:
            c_out = c_out.masked_fill(~c_mask.unsqueeze(-1), 0.0)
        return x_out, c_out


class DoubleBlock(nn.Module):
    """MM-DiT block: joint attention, separate modulation and feed-forward per stream."""

    def __init__(self, dim: int, heads: int, dim_head: int, ff_mult: float, attn_mask_enabled: bool) -> None:
        super().__init__()
        self.attn_norm_c = AdaLayerNorm(dim)
        self.attn_norm_x = AdaLayerNorm(dim)
        self.attn = JointAttention(
            dim=dim, heads=heads, dim_head=dim_head, context_dim=dim, attn_mask_enabled=attn_mask_enabled
        )
        self.ff_norm_c = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.ff_c = FeedForward(dim=dim, mult=ff_mult)
        self.ff_norm_x = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.ff_x = FeedForward(dim=dim, mult=ff_mult)

    def forward(
        self,
        x: torch.Tensor,
        c: torch.Tensor,
        t: torch.Tensor,
        mask: torch.Tensor | None,
        rope: RopeCache,
        c_rope: RopeCache,
        c_mask: torch.Tensor | None,
        bias: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        norm_c, c_gate_msa, c_shift_mlp, c_scale_mlp, c_gate_mlp = self.attn_norm_c(c, t)
        norm_x, x_gate_msa, x_shift_mlp, x_scale_mlp, x_gate_mlp = self.attn_norm_x(x, t)

        x_attn, c_attn = self.attn(x=norm_x, c=norm_c, mask=mask, rope=rope, c_rope=c_rope, c_mask=c_mask, bias=bias)

        c = c + c_gate_msa[:, None] * c_attn
        norm_c = self.ff_norm_c(c) * (1 + c_scale_mlp[:, None]) + c_shift_mlp[:, None]
        c = c + c_gate_mlp[:, None] * self.ff_c(norm_c)

        x = x + x_gate_msa[:, None] * x_attn
        norm_x = self.ff_norm_x(x) * (1 + x_scale_mlp[:, None]) + x_shift_mlp[:, None]
        x = x + x_gate_mlp[:, None] * self.ff_x(norm_x)
        return c, x


class SingleBlock(nn.Module):
    """DiT block over the concatenated text and audio sequence."""

    def __init__(self, dim: int, heads: int, dim_head: int, ff_mult: float, attn_mask_enabled: bool) -> None:
        super().__init__()
        self.attn_norm = AdaLayerNorm(dim)
        self.attn = Attention(dim=dim, heads=heads, dim_head=dim_head, attn_mask_enabled=attn_mask_enabled)
        self.ff_norm = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.ff = FeedForward(dim=dim, mult=ff_mult)

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        mask: torch.Tensor | None,
        rope: RopeCache,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        norm, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.attn_norm(x, t)
        x = x + gate_msa[:, None] * self.attn(x=norm, mask=mask, rope=rope, bias=bias)
        norm = self.ff_norm(x) * (1 + scale_mlp[:, None]) + shift_mlp[:, None]
        return x + gate_mlp[:, None] * self.ff(norm)


class AuKTransformer(nn.Module):
    """Flow-matching velocity predictor for AuK audio latents.

    The forward pass is :meth:`prepare` (per request) followed by
    :meth:`step` (per timestep); an Euler loop calls the two separately.

    Args:
        dim: Model width.
        heads: Attention heads.
        dim_head: Per-head width.
        ff_mult: Feed-forward expansion factor.
        latent_dim: Audio VAE latent channels, the input and output width.
        text_hidden_dim: Width of the pre-encoded text hidden states.
        num_layers: Double-stream (joint attention) block count.
        num_single_layers: Single-stream block count.
        attn_mask_enabled: Apply padding masks inside attention. When ``False``
            attention runs unmasked, though padded outputs are still zeroed,
            which is what the reference does with the flag off.
    """

    # Regional torch.compile targets (see vllm_omni.diffusion.compile).
    _repeated_blocks = ["DoubleBlock", "SingleBlock"]

    def __init__(
        self,
        *,
        dim: int,
        heads: int,
        dim_head: int,
        ff_mult: float,
        latent_dim: int,
        text_hidden_dim: int,
        num_layers: int,
        num_single_layers: int,
        attn_mask_enabled: bool = True,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.latent_dim = latent_dim
        self.attn_mask_enabled = attn_mask_enabled
        self.text_cond: torch.Tensor | None = None
        self.text_uncond: torch.Tensor | None = None

        self.time_embed = TimeEmbedding(dim)
        self.txt_norm = nn.RMSNorm(dim, elementwise_affine=True)
        self.txt_proj = nn.Linear(text_hidden_dim, dim)
        self.audio_embed = AudioEmbedding(latent_dim, dim)
        self.rotary_embed = Rotary(dim_head)

        self.transformer_blocks = nn.ModuleList(
            [
                DoubleBlock(
                    dim=dim, heads=heads, dim_head=dim_head, ff_mult=ff_mult, attn_mask_enabled=attn_mask_enabled
                )
                for _ in range(num_layers)
            ]
        )
        self.single_transformer_blocks = nn.ModuleList(
            [
                SingleBlock(
                    dim=dim, heads=heads, dim_head=dim_head, ff_mult=ff_mult, attn_mask_enabled=attn_mask_enabled
                )
                for _ in range(num_single_layers)
            ]
        )

        self.norm_out = AdaLayerNormFinal(dim)
        self.proj_out = nn.Linear(dim, latent_dim)

    def project_text(self, text: torch.Tensor) -> torch.Tensor:
        """Project LLM hidden states ``[B, nt, text_hidden_dim]`` to model width."""
        return self.txt_norm(self.txt_proj(text))

    def clear_cache(self) -> None:
        """Drop the cached text projection. Call this when the conditioning changes."""
        self.text_cond = None
        self.text_uncond = None

    def _embed_prompt(self, ref: torch.Tensor, ref_mask: torch.Tensor | None, drop: bool) -> torch.Tensor:
        """Embed the reference prompt, zeroed for the audio-dropped CFG branch."""
        if drop:
            ref = torch.zeros_like(ref)
        return self.audio_embed(ref, mask=ref_mask)

    def prepare(
        self,
        text: torch.Tensor,
        *,
        target_len: int,
        mask: torch.Tensor | None = None,
        c_mask: torch.Tensor | None = None,
        ref: torch.Tensor | None = None,
        ref_mask: torch.Tensor | None = None,
        drop_audio_cond: bool = False,
        drop_text: bool = False,
        cfg_infer: bool = False,
        cache: bool = False,
    ) -> "AuKStepContext":
        """Everything a denoise step needs that does not depend on ``x`` or the time.

        An Euler loop over fixed conditioning calls this once and then
        :meth:`step` per timestep, so the text projection, the reference
        prompt embedding, the padding masks and biases and the rotary tables
        are not recomputed on every step. Arguments are those of
        :meth:`forward`; ``target_len`` is the target frame count ``n``.
        """
        batch = text.shape[0]
        if c_mask is None:
            # Padding rows are all-zero in the text encoder's output.
            c_mask = text.abs().sum(-1) > 0
        if ref is not None and ref.shape[1] == 0:
            ref = None

        if cfg_infer:
            if cache and self.text_cond is not None:
                c_cond, c_uncond = self.text_cond, self.text_uncond
            else:
                c_cond = self.project_text(text)
                # The uncond branch drops the text, which is exactly a zeroed projection.
                c_uncond = torch.zeros_like(c_cond)
                if cache:
                    self.text_cond, self.text_uncond = c_cond, c_uncond
            c = torch.cat((c_cond, c_uncond), dim=0)
            prompt = None
            if ref is not None:
                prompt = torch.cat(
                    (self._embed_prompt(ref, ref_mask, False), self._embed_prompt(ref, ref_mask, True)), dim=0
                )
        else:
            c = self.project_text(text)
            if drop_text:
                c = torch.zeros_like(c)
            prompt = None if ref is None else self._embed_prompt(ref, ref_mask, drop_audio_cond)

        audio_mask = mask
        if ref is not None and (mask is not None or ref_mask is not None):
            if mask is None:
                mask = torch.ones((batch, target_len), dtype=torch.bool, device=ref.device)
            if ref_mask is None:
                ref_mask = torch.ones(ref.shape[:2], dtype=torch.bool, device=ref.device)
            audio_mask = torch.cat([ref_mask, mask], dim=1)

        branches = 2 if cfg_infer else 1
        if branches == 2:
            c_mask = torch.cat((c_mask, c_mask), dim=0)
            if audio_mask is not None:
                audio_mask = torch.cat((audio_mask, audio_mask), dim=0)

        prompt_len = 0 if prompt is None else prompt.shape[1]
        text_len = c.shape[1]
        audio_len = prompt_len + target_len
        single_mask = None if audio_mask is None else torch.cat([c_mask, audio_mask], dim=1)
        joint_bias = single_bias = None
        if self.attn_mask_enabled:
            joint_bias = _key_padding_bias(joint_key_mask(audio_mask, c_mask, text_len), c.dtype)
            single_bias = _key_padding_bias(single_mask, c.dtype)

        return AuKStepContext(
            c=c,
            prompt=prompt,
            target_mask=mask,
            c_mask=c_mask,
            audio_mask=audio_mask,
            single_mask=single_mask,
            joint_bias=joint_bias,
            single_bias=single_bias,
            rope_audio=_rope_cos_sin(self.rotary_embed(audio_len, audio_mask)),
            rope_text=_rope_cos_sin(self.rotary_embed(text_len, c_mask)),
            rope_single=_rope_cos_sin(self.rotary_embed(text_len + audio_len, single_mask)),
            branches=branches,
        )

    def step(self, x: torch.Tensor, time: torch.Tensor, ctx: "AuKStepContext") -> torch.Tensor:
        """Predict the velocity of ``x`` at ``time`` under a :meth:`prepare` context.

        The target embedding does not depend on the CFG branch, so it is
        computed once and shared by both branches.
        """
        if time.ndim == 0:
            time = time.repeat(x.shape[0])
        t = self.time_embed(time)
        target = self.audio_embed(x, mask=ctx.target_mask)
        if ctx.branches == 2:
            t = torch.cat((t, t), dim=0)
            target = torch.cat((target, target), dim=0)
        audio = target if ctx.prompt is None else torch.cat([ctx.prompt, target], dim=1)
        c = ctx.c

        for block in self.transformer_blocks:
            c, audio = block(
                audio,
                c,
                t,
                mask=ctx.audio_mask,
                rope=ctx.rope_audio,
                c_rope=ctx.rope_text,
                c_mask=ctx.c_mask,
                bias=ctx.joint_bias,
            )

        h = torch.cat([c, audio], dim=1)
        for block in self.single_transformer_blocks:
            h = block(h, t, mask=ctx.single_mask, rope=ctx.rope_single, bias=ctx.single_bias)

        h = h[:, h.shape[1] - x.shape[1] :]
        return self.proj_out(self.norm_out(h, t))

    def forward(
        self,
        x: torch.Tensor,
        text: torch.Tensor,
        time: torch.Tensor,
        *,
        mask: torch.Tensor | None = None,
        c_mask: torch.Tensor | None = None,
        ref: torch.Tensor | None = None,
        ref_mask: torch.Tensor | None = None,
        drop_audio_cond: bool = False,
        drop_text: bool = False,
        cfg_infer: bool = False,
        cache: bool = False,
    ) -> torch.Tensor:
        """Predict the flow-matching velocity for the target frames of ``x``.

        Args:
            x: Noised target latents ``[B, n, latent_dim]``.
            text: Pre-encoded text hidden states ``[B, nt, text_hidden_dim]``.
            time: Flow-matching timestep, a scalar or ``[B]``.
            mask: Target padding mask ``[B, n]``, ``True`` where valid.
            c_mask: Text padding mask ``[B, nt]``. Inferred from all-zero text
                rows when omitted.
            ref: Reference prompt latents ``[B, np, latent_dim]``, prepended to
                the audio sequence. A zero-length ``ref`` counts as absent.
            ref_mask: Reference padding mask ``[B, np]``.
            drop_audio_cond: Zero the reference prompt, for the CFG uncond branch.
            drop_text: Zero the projected text, for the CFG uncond branch.
            cfg_infer: Run the cond and uncond branches as one batch of ``2B``,
                cond first. ``drop_audio_cond`` and ``drop_text`` are then
                driven per branch and ignored.
            cache: Reuse the projected text across calls, for an ODE loop over
                fixed conditioning. Only the ``cfg_infer`` path caches, as in
                the reference. Call :meth:`clear_cache` when the text changes.

        Returns:
            Velocity ``[B, n, latent_dim]``, or ``[2B, n, latent_dim]`` under
            ``cfg_infer``.
        """
        ctx = self.prepare(
            text,
            target_len=x.shape[1],
            mask=mask,
            c_mask=c_mask,
            ref=ref,
            ref_mask=ref_mask,
            drop_audio_cond=drop_audio_cond,
            drop_text=drop_text,
            cfg_infer=cfg_infer,
            cache=cache and cfg_infer,
        )
        return self.step(x, time, ctx)


@dataclass
class AuKStepContext:
    """Per-request state of the denoise step, built by :meth:`AuKTransformer.prepare`.

    Batch rows are ``branches`` copies of the request (cond first under CFG).
    ``target_mask`` covers the target frames only and has the request's batch
    size; the other masks cover their full sequences and every branch.
    """

    c: torch.Tensor
    prompt: torch.Tensor | None
    target_mask: torch.Tensor | None
    c_mask: torch.Tensor | None
    audio_mask: torch.Tensor | None
    single_mask: torch.Tensor | None
    joint_bias: torch.Tensor | None
    single_bias: torch.Tensor | None
    rope_audio: RopeCache
    rope_text: RopeCache
    rope_single: RopeCache
    branches: int

    def tensors(self) -> list[torch.Tensor | None]:
        """Every tensor field in a fixed order, rope pairs flattened."""
        return [
            self.c,
            self.prompt,
            self.target_mask,
            self.c_mask,
            self.audio_mask,
            self.single_mask,
            self.joint_bias,
            self.single_bias,
            *self.rope_audio,
            *self.rope_text,
            *self.rope_single,
        ]

    def copy_(self, other: "AuKStepContext") -> None:
        """Overwrite these tensors in place with ``other``'s, for CUDA graph static inputs."""
        if other.branches != self.branches:
            raise ValueError(f"AuK step context mismatch: {other.branches} vs {self.branches} branches")
        for dst, src in zip(self.tensors(), other.tensors(), strict=True):
            if (dst is None) != (src is None):
                raise ValueError("AuK step context mismatch: a mask is present in only one context")
            if dst is not None:
                dst.copy_(src)

    def clone(self) -> "AuKStepContext":
        """A copy whose tensors own fresh storage."""

        def _clone(value: torch.Tensor | None) -> torch.Tensor | None:
            return None if value is None else value.clone()

        return AuKStepContext(
            c=self.c.clone(),
            prompt=_clone(self.prompt),
            target_mask=_clone(self.target_mask),
            c_mask=_clone(self.c_mask),
            audio_mask=_clone(self.audio_mask),
            single_mask=_clone(self.single_mask),
            joint_bias=_clone(self.joint_bias),
            single_bias=_clone(self.single_bias),
            rope_audio=(self.rope_audio[0].clone(), self.rope_audio[1].clone()),
            rope_text=(self.rope_text[0].clone(), self.rope_text[1].clone()),
            rope_single=(self.rope_single[0].clone(), self.rope_single[1].clone()),
            branches=self.branches,
        )


def dit_state_dict(
    checkpoint: Mapping[str, torch.Tensor] | Iterable[tuple[str, torch.Tensor]],
    prefix: str = "transformer.",
) -> dict[str, torch.Tensor]:
    """Select the DiT tensors out of a full AuK checkpoint and strip ``prefix``.

    Released checkpoints wrap the backbone under ``transformer.`` alongside the
    text-encoder layer-fusion parameters, which this module does not own.
    """
    items = checkpoint.items() if isinstance(checkpoint, Mapping) else checkpoint
    return {name[len(prefix) :]: tensor for name, tensor in items if name.startswith(prefix)}


def build_time_grid(
    *,
    nfe: int,
    sway_sampling_coef: float | None,
    t_grid: list[float] | None,
    device: torch.device | str,
) -> torch.Tensor:
    """Build and validate the Euler schedule before a graph replay."""
    if t_grid is not None:
        timesteps = torch.tensor(t_grid, device="cpu", dtype=torch.float32)
    else:
        timesteps = torch.linspace(0, 1, nfe + 1, device="cpu", dtype=torch.float32)
        if sway_sampling_coef is not None:
            timesteps = timesteps + sway_sampling_coef * (torch.cos(math.pi / 2 * timesteps) - 1 + timesteps)
    if (
        timesteps.ndim != 1
        or timesteps.numel() < 2
        or not bool(torch.isfinite(timesteps).all() & torch.all(timesteps[1:] > timesteps[:-1]))
    ):
        raise ValueError(
            f"AuK sampling needs a strictly increasing time grid with at least two points; got {timesteps.tolist()}"
        )
    return timesteps.to(device)


@torch.no_grad()
def sample_latents(
    dit: AuKTransformer,
    *,
    text: torch.Tensor,
    c_mask: torch.Tensor | None,
    ref: torch.Tensor,
    ref_mask: torch.Tensor | None,
    gen_frames: int,
    nfe: int = 32,
    cfg_strength: float = 1.0,
    sway_sampling_coef: float | None = None,
    t_grid: list[float] | None = None,
    seed: int | None = None,
    generator: torch.Generator | None = None,
    latent_dim: int | None = None,
    device: torch.device | str | None = None,
    dtype: torch.dtype | None = None,
    sampler: Callable[..., torch.Tensor] | None = None,
) -> torch.Tensor:
    """Integrate the flow from noise to audio latents with explicit Euler steps.

    Single-request only: the batch dimension carries the CFG branches, not
    separate prompts, so no target padding mask is needed.

    Args:
        dit: The velocity model.
        text: Pre-encoded text hidden states ``[1, nt, text_hidden_dim]``.
        c_mask: Text padding mask ``[1, nt]``.
        ref: Reference prompt latents ``[1, np, latent_dim]``.
        ref_mask: Reference padding mask ``[1, np]``.
        gen_frames: Target latent frames to generate.
        nfe: Euler steps, ignored when ``t_grid`` is given.
        cfg_strength: Classifier-free guidance weight. Below ``1e-5`` the
            uncond branch is skipped entirely.
        sway_sampling_coef: Reshapes the uniform time grid towards ``t=0``.
        t_grid: Explicit timesteps, overriding ``nfe`` and the sway reshape.
        seed: Seeds the global RNG before drawing the noise. Ignored when
            ``generator`` is given.
        generator: Draws the initial noise from its own RNG, leaving the global
            one untouched.
        latent_dim: Latent channels; defaults to the model's.
        device: Device for the noise; defaults to the reference's.
        dtype: Dtype for the noise and the Euler accumulator; defaults to the
            reference's, which is the reference implementation's rule. Pass
            ``torch.float32`` explicitly when serving half-precision weights.
            Drawing the noise at bf16 instead of fp32 moves a 32-step CFG
            trajectory by 0.025 relative MSE, roughly 17 times what bf16
            weights themselves cost, because the draw consumes the generator
            differently and starts the ODE somewhere else. Keeping it fp32
            costs a few tens of KB and no measurable time.

    Returns:
        The target latents at ``t=1``, ``[1, gen_frames, latent_dim]``.
    """
    device = torch.device(device) if device is not None else ref.device
    dtype = dtype if dtype is not None else ref.dtype
    latent_dim = latent_dim if latent_dim is not None else dit.latent_dim

    if generator is None:
        if seed is not None:
            torch.manual_seed(seed)
        x = torch.randn(gen_frames, latent_dim, device=device, dtype=dtype).unsqueeze(0)
    else:
        x = (
            torch.randn(gen_frames, latent_dim, generator=generator, device=generator.device, dtype=dtype)
            .to(device)
            .unsqueeze(0)
        )

    t = build_time_grid(
        nfe=nfe,
        sway_sampling_coef=sway_sampling_coef,
        t_grid=t_grid,
        device=device,
    )
    if sampler is not None:
        try:
            for i in range(t.shape[0] - 1):
                velocity = sampler(
                    x=x,
                    text=text,
                    c_mask=c_mask,
                    ref=ref,
                    ref_mask=ref_mask,
                    timestep=t[i],
                    cfg_strength=cfg_strength,
                    new_request=i == 0,
                )
                x = x + (t[i + 1] - t[i]) * velocity
        finally:
            dit.clear_cache()
        return x
    return _sample_latents(
        dit,
        initial_latents=x,
        text=text,
        c_mask=c_mask,
        ref=ref,
        ref_mask=ref_mask,
        timesteps=t,
        cfg_strength=cfg_strength,
    )


def _sample_latents(
    dit: AuKTransformer,
    *,
    initial_latents: torch.Tensor,
    text: torch.Tensor,
    c_mask: torch.Tensor | None,
    ref: torch.Tensor,
    ref_mask: torch.Tensor | None,
    timesteps: torch.Tensor,
    cfg_strength: float,
) -> torch.Tensor:
    """Euler integration shared by eager sampling and CUDA graph capture."""
    x = initial_latents
    guided = cfg_strength >= 1e-5
    try:
        # The conditioning is fixed across the loop: prepare it once.
        ctx = dit.prepare(
            text,
            target_len=x.shape[1],
            c_mask=c_mask,
            ref=ref,
            ref_mask=ref_mask,
            cfg_infer=guided,
            cache=guided,
        )
        for i in range(timesteps.shape[0] - 1):
            v = dit.step(x, timesteps[i], ctx)
            if guided:
                v_cond, v_uncond = v.chunk(2, dim=0)
                v = v_cond + (v_cond - v_uncond) * cfg_strength
            x = x + (timesteps[i + 1] - timesteps[i]) * v
    finally:
        # The cached text projections belong to this request only; a failed
        # step must not leak them into the next one.
        dit.clear_cache()
    return x
