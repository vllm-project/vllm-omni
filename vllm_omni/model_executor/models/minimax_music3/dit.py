# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Condition encoder and flow-matching transformer for MiniMax Music 3.

Stage 1 turns the AR stage's conditioning frames into a DAC latent in two
steps. The condition encoder collapses the eight 4096-wide layer slices of a
frame into one 2048-wide vector and stretches the result from the AR frame
grid onto the vocoder's latent grid. The transformer is then solved as a
flow-matching velocity field: a fixed-step Euler integrator walks Gaussian
noise to the latent the vocoder expects, under classifier-free guidance.

Module names mirror the checkpoint's ``condition_encoder/`` and
``transformer/`` component folders key for key, with one deliberate
exception: the per-block ``to_q``/``to_k``/``to_v`` projections are fused into
a single ``to_qkv`` so each block runs one GEMM instead of three.
:func:`remap_transformer_state` performs that fusion, and is the only place
the checkpoint's layout is reinterpreted.

Attention runs through the project's own diffusion attention layer, so this
model picks up the same backend selection, sequence-parallel plumbing and
KV-quant policy as every other transformer in the tree. That layer resolves
its backend from the diffusion config, which an ``LLM_GENERATION`` stage such
as this one does not set; it tolerates that and falls through to the platform
default. If it cannot be built at all the block runs plain SDPA instead, a
choice made once when the blocks are built and never inside ``forward``.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from vllm.logger import init_logger

from .constants import (
    AR_HIDDEN_SIZE,
    BACKBONE_HIDDEN_SIZE,
    CONDITION_INPUT_HOP,
    CONDITION_INPUT_SAMPLE_RATE,
    CONDITION_OUTPUT_HOP,
    CONDITION_OUTPUT_SAMPLE_RATE,
    DEFAULT_DIT_CFG_SCALE,
    DEFAULT_DIT_STEPS,
    DIT_LATENT_CHANNELS,
    NUM_CODEBOOKS,
)

logger = init_logger(__name__)

__all__ = [
    "MiniMaxMusic3ConditionEncoder",
    "MiniMaxMusic3FlowMatchingDiT",
    "MiniMaxMusic3Transformer1DModel",
    "remap_transformer_state",
]

# Fixed by transformer/config.json. Declared rather than read from the file so
# the skeleton can be built before any checkpoint is on disk, which is what
# vLLM's memory profiler needs at startup.
_DIM = 2_048
_NUM_LAYERS = 36
_HEAD_DIM = 64
_FF_INNER_DIM = 8_192
_FOURIER_DIM = 256
_ROTARY_DIM = 32
_ROTARY_BASE = 10_000.0
_CONDITION_DIM = 2_048
# The solver feeds the transformer the current latent, a zero block of the
# same width, and the condition: 128 + 128 + 2048 = 2304.
_TRANSFORMER_IN_DIM = DIT_LATENT_CHANNELS * 2 + _CONDITION_DIM

# The prompt interpolation floor from the reference sampler. Sigma never
# reaches exactly zero, so the noise term never fully vanishes.
_SIGMA_FLOOR = 1e-6


class FourierFeatures(nn.Module):
    """Random-Fourier timestep features: ``[B, 1] -> [B, out_features]``."""

    def __init__(self, in_features: int, out_features: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.empty(out_features // 2, in_features))

    def forward(self, value: Tensor) -> Tensor:
        f = 2.0 * math.pi * value @ self.weight.T
        return torch.cat((f.cos(), f.sin()), dim=-1)


class TimestepEmbedding(nn.Module):
    """The checkpoint's ``time_embed``: Linear, SiLU, Linear."""

    def __init__(self, in_dim: int, dim: int) -> None:
        super().__init__()
        self.linear_1 = nn.Linear(in_dim, dim)
        self.act = nn.SiLU()
        self.linear_2 = nn.Linear(dim, dim)

    def forward(self, x: Tensor) -> Tensor:
        return self.linear_2(self.act(self.linear_1(x)))


def _rotate_half(x: Tensor) -> Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def _apply_rope(x: Tensor, rope_cos: Tensor, rope_sin: Tensor) -> Tensor:
    """Rotate the leading ``rope_cos.shape[-1]`` channels of ``[B, S, H, D]``.

    Only the first 32 of each head's 64 channels are rotated; the rest pass
    through untouched, which is what this checkpoint was trained with. The
    tables are cached at the exact sequence length, so there is nothing to
    slice off the front.
    """
    rot_dim = rope_cos.shape[-1]
    rotated = x[..., :rot_dim]
    rotated = rotated * rope_cos + _rotate_half(rotated) * rope_sin
    return torch.cat((rotated, x[..., rot_dim:]), dim=-1)


# Set to the failure reason the first time the native attention layer cannot be
# built, so the 36 blocks agree on one path and the reason is logged once.
_NATIVE_ATTENTION_UNAVAILABLE: str | None = None


def _build_native_attention(*, num_heads: int, head_size: int, softmax_scale: float, prefix: str) -> nn.Module | None:
    """Build the project's diffusion attention layer, or ``None`` to use SDPA.

    Anything that goes wrong here is a legitimate reason to run plain SDPA,
    and is reported once rather than once per block.
    """
    global _NATIVE_ATTENTION_UNAVAILABLE
    if _NATIVE_ATTENTION_UNAVAILABLE is not None:
        return None
    try:
        from vllm_omni.diffusion.attention.layer import Attention as DiffusionAttention

        return DiffusionAttention(
            num_heads=num_heads,
            head_size=head_size,
            causal=False,
            softmax_scale=softmax_scale,
            prefix=prefix,
            role="self",
            role_category="self",
        )
    except Exception as exc:  # noqa: BLE001 - any failure means: use SDPA
        _NATIVE_ATTENTION_UNAVAILABLE = repr(exc)
        logger.warning(
            "MiniMax Music 3 DiT could not build the diffusion attention layer "
            "(%s); running torch SDPA instead. The stage decodes in float32, "
            "for which the diffusion layer dispatches to SDPA anyway, so this "
            "changes performance bookkeeping and not the audio.",
            exc,
        )
        return None


class Attention(nn.Module):
    """Bidirectional self-attention with partial rotary position embeddings.

    Q, K and V are kept in ``[B, S, H, D]``, which is the layout the diffusion
    attention backends consume and return.
    """

    def __init__(self, dim: int, *, head_dim: int, prefix: str = "") -> None:
        super().__init__()
        self.num_heads = dim // head_dim
        self.head_dim = head_dim
        self.softmax_scale = float(head_dim**-0.5)
        # Fused from the checkpoint's separate to_q/to_k/to_v; see
        # remap_transformer_state for the concatenation order.
        self.to_qkv = nn.Linear(dim, dim * 3, bias=False)
        self.to_out = nn.Linear(dim, dim, bias=False)
        self.backend = _build_native_attention(
            num_heads=self.num_heads,
            head_size=head_dim,
            softmax_scale=self.softmax_scale,
            prefix=prefix,
        )
        self.backend_name = self.backend.attn_backend.get_name() if self.backend is not None else "TORCH_SDPA"

    def _attend(self, q: Tensor, k: Tensor, v: Tensor) -> Tensor:
        if self.backend is not None:
            return self.backend(q, k, v)
        q, k, v = (t.transpose(1, 2) for t in (q, k, v))
        out = F.scaled_dot_product_attention(q, k, v, is_causal=False, scale=self.softmax_scale)
        return out.transpose(1, 2)

    def forward(self, x: Tensor, rope_cos: Tensor, rope_sin: Tensor) -> Tensor:
        bsz, seq, dim = x.shape
        q, k, v = self.to_qkv(x).chunk(3, dim=-1)
        q = q.reshape(bsz, seq, self.num_heads, self.head_dim)
        k = k.reshape(bsz, seq, self.num_heads, self.head_dim)
        v = v.reshape(bsz, seq, self.num_heads, self.head_dim)
        q = _apply_rope(q, rope_cos, rope_sin)
        k = _apply_rope(k, rope_cos, rope_sin)
        out = self._attend(q, k, v)
        return self.to_out(out.contiguous().reshape(bsz, seq, dim))


class TransformerBlock(nn.Module):
    """Pre-norm attention followed by a pre-norm gated feed-forward."""

    def __init__(self, dim: int, *, head_dim: int, inner_dim: int, prefix: str = "") -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, head_dim=head_dim, prefix=prefix)
        self.norm2 = nn.LayerNorm(dim)
        # One fused projection produces the GLU's value and gate halves.
        self.ff_in = nn.Linear(dim, inner_dim * 2)
        self.ff_out = nn.Linear(inner_dim, dim)

    def forward(self, x: Tensor, rope_cos: Tensor, rope_sin: Tensor) -> Tensor:
        x = x + self.attn(self.norm1(x), rope_cos, rope_sin)
        value, gate = self.ff_in(self.norm2(x)).chunk(2, dim=-1)
        return x + self.ff_out(value * F.silu(gate))


class MiniMaxMusic3Transformer1DModel(nn.Module):
    """The 36-block velocity field: ``([B,128,T], [B], [B,2048,T]) -> [B,128,T]``."""

    def __init__(self) -> None:
        super().__init__()
        self.preprocess_conv = nn.Conv1d(_TRANSFORMER_IN_DIM, _TRANSFORMER_IN_DIM, 1, bias=False)
        self.postprocess_conv = nn.Conv1d(DIT_LATENT_CHANNELS, DIT_LATENT_CHANNELS, 1, bias=False)
        self.proj_in = nn.Linear(_TRANSFORMER_IN_DIM, _DIM, bias=False)
        self.proj_out = nn.Linear(_DIM, DIT_LATENT_CHANNELS, bias=False)
        self.time_proj = FourierFeatures(1, _FOURIER_DIM)
        self.time_embed = TimestepEmbedding(_FOURIER_DIM, _DIM)
        self.transformer_blocks = nn.ModuleList(
            TransformerBlock(
                _DIM,
                head_dim=_HEAD_DIM,
                inner_dim=_FF_INNER_DIM,
                prefix=f"transformer_blocks.{index}.attn",
            )
            for index in range(_NUM_LAYERS)
        )
        inv_freq = 1.0 / (_ROTARY_BASE ** (torch.arange(0, _ROTARY_DIM, 2).float() / _ROTARY_DIM))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._rope_cache: dict[tuple[int, torch.dtype, torch.device], tuple[Tensor, Tensor]] = {}

    @property
    def attention_backend_name(self) -> str:
        """The attention path the blocks actually run, for startup logging."""
        return self.transformer_blocks[0].attn.backend_name

    def _rope(self, seq_len: int, *, dtype: torch.dtype, device: torch.device) -> tuple[Tensor, Tensor]:
        """Return cached ``(cos, sin)`` shaped ``[seq_len, 1, 32]``.

        The head axis is left as a singleton so the tables broadcast over
        ``[B, S, H, D]`` without transposing Q and K.
        """
        key = (seq_len, dtype, device)
        cached = self._rope_cache.get(key)
        if cached is None:
            t = torch.arange(seq_len, device=self.inv_freq.device, dtype=torch.float32)
            freqs = torch.einsum("i,j->ij", t, self.inv_freq)
            freqs = torch.cat((freqs, freqs), dim=-1).to(dtype=dtype, device=device).unsqueeze(1)
            cached = (freqs.cos(), freqs.sin())
            self._rope_cache[key] = cached
        return cached

    def forward(self, x: Tensor, t: Tensor, condition: Tensor) -> Tensor:
        """Predict the flow velocity at latent ``x`` and time ``t``.

        Args:
            x: Current latent, ``[B, 128, T_mel]``.
            t: Flow time in ``[0, 1)``, ``[B]``.
            condition: Aligned condition, ``[B, 2048, T_mel]``.

        Returns:
            The velocity, ``[B, 128, T_mel]``.
        """
        full = torch.cat((x, torch.zeros_like(x), condition), dim=1)
        full = self.preprocess_conv(full) + full
        timestep_embed = self.time_embed(self.time_proj(t[:, None]))
        h = self.proj_in(full.transpose(1, 2))
        # The timestep rides as an extra leading position and is dropped again
        # after the stack, so it shifts every rotary phase by one step.
        h = torch.cat((timestep_embed.unsqueeze(1), h), dim=1)
        rope_cos, rope_sin = self._rope(h.shape[1], dtype=h.dtype, device=h.device)
        for block in self.transformer_blocks:
            h = block(h, rope_cos, rope_sin)
        out = self.proj_out(h[:, 1:]).transpose(1, 2)
        return self.postprocess_conv(out) + out


def remap_transformer_state(state: dict[str, Tensor]) -> dict[str, Tensor]:
    """Rewrite ``transformer/`` checkpoint keys onto this module tree.

    Two rewrites, both confined to the attention block:

    * ``attn.to_q/to_k/to_v.weight`` are concatenated along the output axis
      into ``attn.to_qkv.weight``. ``nn.Linear`` output feature ``i`` reads
      weight row ``i``, and :meth:`Attention.forward` splits the projection
      with ``chunk(3, dim=-1)`` into ``q, k, v``, so query rows come first and
      value rows last.
    * ``attn.to_out.0.weight`` loses its index: the checkpoint models
      ``to_out`` as a two-element list whose second element is a parameterless
      dropout, and inference only needs the projection.

    Every other key is copied through untouched, including the fused ``ff_in``
    whose halves are consumed as ``value, gate`` in that order.

    Raises:
        ValueError: If a block is missing one of its three projections.
    """
    remapped: dict[str, Tensor] = {}
    for key, value in state.items():
        if key.endswith(".attn.to_out.0.weight"):
            remapped[key.replace(".to_out.0.weight", ".to_out.weight")] = value
            continue
        if ".attn.to_q.weight" in key:
            prefix = key[: -len("to_q.weight")]
            try:
                fused = [state[f"{prefix}to_{name}.weight"] for name in "qkv"]
            except KeyError as exc:
                raise ValueError(
                    f"MiniMax Music 3 transformer block {prefix!r} is missing attention projection {exc}"
                ) from exc
            remapped[f"{prefix}to_qkv.weight"] = torch.cat(fused, dim=0)
            continue
        if ".attn.to_k.weight" in key or ".attn.to_v.weight" in key:
            continue
        remapped[key] = value
    return remapped


class MiniMaxMusic3ConditionEncoder(nn.Module):
    """Collapse an AR frame's eight layer slices and resample onto mel time."""

    def __init__(self) -> None:
        super().__init__()
        self.layer_weight_logits = nn.Parameter(torch.zeros(NUM_CODEBOOKS))
        self.layer_scale = nn.Parameter(torch.ones(1))
        self.proj = nn.Conv1d(BACKBONE_HIDDEN_SIZE, _CONDITION_DIM, kernel_size=3, padding=1)

    def aligned_mel_length(self, frames: int) -> int:
        """Return how many vocoder latent frames ``frames`` AR frames cover.

        AR frames sit on a 24 kHz / 960-hop grid (25 fps) and the vocoder
        latent on a 44.1 kHz / 512-hop grid (about 86 fps), so one AR frame is
        roughly 3.445 latent frames. The result is truncated, never rounded.
        """
        return max(
            1,
            int(
                frames
                * CONDITION_OUTPUT_SAMPLE_RATE
                / CONDITION_INPUT_SAMPLE_RATE
                * CONDITION_INPUT_HOP
                / CONDITION_OUTPUT_HOP
            ),
        )

    def condition(self, hidden: Tensor) -> Tensor:
        """Project ``[B, T, 32768]`` AR frames to ``[B, 2048, T]``.

        Raises:
            ValueError: If the conditioning is not ``[B, T, 32768]``.
        """
        if hidden.ndim != 3 or hidden.shape[-1] != AR_HIDDEN_SIZE:
            raise ValueError(f"MiniMax Music 3 condition must be [B,T,{AR_HIDDEN_SIZE}], got {tuple(hidden.shape)}")
        hidden_cf = hidden.transpose(1, 2)
        bsz, _, frames = hidden_cf.shape
        hidden_cf = hidden_cf.reshape(bsz, NUM_CODEBOOKS, BACKBONE_HIDDEN_SIZE, frames)
        weights = torch.softmax(self.layer_weight_logits, dim=0).to(hidden_cf.dtype)
        hidden_cf = torch.einsum("blht,l->bht", hidden_cf, weights)
        hidden_cf = self.layer_scale.to(hidden_cf.dtype) * hidden_cf
        return self.proj(hidden_cf)

    def aligned_condition(self, hidden: Tensor) -> Tensor:
        """Project AR hidden states and resample them onto the latent grid."""
        hidden = hidden.to(dtype=next(self.parameters()).dtype)
        align = self.condition(hidden)
        mel_len = self.aligned_mel_length(hidden.shape[1])
        return F.interpolate(align, size=mel_len, mode="nearest")


class MiniMaxMusic3FlowMatchingDiT(nn.Module):
    """Condition encoder plus velocity field, with the Euler solver on top."""

    def __init__(self) -> None:
        super().__init__()
        self.condition_encoder = MiniMaxMusic3ConditionEncoder()
        self.transformer = MiniMaxMusic3Transformer1DModel()

    def aligned_mel_length(self, frames: int) -> int:
        return self.condition_encoder.aligned_mel_length(frames)

    def condition(self, hidden: Tensor) -> Tensor:
        return self.condition_encoder.condition(hidden)

    def aligned_condition(self, hidden: Tensor) -> Tensor:
        return self.condition_encoder.aligned_condition(hidden)

    @torch.inference_mode()
    def forward(
        self,
        align: Tensor,
        *,
        generator: torch.Generator,
        initial_latent: Tensor | None = None,
        initial_condition: Tensor | None = None,
        num_steps: int = DEFAULT_DIT_STEPS,
        cfg_scale: float = DEFAULT_DIT_CFG_SCALE,
    ) -> Tensor:
        """Solve an aligned condition into a vocoder latent ``[1, 128, T_mel]``.

        The previous window's latent is re-imposed at every step rather than
        only at the start: the leading ``left`` frames are pinned to the exact
        noise-to-latent interpolation the solver would have produced had it
        generated them, so the two windows join without a seam.

        Args:
            align: Aligned condition, ``[1, 2048, T_mel]``. Overwritten in
                place over the prompt span when ``initial_condition`` is
                given, which the caller relies on when it saves the condition
                for the next window.
            generator: Seeded generator for the initial noise draw.
            initial_latent: Previous window's tail latent, or ``None``.
            initial_condition: Previous window's tail condition, or ``None``.
            num_steps: Euler steps.
            cfg_scale: Classifier-free guidance weight.

        Returns:
            The solved latent, ``[1, 128, T_mel]``.

        Raises:
            ValueError: If ``num_steps`` is not positive.
        """
        if num_steps < 1:
            raise ValueError("MiniMax Music 3 DiT num_steps must be positive")
        mel_len = align.shape[-1]
        x = torch.randn(
            (align.shape[0], DIT_LATENT_CHANNELS, mel_len),
            device=align.device,
            dtype=align.dtype,
            generator=generator,
        )

        left = 0
        latent_prompt: Tensor | None = None
        noise_prompt: Tensor | None = None
        if initial_latent is not None:
            left = min(initial_latent.shape[-1], mel_len)
            if left <= 0:
                left = 0
            else:
                latent_prompt = initial_latent[..., :left].to(x)
                noise_prompt = x[..., :left].clone()
                if initial_condition is not None:
                    align[..., :left] = initial_condition[..., :left].to(align)

        dt = 1.0 / num_steps
        # Row 0 carries the condition, row 1 stays zero: one batched forward
        # evaluates both guidance branches.
        cond_cfg = torch.zeros((2, *align.shape[1:]), device=align.device, dtype=align.dtype)
        cond_cfg[0].copy_(align[0])
        # Every timestep in one host-to-device copy. Built from the same
        # Python divisions the per-step construction would have used, so the
        # values are identical, but the solver loop then touches the host only
        # to launch kernels.
        schedule = torch.tensor(
            [step / num_steps for step in range(num_steps)],
            device=align.device,
            dtype=align.dtype,
        )

        for step in range(num_steps):
            t = schedule[step].expand(x.shape[0])
            if left and latent_prompt is not None and noise_prompt is not None:
                x[..., :left] = (1.0 - (1.0 - _SIGMA_FLOOR) * t[0]) * noise_prompt + t[0] * latent_prompt
            d = self.transformer(x.expand(2, -1, -1), t.expand(2), cond_cfg)
            d = cfg_scale * d[:1] + (1.0 - cfg_scale) * d[1:2]
            x = x + dt * d

        if left and latent_prompt is not None:
            x[..., :left] = latent_prompt
        return x
