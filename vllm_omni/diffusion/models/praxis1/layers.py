# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
# Architecture adapted from https://github.com/runwayml/praxis-1-oss (Praxis-1).
"""Native layers of the Praxis-1 video/action transformer.

Attention runs through Omni's shared :class:`Attention` layer. The video
backbone and the action expert attend under block-sparse visibility rules, so
the model owns a FlexAttention kernel (``attention_impl="flex"``, the default).
``attention_impl="backend"`` instead hands a dense Boolean mask to the backend
Omni selected for the deployment; backends that cannot consume dense masks fall
back to SDPA inside the shared layer.
"""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Callable, Sequence
from functools import cache
from typing import Literal

import torch
import torch.nn.functional as F
from diffusers.models.embeddings import get_timestep_embedding
from torch import Tensor, nn
from torch.nn.attention.flex_attention import BlockMask, create_block_mask, create_mask, flex_attention
from vllm.model_executor.layers.linear import MergedColumnParallelLinear, QKVParallelLinear, ReplicatedLinear
from vllm.model_executor.layers.rotary_embedding.common import ApplyRotaryEmb

from vllm_omni.diffusion.attention.backends.abstract import AttentionMetadata
from vllm_omni.diffusion.attention.layer import Attention as OmniAttention
from vllm_omni.diffusion.layers.activation import SiluAndMul
from vllm_omni.diffusion.layers.fused_moe import FusedMoE

AttentionImpl = Literal["flex", "backend"]
ATTENTION_IMPLS: tuple[str, ...] = ("flex", "backend")

# Every FlexAttention tile size, including the short-query decode defaults,
# divides 256, so one block size serves all layouts on every supported GPU.
MASK_BLOCK_SIZE = 256

MaskMod = Callable[[Tensor | None, Tensor | None, Tensor, Tensor], Tensor]


def pad_tags(values: Tensor, length: int, fill: int = -1) -> Tensor:
    """Pad per-token integer tags to the block-rounded ``length`` with ``fill``."""
    return F.pad(values.to(torch.int32), (0, -length % MASK_BLOCK_SIZE), value=fill)


class SparseMask:
    """A per-token visibility rule with lazily built block-sparse and dense forms."""

    def __init__(self, mask_mod: MaskMod, q_len: int, k_len: int, device: torch.device) -> None:
        self.mask_mod = mask_mod
        self.q_len = q_len
        self.k_len = k_len
        self.device = device
        self._block: BlockMask | None = None
        self._dense: Tensor | None = None

    @property
    def block(self) -> BlockMask:
        if self._block is None:
            self._block = create_block_mask(
                self.mask_mod, 1, 1, self.q_len, self.k_len, device=str(self.device), BLOCK_SIZE=MASK_BLOCK_SIZE
            )
        return self._block

    @property
    def dense(self) -> Tensor:
        """Boolean ``(1, 1, Q, K)`` mask for backends that take dense masks."""
        if self._dense is None:
            self._dense = create_mask(self.mask_mod, 1, 1, self.q_len, self.k_len, device=str(self.device))
        return self._dense


class MaskCache:
    """Keep the masks of recently used token layouts (one per layout and device)."""

    def __init__(self, max_entries: int = 8) -> None:
        self.max_entries = max_entries
        self._entries: OrderedDict[tuple, object] = OrderedDict()

    def get(self, key: tuple, build: Callable[[], object]) -> object:
        if key in self._entries:
            self._entries.move_to_end(key)
            return self._entries[key]
        value = build()
        while len(self._entries) >= self.max_entries:
            self._entries.popitem(last=False)
        self._entries[key] = value
        return value


@cache
def _compiled_flex_attention():
    return torch.compile(flex_attention)


class FlexAttentionKernel(nn.Module):
    """Block-sparse FlexAttention behind Omni's model-owned attention contract."""

    def forward(
        self, query: Tensor, key: Tensor, value: Tensor, attn_metadata: AttentionMetadata | None = None
    ) -> Tensor:
        q, k, v = (t.transpose(1, 2) for t in (query, key, value))
        mask = None if attn_metadata is None else attn_metadata.extra.get("block_mask")
        out = _compiled_flex_attention()(q, k, v, block_mask=mask, enable_gqa=q.shape[1] != k.shape[1])
        return out.transpose(1, 2)


class PraxisAttention(nn.Module):
    """Masked attention over ``(B, H, S, D)`` tensors through the shared Omni layer."""

    def __init__(
        self,
        heads: int,
        kv_heads: int,
        head_dim: int,
        *,
        cross: bool = False,
        attention_impl: str = "flex",
    ) -> None:
        super().__init__()
        if attention_impl not in ATTENTION_IMPLS:
            raise ValueError(f"attention_impl must be one of {ATTENTION_IMPLS}, got {attention_impl!r}")
        self.attention_impl = attention_impl
        self.attention = OmniAttention(
            num_heads=heads,
            num_kv_heads=kv_heads,
            head_size=head_dim,
            causal=False,
            softmax_scale=head_dim**-0.5,
            role="cross" if cross else "self",
            qkv_layout="BSND",
            custom_attention=FlexAttentionKernel() if attention_impl == "flex" else None,
            skip_sequence_parallel=True,
            disable_kv_quant=True,
        )

    def forward(self, q: Tensor, k: Tensor, v: Tensor, mask: SparseMask | None = None) -> Tensor:
        if mask is None:
            metadata = None
        elif self.attention_impl == "flex":
            metadata = AttentionMetadata(extra={"block_mask": mask.block})
        else:
            metadata = AttentionMetadata(attn_mask=mask.dense)
        out = self.attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), metadata)
        return out.transpose(1, 2)


def action_rms_norm(x: Tensor, eps: float = 1e-4) -> Tensor:
    """Scale-free RMS normalization of the action expert (squares in the input dtype)."""
    mean_sq = torch.mean(x**2, dim=-1, keepdim=True, dtype=torch.float32)
    return (x.float() * torch.rsqrt(mean_sq + eps)).to(x.dtype)


class ZeroCenteredRMSNorm(nn.Module):
    """RMSNorm whose scale is ``1 + weight``; computed in float32 unless ``fp32=False``."""

    def __init__(self, dim: int, eps: float = 1e-6, *, fp32: bool = True) -> None:
        super().__init__()
        self.dim, self.eps, self.fp32 = dim, eps, fp32
        self.weight = nn.Parameter(torch.zeros(dim))

    def forward(self, x: Tensor) -> Tensor:
        value, weight = (x.float(), self.weight.float()) if self.fp32 else (x, self.weight)
        return F.rms_norm(value, (self.dim,), weight=1 + weight, eps=self.eps).to(x.dtype)


class TimestepEmbedder(nn.Module):
    """Sinusoidal features of a flow time, then a two-layer MLP (action expert)."""

    def __init__(self, dim: int, frequency_dim: int = 256, max_period: float = 10000) -> None:
        super().__init__()
        self.frequency_dim = frequency_dim
        self.max_period = max_period
        self.mlp = nn.Sequential(
            ReplicatedLinear(frequency_dim, dim, return_bias=False),
            nn.SiLU(),
            ReplicatedLinear(dim, dim, return_bias=False),
        )

    def forward(self, t: Tensor) -> Tensor:
        """``t (...)`` -> ``(..., dim)``."""
        features = get_timestep_embedding(
            t.flatten(), self.frequency_dim, flip_sin_to_cos=True, downscale_freq_shift=0, max_period=self.max_period
        )
        first = self.mlp[0]
        assert isinstance(first, ReplicatedLinear)
        features = features.reshape(*t.shape, self.frequency_dim).to(dtype=first.weight.dtype)
        return self.mlp(features)


class ActionAdaRMSNorm(nn.Module):
    """Scale-only adaptive RMS modulation of the action expert."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.proj = ReplicatedLinear(dim, dim, return_bias=False)

    def forward(self, tokens: Tensor, cond: Tensor) -> Tensor:
        """``tokens (B, S, D)``, ``cond (B, S, D)`` -> ``(B, S, D)``."""
        scale = self.proj(F.silu(cond))
        return ((1 + scale.float()) * action_rms_norm(tokens)).to(dtype=tokens.dtype)


class TimestepEmbedding(nn.Module):
    """Fixed random Fourier features of the flow timestep (video backbone)."""

    freqs: Tensor
    phases: Tensor

    def __init__(self, dim: int = 128) -> None:
        super().__init__()
        self.register_buffer("freqs", torch.zeros(1, dim))
        self.register_buffer("phases", torch.zeros(1, dim))

    def forward(self, t: Tensor) -> Tensor:
        """Map ``t`` of shape ``(B, K)`` to features of shape ``(B, K, dim)``."""
        x = t[..., None].float() * self.freqs.float() + self.phases.float()
        x = x.to(t.dtype)
        return (x * 2 * torch.pi).cos() * 0.5


class VideoInputProjection(nn.Module):
    """Linear map of latent channels ``(B, C, T, h, w)`` to tokens ``(B, T, h, w, dim)``."""

    def __init__(self, c_in: int, dim: int) -> None:
        super().__init__()
        self.proj = ReplicatedLinear(c_in, dim, return_bias=False)

    def forward(self, v: Tensor) -> Tensor:
        return self.proj(v.movedim(1, -1))


def modulate(x: Tensor, shift: Tensor, scale: Tensor) -> Tensor:
    """``shift + x * scale`` in float32, rounded back to the dtype of ``x``."""
    return torch.addcmul(shift.float(), x.float(), scale.float()).to(x.dtype)


class MoEFeedForward(nn.Module):
    """One shared SwiGLU expert plus ``top_k`` of ``num_experts`` routed experts.

    The router scores in float32 with a sigmoid. A float32 bias steers which
    experts are picked but not their weights; the picked scores are normalized
    to sum to one and scaled by ``routed_scaling_factor``. The routed experts
    run through vLLM's fused MoE kernels.
    """

    expert_bias: Tensor

    def __init__(
        self,
        dim: int,
        hidden: int,
        num_experts: int,
        top_k: int,
        routed_scaling_factor: float,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.top_k = top_k
        self.routed_scaling_factor = float(routed_scaling_factor)
        self.router = ReplicatedLinear(dim, num_experts, bias=False, return_bias=False)
        self.shared = nn.Sequential(
            OrderedDict(
                gate_up=ReplicatedLinear(dim, 2 * hidden, bias=False, return_bias=False),
                activation=SiluAndMul(),
                down=ReplicatedLinear(hidden, dim, bias=False, return_bias=False),
            )
        )
        self.register_buffer("expert_bias", torch.zeros(num_experts, dtype=torch.float32))
        self.experts = FusedMoE(
            num_experts=num_experts,
            top_k=top_k,
            hidden_size=dim,
            intermediate_size=hidden,
            params_dtype=torch.get_default_dtype(),
            renormalize=False,
            custom_routing_function=self.route,
            routed_scaling_factor=1.0,
            prefix=prefix,
            tp_size=1,
            dp_size=1,
            pcp_size=1,
        )

    def _apply(self, fn, recurse=True):
        bias = self.expert_bias
        super()._apply(fn, recurse)
        # Keep the routing bias in float32 under module dtype casts.
        self.expert_bias = bias.to(device=self.expert_bias.device)
        return self

    def route(
        self, hidden_states: Tensor, gating_output: Tensor, topk: int, renormalize: bool
    ) -> tuple[Tensor, Tensor]:
        """Return the float32 expert weights ``(T, top_k)`` and expert ids."""
        scores = gating_output.float().sigmoid()
        ids = torch.topk(scores + self.expert_bias, topk, dim=-1).indices
        weights = scores.gather(-1, ids)
        weights = weights / weights.sum(-1, keepdim=True) * self.routed_scaling_factor
        return weights.contiguous(), ids.to(torch.int32)

    def forward(self, x: Tensor) -> Tensor:
        shape = x.shape
        flat = x.reshape(-1, shape[-1])
        logits = F.linear(flat.float(), self.router.weight.float())
        routed = self.experts(flat, logits)
        return (routed + self.shared(flat)).reshape(shape)


class RotaryTable(nn.Module):
    """Rotary ``(cos, sin)`` table for large integer positions.

    Only the first ``effective_dim // 2`` frequency pairs rotate; the remaining
    pairs have zero frequency. A position is split as ``high * table_size +
    low`` and the two angles are combined in float64, which keeps positions in
    the millions accurate in float32.
    """

    def __init__(self, dim: int, effective_dim: int, base: int, table_size: int) -> None:
        super().__init__()
        if effective_dim > dim:
            raise ValueError(f"effective_dim {effective_dim} exceeds head_dim {dim}")
        self.dim = dim
        self.table_size = table_size
        exponent = torch.arange(0, effective_dim, 2)[: effective_dim // 2].float() / effective_dim
        theta = 1.0 / base**exponent
        self.theta = torch.cat([theta, torch.zeros((dim - effective_dim) // 2)], dim=0)

    def forward(self, positions: Tensor) -> Tensor:
        """Map ``positions`` (B, N) to a float32 table (B, N, dim // 2, 2)."""
        theta = self.theta.to(positions.device).double()
        high = (positions // self.table_size).to(torch.float32).double() * self.table_size
        low = (positions % self.table_size).to(torch.float32).double()
        angle_high = high[..., None] * theta
        angle_low = low[..., None] * theta
        cos = angle_high.cos() * angle_low.cos() - angle_high.sin() * angle_low.sin()
        sin = angle_high.sin() * angle_low.cos() + angle_high.cos() * angle_low.sin()
        table = torch.stack([cos, sin], dim=-1).float()
        return table.unsqueeze(0) if table.dim() == 3 else table


def apply_rope(x: Tensor, table: Tensor) -> Tensor:
    """Rotate ``x (B, H, N, D)`` with the ``(1, N, D // 2, 2)`` table in float32."""
    cos, sin = table[0].unbind(-1)
    out = ApplyRotaryEmb.forward_static(x.transpose(1, 2), cos, sin, enable_fp32_compute=True)
    return out.transpose(1, 2).contiguous()


class Attention(nn.Module):
    """Grouped-query self-attention with rotary positions and a sigmoid gate per head."""

    def __init__(self, dim: int, head_dim: int, num_kv_heads: int, attention_impl: str) -> None:
        super().__init__()
        self.head_dim = head_dim
        self.num_heads = dim // head_dim
        self.num_kv_heads = num_kv_heads
        self.in_proj = QKVParallelLinear(dim, head_dim, self.num_heads, num_kv_heads, bias=False, return_bias=False)
        self.gate_proj = ReplicatedLinear(dim, self.num_heads, bias=False, return_bias=False)
        self.q_norm = ZeroCenteredRMSNorm(head_dim)
        self.k_norm = ZeroCenteredRMSNorm(head_dim)
        self.out_proj = ReplicatedLinear(self.num_heads * head_dim, dim, return_bias=False)
        self.attention = PraxisAttention(self.num_heads, num_kv_heads, head_dim, attention_impl=attention_impl)

    def forward(self, x: Tensor, rope: Tensor | None, mask: SparseMask | None) -> Tensor:
        b, s, _ = x.shape
        d = self.head_dim
        q, k, v = self.in_proj(x).split([self.num_heads * d, self.num_kv_heads * d, self.num_kv_heads * d], dim=-1)
        gate = self.gate_proj(x).sigmoid().transpose(1, 2).unsqueeze(-1)
        q = q.view(b, s, self.num_heads, d).transpose(1, 2)
        k = k.view(b, s, self.num_kv_heads, d).transpose(1, 2)
        v = v.view(b, s, self.num_kv_heads, d).transpose(1, 2)
        if rope is not None:
            q, k = apply_rope(q, rope), apply_rope(k, rope)
        out = self.attention(self.q_norm(q), self.k_norm(k), v, mask) * gate
        return self.out_proj(out.transpose(1, 2).reshape(b, s, -1))


Modulation = tuple[nn.Module, Tensor]


class TransformerBlock(nn.Module):
    """Pre-norm block: ``x + attn(norm(x) * s)``, then ``x + ffn(norm(x) * s)``."""

    s: Tensor

    def __init__(
        self, ffn: nn.Module, dim: int, head_dim: int, num_kv_heads: int, layer_index: int, attention_impl: str
    ) -> None:
        super().__init__()
        self.register_buffer("s", torch.tensor(1 / layer_index**0.5))
        self.ln_in = ZeroCenteredRMSNorm(dim)
        self.mha = Attention(dim, head_dim, num_kv_heads, attention_impl)
        self.ln_out = ZeroCenteredRMSNorm(dim)
        self.ffn = ffn

    def forward(
        self, x: Tensor, rope: Tensor | None, mask: SparseMask | None, modulation: Modulation | None = None
    ) -> Tensor:
        if modulation is not None:
            modulator, x_mod = modulation
            shift, scale = modulator(x_mod).chunk(2, dim=-1)
            x = modulate(x, shift, 1 + scale)
        x = x + self.mha(self.ln_in(x) * self.s, rope, mask)
        return x + self.ffn(self.ln_out(x) * self.s)


class Transformer(nn.Module):
    """A stack of blocks (dense SwiGLU, then MoE) with a shared rotary table."""

    def __init__(self, cfg: dict, tower: dict, prefix: str, attention_impl: str) -> None:
        super().__init__()
        dim, depth = cfg["dim"], tower["depth"]
        attn, rope, moe = cfg["attention"], cfg["rope"], cfg["moe"]
        self.rope = RotaryTable(attn["head_dim"], rope["effective_dim"], rope["base"], rope["efficient_cache_max"])
        blocks = []
        for i in range(depth):
            ffn: nn.Module
            if i < len(tower["dense_blocks"]):
                hidden = cfg["dense_ffn_hidden"]
                ffn = nn.Sequential(
                    nn.Sequential(
                        OrderedDict(
                            linear=MergedColumnParallelLinear(
                                dim, [hidden, hidden], bias=False, return_bias=False, disable_tp=True
                            ),
                            activation=SiluAndMul(),
                        )
                    ),
                    ReplicatedLinear(hidden, dim, bias=False, return_bias=False),
                )
            else:
                ffn = MoEFeedForward(
                    dim,
                    moe["expert_hidden"],
                    moe["num_routed_experts"],
                    moe["top_k_routed"],
                    moe["routed_scaling_factor"],
                    prefix=f"{prefix}.blocks.{i}.ffn.experts",
                )
            blocks.append(TransformerBlock(ffn, dim, attn["head_dim"], attn["num_kv_heads"], i + 1, attention_impl))
        self.blocks = nn.ModuleList(blocks)
        self.modulators: nn.ModuleList | None = None
        if (mod_dim := tower.get("modulation_dim")) is not None:
            self.modulators = nn.ModuleList(
                [ReplicatedLinear(mod_dim, 2 * dim, return_bias=False) for _ in range(depth)]
            )

    def forward(
        self,
        x: Tensor,
        rope_idx: Tensor,
        x_mod: Tensor | None = None,
        mask: SparseMask | None = None,
        tap_blocks: Sequence[int] = (),
    ) -> tuple[Tensor, list[Tensor]]:
        """Map ``x`` (B, N, C) to (B, N, C) and return the taps of ``tap_blocks``."""
        rope = self.rope(rope_idx)[:, : x.size(-2)]
        taps: dict[int, Tensor] = {}
        for i, block in enumerate(self.blocks):
            modulation = None
            if self.modulators is not None:
                assert x_mod is not None, "a modulated stack needs x_mod"
                modulation = (self.modulators[i], x_mod)
            x = block(x, rope, mask, modulation)
            if i in tap_blocks:
                taps[i] = x
        return x, [taps[i] for i in tap_blocks]
