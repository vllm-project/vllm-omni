# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Ideogram 4 Transformer model adapted for vllm-omni.

This is a single-stream DiT (text + image tokens concatenated) with:
- MRoPE (Multi-modal Rotary Position Embedding) with 3 axes (temporal, height, width)
- AdaLN modulation from timestep embedding
- QK-RMSNorm attention
- Asymmetric CFG: conditional and unconditional branches are separate model instances

Original source: https://github.com/ideogram-ai/ideogram4
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers.models.transformers.transformer_ideogram4 import (
    LLM_TOKEN_INDICATOR,
    OUTPUT_IMAGE_INDICATOR,
)
from vllm.logger import init_logger
from vllm.model_executor.layers.linear import (
    ColumnParallelLinear,
    QKVParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
)
from vllm.model_executor.model_loader.weight_utils import default_weight_loader

if TYPE_CHECKING:
    from vllm.model_executor.layers.quantization.base_config import (
        QuantizationConfig,
    )

from vllm_omni.diffusion.attention.backends.abstract import (
    AttentionMetadata,
)
from vllm_omni.diffusion.attention.layer import Attention
from vllm_omni.diffusion.cache.base import CachedTransformer
from vllm_omni.diffusion.data import OmniDiffusionConfig
from vllm_omni.diffusion.distributed.hsdp_utils import is_transformer_block_module
from vllm_omni.diffusion.distributed.sp_plan import (
    SequenceParallelInput,
    SequenceParallelOutput,
)

logger = init_logger(__name__)


def _pick_nf4_dequant_device(weight: torch.Tensor) -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    return weight.device


_NF4_SIBLING_SUFFIXES = (".absmax", ".quant_map", ".quant_state.bitsandbytes__nf4")


def _consume_nf4(name: str, consumed: set[str]) -> None:
    consumed.add(name)
    for suffix in _NF4_SIBLING_SUFFIXES:
        consumed.add(name + suffix)


@dataclass
class Ideogram4Config:
    """Configuration for Ideogram 4 Transformer."""

    emb_dim: int = 4608
    num_layers: int = 34
    num_heads: int = 18
    intermediate_size: int = 12288
    adanln_dim: int = 512
    in_channels: int = 128
    llm_features_dim: int = 53248  # 4096 * 13 layers
    rope_theta: int = 5_000_000
    mrope_section: tuple[int, ...] = (24, 20, 20)
    norm_eps: float = 1e-5


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    x1 = x[..., :half]
    x2 = x[..., half:]
    return torch.cat((-x2, x1), dim=-1)


def _apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    # q, k: (B, L, num_heads, head_dim); cos/sin: (B, L, head_dim).
    cos = cos.unsqueeze(2)
    sin = sin.unsqueeze(2)
    q_embed = (q * cos) + (_rotate_half(q) * sin)
    k_embed = (k * cos) + (_rotate_half(k) * sin)
    return q_embed, k_embed


class Ideogram4MRoPE(nn.Module):
    """Multi-modal Rotary Position Embedding with 3 axes (temporal, height, width).

    Uses interleaved mrope: H freqs go into idx 1 mod 3, W freqs into idx 2 mod 3,
    matching the original Ideogram4 implementation.
    """

    def __init__(self, head_dim: int, base: int, mrope_section: tuple[int, ...]):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.mrope_section = tuple(mrope_section)
        self.head_dim = head_dim

    @torch.no_grad()
    def forward(self, position_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute MRoPE cos/sin embeddings.

        Args:
            position_ids: (B, L, 3) with axes [temporal, height, width]

        Returns:
            cos, sin: each (B, L, head_dim)
        """
        assert position_ids.ndim == 3 and position_ids.shape[-1] == 3
        batch_size, seq_len, _ = position_ids.shape

        # (3, B, inv_freq_size, L)
        pos = position_ids.permute(2, 0, 1).to(dtype=torch.float32)
        inv_freq = self.inv_freq.to(dtype=torch.float32)[None, None, :, None].expand(3, batch_size, -1, 1)
        freqs = inv_freq @ pos.unsqueeze(2)
        freqs = freqs.transpose(2, 3)  # (3, B, L, inv_freq_size)

        # Interleaved mrope: pull H freqs into idx 1 mod 3, W freqs into idx 2 mod 3.
        freqs_t = freqs[0].clone()
        for axis, offset in ((1, 1), (2, 2)):
            length = self.mrope_section[axis] * 3
            idx = torch.arange(offset, length, 3, device=freqs_t.device)
            freqs_t[..., idx] = freqs[axis][..., idx]

        emb = torch.cat((freqs_t, freqs_t), dim=-1)
        return emb.cos(), emb.sin()


class Ideogram4RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.rms_norm(x, self.weight.shape, self.weight, self.eps)


def _sinusoidal_embedding(t: torch.Tensor, dim: int, scale: float = 1e4) -> torch.Tensor:
    t = t.to(torch.float32)
    half = dim // 2
    freq = math.log(scale) / (half - 1)
    freq = torch.exp(torch.arange(half, dtype=torch.float32, device=t.device) * -freq)
    emb = t.unsqueeze(-1) * freq
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
    if dim % 2 == 1:
        emb = F.pad(emb, (0, 1))
    return emb


class Ideogram4EmbedScalar(nn.Module):
    """Timestep embedding: scalar -> sinusoidal -> two-layer MLP.

    Matches the original Ideogram4 implementation with input_range normalization.
    """

    def __init__(self, dim: int, input_range: tuple[float, float] = (0.0, 1.0)):
        super().__init__()
        self.dim = dim
        self.range_min, self.range_max = input_range
        assert self.range_max > self.range_min
        self.mlp_in = nn.Linear(dim, dim, bias=True)
        self.mlp_out = nn.Linear(dim, dim, bias=True)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        # t is shape (B,) holding a scalar per sample.
        t = t.to(torch.float32)
        scaled = 1e4 * (t - self.range_min) / (self.range_max - self.range_min)
        emb = _sinusoidal_embedding(scaled, self.dim)
        compute_dtype = getattr(self.mlp_in, "compute_dtype", None) or self.mlp_in.weight.dtype
        emb = emb.to(compute_dtype)
        emb = F.silu(self.mlp_in(emb))
        return self.mlp_out(emb)


class Ideogram4Attention(nn.Module):
    """Single-stream attention with QK-RMSNorm and MRoPE."""

    def __init__(
        self,
        config: Ideogram4Config,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ):
        super().__init__()
        self.num_heads = config.num_heads
        self.head_dim = config.emb_dim // config.num_heads

        self.to_qkv = QKVParallelLinear(
            hidden_size=config.emb_dim,
            head_size=self.head_dim,
            total_num_heads=config.num_heads,
            bias=False,
            quant_config=None,
            prefix=f"{prefix}.to_qkv",
        )

        self.norm_q = Ideogram4RMSNorm(self.head_dim, eps=1e-5)
        self.norm_k = Ideogram4RMSNorm(self.head_dim, eps=1e-5)

        self.to_out = RowParallelLinear(
            config.emb_dim,
            config.emb_dim,
            bias=False,
            input_is_parallel=True,
            return_bias=False,
            quant_config=None,
            prefix=f"{prefix}.to_out",
        )

        self.attn = Attention(
            num_heads=self.to_qkv.num_heads,
            head_size=self.head_dim,
            softmax_scale=1.0 / (self.head_dim**0.5),
            causal=False,
            num_kv_heads=self.to_qkv.num_kv_heads,
        )

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        segment_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Args:
        x: (B, L, emb_dim)
        cos, sin: (B, L, head_dim) from MRoPE
        segment_ids: (B, L) for block-diagonal attention mask
        """
        batch_size, seq_len, _ = x.shape

        qkv = self.to_qkv(x)
        if isinstance(qkv, tuple):
            qkv = qkv[0]
        local_num_heads = self.to_qkv.num_heads
        q_size = local_num_heads * self.head_dim
        q, k, v = qkv.split([q_size, q_size, q_size], dim=-1)

        q = q.view(batch_size, seq_len, local_num_heads, self.head_dim)
        k = k.view(batch_size, seq_len, local_num_heads, self.head_dim)
        v = v.view(batch_size, seq_len, local_num_heads, self.head_dim)

        q = self.norm_q(q)
        k = self.norm_k(k)

        cos_emb = cos.to(q.dtype)
        sin_emb = sin.to(q.dtype)
        q, k = _apply_rotary_pos_emb(q, k, cos_emb, sin_emb)

        attn_metadata = None
        if segment_ids is not None:
            attn_mask = _build_segment_mask(segment_ids, q.device)
            if attn_mask is not None:
                attn_metadata = AttentionMetadata(attn_mask=attn_mask)

        hidden_states = self.attn(q, k, v, attn_metadata)
        hidden_states = hidden_states.flatten(2, 3).to(q.dtype)

        output = self.to_out(hidden_states)
        if isinstance(output, tuple):
            output = output[0]
        return output


def _build_segment_mask(
    segment_ids: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    """Build a boolean attention mask from segment_ids.

    Tokens can attend to each other if they have the same segment_id.
    Padding tokens (segment_id == -1) are isolated from all other tokens
    by assigning each a unique negative value.
    """
    ids = segment_ids

    if (ids == -1).any():
        ids = ids.clone()
        pad_mask = ids == -1
        ids[pad_mask] = torch.arange(-2, -2 - pad_mask.sum().item(), -1, device=device)
    mask = (ids.unsqueeze(2) == ids.unsqueeze(1)).unsqueeze(1)

    return mask


class Ideogram4MLP(nn.Module):
    """Feed-forward network with SwiGLU-style gating. No bias in original."""

    def __init__(
        self,
        config: Ideogram4Config,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ):
        super().__init__()
        self.w1 = ColumnParallelLinear(
            config.emb_dim,
            config.intermediate_size,
            bias=False,
            gather_output=False,
            return_bias=False,
            quant_config=None,
            prefix=f"{prefix}.w1",
        )
        self.w3 = ColumnParallelLinear(
            config.emb_dim,
            config.intermediate_size,
            bias=False,
            gather_output=False,
            return_bias=False,
            quant_config=None,
            prefix=f"{prefix}.w3",
        )
        self.w2 = RowParallelLinear(
            config.intermediate_size,
            config.emb_dim,
            bias=False,
            input_is_parallel=True,
            return_bias=False,
            quant_config=None,
            prefix=f"{prefix}.w2",
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = self.w1(x)
        if isinstance(gate, tuple):
            gate = gate[0]
        gate = F.silu(gate)
        up = self.w3(x)
        if isinstance(up, tuple):
            up = up[0]
        hidden = gate * up
        out = self.w2(hidden)
        if isinstance(out, tuple):
            out = out[0]
        return out


class Ideogram4TransformerBlock(nn.Module):
    """Single-stream Transformer block with AdaLN modulation.

    Matches original architecture:
    - attention_norm1: pre-norm for attention input
    - attention_norm2: post-norm on attention output before residual add
    - ffn_norm1: pre-norm for FFN input
    - ffn_norm2: post-norm on FFN output before residual add
    """

    def __init__(
        self,
        config: Ideogram4Config,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ):
        super().__init__()
        self.config = config
        self.emb_dim = config.emb_dim

        self.attention = Ideogram4Attention(
            config,
            quant_config=quant_config,
            prefix=f"{prefix}.attention",
        )
        self.feed_forward = Ideogram4MLP(
            config,
            quant_config=quant_config,
            prefix=f"{prefix}.feed_forward",
        )

        self.attention_norm1 = Ideogram4RMSNorm(config.emb_dim, eps=config.norm_eps)
        self.ffn_norm1 = Ideogram4RMSNorm(config.emb_dim, eps=config.norm_eps)
        self.attention_norm2 = Ideogram4RMSNorm(config.emb_dim, eps=config.norm_eps)
        self.ffn_norm2 = Ideogram4RMSNorm(config.emb_dim, eps=config.norm_eps)

        # AdaLN modulation: adanln_dim -> 4 * emb_dim (scale_msa, gate_msa, scale_mlp, gate_mlp)
        self.adaln_modulation = nn.Linear(config.adanln_dim, 4 * config.emb_dim, bias=True)

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        adaln_input: torch.Tensor,
        segment_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # AdaLN modulation
        mod = self.adaln_modulation(adaln_input)  # (B, 4 * emb_dim)
        scale_msa, gate_msa, scale_mlp, gate_mlp = mod.chunk(4, dim=-1)
        gate_msa = torch.tanh(gate_msa)
        gate_mlp = torch.tanh(gate_mlp)
        scale_msa = 1.0 + scale_msa
        scale_mlp = 1.0 + scale_mlp

        # Attention: pre-norm * scale -> attention -> post-norm -> gate residual
        attn_out = self.attention(
            self.attention_norm1(x) * scale_msa,
            cos,
            sin,
            segment_ids,
        )
        x = x + gate_msa * self.attention_norm2(attn_out)

        # FFN: pre-norm * scale -> FFN -> post-norm -> gate residual
        x = x + gate_mlp * self.ffn_norm2(self.feed_forward(self.ffn_norm1(x) * scale_mlp))

        return x


class Ideogram4FinalLayer(nn.Module):
    """Final layer: AdaLN + LayerNorm + Linear projection to velocity.

    Matches original: scale = 1 + adaln(silu(c)), then norm(x) * scale.
    """

    def __init__(
        self,
        config: Ideogram4Config,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ):
        super().__init__()
        self.norm_final = nn.LayerNorm(config.emb_dim, eps=1e-6, elementwise_affine=False)
        self.linear = ReplicatedLinear(
            config.emb_dim,
            config.in_channels,
            bias=True,
            return_bias=False,
            quant_config=None,
            prefix=f"{prefix}.linear",
        )
        self.adaln_modulation = nn.Linear(config.adanln_dim, config.emb_dim, bias=True)

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        scale = 1.0 + self.adaln_modulation(F.silu(c))
        output = self.linear(self.norm_final(x) * scale)
        if isinstance(output, tuple):
            output = output[0]
        return output


class Ideogram4Transformer(CachedTransformer):
    """Ideogram 4 single-stream DiT Transformer.

    Architecture:
    - Input: text features (from Qwen3-VL) + image latent tokens, concatenated
    - MRoPE for positional encoding (3 axes: temporal, height, width)
    - 34 Transformer blocks with AdaLN modulation
    - Final layer projects to velocity prediction (128 channels)

    For asymmetric CFG, two separate instances are used:
    - conditional_transformer: processes text + image tokens
    - unconditional_transformer: processes image tokens only (with zero text features)
    """

    _repeated_blocks = ["Ideogram4TransformerBlock"]
    _layerwise_offload_blocks_attrs = ["layers"]
    packed_modules_mapping = {
        "to_qkv": ["to_q", "to_k", "to_v"],
    }

    _hsdp_shard_conditions = [is_transformer_block_module]

    _sp_plan = {
        "input_proj": {
            0: SequenceParallelInput(split_dim=1, expected_dims=3, split_output=True, auto_pad=True),
        },
        "final_layer.linear": SequenceParallelOutput(gather_dim=1, expected_dims=3),
    }

    def __init__(
        self,
        od_config: OmniDiffusionConfig,
        config: Ideogram4Config | None = None,
        quant_config: QuantizationConfig | None = None,
    ):
        super().__init__()
        self.od_config = od_config
        self.config = config or Ideogram4Config()
        self.parallel_config = od_config.parallel_config

        head_dim = self.config.emb_dim // self.config.num_heads

        self.input_proj = nn.Linear(self.config.in_channels, self.config.emb_dim, bias=True)

        self.llm_cond_norm = Ideogram4RMSNorm(self.config.llm_features_dim, eps=1e-6)
        self.llm_cond_proj = nn.Linear(self.config.llm_features_dim, self.config.emb_dim, bias=True)

        self.t_embedding = Ideogram4EmbedScalar(self.config.emb_dim, input_range=(0.0, 1.0))

        self.adaln_proj = nn.Linear(self.config.emb_dim, self.config.adanln_dim, bias=True)

        self.embed_image_indicator = nn.Embedding(2, self.config.emb_dim)

        self.rotary_emb = Ideogram4MRoPE(
            head_dim=head_dim,
            base=self.config.rope_theta,
            mrope_section=self.config.mrope_section,
        )

        self.layers = nn.ModuleList(
            [
                Ideogram4TransformerBlock(
                    self.config,
                    quant_config=quant_config,
                    prefix=f"layers.{i}",
                )
                for i in range(self.config.num_layers)
            ]
        )

        self.final_layer = Ideogram4FinalLayer(
            self.config,
            quant_config=quant_config,
            prefix="final_layer",
        )

    def forward(
        self,
        llm_features: torch.Tensor,
        x: torch.Tensor,
        t: torch.Tensor,
        position_ids: torch.Tensor,
        segment_ids: torch.Tensor,
        indicator: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            llm_features: (B, L_text, llm_features_dim) text features from Qwen3-VL
            x: (B, L, in_channels) latent tokens (for conditional: text_padding + image,
               for unconditional: image only)
            t: (B,) timestep in [0, 1]
            position_ids: (B, L, 3) MRoPE position ids
            segment_ids: (B, L) segment ids for attention masking
            indicator: (B, L) token type indicator (2=image, 3=text, 0=padding)

        Returns:
            velocity: (B, L, in_channels) predicted velocity
        """
        batch_size, seq_len, in_channels = x.shape
        assert in_channels == self.config.in_channels

        param_dtype = getattr(self.input_proj, "compute_dtype", None) or self.input_proj.weight.dtype
        x = x.to(param_dtype)
        t = t.to(param_dtype)
        llm_features = llm_features.to(param_dtype)

        indicator = indicator.to(torch.long)
        llm_token_mask = (indicator == LLM_TOKEN_INDICATOR).to(x.dtype).unsqueeze(-1)
        output_image_mask = (indicator == OUTPUT_IMAGE_INDICATOR).to(x.dtype).unsqueeze(-1)

        llm_features = llm_features * llm_token_mask
        x = x * output_image_mask

        x = self.input_proj(x) * output_image_mask

        t_cond = self.t_embedding(t)  # (B, emb_dim)
        if t.dim() == 1:
            t_cond = t_cond.unsqueeze(1)  # (B, 1, emb_dim)
        adaln_input = F.silu(self.adaln_proj(t_cond))  # (B, 1, adanln_dim)

        llm_features = self.llm_cond_norm(llm_features)
        llm_features = self.llm_cond_proj(llm_features) * llm_token_mask

        h = x + llm_features

        image_indicator_embedding = self.embed_image_indicator((indicator == OUTPUT_IMAGE_INDICATOR).to(torch.long))
        h = h + image_indicator_embedding

        cos, sin = self.rotary_emb(position_ids)
        cos = cos.to(h.dtype)
        sin = sin.to(h.dtype)

        for layer_idx, layer in enumerate(self.layers):
            h = layer(h, cos, sin, adaln_input, segment_ids)
        output = self.final_layer(h, c=adaln_input)  # (B, L, in_channels)

        return output.to(torch.float32)

    def _preprocess_nf4_weights(
        self,
        all_weights: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        """Dequantize NF4 tensors to bf16 before loading."""
        is_nf4 = any(".quant_state.bitsandbytes__" in key for key in all_weights)
        if not is_nf4:
            return all_weights

        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError(
                "bitsandbytes is required to load NF4 quantized Ideogram4 "
                "weights. Install it with: pip install bitsandbytes"
            )

        import json

        consumed: set[str] = set()
        dequantized: dict[str, torch.Tensor] = {}

        for name, source_tensor in list(all_weights.items()):
            if name in consumed:
                continue
            if not (source_tensor.ndim == 2 and source_tensor.dtype == torch.uint8 and source_tensor.shape[1] == 1):
                continue

            absmax = all_weights.get(name + ".absmax")
            quant_map = all_weights.get(name + ".quant_map")
            qs_tensor = all_weights.get(name + ".quant_state.bitsandbytes__nf4")

            has_bnb_siblings = absmax is not None and quant_map is not None
            if not has_bnb_siblings and qs_tensor is None:
                continue

            try:
                if qs_tensor is not None:
                    data = qs_tensor.cpu().numpy().tobytes()
                    end = data.find(0)
                    json_str = data[: end if end != -1 else len(data)].decode("utf-8")
                    qs = json.loads(json_str)
                else:
                    qs = {
                        "quant_type": "nf4",
                        "blocksize": 64,
                        "dtype": "bfloat16",
                        "shape": None,
                    }
                target_device = _pick_nf4_dequant_device(source_tensor)

                if absmax is None:
                    raise ValueError(f"Missing absmax for {name}")
                if quant_map is None:
                    raise ValueError(f"Missing quant_map for {name}")

                quant_state = bnb.functional.QuantState(
                    quant_type=qs["quant_type"],
                    absmax=absmax.to(target_device),
                    blocksize=qs["blocksize"],
                    code=quant_map.to(target_device),
                    dtype=getattr(torch, qs["dtype"]),
                    shape=torch.Size(qs["shape"]) if qs.get("shape") else None,
                )

                w_dequant = bnb.functional.dequantize_4bit(
                    source_tensor.to(target_device),
                    quant_state=quant_state,
                    quant_type="nf4",
                )
                dequantized[name] = w_dequant.cpu()
                del w_dequant
                if target_device.type == "cuda":
                    torch.accelerator.empty_cache()
                _consume_nf4(name, consumed)
            except RuntimeError as e:
                if "out of memory" in str(e).lower() and target_device.type == "cuda":
                    logger.warning(
                        "GPU OOM dequantizing %s, falling back to CPU",
                        name,
                    )
                    try:
                        cpu_device = torch.device("cpu")
                        quant_state_cpu = bnb.functional.QuantState(
                            quant_type=qs["quant_type"],
                            absmax=absmax.to(cpu_device),
                            blocksize=qs["blocksize"],
                            code=quant_map.to(cpu_device),
                            dtype=getattr(torch, qs["dtype"]),
                            shape=torch.Size(qs["shape"]) if qs.get("shape") else None,
                        )
                        w_dequant = bnb.functional.dequantize_4bit(
                            source_tensor.to(cpu_device),
                            quant_state=quant_state_cpu,
                            quant_type="nf4",
                        )
                        dequantized[name] = w_dequant
                        del w_dequant
                        _consume_nf4(name, consumed)
                    except Exception as e2:
                        logger.error(
                            "Failed to dequantize NF4 weight %s on CPU too: %s",
                            name,
                            e2,
                        )
                        _consume_nf4(name, consumed)
                else:
                    logger.warning("Failed to dequantize NF4 weight %s: %s", name, e)
                    _consume_nf4(name, consumed)
            except Exception as e:
                logger.warning("Failed to dequantize NF4 weight %s: %s", name, e)
                _consume_nf4(name, consumed)

        for name in consumed:
            all_weights.pop(name, None)
        all_weights.update(dequantized)
        return all_weights

    def _remap_packed_qkv_weight(self, weight: torch.Tensor) -> torch.Tensor:
        """Convert packed qkv weights to the split layout expected by QKVParallelLinear."""
        num_heads = self.config.num_heads
        head_dim = self.config.emb_dim // num_heads
        if weight.shape[0] != 3 * num_heads * head_dim:
            return weight

        weight = weight.view(3, num_heads, head_dim, -1)
        q = weight[0].reshape(num_heads * head_dim, -1)
        k = weight[1].reshape(num_heads * head_dim, -1)
        v = weight[2].reshape(num_heads * head_dim, -1)
        return torch.cat([q, k, v], dim=0)

    def _load_one_weight(
        self,
        params_dict: dict[str, torch.Tensor],
        stacked_params_mapping: list[tuple[str, str, str]],
        original_name: str,
        loaded_weight: torch.Tensor,
        loaded_params: set[str],
    ) -> None:
        quant_aux_suffixes = (
            ".absmax",
            ".quant_map",
            ".quant_state.",
            ".nested_absmax",
            ".nested_quant_map",
        )
        if any(suffix in original_name for suffix in quant_aux_suffixes):
            return

        lookup_name = original_name

        if ".attention.qkv.weight" in original_name:
            loaded_weight = self._remap_packed_qkv_weight(loaded_weight)
            lookup_name = original_name.replace(".attention.qkv.", ".attention.to_qkv.")
            if lookup_name in params_dict:
                param = params_dict[lookup_name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, loaded_weight)
                loaded_params.add(original_name)
                loaded_params.add(lookup_name)
                return

        for param_name, weight_name, shard_id in stacked_params_mapping:
            if weight_name not in original_name or param_name in original_name:
                continue
            lookup_name = original_name.replace(weight_name, param_name)
            if lookup_name in params_dict:
                param = params_dict[lookup_name]
                param.weight_loader(param, loaded_weight, shard_id)
                loaded_params.add(original_name)
                loaded_params.add(lookup_name)
                return

        if lookup_name not in params_dict:
            lookup_name = lookup_name.replace(".attention.o.", ".attention.to_out.")

        if lookup_name in params_dict:
            param = params_dict[lookup_name]
            weight_loader = getattr(param, "weight_loader", default_weight_loader)
            weight_loader(param, loaded_weight)
        else:
            logger.warning(
                "Could not find parameter for weight: %s (tried %s)",
                original_name,
                lookup_name,
            )

        loaded_params.add(original_name)
        loaded_params.add(lookup_name)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        """Load weights from a diffusers checkpoint."""
        stacked_params_mapping = [
            (".to_qkv", ".to_q", "q"),
            (".to_qkv", ".to_k", "k"),
            (".to_qkv", ".to_v", "v"),
        ]
        self.stacked_params_mapping = stacked_params_mapping

        params_dict = dict(self.named_parameters())
        for name, buffer in self.named_buffers():
            params_dict[name] = buffer

        all_weights = {name: tensor for name, tensor in weights}
        all_weights = self._preprocess_nf4_weights(all_weights)

        loaded_params: set[str] = set()
        for name, loaded_weight in all_weights.items():
            self._load_one_weight(
                params_dict=params_dict,
                stacked_params_mapping=stacked_params_mapping,
                original_name=name,
                loaded_weight=loaded_weight,
                loaded_params=loaded_params,
            )

        return loaded_params
