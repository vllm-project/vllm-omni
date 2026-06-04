# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Cosmos3 VFM Transformer for vllm-omni.

Implements the Mixture-of-Transformers architecture with two pathways:
- Understanding (UND): causal self-attention on text tokens (Qwen3-VL backbone)
- Generation (GEN): cross-attention where visual Q attends to [K_und, K_gen]

Ported from the TRT-LLM integration (tekit branch user/shreyasm/cosmos3).
"""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from vllm.distributed import get_tensor_model_parallel_world_size
from vllm.logger import init_logger
from vllm.model_executor.layers.linear import (
    ColumnParallelLinear,
    RowParallelLinear,
)
from vllm.model_executor.layers.quantization.base_config import (
    QuantizationConfig,
)

from vllm_omni.diffusion.attention.backends.abstract import AttentionMetadata
from vllm_omni.diffusion.attention.layer import Attention as FrameworkAttention
from vllm_omni.diffusion.data import OmniDiffusionConfig
from vllm_omni.diffusion.distributed.sp_plan import SequenceParallelInput, SequenceParallelOutput
from vllm_omni.diffusion.forward_context import get_forward_context, is_forward_context_available
from vllm_omni.diffusion.layers.norm import RMSNorm

logger = init_logger(__name__)


def _get_ulysses_state() -> tuple[int, int, dist.ProcessGroup | None]:
    """Return (ulysses_size, ulysses_rank, ulysses_pg) from vllm-omni parallel state.

    Returns (1, 0, None) when sequence parallelism is not active.
    """
    from vllm_omni.diffusion.distributed.parallel_state import (
        get_sp_group,
        get_ulysses_parallel_rank,
        get_ulysses_parallel_world_size,
    )

    size = get_ulysses_parallel_world_size()
    if size <= 1:
        return 1, 0, None
    return size, get_ulysses_parallel_rank(), get_sp_group().ulysses_group


def _is_sp_active() -> bool:
    """Check whether sequence parallelism is active in the current forward context.

    Follows the Bagel pattern: read ``forward_context.sp_active`` which returns
    True when ``sequence_parallel_size > 1`` even without ``_sp_plan`` hooks.
    """

    if not is_forward_context_available():
        return False
    return get_forward_context().sp_active


def _tf_config_get(config: Any, key: str, default: Any) -> Any:
    """Read a value from TransformerConfig, dict, or simple namespace."""
    if config is None:
        return default
    if hasattr(config, "get"):
        return config.get(key, default)
    return getattr(config, key, default)


def _nested_get(value: Any, key: str) -> Any:
    if isinstance(value, dict):
        if key in value:
            return value[key]
        for child in value.values():
            found = _nested_get(child, key)
            if found is not None:
                return found
    elif isinstance(value, list | tuple):
        for child in value:
            found = _nested_get(child, key)
            if found is not None:
                return found
    return None


def _od_config_get(od_config: Any, key: str, default: Any = None) -> Any:
    """Read Cosmos3 options from runtime, model, or transformer config."""
    if od_config is None:
        return default
    for attr in ("custom_pipeline_args", "model_config"):
        source = getattr(od_config, attr, None) or {}
        if isinstance(source, dict):
            if key in source:
                return source[key]
            found = _nested_get(source, key)
            if found is not None:
                return found
    tf_model_config = getattr(od_config, "tf_model_config", None)
    if isinstance(tf_model_config, dict):
        if key in tf_model_config:
            return tf_model_config[key]
        found = _nested_get(tf_model_config, key)
        if found is not None:
            return found
    value = _tf_config_get(tf_model_config, key, None)
    return default if value is None else value


def _as_bool(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def resolve_sound_gen(od_config: Any) -> bool:
    """Capability gate shared by the pipeline and transformer.

    Explicit ``sound_gen`` flag wins (including an explicit False);
    otherwise infer from the presence of any sound-width key in od_config.
    """
    sound_gen_value = _od_config_get(od_config, "sound_gen", None)
    if sound_gen_value is not None:
        return _as_bool(sound_gen_value)
    for key in ("sound_dim", "io_channels", "vocoder_input_dim", "latent_ch"):
        if _od_config_get(od_config, key, None) is not None:
            return True
    return False


class DomainAwareLinear(nn.Module):
    """Linear projection with one weight/bias pair per action embodiment domain."""

    def __init__(
        self,
        input_size: int,
        output_size: int,
        num_domains: int,
        *,
        dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        super().__init__()
        self.input_size = int(input_size)
        self.output_size = int(output_size)
        self.num_domains = int(num_domains)
        self.fc = nn.Embedding(self.num_domains, self.output_size * self.input_size, dtype=dtype)
        self.bias = nn.Embedding(self.num_domains, self.output_size, dtype=dtype)
        nn.init.xavier_uniform_(self.fc.weight)
        nn.init.zeros_(self.bias.weight)

    def forward(self, x: torch.Tensor, domain_id: torch.Tensor) -> torch.Tensor:
        if domain_id.ndim == 0:
            domain_id = domain_id.unsqueeze(0)
        domain_id = domain_id.to(device=x.device, dtype=torch.long).reshape(-1)
        if x.shape[0] != domain_id.shape[0]:
            raise ValueError(
                "Cosmos3 action domain_id batch size must match action tokens: "
                f"tokens={x.shape[0]}, domain_id={domain_id.shape[0]}."
            )
        if torch.any((domain_id < 0) | (domain_id >= self.num_domains)):
            raise ValueError(f"Cosmos3 action domain_id must be in [0, {self.num_domains}), got {domain_id.tolist()}.")

        weight = self.fc(domain_id).view(domain_id.shape[0], self.input_size, self.output_size)
        bias = self.bias(domain_id).view(domain_id.shape[0], self.output_size)
        if x.ndim == 2:
            return torch.bmm(x.unsqueeze(1), weight).squeeze(1) + bias
        if x.ndim == 3:
            return torch.bmm(x, weight) + bias.unsqueeze(1)
        raise ValueError(f"Cosmos3 DomainAwareLinear expected rank-2 or rank-3 input, got {tuple(x.shape)}.")


# ---------------------------------------------------------------------------
# Rotary Position Embeddings (mRoPE)
# ---------------------------------------------------------------------------
def compute_mrope_position_ids_text(
    num_tokens: int,
    temporal_offset: int,
) -> tuple[torch.Tensor, int]:
    """Generate 3D mRoPE position IDs for text tokens.

    Text tokens: all three axes (t, h, w) share the same monotonically
    increasing position IDs.
    """
    ids = torch.arange(num_tokens, dtype=torch.long) + temporal_offset
    mrope_ids = ids.unsqueeze(0).expand(3, -1).contiguous()
    return mrope_ids, temporal_offset + num_tokens


def compute_mrope_position_ids_vision(
    grid_t: int,
    grid_h: int,
    grid_w: int,
    temporal_offset: int | float,
    fps: float | None = None,
    base_fps: float = 24.0,
    temporal_compression_factor: int = 4,
    base_temporal_compression_factor: int | None = None,
    enable_fps_modulation: bool = True,
    start_frame_offset: int = 0,
) -> tuple[torch.Tensor, int | float]:
    """Generate 3D mRoPE position IDs for vision tokens.

    Creates a (t, h, w) position grid with spatial indices reset per segment
    (Qwen3VL-style). Flattened in t-major order.
    """
    fps_modulation = enable_fps_modulation and fps is not None

    if fps_modulation:
        tps = fps / temporal_compression_factor
        effective_base_tcf = (
            base_temporal_compression_factor
            if base_temporal_compression_factor is not None
            else temporal_compression_factor
        )
        base_tps = base_fps / effective_base_tcf
        frame_indices = torch.arange(grid_t, dtype=torch.float32)
        t_index = (
            ((frame_indices + start_frame_offset) / tps * base_tps + temporal_offset)
            .view(-1, 1)
            .expand(-1, grid_h * grid_w)
            .flatten()
        )
    else:
        t_index = (
            torch.arange(grid_t, dtype=torch.long).view(-1, 1).expand(-1, grid_h * grid_w).flatten()
            + int(temporal_offset)
            + start_frame_offset
        )

    h_index = torch.arange(grid_h, dtype=torch.long).view(1, -1, 1).expand(grid_t, -1, grid_w).flatten()
    w_index = torch.arange(grid_w, dtype=torch.long).view(1, 1, -1).expand(grid_t, grid_h, -1).flatten()

    if fps_modulation:
        mrope_ids = torch.stack([t_index, h_index.to(torch.float32), w_index.to(torch.float32)], dim=0)
    else:
        mrope_ids = torch.stack([t_index, h_index, w_index], dim=0)

    next_offset = math.floor(mrope_ids.max().item()) + 1
    return mrope_ids, next_offset


def compute_mrope_position_ids_sound(
    grid_t: int,
    temporal_offset: int | float,
    sound_latent_fps: float,
    base_fps: float = 24.0,
    temporal_compression_factor_sound: int = 1,
    enable_fps_modulation: bool = True,
) -> tuple[torch.Tensor, int | float]:
    """Generate mRoPE IDs for sound tokens as a (T, 1, 1) grid."""
    return compute_mrope_position_ids_vision(
        grid_t=grid_t,
        grid_h=1,
        grid_w=1,
        temporal_offset=temporal_offset,
        fps=sound_latent_fps,
        base_fps=base_fps,
        temporal_compression_factor=temporal_compression_factor_sound,
        base_temporal_compression_factor=temporal_compression_factor_sound,
        enable_fps_modulation=enable_fps_modulation,
    )


def compute_mrope_position_ids_action(
    grid_t: int,
    temporal_offset: int | float,
    action_fps: float | None,
    base_fps: float = 24.0,
    base_temporal_compression_factor: int = 4,
    enable_fps_modulation: bool = True,
    start_frame_offset: int = 1,
) -> tuple[torch.Tensor, int | float]:
    """Generate mRoPE IDs for action tokens as a frame-rate (T, 1, 1) grid."""
    return compute_mrope_position_ids_vision(
        grid_t=grid_t,
        grid_h=1,
        grid_w=1,
        temporal_offset=temporal_offset,
        fps=action_fps,
        base_fps=base_fps,
        temporal_compression_factor=1,
        base_temporal_compression_factor=base_temporal_compression_factor,
        enable_fps_modulation=enable_fps_modulation,
        start_frame_offset=start_frame_offset,
    )


class Qwen3VLTextRotaryEmbedding(nn.Module):
    """Multi-dimensional rotary position embedding for Qwen3-VL."""

    def __init__(
        self,
        *,
        head_dim: int,
        rope_theta: float,
        mrope_section: list[int],
    ) -> None:
        super().__init__()
        self.head_dim = head_dim
        self.rope_theta = rope_theta
        self.mrope_section = mrope_section
        inv_freq = 1.0 / (
            rope_theta ** (torch.arange(0, head_dim, 2, dtype=torch.int64).to(dtype=torch.float) / head_dim)
        )
        self.attention_scaling = 1.0
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def apply_interleaved_mrope(self, freqs: torch.Tensor, mrope_section: list[int]) -> torch.Tensor:
        """Reorganize from chunked [TTT...HHH...WWW] to interleaved [THTHW...]."""
        freqs_t = freqs[0]
        for dim, offset in enumerate((1, 2), start=1):
            length = mrope_section[dim] * 3
            idx = slice(offset, length, 3)
            freqs_t[..., idx] = freqs[dim, ..., idx]
        return freqs_t

    @torch.no_grad()
    def forward(self, x: torch.Tensor, position_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if position_ids.ndim == 2:
            position_ids = position_ids[None, ...].expand(3, position_ids.shape[0], -1)
        inv_freq_expanded = (
            self.inv_freq[None, None, :, None].float().expand(3, position_ids.shape[1], -1, 1).to(x.device)
        )
        position_ids_expanded = position_ids[:, :, None, :].float()

        freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(2, 3)
        freqs = self.apply_interleaved_mrope(freqs, self.mrope_section)
        emb = torch.cat((freqs, freqs), dim=-1)
        cos = emb.cos() * self.attention_scaling
        sin = emb.sin() * self.attention_scaling
        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


# ---------------------------------------------------------------------------
# RoPE application (Qwen3/Llama style)
# ---------------------------------------------------------------------------
def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def _apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Qwen3-style RoPE: (x * cos) + (rotate_half(x) * sin).

    Args:
        q: [B, S, h, D]
        k: [B, S, H_kv, D]
        cos: [1, S, 1, D] or broadcastable
        sin: [1, S, 1, D] or broadcastable
    """
    q_embed = (q * cos) + (_rotate_half(q) * sin)
    k_embed = (k * cos) + (_rotate_half(k) * sin)
    return q_embed, k_embed


# ---------------------------------------------------------------------------
# Timestep Embedder
# ---------------------------------------------------------------------------
class TimestepEmbedder(nn.Module):
    """Embeds scalar timesteps into vector representations."""

    def __init__(
        self,
        hidden_size: int,
        frequency_embedding_size: int = 256,
        max_period: int = 10000,
        target_dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        super().__init__()
        # Following diffusers naming pattern here for checkpoint compatibility.
        self.linear_1 = nn.Linear(frequency_embedding_size, hidden_size, bias=True)
        self.act = nn.SiLU()
        self.linear_2 = nn.Linear(hidden_size, hidden_size, bias=True)
        self.frequency_embedding_size = frequency_embedding_size
        self.hidden_size = hidden_size

        half = frequency_embedding_size // 2
        freqs = torch.exp(-math.log(max_period) * torch.arange(start=0, end=half, dtype=target_dtype) / half)
        self.register_buffer("freqs", freqs, persistent=False)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        args = t[:, None] * self.freqs[None]
        t_freq = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        return self.linear_2(self.act(self.linear_1(t_freq)))


# ---------------------------------------------------------------------------
# GatedMLP (replaces TRT-LLM GatedMLP)
# ---------------------------------------------------------------------------
class Cosmos3GatedMLP(nn.Module):
    def __init__(
        self,
        hidden_size: int = 4096,
        intermediate_size: int = 12288,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.gate_proj = ColumnParallelLinear(
            hidden_size,
            intermediate_size,
            bias=False,
            gather_output=False,
            return_bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.gate_proj",
        )
        self.up_proj = ColumnParallelLinear(
            hidden_size,
            intermediate_size,
            bias=False,
            gather_output=False,
            return_bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.up_proj",
        )
        self.down_proj = RowParallelLinear(
            intermediate_size,
            hidden_size,
            bias=False,
            input_is_parallel=True,
            return_bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.down_proj",
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


# ---------------------------------------------------------------------------
# Attention Modules
# ---------------------------------------------------------------------------
class Cosmos3CausalAttention(nn.Module):
    """Understanding pathway: causal self-attention on text tokens.

    Returns (output, K, V) where K/V are post-norm, post-RoPE for the
    generation pathway's cross-attention.
    """

    def __init__(
        self,
        *,
        hidden_size: int,
        num_attention_heads: int,
        num_key_value_heads: int,
        head_dim: int,
        rms_norm_eps: float,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_attention_heads
        self.num_kv_heads = num_key_value_heads
        self.head_dim = head_dim

        tp_size = get_tensor_model_parallel_world_size()
        self.num_heads_local = self.num_heads // tp_size
        self.num_kv_heads_local = self.num_kv_heads // tp_size

        self.to_q = ColumnParallelLinear(
            hidden_size,
            self.num_heads * self.head_dim,
            bias=False,
            gather_output=False,
            return_bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.to_q",
        )
        self.to_k = ColumnParallelLinear(
            hidden_size,
            self.num_kv_heads * self.head_dim,
            bias=False,
            gather_output=False,
            return_bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.to_k",
        )
        self.to_v = ColumnParallelLinear(
            hidden_size,
            self.num_kv_heads * self.head_dim,
            bias=False,
            gather_output=False,
            return_bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.to_v",
        )
        self.to_out = RowParallelLinear(
            self.num_heads * self.head_dim,
            hidden_size,
            bias=False,
            input_is_parallel=True,
            return_bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.to_out",
        )

        self.norm_q = RMSNorm(self.head_dim, eps=rms_norm_eps)
        self.norm_k = RMSNorm(self.head_dim, eps=rms_norm_eps)

        # skip_sequence_parallel=True because the UND pathway is
        # computed once and replicated across SP ranks.
        # Only the GEN pathway is sequence-sharded.
        self.attn = FrameworkAttention(
            num_heads=self.num_heads,
            head_size=self.head_dim,
            causal=True,
            softmax_scale=1.0 / (self.head_dim**0.5),
            num_kv_heads=self.num_kv_heads,
            skip_sequence_parallel=True,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        freqs_cos: torch.Tensor,
        freqs_sin: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        B, S, _ = hidden_states.shape

        q = self.to_q(hidden_states).view(B, S, self.num_heads_local, self.head_dim)
        k = self.to_k(hidden_states).view(B, S, self.num_kv_heads_local, self.head_dim)
        v = self.to_v(hidden_states).view(B, S, self.num_kv_heads_local, self.head_dim)

        # Per-head QK norm
        q = F.rms_norm(q, (q.shape[-1],), self.norm_q.weight, self.norm_q.variance_epsilon)
        k = F.rms_norm(k, (k.shape[-1],), self.norm_k.weight, self.norm_k.variance_epsilon)

        # Qwen3-style RoPE
        q, k = _apply_rotary_pos_emb(q, k, freqs_cos, freqs_sin)

        out = self.attn(q, k, v).reshape(B, S, -1)
        return self.to_out(out), k, v


class Cosmos3CrossAttention(nn.Module):
    """Generation pathway: full attention where visual Q attends to all K/V.

    * **Non-SP path**: explicit ``cat([k_und, k_gen])``.  Text conditioning is
      always present because K/V are physically concatenated.

    * **SP path** (Ulysses active): ``k_und``/``v_und`` are passed as
      ``joint_key``/``joint_value`` in ``AttentionMetadata``.  The Ulysses
      wrapper head-slices the replicated UND K/V and performs all-to-all on the
      sharded GEN Q/K/V so every query sees the full context.
    """

    def __init__(
        self,
        *,
        hidden_size: int,
        num_attention_heads: int,
        num_key_value_heads: int,
        head_dim: int,
        rms_norm_eps: float,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_attention_heads
        self.num_kv_heads = num_key_value_heads
        self.head_dim = head_dim

        tp_size = get_tensor_model_parallel_world_size()
        self.num_heads_local = self.num_heads // tp_size
        self.num_kv_heads_local = self.num_kv_heads // tp_size

        self.to_q = ColumnParallelLinear(
            hidden_size,
            self.num_heads * self.head_dim,
            bias=False,
            gather_output=False,
            return_bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.to_q",
        )
        self.to_k = ColumnParallelLinear(
            hidden_size,
            self.num_kv_heads * self.head_dim,
            bias=False,
            gather_output=False,
            return_bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.to_k",
        )
        self.to_v = ColumnParallelLinear(
            hidden_size,
            self.num_kv_heads * self.head_dim,
            bias=False,
            gather_output=False,
            return_bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.to_v",
        )
        self.to_out = RowParallelLinear(
            self.num_heads * self.head_dim,
            hidden_size,
            bias=False,
            input_is_parallel=True,
            return_bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.to_out",
        )

        self.norm_q = RMSNorm(self.head_dim, eps=rms_norm_eps)
        self.norm_k = RMSNorm(self.head_dim, eps=rms_norm_eps)

        self.attn = FrameworkAttention(
            num_heads=self.num_heads,
            head_size=self.head_dim,
            causal=False,
            softmax_scale=1.0 / (self.head_dim**0.5),
            num_kv_heads=self.num_kv_heads,
        )

    # TODO(follow-up): collapse _forward_local and _forward_sp into a single
    # joint-based path when NoParallelAttention can process joint_key/value.
    # Currently the non-SP path must concatenate the replicated UND K/V explicitly,
    # while the SP path passes it as joint_*.

    # -- Non-SP path: explicit K/V concatenation + framework Attention --------

    def _forward_local(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        k_und: torch.Tensor,
        v_und: torch.Tensor,
    ) -> torch.Tensor:
        B, S_gen = q.shape[:2]
        k_all = torch.cat([k_und, k], dim=1)
        v_all = torch.cat([v_und, v], dim=1)

        out = self.attn(q, k_all, v_all)
        return out.reshape(B, S_gen, -1)

    # -- SP path: framework Attention with joint_key/value -------------------

    def _forward_sp(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        k_und: torch.Tensor,
        v_und: torch.Tensor,
    ) -> torch.Tensor:
        B, S_gen = q.shape[:2]

        # Zero-length joint_query satisfies the Ulysses contract
        # (joint_query, joint_key, joint_value must all be present) without
        # adding text tokens to Q.  joint_len=0 keeps post_attention on the
        # standard reverse-all-to-all path (no joint-output splitting).
        joint_q = q.new_empty(B, 0, self.num_heads_local, self.head_dim)

        attn_metadata = AttentionMetadata(
            joint_query=joint_q,
            joint_key=k_und,
            joint_value=v_und,
            joint_strategy="front",
        )
        out = self.attn(q, k, v, attn_metadata)
        return out.reshape(B, S_gen, -1)

    # -- Public forward: routes to the appropriate path ----------------------

    def forward(
        self,
        hidden_states: torch.Tensor,
        k_und: torch.Tensor,
        v_und: torch.Tensor,
        freqs_cos: torch.Tensor,
        freqs_sin: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            hidden_states: [B, S_gen_local, hidden_size] (may be sequence-sharded)
            k_und: [B, S_und, H_kv_local, D] pre-computed UND keys (TP-sharded, post-norm/RoPE)
            v_und: [B, S_und, H_kv_local, D] pre-computed UND values (TP-sharded)
            freqs_cos: [B, S_gen_local, 1, D]
            freqs_sin: [B, S_gen_local, 1, D]
        """
        B, S_gen, _ = hidden_states.shape

        q = self.to_q(hidden_states).view(B, S_gen, self.num_heads_local, self.head_dim)
        k = self.to_k(hidden_states).view(B, S_gen, self.num_kv_heads_local, self.head_dim)
        v = self.to_v(hidden_states).view(B, S_gen, self.num_kv_heads_local, self.head_dim)

        # Per-head QK norm
        q = F.rms_norm(q, (q.shape[-1],), self.norm_q.weight, self.norm_q.variance_epsilon)
        k = F.rms_norm(k, (k.shape[-1],), self.norm_k.weight, self.norm_k.variance_epsilon)

        # Qwen3-style RoPE
        q, k = _apply_rotary_pos_emb(q, k, freqs_cos, freqs_sin)

        if _is_sp_active():
            out = self._forward_sp(q, k, v, k_und, v_und)
        else:
            out = self._forward_local(q, k, v, k_und, v_und)

        return self.to_out(out)


# ---------------------------------------------------------------------------
# Decoder Layers
# ---------------------------------------------------------------------------
class Cosmos3UndDecoderLayer(nn.Module):
    """Understanding pathway decoder layer: causal self-attention + MLP."""

    def __init__(
        self,
        *,
        hidden_size: int,
        intermediate_size: int,
        num_attention_heads: int,
        num_key_value_heads: int,
        head_dim: int,
        rms_norm_eps: float,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.self_attn = Cosmos3CausalAttention(
            hidden_size=hidden_size,
            num_attention_heads=num_attention_heads,
            num_key_value_heads=num_key_value_heads,
            head_dim=head_dim,
            rms_norm_eps=rms_norm_eps,
            quant_config=quant_config,
            prefix=f"{prefix}.self_attn",
        )
        self.input_layernorm = RMSNorm(hidden_size, eps=rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(hidden_size, eps=rms_norm_eps)
        self.mlp = Cosmos3GatedMLP(
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            quant_config=quant_config,
            prefix=f"{prefix}.mlp",
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        freqs: tuple[torch.Tensor, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns (hidden_states, K, V) where K/V are for GEN cross-attention."""
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        cos, sin = freqs
        attn_out, k, v = self.self_attn(hidden_states, cos, sin)
        hidden_states = residual + attn_out

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = residual + self.mlp(hidden_states)

        return hidden_states, k, v


class Cosmos3GenDecoderLayer(nn.Module):
    """Generation pathway decoder layer: cross-attention (to UND K/V) + MLP."""

    def __init__(
        self,
        *,
        layer_idx: int | None = None,
        hidden_size: int,
        intermediate_size: int,
        num_attention_heads: int,
        num_key_value_heads: int,
        head_dim: int,
        rms_norm_eps: float,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.layer_idx = layer_idx
        self.cross_attention = Cosmos3CrossAttention(
            hidden_size=hidden_size,
            num_attention_heads=num_attention_heads,
            num_key_value_heads=num_key_value_heads,
            head_dim=head_dim,
            rms_norm_eps=rms_norm_eps,
            quant_config=quant_config,
            prefix=f"{prefix}.cross_attention",
        )
        self.input_layernorm = RMSNorm(hidden_size, eps=rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(hidden_size, eps=rms_norm_eps)
        self.mlp = Cosmos3GatedMLP(
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            quant_config=quant_config,
            prefix=f"{prefix}.mlp",
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        *,
        k_und: torch.Tensor | None = None,
        v_und: torch.Tensor | None = None,
        freqs_cos: torch.Tensor | None = None,
        freqs_sin: torch.Tensor | None = None,
        cached_kv: list[tuple[torch.Tensor, torch.Tensor]] | None = None,
        freqs_gen: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        if cached_kv is not None:
            if self.layer_idx is None:
                raise ValueError("Cosmos3 GEN layer requires layer_idx when cached_kv is provided.")
            k_und, v_und = cached_kv[self.layer_idx]
        if freqs_gen is not None:
            freqs_cos, freqs_sin = freqs_gen
        if k_und is None or v_und is None or freqs_cos is None or freqs_sin is None:
            raise ValueError("Cosmos3 GEN layer requires k_und/v_und/freqs_cos/freqs_sin.")

        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        hidden_states = self.cross_attention(
            hidden_states, k_und=k_und, v_und=v_und, freqs_cos=freqs_cos, freqs_sin=freqs_sin
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = residual + self.mlp(hidden_states)

        return hidden_states


# ---------------------------------------------------------------------------
# Language Model (Understanding pathway)
# ---------------------------------------------------------------------------
class Cosmos3LanguageModel(nn.Module):
    """Understanding pathway: a standard causal LM that processes text tokens.

    Returns per-layer K/V tensors for the generation pathway's cross-attention.
    The UND pathway is independent of the denoising step, so its K/V can be
    computed once and reused across all sampling steps.
    """

    def __init__(
        self,
        *,
        hidden_size: int,
        intermediate_size: int,
        num_hidden_layers: int,
        num_attention_heads: int,
        num_key_value_heads: int,
        head_dim: int,
        vocab_size: int,
        rms_norm_eps: float,
        rope_theta: float,
        mrope_section: list[int],
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.embed_tokens = nn.Embedding(vocab_size, hidden_size)
        self.rotary_emb = Qwen3VLTextRotaryEmbedding(
            head_dim=head_dim,
            rope_theta=rope_theta,
            mrope_section=mrope_section,
        )
        self.layers = nn.ModuleList(
            [
                Cosmos3UndDecoderLayer(
                    hidden_size=hidden_size,
                    intermediate_size=intermediate_size,
                    num_attention_heads=num_attention_heads,
                    num_key_value_heads=num_key_value_heads,
                    head_dim=head_dim,
                    rms_norm_eps=rms_norm_eps,
                    quant_config=quant_config,
                    prefix=f"{prefix}.layers.{i}",
                )
                for i in range(num_hidden_layers)
            ]
        )
        # TODO: Not used right now, will be used in the future for prompt upsampler.
        self.norm = RMSNorm(hidden_size, eps=rms_norm_eps)

    def forward(
        self,
        text_ids: torch.Tensor,
        freqs: tuple[torch.Tensor, torch.Tensor],
    ) -> list[tuple[torch.Tensor, torch.Tensor]]:
        """
        Args:
            text_ids: [B, S] token IDs
            freqs: (cos, sin) each [B, S, 1, D]

        Returns:
            List of (K, V) per layer, each [B, S, H_kv, D].

        No padding mask is applied: with right-padding + causal self-attention,
        real query positions only attend to real keys, and the caller trims pad
        K/V via ``max_real_len`` before the GEN cross-attention sees them.
        """
        hidden = self.embed_tokens(text_ids)

        cached_kv: list[tuple[torch.Tensor, torch.Tensor]] = []
        for layer in self.layers:
            hidden, k, v = layer(hidden, freqs)
            cached_kv.append((k, v))

        return cached_kv


# ---------------------------------------------------------------------------
# Main Transformer
# ---------------------------------------------------------------------------
class Cosmos3GenSPPrepare(nn.Module):
    """Module boundary used by _sp_plan to shard GEN states and RoPE together."""

    def forward(
        self,
        hidden_gen: torch.Tensor,
        freqs_cos: torch.Tensor,
        freqs_sin: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return hidden_gen, freqs_cos, freqs_sin


class Cosmos3VFMTransformer(nn.Module):
    """Cosmos3 VFM Transformer: UND language model + GEN denoising layers.

    The UND pathway runs once per generation (K/V cached). The GEN pathway
    runs at each denoising step.

    Layerwise offloading uses ``gen_layers`` as the block container.

    Sequence parallelism uses ``_sp_plan`` to shard/gather the GEN pathway at
    module boundaries. ``Cosmos3CrossAttention`` checks
    ``forward_context.sp_active`` at runtime and routes to the framework
    ``Attention`` layer (with Ulysses all-to-all) or plain SDPA accordingly.
    """

    _repeated_blocks = ["Cosmos3GenDecoderLayer"]

    _layerwise_offload_blocks_attrs = ["gen_layers"]

    packed_modules_mapping = {}

    @staticmethod
    def _is_transformer_block(name: str, module) -> bool:
        return ("gen_layers" in name or "language_model.layers" in name) and name.split(".")[-1].isdigit()

    _hsdp_shard_conditions = [_is_transformer_block]

    _sp_plan = {
        "gen_sp_prepare": {
            0: SequenceParallelInput(split_dim=1, expected_dims=3, split_output=True),
            1: SequenceParallelInput(split_dim=1, expected_dims=4, split_output=True),
            2: SequenceParallelInput(split_dim=1, expected_dims=4, split_output=True),
        },
        "gen_sp_gather": SequenceParallelOutput(gather_dim=1, expected_dims=3),
    }

    @staticmethod
    def _validate_supported_config(model_config: Any) -> None:
        """Fail loudly when a checkpoint requests an unsupported architecture."""
        expected_values = {
            "qk_norm_for_diffusion": True,
            "qk_norm_for_text": True,
            "position_embedding_type": "unified_3d_mrope",
            "unified_3d_mrope_reset_spatial_ids": True,
            "joint_attn_implementation": "two_way",
        }
        for key, expected in expected_values.items():
            actual = _tf_config_get(model_config, key, expected)
            if actual != expected:
                raise ValueError(f"Unsupported Cosmos3 transformer config: {key}={actual!r}; expected {expected!r}.")

    def __init__(
        self,
        od_config: OmniDiffusionConfig,
        *,
        temporal_compression_factor: int | None = None,
        sound_gen: bool = False,
        sound_dim: int | None = None,
        sound_latent_fps: float | None = None,
    ) -> None:
        super().__init__()
        model_config = od_config.tf_model_config
        self._validate_supported_config(model_config)
        rope_scaling = _tf_config_get(model_config, "rope_scaling", {}) or {}

        self.hidden_size = int(_tf_config_get(model_config, "hidden_size", 4096))
        self.num_hidden_layers = int(_tf_config_get(model_config, "num_hidden_layers", 36))
        self.num_attention_heads = int(_tf_config_get(model_config, "num_attention_heads", 32))
        self.num_key_value_heads = int(_tf_config_get(model_config, "num_key_value_heads", 8))
        self.head_dim = int(_tf_config_get(model_config, "head_dim", 128))
        self.intermediate_size = int(_tf_config_get(model_config, "intermediate_size", 12288))
        self.vocab_size = int(_tf_config_get(model_config, "vocab_size", 151936))
        self.rms_norm_eps = float(_tf_config_get(model_config, "rms_norm_eps", 1e-6))
        self.rope_theta = float(_tf_config_get(model_config, "rope_theta", 5_000_000))
        self.mrope_section = list(rope_scaling.get("mrope_section", [24, 20, 20]))
        self.latent_patch_size = int(_tf_config_get(model_config, "latent_patch_size", 2))
        self.latent_channel_size = int(_tf_config_get(model_config, "latent_channel", 48))
        self.timestep_scale = float(_tf_config_get(model_config, "timestep_scale", 0.001))
        self.base_fps = float(_tf_config_get(model_config, "base_fps", 24.0))
        self.sound_gen = sound_gen
        self.sound_dim = sound_dim
        self.sound_latent_fps = sound_latent_fps
        if self.sound_gen and (sound_dim is None or sound_latent_fps is None):
            raise ValueError(
                "Cosmos3VFMTransformer requires an explicit sound_dim and sound_latent_fps when sound_gen is True; "
                "the pipeline must pass Cosmos3SoundTokenizer.latent_ch so the audio projection "
                "layers are sized from the authoritative AVAE latent width."
            )
        action_gen_value = _od_config_get(od_config, "action_gen", None)
        action_dim_value = _od_config_get(od_config, "action_dim", None)
        if action_dim_value is None:
            action_dim_value = _od_config_get(od_config, "max_action_dim", None)
        self.action_gen = _as_bool(action_gen_value) if action_gen_value is not None else False
        self.action_dim = int(action_dim_value if action_dim_value is not None else 64)
        self.num_embodiment_domains = int(_od_config_get(od_config, "num_embodiment_domains", 32))
        if temporal_compression_factor is None:
            temporal_compression_factor = _tf_config_get(model_config, "temporal_compression_factor", 4)
        self.temporal_compression_factor = int(temporal_compression_factor)
        self.temporal_compression_factor_sound = int(
            _tf_config_get(model_config, "temporal_compression_factor_sound", 1)
        )
        self.temporal_compression_factor_sound = int(
            _tf_config_get(model_config, "temporal_compression_factor_sound", 1)
        )
        self.enable_fps_modulation = bool(_tf_config_get(model_config, "enable_fps_modulation", True))
        self.temporal_modality_margin = int(
            _tf_config_get(
                model_config,
                "unified_3d_mrope_temporal_modality_margin",
                15000,
            )
        )
        self.patch_latent_dim = (self.latent_patch_size**2) * self.latent_channel_size

        dtype = od_config.dtype
        quant_config = getattr(od_config, "quantization_config", None) if od_config else None

        self.language_model = Cosmos3LanguageModel(
            hidden_size=self.hidden_size,
            intermediate_size=self.intermediate_size,
            num_hidden_layers=self.num_hidden_layers,
            num_attention_heads=self.num_attention_heads,
            num_key_value_heads=self.num_key_value_heads,
            head_dim=self.head_dim,
            vocab_size=self.vocab_size,
            rms_norm_eps=self.rms_norm_eps,
            rope_theta=self.rope_theta,
            mrope_section=self.mrope_section,
            quant_config=quant_config,
            prefix="language_model",
        )

        # Video projection layers are small; not worth quantizing.
        self.proj_in = nn.Linear(self.patch_latent_dim, self.hidden_size)
        self.proj_out = nn.Linear(self.hidden_size, self.patch_latent_dim)
        self.time_embedder = TimestepEmbedder(self.hidden_size, target_dtype=dtype)
        if self.action_gen:
            self.action_proj_in = DomainAwareLinear(
                self.action_dim,
                self.hidden_size,
                self.num_embodiment_domains,
                dtype=dtype,
            )
            self.action_proj_out = DomainAwareLinear(
                self.hidden_size,
                self.action_dim,
                self.num_embodiment_domains,
                dtype=dtype,
            )
            self.action_modality_embed = nn.Parameter(torch.zeros(self.hidden_size, dtype=dtype))
        if self.sound_gen:
            self.audio_proj_in = nn.Linear(self.sound_dim, self.hidden_size)
            self.audio_proj_out = nn.Linear(self.hidden_size, self.sound_dim)
            self.audio_modality_embed = nn.Parameter(torch.zeros(self.hidden_size))

        self.gen_layers = nn.ModuleList(
            [
                Cosmos3GenDecoderLayer(
                    layer_idx=i,
                    hidden_size=self.hidden_size,
                    intermediate_size=self.intermediate_size,
                    num_attention_heads=self.num_attention_heads,
                    num_key_value_heads=self.num_key_value_heads,
                    head_dim=self.head_dim,
                    rms_norm_eps=self.rms_norm_eps,
                    quant_config=quant_config,
                    prefix=f"gen_layers.{i}",
                )
                for i in range(self.num_hidden_layers)
            ]
        )

        self.norm_moe_gen = RMSNorm(self.hidden_size, eps=self.rms_norm_eps)
        self.gen_sp_prepare = Cosmos3GenSPPrepare()
        self.gen_sp_gather = nn.Identity()

        # Cached state (populated on first forward, reused across denoising steps)
        self.cached_kv: list[tuple[torch.Tensor, torch.Tensor]] | None = None
        self.cached_freqs_gen: tuple[torch.Tensor, torch.Tensor] | None = None

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    # -- Patchify / Unpatchify -----------------------------------------------

    def _pad_to_patch_size(self, h: int, w: int) -> tuple[int, int, int, int]:
        """Returns (hp, wp, H_padded, W_padded)."""
        p = self.latent_patch_size
        H_padded = ((h + p - 1) // p) * p
        W_padded = ((w + p - 1) // p) * p
        return H_padded // p, W_padded // p, H_padded, W_padded

    def patchify(self, latents: torch.Tensor, t: int, h: int, w: int) -> torch.Tensor:
        """[B, C, t, h, w] -> [B, t*hp*wp, p*p*C], padding h/w if needed."""
        B = latents.shape[0]
        p = self.latent_patch_size
        C = self.latent_channel_size
        hp, wp, H_padded, W_padded = self._pad_to_patch_size(h, w)

        if H_padded != h or W_padded != w:
            latents = F.pad(latents, (0, W_padded - w, 0, H_padded - h))

        x = latents.reshape(B, C, t, hp, p, wp, p)
        x = x.permute(0, 2, 3, 5, 4, 6, 1)  # [B, t, hp, wp, p, p, C]
        return x.reshape(B, t * hp * wp, p * p * C)

    def unpatchify(self, tokens: torch.Tensor, t: int, h: int, w: int) -> torch.Tensor:
        """[B, t*hp*wp, p*p*C] -> [B, C, t, h, w], cropping padding if needed."""
        B = tokens.shape[0]
        p = self.latent_patch_size
        C = self.latent_channel_size
        hp, wp, H_padded, W_padded = self._pad_to_patch_size(h, w)

        x = tokens.reshape(B, t, hp, wp, p, p, C)
        x = x.permute(0, 6, 1, 2, 4, 3, 5)  # [B, C, t, hp, p, wp, p]
        x = x.reshape(B, C, t, H_padded, W_padded)

        if H_padded != h or W_padded != w:
            x = x[:, :, :, :h, :w]
        return x

    def pack_sound(self, sound_latents: torch.Tensor) -> torch.Tensor:
        """[B, C_sound, T_sound] -> [B, T_sound, C_sound]."""
        if sound_latents.ndim != 3:
            raise ValueError(f"Cosmos3 sound latents must have shape [B, C, T], got {tuple(sound_latents.shape)}.")
        if sound_latents.shape[1] != self.sound_dim:
            raise ValueError(
                f"Cosmos3 sound latent channel mismatch: expected {self.sound_dim}, got {sound_latents.shape[1]}."
            )
        return sound_latents.permute(0, 2, 1).contiguous()

    @staticmethod
    def unpack_sound(tokens: torch.Tensor) -> torch.Tensor:
        """[B, T_sound, C_sound] -> [B, C_sound, T_sound]."""
        return tokens.permute(0, 2, 1).contiguous()

    def pack_action(self, action_latents: torch.Tensor) -> torch.Tensor:
        """Validate and return action latents as [B, T_action, D_action] tokens."""
        if action_latents.ndim != 3:
            raise ValueError(f"Cosmos3 action latents must have shape [B, T, D], got {tuple(action_latents.shape)}.")
        if action_latents.shape[-1] != self.action_dim:
            raise ValueError(
                f"Cosmos3 action latent dimension mismatch: expected {self.action_dim}, got {action_latents.shape[-1]}."
            )
        return action_latents.contiguous()

    @staticmethod
    def unpack_action(tokens: torch.Tensor) -> torch.Tensor:
        """Return [B, T_action, D_action] action predictions."""
        return tokens.contiguous()

    # -- RoPE computation ----------------------------------------------------

    def _compute_rope_freqs(
        self,
        text_mask: torch.Tensor,
        t: int,
        hp: int,
        wp: int,
        fps: float | None,
        device: torch.device,
        dtype: torch.dtype,
        t_action: int | None = None,
        action_start_frame_offset: int = 1,
        action_fps: float | None = None,
        t_sound: int | None = None,
    ) -> tuple[tuple[torch.Tensor, torch.Tensor], tuple[torch.Tensor, torch.Tensor]]:
        """Compute mRoPE cos/sin for UND text and GEN media pathways."""
        B = text_mask.shape[0]
        S_text = text_mask.shape[1]
        text_lengths = text_mask.sum(dim=1).long()
        effective_fps = fps if fps is not None and t > 1 else None
        action_frames = int(t_action or 0)
        sound_frames = int(t_sound or 0)

        text_pos_list = []
        gen_pos_list = []
        for b in range(B):
            real_len = int(text_lengths[b].item())
            t_pos, t_offset = compute_mrope_position_ids_text(real_len, temporal_offset=0)
            media_temporal_offset = t_offset + self.temporal_modality_margin
            v_pos, _ = compute_mrope_position_ids_vision(
                t,
                hp,
                wp,
                temporal_offset=media_temporal_offset,
                fps=effective_fps,
                base_fps=self.base_fps,
                temporal_compression_factor=self.temporal_compression_factor,
                enable_fps_modulation=self.enable_fps_modulation,
            )
            gen_positions = [v_pos]
            if action_frames > 0:
                a_pos, _ = compute_mrope_position_ids_action(
                    action_frames,
                    temporal_offset=media_temporal_offset,
                    action_fps=action_fps if action_fps is not None else fps,
                    base_fps=self.base_fps,
                    base_temporal_compression_factor=self.temporal_compression_factor,
                    enable_fps_modulation=self.enable_fps_modulation,
                    start_frame_offset=action_start_frame_offset,
                )
                gen_positions.append(a_pos)
            if sound_frames > 0:
                s_pos, _ = compute_mrope_position_ids_sound(
                    sound_frames,
                    temporal_offset=media_temporal_offset,
                    sound_latent_fps=self.sound_latent_fps,
                    base_fps=self.base_fps,
                    temporal_compression_factor_sound=getattr(self, "temporal_compression_factor_sound", 1),
                    enable_fps_modulation=self.enable_fps_modulation,
                )
                gen_positions.append(s_pos)
            pos_dtype = gen_positions[0].dtype
            for pos in gen_positions[1:]:
                pos_dtype = torch.promote_types(pos_dtype, pos.dtype)
            v_pos = torch.cat([pos.to(pos_dtype) for pos in gen_positions], dim=1)
            if real_len < S_text:
                t_pos = torch.cat(
                    [t_pos, torch.zeros(3, S_text - real_len, dtype=t_pos.dtype)],
                    dim=1,
                )
            text_pos_list.append(t_pos)
            gen_pos_list.append(v_pos)

        text_pos_ids = torch.stack(text_pos_list, dim=1).to(device)  # [3, B, S_text]
        gen_pos_ids = torch.stack(gen_pos_list, dim=1).to(device)  # [3, B, S_gen]

        rotary_emb = self.language_model.rotary_emb
        _dummy = torch.tensor([], dtype=dtype, device=device)
        cos_und, sin_und = rotary_emb(_dummy, position_ids=text_pos_ids)
        cos_gen, sin_gen = rotary_emb(_dummy, position_ids=gen_pos_ids)

        freqs_und = (cos_und.unsqueeze(2), sin_und.unsqueeze(2))  # (B, S, 1, D)
        freqs_gen = (cos_gen.unsqueeze(2), sin_gen.unsqueeze(2))
        return freqs_und, freqs_gen

    # -- Cache management ----------------------------------------------------

    def reset_cache(self) -> None:
        self.cached_kv = None
        self.cached_freqs_gen = None

    @staticmethod
    def _validate_gen_sequence_parallel(
        *,
        s_gen: int,
        s_video: int,
        s_action: int,
        s_sound: int,
        has_action: bool,
        has_sound: bool,
        ulysses_size: int,
    ) -> None:
        if ulysses_size <= 1 or s_gen % ulysses_size == 0:
            return

        detail_parts = [f"video tokens {s_video}"]
        if has_action:
            detail_parts.append(f"action tokens {s_action}")
        if has_sound:
            detail_parts.append(f"sound tokens {s_sound}")
        detail = " = " + " + ".join(detail_parts) if len(detail_parts) > 1 else ""
        adjust_detail = (
            "Adjust the spatial resolution, frame count, action chunk size, "
            "sound duration, or sound latent FPS so the combined media sequence is a "
            "multiple of ulysses_degree."
            if has_action or has_sound
            else (
                "Adjust the spatial resolution so that "
                "t * ceil(h/patch) * ceil(w/patch) is a multiple "
                "of ulysses_degree."
            )
        )
        raise ValueError(
            f"GEN sequence length ({s_gen}{detail}) must be divisible by "
            f"ulysses_degree ({ulysses_size}). {adjust_detail}"
        )

    # -- Forward -------------------------------------------------------------

    def forward(
        self,
        hidden_states: torch.Tensor,
        timestep: torch.Tensor,
        text_ids: torch.Tensor,
        text_mask: torch.Tensor,
        video_shape: tuple[int, int, int],
        fps: float | None = None,
        action_latents: torch.Tensor | None = None,
        action_domain_ids: torch.Tensor | None = None,
        action_noisy_mask: torch.Tensor | None = None,
        action_start_frame_offset: int = 1,
        action_fps: float | None = None,
        sound_latents: torch.Tensor | None = None,
        noisy_frame_mask: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor | tuple[torch.Tensor, ...]:
        """
        Args:
            hidden_states: [B, C, t, h, w] noisy latents
            timestep: [B] diffusion timestep
            text_ids: [B, S_text] tokenized text
            text_mask: [B, S_text] attention mask (1=real, 0=pad)
            video_shape: (t, h, w) in latent space
            fps: video frame rate for temporal mRoPE modulation
            action_latents: Optional [B, T_action, D_action] noisy action latents.
            action_domain_ids: Optional [B] embodiment domain IDs for action projections.
            action_noisy_mask: Optional [B, T_action, 1] mask where 1=noisy
                action token and 0=clean conditioned token.
            sound_latents: Optional [B, C_sound, T_sound] noisy sound latents.
            noisy_frame_mask: Optional [B, 1, t, 1, 1] mask where 1=noisy (add
                timestep embedding, predict velocity) and 0=conditioned (clean
                context, skip timestep embedding).  None means all frames noisy
                (T2V mode).

        Returns:
            [B, C, t, h, w] velocity prediction, or
            tuple outputs in video, action, sound order when extra modalities are provided.
        """
        t, h, w = video_shape
        hp, wp, _, _ = self._pad_to_patch_size(h, w)
        text_lengths = text_mask.sum(dim=1)
        min_real_len = int(text_lengths.min().item())
        max_real_len = int(text_lengths.max().item())
        if min_real_len != max_real_len:
            raise ValueError(
                f"Cosmos3 requires identical real text lengths within a batch "
                f"(got min={min_real_len}, max={max_real_len})."
            )
        has_action = action_latents is not None
        has_sound = sound_latents is not None
        if has_action and not self.action_gen:
            raise ValueError(
                "Cosmos3 action generation was requested, but this transformer "
                "was initialized without action modules. Check that the "
                "transformer config enables action_gen."
            )
        if has_sound and not self.sound_gen:
            raise ValueError(
                "Cosmos3 sound generation was requested, but this transformer "
                "was initialized without sound modules. Check that the "
                "transformer config enables sound_gen or defines sound_dim."
            )

        # Query Ulysses state at runtime
        ulysses_size, _, _ = _get_ulysses_state()

        # Patchify latents and project to hidden space
        hidden_video = self.proj_in(self.patchify(hidden_states, t, h, w))
        s_video = hidden_video.shape[1]
        s_action = 0
        hidden_action = None
        s_sound = 0
        hidden_sound = None
        if action_latents is not None:
            if action_latents.shape[0] != hidden_states.shape[0]:
                raise ValueError(
                    "Cosmos3 action and video batch sizes must match: "
                    f"video={hidden_states.shape[0]}, action={action_latents.shape[0]}."
                )
            if action_domain_ids is None:
                action_domain_ids = torch.zeros(action_latents.shape[0], dtype=torch.long, device=action_latents.device)
            hidden_action = self.action_proj_in(self.pack_action(action_latents), action_domain_ids)
            hidden_action = hidden_action + self.action_modality_embed.to(hidden_action.dtype)
            s_action = hidden_action.shape[1]
        if sound_latents is not None:
            if sound_latents.shape[0] != hidden_states.shape[0]:
                raise ValueError(
                    "Cosmos3 sound and video batch sizes must match: "
                    f"video={hidden_states.shape[0]}, sound={sound_latents.shape[0]}."
                )
            hidden_sound = self.audio_proj_in(self.pack_sound(sound_latents))
            hidden_sound = hidden_sound + self.audio_modality_embed.to(hidden_sound.dtype)
            s_sound = hidden_sound.shape[1]

        # Timestep embedding (fp32 for precision).
        # For I2V: only add to noisy tokens, not conditioned ones.
        # Conditioned frames are clean context and should not receive
        # the diffusion timestep signal.
        with torch.autocast("cuda", enabled=True, dtype=torch.float32):
            time_embed = self.time_embedder(timestep * self.timestep_scale)
        time_embed = time_embed.to(hidden_states.dtype)

        if noisy_frame_mask is not None:
            # Build per-token mask from per-frame mask.
            # noisy_frame_mask: [B, 1, t, 1, 1] → token mask: [B, t*hp*wp, 1]
            token_noisy_mask = (
                noisy_frame_mask[:, 0, :, 0, 0]  # [B, t]
                .unsqueeze(-1)  # [B, t, 1]
                .expand(-1, -1, hp * wp)  # [B, t, hp*wp]
                .reshape(hidden_video.shape[0], -1, 1)  # [B, t*hp*wp, 1]
            )
            hidden_video = hidden_video + time_embed.unsqueeze(1) * token_noisy_mask
        else:
            hidden_video = hidden_video + time_embed.unsqueeze(1)

        if hidden_action is not None:
            if action_noisy_mask is None:
                hidden_action = hidden_action + time_embed.unsqueeze(1)
            else:
                if action_noisy_mask.shape != (hidden_action.shape[0], hidden_action.shape[1], 1):
                    raise ValueError(
                        "Cosmos3 action_noisy_mask must have shape [B, T_action, 1], "
                        f"got {tuple(action_noisy_mask.shape)}."
                    )
                hidden_action = hidden_action + time_embed.unsqueeze(1) * action_noisy_mask.to(hidden_action.dtype)

        if hidden_sound is not None:
            hidden_sound = hidden_sound + time_embed.unsqueeze(1)
        hidden_parts = [hidden_video]
        if hidden_action is not None:
            hidden_parts.append(hidden_action)
        if hidden_sound is not None:
            hidden_parts.append(hidden_sound)
        hidden_gen = torch.cat(hidden_parts, dim=1)

        # Run UND pathway once and cache K/V (replicated across all ranks)
        if self.cached_kv is None:
            freqs_und, freqs_gen = self._compute_rope_freqs(
                text_mask,
                t,
                hp,
                wp,
                fps,
                hidden_states.device,
                hidden_states.dtype,
                t_action=s_action,
                action_start_frame_offset=action_start_frame_offset,
                action_fps=action_fps,
                t_sound=s_sound,
            )
            cached_kv_full = self.language_model(text_ids, freqs_und)
            self.cached_freqs_gen = freqs_gen

            # Trim to real text length (remove padding).  K/V stay replicated;
            # the framework Attention layer head-slices them via joint_key/value.
            self.cached_kv = [(k[:, :max_real_len], v[:, :max_real_len]) for k, v in cached_kv_full]

        # Run GEN layers.  UND K/V (replicated) is passed to each layer;
        # the Cosmos3CrossAttention forwards them as joint_key/value so the
        # framework Attention handles the Ulysses head-slicing internally.
        if self.cached_kv is None or self.cached_freqs_gen is None:
            raise RuntimeError("Cosmos3 GEN cache was not initialized before running GEN layers.")
        self._validate_gen_sequence_parallel(
            s_gen=hidden_gen.shape[1],
            s_video=s_video,
            s_action=s_action,
            s_sound=s_sound,
            has_action=has_action,
            has_sound=has_sound,
            ulysses_size=ulysses_size,
        )
        freqs_cos, freqs_sin = self.cached_freqs_gen
        hidden_gen, freqs_cos, freqs_sin = self.gen_sp_prepare(hidden_gen, freqs_cos, freqs_sin)
        freqs_gen = (freqs_cos, freqs_sin)

        if len(self.gen_layers) == len(self.cached_kv):
            for layer, (k_und, v_und) in zip(self.gen_layers, self.cached_kv, strict=True):
                hidden_gen = layer(
                    hidden_gen,
                    k_und=k_und,
                    v_und=v_und,
                    freqs_cos=freqs_cos,
                    freqs_sin=freqs_sin,
                )
                # Cache-dit's block wrapper may return a tuple; unwrap it.
                if isinstance(hidden_gen, tuple):
                    hidden_gen = hidden_gen[0]
        else:
            # Cache-dit patches gen_layers to a grouped wrapper.
            for layer in self.gen_layers:
                hidden_gen = layer(
                    hidden_gen,
                    cached_kv=self.cached_kv,
                    freqs_gen=freqs_gen,
                )
                if isinstance(hidden_gen, tuple):
                    hidden_gen = hidden_gen[0]

            hidden_gen = self.gen_sp_gather(hidden_gen)

        # Final norm and project back to latent space
        hidden_gen = self.norm_moe_gen(hidden_gen)
        if not has_action and not has_sound:
            return self.unpatchify(self.proj_out(hidden_gen), t, h, w)

        split_sizes = [s_video]
        if has_action:
            split_sizes.append(s_action)
        if has_sound:
            split_sizes.append(s_sound)
        split_hidden = hidden_gen.split(split_sizes, dim=1)
        hidden_video = split_hidden[0]
        video_pred = self.unpatchify(self.proj_out(hidden_video), t, h, w)
        outputs: list[torch.Tensor] = [video_pred]
        split_idx = 1
        if has_action:
            hidden_action = split_hidden[split_idx]
            split_idx += 1
            assert action_domain_ids is not None
            outputs.append(self.unpack_action(self.action_proj_out(hidden_action, action_domain_ids)))
        if has_sound:
            hidden_sound = split_hidden[split_idx]
            outputs.append(self.unpack_sound(self.audio_proj_out(hidden_sound)))
        return tuple(outputs)

    def post_load_weights(self) -> None:
        """Post-load processing: ensure correct dtypes."""
        self.time_embedder.to(torch.float32)
