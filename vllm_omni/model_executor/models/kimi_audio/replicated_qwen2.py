# SPDX-License-Identifier: Apache-2.0
"""Numerically-exact Qwen2 decoder building blocks for the Kimi-Audio MIMO branch.

The MIMO / audio branch is extremely sensitive: its residual stream runs hot
(absmax ~400 vs ~4.7 for the text branch) and its output is a competitive
audio-token softmax, so tiny per-step differences flip the argmax and — once the
audio-token feedback loop amplifies them — collapse generation under TP>1
(greedy argmax first flips at decode step 4, then the stream degenerates and
never emits EOD).

The cross-rank numerical differences between TP=1 and TP=2 inside the branch
come from TWO kinds of sharded GEMM:

1. ``RowParallelLinear`` reductions (``o_proj`` / ``down_proj``): a bf16
   all-reduce of two partial sums rounds differently than TP=1's single fused
   GEMM.
2. Column-parallel ``qkv_proj`` / ``gate_up_proj``: although they have no
   cross-rank reduction, cuBLAS selects a different kernel / split-K accumulation
   order for the half-width TP=2 GEMM than the full-width TP=1 GEMM, so the two
   round differently. Verified in-vivo on the Kimi whisper tower (the identical
   ``QKVParallelLinear`` class): q/k/v differed by up to ~7.8e-3 before
   attention. These column-parallel diffs are the seed that the audio-token
   feedback loop amplifies into collapse.

So this module replaces every sharded GEMM with a **full-width** one that is
bit-identical to TP=1. ``qkv_proj`` / ``gate_up_proj`` stay sharded modules only
so the existing stacked-params loader works unchanged (q/k/v and gate/up shard
via the mapping); at first forward the per-rank shards are all-gathered and
reordered into the full fused weight, and the forward runs a full-width
``F.linear``. ``o_proj`` / ``down_proj`` are full replicated GEMMs fed by an
all-gathered activation (no all-reduce). Attention keeps a uniform per-layer
KV-cache layout (vLLM V1 "same physical memory per token per layer") by running
full-head geometry on every rank. Result: TP=N audio output matches the
known-good TP=1 trajectory.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn
from transformers import Qwen2Config
from vllm.config import CacheConfig
from vllm.distributed import (
    get_tensor_model_parallel_world_size,
    tensor_model_parallel_all_gather,
)
from vllm.model_executor.layers.activation import SiluAndMul
from vllm.model_executor.layers.attention import Attention
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import (
    MergedColumnParallelLinear,
    QKVParallelLinear,
    ReplicatedLinear,
)
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.transformers_utils.config import set_default_rope_theta
from vllm.v1.attention.backend import AttentionType


def _gather_if_tp(x: torch.Tensor) -> torch.Tensor:
    """All-gather a feature-sharded activation across the TP group (lossless).

    The input is made contiguous first: ``all_gather_into_tensor`` rejects
    non-contiguous tensors, and feature-sliced activations (e.g. q/k/v from
    ``qkv.split(dim=-1)``) are strided views. ``.contiguous()`` is a no-op when
    the tensor is already contiguous.
    """
    if get_tensor_model_parallel_world_size() > 1:
        return tensor_model_parallel_all_gather(x.contiguous(), dim=-1)
    return x


def _reconstruct_fused_rows(
    local: torch.Tensor,
    block_sizes_local: list[int],
) -> torch.Tensor:
    """Rebuild a full fused weight/bias from a column-parallel shard.

    ``local`` is this rank's shard (``[sum(block_sizes_local), ...]``), laid out as
    concatenated per-block row ranges for the rank's head/intermediate slice.
    All-gathering along dim 0 yields a rank-major tensor; reordering the blocks
    restores TP=1's ``[block0_full; block1_full; ...]`` layout. The gather is a
    lossless copy, so the result equals TP=1's full weight exactly.
    """
    world = get_tensor_model_parallel_world_size()
    per = sum(block_sizes_local)
    all_rows = tensor_model_parallel_all_gather(local.detach().contiguous(), dim=0)
    all_rows = all_rows.view(world, per, *local.shape[1:])
    blocks = []
    off = 0
    for size in block_sizes_local:
        blocks.append(all_rows[:, off : off + size].reshape(world * size, *local.shape[1:]))
        off += size
    return torch.cat(blocks, dim=0)


class ExactQwen2Attention(nn.Module):
    """Qwen2 attention: sharded qkv + local heads (uniform KV cache), exact o_proj.

    ``qkv_proj`` stays column-parallel and ``self.attn`` keeps the per-rank head
    count (so the KV-cache layout matches the text layers). ``o_proj`` is a full
    replicated GEMM fed by an all-gathered attention output instead of a
    partial-sum + all-reduce, removing the bf16 reduction error.
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        rope_parameters: dict,
        max_position: int = 4096 * 32,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
        attn_type: str = AttentionType.DECODER,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        tp_size = get_tensor_model_parallel_world_size()
        self.total_num_heads = num_heads
        assert self.total_num_heads % tp_size == 0
        self.num_heads = self.total_num_heads // tp_size
        self.total_num_kv_heads = num_kv_heads
        if self.total_num_kv_heads >= tp_size:
            assert self.total_num_kv_heads % tp_size == 0
        else:
            assert tp_size % self.total_num_kv_heads == 0
        self.num_kv_heads = max(1, self.total_num_kv_heads // tp_size)
        self.head_dim = hidden_size // self.total_num_heads
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim**-0.5

        self.qkv_proj = QKVParallelLinear(
            hidden_size,
            self.head_dim,
            self.total_num_heads,
            self.total_num_kv_heads,
            bias=True,
            quant_config=quant_config,
            prefix=f"{prefix}.qkv_proj",
        )
        # Exact: full replicated weight, input all-gathered (no all-reduce).
        self.o_proj = ReplicatedLinear(
            self.total_num_heads * self.head_dim,
            hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.o_proj",
        )
        self.rotary_emb = get_rope(
            self.head_dim,
            max_position=max_position,
            rope_parameters=rope_parameters,
        )
        self.attn = Attention(
            self.num_heads,
            self.head_dim,
            self.scaling,
            num_kv_heads=self.num_kv_heads,
            cache_config=cache_config,
            quant_config=quant_config,
            attn_type=attn_type,
            prefix=f"{prefix}.attn",
        )

    def forward(self, positions: torch.Tensor, hidden_states: torch.Tensor) -> torch.Tensor:
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        q, k = self.rotary_emb(positions, q, k)
        attn_output = self.attn(q, k, v)  # [tokens, num_heads_local * head_dim]
        attn_output = _gather_if_tp(attn_output)  # [tokens, total_heads * head_dim]
        output, _ = self.o_proj(attn_output)
        return output


class ReplicatedQwen2Attention(nn.Module):
    """Qwen2 attention with TP=1-identical per-rank geometry (attention replication).

    The numerically-sensitive Kimi audio (mimo) branch collapses under TP>1
    because the attention kernel's per-head reduction order depends on the
    heads-per-rank / split-KV geometry: TP=1 runs all 4 KV heads on one rank,
    TP=2 runs 2 per rank, so the flash kernel tiles/reduces differently and the
    layer-21 bifurcation shifts ~1.5% — which the audio-token feedback loop then
    amplifies into total collapse. All linear reductions are already exact (see
    ``ExactQwen2Attention``), so attention geometry is the only remaining
    cross-rank difference.

    This keeps ``qkv_proj`` SHARDED (so the existing TP-aware weight loading is
    unchanged — q/k/v still shard via the stacked-params mapping) but
    reconstructs the FULL head set on every rank before attention: the local
    q/k/v slices are all-gathered (a lossless concatenation — column-parallel
    slices are exact columns of the full GEMM), rope is applied to the full
    tensors, and ``self.attn`` is built with the FULL head count so its KV-cache
    spec (``FullAttentionSpec(num_kv_heads=self.num_kv_heads)``) is full-head.
    Every rank therefore runs the identical kernel configuration as TP=1 and
    produces bit-identical attention. ``o_proj`` is a full replicated GEMM (no
    output gather needed — the attention output is already full).

    Cost: the KV cache is replicated (full heads per rank, so TP no longer
    shrinks KV memory) and each rank does the full attention FLOPs plus two
    small per-step all-gathers. This trades TP efficiency for TP=1 numerical
    equivalence on the chaos-sensitive path. Requires
    ``total_num_kv_heads >= tp_size`` (Kimi: 4 KV heads, so TP<=4).
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        rope_parameters: dict,
        max_position: int = 4096 * 32,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
        attn_type: str = AttentionType.DECODER,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        tp_size = get_tensor_model_parallel_world_size()
        self.total_num_heads = num_heads
        assert self.total_num_heads % tp_size == 0
        self.num_heads_local = self.total_num_heads // tp_size
        self.total_num_kv_heads = num_kv_heads
        # KV heads must cover every rank so the all-gather reconstructs the full
        # set exactly once (no duplicated heads). Kimi has 4 KV heads -> TP<=4.
        assert self.total_num_kv_heads >= tp_size, (
            f"attention replication requires num_kv_heads ({self.total_num_kv_heads}) >= tp_size ({tp_size})"
        )
        assert self.total_num_kv_heads % tp_size == 0
        self.num_kv_heads_local = self.total_num_kv_heads // tp_size
        self.head_dim = hidden_size // self.total_num_heads
        self.q_size_local = self.num_heads_local * self.head_dim
        self.kv_size_local = self.num_kv_heads_local * self.head_dim
        self.q_size = self.total_num_heads * self.head_dim  # full
        self.kv_size = self.total_num_kv_heads * self.head_dim  # full
        self.scaling = self.head_dim**-0.5

        # Sharded qkv — weight loading unchanged (q/k/v shard via stacked-params).
        self.qkv_proj = QKVParallelLinear(
            hidden_size,
            self.head_dim,
            self.total_num_heads,
            self.total_num_kv_heads,
            bias=True,
            quant_config=quant_config,
            prefix=f"{prefix}.qkv_proj",
        )
        # Full replicated weight; the attention output is already full (no gather).
        self.o_proj = ReplicatedLinear(
            self.total_num_heads * self.head_dim,
            hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.o_proj",
        )
        self.rotary_emb = get_rope(
            self.head_dim,
            max_position=max_position,
            rope_parameters=rope_parameters,
        )
        # FULL head count -> full-head KV-cache spec (replicated across ranks).
        self.attn = Attention(
            self.total_num_heads,
            self.head_dim,
            self.scaling,
            num_kv_heads=self.total_num_kv_heads,
            cache_config=cache_config,
            quant_config=quant_config,
            attn_type=attn_type,
            prefix=f"{prefix}.attn",
        )

        # Lazily-built full fused qkv weight/bias for the TP=1-identical GEMM.
        self._qkv_full_weight: torch.Tensor | None = None
        self._qkv_full_bias: torch.Tensor | None = None

    def _build_full_qkv(self) -> None:
        """Reconstruct the full fused ``[q;k;v]`` weight/bias from the TP shards."""
        blocks = [self.q_size_local, self.kv_size_local, self.kv_size_local]
        self._qkv_full_weight = _reconstruct_fused_rows(self.qkv_proj.weight, blocks)
        if self.qkv_proj.bias is not None:
            self._qkv_full_bias = _reconstruct_fused_rows(self.qkv_proj.bias, blocks)

    def forward(self, positions: torch.Tensor, hidden_states: torch.Tensor) -> torch.Tensor:
        if get_tensor_model_parallel_world_size() > 1:
            # Full-width GEMM bit-identical to TP=1 (the sharded GEMM is not).
            if self._qkv_full_weight is None:
                self._build_full_qkv()
            qkv = F.linear(hidden_states, self._qkv_full_weight, self._qkv_full_bias)
        else:
            qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        q, k = self.rotary_emb(positions, q, k)
        attn_output = self.attn(q, k, v)  # [tokens, total_heads * head_dim]
        output, _ = self.o_proj(attn_output)
        return output


class ReplicatedQwen2DecoderLayer(nn.Module):
    """``ExactQwen2DecoderLayer`` with attention replication (TP=1-identical attention).

    Same attribute names and weight-loading contract as ``ExactQwen2DecoderLayer``
    (q/k/v shard, o_proj/down_proj/norms load full), so the existing
    ``load_weights`` path works unchanged. The MLP is the exact all-gather path;
    only the attention differs (full-head geometry on every rank).
    """

    def __init__(
        self,
        config: Qwen2Config,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        set_default_rope_theta(config, default_theta=1000000)
        attn_type = AttentionType.DECODER if getattr(config, "is_causal", True) else AttentionType.ENCODER_ONLY

        self.self_attn = ReplicatedQwen2Attention(
            hidden_size=self.hidden_size,
            num_heads=config.num_attention_heads,
            max_position=config.max_position_embeddings,
            num_kv_heads=config.num_key_value_heads,
            cache_config=cache_config,
            quant_config=quant_config,
            rope_parameters=config.rope_parameters,
            prefix=f"{prefix}.self_attn",
            attn_type=attn_type,
        )
        self.mlp = ExactQwen2MLP(
            hidden_size=self.hidden_size,
            intermediate_size=config.intermediate_size,
            hidden_act=config.hidden_act,
            quant_config=quant_config,
            prefix=f"{prefix}.mlp",
        )
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Self Attention
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)
        hidden_states = self.self_attn(positions=positions, hidden_states=hidden_states)

        # Fully Connected
        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        hidden_states = self.mlp(hidden_states)
        return hidden_states, residual


class ExactQwen2MLP(nn.Module):
    """Qwen2 MLP: full-width gate_up and down_proj GEMMs (bit-exact TP=1).

    ``gate_up_proj`` is kept as a sharded ``MergedColumnParallelLinear`` only so
    the stacked-params loader shards gate/up unchanged; at first forward the
    shards are all-gathered and reordered into the full fused ``[gate;up]``
    weight, and the forward runs a full-width ``F.linear`` identical to TP=1
    (the sharded column-parallel GEMM rounds differently across TP).
    ``down_proj`` is a full replicated GEMM (no all-reduce).
    """

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        tp_size = get_tensor_model_parallel_world_size()
        self.intermediate_size_local = intermediate_size // tp_size
        self.gate_up_proj = MergedColumnParallelLinear(
            hidden_size,
            [intermediate_size] * 2,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.gate_up_proj",
        )
        # Exact: full replicated weight, input all-gathered (no all-reduce).
        self.down_proj = ReplicatedLinear(
            intermediate_size,
            hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.down_proj",
        )
        if hidden_act != "silu":
            raise ValueError(f"Unsupported activation: {hidden_act}. Only silu is supported for now.")
        self.act_fn = SiluAndMul()

        # Lazily-built full fused [gate;up] weight for the TP=1-identical GEMM.
        self._gate_up_full_weight: torch.Tensor | None = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if get_tensor_model_parallel_world_size() > 1:
            # Full-width GEMM bit-identical to TP=1 (the sharded GEMM is not).
            if self._gate_up_full_weight is None:
                blocks = [self.intermediate_size_local, self.intermediate_size_local]
                self._gate_up_full_weight = _reconstruct_fused_rows(self.gate_up_proj.weight, blocks)
            gate_up = F.linear(x, self._gate_up_full_weight)
        else:
            gate_up, _ = self.gate_up_proj(x)
        x = self.act_fn(gate_up)  # [tokens, intermediate]
        x, _ = self.down_proj(x)
        return x


class ExactQwen2DecoderLayer(nn.Module):
    """Drop-in replacement for ``Qwen2DecoderLayer`` with exact row-parallel ops.

    Attribute names mirror ``Qwen2DecoderLayer`` (``self_attn.qkv_proj`` /
    ``self_attn.o_proj`` / ``mlp.gate_up_proj`` / ``mlp.down_proj`` / norms) so
    the existing TP-aware ``load_weights`` path works unchanged: q/k/v and
    gate/up shard via the stacked-params mapping, while o_proj/down_proj/norms
    load full through the default loader.
    """

    def __init__(
        self,
        config: Qwen2Config,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        set_default_rope_theta(config, default_theta=1000000)
        attn_type = AttentionType.DECODER if getattr(config, "is_causal", True) else AttentionType.ENCODER_ONLY

        self.self_attn = ExactQwen2Attention(
            hidden_size=self.hidden_size,
            num_heads=config.num_attention_heads,
            max_position=config.max_position_embeddings,
            num_kv_heads=config.num_key_value_heads,
            cache_config=cache_config,
            quant_config=quant_config,
            rope_parameters=config.rope_parameters,
            prefix=f"{prefix}.self_attn",
            attn_type=attn_type,
        )
        self.mlp = ExactQwen2MLP(
            hidden_size=self.hidden_size,
            intermediate_size=config.intermediate_size,
            hidden_act=config.hidden_act,
            quant_config=quant_config,
            prefix=f"{prefix}.mlp",
        )
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Self Attention
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)
        hidden_states = self.self_attn(positions=positions, hidden_states=hidden_states)

        # Fully Connected
        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        hidden_states = self.mlp(hidden_states)
        return hidden_states, residual
