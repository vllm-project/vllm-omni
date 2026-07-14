# SPDX-License-Identifier: Apache-2.0
"""Numerically-exact Whisper-encoder building blocks for the Kimi-Audio audio tower.

The Kimi-Audio S2S pipeline feeds the Whisper encoder's features into the LLM's
input embedding, so a cross-rank difference in the *audio input path* shifts the
layer-21 bifurcation and — amplified by the audio-token feedback loop — collapses
TP>1 audio generation. There are two cross-rank numerical differences inside the
encoder:

1. The ``RowParallelLinear`` reductions (``self_attn.out_proj`` and ``mlp.fc2``):
   a bf16 all-reduce of two partial sums rounds differently than TP=1's single
   fused GEMM.
2. The column-parallel ``mlp.fc1``: although it has no cross-rank reduction,
   cuBLAS picks a different kernel / split-K accumulation order for the
   half-width TP=2 GEMM than the full-width TP=1 GEMM, so the two round
   differently (verified offline: [185,1280]@[5120,1280] full vs 2x half differ
   by up to 1.6e-2). This compounds through all 32 layers.

``qkv_proj`` is column-parallel and is NOT bit-exact across TP: the half-width
TP=2 GEMM selects a different cuBLAS kernel / accumulation order than TP=1's
full-width GEMM, so the resulting q/k/v differ by up to ~7.8e-3 on a sparse set
of elements (verified in-vivo at whisper layer 0). That seed compounds through
all 32 layers and collapses TP>1 audio. The encoder attention is full (per-head
local, no KV cache).

So this module makes ``qkv_proj`` / ``out_proj`` / ``fc1`` / ``fc2`` **full
replicated GEMMs** — the identical operation TP=1 performs on every rank. For
``qkv_proj`` the weight is kept as a ``QKVParallelLinear`` only so the existing
stacked-params loader shards q/k/v unchanged; at first forward the per-rank
shards are all-gathered and reordered into the full fused ``[q;k;v]`` weight and
the forward runs a full-width ``F.linear`` bit-identical to TP=1. Attribute
names mirror ``WhisperEncoderLayer``
(``self_attn.qkv_proj`` / ``self_attn.out_proj`` / ``mlp.fc1`` / ``mlp.fc2`` /
``self_attn_layer_norm`` / ``final_layer_norm``) so the existing TP-aware
``KimiAudioWhisperEncoder.load_weights`` path works unchanged: q/k/v shard via
the stacked-params mapping, while out_proj/fc1/fc2/norms load full through the
default loader. Result: TP=N audio features match the known-good TP=1 features.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn
from vllm.distributed import (
    get_tensor_model_parallel_world_size,
    tensor_model_parallel_all_gather,
)
from vllm.model_executor.layers.linear import (
    QKVParallelLinear,
    ReplicatedLinear,
)
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.models.utils import cast_overflow_tensors
from vllm.model_executor.models.whisper import WhisperEncoderAttention

from .replicated_qwen2 import _gather_if_tp


class ExactWhisperAttention(nn.Module):
    """Whisper encoder attention: sharded qkv + local full attention, exact out_proj.

    ``qkv_proj`` stays column-parallel and ``self.attn`` keeps the per-rank head
    count (encoder attention has no KV cache). ``out_proj`` is a full replicated
    GEMM fed by an all-gathered attention output instead of a partial-sum +
    all-reduce, removing the bf16 reduction error.
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        num_heads_local: int,
        num_kv_heads_local: int,
        head_dim: int,
        scaling: float,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.total_num_heads = num_heads
        self.num_heads = num_heads_local
        self.num_kv_heads = num_kv_heads_local
        self.head_dim = head_dim
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = scaling

        self.qkv_proj = QKVParallelLinear(
            hidden_size=embed_dim,
            head_size=self.head_dim,
            total_num_heads=self.total_num_heads,
            total_num_kv_heads=self.total_num_heads,
            bias=True,
            quant_config=quant_config,
            prefix=f"{prefix}.qkv_proj",
        )
        self.attn = WhisperEncoderAttention(
            self.num_heads,
            self.head_dim,
            self.scaling,
            num_kv_heads=self.num_kv_heads,
        )
        # Exact: full replicated weight, input all-gathered (no all-reduce).
        self.out_proj = ReplicatedLinear(
            self.total_num_heads * self.head_dim,
            embed_dim,
            bias=True,
            quant_config=quant_config,
            prefix=f"{prefix}.out_proj",
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        attn_output = self.attn(q, k, v)  # [tokens, num_heads_local * head_dim]
        attn_output = _gather_if_tp(attn_output)  # [tokens, embed_dim]
        output, _ = self.out_proj(attn_output)
        return output


class ReplicatedWhisperAttention(nn.Module):
    """Whisper encoder attention with TP=1-identical geometry AND qkv GEMM.

    Two cross-rank differences must be removed for bit-exact features:

    1. Encoder attention runs on the *sharded* head set, so the kernel tiles
       differently than TP=1. Fixed by all-gathering q/k/v to the FULL head set
       (a lossless concat of exact column-parallel slices) and attending over the
       full TP=1 geometry (encoder attention has no KV cache).
    2. The sharded ``qkv_proj`` GEMM itself rounds differently than TP=1's
       full-width GEMM (cuBLAS kernel / accumulation order depends on output
       width) — verified in-vivo: q/k/v differ by up to ~7.8e-3 before attention.
       Fixed by reconstructing the FULL fused ``[q;k;v]`` weight on every rank
       (all-gather the per-rank shards, reorder to ``[q_full;k_full;v_full]``) and
       running a full-width ``F.linear`` identical to TP=1.

    ``qkv_proj`` stays a ``QKVParallelLinear`` so weight loading is unchanged
    (q/k/v shard via the stacked-params map); the full weight is rebuilt lazily
    on first forward. ``out_proj`` is a full replicated GEMM.
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        num_heads_local: int,
        num_kv_heads_local: int,
        head_dim: int,
        scaling: float,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.total_num_heads = num_heads
        self.num_heads_local = num_heads_local
        self.num_kv_heads_local = num_kv_heads_local
        self.head_dim = head_dim
        self.q_size = num_heads * head_dim  # full
        self.kv_size = num_heads * head_dim  # full (MHA: kv heads == q heads)
        self.q_size_local = num_heads_local * head_dim
        self.kv_size_local = num_kv_heads_local * head_dim
        self.scaling = scaling

        # Sharded qkv — weight loading unchanged (q/k/v shard via stacked-params).
        self.qkv_proj = QKVParallelLinear(
            hidden_size=embed_dim,
            head_size=self.head_dim,
            total_num_heads=self.total_num_heads,
            total_num_kv_heads=self.total_num_heads,
            bias=True,
            quant_config=quant_config,
            prefix=f"{prefix}.qkv_proj",
        )
        # FULL head count -> TP=1-identical encoder attention (no KV cache).
        self.attn = WhisperEncoderAttention(
            self.total_num_heads,
            self.head_dim,
            self.scaling,
            num_kv_heads=self.total_num_heads,
        )
        # Full replicated weight; the attention output is already full (no gather).
        self.out_proj = ReplicatedLinear(
            self.total_num_heads * self.head_dim,
            embed_dim,
            bias=True,
            quant_config=quant_config,
            prefix=f"{prefix}.out_proj",
        )

        # Lazily-built full fused qkv weight/bias for the TP=1-identical GEMM.
        self._qkv_full_weight: torch.Tensor | None = None
        self._qkv_full_bias: torch.Tensor | None = None

    def _build_full_qkv(self) -> None:
        """Reconstruct the full fused ``[q;k;v]`` weight/bias from the TP shards.

        Each rank holds ``[q_local; k_local; v_local]`` rows for its head range.
        All-gathering along dim 0 yields a rank-major block; reordering the q/k/v
        sub-blocks restores TP=1's ``[q_full; k_full; v_full]`` layout. The gather
        is a lossless copy, so the rebuilt tensor equals TP=1's weight exactly.
        """
        world = get_tensor_model_parallel_world_size()
        ql, kvl = self.q_size_local, self.kv_size_local
        per = ql + 2 * kvl  # output rows held by one rank
        w = self.qkv_proj.weight.detach()
        w_all = tensor_model_parallel_all_gather(w.contiguous(), dim=0)
        w_all = w_all.view(world, per, -1)
        q_full = w_all[:, :ql].reshape(world * ql, -1)
        k_full = w_all[:, ql : ql + kvl].reshape(world * kvl, -1)
        v_full = w_all[:, ql + kvl :].reshape(world * kvl, -1)
        self._qkv_full_weight = torch.cat([q_full, k_full, v_full], dim=0)

        b = self.qkv_proj.bias
        if b is not None:
            b_all = tensor_model_parallel_all_gather(b.detach().contiguous(), dim=0)
            b_all = b_all.view(world, per)
            bq = b_all[:, :ql].reshape(world * ql)
            bk = b_all[:, ql : ql + kvl].reshape(world * kvl)
            bv = b_all[:, ql + kvl :].reshape(world * kvl)
            self._qkv_full_bias = torch.cat([bq, bk, bv], dim=0)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if get_tensor_model_parallel_world_size() > 1:
            # Full-width GEMM bit-identical to TP=1 (the sharded GEMM is not).
            if self._qkv_full_weight is None:
                self._build_full_qkv()
            qkv = F.linear(hidden_states, self._qkv_full_weight, self._qkv_full_bias)
        else:
            qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        attn_output = self.attn(q, k, v)  # [tokens, embed_dim]
        output, _ = self.out_proj(attn_output)
        return output


class ExactWhisperMLP(nn.Module):
    """Whisper MLP: both fc1 and fc2 are full replicated GEMMs (bit-exact TP=1).

    fc1 MUST be replicated, not column-parallel: although a column-parallel fc1
    has no cross-rank reduction, cuBLAS selects a different kernel / split-K
    accumulation order for the half-width TP=2 GEMM than for the full-width TP=1
    GEMM, so the two round differently (verified offline: [185,1280]@[5120,1280]
    full vs 2x half differ by up to 1.6e-2). That per-layer rounding compounds
    through the 32 whisper layers and seeds the audio-feature divergence. A full
    ReplicatedLinear runs the identical GEMM as TP=1 on every rank → bit-exact.
    """

    def __init__(
        self,
        embed_dim: int,
        ffn_dim: int,
        activation_fn: nn.Module,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.activation_fn = activation_fn
        # Replicated (NOT column-parallel): full GEMM on every rank = TP=1-exact.
        self.fc1 = ReplicatedLinear(
            input_size=embed_dim,
            output_size=ffn_dim,
            quant_config=quant_config,
            prefix=f"{prefix}.fc1",
        )
        self.fc2 = ReplicatedLinear(
            input_size=ffn_dim,
            output_size=embed_dim,
            quant_config=quant_config,
            prefix=f"{prefix}.fc2",
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states, _ = self.fc1(hidden_states)
        hidden_states = self.activation_fn(hidden_states)
        hidden_states, _ = self.fc2(hidden_states)
        return hidden_states


class ExactWhisperEncoderLayer(nn.Module):
    """Drop-in replacement for ``WhisperEncoderLayer`` with exact row-parallel ops.

    Mirrors ``WhisperEncoderLayer.forward`` (pre-LN residual structure). The
    ``activation_fn`` module is reused from the original layer (stateless), so no
    activation-name guessing is required.
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        num_heads_local: int,
        num_kv_heads_local: int,
        head_dim: int,
        scaling: float,
        ffn_dim: int,
        activation_fn: nn.Module,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.self_attn = ExactWhisperAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            num_heads_local=num_heads_local,
            num_kv_heads_local=num_kv_heads_local,
            head_dim=head_dim,
            scaling=scaling,
            quant_config=quant_config,
            prefix=f"{prefix}.self_attn",
        )
        self.self_attn_layer_norm = nn.LayerNorm(self.embed_dim)
        self.mlp = ExactWhisperMLP(
            embed_dim=embed_dim,
            ffn_dim=ffn_dim,
            activation_fn=activation_fn,
            quant_config=quant_config,
            prefix=f"{prefix}.mlp",
        )
        self.final_layer_norm = nn.LayerNorm(self.embed_dim)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.self_attn_layer_norm(hidden_states)
        hidden_states = self.self_attn(hidden_states=hidden_states)
        hidden_states = residual + hidden_states
        residual = hidden_states
        hidden_states = self.final_layer_norm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        hidden_states = cast_overflow_tensors(hidden_states)
        return hidden_states


class ReplicatedWhisperEncoderLayer(nn.Module):
    """``ExactWhisperEncoderLayer`` with attention replication (TP=1-identical).

    Same attribute names and weight-loading contract as ``ExactWhisperEncoderLayer``
    (q/k/v shard, out_proj/fc2/norms load full), so the existing
    ``KimiAudioWhisperEncoder.load_weights`` path works unchanged. The MLP is the
    exact all-gather path; only the attention differs (full-head geometry on every
    rank). Use this so the audio features spliced into the LLM input are
    bit-identical to TP=1 — the divergence that otherwise collapses TP>1 audio.
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        num_heads_local: int,
        num_kv_heads_local: int,
        head_dim: int,
        scaling: float,
        ffn_dim: int,
        activation_fn: nn.Module,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.self_attn = ReplicatedWhisperAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            num_heads_local=num_heads_local,
            num_kv_heads_local=num_kv_heads_local,
            head_dim=head_dim,
            scaling=scaling,
            quant_config=quant_config,
            prefix=f"{prefix}.self_attn",
        )
        self.self_attn_layer_norm = nn.LayerNorm(self.embed_dim)
        self.mlp = ExactWhisperMLP(
            embed_dim=embed_dim,
            ffn_dim=ffn_dim,
            activation_fn=activation_fn,
            quant_config=quant_config,
            prefix=f"{prefix}.mlp",
        )
        self.final_layer_norm = nn.LayerNorm(self.embed_dim)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.self_attn_layer_norm(hidden_states)
        hidden_states = self.self_attn(hidden_states=hidden_states)
        hidden_states = residual + hidden_states
        residual = hidden_states
        hidden_states = self.final_layer_norm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        hidden_states = cast_overflow_tensors(hidden_states)
        return hidden_states
