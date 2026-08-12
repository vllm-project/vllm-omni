# SPDX-License-Identifier: Apache-2.0
"""MiniMax H3 Qwen3-VL layer-50 text/vision encoder.

The encoder is reimplemented on top of vLLM-style tensor-parallel building
blocks (column/row/vocab-parallel linears bound to the MiniMax H3 encoder
process group) instead of the single-GPU ``transformers`` backbone.  This
makes the large Qwen3-VL text model TP-shardable across ``text_encoder_tp_size``
ranks, which removes the rank-0 memory hotspot that dominates no-offload peak
memory (the retained 50-layer encoder is ~51.5 GB in BF16).

The computation contract is unchanged from the HF reference path:

- plain BF16 weights, only the first ``MINIMAX_H3_QWEN3VL_SELECTED_LM_LAYER``
  decoder layers are kept (the checkpoint consumes the unnormalized hidden
  state after decoder layer 49);
- cuDNN SDP is enabled during encode;
- the multimodal backbone runs with an all-ones attention mask and
  ``mm_token_type_ids`` derived from the image/video token ids;
- DeepStack visual features are injected at the first
  ``len(deepstack_visual_indexes)`` decoder layers;
- the output is ``hidden_states[50]`` of shape ``[seq, 5120]``.

Tensor parallelism follows vLLM's TP semantics: vocab-parallel embeddings and
row-parallel projections all-reduce so the hidden state stays fully replicated
on every encoder rank, while column-parallel projections (QKV / MLP up) keep a
local shard.  After the final layer every encoder rank therefore holds the
complete ``[seq, 5120]`` hidden state.
"""

from __future__ import annotations

import itertools
import json
import os
import re
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from vllm.logger import init_logger

MINIMAX_H3_QWEN3VL_SELECTED_LM_LAYER = 50
MINIMAX_H3_QWEN3VL_HIDDEN_DIM = 5120

logger = init_logger(__name__)

_TEXT_ENCODER_QUANT_ENV = "VLLM_OMNI_H3_TEXT_ENCODER_QUANTIZATION"


def minimax_h3_text_encoder_quantization() -> bool:
    """Return whether opt-in online W4A16 NVFP4 is requested."""
    value = os.getenv(_TEXT_ENCODER_QUANT_ENV, "").strip().lower()
    if not value or value in {"none", "bf16"}:
        return False
    if value in {"online_nvfp4", "online_nvfp4_w4a16", "nvfp4_online"}:
        return True
    raise ValueError(
        f"{_TEXT_ENCODER_QUANT_ENV} must be bf16 or online_nvfp4_w4a16; got {value!r}"
    )


def _default_weight_loader(
    param: nn.Parameter,
    loaded_weight: torch.Tensor,
    loaded_shard_id: str | None = None,
) -> None:
    del loaded_shard_id
    param.data.copy_(loaded_weight)


def _tp_range(group: Any) -> tuple[int, int]:
    return group.rank_in_group, group.world_size


class _MiniMaxH3OnlineNvFp4Linear:
    """Convert a CUDA-resident BF16 TP linear to ModelOpt W4A16 NVFP4.

    vLLM's ModelOpt method intentionally rejects dynamic conversion at its
    public checkpoint loader.  This experiment explicitly performs ModelOpt's
    in-memory packing at load time, then feeds those serialized tensors to the
    same W4A16 Marlin method.  It neither reads an offline artifact nor uses
    FP8 as a fallback.
    """

    _online_nvfp4_method: Any | None
    _online_nvfp4_layer: nn.Module | None

    def _init_online_nvfp4(self, enabled: bool) -> None:
        self._online_nvfp4_enabled = enabled
        self._online_nvfp4_method = None
        self._online_nvfp4_layer = None

    @property
    def is_online_nvfp4_quantized(self) -> bool:
        return self._online_nvfp4_layer is not None

    def _quantize_online_nvfp4(
        self,
        *,
        input_size: int,
        output_partition_sizes: list[int],
    ) -> tuple[int, int]:
        if not self._online_nvfp4_enabled or self._online_nvfp4_layer is not None:
            return 0, 0
        if not self.weight.is_cuda:
            raise RuntimeError("MiniMax H3 online NVFP4 conversion requires CUDA-resident BF16 weights")

        from modelopt.torch.quantization.qtensor import NVFP4QTensor
        from vllm.model_executor.layers.quantization.modelopt import (
            ModelOptNvFp4Config,
            ModelOptNvFp4W4A16LinearMethod,
        )

        source_nbytes = self.weight.numel() * self.weight.element_size()
        packed, weight_scale, weight_scale_2 = NVFP4QTensor.quantize(self.weight.detach(), 16)
        config = ModelOptNvFp4Config(
            quant_method="W4A16_NVFP4",
            # This is an in-memory serialized representation created above;
            # the method otherwise correctly rejects unsupported dynamic I/O.
            is_checkpoint_nvfp4_serialized=True,
            exclude_modules=[],
            group_size=16,
        )
        method = ModelOptNvFp4W4A16LinearMethod(config)
        layer = nn.Module()
        layer.params_dtype = torch.bfloat16
        method.create_weights(
            layer,
            input_size_per_partition=self.weight.shape[1],
            output_partition_sizes=output_partition_sizes,
            input_size=input_size,
            output_size=sum(output_partition_sizes),
            params_dtype=torch.bfloat16,
        )
        layer.to(self.weight.device)
        layer.weight.data.copy_(packed._quantized_data)
        layer.weight_scale.data.copy_(weight_scale)
        # ModelOpt's public quantizer returns a scalar global scale. vLLM's
        # fused QKV/gate-up loader stores one equal value per logical part.
        layer.weight_scale_2.data.copy_(weight_scale_2.reshape(1).repeat(len(output_partition_sizes)))
        method.process_weights_after_loading(layer)
        self._online_nvfp4_method = method
        self._online_nvfp4_layer = layer
        del self._parameters["weight"]
        state_nbytes = sum(parameter.numel() * parameter.element_size() for parameter in layer.parameters())
        return source_nbytes, state_nbytes

    def _linear(self, input_: torch.Tensor) -> torch.Tensor:
        if self._online_nvfp4_layer is None:
            return F.linear(input_, self.weight)
        return self._online_nvfp4_method.apply(self._online_nvfp4_layer, input_)


# ---------------------------------------------------------------------------
# Tensor-parallel primitives bound to the encoder process group.
#
# These mirror the math of vLLM's ``VocabParallelEmbedding``,
# ``ColumnParallelLinear``, ``MergedColumnParallelLinear``, ``QKVParallelLinear``
# and ``RowParallelLinear`` but route their collectives through the encoder
# group instead of the (DiT) tensor-model-parallel group, so the text encoder
# can be sharded independently of the DiT's TP configuration.
# ---------------------------------------------------------------------------


class MiniMaxH3Qwen3VLVocabParallelEmbedding(nn.Module):
    def __init__(self, group: Any, num_embeddings: int, embedding_dim: int, dtype: torch.dtype) -> None:
        super().__init__()
        self.group = group
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        tp_rank, tp_size = _tp_range(group)
        assert num_embeddings % tp_size == 0, (
            f"vocab_size {num_embeddings} must be divisible by text_encoder_tp_size {tp_size}"
        )
        self.num_embeddings_per_partition = num_embeddings // tp_size
        self.weight = nn.Parameter(torch.empty(self.num_embeddings_per_partition, embedding_dim, dtype=dtype))
        self.weight.weight_loader = self.weight_loader  # type: ignore[attr-defined]
        self._tp_rank = tp_rank
        self._tp_size = tp_size

    def forward(self, input_: torch.Tensor) -> torch.Tensor:
        if self._tp_size == 1:
            return F.embedding(input_, self.weight)
        # Mirror vLLM's `VocabParallelEmbedding`: token ids that fall outside
        # this rank's vocab partition are masked to 0 and in-partition ids are
        # shifted to local indices before the local gather.  The out-of-range
        # rows are zeroed so the subsequent all-reduce reconstructs the full
        # embedding for every token.
        start_idx = self._tp_rank * self.num_embeddings_per_partition
        end_idx = start_idx + self.num_embeddings_per_partition
        input_mask = (input_ < start_idx) | (input_ >= end_idx)
        masked_input = input_.clone() - start_idx
        masked_input[input_mask] = 0
        output = F.embedding(masked_input, self.weight)
        output[input_mask, :] = 0.0
        self.group.all_reduce(output)
        return output

    def weight_loader(
        self, param: nn.Parameter, loaded_weight: torch.Tensor, loaded_shard_id: str | None = None
    ) -> None:
        del loaded_shard_id
        shard_size = self.num_embeddings_per_partition
        start_idx = self._tp_rank * shard_size
        param.data.copy_(loaded_weight.narrow(0, start_idx, shard_size))


class MiniMaxH3Qwen3VLMergedColumnParallelLinear(_MiniMaxH3OnlineNvFp4Linear, nn.Module):
    """Packed gate/up projection sharded along the output dimension."""

    def __init__(
        self,
        group: Any,
        input_size: int,
        intermediate_size: int,
        dtype: torch.dtype,
        online_nvfp4: bool = False,
    ) -> None:
        super().__init__()
        self.group = group
        self.input_size = input_size
        self.intermediate_size = intermediate_size
        tp_rank, tp_size = _tp_range(group)
        assert intermediate_size % tp_size == 0, (
            f"intermediate_size {intermediate_size} must be divisible by text_encoder_tp_size {tp_size}"
        )
        self.intermediate_size_per_partition = intermediate_size // tp_size
        self.weight = nn.Parameter(torch.empty(2 * self.intermediate_size_per_partition, input_size, dtype=dtype))
        self.weight.weight_loader = self.weight_loader  # type: ignore[attr-defined]
        self._tp_rank = tp_rank
        self._tp_size = tp_size
        self._init_online_nvfp4(online_nvfp4)

    def forward(self, input_: torch.Tensor) -> torch.Tensor:
        return self._linear(input_)

    def quantize_for_inference(self) -> tuple[int, int]:
        return self._quantize_online_nvfp4(
            input_size=self.input_size,
            output_partition_sizes=[self.intermediate_size_per_partition] * 2,
        )

    def weight_loader(
        self, param: nn.Parameter, loaded_weight: torch.Tensor, loaded_shard_id: str | None = None
    ) -> None:
        shard_size = self.intermediate_size_per_partition
        start_idx = self._tp_rank * shard_size
        if loaded_shard_id == 1:  # up_proj
            param.data[shard_size : 2 * shard_size].copy_(loaded_weight.narrow(0, start_idx, shard_size))
        else:  # gate_proj
            param.data[0:shard_size].copy_(loaded_weight.narrow(0, start_idx, shard_size))


class MiniMaxH3Qwen3VLQKVParallelLinear(_MiniMaxH3OnlineNvFp4Linear, nn.Module):
    """QKV projection with GQA head sharding along the output dimension."""

    def __init__(
        self,
        group: Any,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        dtype: torch.dtype,
        online_nvfp4: bool = False,
    ) -> None:
        super().__init__()
        self.group = group
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        tp_rank, tp_size = _tp_range(group)
        assert num_heads % tp_size == 0, (
            f"num_attention_heads {num_heads} must be divisible by text_encoder_tp_size {tp_size}"
        )
        assert num_kv_heads % tp_size == 0, (
            f"num_key_value_heads {num_kv_heads} must be divisible by text_encoder_tp_size {tp_size}"
        )
        self.local_num_heads = num_heads // tp_size
        self.local_num_kv_heads = num_kv_heads // tp_size
        q_local = self.local_num_heads * head_dim
        kv_local = self.local_num_kv_heads * head_dim
        self.weight = nn.Parameter(torch.empty(q_local + 2 * kv_local, hidden_size, dtype=dtype))
        self.weight.weight_loader = self.weight_loader  # type: ignore[attr-defined]
        self._tp_rank = tp_rank
        self._tp_size = tp_size
        self._init_online_nvfp4(online_nvfp4)

    def forward(self, input_: torch.Tensor) -> torch.Tensor:
        return self._linear(input_)

    def quantize_for_inference(self) -> tuple[int, int]:
        q_local = self.local_num_heads * self.head_dim
        kv_local = self.local_num_kv_heads * self.head_dim
        return self._quantize_online_nvfp4(
            input_size=self.hidden_size,
            output_partition_sizes=[q_local, kv_local, kv_local],
        )

    def weight_loader(
        self, param: nn.Parameter, loaded_weight: torch.Tensor, loaded_shard_id: str | None = None
    ) -> None:
        head_dim = self.head_dim
        q_local = self.local_num_heads * head_dim
        kv_local = self.local_num_kv_heads * head_dim
        if loaded_shard_id == "q":
            start_idx = self._tp_rank * q_local
            param.data[0:q_local].copy_(loaded_weight.narrow(0, start_idx, q_local))
        elif loaded_shard_id == "k":
            start_idx = self._tp_rank * kv_local
            param.data[q_local : q_local + kv_local].copy_(loaded_weight.narrow(0, start_idx, kv_local))
        elif loaded_shard_id == "v":
            start_idx = self._tp_rank * kv_local
            param.data[q_local + kv_local : q_local + 2 * kv_local].copy_(loaded_weight.narrow(0, start_idx, kv_local))
        else:
            raise ValueError(f"unexpected QKV shard id {loaded_shard_id!r}")


# A fused weight is filled from several checkpoint tensors, so a partial load leaves
# uninitialized `torch.empty` rows that no later check would catch. Keyed by the class
# owning the multi-shard `weight_loader`; each entry is the source name reported to the
# user paired with the shard id that loader expects, in projection order.
_FUSED_SOURCE_SHARDS: dict[type[nn.Module], tuple[tuple[str, str | int], ...]] = {
    MiniMaxH3Qwen3VLQKVParallelLinear: (("q", "q"), ("k", "k"), ("v", "v")),
    MiniMaxH3Qwen3VLMergedColumnParallelLinear: (("gate", 0), ("up", 1)),
}


def _fused_source_shards(module: nn.Module) -> tuple[tuple[str, str | int], ...] | None:
    for cls, sources in _FUSED_SOURCE_SHARDS.items():
        if isinstance(module, cls):
            return sources
    return None


def _verify_fused_shards_complete(
    expected: dict[str, tuple[tuple[str, str | int], ...]],
    loaded: dict[str, set[str | int]],
) -> None:
    missing = {
        name: [source for source, shard_id in sources if shard_id not in loaded[name]]
        for name, sources in expected.items()
    }
    missing = {name: sources for name, sources in missing.items() if sources}
    if not missing:
        return
    details = "; ".join(f"{name}: {sources}" for name, sources in sorted(missing.items()))
    raise RuntimeError(f"MiniMax H3 text encoder fused weights are missing source shards: {details}")


# A fused weight is filled from several checkpoint tensors, so a partial load leaves
# uninitialized `torch.empty` rows that no later check would catch. Keyed by the class
# owning the multi-shard `weight_loader`; each entry is the source name reported to the
# user paired with the shard id that loader expects, in projection order.
_FUSED_SOURCE_SHARDS: dict[type[nn.Module], tuple[tuple[str, str | int], ...]] = {
    MiniMaxH3Qwen3VLQKVParallelLinear: (("q", "q"), ("k", "k"), ("v", "v")),
    MiniMaxH3Qwen3VLMergedColumnParallelLinear: (("gate", 0), ("up", 1)),
}


def _fused_source_shards(module: nn.Module) -> tuple[tuple[str, str | int], ...] | None:
    for cls, sources in _FUSED_SOURCE_SHARDS.items():
        if isinstance(module, cls):
            return sources
    return None


def _verify_fused_shards_complete(
    expected: dict[str, tuple[tuple[str, str | int], ...]],
    loaded: dict[str, set[str | int]],
) -> None:
    missing = {
        name: [source for source, shard_id in sources if shard_id not in loaded[name]]
        for name, sources in expected.items()
    }
    missing = {name: sources for name, sources in missing.items() if sources}
    if not missing:
        return
    details = "; ".join(f"{name}: {sources}" for name, sources in sorted(missing.items()))
    raise RuntimeError(f"MiniMax H3 text encoder fused weights are missing source shards: {details}")


class MiniMaxH3Qwen3VLRowParallelLinear(_MiniMaxH3OnlineNvFp4Linear, nn.Module):
    def __init__(
        self,
        group: Any,
        input_size: int,
        output_size: int,
        dtype: torch.dtype,
        input_is_parallel: bool = True,
        online_nvfp4: bool = False,
    ) -> None:
        super().__init__()
        self.group = group
        self.input_size = input_size
        self.output_size = output_size
        self.input_is_parallel = input_is_parallel
        tp_rank, tp_size = _tp_range(group)
        assert input_size % tp_size == 0, f"input_size {input_size} must be divisible by text_encoder_tp_size {tp_size}"
        self.input_size_per_partition = input_size // tp_size
        self.weight = nn.Parameter(torch.empty(output_size, self.input_size_per_partition, dtype=dtype))
        self.weight.weight_loader = self.weight_loader  # type: ignore[attr-defined]
        self._tp_rank = tp_rank
        self._tp_size = tp_size
        self._init_online_nvfp4(online_nvfp4)

    def forward(self, input_: torch.Tensor) -> torch.Tensor:
        if self.input_is_parallel:
            input_parallel = input_
        else:
            split_input = input_.split(self.input_size_per_partition, dim=-1)
            input_parallel = split_input[self._tp_rank].contiguous()
        output_parallel = self._linear(input_parallel)
        if self._tp_size > 1:
            # Reduce in fp32: the reference path accumulates the full (K=8192)
            # dot product inside a single cuBLAS GEMM before rounding to bf16.
            # A bf16 all-reduce of the per-rank partial GEMMs would round twice
            # and amplify error on this model's large-magnitude activations.
            output_parallel = output_parallel.float()
            self.group.all_reduce(output_parallel)
            # Online W4A16 conversion releases the BF16 source weight, while
            # the text encoder's externally visible row-parallel contract
            # remains BF16.
            output_parallel = output_parallel.to(torch.bfloat16)
        return output_parallel

    def quantize_for_inference(self) -> tuple[int, int]:
        return self._quantize_online_nvfp4(
            input_size=self.input_size,
            output_partition_sizes=[self.output_size],
        )

    def weight_loader(
        self, param: nn.Parameter, loaded_weight: torch.Tensor, loaded_shard_id: str | None = None
    ) -> None:
        del loaded_shard_id
        shard_size = self.input_size_per_partition
        start_idx = self._tp_rank * shard_size
        param.data.copy_(loaded_weight.narrow(1, start_idx, shard_size))


# ---------------------------------------------------------------------------
# Shared math ports (identical to the HF reference path).
# ---------------------------------------------------------------------------


class MiniMaxH3Qwen3VLRMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def _apply_rotary_pos_emb(
    q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)
    q_embed = (q * cos) + (_rotate_half(q) * sin)
    k_embed = (k * cos) + (_rotate_half(k) * sin)
    return q_embed, k_embed


def _apply_interleaved_mrope(freqs: torch.Tensor, mrope_section: list[int]) -> torch.Tensor:
    """Reorganize chunked [TTT...HHH...WWW] into interleaved [THWTHW...] layout."""
    freqs_t = freqs[0]
    for dim, offset in enumerate((1, 2), start=1):
        length = mrope_section[dim] * 3
        idx = slice(offset, length, 3)
        freqs_t[..., idx] = freqs[dim, ..., idx]
    return freqs_t


class MiniMaxH3Qwen3VLTextRotaryEmbedding(nn.Module):
    """Qwen3-VL text M-RoPE (interleaved sections). Same math as the reference."""

    def __init__(self, config: Any) -> None:
        super().__init__()
        self.config = config
        # ``dim`` and ``base`` are plain attributes so the inverse frequencies
        # are always recomputed in float32 (module-wide dtype casts must not
        # degrade the RoPE cache precision).
        self.dim = int(getattr(config, "head_dim", None) or config.hidden_size // config.num_attention_heads)
        self.base = float(config.rope_parameters["rope_theta"])
        self.mrope_section = config.rope_parameters.get("mrope_section", [24, 20, 20])

    def _get_inv_freq(self, device: torch.device) -> torch.Tensor:
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2, dtype=torch.int64).float() / self.dim))
        return inv_freq.to(device)

    def forward(self, x: torch.Tensor, position_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # position_ids: (3, 1, seq)
        inv_freq = self._get_inv_freq(position_ids.device)
        inv_freq_expanded = inv_freq[None, None, :, None].float().expand(3, position_ids.shape[1], -1, 1)
        position_ids_expanded = position_ids[:, :, None, :].float()
        freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(2, 3)
        freqs = _apply_interleaved_mrope(freqs, self.mrope_section)
        emb = torch.cat((freqs, freqs), dim=-1)
        cos = emb.cos()
        sin = emb.sin()
        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


# ---------------------------------------------------------------------------
# Qwen3-VL vision tower (replicated on every encoder rank; ~1.2 GB BF16).
# ---------------------------------------------------------------------------


class MiniMaxH3Qwen3VLVisionPatchEmbed(nn.Module):
    def __init__(self, config: Any) -> None:
        super().__init__()
        self.patch_size = config.patch_size
        self.temporal_patch_size = config.temporal_patch_size
        self.in_channels = config.in_channels
        self.embed_dim = config.hidden_size
        kernel_size = [self.temporal_patch_size, self.patch_size, self.patch_size]
        self.proj = nn.Conv3d(self.in_channels, self.embed_dim, kernel_size=kernel_size, stride=kernel_size, bias=True)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        target_dtype = self.proj.weight.dtype
        hidden_states = hidden_states.view(
            -1, self.in_channels, self.temporal_patch_size, self.patch_size, self.patch_size
        )
        hidden_states = self.proj(hidden_states.to(dtype=target_dtype)).view(-1, self.embed_dim)
        return hidden_states


class MiniMaxH3Qwen3VLVisionRotaryEmbedding(nn.Module):
    def __init__(self, dim: int, theta: float = 10000.0) -> None:
        super().__init__()
        self.dim = dim
        self.theta = theta
        # Plain attribute (not a registered buffer): the reference keeps this
        # table in float32 and module-wide dtype casts must not touch it.
        self._inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float) / dim))

    def forward(self, seqlen: int) -> torch.Tensor:
        inv_freq = self._inv_freq.to(device=self._inv_freq.device)
        seq = torch.arange(seqlen, device=inv_freq.device, dtype=torch.float)
        freqs = torch.outer(seq, inv_freq)
        return freqs


class MiniMaxH3Qwen3VLVisionPatchMerger(nn.Module):
    def __init__(self, config: Any, use_postshuffle_norm: bool = False) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size * (config.spatial_merge_size**2)
        self.use_postshuffle_norm = use_postshuffle_norm
        self.norm = nn.LayerNorm(self.hidden_size if use_postshuffle_norm else config.hidden_size, eps=1e-6)
        self.linear_fc1 = nn.Linear(self.hidden_size, self.hidden_size)
        self.act_fn = nn.GELU()
        self.linear_fc2 = nn.Linear(self.hidden_size, config.out_hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.norm(x.view(-1, self.hidden_size) if self.use_postshuffle_norm else x).view(-1, self.hidden_size)
        x = self.linear_fc2(self.act_fn(self.linear_fc1(x)))
        return x


class MiniMaxH3Qwen3VLVisionMLP(nn.Module):
    def __init__(self, config: Any) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.linear_fc1 = nn.Linear(self.hidden_size, self.intermediate_size, bias=True)
        self.linear_fc2 = nn.Linear(self.intermediate_size, self.hidden_size, bias=True)
        # The reference vision MLP uses the tanh GELU approximation
        # (``gelu_pytorch_tanh``), which differs from the exact erf GELU.
        self.act_fn = nn.GELU(approximate="tanh")

    def forward(self, hidden_state: torch.Tensor) -> torch.Tensor:
        return self.linear_fc2(self.act_fn(self.linear_fc1(hidden_state)))


def _apply_rotary_pos_emb_vision(
    q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    orig_q_dtype = q.dtype
    orig_k_dtype = k.dtype
    q, k = q.float(), k.float()
    cos, sin = cos.unsqueeze(-2).float(), sin.unsqueeze(-2).float()
    q_embed = (q * cos) + (_rotate_half(q) * sin)
    k_embed = (k * cos) + (_rotate_half(k) * sin)
    q_embed = q_embed.to(orig_q_dtype)
    k_embed = k_embed.to(orig_k_dtype)
    return q_embed, k_embed


class MiniMaxH3Qwen3VLVisionAttention(nn.Module):
    def __init__(self, config: Any) -> None:
        super().__init__()
        self.dim = config.hidden_size
        self.num_heads = config.num_heads
        self.head_dim = self.dim // self.num_heads
        self.num_key_value_groups = 1
        self.qkv = nn.Linear(self.dim, self.dim * 3, bias=True)
        self.proj = nn.Linear(self.dim, self.dim)
        self.scaling = self.head_dim**-0.5

    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        seq_length = hidden_states.shape[0]
        query_states, key_states, value_states = (
            self.qkv(hidden_states).reshape(seq_length, 3, self.num_heads, -1).permute(1, 0, 2, 3).unbind(0)
        )
        cos, sin = position_embeddings
        query_states, key_states = _apply_rotary_pos_emb_vision(query_states, key_states, cos, sin)

        query_states = query_states.transpose(0, 1).unsqueeze(0)
        key_states = key_states.transpose(0, 1).unsqueeze(0)
        value_states = value_states.transpose(0, 1).unsqueeze(0)

        lengths = cu_seqlens[1:] - cu_seqlens[:-1]
        splits = [torch.split(tensor, lengths.tolist(), dim=2) for tensor in (query_states, key_states, value_states)]
        attn_outputs = [
            F.scaled_dot_product_attention(q, k, v, dropout_p=0.0, is_causal=False) for q, k, v in zip(*splits)
        ]
        attn_output = torch.cat(attn_outputs, dim=2)
        # The SDPA kernel may return a non-contiguous (1, num_heads, seq_len, head_dim)
        # tensor whose memory layout is seq-major; transpose before flattening exactly
        # like the reference `sdpa_attention_forward` to avoid scrambling heads.
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(seq_length, -1).contiguous()
        attn_output = self.proj(attn_output)
        return attn_output


class MiniMaxH3Qwen3VLVisionBlock(nn.Module):
    def __init__(self, config: Any) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(config.hidden_size, eps=1e-6)
        self.norm2 = nn.LayerNorm(config.hidden_size, eps=1e-6)
        self.attn = MiniMaxH3Qwen3VLVisionAttention(config=config)
        self.mlp = MiniMaxH3Qwen3VLVisionMLP(config=config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        hidden_states = hidden_states + self.attn(
            self.norm1(hidden_states),
            cu_seqlens=cu_seqlens,
            position_embeddings=position_embeddings,
        )
        hidden_states = hidden_states + self.mlp(self.norm2(hidden_states))
        return hidden_states


class MiniMaxH3Qwen3VLVisionModel(nn.Module):
    """Qwen3-VL vision tower (patch embed + blocks + merger + DeepStack)."""

    def __init__(self, config: Any) -> None:
        super().__init__()
        self.config = config
        self.spatial_merge_size = config.spatial_merge_size
        self.patch_size = config.patch_size
        self.spatial_merge_unit = self.spatial_merge_size * self.spatial_merge_size

        self.patch_embed = MiniMaxH3Qwen3VLVisionPatchEmbed(config=config)
        self.pos_embed = nn.Embedding(config.num_position_embeddings, config.hidden_size)
        self.num_grid_per_side = int(config.num_position_embeddings**0.5)

        head_dim = config.hidden_size // config.num_heads
        self.rotary_pos_emb = MiniMaxH3Qwen3VLVisionRotaryEmbedding(head_dim // 2)

        self.blocks = nn.ModuleList([MiniMaxH3Qwen3VLVisionBlock(config=config) for _ in range(config.depth)])
        self.merger = MiniMaxH3Qwen3VLVisionPatchMerger(config=config, use_postshuffle_norm=False)
        self.deepstack_visual_indexes = config.deepstack_visual_indexes
        self.deepstack_merger_list = nn.ModuleList(
            [
                MiniMaxH3Qwen3VLVisionPatchMerger(config=config, use_postshuffle_norm=True)
                for _ in range(len(config.deepstack_visual_indexes))
            ]
        )

    def rot_pos_emb(self, grid_thw: torch.Tensor) -> torch.Tensor:
        merge_size = self.spatial_merge_size
        grid_thw_list = grid_thw.tolist()
        max_hw = max(max(h, w) for _, h, w in grid_thw_list)
        freq_table = self.rotary_pos_emb(max_hw).to(device=grid_thw.device)
        device = freq_table.device
        total_tokens = sum(t * h * w for t, h, w in grid_thw_list)
        pos_ids = torch.empty((total_tokens, 2), dtype=torch.long, device=device)
        offset = 0
        for num_frames, height, width in grid_thw_list:
            merged_h, merged_w = height // merge_size, width // merge_size
            block_rows = torch.arange(merged_h, device=device)
            block_cols = torch.arange(merged_w, device=device)
            intra_row = torch.arange(merge_size, device=device)
            intra_col = torch.arange(merge_size, device=device)
            row_idx = block_rows[:, None, None, None] * merge_size + intra_row[None, None, :, None]
            col_idx = block_cols[None, :, None, None] * merge_size + intra_col[None, None, None, :]
            row_idx = row_idx.expand(merged_h, merged_w, merge_size, merge_size).reshape(-1)
            col_idx = col_idx.expand(merged_h, merged_w, merge_size, merge_size).reshape(-1)
            coords = torch.stack((row_idx, col_idx), dim=-1)
            if num_frames > 1:
                coords = coords.repeat(num_frames, 1)
            num_tokens = coords.shape[0]
            pos_ids[offset : offset + num_tokens] = coords
            offset += num_tokens
        embeddings = freq_table[pos_ids]
        embeddings = embeddings.flatten(1)
        return embeddings

    def fast_pos_embed_interpolate(self, grid_thw: torch.Tensor) -> torch.Tensor:
        grid_thw_list = grid_thw.tolist()
        grid_ts = [row[0] for row in grid_thw_list]
        grid_hs = [row[1] for row in grid_thw_list]
        grid_ws = [row[2] for row in grid_thw_list]
        device = self.pos_embed.weight.device
        idx_list = [[] for _ in range(4)]
        weight_list = [[] for _ in range(4)]
        for _, h, w in grid_thw_list:
            h_idxs = torch.linspace(0, self.num_grid_per_side - 1, h)
            w_idxs = torch.linspace(0, self.num_grid_per_side - 1, w)
            h_idxs_floor = h_idxs.int()
            w_idxs_floor = w_idxs.int()
            h_idxs_ceil = (h_idxs.int() + 1).clip(max=self.num_grid_per_side - 1)
            w_idxs_ceil = (w_idxs.int() + 1).clip(max=self.num_grid_per_side - 1)
            dh = h_idxs - h_idxs_floor
            dw = w_idxs - w_idxs_floor
            base_h = h_idxs_floor * self.num_grid_per_side
            base_h_ceil = h_idxs_ceil * self.num_grid_per_side
            indices = [
                (base_h[None].T + w_idxs_floor[None]).flatten(),
                (base_h[None].T + w_idxs_ceil[None]).flatten(),
                (base_h_ceil[None].T + w_idxs_floor[None]).flatten(),
                (base_h_ceil[None].T + w_idxs_ceil[None]).flatten(),
            ]
            weights = [
                ((1 - dh)[None].T * (1 - dw)[None]).flatten(),
                ((1 - dh)[None].T * dw[None]).flatten(),
                (dh[None].T * (1 - dw)[None]).flatten(),
                (dh[None].T * dw[None]).flatten(),
            ]
            for i in range(4):
                idx_list[i].extend(indices[i].tolist())
                weight_list[i].extend(weights[i].tolist())
        idx_tensor = torch.tensor(idx_list, dtype=torch.long, device=device)
        weight_tensor = torch.tensor(weight_list, dtype=self.pos_embed.weight.dtype, device=device)
        pos_embeds = self.pos_embed(idx_tensor).to(device) * weight_tensor[:, :, None]
        patch_pos_embeds = pos_embeds[0] + pos_embeds[1] + pos_embeds[2] + pos_embeds[3]
        patch_pos_embeds = patch_pos_embeds.split([h * w for h, w in zip(grid_hs, grid_ws)])
        patch_pos_embeds_permute = []
        merge_size = self.spatial_merge_size
        for pos_embed, t, h, w in zip(patch_pos_embeds, grid_ts, grid_hs, grid_ws):
            pos_embed = pos_embed.repeat(t, 1)
            pos_embed = (
                pos_embed.view(t, h // merge_size, merge_size, w // merge_size, merge_size, -1)
                .permute(0, 1, 3, 2, 4, 5)
                .flatten(0, 4)
            )
            patch_pos_embeds_permute.append(pos_embed)
        return torch.cat(patch_pos_embeds_permute)

    def forward(
        self,
        hidden_states: torch.Tensor,
        grid_thw: torch.Tensor,
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        hidden_states = self.patch_embed(hidden_states)
        pos_embeds = self.fast_pos_embed_interpolate(grid_thw)
        hidden_states = hidden_states + pos_embeds
        rotary_pos_emb = self.rot_pos_emb(grid_thw)
        seq_len, _ = hidden_states.size()
        hidden_states = hidden_states.reshape(seq_len, -1)
        rotary_pos_emb = rotary_pos_emb.reshape(seq_len, -1)
        emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)
        position_embeddings = (emb.cos(), emb.sin())
        cu_seqlens = torch.repeat_interleave(grid_thw[:, 1] * grid_thw[:, 2], grid_thw[:, 0]).cumsum(
            dim=0, dtype=torch.int32
        )
        cu_seqlens = F.pad(cu_seqlens, (1, 0), value=0)
        deepstack_feature_lists = []
        for layer_num, blk in enumerate(self.blocks):
            hidden_states = blk(
                hidden_states,
                cu_seqlens=cu_seqlens,
                position_embeddings=position_embeddings,
            )
            if layer_num in self.deepstack_visual_indexes:
                deepstack_feature = self.deepstack_merger_list[self.deepstack_visual_indexes.index(layer_num)](
                    hidden_states
                )
                deepstack_feature_lists.append(deepstack_feature)
        merged_hidden_states = self.merger(hidden_states)
        return merged_hidden_states, deepstack_feature_lists


# ---------------------------------------------------------------------------
# Qwen3-VL text decoder layers (TP-sharded).
# ---------------------------------------------------------------------------


class MiniMaxH3Qwen3VLTextAttention(nn.Module):
    def __init__(self, group: Any, config: Any, dtype: torch.dtype, *, online_nvfp4: bool = False) -> None:
        super().__init__()
        self.group = group
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        self.scaling = self.head_dim**-0.5
        self.is_causal = True

        self.qkv_proj = MiniMaxH3Qwen3VLQKVParallelLinear(
            group,
            hidden_size=self.hidden_size,
            num_heads=self.num_heads,
            num_kv_heads=self.num_kv_heads,
            head_dim=self.head_dim,
            dtype=dtype,
            online_nvfp4=online_nvfp4,
        )
        self.o_proj = MiniMaxH3Qwen3VLRowParallelLinear(
            group,
            input_size=self.num_heads * self.head_dim,
            output_size=self.hidden_size,
            dtype=dtype,
            input_is_parallel=True,
            online_nvfp4=online_nvfp4,
        )
        self.q_norm = MiniMaxH3Qwen3VLRMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = MiniMaxH3Qwen3VLRMSNorm(self.head_dim, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)
        q_local = self.qkv_proj.local_num_heads * self.head_dim
        kv_local = self.qkv_proj.local_num_kv_heads * self.head_dim
        if self.qkv_proj.is_online_nvfp4_quantized:
            packed_qkv = self.qkv_proj(hidden_states)
            query_states = self.q_norm(packed_qkv[..., :q_local].view(hidden_shape)).transpose(1, 2)
            key_states = self.k_norm(packed_qkv[..., q_local : q_local + kv_local].view(hidden_shape)).transpose(1, 2)
            value_states = (
                packed_qkv[..., q_local + kv_local : q_local + 2 * kv_local]
                .view(hidden_shape)
                .transpose(1, 2)
            )
        else:
            q_weight = self.qkv_proj.weight[0:q_local]
            k_weight = self.qkv_proj.weight[q_local : q_local + kv_local]
            v_weight = self.qkv_proj.weight[q_local + kv_local : q_local + 2 * kv_local]
            # Three separate GEMMs keep BF16 numerics identical to the reference.
            query_states = self.q_norm(F.linear(hidden_states, q_weight).view(hidden_shape)).transpose(1, 2)
            key_states = self.k_norm(F.linear(hidden_states, k_weight).view(hidden_shape)).transpose(1, 2)
            value_states = F.linear(hidden_states, v_weight).view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = _apply_rotary_pos_emb(query_states, key_states, cos, sin)

        # Expand GQA keys/values to the local query-head count so SDPA works
        # uniformly across backends (mirrors the reference eager path).
        num_key_value_groups = self.num_heads // self.num_kv_heads
        key_states = key_states.repeat_interleave(num_key_value_groups, dim=1)
        value_states = value_states.repeat_interleave(num_key_value_groups, dim=1)

        attn_output = F.scaled_dot_product_attention(
            query_states,
            key_states,
            value_states,
            dropout_p=0.0,
            is_causal=True,
        )
        attn_output = attn_output.transpose(1, 2).reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output


class MiniMaxH3Qwen3VLTextMLP(nn.Module):
    def __init__(self, group: Any, config: Any, dtype: torch.dtype, *, online_nvfp4: bool = False) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_up_proj = MiniMaxH3Qwen3VLMergedColumnParallelLinear(
            group,
            input_size=self.hidden_size,
            intermediate_size=self.intermediate_size,
            dtype=dtype,
            online_nvfp4=online_nvfp4,
        )
        self.down_proj = MiniMaxH3Qwen3VLRowParallelLinear(
            group,
            input_size=self.intermediate_size,
            output_size=self.hidden_size,
            dtype=dtype,
            input_is_parallel=True,
            online_nvfp4=online_nvfp4,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ip_local = self.gate_up_proj.intermediate_size_per_partition
        if self.gate_up_proj.is_online_nvfp4_quantized:
            gate, up = self.gate_up_proj(x).split(ip_local, dim=-1)
        else:
            gate = F.linear(x, self.gate_up_proj.weight[0:ip_local])
            up = F.linear(x, self.gate_up_proj.weight[ip_local : 2 * ip_local])
        x = F.silu(gate) * up
        return self.down_proj(x)


class MiniMaxH3Qwen3VLTextDecoderLayer(nn.Module):
    def __init__(self, group: Any, config: Any, dtype: torch.dtype, *, online_nvfp4: bool = False) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = MiniMaxH3Qwen3VLTextAttention(group, config, dtype, online_nvfp4=online_nvfp4)
        self.mlp = MiniMaxH3Qwen3VLTextMLP(group, config, dtype, online_nvfp4=online_nvfp4)
        self.input_layernorm = MiniMaxH3Qwen3VLRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = MiniMaxH3Qwen3VLRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(hidden_states, position_embeddings=position_embeddings)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states


class MiniMaxH3Qwen3VLTextModel(nn.Module):
    """TP-sharded Qwen3-VL text model returning unnormalized layer-50 states."""

    def __init__(
        self,
        group: Any,
        config: Any,
        selected_layer: int,
        dtype: torch.dtype,
        *,
        online_nvfp4: bool = False,
    ) -> None:
        super().__init__()
        self.group = group
        self.config = config
        self.selected_layer = selected_layer
        self.vocab_size = config.vocab_size
        self.hidden_size = config.hidden_size
        self.num_layers = min(int(getattr(config, "num_hidden_layers", 0)), selected_layer)
        self.embed_tokens = MiniMaxH3Qwen3VLVocabParallelEmbedding(
            group, num_embeddings=config.vocab_size, embedding_dim=config.hidden_size, dtype=dtype
        )
        self.rotary_emb = MiniMaxH3Qwen3VLTextRotaryEmbedding(config)
        self.layers = nn.ModuleList(
            [
                MiniMaxH3Qwen3VLTextDecoderLayer(group, config, dtype, online_nvfp4=online_nvfp4)
                for _ in range(self.num_layers)
            ]
        )

    def quantize_linears_for_inference(self) -> tuple[int, int, int]:
        """Online-convert every retained custom TP linear exactly once."""
        converted = source_nbytes = state_nbytes = 0
        for decoder in self.layers:
            for linear in (
                decoder.self_attn.qkv_proj,
                decoder.self_attn.o_proj,
                decoder.mlp.gate_up_proj,
                decoder.mlp.down_proj,
            ):
                source, state = linear.quantize_for_inference()
                if source:
                    converted += 1
                    source_nbytes += source
                    state_nbytes += state
        return converted, source_nbytes, state_nbytes

    def _deepstack_process(
        self,
        hidden_states: torch.Tensor,
        visual_pos_masks: torch.Tensor,
        visual_embeds: torch.Tensor,
    ) -> torch.Tensor:
        visual_pos_masks = visual_pos_masks.to(hidden_states.device)
        visual_embeds = visual_embeds.to(hidden_states.device, hidden_states.dtype)
        hidden_states = hidden_states.clone()
        hidden_states[visual_pos_masks, :] = hidden_states[visual_pos_masks, :] + visual_embeds
        return hidden_states

    def forward(
        self,
        inputs_embeds: torch.Tensor,
        positions: torch.Tensor,
        *,
        visual_pos_masks: torch.Tensor | None = None,
        deepstack_visual_embeds: list[torch.Tensor] | None = None,
    ) -> torch.Tensor:
        hidden_states = inputs_embeds
        position_embeddings = self.rotary_emb(hidden_states, positions)
        for layer_idx, decoder_layer in enumerate(self.layers):
            hidden_states = decoder_layer(hidden_states, position_embeddings=position_embeddings)
            if deepstack_visual_embeds is not None and layer_idx in range(len(deepstack_visual_embeds)):
                hidden_states = self._deepstack_process(
                    hidden_states,
                    visual_pos_masks,
                    deepstack_visual_embeds[layer_idx],
                )
        return hidden_states


# ---------------------------------------------------------------------------
# Multimodal position / placeholder-mask helpers (ported from the reference).
# ---------------------------------------------------------------------------


def _get_vision_position_ids(
    start_position: int,
    grid_thw: list[int] | torch.Tensor,
    temp_merge_size: int,
    spatial_merge_size: int,
    device: str | torch.device,
) -> torch.Tensor:
    llm_grid_t, llm_grid_h, llm_grid_w = (
        grid_thw[0].item() // temp_merge_size,
        grid_thw[1].item() // spatial_merge_size,
        grid_thw[2].item() // spatial_merge_size,
    )
    position_temporal = torch.arange(llm_grid_t, device=device)
    position_width = torch.arange(llm_grid_w, device=device) + start_position
    position_height = torch.arange(llm_grid_h, device=device) + start_position
    position_width = position_width.repeat(llm_grid_h * llm_grid_t)
    position_height = position_height.repeat_interleave(llm_grid_w).repeat(llm_grid_t)
    position_temporal = position_temporal.repeat_interleave(llm_grid_h * llm_grid_w) + start_position
    return torch.stack([position_temporal, position_height, position_width], dim=0)


def _get_rope_index(
    input_ids: torch.Tensor,
    mm_token_type_ids: torch.Tensor,
    image_grid_thw: torch.Tensor | None,
    video_grid_thw: torch.Tensor | None,
    attention_mask: torch.Tensor | None,
    spatial_merge_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    if video_grid_thw is not None:
        video_grid_thw = torch.repeat_interleave(video_grid_thw, video_grid_thw[:, 0], dim=0)
        video_grid_thw[:, 0] = 1

    mrope_position_deltas = []
    position_ids = torch.zeros(
        3,
        input_ids.shape[0],
        input_ids.shape[1],
        dtype=input_ids.dtype,
        device=input_ids.device,
    )
    grid_iters = {
        1: iter(image_grid_thw) if image_grid_thw is not None else None,
        2: iter(video_grid_thw) if video_grid_thw is not None else None,
    }

    for batch_idx, current_input_ids in enumerate(input_ids):
        input_token_type = mm_token_type_ids[batch_idx]
        if attention_mask is not None:
            current_input_ids = current_input_ids[attention_mask[batch_idx].bool()]
            input_token_type = input_token_type[attention_mask[batch_idx].bool()]

        input_type_group = []
        for key, group in itertools.groupby(enumerate(input_token_type.tolist()), lambda x: x[1]):
            group = list(group)
            start_index = group[0][0]
            end_index = group[-1][0] + 1
            input_type_group.append((key, start_index, end_index))

        current_pos = 0
        llm_pos_ids_list = []
        for modality_type, start_idx, end_idx in input_type_group:
            if modality_type == 0:
                text_len = end_idx - start_idx
                llm_pos_ids_list.append(
                    torch.arange(text_len, device=input_ids.device).view(1, -1).expand(3, -1) + current_pos
                )
                current_pos += text_len
            else:
                grid_thw = next(grid_iters[modality_type])
                vision_position_ids = _get_vision_position_ids(
                    current_pos, grid_thw, 1, spatial_merge_size, device=input_ids.device
                )
                llm_pos_ids_list.append(vision_position_ids)
                current_pos += max(grid_thw[1], grid_thw[2]) // spatial_merge_size
        llm_positions = torch.cat(llm_pos_ids_list, dim=1).reshape(3, -1)
        if attention_mask is not None:
            position_ids[:, batch_idx, attention_mask[batch_idx].bool()] = llm_positions.to(position_ids.device)
        else:
            position_ids[:, batch_idx] = llm_positions.to(position_ids.device)
        mrope_position_deltas.append(llm_positions.max() + 1 - len(current_input_ids))
    mrope_position_deltas = torch.tensor(mrope_position_deltas, device=input_ids.device).unsqueeze(1)
    return position_ids, mrope_position_deltas


# ---------------------------------------------------------------------------
# Encoder entry point.
# ---------------------------------------------------------------------------


class MiniMaxH3Qwen3VLEncoder(nn.Module):
    """Frozen, TP-capable Qwen3-VL backbone returning layer-50 states.

    ``encoder_group`` is a ``GroupCoordinator`` over the encoder tensor-parallel
    ranks (by default the first ``text_encoder_tp_size`` DiT ranks).  Only
    ranks inside the group load weights; other ranks construct a parameter-free
    stub that never runs the forward.
    """

    def __init__(
        self,
        model_path: str,
        *,
        device: torch.device,
        load_model: bool,
        encoder_group: Any | None = None,
        online_nvfp4: bool = False,
    ) -> None:
        super().__init__()
        self.device_target = device
        self.encoder_group = encoder_group
        self.image_token_id = 151655
        self.video_token_id = 151656
        self._tp_size = 1
        self.online_nvfp4 = online_nvfp4
        self._online_nvfp4_loaded = False
        if not load_model:
            return

        from transformers import Qwen3VLConfig

        config = Qwen3VLConfig.from_pretrained(model_path, trust_remote_code=False)
        self.image_token_id = int(config.image_token_id)
        self.video_token_id = int(config.video_token_id)
        self._tp_size = int(encoder_group.world_size) if encoder_group is not None else 1
        dtype = torch.bfloat16
        self.vision = MiniMaxH3Qwen3VLVisionModel(config.vision_config)
        self.vision.to(dtype=dtype)
        self.text_model = MiniMaxH3Qwen3VLTextModel(
            encoder_group,
            config.text_config,
            MINIMAX_H3_QWEN3VL_SELECTED_LM_LAYER,
            dtype,
            online_nvfp4=online_nvfp4,
        )
        self.text_model.to(dtype=dtype)
        self._load_weights(model_path)
        logger.info(
            "MiniMax H3 Qwen3-VL encoder: %d retained decoder layers, text_encoder_tp_size=%d, "
            "vision replicated, online_nvfp4=%s",
            MINIMAX_H3_QWEN3VL_SELECTED_LM_LAYER,
            self._tp_size,
            online_nvfp4,
        )

    @property
    def is_loaded(self) -> bool:
        return hasattr(self, "text_model")

    @property
    def tp_size(self) -> int:
        return self._tp_size

    def _map_weight_name(self, name: str) -> tuple[str, str | int | None] | None:
        if name == "lm_head.weight" or name == "model.language_model.norm.weight":
            return None
        if name.startswith("model.visual."):
            return ("vision." + name[len("model.visual.") :], None)
        if name.startswith("model.language_model."):
            rest = name[len("model.language_model.") :]
            if rest.startswith("embed_tokens."):
                return ("text_model." + rest, None)
            match = re.match(r"layers\.(\d+)\.", rest)
            if match is not None and int(match.group(1)) >= MINIMAX_H3_QWEN3VL_SELECTED_LM_LAYER:
                return None
            if ".self_attn.q_proj.weight" in rest:
                return (
                    "text_model." + rest.replace(".self_attn.q_proj.weight", ".self_attn.qkv_proj.weight"),
                    "q",
                )
            if ".self_attn.k_proj.weight" in rest:
                return (
                    "text_model." + rest.replace(".self_attn.k_proj.weight", ".self_attn.qkv_proj.weight"),
                    "k",
                )
            if ".self_attn.v_proj.weight" in rest:
                return (
                    "text_model." + rest.replace(".self_attn.v_proj.weight", ".self_attn.qkv_proj.weight"),
                    "v",
                )
            if ".mlp.gate_proj.weight" in rest:
                return (
                    "text_model." + rest.replace(".mlp.gate_proj.weight", ".mlp.gate_up_proj.weight"),
                    0,
                )
            if ".mlp.up_proj.weight" in rest:
                return (
                    "text_model." + rest.replace(".mlp.up_proj.weight", ".mlp.gate_up_proj.weight"),
                    1,
                )
            return ("text_model." + rest, None)
        return None

    def _load_weights(self, model_path: str) -> None:
        import os

        import safetensors.torch

        index_path = os.path.join(model_path, "model.safetensors.index.json")
        with open(index_path, encoding="utf-8") as handle:
            index = json.load(handle)
        weight_map: dict[str, str] = index["weight_map"]
        files = sorted(set(weight_map.values()))
        params = dict(self.named_parameters())
        loaded: set[str] = set()
        # Seeded from the modules that own a multi-shard loader, so a fused weight
        # that receives no source shard at all is caught too, not only a partially
        # filled one.
        expected_fused_shards: dict[str, tuple[tuple[str, str | int], ...]] = {
            f"{module_name}.weight": sources
            for module_name, module in self.named_modules()
            if (sources := _fused_source_shards(module)) is not None
        }
        loaded_fused_shards: dict[str, set[str | int]] = {name: set() for name in expected_fused_shards}
        for shard in files:
            tensors = safetensors.torch.load_file(os.path.join(model_path, shard), device="cpu")
            for name, tensor in tensors.items():
                mapped = self._map_weight_name(name)
                if mapped is None:
                    continue
                param_name, shard_id = mapped
                param = params.get(param_name)
                if param is None:
                    logger.warning("MiniMax H3 text encoder weight %s has no target parameter", name)
                    continue
                weight_loader = getattr(param, "weight_loader", _default_weight_loader)
                weight_loader(param, tensor, shard_id)
                loaded.add(param_name)
                if shard_id is not None and param_name in loaded_fused_shards:
                    loaded_fused_shards[param_name].add(shard_id)
        _verify_fused_shards_complete(expected_fused_shards, loaded_fused_shards)
        missing = sorted(set(params) - loaded)
        if missing:
            # Listed in full: this aborts startup, so the message is the only diagnostic.
            raise RuntimeError(f"MiniMax H3 text encoder weights not loaded: {len(missing)} params: {missing}")

    def load_to_device(self) -> None:
        if not self.is_loaded:
            return
        if getattr(self, "_omni_layerwise_enabled", False):
            for name, child in self.vision.named_children():
                if name != "blocks":
                    child.to(self.device_target)
            for name, child in self.text_model.named_children():
                if name != "layers":
                    child.to(self.device_target)
            return
        self.vision.to(self.device_target)
        self.text_model.to(self.device_target)
        if self.online_nvfp4 and not self._online_nvfp4_loaded:
            started = torch.cuda.Event(enable_timing=True)
            finished = torch.cuda.Event(enable_timing=True)
            started.record()
            converted, source_nbytes, state_nbytes = self.text_model.quantize_linears_for_inference()
            finished.record()
            finished.synchronize()
            expected = MINIMAX_H3_QWEN3VL_SELECTED_LM_LAYER * 4
            if converted != expected:
                raise RuntimeError(
                    f"MiniMax H3 online NVFP4 converted {converted} TP linears; expected {expected}"
                )
            self._online_nvfp4_loaded = True
            # The BF16 source parameters were released by each linear, but
            # their blocks remain reserved in the CUDA caching allocator until
            # explicitly returned. Online conversion would otherwise retain
            # roughly the original text-linear footprint at process level.
            torch.accelerator.empty_cache()
            gib = 1024**3
            logger.info(
                "MiniMax H3 text encoder online W4A16_NVFP4 converted %d TP linears in %.3f ms: "
                "%.2f GiB BF16 -> %.2f GiB post-Marlin state (%.2f GiB released)",
                converted,
                started.elapsed_time(finished),
                source_nbytes / gib,
                state_nbytes / gib,
                (source_nbytes - state_nbytes) / gib,
            )

    def offload_to_cpu(self) -> None:
        if not self.is_loaded:
            return
        if self.online_nvfp4:
            raise RuntimeError(
                "MiniMax H3 online NVFP4 is GPU-resident after Marlin packing and does not support CPU offload"
            )
        if getattr(self, "_omni_layerwise_enabled", False):
            for hook in self._omni_layerwise_hooks:
                hook.offload_layer()
            for name, child in self.vision.named_children():
                if name != "blocks":
                    child.to("cpu")
            for name, child in self.text_model.named_children():
                if name != "layers":
                    child.to("cpu")
            torch.accelerator.empty_cache()
            return
        self.vision.to("cpu")
        self.text_model.to("cpu")
        torch.accelerator.empty_cache()

    def enable_omni_layerwise_offload(self, *, pin_memory: bool = True) -> None:
        """Stream the TP-local Qwen vision/text blocks for low-HBM serving.

        The encoder has its own TP process group, so these blocks must remain
        rank-local.  Reusing the DiT AllGather group would concatenate
        different TP shards and corrupt the encoder weights.
        """
        if not self.is_loaded or getattr(self, "_omni_layerwise_enabled", False):
            return

        from vllm_omni.diffusion.offloader.layerwise_backend import apply_block_hook
        from vllm_omni.platforms import current_omni_platform

        self._omni_layerwise_hooks = []
        self._omni_layerwise_block_groups = []
        copy_stream = current_omni_platform.Stream()
        for blocks in (self.vision.blocks, self.text_model.layers):
            if len(blocks) <= 1:
                continue
            last_hook = apply_block_hook(
                blocks[-1],
                blocks[0],
                self.device_target,
                copy_stream,
                pin_memory,
            )
            hooks = [last_hook]
            for index, block in enumerate(blocks[:-1]):
                hooks.append(
                    apply_block_hook(
                        block,
                        blocks[index + 1],
                        self.device_target,
                        copy_stream,
                        pin_memory,
                    )
                )
            for index, hook in enumerate(hooks):
                hook._prev_hook = hooks[index - 1]
            self._omni_layerwise_hooks.extend(hooks)
            self._omni_layerwise_block_groups.append(blocks)

        self._omni_layerwise_enabled = bool(self._omni_layerwise_hooks)
        logger.info(
            "MiniMax H3 encoder layerwise offload enabled on %d blocks across %d stacks",
            sum(len(blocks) for blocks in self._omni_layerwise_block_groups),
            len(self._omni_layerwise_block_groups),
        )

    def _encode(
        self,
        input_ids: torch.Tensor,
        *,
        pixel_values: torch.Tensor | None = None,
        image_grid_thw: torch.Tensor | None = None,
        pixel_values_videos: torch.Tensor | None = None,
        video_grid_thw: torch.Tensor | None = None,
    ) -> torch.Tensor:
        device = self.device_target
        ids = input_ids.to(device, torch.long)[None]
        attention_mask = torch.ones_like(ids)
        mm_types = torch.zeros_like(ids, dtype=torch.int32)
        mm_types[ids == self.image_token_id] = 1
        mm_types[ids == self.video_token_id] = 2

        inputs_embeds = self.text_model.embed_tokens(ids)
        image_mask = None
        video_mask = None
        deepstack_image_embeds: list[torch.Tensor] | None = None
        deepstack_video_embeds: list[torch.Tensor] | None = None

        if pixel_values is not None:
            image_embeds, deepstack_image_embeds = self.vision(
                pixel_values.to(device, torch.bfloat16),
                image_grid_thw.to(device, torch.long),
            )
            image_embeds = image_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
            image_mask, _ = self._get_placeholder_mask(ids, inputs_embeds=inputs_embeds, image_features=image_embeds)
            inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)
        if pixel_values_videos is not None:
            video_embeds, deepstack_video_embeds = self.vision(
                pixel_values_videos.to(device, torch.bfloat16),
                video_grid_thw.to(device, torch.long),
            )
            video_embeds = video_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
            _, video_mask = self._get_placeholder_mask(ids, inputs_embeds=inputs_embeds, video_features=video_embeds)
            inputs_embeds = inputs_embeds.masked_scatter(video_mask, video_embeds)

        visual_pos_masks, deepstack_visual_embeds = self._aggregate_deepstack(
            image_mask,
            video_mask,
            deepstack_image_embeds,
            deepstack_video_embeds,
        )

        positions, _ = _get_rope_index(
            ids,
            mm_token_type_ids=mm_types,
            image_grid_thw=image_grid_thw.to(device, torch.long) if image_grid_thw is not None else None,
            video_grid_thw=video_grid_thw.to(device, torch.long) if video_grid_thw is not None else None,
            attention_mask=attention_mask,
            spatial_merge_size=int(self.vision.config.spatial_merge_size),
        )
        hidden = self.text_model(
            inputs_embeds,
            positions,
            visual_pos_masks=visual_pos_masks,
            deepstack_visual_embeds=deepstack_visual_embeds,
        )[0]
        expected = (int(ids.shape[1]), MINIMAX_H3_QWEN3VL_HIDDEN_DIM)
        if tuple(hidden.shape) != expected:
            raise ValueError(f"unexpected Qwen3-VL hidden shape {tuple(hidden.shape)}, expected {expected}")
        return hidden.to(torch.bfloat16)

    def _get_placeholder_mask(
        self,
        input_ids: torch.Tensor,
        *,
        inputs_embeds: torch.Tensor,
        image_features: torch.Tensor | None = None,
        video_features: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        special_image_mask = input_ids == self.image_token_id
        special_video_mask = input_ids == self.video_token_id
        special_image_mask = special_image_mask.unsqueeze(-1).expand_as(inputs_embeds).to(inputs_embeds.device)
        special_video_mask = special_video_mask.unsqueeze(-1).expand_as(inputs_embeds).to(inputs_embeds.device)
        if image_features is not None and special_image_mask.sum().item() != image_features.numel():
            raise ValueError(
                "Image features and image tokens do not match: "
                f"tokens: {special_image_mask.sum().item() // inputs_embeds.shape[-1]}, "
                f"features: {image_features.shape[0]}"
            )
        if video_features is not None and special_video_mask.sum().item() != video_features.numel():
            raise ValueError(
                "Video features and video tokens do not match: "
                f"tokens: {special_video_mask.sum().item() // inputs_embeds.shape[-1]}, "
                f"features: {video_features.shape[0]}"
            )
        return special_image_mask, special_video_mask

    def _aggregate_deepstack(
        self,
        image_mask: torch.Tensor | None,
        video_mask: torch.Tensor | None,
        deepstack_image_embeds: list[torch.Tensor] | None,
        deepstack_video_embeds: list[torch.Tensor] | None,
    ) -> tuple[torch.Tensor | None, list[torch.Tensor] | None]:
        if image_mask is not None and video_mask is not None:
            image_mask = image_mask[..., 0]
            video_mask = video_mask[..., 0]
            visual_pos_masks = image_mask | video_mask
            deepstack_visual_embeds = []
            image_mask_joint = image_mask[visual_pos_masks]
            video_mask_joint = video_mask[visual_pos_masks]
            for img_embed, vid_embed in zip(deepstack_image_embeds, deepstack_video_embeds):
                embed_joint = img_embed.new_zeros(visual_pos_masks.sum(), img_embed.shape[-1]).to(img_embed.device)
                embed_joint[image_mask_joint, :] = img_embed
                embed_joint[video_mask_joint, :] = vid_embed
                deepstack_visual_embeds.append(embed_joint)
            return visual_pos_masks, deepstack_visual_embeds
        if image_mask is not None:
            image_mask = image_mask[..., 0]
            return image_mask, deepstack_image_embeds
        if video_mask is not None:
            video_mask = video_mask[..., 0]
            return video_mask, deepstack_video_embeds
        return None, None

    @torch.inference_mode()
    def encode_ids(
        self,
        input_ids: torch.Tensor,
        *,
        pixel_values: torch.Tensor | None = None,
        image_grid_thw: torch.Tensor | None = None,
        pixel_values_videos: torch.Tensor | None = None,
        video_grid_thw: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if not self.is_loaded:
            raise RuntimeError("MiniMax H3 text encoder is only loaded on the encoder TP ranks")
        if input_ids.ndim != 1:
            raise ValueError(f"input_ids must be 1-D, got {list(input_ids.shape)}")
        if (pixel_values is None) != (image_grid_thw is None):
            raise ValueError("pixel_values and image_grid_thw must be provided together")
        if (pixel_values_videos is None) != (video_grid_thw is None):
            raise ValueError("pixel_values_videos and video_grid_thw must be provided together")
        if next(self.parameters()).device.type != self.device_target.type:
            raise RuntimeError("call load_to_device() before encode_ids()")

        torch.backends.cuda.enable_cudnn_sdp(True)
        hidden = self._encode(
            input_ids,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            pixel_values_videos=pixel_values_videos,
            video_grid_thw=video_grid_thw,
        )
        return hidden.cpu()

    def forward(
        self,
        input_ids: torch.Tensor,
        *,
        pixel_values: torch.Tensor | None = None,
        image_grid_thw: torch.Tensor | None = None,
        pixel_values_videos: torch.Tensor | None = None,
        video_grid_thw: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Hook-compatible entry point for model-level CPU offloading."""
        kwargs = {
            "pixel_values": pixel_values,
            "image_grid_thw": image_grid_thw,
        }
        if pixel_values_videos is not None or video_grid_thw is not None:
            kwargs.update(
                pixel_values_videos=pixel_values_videos,
                video_grid_thw=video_grid_thw,
            )
        return self.encode_ids(input_ids, **kwargs)


__all__ = ["MiniMaxH3Qwen3VLEncoder", "minimax_h3_text_encoder_quantization"]
