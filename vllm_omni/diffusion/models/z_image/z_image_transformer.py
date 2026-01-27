# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# _sp_plan definition adapted from HuggingFace diffusers library (_cp_plan)

# Copyright 2025 Alibaba Z-Image Team and The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math
from collections.abc import Iterable

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from vllm.logger import init_logger
from vllm.model_executor.layers.activation import SiluAndMul
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import (
    MergedColumnParallelLinear,
    QKVParallelLinear,
    RowParallelLinear,
)
from vllm.model_executor.model_loader.weight_utils import default_weight_loader

from vllm_omni.diffusion.attention.layer import Attention
from vllm_omni.diffusion.cache.base import CachedTransformer
from vllm_omni.diffusion.distributed.sp_plan import (
    SequenceParallelInput,
    SequenceParallelOutput,
)
from vllm_omni.diffusion.forward_context import (
    get_forward_context,
    is_forward_context_available,
)
from vllm_omni.diffusion.layers.rope import RotaryEmbedding

ADALN_EMBED_DIM = 256
SEQ_MULTI_OF = 32

logger = init_logger(__name__)


class UnifiedPrepare(nn.Module):
    """Prepares unified tensors for transformer blocks.

    This module encapsulates the unification of x and cap tensors into unified
    sequences. Similar to how Wan's `rope` module outputs rotary embeddings,
    this module outputs unified tensors that can be sharded via _sp_plan's
    split_output=True mechanism.

    This follows the diffusers pattern where tensor preparation happens in
    a dedicated submodule, enabling _sp_plan hooks to work at module boundaries.
    """

    def forward(
        self,
        x: torch.Tensor,
        x_cos: torch.Tensor,
        x_sin: torch.Tensor,
        cap_feats: torch.Tensor,
        cap_cos: torch.Tensor,
        cap_sin: torch.Tensor,
        x_item_seqlens: list[int],
        cap_item_seqlens: list[int],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Combine x and cap tensors into unified sequences.

        Returns:
            unified: Combined hidden states [batch, seq_len, dim]
            unified_cos: Combined RoPE cos [batch, seq_len, rope_dim]
            unified_sin: Combined RoPE sin [batch, seq_len, rope_dim]
            unified_attn_mask: Combined attention mask [batch, seq_len]
        """
        bsz = x.shape[0]
        device = x.device

        unified = []
        unified_cos = []
        unified_sin = []
        for i in range(bsz):
            x_len = x_item_seqlens[i]
            cap_len = cap_item_seqlens[i]
            unified.append(torch.cat([x[i][:x_len], cap_feats[i][:cap_len]]))
            unified_cos.append(torch.cat([x_cos[i][:x_len], cap_cos[i][:cap_len]]))
            unified_sin.append(torch.cat([x_sin[i][:x_len], cap_sin[i][:cap_len]]))

        unified_item_seqlens = [a + b for a, b in zip(cap_item_seqlens, x_item_seqlens)]
        unified_max_item_seqlen = max(unified_item_seqlens)

        unified = pad_sequence(unified, batch_first=True, padding_value=0.0)
        unified_cos = pad_sequence(unified_cos, batch_first=True, padding_value=0.0)
        unified_sin = pad_sequence(unified_sin, batch_first=True, padding_value=0.0)
        unified_attn_mask = torch.zeros((bsz, unified_max_item_seqlen), dtype=torch.bool, device=device)
        for i, seq_len in enumerate(unified_item_seqlens):
            unified_attn_mask[i, :seq_len] = 1

        return unified, unified_cos, unified_sin, unified_attn_mask


def _positive_divisors(n: int) -> set[int]:
    if n <= 0:
        return set()
    divs: set[int] = set()
    for d in range(1, int(math.isqrt(n)) + 1):
        if n % d == 0:
            divs.add(d)
            divs.add(n // d)
    return divs


def _get_tensor_parallel_size_from_context() -> int:
    if not is_forward_context_available():
        return 1
    try:
        od_config = get_forward_context().omni_diffusion_config
        if od_config is None:
            return 1
        return int(od_config.parallel_config.tensor_parallel_size)
    except Exception:
        return 1


def validate_zimage_tp_constraints(
    *,
    dim: int,
    n_heads: int,
    n_kv_heads: int,
    in_channels: int,
    all_patch_size: tuple[int, ...],
    all_f_patch_size: tuple[int, ...],
    tensor_parallel_size: int,
) -> tuple[int, list[int], list[int]]:
    """Validate Z-Image TP constraints without requiring a distributed context.

    Returns:
        (ffn_hidden_dim, final_out_dims, supported_tp_candidates)
    """
    tp_size = int(tensor_parallel_size)
    if tp_size <= 0:
        raise ValueError(f"tensor_parallel_size must be > 0, got {tp_size}")
    if dim % n_heads != 0:
        raise ValueError(f"dim must be divisible by n_heads, got dim={dim}, n_heads={n_heads}")
    if dim % tp_size != 0:
        supported = sorted(_positive_divisors(dim))
        raise ValueError(
            f"Z-Image requires dim % tensor_parallel_size == 0, but got dim={dim}, tp={tp_size}. "
            f"Supported tp candidates by dim: {supported}"
        )
    if n_heads % tp_size != 0:
        supported = sorted(_positive_divisors(n_heads))
        raise ValueError(
            f"Z-Image requires n_heads % tensor_parallel_size == 0, but got n_heads={n_heads}, tp={tp_size}. "
            f"Supported tp candidates by n_heads: {supported}"
        )
    if n_kv_heads % tp_size != 0:
        supported = sorted(_positive_divisors(n_kv_heads))
        raise ValueError(
            f"Z-Image requires n_kv_heads % tensor_parallel_size == 0, but got n_kv_heads={n_kv_heads}, "
            f"tp={tp_size}. Supported tp candidates by n_kv_heads: {supported}"
        )

    ffn_hidden_dim = int(dim / 3 * 8)
    if ffn_hidden_dim % tp_size != 0:
        supported = sorted(_positive_divisors(ffn_hidden_dim))
        raise ValueError(
            "Z-Image requires ffn_hidden_dim % tensor_parallel_size == 0 (for TP-sharded MLP), but got "
            f"ffn_hidden_dim={ffn_hidden_dim}, tp={tp_size}. Supported tp candidates by ffn_hidden_dim: {supported}"
        )

    final_out_dims = [
        int(patch_size) * int(patch_size) * int(f_patch_size) * int(in_channels)
        for patch_size, f_patch_size in zip(all_patch_size, all_f_patch_size)
    ]
    bad_final_out_dims = [d for d in final_out_dims if d % tp_size != 0]
    if bad_final_out_dims:
        supported = sorted(_positive_divisors(math.gcd(*final_out_dims)))
        raise ValueError(
            "Z-Image requires final projection out_features divisible by tensor_parallel_size, but got "
            f"final_out_dims={final_out_dims}, tp={tp_size}. "
            f"Supported tp candidates by final_out_dims gcd: {supported}"
        )

    supported_tp_candidates = sorted(
        _positive_divisors(n_heads)
        & _positive_divisors(n_kv_heads)
        & _positive_divisors(dim)
        & _positive_divisors(ffn_hidden_dim)
        & _positive_divisors(math.gcd(*final_out_dims))
    )
    return ffn_hidden_dim, final_out_dims, supported_tp_candidates


class TimestepEmbedder(nn.Module):
    def __init__(self, out_size, mid_size=None, frequency_embedding_size=256):
        super().__init__()
        if mid_size is None:
            mid_size = out_size
        self.mlp = nn.Sequential(
            nn.Linear(
                frequency_embedding_size,
                mid_size,
                bias=True,
            ),
            nn.SiLU(),
            nn.Linear(
                mid_size,
                out_size,
                bias=True,
            ),
        )

        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32, device=t.device) / half
        )
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        weight_dtype = self.mlp[0].weight.dtype
        if weight_dtype.is_floating_point:
            t_freq = t_freq.to(weight_dtype)
        t_emb = self.mlp(t_freq)
        return t_emb


class ZImageAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        num_kv_heads: int,
        qk_norm: bool = True,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.total_num_heads = num_heads
        self.total_num_kv_heads = num_kv_heads
        self.head_dim = dim // num_heads
        self.qk_norm = qk_norm

        self.to_qkv = QKVParallelLinear(
            hidden_size=dim,
            head_size=self.head_dim,
            total_num_heads=num_heads,
            total_num_kv_heads=num_kv_heads,
            bias=False,
        )

        assert qk_norm is True
        self.norm_q = RMSNorm(self.head_dim, eps=eps)
        self.norm_k = RMSNorm(self.head_dim, eps=eps)

        # NOTE: QKV is column-parallel on heads, so attention output is sharded
        # on the last dim (dim / tp). Use row-parallel output projection to
        # all-reduce back to full dim.
        self.to_out = nn.ModuleList(
            [
                RowParallelLinear(
                    dim,
                    dim,
                    bias=False,
                    input_is_parallel=True,
                    return_bias=False,
                )
            ]
        )

        self.attn = Attention(
            num_heads=self.to_qkv.num_heads,
            head_size=self.head_dim,
            softmax_scale=1.0 / (self.head_dim**0.5),
            causal=False,
            num_kv_heads=self.to_qkv.num_kv_heads,
        )
        self.rope = RotaryEmbedding(is_neox_style=False)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ):
        qkv, _ = self.to_qkv(hidden_states)
        q_size = self.to_qkv.num_heads * self.head_dim
        kv_size = self.to_qkv.num_kv_heads * self.head_dim
        query, key, value = qkv.split([q_size, kv_size, kv_size], dim=-1)

        query = query.unflatten(-1, (self.to_qkv.num_heads, -1))
        key = key.unflatten(-1, (self.to_qkv.num_kv_heads, -1))
        value = value.unflatten(-1, (self.to_qkv.num_kv_heads, -1))

        query = self.norm_q(query)
        key = self.norm_k(key)

        cos = cos.to(query.dtype)
        sin = sin.to(query.dtype)
        query = self.rope(query, cos, sin)
        key = self.rope(key, cos, sin)
        # Cast to correct dtype
        dtype = query.dtype
        query, key = query.to(dtype), key.to(dtype)

        # From [batch, seq_len] to [batch, 1, 1, seq_len] -> broadcast to [batch, heads, seq_len, seq_len]
        if attention_mask is not None and attention_mask.ndim == 2:
            attention_mask = attention_mask[:, None, None, :]

        # Compute joint attention
        hidden_states = self.attn(
            query,
            key,
            value,
            # attn_mask=attention_mask, # we don't support multi prompts now.
        )

        # Reshape back
        hidden_states = hidden_states.flatten(2, 3)
        hidden_states = hidden_states.to(dtype)

        hidden_states = self.to_out[0](hidden_states)

        return hidden_states


class FeedForward(nn.Module):
    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        self.w13 = MergedColumnParallelLinear(
            dim,
            [hidden_dim] * 2,
            bias=False,
            return_bias=False,
        )
        self.act = SiluAndMul()
        self.w2 = RowParallelLinear(
            hidden_dim,
            dim,
            bias=False,
            input_is_parallel=True,
            return_bias=False,
        )

    def forward(self, x):
        return self.w2(self.act(self.w13(x)))


class ZImageTransformerBlock(nn.Module):
    def __init__(
        self,
        layer_id: int,
        dim: int,
        n_heads: int,
        n_kv_heads: int,
        norm_eps: float,
        qk_norm: bool,
        modulation=True,
    ):
        super().__init__()
        self.dim = dim

        self.attention = ZImageAttention(
            dim=dim,
            num_heads=n_heads,
            num_kv_heads=n_kv_heads,
            qk_norm=qk_norm,
            eps=1e-5,
        )

        self.feed_forward = FeedForward(dim=dim, hidden_dim=int(dim / 3 * 8))
        self.layer_id = layer_id

        self.attention_norm1 = RMSNorm(dim, eps=norm_eps)
        self.ffn_norm1 = RMSNorm(dim, eps=norm_eps)

        self.attention_norm2 = RMSNorm(dim, eps=norm_eps)
        self.ffn_norm2 = RMSNorm(dim, eps=norm_eps)

        self.modulation = modulation
        if modulation:
            self.adaLN_modulation = nn.Sequential(
                nn.Linear(min(dim, ADALN_EMBED_DIM), 4 * dim, bias=True),
            )

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        adaln_input: torch.Tensor | None = None,
    ):
        if self.modulation:
            assert adaln_input is not None
            scale_msa, gate_msa, scale_mlp, gate_mlp = self.adaLN_modulation(adaln_input).unsqueeze(1).chunk(4, dim=2)
            gate_msa, gate_mlp = gate_msa.tanh(), gate_mlp.tanh()
            scale_msa, scale_mlp = 1.0 + scale_msa, 1.0 + scale_mlp

            # Attention block
            attn_out = self.attention(
                self.attention_norm1(x) * scale_msa,
                attention_mask=attn_mask,
                cos=cos,
                sin=sin,
            )
            x = x + gate_msa * self.attention_norm2(attn_out)

            # FFN block
            x = x + gate_mlp * self.ffn_norm2(
                self.feed_forward(
                    self.ffn_norm1(x) * scale_mlp,
                )
            )
        else:
            # Attention block
            attn_out = self.attention(
                self.attention_norm1(x),
                attention_mask=attn_mask,
                cos=cos,
                sin=sin,
            )
            x = x + self.attention_norm2(attn_out)

            # FFN block
            x = x + self.ffn_norm2(
                self.feed_forward(
                    self.ffn_norm1(x),
                )
            )

        return x


class FinalLayer(nn.Module):
    def __init__(self, hidden_size, out_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, out_channels, bias=True)

        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(min(hidden_size, ADALN_EMBED_DIM), hidden_size, bias=True),
        )

    def forward(self, x, c):
        scale = 1.0 + self.adaLN_modulation(c)
        x = self.norm_final(x) * scale.unsqueeze(1)
        x = self.linear(x)
        return x


class RopeEmbedder:
    def __init__(
        self,
        theta: float = 256.0,
        axes_dims: list[int] = (16, 56, 56),
        axes_lens: list[int] = (64, 128, 128),
    ):
        self.theta = theta
        self.axes_dims = axes_dims
        self.axes_lens = axes_lens
        assert len(axes_dims) == len(axes_lens), "axes_dims and axes_lens must have the same length"
        self.cos_cached = None
        self.sin_cached = None

    @staticmethod
    def precompute_freqs(dim: list[int], end: list[int], theta: float = 256.0):
        with torch.device("cpu"):
            cos_list = []
            sin_list = []
            for i, (d, e) in enumerate(zip(dim, end)):
                freqs = 1.0 / (theta ** (torch.arange(0, d, 2, dtype=torch.float64, device="cpu") / d))
                timestep = torch.arange(e, device=freqs.device, dtype=torch.float64)
                freqs = torch.outer(timestep, freqs).float()
                cos_list.append(torch.cos(freqs))
                sin_list.append(torch.sin(freqs))

            return cos_list, sin_list

    def __call__(self, ids: torch.Tensor):
        assert ids.ndim == 2
        assert ids.shape[-1] == len(self.axes_dims)
        device = ids.device

        if self.cos_cached is None:
            self.cos_cached, self.sin_cached = self.precompute_freqs(self.axes_dims, self.axes_lens, theta=self.theta)
            self.cos_cached = [c.to(device) for c in self.cos_cached]
            self.sin_cached = [s.to(device) for s in self.sin_cached]
        else:
            # Ensure cached tensors are on the same device as ids
            if self.cos_cached[0].device != device:
                self.cos_cached = [c.to(device) for c in self.cos_cached]
                self.sin_cached = [s.to(device) for s in self.sin_cached]

        cos_result = []
        sin_result = []
        for i in range(len(self.axes_dims)):
            index = ids[:, i]
            cos_result.append(self.cos_cached[i][index])
            sin_result.append(self.sin_cached[i][index])

        return torch.cat(cos_result, dim=-1), torch.cat(sin_result, dim=-1)


class ZImageTransformer2DModel(CachedTransformer):
    """Z-Image Transformer model for image generation.

    Sequence Parallelism:
        This model supports non-intrusive SP via _sp_plan. The plan specifies:
        - Input splitting at first main transformer block (unified sequence)
        - RoPE (cos/sin) splitting along sequence dimension
        - Attention mask splitting along sequence dimension
        - Output gathering at final_layer

        The SP is applied to the main `layers` transformer blocks where the
        unified image+caption sequence is processed jointly.

        Note: noise_refiner and context_refiner are NOT parallelized as they
        process image and caption separately before unification.

        Important: The default _sp_plan assumes patch_size=2 and f_patch_size=1.
        If using different patch configurations, update _sp_plan accordingly.

        Note: Our "Sequence Parallelism" (SP) corresponds to "Context Parallelism" (CP) in diffusers.
    """

    _repeated_blocks = ["ZImageTransformerBlock"]
    packed_modules_mapping = {
        "to_qkv": ["to_q", "to_k", "to_v"],
        "w13": ["w1", "w3"],
    }

    # Sequence Parallelism for Z-Image (following diffusers' _cp_plan pattern)
    # Similar to how Wan uses `rope` module's split_output to shard rotary embeddings,
    # Z-Image uses `unified_prepare` module's split_output to shard unified tensors.
    #
    # The _sp_plan specifies sharding/gathering at module boundaries:
    # - unified_prepare: Split all 4 outputs (unified, cos, sin, attn_mask) via split_output=True
    # - layers.0: hidden_states input is already sharded from unified_prepare output
    # - all_final_layer.2-1: Gather outputs after the final layer
    #
    # Note: _sp_plan corresponds to diffusers' _cp_plan (Context Parallelism)
    _sp_plan = {
        # Shard unified_prepare outputs (similar to Wan's rope module)
        # This shards all 4 return values: unified, unified_cos, unified_sin, unified_attn_mask
        "unified_prepare": {
            0: SequenceParallelInput(split_dim=1, expected_dims=3, split_output=True),  # unified
            1: SequenceParallelInput(split_dim=1, expected_dims=3, split_output=True),  # unified_cos
            2: SequenceParallelInput(split_dim=1, expected_dims=3, split_output=True),  # unified_sin
            3: SequenceParallelInput(split_dim=1, expected_dims=2, split_output=True),  # unified_attn_mask
        },
        # Gather output at final_layer (default: patch_size=2, f_patch_size=1)
        "all_final_layer.2-1": SequenceParallelOutput(gather_dim=1, expected_dims=3),
    }

    def __init__(
        self,
        all_patch_size=(2,),
        all_f_patch_size=(1,),
        in_channels=16,
        dim=3840,
        n_layers=30,
        n_refiner_layers=2,
        n_heads=30,
        n_kv_heads=30,
        norm_eps=1e-5,
        qk_norm=True,
        cap_feat_dim=2560,
        rope_theta=256.0,
        t_scale=1000.0,
        axes_dims=[32, 48, 48],
        axes_lens=[1024, 512, 512],
    ) -> None:
        super().__init__()
        self.dtype = torch.bfloat16
        self.in_channels = in_channels
        self.out_channels = in_channels
        self.all_patch_size = all_patch_size
        self.all_f_patch_size = all_f_patch_size
        self.dim = dim
        self.n_heads = n_heads

        self.rope_theta = rope_theta
        self.t_scale = t_scale
        self.gradient_checkpointing = False

        assert len(all_patch_size) == len(all_f_patch_size)

        tp_size = _get_tensor_parallel_size_from_context()
        ffn_hidden_dim, final_out_dims, supported_tp_candidates = validate_zimage_tp_constraints(
            dim=dim,
            n_heads=n_heads,
            n_kv_heads=n_kv_heads,
            in_channels=self.out_channels,
            all_patch_size=tuple(all_patch_size),
            all_f_patch_size=tuple(all_f_patch_size),
            tensor_parallel_size=tp_size,
        )

        logger.info_once(
            "Z-Image init: dim=%d n_heads=%d n_kv_heads=%d ffn_hidden_dim=%d final_out_dims=%s tp=%d (supported_tp=%s)",
            dim,
            n_heads,
            n_kv_heads,
            ffn_hidden_dim,
            tuple(final_out_dims),
            tp_size,
            tuple(supported_tp_candidates),
        )

        all_x_embedder = {}
        all_final_layer = {}
        for patch_idx, (patch_size, f_patch_size) in enumerate(zip(all_patch_size, all_f_patch_size)):
            x_embedder = nn.Linear(f_patch_size * patch_size * patch_size * in_channels, dim, bias=True)
            all_x_embedder[f"{patch_size}-{f_patch_size}"] = x_embedder

            final_layer = FinalLayer(dim, patch_size * patch_size * f_patch_size * self.out_channels)
            all_final_layer[f"{patch_size}-{f_patch_size}"] = final_layer

        self.all_x_embedder = nn.ModuleDict(all_x_embedder)
        self.all_final_layer = nn.ModuleDict(all_final_layer)
        self.noise_refiner = nn.ModuleList(
            [
                ZImageTransformerBlock(
                    1000 + layer_id,
                    dim,
                    n_heads,
                    n_kv_heads,
                    norm_eps,
                    qk_norm,
                    modulation=True,
                )
                for layer_id in range(n_refiner_layers)
            ]
        )
        self.context_refiner = nn.ModuleList(
            [
                ZImageTransformerBlock(
                    layer_id,
                    dim,
                    n_heads,
                    n_kv_heads,
                    norm_eps,
                    qk_norm,
                    modulation=False,
                )
                for layer_id in range(n_refiner_layers)
            ]
        )
        self.t_embedder = TimestepEmbedder(min(dim, ADALN_EMBED_DIM), mid_size=1024)
        self.cap_embedder = nn.Sequential(
            RMSNorm(cap_feat_dim, eps=norm_eps),
            nn.Linear(cap_feat_dim, dim, bias=True),
        )

        self.x_pad_token = nn.Parameter(torch.empty((1, dim)))
        self.cap_pad_token = nn.Parameter(torch.empty((1, dim)))

        self.layers = nn.ModuleList(
            [
                ZImageTransformerBlock(layer_id, dim, n_heads, n_kv_heads, norm_eps, qk_norm)
                for layer_id in range(n_layers)
            ]
        )
        self.axes_dims = axes_dims
        self.axes_lens = axes_lens

        self.rope_embedder = RopeEmbedder(theta=rope_theta, axes_dims=axes_dims, axes_lens=axes_lens)

        # UnifiedPrepare module for combining x and cap tensors
        # This enables _cp_plan to shard outputs via split_output=True
        # Similar to how Wan's rope module enables rotary embedding sharding
        self.unified_prepare = UnifiedPrepare()

    def unpatchify(self, x: list[torch.Tensor], size: list[tuple], patch_size, f_patch_size) -> list[torch.Tensor]:
        pH = pW = patch_size
        pF = f_patch_size
        bsz = len(x)
        assert len(size) == bsz
        for i in range(bsz):
            F, H, W = size[i]
            ori_len = (F // pF) * (H // pH) * (W // pW)
            # "f h w pf ph pw c -> c (f pf) (h ph) (w pw)"
            x[i] = (
                x[i][:ori_len]
                .view(F // pF, H // pH, W // pW, pF, pH, pW, self.out_channels)
                .permute(6, 0, 3, 1, 4, 2, 5)
                .reshape(self.out_channels, F, H, W)
            )
        return x

    @staticmethod
    def create_coordinate_grid(size, start=None, device=None):
        if start is None:
            start = (0 for _ in size)

        axes = [torch.arange(x0, x0 + span, dtype=torch.int32, device=device) for x0, span in zip(start, size)]
        grids = torch.meshgrid(axes, indexing="ij")
        return torch.stack(grids, dim=-1)

    def patchify_and_embed(
        self,
        all_image: list[torch.Tensor],
        all_cap_feats: list[torch.Tensor],
        patch_size: int,
        f_patch_size: int,
    ):
        pH = pW = patch_size
        pF = f_patch_size
        device = all_image[0].device

        all_image_out = []
        all_image_size = []
        all_image_pos_ids = []
        all_image_pad_mask = []
        all_cap_pos_ids = []
        all_cap_pad_mask = []
        all_cap_feats_out = []

        for i, (image, cap_feat) in enumerate(zip(all_image, all_cap_feats)):
            ### Process Caption
            cap_ori_len = len(cap_feat)
            cap_padding_len = (-cap_ori_len) % SEQ_MULTI_OF
            # padded position ids
            cap_padded_pos_ids = self.create_coordinate_grid(
                size=(cap_ori_len + cap_padding_len, 1, 1),
                start=(1, 0, 0),
                device=device,
            ).flatten(0, 2)
            all_cap_pos_ids.append(cap_padded_pos_ids)
            # pad mask
            all_cap_pad_mask.append(
                torch.cat(
                    [
                        torch.zeros((cap_ori_len,), dtype=torch.bool, device=device),
                        torch.ones((cap_padding_len,), dtype=torch.bool, device=device),
                    ],
                    dim=0,
                )
            )
            # padded feature
            cap_padded_feat = torch.cat(
                [cap_feat, cap_feat[-1:].repeat(cap_padding_len, 1)],
                dim=0,
            )
            all_cap_feats_out.append(cap_padded_feat)

            ### Process Image
            C, F, H, W = image.size()
            all_image_size.append((F, H, W))
            F_tokens, H_tokens, W_tokens = F // pF, H // pH, W // pW

            image = image.view(C, F_tokens, pF, H_tokens, pH, W_tokens, pW)
            # "c f pf h ph w pw -> (f h w) (pf ph pw c)"
            image = image.permute(1, 3, 5, 2, 4, 6, 0).reshape(F_tokens * H_tokens * W_tokens, pF * pH * pW * C)

            image_ori_len = len(image)
            image_padding_len = (-image_ori_len) % SEQ_MULTI_OF

            image_ori_pos_ids = self.create_coordinate_grid(
                size=(F_tokens, H_tokens, W_tokens),
                start=(cap_ori_len + cap_padding_len + 1, 0, 0),
                device=device,
            ).flatten(0, 2)
            image_padding_pos_ids = (
                self.create_coordinate_grid(
                    size=(1, 1, 1),
                    start=(0, 0, 0),
                    device=device,
                )
                .flatten(0, 2)
                .repeat(image_padding_len, 1)
            )
            image_padded_pos_ids = torch.cat([image_ori_pos_ids, image_padding_pos_ids], dim=0)
            all_image_pos_ids.append(image_padded_pos_ids)
            # pad mask
            all_image_pad_mask.append(
                torch.cat(
                    [
                        torch.zeros((image_ori_len,), dtype=torch.bool, device=device),
                        torch.ones((image_padding_len,), dtype=torch.bool, device=device),
                    ],
                    dim=0,
                )
            )
            # padded feature
            image_padded_feat = torch.cat([image, image[-1:].repeat(image_padding_len, 1)], dim=0)
            all_image_out.append(image_padded_feat)

        return (
            all_image_out,
            all_cap_feats_out,
            all_image_size,
            all_image_pos_ids,
            all_cap_pos_ids,
            all_image_pad_mask,
            all_cap_pad_mask,
        )

    def forward(
        self,
        x: list[torch.Tensor],
        t,
        cap_feats: list[torch.Tensor],
        patch_size=2,
        f_patch_size=1,
    ):
        assert patch_size in self.all_patch_size
        assert f_patch_size in self.all_f_patch_size

        bsz = len(x)
        device = x[0].device
        t = t * self.t_scale
        t = self.t_embedder(t)

        (
            x,
            cap_feats,
            x_size,
            x_pos_ids,
            cap_pos_ids,
            x_inner_pad_mask,
            cap_inner_pad_mask,
        ) = self.patchify_and_embed(x, cap_feats, patch_size, f_patch_size)

        # x embed & refine
        x_item_seqlens = [len(_) for _ in x]
        assert all(_ % SEQ_MULTI_OF == 0 for _ in x_item_seqlens)
        x_max_item_seqlen = max(x_item_seqlens)

        x = torch.cat(x, dim=0)
        x = self.all_x_embedder[f"{patch_size}-{f_patch_size}"](x)

        # Match t_embedder output dtype to x for layerwise casting compatibility
        adaln_input = t.type_as(x)
        x[torch.cat(x_inner_pad_mask)] = self.x_pad_token
        x = list(x.split(x_item_seqlens, dim=0))
        x_cos, x_sin = self.rope_embedder(torch.cat(x_pos_ids, dim=0))
        x_cos = list(x_cos.split(x_item_seqlens, dim=0))
        x_sin = list(x_sin.split(x_item_seqlens, dim=0))

        x = pad_sequence(x, batch_first=True, padding_value=0.0)
        x_cos = pad_sequence(x_cos, batch_first=True, padding_value=0.0)
        x_sin = pad_sequence(x_sin, batch_first=True, padding_value=0.0)
        x_attn_mask = torch.zeros((bsz, x_max_item_seqlen), dtype=torch.bool, device=device)
        for i, seq_len in enumerate(x_item_seqlens):
            x_attn_mask[i, :seq_len] = 1

        for layer in self.noise_refiner:
            x = layer(x, x_attn_mask, x_cos, x_sin, adaln_input)

        # cap embed & refine
        cap_item_seqlens = [len(_) for _ in cap_feats]
        assert all(_ % SEQ_MULTI_OF == 0 for _ in cap_item_seqlens)
        cap_max_item_seqlen = max(cap_item_seqlens)

        cap_feats = torch.cat(cap_feats, dim=0)
        cap_feats = self.cap_embedder(cap_feats)
        cap_feats[torch.cat(cap_inner_pad_mask)] = self.cap_pad_token
        cap_feats = list(cap_feats.split(cap_item_seqlens, dim=0))
        cap_cos, cap_sin = self.rope_embedder(torch.cat(cap_pos_ids, dim=0))
        cap_cos = list(cap_cos.split(cap_item_seqlens, dim=0))
        cap_sin = list(cap_sin.split(cap_item_seqlens, dim=0))

        cap_feats = pad_sequence(cap_feats, batch_first=True, padding_value=0.0)
        cap_cos = pad_sequence(cap_cos, batch_first=True, padding_value=0.0)
        cap_sin = pad_sequence(cap_sin, batch_first=True, padding_value=0.0)
        cap_attn_mask = torch.zeros((bsz, cap_max_item_seqlen), dtype=torch.bool, device=device)
        for i, seq_len in enumerate(cap_item_seqlens):
            cap_attn_mask[i, :seq_len] = 1

        for layer in self.context_refiner:
            cap_feats = layer(cap_feats, cap_attn_mask, cap_cos, cap_sin)

        # Prepare unified tensors via UnifiedPrepare module
        # This enables _cp_plan to shard outputs via split_output=True
        unified, unified_cos, unified_sin, unified_attn_mask = self.unified_prepare(
            x, x_cos, x_sin, cap_feats, cap_cos, cap_sin, x_item_seqlens, cap_item_seqlens
        )

        # Main transformer blocks
        for layer in self.layers:
            unified = layer(unified, unified_attn_mask, unified_cos, unified_sin, adaln_input)

        # Final layer
        unified = self.all_final_layer[f"{patch_size}-{f_patch_size}"](unified, adaln_input)

        unified = list(unified.unbind(dim=0))
        x = self.unpatchify(unified, x_size, patch_size, f_patch_size)

        return x, {}

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            # self-attn
            (".to_qkv", ".to_q", "q"),
            (".to_qkv", ".to_k", "k"),
            (".to_qkv", ".to_v", "v"),
            # ffn
            (".w13", ".w1", 0),
            (".w13", ".w3", 1),
        ]

        params_dict = dict(self.named_parameters())

        loaded_params = set[str]()
        for name, loaded_weight in weights:
            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in name:
                    continue
                name = name.replace(weight_name, param_name)
                param = params_dict[name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                break
            else:
                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, loaded_weight)
            loaded_params.add(name)
        return loaded_params
