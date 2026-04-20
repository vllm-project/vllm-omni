# SPDX-License-Identifier: Apache-2.0
# Copyright 2025 The vLLM-Omni team.
# Copyright 2023 Antgroup and The HuggingFace Inc. team. All rights reserved.
# Adapted from Ming
# https://github.com/inclusionAI/Ming/blob/2a0c02ae3130190160c215f89fce7de3005db483/modeling_bailing_moe_v2.py
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT used by the Meta AI team that trained the model.
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

from collections.abc import Iterable

import torch
from torch import nn
from vllm.compilation.decorators import support_torch_compile
from vllm.config import VllmConfig
from vllm.config.cache import CacheConfig
from vllm.distributed import get_pp_group, get_tensor_model_parallel_world_size
from vllm.logger import init_logger
from vllm.model_executor.layers.activation import SiluAndMul
from vllm.model_executor.layers.attention import Attention
from vllm.model_executor.layers.fused_moe import FusedMoE, SharedFusedMoE
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import (
    MergedColumnParallelLinear,
    QKVParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
)
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.model_executor.layers.rotary_embedding.mrope import MRotaryEmbedding
from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead, VocabParallelEmbedding
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.models.utils import (
    PPMissingLayer,
    WeightsMapper,
    make_empty_intermediate_tensors_factory,
    make_layers,
    maybe_prefix,
)
from vllm.sequence import IntermediateTensors
from vllm.v1.outputs import SamplerOutput
from vllm.v1.sample.sampler import Sampler

from vllm_omni.model_executor.custom_process_mixin import CustomProcessMixin
from vllm_omni.transformers_utils.configs.ming_flash_omni import BailingMoeV2Config

logger = init_logger(__name__)


class MingVideoRopeMRotaryEmbedding(MRotaryEmbedding):
    """MRotaryEmbedding with Ming's video_rope cos/sin interleaving.

    Unlike standard mrope which maps contiguous frequency sections to T/H/W,
    video_rope alternates H/W frequencies element-wise in the spatial section
    and places temporal frequencies at the end:
        Standard mrope:  [T T T ... H H H ... W W W ...]
        Video rope:      [H W H W ... H W ... T T T ...]

    Refer to Ming's BailingMoeV2RotaryEmbedding3D
    https://github.com/inclusionAI/Ming/blob/2a0c02ae3130190160c215f89fce7de3005db483/modeling_bailing_moe_v2.py#L174
    """

    def _remap_video_rope(
        self,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Remap 3D cos/sin to video_rope interleaved layout.

        Args:
            cos, sin: [3, num_tokens, rotary_dim // 2]
        Returns:
            cos, sin: [num_tokens, rotary_dim // 2]

        Refer to Ming's apply_3d_rotary_pos_emb
        https://github.com/inclusionAI/Ming/blob/2a0c02ae3130190160c215f89fce7de3005db483/modeling_bailing_moe_v2.py#L226
        """
        assert self.mrope_section is not None
        hw_size = self.mrope_section[1] + self.mrope_section[2]

        result_cos = torch.empty_like(cos[0])
        result_sin = torch.empty_like(sin[0])

        # Spatial frequencies: even indices from H (dim 1), odd from W (dim 2)
        result_cos[:, 0:hw_size:2] = cos[1, :, 0:hw_size:2]
        result_cos[:, 1:hw_size:2] = cos[2, :, 1:hw_size:2]
        result_sin[:, 0:hw_size:2] = sin[1, :, 0:hw_size:2]
        result_sin[:, 1:hw_size:2] = sin[2, :, 1:hw_size:2]

        # Temporal frequencies at the end
        result_cos[:, hw_size:] = cos[0, :, hw_size:]
        result_sin[:, hw_size:] = sin[0, :, hw_size:]

        return result_cos, result_sin

    def forward_native(
        self,
        positions: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor | None = None,
        offsets: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        assert positions.ndim == 1 or positions.ndim == 2
        assert key is not None

        cos_sin_cache = self._match_cos_sin_cache_dtype(query)
        num_tokens = positions.shape[-1]
        cos_sin = cos_sin_cache[positions]
        cos, sin = cos_sin.chunk(2, dim=-1)

        if positions.ndim == 2:
            cos, sin = self._remap_video_rope(cos, sin)

        query_shape = query.shape
        query = query.view(num_tokens, -1, self.head_size)
        query_rot = query[..., : self.rotary_dim]
        query_pass = query[..., self.rotary_dim :]
        query_rot = self.apply_rotary_emb.forward_native(query_rot, cos, sin)
        query = torch.cat((query_rot, query_pass), dim=-1).reshape(query_shape)

        key_shape = key.shape
        key = key.view(num_tokens, -1, self.head_size)
        key_rot = key[..., : self.rotary_dim]
        key_pass = key[..., self.rotary_dim :]
        key_rot = self.apply_rotary_emb.forward_native(key_rot, cos, sin)
        key = torch.cat((key_rot, key_pass), dim=-1).reshape(key_shape)

        return query, key

    def forward_cuda(
        self,
        positions: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor | None = None,
        offsets: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        # No custom Triton kernel for video_rope; fall back to native for 3D
        # TODO: Consider custom optimization
        if positions.ndim == 2:
            return self.forward_native(positions, query, key, offsets)
        return super().forward_cuda(positions, query, key, offsets)

    def forward_cpu(
        self,
        positions: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor | None = None,
        offsets: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        return self.forward_native(positions, query, key, offsets)


class BailingMoeV2MLP(nn.Module):
    def __init__(
        self,
        config: BailingMoeV2Config,
        intermediate_size: int,
        hidden_act: str = "silu",
        quant_config: QuantizationConfig | None = None,
        reduce_results: bool = True,
        prefix: str = "",
    ):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = intermediate_size

        self.gate_up_proj = MergedColumnParallelLinear(
            self.hidden_size,
            [self.intermediate_size] * 2,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.gate_up_proj",
        )
        self.down_proj = RowParallelLinear(
            self.intermediate_size,
            self.hidden_size,
            bias=False,
            quant_config=quant_config,
            reduce_results=reduce_results,
            prefix=f"{prefix}.down_proj",
        )

        if hidden_act != "silu":
            raise ValueError(f"Unsupported activation: {hidden_act}")
        self.act_fn = SiluAndMul()

    def forward(self, x):
        gate_up, _ = self.gate_up_proj(x)
        x = self.act_fn(gate_up)
        x, _ = self.down_proj(x)
        return x


class BailingMoeV2Gate(nn.Module):
    """MoE routing gate with grouped expert selection."""

    def __init__(
        self,
        config: BailingMoeV2Config,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ):
        super().__init__()
        self.config = config
        self.top_k = config.num_experts_per_tok
        self.num_experts = config.num_experts

        self.n_group = config.n_group
        self.topk_group = config.topk_group

        self.gating_dim = config.hidden_size

        self.gate = ReplicatedLinear(
            self.gating_dim,
            self.num_experts,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.gate",
        )

        self.routed_scaling_factor = config.routed_scaling_factor

        self.expert_bias = nn.Parameter(torch.zeros(self.num_experts), requires_grad=False)

    def group_limited_topk(self, scores: torch.Tensor):
        """Group-limited top-k selection for expert routing."""
        num_tokens, _ = scores.size()
        # Organize experts into groups
        group_scores = scores.view(num_tokens, self.n_group, -1).topk(2, dim=-1)[0].sum(dim=-1)
        group_idx = torch.topk(group_scores, k=self.topk_group, dim=-1, sorted=False)[1]
        group_mask = torch.zeros_like(group_scores)
        group_mask.scatter_(1, group_idx, 1)

        # Mask experts based on selected groups
        score_mask = (
            group_mask.unsqueeze(-1)
            .expand(num_tokens, self.n_group, self.num_experts // self.n_group)
            .reshape(num_tokens, -1)
        )

        masked_scores = scores.masked_fill(~score_mask.bool(), float("-inf"))
        probs, top_indices = torch.topk(masked_scores, k=self.top_k, dim=-1, sorted=False)

        return probs, top_indices

    def forward(self, hidden_states):
        # compute gating score
        hidden_states = hidden_states.view(-1, hidden_states.shape[-1])
        logits, _ = self.gate(hidden_states)

        logits = logits.float()
        scores = torch.sigmoid(logits)

        scores_for_routing = scores + self.expert_bias
        _, topk_idx = self.group_limited_topk(scores_for_routing)

        scores = torch.gather(scores, dim=1, index=topk_idx).type_as(logits)

        topk_weight = scores / (scores.sum(dim=-1, keepdim=True) + 1e-20) if self.top_k > 1 else scores
        topk_weight = topk_weight * self.routed_scaling_factor

        return topk_idx, topk_weight, logits


def _unpack_multi_routing(
    hidden_states: torch.Tensor,
    gating_output: torch.Tensor,
    topk: int,
    renormalize: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Stateless routing function that unpacks pre-computed routing results.

    Used as `custom_routing_function` for `FusedMoE`. The caller is expected
    to pack (topk_weight, topk_idx) into `gating_output` before
    calling FusedMoE.forward(), and this function unpacks them.

    Args:
        gating_output: [num_tokens, top_k * 2]
            - [:, :top_k], topk_weight (float)
            - [:, top_k:], topk_idx   (float, cast back to int)
    """
    topk_weight = gating_output[:, :topk].contiguous()
    topk_idx = gating_output[:, topk:]
    return topk_weight.to(torch.float32), topk_idx.to(torch.int32)


class BailingMoeV2SparseMoeBlock(nn.Module):
    """Sparse MoE block with MultiRouter support for multimodal routing.

    Keep the custom multi-router gating logic external.
    """

    def __init__(
        self,
        config: BailingMoeV2Config,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ):
        super().__init__()
        self.config = config
        self.tp_size = get_tensor_model_parallel_world_size()
        self.num_experts_per_tok = config.num_experts_per_tok

        if isinstance(self.config.num_shared_experts, int) and self.config.num_shared_experts > 0:
            self.shared_experts = BailingMoeV2MLP(
                config=self.config,
                intermediate_size=self.config.moe_intermediate_size * self.config.num_shared_experts,
                quant_config=quant_config,
                reduce_results=False,
                prefix=f"{prefix}.shared_experts",
            )
        else:
            self.shared_experts = None

        self.experts = SharedFusedMoE(
            shared_experts=self.shared_experts,
            num_experts=config.num_experts,
            top_k=config.num_experts_per_tok,
            hidden_size=config.hidden_size,
            intermediate_size=config.moe_intermediate_size,
            custom_routing_function=_unpack_multi_routing,
            renormalize=False,  # we handle normalization in the gate
            reduce_results=True,
            quant_config=quant_config,
            prefix=f"{prefix}.experts",
        )

        self.experts.expert_mapping = FusedMoE.make_expert_params_mapping(
            self.experts,
            ckpt_gate_proj_name="gate_proj",
            ckpt_down_proj_name="down_proj",
            ckpt_up_proj_name="up_proj",
            num_experts=config.num_experts,
        )

        self.router_type = self.config.router_type
        if self.router_type == "topN":
            self.gate = BailingMoeV2Gate(self.config, quant_config, prefix=f"{prefix}.gate")
        elif self.router_type == "MultiRouter":
            self.gate = BailingMoeV2Gate(self.config, quant_config, prefix=f"{prefix}.gate")
            self.image_gate = BailingMoeV2Gate(self.config, quant_config, prefix=f"{prefix}.image_gate")
            self.audio_gate = BailingMoeV2Gate(self.config, quant_config, prefix=f"{prefix}.audio_gate")
        else:
            raise ValueError(f"Unsupported router_type: {self.router_type}")

    @staticmethod
    def _normalize_mask(
        mask: torch.Tensor,
        bsz: int,
        seq_len: int,
        name: str,
    ) -> torch.Tensor:
        """Validate and reshape a modality mask to [bsz*seq_len, 1] bool."""
        N = bsz * seq_len
        if mask.ndim == 1:
            # vLLM path: flat tokens [N]
            assert mask.shape[0] == N, f"{name} length {mask.shape[0]} != N={N}"
        elif mask.ndim == 2:
            assert mask.shape == (bsz, seq_len), f"{name} shape {mask.shape} != ({bsz}, {seq_len})"
        elif mask.ndim == 3:
            assert mask.shape == (bsz, seq_len, 1), f"{name} shape {mask.shape} != ({bsz}, {seq_len}, 1)"
        else:
            raise ValueError(f"Unsupported {name} shape: {mask.shape}")

        return mask.reshape(N, 1).bool()

    def forward(self, hidden_states, image_mask: torch.Tensor, audio_mask: torch.Tensor):
        # TODO(yuanheng-zhao): revise the shapes in the flow
        assert 2 <= hidden_states.dim() <= 3, f"{self.__class__.__name__} only supports 2D or 3D inputs"
        input_is_2d = hidden_states.ndim == 2
        if input_is_2d:
            hidden_states = hidden_states.unsqueeze(0)

        bsz, seq_len, h = hidden_states.shape

        if self.router_type == "MultiRouter":
            image_mask = self._normalize_mask(image_mask, bsz, seq_len, "image_mask").to(hidden_states.device)
            audio_mask = self._normalize_mask(audio_mask, bsz, seq_len, "audio_mask").to(hidden_states.device)

            # if image_mask is not None and audio_mask is not None:
            #     assert torch.logical_and(image_mask, audio_mask).sum() == 0

            image_topk_idx, image_topk_weight, _ = self.image_gate(hidden_states)
            audio_topk_idx, audio_topk_weight, _ = self.audio_gate(hidden_states)
            topk_idx, topk_weight, _ = self.gate(hidden_states)

            topk_idx = torch.where(image_mask, image_topk_idx, topk_idx)
            topk_weight = torch.where(image_mask, image_topk_weight, topk_weight)
            topk_idx = torch.where(audio_mask, audio_topk_idx, topk_idx)
            topk_weight = torch.where(audio_mask, audio_topk_weight, topk_weight)
        else:
            topk_idx, topk_weight, _ = self.gate(hidden_states)

        # Pack pre-computed routing into a single tensor
        packed_routing = torch.cat(
            [
                topk_weight.to(hidden_states.dtype),
                topk_idx.to(hidden_states.dtype),
            ],
            dim=-1,
        )

        # SharedFusedMoE expects 2D hidden_states
        hidden_states_2d = hidden_states.view(-1, h)
        result = self.experts(hidden_states_2d, packed_routing)

        if self.shared_experts is not None:
            shared_output, fused_out = result
        else:
            shared_output, fused_out = None, result

        final_hidden_states = fused_out + shared_output if shared_output is not None else fused_out

        final_hidden_states = final_hidden_states.view(bsz, seq_len, h)

        return final_hidden_states.squeeze(0) if input_is_2d else final_hidden_states


class BailingMoeV2Attention(nn.Module):
    """Multi-headed attention using vLLM's Attention layer with 3D RoPE support."""

    def __init__(
        self,
        config: BailingMoeV2Config,
        layer_idx: int,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx

        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = config.head_dim

        tp_size = get_tensor_model_parallel_world_size()
        assert self.num_heads % tp_size == 0
        self.num_heads = self.num_heads // tp_size
        self.num_kv_heads = max(1, self.num_kv_heads // tp_size)

        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim**-0.5

        partial_rotary_factor = config.partial_rotary_factor
        self.rope_dim = int(self.head_dim * partial_rotary_factor)

        total_num_heads = config.num_attention_heads
        total_num_kv_heads = config.num_key_value_heads
        self.qkv_proj = QKVParallelLinear(
            self.hidden_size,
            self.head_dim,
            total_num_heads,
            total_num_kv_heads,
            bias=config.use_qkv_bias,
            quant_config=quant_config,
            prefix=f"{prefix}.qkv_proj",
        )

        self.dense = RowParallelLinear(
            total_num_heads * self.head_dim,
            self.hidden_size,
            bias=config.use_bias,
            quant_config=quant_config,
            prefix=f"{prefix}.dense",
        )

        # apply vLLM RMSNorm here rather than BailingMoeV2RMSNorm, diff might exist
        self.q_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)

        # 3D Rotary embeddings for multimodal
        if config.rope_scaling is None:
            raise ValueError("rope_scaling must not be None")

        rope_type = config.rope_scaling.get("rope_type", config.rope_scaling.get("type"))
        mrope_section = config.rope_scaling.get("mrope_section", [8, 12, 12])

        if rope_type == "video_rope":
            # Ming-specific video_rope with custom H/W interleaving
            self.rotary_emb = MingVideoRopeMRotaryEmbedding(
                head_size=self.head_dim,
                rotary_dim=self.rope_dim,
                max_position_embeddings=config.max_position_embeddings,
                base=config.rope_theta,
                is_neox_style=True,
                dtype=torch.get_default_dtype(),
                mrope_section=mrope_section,
            )
        else:
            # Standard m_rope (rope_type "3D", "default", or None)
            rope_scaling = dict(config.rope_scaling)
            rope_scaling["rope_type"] = "default"  # normalize for get_rope dispatch
            rope_scaling["mrope_section"] = mrope_section
            self.rotary_emb = get_rope(
                head_size=self.head_dim,
                max_position=config.max_position_embeddings,
                is_neox_style=True,
                rope_parameters={
                    "rope_theta": config.rope_theta,
                    "partial_rotary_factor": config.partial_rotary_factor,
                    **rope_scaling,
                },
            )

        self.attn = Attention(
            self.num_heads,
            self.head_dim,
            self.scaling,
            num_kv_heads=self.num_kv_heads,
            cache_config=cache_config,
            quant_config=quant_config,
            prefix=f"{prefix}.attn",
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass for attention with 3D RoPE.

        Args:
            positions: Position IDs, shape (3, num_tokens) for 3D rope
                or (num_tokens,) for text-only
            hidden_states: Input hidden states, shape (num_tokens, hidden_size)

        Returns:
            Attention output tensor, shape (num_tokens, hidden_size)
        """
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)

        num_tokens = q.shape[0]
        q = self.q_norm(q.view(num_tokens, self.num_heads, self.head_dim)).view(num_tokens, self.q_size)
        k = self.k_norm(k.view(num_tokens, self.num_kv_heads, self.head_dim)).view(num_tokens, self.kv_size)

        q, k = self.rotary_emb(positions, q, k)

        attn_output = self.attn(q, k, v)

        output, _ = self.dense(attn_output)
        return output


class BailingMoeV2DecoderLayer(nn.Module):
    """Decoder layer with attention and MoE MLP."""

    def __init__(
        self,
        config: BailingMoeV2Config,
        layer_idx: int,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.layer_idx = layer_idx

        self.attention = BailingMoeV2Attention(
            config=config,
            layer_idx=layer_idx,
            cache_config=cache_config,
            quant_config=quant_config,
            prefix=f"{prefix}.attention",
        )

        # MLP or MoE based on layer index
        if config.num_experts is not None and layer_idx >= config.first_k_dense_replace:
            self.mlp = BailingMoeV2SparseMoeBlock(
                config=config,
                quant_config=quant_config,
                prefix=f"{prefix}.mlp",
            )
            self.is_moe = True
        else:
            self.mlp = BailingMoeV2MLP(
                config=config,
                intermediate_size=config.intermediate_size,
                quant_config=quant_config,
                prefix=f"{prefix}.mlp",
            )
            self.is_moe = False

        # apply vLLM RMSNorm to replace BailingMoeV2RMSNorm, diff might exist
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
        image_mask: torch.Tensor | None = None,
        audio_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass for decoder layer.

        Args:
            positions: Position IDs
            hidden_states: Input hidden states
            residual: Residual connection from previous layer
            image_mask: Mask for image tokens (for MultiRouter MoE)
            audio_mask: Mask for audio tokens (for MultiRouter MoE)

        Returns:
            Tuple of (hidden_states, residual)
        """
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)

        hidden_states = self.attention(
            positions=positions,
            hidden_states=hidden_states,
        )

        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)

        if self.is_moe:
            hidden_states = self.mlp(hidden_states, image_mask, audio_mask)
        else:
            # Dense MLP only takes hidden_states (no routing masks)
            hidden_states = self.mlp(hidden_states)

        return hidden_states, residual


@support_torch_compile(
    dynamic_arg_dims={
        "input_ids": 0,
        "positions": -1,
        "intermediate_tensors": 0,
        "inputs_embeds": 0,
        "image_mask": 0,
        "audio_mask": 0,
    }
)
class BailingMoeV2Model(nn.Module):
    """BailingMoeV2 Model adapted from:

    Ming repo BailingMoeV2Model
    https://github.com/inclusionAI/Ming/blob/2a0c02ae3130190160c215f89fce7de3005db483/modeling_bailing_moe_v2.py
    vLLM repo BailingMoeModel
    https://github.com/vllm-project/vllm/blob/7291d1b288558d48508e1a17c37b0aa170332264/vllm/model_executor/models/bailing_moe.py
    """

    def __init__(
        self,
        *,
        vllm_config: VllmConfig,
        prefix: str = "",
    ):
        super().__init__()

        # BailingMoeV2Config
        config = vllm_config.model_config.hf_text_config

        cache_config = vllm_config.cache_config
        quant_config = vllm_config.quant_config

        self.config = config
        self.quant_config = quant_config
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.tie_word_embeddings = getattr(config, "tie_word_embeddings", False)

        if get_pp_group().is_first_rank or (self.tie_word_embeddings and get_pp_group().is_last_rank):
            self.word_embeddings = VocabParallelEmbedding(
                config.vocab_size,
                config.hidden_size,
                quant_config=quant_config,
                prefix=f"{prefix}.word_embeddings",
            )
        else:
            self.word_embeddings = PPMissingLayer()

        # Decoder layers with later pipeline parallelism support
        self.start_layer, self.end_layer, self.layers = make_layers(
            config.num_hidden_layers,
            lambda prefix: BailingMoeV2DecoderLayer(
                config=config,
                layer_idx=int(prefix.split(".")[-1]),
                cache_config=cache_config,
                quant_config=quant_config,
                prefix=prefix,
            ),
            prefix=f"{prefix}.layers",
        )

        if get_pp_group().is_last_rank:
            # apply vLLM RMSNorm to replace BailingMoeV2RMSNorm, diff might exist
            self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        else:
            self.norm = PPMissingLayer()

        self.make_empty_intermediate_tensors = make_empty_intermediate_tensors_factory(
            ["hidden_states", "residual"], config.hidden_size
        )

    def get_input_embeddings(self):
        return self.word_embeddings

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        image_mask: torch.Tensor | None = None,
        audio_mask: torch.Tensor | None = None,
    ) -> torch.Tensor | IntermediateTensors:
        if get_pp_group().is_first_rank:
            if inputs_embeds is not None:
                hidden_states = inputs_embeds
            else:
                hidden_states = self.word_embeddings(input_ids)
            residual = None
        else:
            assert intermediate_tensors is not None
            hidden_states = intermediate_tensors["hidden_states"]
            residual = intermediate_tensors["residual"]

        for layer in self.layers[self.start_layer : self.end_layer]:
            hidden_states, residual = layer(
                positions,
                hidden_states,
                residual,
                image_mask=image_mask,
                audio_mask=audio_mask,
            )

        if not get_pp_group().is_last_rank:
            return IntermediateTensors({"hidden_states": hidden_states, "residual": residual})

        hidden_states, _ = self.norm(hidden_states, residual)
        return hidden_states


class BailingMoeV2ForCausalLM(nn.Module, CustomProcessMixin):
    """BailingMoeV2 model for causal language modeling, adapted for vLLM.

    Inherits from CustomProcessMixin to support custom preprocessing and postprocessing
    for integration with omni model pipelines.
    """

    packed_modules_mapping = {
        "qkv_proj": ["q_proj", "k_proj", "v_proj"],
        "gate_up_proj": ["gate_proj", "up_proj"],
    }

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()

        # BailingMoeV2Config
        config = vllm_config.model_config.hf_text_config
        quant_config = vllm_config.quant_config

        self.config = config
        self.quant_config = quant_config

        self.model = BailingMoeV2Model(
            vllm_config=vllm_config,
            prefix=maybe_prefix(prefix, "model"),
        )

        self.tie_word_embeddings = getattr(config, "tie_word_embeddings", False)
        if get_pp_group().is_last_rank:
            self.lm_head = ParallelLMHead(
                config.vocab_size,
                config.hidden_size,
                quant_config=quant_config,
                prefix=maybe_prefix(prefix, "lm_head"),
            )
            if self.tie_word_embeddings:
                self.lm_head.weight = self.model.word_embeddings.weight
        else:
            self.lm_head = PPMissingLayer()

        self.logits_processor = LogitsProcessor(config.vocab_size)
        self.sampler = Sampler()
        self.make_empty_intermediate_tensors = self.model.make_empty_intermediate_tensors

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        image_mask: torch.Tensor | None = None,
        audio_mask: torch.Tensor | None = None,
    ):
        hidden_states = self.model(
            input_ids=input_ids,
            positions=positions,
            intermediate_tensors=intermediate_tensors,
            inputs_embeds=inputs_embeds,
            image_mask=image_mask,
            audio_mask=audio_mask,
        )
        return hidden_states

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
        sampling_metadata,
    ) -> torch.Tensor | None:
        logits = self.logits_processor(self.lm_head, hidden_states, sampling_metadata)
        return logits

    def sample(
        self,
        logits: torch.Tensor,
        sampling_metadata,
    ) -> SamplerOutput | None:
        next_tokens = self.sampler(logits, sampling_metadata)
        return next_tokens

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        stacked_params_mapping = [
            # (param_name, weight_name, shard_id)
            # BailingMoE stores fused QKV in checkpoint as query_key_value
            ("qkv_proj", "query_key_value", None),
            # Dense MLP and shared_experts gate/up are stored separately
            ("gate_up_proj", "gate_proj", 0),
            ("gate_up_proj", "up_proj", 1),
        ]

        # Gate router linear layers: checkpoint `{r}.weight` -> model `{r}.gate.weight`
        gate_name_mapper = WeightsMapper(
            orig_to_new_substr={f".{r}.weight": f".{r}.gate.weight" for r in ("gate", "image_gate", "audio_gate")}
        )

        # FusedMoE expert params mapping is identical across all MoE layers
        expert_params_mapping: list[tuple[str, str, int, str]] = []
        for layer in self.model.layers:
            if hasattr(layer, "mlp") and hasattr(layer.mlp, "experts"):
                expert_params_mapping = layer.mlp.experts.expert_mapping
                break

        params_dict = dict(self.named_parameters(remove_duplicate=False))
        loaded_params: set[str] = set()

        for name, loaded_weight in gate_name_mapper.apply(weights):
            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in name or "mlp.experts" in name:
                    continue
                name = name.replace(weight_name, param_name)
                param = params_dict.get(name)
                if param is not None:
                    weight_loader = param.weight_loader
                    weight_loader(param, loaded_weight, shard_id)
                    loaded_params.add(name)
                break
            else:
                for param_name, weight_name, expert_id, shard_id in expert_params_mapping:
                    if weight_name not in name:
                        continue
                    name = name.replace(weight_name, param_name)
                    param = params_dict.get(name)
                    if param is not None:
                        weight_loader = param.weight_loader
                        weight_loader(param, loaded_weight, name, shard_id=shard_id, expert_id=expert_id)
                        loaded_params.add(name)
                    break
                else:
                    param = params_dict.get(name)
                    if param is not None:
                        weight_loader = getattr(param, "weight_loader", default_weight_loader)
                        weight_loader(param, loaded_weight)
                        loaded_params.add(name)

        return loaded_params
