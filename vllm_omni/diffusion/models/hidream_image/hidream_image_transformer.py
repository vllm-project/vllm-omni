# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project


from collections.abc import Iterable
from typing import TYPE_CHECKING

import torch
import torch.nn as nn
import torch.nn.functional as F

if TYPE_CHECKING:
    from vllm.model_executor.layers.quantization.base_config import (
        QuantizationConfig,
    )

from diffusers.models.embeddings import TimestepEmbedding, Timesteps
from diffusers.models.modeling_outputs import Transformer2DModelOutput
from diffusers.utils import deprecate, logging
from vllm.distributed import (
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
    tensor_model_parallel_all_reduce,
)
from vllm.model_executor.layers.linear import (
    QKVParallelLinear,
    RowParallelLinear,
)
from vllm.model_executor.model_loader.weight_utils import default_weight_loader

from vllm_omni.diffusion.data import OmniDiffusionConfig
from vllm_omni.diffusion.layers.rope import RotaryEmbedding

logger = logging.get_logger(__name__)


class HiDreamImageFeedForwardSwiGLU(nn.Module):
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        multiple_of: int = 256,
        ffn_dim_multiplier: float | None = None,
    ):
        super().__init__()
        hidden_dim = int(2 * hidden_dim / 3)
        # custom dim factor multiplier
        if ffn_dim_multiplier is not None:
            hidden_dim = int(ffn_dim_multiplier * hidden_dim)
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)

        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(torch.nn.functional.silu(self.w1(x)) * self.w3(x))


class HiDreamImagePooledEmbed(nn.Module):
    def __init__(self, text_emb_dim, hidden_size):
        super().__init__()
        self.pooled_embedder = TimestepEmbedding(in_channels=text_emb_dim, time_embed_dim=hidden_size)

    def forward(self, pooled_embed: torch.Tensor) -> torch.Tensor:
        #
        return self.pooled_embedder(pooled_embed.to(dtype=torch.bfloat16))


class HiDreamImageTimestepEmbed(nn.Module):
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.time_proj = Timesteps(num_channels=frequency_embedding_size, flip_sin_to_cos=True, downscale_freq_shift=0)
        self.timestep_embedder = TimestepEmbedding(in_channels=frequency_embedding_size, time_embed_dim=hidden_size)

    def forward(self, timesteps: torch.Tensor, wdtype: torch.dtype | None = None) -> torch.Tensor:
        t_emb = self.time_proj(timesteps).to(dtype=torch.bfloat16)
        t_emb = self.timestep_embedder(t_emb)
        return t_emb


class HiDreamImageOutEmbed(nn.Module):
    def __init__(self, hidden_size, patch_size, out_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, 2 * hidden_size, bias=True))

    def forward(self, hidden_states: torch.Tensor, temb: torch.Tensor) -> torch.Tensor:
        shift, scale = self.adaLN_modulation(temb).chunk(2, dim=1)
        hidden_states = self.norm_final(hidden_states) * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)
        hidden_states = self.linear(hidden_states)
        return hidden_states


class HiDreamImagePatchEmbed(nn.Module):
    def __init__(
        self,
        patch_size=2,
        in_channels=4,
        out_channels=1024,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.out_channels = out_channels
        self.proj = nn.Linear(in_channels * patch_size * patch_size, out_channels, bias=True)

    def forward(self, latent) -> torch.Tensor:
        latent = self.proj(latent.to(dtype=torch.bfloat16))
        return latent


def rope(pos: torch.Tensor, dim: int, theta: int) -> torch.Tensor:
    assert dim % 2 == 0, "The dimension must be even."

    is_mps = pos.device.type == "mps"
    is_npu = pos.device.type == "npu"

    dtype = torch.float32 if (is_mps or is_npu) else torch.float64

    scale = torch.arange(0, dim, 2, dtype=dtype, device=pos.device) / dim
    omega = 1.0 / (theta**scale)

    batch_size, seq_length = pos.shape
    out = torch.einsum("...n,d->...nd", pos, omega)
    cos_out = torch.cos(out)
    sin_out = torch.sin(out)

    stacked_out = torch.stack([cos_out, -sin_out, sin_out, cos_out], dim=-1)
    out = stacked_out.view(batch_size, -1, dim // 2, 2, 2)
    return out.float()


class HiDreamImageEmbedND(nn.Module):
    def __init__(self, theta: int, axes_dim: list[int]):
        super().__init__()
        self.theta = theta
        self.axes_dim = axes_dim

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        n_axes = ids.shape[-1]
        emb = torch.cat(
            [rope(ids[..., i], self.axes_dim[i], self.theta) for i in range(n_axes)],
            dim=-3,
        )
        return emb.unsqueeze(2)


class DistributedRMSNorm(nn.Module):
    """RMSNorm that computes global RMS across tensor parallel ranks."""

    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(hidden_size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tp_size = get_tensor_model_parallel_world_size()

        input_dtype = x.dtype
        x_float = x.float()
        local_sum_sq = (x_float**2).sum(dim=-1, keepdim=True)
        local_count = x.shape[-1]

        if tp_size > 1:
            global_sum_sq = local_sum_sq.clone()
            tensor_model_parallel_all_reduce(global_sum_sq)
            global_count = local_count * tp_size
        else:
            global_sum_sq = local_sum_sq
            global_count = local_count

        rms = torch.sqrt(global_sum_sq / global_count + self.eps)

        output = (x_float / rms) * self.weight.float()
        return output.to(input_dtype)


class HiDreamAttention(nn.Module):
    def __init__(
        self,
        query_dim: int,
        heads: int = 8,
        dim_head: int = 64,
        upcast_attention: bool = False,
        upcast_softmax: bool = False,
        scale_qk: bool = True,
        eps: float = 1e-5,
        processor=None,
        out_dim: int = None,
        single: bool = False,
        quant_config: "QuantizationConfig | None" = None,
    ):
        super().__init__()
        self.head_dim = dim_head

        self.inner_dim = out_dim if out_dim is not None else dim_head * heads
        self.query_dim = query_dim
        self.upcast_attention = upcast_attention
        self.upcast_softmax = upcast_softmax
        self.out_dim = out_dim if out_dim is not None else query_dim

        self.scale_qk = scale_qk
        self.scale = dim_head**-0.5 if self.scale_qk else 1.0

        self.heads = out_dim // dim_head if out_dim is not None else heads
        self.sliceable_head_dim = heads
        self.single = single

        # Fused QKV projection using vLLM's optimized layer
        self.to_qkv = QKVParallelLinear(
            hidden_size=query_dim,
            head_size=self.head_dim,
            total_num_heads=self.heads,
            quant_config=quant_config,
            prefix="to_qkv",
        )
        self.query_num_heads = self.to_qkv.num_heads
        self.kv_num_heads = self.to_qkv.num_kv_heads

        self.to_out = RowParallelLinear(
            self.inner_dim,
            self.out_dim,
            bias=True,
            input_is_parallel=True,
            return_bias=False,
            quant_config=quant_config,
            prefix="to_out",
        )

        self.tp_inner_dim = self.query_num_heads * dim_head
        if get_tensor_model_parallel_world_size() > 1:
            self.q_rms_norm = DistributedRMSNorm(self.tp_inner_dim, eps=eps)
            self.k_rms_norm = DistributedRMSNorm(self.tp_inner_dim, eps=eps)
        else:
            self.q_rms_norm = nn.RMSNorm(self.tp_inner_dim, eps)
            self.k_rms_norm = nn.RMSNorm(self.tp_inner_dim, eps)

        if not single:
            self.to_qkv_t = QKVParallelLinear(
                hidden_size=query_dim,
                head_size=self.head_dim,
                total_num_heads=self.heads,
                quant_config=quant_config,
                prefix="to_qkv_t",
            )
            self.to_out_t = RowParallelLinear(
                self.inner_dim,
                self.out_dim,
                bias=True,
                input_is_parallel=True,
                return_bias=False,
                quant_config=quant_config,
                prefix="to_out_t",
            )
            if get_tensor_model_parallel_world_size() > 1:
                self.q_rms_norm_t = DistributedRMSNorm(self.tp_inner_dim, eps=eps)
                self.k_rms_norm_t = DistributedRMSNorm(self.tp_inner_dim, eps=eps)
            else:
                self.q_rms_norm_t = nn.RMSNorm(self.tp_inner_dim, eps)
                self.k_rms_norm_t = nn.RMSNorm(self.tp_inner_dim, eps)
        self.rope = RotaryEmbedding(is_neox_style=False)

    def forward(
        self,
        hidden_states: torch.Tensor,
        hidden_states_masks: torch.Tensor | None = None,
        encoder_hidden_states: torch.Tensor | None = None,
        image_rotary_emb: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        dtype = hidden_states.dtype
        batch_size = hidden_states.shape[0]

        qkv, _ = self.to_qkv(hidden_states)
        q_size = self.to_qkv.num_heads * self.head_dim
        kv_size = self.to_qkv.num_kv_heads * self.head_dim
        query, key, value = qkv.split([q_size, kv_size, kv_size], dim=-1)

        query_i = self.q_rms_norm(query).to(dtype=dtype)
        key_i = self.k_rms_norm(key).to(dtype=dtype)
        value_i = value

        inner_dim = key_i.shape[-1]
        head_dim = inner_dim // self.query_num_heads

        query_i = query_i.view(batch_size, -1, self.query_num_heads, head_dim)
        key_i = key_i.view(batch_size, -1, self.kv_num_heads, head_dim)
        value_i = value_i.view(batch_size, -1, self.kv_num_heads, head_dim)
        if hidden_states_masks is not None:
            key_i = key_i * hidden_states_masks.view(batch_size, -1, 1, 1)

        if not self.single:
            qkv_t, _ = self.to_qkv_t(encoder_hidden_states)
            q_size_t = self.to_qkv_t.num_heads * self.head_dim
            kv_size_t = self.to_qkv_t.num_kv_heads * self.head_dim
            query_t, key_t, value_t = qkv_t.split([q_size_t, kv_size_t, kv_size_t], dim=-1)
            query_t = self.q_rms_norm_t(query_t).to(dtype=dtype)
            key_t = self.k_rms_norm_t(key_t).to(dtype=dtype)
            value_t = value_t

            query_t = query_t.view(batch_size, -1, self.to_qkv_t.num_heads, head_dim)
            key_t = key_t.view(batch_size, -1, self.to_qkv_t.num_kv_heads, head_dim)
            value_t = value_t.view(batch_size, -1, self.to_qkv_t.num_kv_heads, head_dim)

            num_image_tokens = query_i.shape[1]
            num_text_tokens = query_t.shape[1]
            query = torch.cat([query_i, query_t], dim=1)
            key = torch.cat([key_i, key_t], dim=1)
            value = torch.cat([value_i, value_t], dim=1)
        else:
            query = query_i
            key = key_i
            value = value_i

        cos, sin = image_rotary_emb
        cos = cos.squeeze(2).to(query.dtype)
        sin = sin.squeeze(2).to(query.dtype)
        if query.shape[-1] == image_rotary_emb[0].shape[-1] * 2:
            query = self.rope(query, cos, sin)
            key = self.rope(key, cos, sin)
        else:
            query_1, query_2 = query.chunk(2, dim=-1)
            key_1, key_2 = key.chunk(2, dim=-1)
            key_1 = self.rope(query_1, cos, sin)
            key_1 = self.rope(key_1, cos, sin)
            query = torch.cat([query_1, query_2], dim=-1)
            key = torch.cat([key_1, key_2], dim=-1)

        hidden_states = F.scaled_dot_product_attention(
            query.transpose(1, 2), key.transpose(1, 2), value.transpose(1, 2), dropout_p=0.0, is_causal=False
        )

        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, self.query_num_heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)

        if not self.single:
            hidden_states_i, hidden_states_t = torch.split(hidden_states, [num_image_tokens, num_text_tokens], dim=1)
            hidden_states_i = self.to_out(hidden_states_i)
            hidden_states_t = self.to_out_t(hidden_states_t)
            return hidden_states_i, hidden_states_t
        else:
            hidden_states = self.to_out(hidden_states)
            return hidden_states


# Modified from https://github.com/deepseek-ai/DeepSeek-V3/blob/main/inference/model.py
class MoEGate(nn.Module):
    def __init__(
        self,
        embed_dim,
        num_routed_experts=4,
        num_activated_experts=2,
        aux_loss_alpha=0.01,
        _force_inference_output=False,
    ):
        super().__init__()
        self.top_k = num_activated_experts
        self.n_routed_experts = num_routed_experts

        self.scoring_func = "softmax"
        self.alpha = aux_loss_alpha
        self.seq_aux = False

        # topk selection algorithm
        self.norm_topk_prob = False
        self.gating_dim = embed_dim
        self.weight = nn.Parameter(torch.randn(self.n_routed_experts, self.gating_dim) / embed_dim**0.5)

        self._force_inference_output = _force_inference_output

    def forward(self, hidden_states):
        bsz, seq_len, h = hidden_states.shape
        ### compute gating score
        hidden_states = hidden_states.view(-1, h)
        logits = F.linear(hidden_states, self.weight, None)
        if self.scoring_func == "softmax":
            scores = logits.softmax(dim=-1)
        else:
            raise NotImplementedError(f"insupportable scoring function for MoE gating: {self.scoring_func}")

        ### select top-k experts
        topk_weight, topk_idx = torch.topk(scores, k=self.top_k, dim=-1, sorted=False)

        ### norm gate to sum 1
        if self.top_k > 1 and self.norm_topk_prob:
            denominator = topk_weight.sum(dim=-1, keepdim=True) + 1e-20
            topk_weight = topk_weight / denominator

        ### expert-level computation auxiliary loss
        if self.training and self.alpha > 0.0 and not self._force_inference_output:
            scores_for_aux = scores
            aux_topk = self.top_k
            # always compute aux loss based on the naive greedy topk method
            topk_idx_for_aux_loss = topk_idx.view(bsz, -1)
            if self.seq_aux:
                scores_for_seq_aux = scores_for_aux.view(bsz, seq_len, -1)
                ce = torch.zeros(bsz, self.n_routed_experts, device=hidden_states.device)
                ce.scatter_add_(
                    1, topk_idx_for_aux_loss, torch.ones(bsz, seq_len * aux_topk, device=hidden_states.device)
                ).div_(seq_len * aux_topk / self.n_routed_experts)
                aux_loss = (ce * scores_for_seq_aux.mean(dim=1)).sum(dim=1).mean() * self.alpha
            else:
                mask_ce = F.one_hot(topk_idx_for_aux_loss.view(-1), num_classes=self.n_routed_experts)
                ce = mask_ce.float().mean(0)

                Pi = scores_for_aux.mean(0)
                fi = ce * self.n_routed_experts
                aux_loss = (Pi * fi).sum() * self.alpha
        else:
            aux_loss = None
        return topk_idx, topk_weight, aux_loss


# Modified from https://github.com/deepseek-ai/DeepSeek-V3/blob/main/inference/model.py
class MOEFeedForwardSwiGLU(nn.Module):
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        num_routed_experts: int,
        num_activated_experts: int,
        _force_inference_output: bool = False,
    ):
        super().__init__()
        self.shared_experts = HiDreamImageFeedForwardSwiGLU(dim, hidden_dim // 2)
        self.experts = nn.ModuleList(
            [HiDreamImageFeedForwardSwiGLU(dim, hidden_dim) for i in range(num_routed_experts)]
        )
        self._force_inference_output = _force_inference_output
        self.gate = MoEGate(
            embed_dim=dim,
            num_routed_experts=num_routed_experts,
            num_activated_experts=num_activated_experts,
            _force_inference_output=_force_inference_output,
        )
        self.num_activated_experts = num_activated_experts

    def forward(self, x):
        wtype = x.dtype
        identity = x
        orig_shape = x.shape
        topk_idx, topk_weight, aux_loss = self.gate(x)
        x = x.view(-1, x.shape[-1])
        flat_topk_idx = topk_idx.view(-1)
        if self.training and not self._force_inference_output:
            x = x.repeat_interleave(self.num_activated_experts, dim=0)
            y = torch.empty_like(x, dtype=wtype)
            for i, expert in enumerate(self.experts):
                y[flat_topk_idx == i] = expert(x[flat_topk_idx == i]).to(dtype=wtype)
            y = (y.view(*topk_weight.shape, -1) * topk_weight.unsqueeze(-1)).sum(dim=1)
            y = y.view(*orig_shape).to(dtype=wtype)
        else:
            y = self.moe_infer(x, flat_topk_idx, topk_weight.view(-1, 1)).view(*orig_shape)
        y = y + self.shared_experts(identity)
        return y

    @torch.no_grad()
    def moe_infer(self, x, flat_expert_indices, flat_expert_weights):
        expert_cache = torch.zeros_like(x)
        idxs = flat_expert_indices.argsort()
        tokens_per_expert = flat_expert_indices.bincount().cpu().numpy().cumsum(0)
        token_idxs = idxs // self.num_activated_experts

        # TODO: This MoE expert routing is sequential and can be a bottleneck for models
        # with MoE in every transformer block. Consider optimizing this in the future
        # using techniques like grouped GEMM or batched expert execution.
        for i, end_idx in enumerate(tokens_per_expert):
            start_idx = 0 if i == 0 else tokens_per_expert[i - 1]
            if start_idx == end_idx:
                continue
            expert = self.experts[i]
            exp_token_idx = token_idxs[start_idx:end_idx]
            expert_tokens = x[exp_token_idx]
            expert_out = expert(expert_tokens)
            expert_out.mul_(flat_expert_weights[idxs[start_idx:end_idx]])

            # for fp16 and other dtype
            expert_cache = expert_cache.to(expert_out.dtype)
            expert_cache.scatter_reduce_(0, exp_token_idx.view(-1, 1).repeat(1, x.shape[-1]), expert_out, reduce="sum")
        return expert_cache


class TextProjection(nn.Module):
    def __init__(self, in_features, hidden_size):
        super().__init__()
        self.linear = nn.Linear(in_features=in_features, out_features=hidden_size, bias=False)

    def forward(self, caption):
        hidden_states = self.linear(caption)
        return hidden_states


class HiDreamImageSingleTransformerBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_attention_heads: int,
        attention_head_dim: int,
        num_routed_experts: int = 4,
        num_activated_experts: int = 2,
        _force_inference_output: bool = False,
    ):
        super().__init__()
        self.num_attention_heads = num_attention_heads
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(dim, 6 * dim, bias=True))

        # 1. Attention
        self.norm1_i = nn.LayerNorm(dim, eps=1e-06, elementwise_affine=False)
        self.attn1 = HiDreamAttention(
            query_dim=dim,
            heads=num_attention_heads,
            dim_head=attention_head_dim,
            single=True,
        )

        # 3. Feed-forward
        self.norm3_i = nn.LayerNorm(dim, eps=1e-06, elementwise_affine=False)
        if num_routed_experts > 0:
            self.ff_i = MOEFeedForwardSwiGLU(
                dim=dim,
                hidden_dim=4 * dim,
                num_routed_experts=num_routed_experts,
                num_activated_experts=num_activated_experts,
                _force_inference_output=_force_inference_output,
            )
        else:
            self.ff_i = HiDreamImageFeedForwardSwiGLU(dim=dim, hidden_dim=4 * dim)

    def forward(
        self,
        hidden_states: torch.Tensor,
        hidden_states_masks: torch.Tensor | None = None,
        encoder_hidden_states: torch.Tensor | None = None,
        temb: torch.Tensor | None = None,
        image_rotary_emb: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        wtype = hidden_states.dtype
        shift_msa_i, scale_msa_i, gate_msa_i, shift_mlp_i, scale_mlp_i, gate_mlp_i = self.adaLN_modulation(temb)[
            :, None
        ].chunk(6, dim=-1)

        # 1. MM-Attention
        norm_hidden_states = self.norm1_i(hidden_states).to(dtype=wtype)
        norm_hidden_states = norm_hidden_states * (1 + scale_msa_i) + shift_msa_i
        attn_output_i = self.attn1(
            norm_hidden_states,
            hidden_states_masks,
            image_rotary_emb=image_rotary_emb,
        )
        hidden_states = gate_msa_i * attn_output_i + hidden_states

        # 2. Feed-forward
        norm_hidden_states = self.norm3_i(hidden_states).to(dtype=wtype)
        norm_hidden_states = norm_hidden_states * (1 + scale_mlp_i) + shift_mlp_i
        ff_output_i = gate_mlp_i * self.ff_i(norm_hidden_states.to(dtype=wtype))
        hidden_states = ff_output_i + hidden_states
        return hidden_states


class HiDreamImageTransformerBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_attention_heads: int,
        attention_head_dim: int,
        num_routed_experts: int = 4,
        num_activated_experts: int = 2,
        _force_inference_output: bool = False,
    ):
        super().__init__()
        self.num_attention_heads = num_attention_heads
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(dim, 12 * dim, bias=True))

        # 1. Attention
        self.norm1_i = nn.LayerNorm(dim, eps=1e-06, elementwise_affine=False)
        self.norm1_t = nn.LayerNorm(dim, eps=1e-06, elementwise_affine=False)
        self.attn1 = HiDreamAttention(
            query_dim=dim,
            heads=num_attention_heads,
            dim_head=attention_head_dim,
            single=False,
        )

        # 3. Feed-forward
        self.norm3_i = nn.LayerNorm(dim, eps=1e-06, elementwise_affine=False)
        if num_routed_experts > 0:
            self.ff_i = MOEFeedForwardSwiGLU(
                dim=dim,
                hidden_dim=4 * dim,
                num_routed_experts=num_routed_experts,
                num_activated_experts=num_activated_experts,
                _force_inference_output=_force_inference_output,
            )
        else:
            self.ff_i = HiDreamImageFeedForwardSwiGLU(dim=dim, hidden_dim=4 * dim)
        self.norm3_t = nn.LayerNorm(dim, eps=1e-06, elementwise_affine=False)
        self.ff_t = HiDreamImageFeedForwardSwiGLU(dim=dim, hidden_dim=4 * dim)

    def forward(
        self,
        hidden_states: torch.Tensor,
        hidden_states_masks: torch.Tensor | None = None,
        encoder_hidden_states: torch.Tensor | None = None,
        temb: torch.Tensor | None = None,
        image_rotary_emb: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        wtype = hidden_states.dtype
        (
            shift_msa_i,
            scale_msa_i,
            gate_msa_i,
            shift_mlp_i,
            scale_mlp_i,
            gate_mlp_i,
            shift_msa_t,
            scale_msa_t,
            gate_msa_t,
            shift_mlp_t,
            scale_mlp_t,
            gate_mlp_t,
        ) = self.adaLN_modulation(temb)[:, None].chunk(12, dim=-1)

        # 1. MM-Attention
        norm_hidden_states = self.norm1_i(hidden_states).to(dtype=wtype)
        norm_hidden_states = norm_hidden_states * (1 + scale_msa_i) + shift_msa_i
        norm_encoder_hidden_states = self.norm1_t(encoder_hidden_states).to(dtype=wtype)
        norm_encoder_hidden_states = norm_encoder_hidden_states * (1 + scale_msa_t) + shift_msa_t

        attn_output_i, attn_output_t = self.attn1(
            norm_hidden_states,
            hidden_states_masks,
            norm_encoder_hidden_states,
            image_rotary_emb=image_rotary_emb,
        )

        hidden_states = gate_msa_i * attn_output_i + hidden_states
        encoder_hidden_states = gate_msa_t * attn_output_t + encoder_hidden_states

        # 2. Feed-forward
        norm_hidden_states = self.norm3_i(hidden_states).to(dtype=wtype)
        norm_hidden_states = norm_hidden_states * (1 + scale_mlp_i) + shift_mlp_i
        norm_encoder_hidden_states = self.norm3_t(encoder_hidden_states).to(dtype=wtype)
        norm_encoder_hidden_states = norm_encoder_hidden_states * (1 + scale_mlp_t) + shift_mlp_t

        ff_output_i = gate_mlp_i * self.ff_i(norm_hidden_states)
        ff_output_t = gate_mlp_t * self.ff_t(norm_encoder_hidden_states)
        hidden_states = ff_output_i + hidden_states
        encoder_hidden_states = ff_output_t + encoder_hidden_states
        return hidden_states, encoder_hidden_states


class HiDreamBlock(nn.Module):
    def __init__(self, block: HiDreamImageTransformerBlock | HiDreamImageSingleTransformerBlock):
        super().__init__()
        self.block = block

    def forward(
        self,
        hidden_states: torch.Tensor,
        hidden_states_masks: torch.Tensor | None = None,
        encoder_hidden_states: torch.Tensor | None = None,
        temb: torch.Tensor | None = None,
        image_rotary_emb: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        return self.block(
            hidden_states=hidden_states,
            hidden_states_masks=hidden_states_masks,
            encoder_hidden_states=encoder_hidden_states,
            temb=temb,
            image_rotary_emb=image_rotary_emb,
        )


class HiDreamImageTransformer2DModel(nn.Module):
    def __init__(
        self,
        od_config: OmniDiffusionConfig = None,
        patch_size: int | None = None,
        in_channels: int = 64,
        out_channels: int | None = None,
        num_layers: int = 16,
        num_single_layers: int = 32,
        attention_head_dim: int = 128,
        num_attention_heads: int = 20,
        caption_channels: list[int] = None,
        text_emb_dim: int = 2048,
        num_routed_experts: int = 4,
        num_activated_experts: int = 2,
        axes_dims_rope: tuple[int, int] = (32, 32),
        max_resolution: tuple[int, int] = (128, 128),
        llama_layers: list[int] = None,
        force_inference_output: bool = False,
        quant_config: "QuantizationConfig | None" = None,
    ):
        super().__init__()
        self.parallel_config = od_config.parallel_config

        self.in_channels = in_channels
        self.patch_size = patch_size
        self.force_inference_output = force_inference_output
        self.llama_layers = llama_layers

        self.out_channels = out_channels or in_channels
        self.inner_dim = num_attention_heads * attention_head_dim

        self.t_embedder = HiDreamImageTimestepEmbed(self.inner_dim)
        self.p_embedder = HiDreamImagePooledEmbed(text_emb_dim, self.inner_dim)
        self.x_embedder = HiDreamImagePatchEmbed(
            patch_size=patch_size,
            in_channels=in_channels,
            out_channels=self.inner_dim,
        )
        self.pe_embedder = HiDreamImageEmbedND(theta=10000, axes_dim=axes_dims_rope)

        self.double_stream_blocks = nn.ModuleList(
            [
                HiDreamBlock(
                    HiDreamImageTransformerBlock(
                        dim=self.inner_dim,
                        num_attention_heads=num_attention_heads,
                        attention_head_dim=attention_head_dim,
                        num_routed_experts=num_routed_experts,
                        num_activated_experts=num_activated_experts,
                        _force_inference_output=force_inference_output,
                    )
                )
                for _ in range(num_layers)
            ]
        )

        self.single_stream_blocks = nn.ModuleList(
            [
                HiDreamBlock(
                    HiDreamImageSingleTransformerBlock(
                        dim=self.inner_dim,
                        num_attention_heads=num_attention_heads,
                        attention_head_dim=attention_head_dim,
                        num_routed_experts=num_routed_experts,
                        num_activated_experts=num_activated_experts,
                        _force_inference_output=force_inference_output,
                    )
                )
                for _ in range(num_single_layers)
            ]
        )

        self.final_layer = HiDreamImageOutEmbed(self.inner_dim, patch_size, self.out_channels)

        num_llama_projections = len(self.llama_layers) if self.llama_layers is not None else 0
        caption_channels = [caption_channels[1]] * num_llama_projections + [caption_channels[0]]
        caption_projection = []
        for caption_channel in caption_channels:
            caption_projection.append(TextProjection(in_features=caption_channel, hidden_size=self.inner_dim))
        self.caption_projection = nn.ModuleList(caption_projection)
        self.max_seq = max_resolution[0] * max_resolution[1] // (patch_size * patch_size)

    def unpatchify(self, x: torch.Tensor, img_sizes: list[tuple[int, int]], is_training: bool) -> list[torch.Tensor]:
        if is_training and not self.force_inference_output:
            B, S, F = x.shape
            C = F // (self.patch_size * self.patch_size)
            x = (
                x.reshape(B, S, self.patch_size, self.patch_size, C)
                .permute(0, 4, 1, 2, 3)
                .reshape(B, C, S, self.patch_size * self.patch_size)
            )
        else:
            x_arr = []
            p1 = self.patch_size
            p2 = self.patch_size
            for i, img_size in enumerate(img_sizes):
                pH, pW = img_size
                t = x[i, : pH * pW].reshape(1, pH, pW, -1)
                F_token = t.shape[-1]
                C = F_token // (p1 * p2)
                t = t.reshape(1, pH, pW, p1, p2, C)
                t = t.permute(0, 5, 1, 3, 2, 4)
                t = t.reshape(1, C, pH * p1, pW * p2)
                x_arr.append(t)
            x = torch.cat(x_arr, dim=0)
        return x

    def patchify(self, hidden_states):
        batch_size, channels, height, width = hidden_states.shape
        patch_size = self.patch_size
        patch_height, patch_width = height // patch_size, width // patch_size
        device = hidden_states.device
        dtype = hidden_states.dtype

        # create img_sizes
        img_sizes = torch.tensor([patch_height, patch_width], dtype=torch.int64, device=device).reshape(-1)
        img_sizes = img_sizes.unsqueeze(0).repeat(batch_size, 1)

        # create hidden_states_masks
        if hidden_states.shape[-2] != hidden_states.shape[-1]:
            hidden_states_masks = torch.zeros((batch_size, self.max_seq), dtype=dtype, device=device)
            hidden_states_masks[:, : patch_height * patch_width] = 1.0
        else:
            hidden_states_masks = None

        # create img_ids
        img_ids = torch.zeros(patch_height, patch_width, 3, device=device)
        row_indices = torch.arange(patch_height, device=device)[:, None]
        col_indices = torch.arange(patch_width, device=device)[None, :]
        img_ids[..., 1] = img_ids[..., 1] + row_indices
        img_ids[..., 2] = img_ids[..., 2] + col_indices
        img_ids = img_ids.reshape(patch_height * patch_width, -1)

        if hidden_states.shape[-2] != hidden_states.shape[-1]:
            # Handle non-square latents
            img_ids_pad = torch.zeros(self.max_seq, 3, device=device)
            img_ids_pad[: patch_height * patch_width, :] = img_ids
            img_ids = img_ids_pad.unsqueeze(0).repeat(batch_size, 1, 1)
        else:
            img_ids = img_ids.unsqueeze(0).repeat(batch_size, 1, 1)

        # patchify hidden_states
        if hidden_states.shape[-2] != hidden_states.shape[-1]:
            # Handle non-square latents
            out = torch.zeros(
                (batch_size, channels, self.max_seq, patch_size * patch_size),
                dtype=dtype,
                device=device,
            )
            hidden_states = hidden_states.reshape(
                batch_size, channels, patch_height, patch_size, patch_width, patch_size
            )
            hidden_states = hidden_states.permute(0, 1, 2, 4, 3, 5)
            hidden_states = hidden_states.reshape(
                batch_size, channels, patch_height * patch_width, patch_size * patch_size
            )
            out[:, :, 0 : patch_height * patch_width] = hidden_states
            hidden_states = out
            hidden_states = hidden_states.permute(0, 2, 3, 1).reshape(
                batch_size, self.max_seq, patch_size * patch_size * channels
            )

        else:
            # Handle square latents
            hidden_states = hidden_states.reshape(
                batch_size, channels, patch_height, patch_size, patch_width, patch_size
            )
            hidden_states = hidden_states.permute(0, 2, 4, 3, 5, 1)
            hidden_states = hidden_states.reshape(
                batch_size, patch_height * patch_width, patch_size * patch_size * channels
            )

        return hidden_states, hidden_states_masks, img_sizes, img_ids

    def forward(
        self,
        hidden_states: torch.Tensor,
        timesteps: torch.LongTensor = None,
        encoder_hidden_states_t5: torch.Tensor = None,
        encoder_hidden_states_llama3: torch.Tensor = None,
        pooled_embeds: torch.Tensor = None,
        img_ids: torch.Tensor | None = None,
        img_sizes: list[tuple[int, int]] | None = None,
        hidden_states_masks: torch.Tensor | None = None,
        return_dict: bool = True,
        **kwargs,
    ) -> tuple[torch.Tensor] | Transformer2DModelOutput:
        encoder_hidden_states = kwargs.get("encoder_hidden_states", None)

        if encoder_hidden_states is not None:
            deprecation_message = (
                "The `encoder_hidden_states` argument is deprecated."
                "Please use `encoder_hidden_states_t5` and `encoder_hidden_states_llama3` instead."
            )
            deprecate("encoder_hidden_states", "0.35.0", deprecation_message)
            encoder_hidden_states_t5 = encoder_hidden_states[0]
            encoder_hidden_states_llama3 = encoder_hidden_states[1]

        if img_ids is not None and img_sizes is not None and hidden_states_masks is None:
            deprecation_message = (
                "Passing `img_ids` and `img_sizes` with unpachified `hidden_states` is deprecated and will be ignored."
            )
            deprecate("img_ids", "0.35.0", deprecation_message)

        if hidden_states_masks is not None and (img_ids is None or img_sizes is None):
            raise ValueError("if `hidden_states_masks` is passed, `img_ids` and `img_sizes` must also be passed.")
        elif hidden_states_masks is not None and hidden_states.ndim != 3:
            raise ValueError(
                "if `hidden_states_masks` is passed, `hidden_states` must be a 3D tensors with shape"
                "(batch_size, patch_height * patch_width, patch_size * patch_size * channels)"
            )

        # spatial forward
        batch_size = hidden_states.shape[0]
        hidden_states_type = hidden_states.dtype

        # Patchify the input
        if hidden_states_masks is None:
            hidden_states, hidden_states_masks, img_sizes, img_ids = self.patchify(hidden_states)

        # Embed the hidden states
        hidden_states = self.x_embedder(hidden_states)

        # 0. time
        timesteps = self.t_embedder(timesteps, hidden_states_type)
        p_embedder = self.p_embedder(pooled_embeds)
        temb = timesteps + p_embedder

        encoder_hidden_states = [encoder_hidden_states_llama3[k] for k in self.llama_layers]

        if self.caption_projection is not None:
            new_encoder_hidden_states = []
            for i, enc_hidden_state in enumerate(encoder_hidden_states):
                enc_hidden_state = self.caption_projection[i](enc_hidden_state)
                enc_hidden_state = enc_hidden_state.view(batch_size, -1, hidden_states.shape[-1])
                new_encoder_hidden_states.append(enc_hidden_state)
            encoder_hidden_states = new_encoder_hidden_states
            encoder_hidden_states_t5 = self.caption_projection[-1](encoder_hidden_states_t5)
            encoder_hidden_states_t5 = encoder_hidden_states_t5.view(batch_size, -1, hidden_states.shape[-1])
            encoder_hidden_states.append(encoder_hidden_states_t5)

        txt_ids = torch.zeros(
            batch_size,
            encoder_hidden_states[-1].shape[1] + encoder_hidden_states[-2].shape[1] + encoder_hidden_states[0].shape[1],
            3,
            device=img_ids.device,
            dtype=img_ids.dtype,
        )
        image_rotary_emb = self.pe_embedder(img_ids)
        text_rotary_emb = self.pe_embedder(txt_ids)
        concat_rotary_emb = (
            torch.cat([image_rotary_emb[..., 0, 0], text_rotary_emb[..., 0, 0]], dim=1),
            torch.cat([image_rotary_emb[..., 1, 0], text_rotary_emb[..., 1, 0]], dim=1),
        )

        # 2. Blocks
        block_id = 0
        initial_encoder_hidden_states = torch.cat([encoder_hidden_states[-1], encoder_hidden_states[-2]], dim=1)
        initial_encoder_hidden_states_seq_len = initial_encoder_hidden_states.shape[1]
        for bid, block in enumerate(self.double_stream_blocks):
            cur_llama31_encoder_hidden_states = encoder_hidden_states[block_id]
            cur_encoder_hidden_states = torch.cat(
                [initial_encoder_hidden_states, cur_llama31_encoder_hidden_states], dim=1
            )
            hidden_states, initial_encoder_hidden_states = block(
                hidden_states=hidden_states,
                hidden_states_masks=hidden_states_masks,
                encoder_hidden_states=cur_encoder_hidden_states,
                temb=temb,
                image_rotary_emb=concat_rotary_emb,
            )
            initial_encoder_hidden_states = initial_encoder_hidden_states[:, :initial_encoder_hidden_states_seq_len]
            block_id += 1

        image_tokens_seq_len = hidden_states.shape[1]
        hidden_states = torch.cat([hidden_states, initial_encoder_hidden_states], dim=1)
        hidden_states_seq_len = hidden_states.shape[1]
        if hidden_states_masks is not None:
            encoder_attention_mask_ones = torch.ones(
                (batch_size, initial_encoder_hidden_states.shape[1] + cur_llama31_encoder_hidden_states.shape[1]),
                device=hidden_states_masks.device,
                dtype=hidden_states_masks.dtype,
            )
            hidden_states_masks = torch.cat([hidden_states_masks, encoder_attention_mask_ones], dim=1)

        for bid, block in enumerate(self.single_stream_blocks):
            cur_llama31_encoder_hidden_states = encoder_hidden_states[block_id]
            hidden_states = torch.cat([hidden_states, cur_llama31_encoder_hidden_states], dim=1)

            hidden_states = block(
                hidden_states=hidden_states,
                hidden_states_masks=hidden_states_masks,
                encoder_hidden_states=None,
                temb=temb,
                image_rotary_emb=concat_rotary_emb,
            )
            hidden_states = hidden_states[:, :hidden_states_seq_len]
            block_id += 1

        hidden_states = hidden_states[:, :image_tokens_seq_len, ...]
        output = self.final_layer(hidden_states, temb)
        output = self.unpatchify(output, img_sizes, self.training)
        if hidden_states_masks is not None:
            hidden_states_masks = hidden_states_masks[:, :image_tokens_seq_len]

        if not return_dict:
            return (output,)
        return Transformer2DModelOutput(sample=output)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        tp_rank = get_tensor_model_parallel_rank()
        tp_size = get_tensor_model_parallel_world_size()

        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            # self-attn
            (".to_qkv", ".to_q", "q"),
            (".to_qkv", ".to_k", "k"),
            (".to_qkv", ".to_v", "v"),
            # self-attn no single
            (".to_qkv_t", ".to_q_t", "q"),
            (".to_qkv_t", ".to_k_t", "k"),
            (".to_qkv_t", ".to_v_t", "v"),
            # cross-attn
            (".add_kv_proj", ".add_q_proj", "q"),
            (".add_kv_proj", ".add_k_proj", "k"),
            (".add_kv_proj", ".add_v_proj", "v"),
        ]

        params_dict = dict(self.named_parameters())

        # we need to load the buffers for beta and eps (XIELU)
        for name, buffer in self.named_buffers():
            if name.endswith(".beta") or name.endswith(".eps"):
                params_dict[name] = buffer

        loaded_params: set[str] = set()
        for name, loaded_weight in weights:
            original_name = name
            lookup_name = name
            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in original_name:
                    continue
                lookup_name = original_name.replace(weight_name, param_name)
                param = params_dict[lookup_name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                break
            else:
                if lookup_name not in params_dict and ".to_out.0." in lookup_name:
                    lookup_name = lookup_name.replace(".to_out.0.", ".to_out.")
                if lookup_name not in params_dict:
                    continue
                param = params_dict[lookup_name]
                if tp_size > 1 and any(
                    norm_name in lookup_name
                    for norm_name in [
                        ".attn1.q_rms_norm",
                        ".attn1.k_rms_norm",
                    ]
                ):
                    shard_size = loaded_weight.shape[0] // tp_size
                    loaded_weight = loaded_weight[tp_rank * shard_size : (tp_rank + 1) * shard_size]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, loaded_weight)
            loaded_params.add(original_name)
            loaded_params.add(lookup_name)
        return loaded_params
