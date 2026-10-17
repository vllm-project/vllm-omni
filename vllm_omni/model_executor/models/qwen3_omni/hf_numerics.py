# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""HF-numerics mode for the Qwen3-Omni thinker text decoder.

Enabled with ``additional_config: {hf_numerics: true}`` on the thinker stage. RMSNorm,
the residual and deepstack adds, rotary embedding and the MoE block then follow the transformers
implementation op for op, so a trainer that runs the HF model under batch-invariant
kernels recomputes the rollout log-probs bitwise. Each replaced op is a custom op,
opaque to torch.compile.
"""

import torch
import torch.nn.functional as F
from torch import nn
from transformers import PretrainedConfig
from transformers.activations import ACT2FN
from vllm.config import CUDAGraphMode, VllmConfig
from vllm.model_executor.layers.fused_moe import UnquantizedFusedMoEMethod
from vllm.model_executor.layers.fused_moe.oracle.unquantized import UnquantizedMoeBackend
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.models.qwen3_moe import Qwen3MoeDecoderLayer, Qwen3MoeSparseMoeBlock
from vllm.platforms import current_platform


def _rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    h = x.float()
    h = h * torch.rsqrt(h.pow(2).mean(-1, keepdim=True) + eps)
    return weight * h.to(x.dtype)


@torch.library.custom_op("vllm_omni::hf_rms_norm", mutates_args=())
def hf_rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    return _rms_norm(x, weight, eps)


@hf_rms_norm.register_fake
def _(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    return torch.empty_like(x)


@torch.library.custom_op("vllm_omni::hf_add_rms_norm", mutates_args=())
def hf_add_rms_norm(
    x: torch.Tensor, residual: torch.Tensor, weight: torch.Tensor, eps: float
) -> tuple[torch.Tensor, torch.Tensor]:
    s = x + residual
    return _rms_norm(s, weight, eps), s


@hf_add_rms_norm.register_fake
def _(x: torch.Tensor, residual: torch.Tensor, weight: torch.Tensor, eps: float) -> tuple[torch.Tensor, torch.Tensor]:
    return torch.empty_like(x), torch.empty_like(x)


@torch.library.custom_op("vllm_omni::hf_add_deepstack", mutates_args=())
def hf_add_deepstack(x: torch.Tensor, residual: torch.Tensor, deepstack: torch.Tensor) -> torch.Tensor:
    """HF adds the deepstack embeds to the decoder layer output, i.e. after the residual add."""
    return (x + residual) + deepstack


@hf_add_deepstack.register_fake
def _(x: torch.Tensor, residual: torch.Tensor, deepstack: torch.Tensor) -> torch.Tensor:
    return torch.empty_like(x)


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


@torch.library.custom_op("vllm_omni::hf_mrope", mutates_args=())
def hf_mrope(
    positions: torch.Tensor,
    query: torch.Tensor,
    key: torch.Tensor,
    inv_freq: torch.Tensor,
    mrope_section: list[int],
    head_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    pos = positions if positions.dim() == 2 else positions[None].expand(3, -1)
    inv_freq_expanded = inv_freq[None, :, None].float().expand(3, -1, 1)
    freqs = (inv_freq_expanded @ pos[:, None, :].float()).transpose(1, 2)
    freqs_t = freqs[0]
    for dim, offset in enumerate((1, 2), start=1):
        idx = slice(offset, mrope_section[dim] * 3, 3)
        freqs_t[..., idx] = freqs[dim, ..., idx]
    emb = torch.cat((freqs_t, freqs_t), dim=-1)
    cos, sin = emb.cos().to(query.dtype)[:, None, :], emb.sin().to(query.dtype)[:, None, :]
    n = query.shape[0]
    q, k = query.view(n, -1, head_size), key.view(n, -1, head_size)
    q = q * cos + _rotate_half(q) * sin
    k = k * cos + _rotate_half(k) * sin
    return q.reshape(n, -1), k.reshape(n, -1)


@hf_mrope.register_fake
def _(
    positions: torch.Tensor,
    query: torch.Tensor,
    key: torch.Tensor,
    inv_freq: torch.Tensor,
    mrope_section: list[int],
    head_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    return torch.empty_like(query), torch.empty_like(key)


class _Experts:
    """The attributes ``grouped_mm_experts_forward`` reads from an HF experts module."""

    has_gate = True
    has_bias = False
    is_transposed = False
    is_concatenated = True
    _is_expert_parallel = False

    def __init__(self, gate_up_proj: torch.Tensor, down_proj: torch.Tensor, hidden_act: str):
        self.gate_up_proj = gate_up_proj
        self.down_proj = down_proj
        self.num_experts = gate_up_proj.shape[0]
        self.act_fn = ACT2FN[hidden_act]

    def _apply_gate(self, gate_up_out: torch.Tensor) -> torch.Tensor:
        from transformers.integrations.moe import _default_apply_gate

        return _default_apply_gate(self, gate_up_out)


@torch.library.custom_op("vllm_omni::hf_moe", mutates_args=())
def hf_moe(
    x: torch.Tensor,
    gate_weight: torch.Tensor,
    w13: torch.Tensor,
    w2: torch.Tensor,
    top_k: int,
    norm_topk_prob: bool,
    hidden_act: str,
) -> torch.Tensor:
    # Imported here so the default thinker path never loads this private transformers helper.
    from transformers.integrations.moe import grouped_mm_experts_forward

    router_logits = F.linear(x, gate_weight)
    router_probs = F.softmax(router_logits, dtype=torch.float, dim=-1)
    top_value, top_index = torch.topk(router_probs, top_k, dim=-1)
    if norm_topk_prob:
        top_value = top_value / top_value.sum(dim=-1, keepdim=True)
    top_value = top_value.to(router_logits.dtype)
    return grouped_mm_experts_forward(_Experts(w13, w2, hidden_act), x, top_index, top_value)


@hf_moe.register_fake
def _(
    x: torch.Tensor,
    gate_weight: torch.Tensor,
    w13: torch.Tensor,
    w2: torch.Tensor,
    top_k: int,
    norm_topk_prob: bool,
    hidden_act: str,
) -> torch.Tensor:
    return torch.empty_like(x)


class HFNumericsRMSNorm(RMSNorm):
    def forward(self, x: torch.Tensor, residual: torch.Tensor | None = None):
        if residual is None:
            return torch.ops.vllm_omni.hf_rms_norm(x, self.weight, self.variance_epsilon)
        return torch.ops.vllm_omni.hf_add_rms_norm(x, residual, self.weight, self.variance_epsilon)


class HFNumericsRotaryEmbedding(nn.Module):
    def __init__(self, config: PretrainedConfig) -> None:
        super().__init__()
        rope = config.rope_parameters
        if rope.get("rope_type", "default") != "default":
            raise ValueError(f"hf_numerics supports rope_type 'default' only, got {rope['rope_type']!r}")
        dim = getattr(config, "head_dim", None) or config.hidden_size // config.num_attention_heads
        inv_freq = 1.0 / (rope["rope_theta"] ** (torch.arange(0, dim, 2, dtype=torch.int64).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.mrope_section = list(rope["mrope_section"])
        self.head_size = dim

    def forward(self, positions: torch.Tensor, query: torch.Tensor, key: torch.Tensor):
        return torch.ops.vllm_omni.hf_mrope(positions, query, key, self.inv_freq, self.mrope_section, self.head_size)


class HFNumericsSparseMoeBlock(Qwen3MoeSparseMoeBlock):
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        experts = self.experts.routed_experts
        out = torch.ops.vllm_omni.hf_moe(
            hidden_states.view(-1, hidden_states.shape[-1]),
            self.gate.weight,
            experts.w13_weight,
            experts.w2_weight,
            self.top_k,
            self.norm_topk_prob,
            self.hidden_act,
        )
        return out.view(hidden_states.shape)


class HFNumericsDecoderLayer(Qwen3MoeDecoderLayer):
    def __init__(self, vllm_config: VllmConfig, prefix: str = "", is_fused_checkpoint_transposed: bool = False) -> None:
        super().__init__(
            vllm_config=vllm_config, prefix=prefix, is_fused_checkpoint_transposed=is_fused_checkpoint_transposed
        )
        config = vllm_config.model_config.hf_text_config
        eps = config.rms_norm_eps
        self.input_layernorm = HFNumericsRMSNorm(config.hidden_size, eps=eps)
        self.post_attention_layernorm = HFNumericsRMSNorm(config.hidden_size, eps=eps)
        self.self_attn.q_norm = HFNumericsRMSNorm(self.self_attn.head_dim, eps=eps)
        self.self_attn.k_norm = HFNumericsRMSNorm(self.self_attn.head_dim, eps=eps)
        self.self_attn.rotary_emb = HFNumericsRotaryEmbedding(config)
        if not isinstance(self.mlp, Qwen3MoeSparseMoeBlock):
            raise ValueError("hf_numerics supports MoE decoder layers only")
        if self.mlp.tp_size != 1 or self.mlp.ep_size != 1 or self.mlp.is_sequence_parallel:
            raise ValueError(
                "hf_numerics needs tensor_parallel_size=1, no expert parallelism and no sequence-parallel MoE"
            )
        if self.mlp.shared_expert is not None:
            raise ValueError("hf_numerics does not support shared experts")
        quant_method = self.mlp.experts.routed_experts.quant_method
        if not isinstance(quant_method, UnquantizedFusedMoEMethod):
            raise ValueError(f"hf_numerics needs unquantized MoE weights, got {type(quant_method).__name__}")
        # Other backends repack the expert weights after loading, away from the HF layout the forward reads.
        if quant_method.unquantized_backend != UnquantizedMoeBackend.TRITON:
            raise ValueError(
                f"hf_numerics needs the Triton MoE backend, got {quant_method.unquantized_backend.value}; "
                "set kernel_config.moe_backend='triton' (--moe-backend triton)"
            )
        native_grouped_mm = (
            current_platform.is_cuda()
            and (current_platform.is_device_capability_family(90) or current_platform.is_device_capability_family(100))
            and vllm_config.model_config.dtype == torch.bfloat16
        )
        if vllm_config.compilation_config.cudagraph_mode != CUDAGraphMode.NONE and not native_grouped_mm:
            raise ValueError(
                "hf_numerics can run under CUDA graphs only with bf16 on SM90/SM100 GPUs, where torch grouped_mm "
                "has a native kernel; set enforce_eager=True"
            )
        # The stock layer builds the MoE block itself; swapping the class keeps its loaded experts and
        # replaces only the forward.
        self.mlp.__class__ = HFNumericsSparseMoeBlock
        self.mlp.top_k = config.num_experts_per_tok
        self.mlp.norm_topk_prob = config.norm_topk_prob
        self.mlp.hidden_act = config.hidden_act
