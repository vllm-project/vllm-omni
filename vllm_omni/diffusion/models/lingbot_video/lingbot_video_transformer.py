# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Adapted from LingBot-Video (https://github.com/Robbyant/lingbot-video).

import math
from collections.abc import Iterable
from types import SimpleNamespace

import torch
import torch.nn as nn
import torch.nn.functional as F
from cache_dit import ForwardPattern
from diffusers.configuration_utils import ConfigMixin
from diffusers.models.embeddings import TimestepEmbedding, Timesteps
from diffusers.models.modeling_outputs import Transformer2DModelOutput
from vllm.config import get_current_vllm_config
from vllm.forward_context import set_forward_context as set_vllm_forward_context
from vllm.model_executor.layers.fused_moe.router.gate_linear import GateLinear
from vllm.model_executor.model_loader.weight_utils import default_weight_loader

from vllm_omni.diffusion.attention.backends.abstract import AttentionMetadata
from vllm_omni.diffusion.attention.layer import Attention
from vllm_omni.diffusion.cache.cachedit import CacheDiTAdapterConfig
from vllm_omni.diffusion.distributed.sp_plan import (
    SequenceParallelInput,
    SequenceParallelOutput,
)
from vllm_omni.diffusion.forward_context import (
    get_forward_context,
    is_forward_context_available,
)
from vllm_omni.diffusion.layers.fused_moe import FusedMoE

LINGBOT_VIDEO_FP32_MODULES = (
    "time_embedder",
    "time_modulation",
    "scale_shift_table",
    "norm",
    "norm1",
    "norm2",
    "norm_q",
    "norm_k",
    "norm_post_attn",
    "norm_post_ffn",
    "norm_out",
    "norm_out_modulation",
    "router",
    "gate",
    "e_score_correction_bias",
)


def should_keep_in_fp32(name: str) -> bool:
    return any(module_name in name.split(".") for module_name in LINGBOT_VIDEO_FP32_MODULES)


class LingBotVideoRMSNorm(nn.Module):
    """RMSNorm with fp32 accumulation."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return (self.weight * hidden_states).to(input_dtype)


def apply_rotary_emb(x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
    """Apply complex RoPE to `(B, S, H, D)` attention tensors."""
    with torch.amp.autocast("cuda", enabled=False):
        x_c = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
        out = torch.view_as_real(x_c * freqs_cis.unsqueeze(2)).flatten(3)
        return out.type_as(x)


class LingBotVideoRotaryEmbedding(nn.Module):
    """Complex64 RoPE table indexed by position ids."""

    def __init__(self, axes_dims: tuple[int, ...], axes_lens: tuple[int, ...], theta: float):
        super().__init__()
        self.axes_dims = tuple(axes_dims)
        self.axes_lens = tuple(axes_lens)
        self.theta = theta
        for axis, freqs_cis in enumerate(self.precompute_freqs_cis(self.axes_dims, self.axes_lens, theta=self.theta)):
            self.register_buffer(f"freqs_cis_{axis}", freqs_cis, persistent=False)

    @staticmethod
    def precompute_freqs_cis(dim: tuple[int, ...], end: tuple[int, ...], theta: float):
        freqs_cis = []
        for d, e in zip(dim, end):
            freqs = 1.0 / (theta ** (torch.arange(0, d, 2, dtype=torch.float64, device="cpu") / d))
            timestep = torch.arange(e, device=freqs.device, dtype=torch.float64)
            freqs = torch.outer(timestep, freqs).float()
            freqs_cis.append(torch.polar(torch.ones_like(freqs), freqs).to(torch.complex64))
        return freqs_cis

    def forward(self, position_ids: torch.Tensor) -> torch.Tensor:
        # position_ids: (..., S, 3) int -> (..., S, head_dim/2) complex64
        axis_embeddings = []
        for axis in range(len(self.axes_dims)):
            buffer_name = f"freqs_cis_{axis}"
            freqs_cis = getattr(self, buffer_name)
            if freqs_cis.device != position_ids.device:
                freqs_cis = freqs_cis.to(position_ids.device)
                setattr(self, buffer_name, freqs_cis)
            axis_embeddings.append(freqs_cis[position_ids[..., axis]])
        return torch.cat(axis_embeddings, dim=-1)


def make_joint_position_ids(text_len: int, grid_t: int, grid_h: int, grid_w: int, device: torch.device) -> torch.Tensor:
    """Return ``(t, h, w)`` positions with video rows before text rows.

    Text t-axis positions are 1..text_len, and video t-axis positions start at
    text_len + 1. This matches the token order produced by ``_cat_interleave``.
    """
    tt = torch.arange(grid_t, device=device, dtype=torch.int32) + (text_len + 1)
    hh = torch.arange(grid_h, device=device, dtype=torch.int32)
    ww = torch.arange(grid_w, device=device, dtype=torch.int32)
    grid = torch.stack(torch.meshgrid(tt, hh, ww, indexing="ij"), dim=-1).flatten(0, 2)
    text_t = torch.arange(text_len, device=device, dtype=torch.int32) + 1
    text_pos = torch.stack([text_t, torch.zeros_like(text_t), torch.zeros_like(text_t)], dim=-1)
    return torch.cat([grid, text_pos], dim=0)  # (Nx + L, 3)


def make_batched_joint_position_ids(
    text_lens: torch.Tensor,
    max_text_len: int,
    grid_t: int,
    grid_h: int,
    grid_w: int,
) -> torch.Tensor:
    """Build per-sample ``[video; text]`` positions without host syncs."""
    device = text_lens.device
    tt = torch.arange(grid_t, device=device, dtype=torch.int32)
    hh = torch.arange(grid_h, device=device, dtype=torch.int32)
    ww = torch.arange(grid_w, device=device, dtype=torch.int32)
    video_pos = torch.stack(torch.meshgrid(tt, hh, ww, indexing="ij"), dim=-1).flatten(0, 2)
    video_pos = video_pos.unsqueeze(0).expand(text_lens.shape[0], -1, -1).clone()
    video_pos[..., 0] += text_lens.to(torch.int32).unsqueeze(1) + 1

    text_t = torch.arange(max_text_len, device=device, dtype=torch.int32) + 1
    text_pos = torch.stack(
        [text_t, torch.zeros_like(text_t), torch.zeros_like(text_t)],
        dim=-1,
    )
    text_pos = text_pos.unsqueeze(0).expand(text_lens.shape[0], -1, -1)
    return torch.cat([video_pos, text_pos], dim=1)


def _packed_block_attention_mask(
    packed_sample_ids: torch.Tensor,
    *,
    total_seq_len: int | None = None,
) -> torch.Tensor:
    unpadded_seq_len = packed_sample_ids.numel()
    total_seq_len = total_seq_len or unpadded_seq_len
    if total_seq_len < unpadded_seq_len:
        raise ValueError("Packed attention mask cannot truncate sample tokens.")
    mask = packed_sample_ids.unsqueeze(0) == packed_sample_ids.unsqueeze(1)
    if total_seq_len > unpadded_seq_len:
        mask = F.pad(mask, (0, total_seq_len - unpadded_seq_len, 0, total_seq_len - unpadded_seq_len))
    return mask.unsqueeze(0).unsqueeze(0)


class LingBotVideoTextEmbedder(nn.Module):
    """Matches CondProjection: RMSNorm(text_dim, eps=1e-6 fixed) -> Linear-SiLU-Linear."""

    def __init__(self, text_dim: int, hidden_size: int):
        super().__init__()
        self.norm = LingBotVideoRMSNorm(text_dim, eps=1e-6)
        self.linear_1 = nn.Linear(text_dim, hidden_size, bias=True)
        self.linear_2 = nn.Linear(hidden_size, hidden_size, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.norm(x)
        return self.linear_2(F.silu(self.linear_1(x)))


class LingBotVideoAttention(nn.Module):
    def __init__(self, hidden_size, num_heads, norm_eps, qkv_bias, out_bias):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.to_q = nn.Linear(hidden_size, hidden_size, bias=qkv_bias)
        self.to_k = nn.Linear(hidden_size, hidden_size, bias=qkv_bias)
        self.to_v = nn.Linear(hidden_size, hidden_size, bias=qkv_bias)
        self.norm_q = LingBotVideoRMSNorm(self.head_dim, norm_eps)
        self.norm_k = LingBotVideoRMSNorm(self.head_dim, norm_eps)
        self.to_out = nn.Linear(hidden_size, hidden_size, bias=out_bias)
        self.attn = Attention(
            num_heads=num_heads,
            head_size=self.head_dim,
            softmax_scale=1.0 / math.sqrt(self.head_dim),
            causal=False,
            num_kv_heads=num_heads,
            role="self",
        )

    def forward(
        self,
        x,
        rotary_emb,
        attention_mask=None,
    ):
        q = self.to_q(x).unflatten(2, (self.num_heads, self.head_dim))
        k = self.to_k(x).unflatten(2, (self.num_heads, self.head_dim))
        v = self.to_v(x).unflatten(2, (self.num_heads, self.head_dim))
        q = apply_rotary_emb(self.norm_q(q), rotary_emb)
        k = apply_rotary_emb(self.norm_k(k), rotary_emb)
        metadata = AttentionMetadata(attn_mask=attention_mask)
        if attention_mask is not None and attention_mask.ndim > 2:
            # Packed CFG uses a block-diagonal QK mask. Let the common Attention
            # layer perform SP resharding, then select its exact SDPA fallback.
            metadata.extra["force_sdpa"] = True
        out = self.attn(q, k, v, attn_metadata=metadata)
        return self.to_out(out.flatten(2, 3).type_as(x))


class LingBotVideoMLP(nn.Module):
    def __init__(self, hidden_size, intermediate_size):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class LingBotVideoSparseMoeBlock(nn.Module):
    """LingBot routing semantics backed by vLLM's common FusedMoE runner."""

    def __init__(
        self,
        hidden_size: int,
        num_experts: int,
        top_k: int,
        moe_intermediate_size: int,
        score_func: str,
        norm_topk_prob: bool,
        n_group: int | None,
        topk_group: int | None,
        routed_scaling_factor: float,
        n_shared_experts: int | None,
        *,
        prefix: str,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_experts = num_experts
        gate = GateLinear(
            hidden_size,
            num_experts,
            bias=False,
            out_dtype=torch.float32,
            params_dtype=torch.float32,
            force_fp32_compute=True,
            prefix=f"{prefix}.gate",
        )
        correction_bias = nn.Parameter(
            torch.zeros(num_experts, dtype=torch.float32),
            requires_grad=False,
        )
        self.experts = FusedMoE(
            num_experts=num_experts,
            top_k=top_k,
            hidden_size=hidden_size,
            intermediate_size=moe_intermediate_size,
            renormalize=norm_topk_prob,
            use_grouped_topk=n_group is not None and n_group > 1,
            num_expert_group=n_group,
            topk_group=topk_group,
            scoring_func=score_func,
            routed_scaling_factor=routed_scaling_factor,
            e_score_correction_bias=correction_bias,
            activation="silu",
            gate=gate,
            ckpt_names=("w1", "w2", "w3"),
            tp_size=1,
            dp_size=1,
            pcp_size=1,
            prefix=prefix,
        )
        self.shared_experts = None
        if n_shared_experts is not None and n_shared_experts > 0:
            self.shared_experts = LingBotVideoMLP(
                hidden_size,
                moe_intermediate_size * n_shared_experts,
            )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch_size = hidden_states.shape[0]
        tokens = hidden_states.reshape(-1, self.hidden_size)
        out = self._run_routed_experts(tokens).reshape(
            batch_size,
            -1,
            self.hidden_size,
        )
        if self.shared_experts is not None:
            out = out + self.shared_experts(hidden_states)
        return out

    @torch.compiler.disable
    def _run_routed_experts(self, tokens: torch.Tensor) -> torch.Tensor:
        """Keep vLLM's opaque MoE runner outside the Inductor graph."""
        # The runner owns the FP32 gate and overwrites this placeholder inside
        # its opaque custom op before routing.
        router_logits = tokens.new_empty(0)
        with set_vllm_forward_context(
            attn_metadata=None,
            vllm_config=get_current_vllm_config(),
            num_tokens=tokens.shape[0],
        ):
            return self.experts(
                hidden_states=tokens,
                router_logits=router_logits,
            )


class LingBotVideoBlock(nn.Module):
    def __init__(
        self,
        hidden_size,
        num_attention_heads,
        intermediate_size,
        norm_eps,
        qkv_bias,
        out_bias,
        num_experts,
        num_experts_per_tok,
        moe_intermediate_size,
        decoder_sparse_step,
        mlp_only_layers,
        n_shared_experts,
        score_func,
        norm_topk_prob,
        n_group,
        topk_group,
        routed_scaling_factor,
        layer_idx: int,
        *,
        prefix: str,
    ):
        super().__init__()
        self.layer_idx = layer_idx
        h = hidden_size
        self.scale_shift_table = nn.Parameter(torch.zeros(1, 6 * h))
        self.norm1 = LingBotVideoRMSNorm(h, norm_eps)
        self.attn = LingBotVideoAttention(h, num_attention_heads, norm_eps, qkv_bias, out_bias)
        self.norm_post_attn = LingBotVideoRMSNorm(h, norm_eps)
        self.norm2 = LingBotVideoRMSNorm(h, norm_eps)
        use_sparse_moe = (
            num_experts > 0 and layer_idx not in mlp_only_layers and (layer_idx + 1) % decoder_sparse_step == 0
        )
        if use_sparse_moe:
            self.ffn = LingBotVideoSparseMoeBlock(
                hidden_size=h,
                num_experts=num_experts,
                top_k=num_experts_per_tok,
                moe_intermediate_size=moe_intermediate_size,
                score_func=score_func,
                norm_topk_prob=norm_topk_prob,
                n_group=n_group,
                topk_group=topk_group,
                routed_scaling_factor=routed_scaling_factor,
                n_shared_experts=n_shared_experts,
                prefix=f"{prefix}.ffn.experts",
            )
        else:
            self.ffn = LingBotVideoMLP(h, intermediate_size)
        self.norm_post_ffn = LingBotVideoRMSNorm(h, norm_eps)

    def forward(
        self,
        hidden_states,
        temb6,
        rotary_emb,
        attention_mask=None,
    ):
        expected_tokens = hidden_states.shape[0] * hidden_states.shape[1]
        if temb6.ndim != 2 or temb6.shape[0] != expected_tokens:
            raise ValueError(
                "LingBotVideoBlock expects token-level temb6 with shape "
                f"(B*S, 6D); got {tuple(temb6.shape)} for hidden states {tuple(hidden_states.shape)}."
            )
        # AdaLN modulation and normalization stay in fp32 for the sensitive path.
        mod = temb6.view(hidden_states.shape[0], hidden_states.shape[1], -1) + self.scale_shift_table.unsqueeze(0)
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = mod.chunk(6, dim=-1)
        gate_msa, gate_mlp = gate_msa.tanh(), gate_mlp.tanh()
        scale_msa, scale_mlp = 1.0 + scale_msa, 1.0 + scale_mlp

        # AdaLN modulation and norms stay in fp32; cast to the transformer
        # compute dtype only at Linear boundaries.
        bulk_dtype = self.attn.to_q.weight.dtype
        attn_in = (self.norm1(hidden_states) * scale_msa + shift_msa).to(bulk_dtype)
        attn_out = self.attn(attn_in, rotary_emb, attention_mask)
        hidden_states = hidden_states + (gate_msa * self.norm_post_attn(attn_out)).to(hidden_states.dtype)

        ffn_in = (self.norm2(hidden_states) * scale_mlp + shift_mlp).to(bulk_dtype)
        ffn_out = self.ffn(ffn_in)
        ffn_normed = self.norm_post_ffn(ffn_out)
        hidden_states = hidden_states + (gate_mlp * ffn_normed).to(hidden_states.dtype)
        return hidden_states


class _LingBotSPInputBoundary(nn.Module):
    """Keep all token-aligned tensors on one standard SP split boundary."""

    def forward(
        self,
        joint: torch.Tensor,
        rotary: torch.Tensor,
        temb_input: torch.Tensor,
        temb6: torch.Tensor,
        token_validity: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if is_forward_context_available():
            ctx = get_forward_context()
            ctx.sp_original_seq_len = None
            ctx.sp_padding_size = 0
        return joint, rotary, temb_input, temb6, token_validity


class _LingBotSPOutputBoundary(nn.Module):
    """Gather projected joint tokens and remove any SP auto-padding."""

    def forward(self, projected: torch.Tensor) -> torch.Tensor:
        return projected


class _LingBotVideoConfigLoader(ConfigMixin):
    config_name = "config.json"


class LingBotVideoTransformer3DModel(nn.Module):
    _supports_gradient_checkpointing = False
    _cache_dit_adapter_config = CacheDiTAdapterConfig(
        block_forward_patterns={"blocks": ForwardPattern.Pattern_3},
        has_separate_cfg=True,
    )
    _repeated_blocks = ["LingBotVideoBlock"]
    _layerwise_offload_blocks_attrs = ["blocks"]
    _no_split_modules = ["LingBotVideoBlock"]
    _keep_in_fp32_modules = list(LINGBOT_VIDEO_FP32_MODULES)

    @staticmethod
    def _is_transformer_block(name: str, module: nn.Module) -> bool:
        return (
            isinstance(module, LingBotVideoBlock)
            and name.startswith("blocks.")
            and name.removeprefix("blocks.").isdigit()
        )

    _hsdp_shard_conditions = [_is_transformer_block]
    # AdaLN, normalization, and MoE router parameters intentionally remain
    # FP32 while the bulk weights are BF16. Preserve those loaded dtypes when
    # FSDP all-gathers each block instead of applying one dtype to all params.
    _hsdp_preserve_param_dtype = True

    # Auto-padding is exact for pure Ulysses. Ring/hybrid remain limited to
    # divisible, unmasked sequences because Ring attention cannot consume masks.
    _sp_plan = {
        "sp_input_boundary": {
            0: SequenceParallelInput(split_dim=1, expected_dims=3, split_output=True, auto_pad=True),
            1: SequenceParallelInput(split_dim=1, expected_dims=3, split_output=True, auto_pad=True),
            2: SequenceParallelInput(split_dim=1, expected_dims=3, split_output=True, auto_pad=True),
            3: SequenceParallelInput(split_dim=1, expected_dims=3, split_output=True, auto_pad=True),
            4: SequenceParallelInput(split_dim=1, expected_dims=2, split_output=True, auto_pad=True),
        },
        "sp_output_boundary": SequenceParallelOutput(gather_dim=1, expected_dims=3),
    }

    def to(self, *args, **kwargs):
        device, dtype, non_blocking, _ = torch._C._nn._parse_to(*args, **kwargs)
        if dtype is None or dtype == torch.float32:
            return super().to(*args, **kwargs)

        dtype_is_floating = torch.is_floating_point(torch.empty((), dtype=dtype))
        if not dtype_is_floating:
            return super().to(*args, **kwargs)

        if device is not None:
            super().to(device=device, non_blocking=non_blocking)

        for name, param in self.named_parameters():
            if not torch.is_floating_point(param):
                continue
            target_dtype = torch.float32 if should_keep_in_fp32(name) else dtype
            param.data = param.data.to(dtype=target_dtype, non_blocking=non_blocking)
            if param.grad is not None:
                param.grad.data = param.grad.data.to(dtype=target_dtype, non_blocking=non_blocking)

        for name, buffer in self.named_buffers():
            if not torch.is_floating_point(buffer):
                continue
            target_dtype = torch.float32 if should_keep_in_fp32(name) else dtype
            buffer.data = buffer.data.to(dtype=target_dtype, non_blocking=non_blocking)

        return self

    def __init__(
        self,
        patch_size: tuple[int, int, int] = (1, 2, 2),
        in_channels: int = 16,
        out_channels: int = 16,
        hidden_size: int = 2048,
        num_attention_heads: int = 16,
        depth: int = 24,
        intermediate_size: int = 6144,
        text_dim: int = 2560,
        freq_dim: int = 256,
        norm_eps: float = 1e-6,
        rope_theta: float = 256.0,
        axes_dims: tuple[int, int, int] = (32, 48, 48),
        axes_lens: tuple[int, int, int] = (8192, 1024, 1024),
        qkv_bias: bool = False,
        out_bias: bool = True,
        patch_embed_bias: bool = True,
        timestep_mlp_bias: bool = True,
        num_experts: int = 0,
        num_experts_per_tok: int = 8,
        moe_intermediate_size: int = 512,
        decoder_sparse_step: int = 1,
        mlp_only_layers: tuple[int, ...] = (),
        n_shared_experts: int | None = None,
        score_func: str = "sigmoid",
        norm_topk_prob: bool = True,
        n_group: int | None = None,
        topk_group: int | None = None,
        routed_scaling_factor: float = 1.0,
        prefix: str = "lingbot_video",
    ):
        super().__init__()
        head_dim = hidden_size // num_attention_heads
        assert head_dim == sum(axes_dims), f"head_dim {head_dim} != sum(axes_dims) {sum(axes_dims)}"
        mlp_only_layers = tuple(mlp_only_layers)
        self.config = SimpleNamespace(
            patch_size=tuple(patch_size),
            in_channels=in_channels,
            out_channels=out_channels,
            hidden_size=hidden_size,
            num_attention_heads=num_attention_heads,
            depth=depth,
            intermediate_size=intermediate_size,
            text_dim=text_dim,
            freq_dim=freq_dim,
            norm_eps=norm_eps,
            rope_theta=rope_theta,
            axes_dims=tuple(axes_dims),
            axes_lens=tuple(axes_lens),
            qkv_bias=qkv_bias,
            out_bias=out_bias,
            patch_embed_bias=patch_embed_bias,
            timestep_mlp_bias=timestep_mlp_bias,
            num_experts=num_experts,
            num_experts_per_tok=num_experts_per_tok,
            moe_intermediate_size=moe_intermediate_size,
            decoder_sparse_step=decoder_sparse_step,
            mlp_only_layers=mlp_only_layers,
            n_shared_experts=n_shared_experts,
            score_func=score_func,
            norm_topk_prob=norm_topk_prob,
            n_group=n_group,
            topk_group=topk_group,
            routed_scaling_factor=routed_scaling_factor,
        )

        self.patch_embedder = nn.Linear(in_channels * math.prod(patch_size), hidden_size, bias=patch_embed_bias)
        self.time_proj = Timesteps(freq_dim, flip_sin_to_cos=True, downscale_freq_shift=0)
        self.time_embedder = TimestepEmbedding(freq_dim, hidden_size, act_fn="silu", sample_proj_bias=timestep_mlp_bias)
        self.time_modulation = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, 6 * hidden_size))
        self.text_embedder = LingBotVideoTextEmbedder(text_dim, hidden_size)
        self.rope = LingBotVideoRotaryEmbedding(axes_dims, axes_lens, rope_theta)
        self.blocks = nn.ModuleList(
            [
                LingBotVideoBlock(
                    hidden_size=hidden_size,
                    num_attention_heads=num_attention_heads,
                    intermediate_size=intermediate_size,
                    norm_eps=norm_eps,
                    qkv_bias=qkv_bias,
                    out_bias=out_bias,
                    num_experts=num_experts,
                    num_experts_per_tok=num_experts_per_tok,
                    moe_intermediate_size=moe_intermediate_size,
                    decoder_sparse_step=decoder_sparse_step,
                    mlp_only_layers=mlp_only_layers,
                    n_shared_experts=n_shared_experts,
                    score_func=score_func,
                    norm_topk_prob=norm_topk_prob,
                    n_group=n_group,
                    topk_group=topk_group,
                    routed_scaling_factor=routed_scaling_factor,
                    layer_idx=i,
                    prefix=f"{prefix}.blocks.{i}",
                )
                for i in range(depth)
            ]
        )
        self.norm_out = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=norm_eps)
        self.norm_out_modulation = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, 2 * hidden_size))
        self.proj_out = nn.Linear(hidden_size, math.prod(patch_size) * out_channels)
        self.sp_input_boundary = _LingBotSPInputBoundary()
        self.sp_output_boundary = _LingBotSPOutputBoundary()

    @classmethod
    def load_config(cls, *args, **kwargs):
        """Load a diffusers-style config without inheriting its model mixins."""
        return _LingBotVideoConfigLoader.load_config(*args, **kwargs)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        """Load LingBot checkpoints and pack aggregate ``w1/w3`` into ``w13``."""
        params = dict(self.named_parameters())
        tensors = {**dict(self.named_buffers()), **params}
        loaded: set[str] = set()

        for name, weight in weights:
            if name.endswith((".ffn.experts.w1", ".ffn.experts.w2", ".ffn.experts.w3")):
                source_prefix, shard_id = name.rsplit(".", 1)
                target_suffix = "w2_weight" if shard_id == "w2" else "w13_weight"
                target_name = f"{source_prefix}.routed_experts.{target_suffix}"
                param = params[target_name]
                weight_loader = getattr(param, "weight_loader")
                for expert_id, expert_weight in enumerate(weight):
                    weight_loader(
                        param,
                        expert_weight,
                        target_name,
                        shard_id,
                        expert_id,
                    )
                loaded.add(target_name)
                continue

            if name.endswith(".ffn.router.weight"):
                target_name = name.replace(".ffn.router.weight", ".ffn.experts.gate.weight")
            elif name.endswith(".ffn.router.e_score_correction_bias"):
                target_name = name.replace(
                    ".ffn.router.e_score_correction_bias",
                    ".ffn.experts.routed_experts.e_score_correction_bias",
                )
            else:
                target_name = name

            target = tensors.get(target_name)
            if target is None:
                raise KeyError(f"LingBot checkpoint tensor {name!r} does not map to a model parameter or buffer.")
            weight_loader = getattr(target, "weight_loader", default_weight_loader)
            weight_loader(target, weight)
            loaded.add(target_name)

        return loaded

    def forward(
        self,
        hidden_states: torch.Tensor,  # (B, C, T, H, W)
        timestep: torch.Tensor,  # (B,) in [0, 1000](= sigma*1000)
        encoder_hidden_states: torch.Tensor,  # (B, L, text_dim)
        encoder_attention_mask: torch.Tensor | None = None,  # (B, L) 1=valid
        return_dict: bool = True,
    ):
        B, C, T, H, W = hidden_states.shape
        pF, pH, pW = self.config.patch_size
        gt, gh, gw = T // pF, H // pH, W // pW
        n_video = gt * gh * gw
        L = encoder_hidden_states.shape[1]
        device = hidden_states.device
        if encoder_attention_mask is not None:
            text_validity = encoder_attention_mask.bool()
            text_lens = text_validity.sum(dim=-1).long()
        else:
            text_validity = torch.ones(B, L, dtype=torch.bool, device=device)
            text_lens = torch.full((B,), L, dtype=torch.long, device=device)
        packed_batch = B > 1

        # patchify: token order (f h w), feature order (pf ph pw c) -- matches patchify_and_embed
        patch_tokens = hidden_states.reshape(B, C, gt, pF, gh, pH, gw, pW)
        patch_tokens = patch_tokens.permute(0, 2, 4, 6, 3, 5, 7, 1).reshape(
            B,
            n_video,
            pF * pH * pW * C,
        )
        video = self.patch_embedder(patch_tokens)
        text = self.text_embedder(encoder_hidden_states)
        sample_joint = torch.cat([video, text], dim=1)
        video_validity = torch.ones(B, n_video, dtype=torch.bool, device=device)
        sample_token_validity = torch.cat([video_validity, text_validity], dim=1)
        sample_video_selector = torch.cat(
            [video_validity, torch.zeros_like(text_validity)],
            dim=1,
        )
        sample_ids = torch.arange(B, device=device).unsqueeze(1).expand_as(sample_token_validity)
        sample_positions = make_batched_joint_position_ids(text_lens, L, gt, gh, gw)

        timestep_for_embed = timestep.float()
        timestep_proj = self.time_proj(timestep_for_embed)
        t_emb = self.time_embedder(timestep_proj)  # (B, D)
        sample_temb = t_emb.unsqueeze(1).expand(B, n_video + L, -1)

        if packed_batch:
            joint = sample_joint[sample_token_validity].unsqueeze(0)
            rotary = self.rope(sample_positions[sample_token_validity]).unsqueeze(0)
            temb_input = sample_temb[sample_token_validity].unsqueeze(0)
            packed_sample_ids = sample_ids[sample_token_validity]
            packed_video_selector = sample_video_selector[sample_token_validity]
            global_token_validity = torch.ones(
                1,
                joint.shape[1],
                dtype=torch.bool,
                device=device,
            )
        else:
            joint = sample_joint
            rotary = self.rope(sample_positions)
            temb_input = sample_temb
            packed_sample_ids = None
            packed_video_selector = None
            global_token_validity = sample_token_validity

        joint_seq_len = joint.shape[1]
        temb6 = self.time_modulation(temb_input.reshape(-1, temb_input.shape[-1]))
        temb6 = temb6.reshape(joint.shape[0], joint_seq_len, -1)

        joint, rotary, temb_input, temb6, _local_token_validity = self.sp_input_boundary(
            joint, rotary, temb_input, temb6, global_token_validity
        )

        ctx = get_forward_context() if is_forward_context_available() else None
        padding_size = ctx.sp_padding_size if ctx is not None else 0
        parallel_config = (
            getattr(ctx.omni_diffusion_config, "parallel_config", None)
            if ctx is not None and ctx.omni_diffusion_config is not None
            else None
        )
        ring_degree = int(getattr(parallel_config, "ring_degree", 1))

        padded_global_token_validity = (
            F.pad(global_token_validity, (0, padding_size), value=False) if padding_size else global_token_validity
        )
        if packed_batch:
            assert packed_sample_ids is not None
            attention_mask = _packed_block_attention_mask(
                packed_sample_ids,
                total_seq_len=padded_global_token_validity.shape[1],
            )
        else:
            attention_mask = padded_global_token_validity
            if encoder_attention_mask is None and padding_size == 0:
                attention_mask = None

        if ring_degree > 1 and attention_mask is not None:
            raise ValueError("LingBot Ring SP requires divisible sequences without text padding or packed CFG.")
        temb6 = temb6.reshape(temb6.shape[0] * temb6.shape[1], -1)

        for block in self.blocks:
            joint = block(
                joint,
                temb6,
                rotary,
                attention_mask,
            )
        final_mod = self.norm_out_modulation(temb_input.reshape(joint.shape[0] * joint.shape[1], -1))
        shift, scale = final_mod.reshape(joint.shape[0], joint.shape[1], -1).chunk(2, dim=-1)
        final_hidden = self.norm_out(joint) * (1.0 + scale) + shift
        projected = self.proj_out(final_hidden.to(self.proj_out.weight.dtype))
        projected = self.sp_output_boundary(projected)
        if packed_batch:
            assert packed_video_selector is not None
            x = projected[:, packed_video_selector].reshape(B, n_video, -1)
        else:
            x = projected[:, :n_video]

        # unpatchify (matches the rearrange in postprocess)
        Cout = self.config.out_channels
        x = x.reshape(B, gt, gh, gw, pF, pH, pW, Cout)
        x = x.permute(0, 7, 1, 4, 2, 5, 3, 6).reshape(B, Cout, T, H, W)

        if not return_dict:
            return (x,)
        return Transformer2DModelOutput(sample=x)
