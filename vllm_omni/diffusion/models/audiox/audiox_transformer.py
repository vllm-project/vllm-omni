from __future__ import annotations

import logging
from collections.abc import Iterable
from typing import Any

import torch
import torch.nn as nn
from einops import rearrange
from torch.nn import functional as F
from vllm.distributed import (
    get_tensor_model_parallel_world_size,
    tensor_model_parallel_all_gather,
    tensor_model_parallel_all_reduce,
)
from vllm.model_executor.layers.linear import (
    ColumnParallelLinear,
    MergedColumnParallelLinear,
    QKVParallelLinear,
)
from vllm.model_executor.model_loader.weight_utils import default_weight_loader, sharded_weight_loader

from vllm_omni.diffusion.attention.layer import Attention
from vllm_omni.diffusion.layers.fourier import GaussianFourierProjection
from vllm_omni.diffusion.layers.rope import RotaryEmbedding


class AudioXCrossAttention(nn.Module):
    def __init__(self, dim: int, nheads: int, prefix: str = ""):
        super().__init__()
        self.dim = dim
        self.nheads = nheads
        head_dim = dim // nheads
        self.head_dim = head_dim

        # to_kv bundle weights arrive in (head, dim, VK-index) interleaved order; pipeline's
        # load_weights restacks them to [V|K] before MergedColumnParallelLinear consumes them.
        self.to_q = ColumnParallelLinear(dim, dim, bias=False, gather_output=False, prefix=f"{prefix}.to_q")
        self.to_kv = MergedColumnParallelLinear(
            input_size=dim, output_sizes=[dim, dim], bias=False, gather_output=False, prefix=f"{prefix}.to_kv"
        )
        self.q_norm = AudioXRMSNorm(head_dim)
        self.k_norm = AudioXRMSNorm(head_dim)
        local_nheads = nheads // get_tensor_model_parallel_world_size()
        self.attn = Attention(
            num_heads=local_nheads,
            head_size=head_dim,
            softmax_scale=head_dim**-0.5,
            causal=False,
        )

    def forward(self, x: torch.Tensor, context: torch.Tensor | None = None) -> torch.Tensor:
        tp = get_tensor_model_parallel_world_size()
        local_h = self.nheads // tp
        d = self.head_dim

        # ROCm's ``wvSplitK`` GEMM bypasses autocast and rejects mismatched dtypes,
        # so pre-match the input to the linear's weight dtype.
        weight_dtype = self.to_q.weight.dtype
        q_flat, _ = self.to_q(x.to(weight_dtype))
        q = rearrange(q_flat, "b n (h d) -> b n h d", h=local_h, d=d)
        kv_flat, _ = self.to_kv(context.to(weight_dtype))
        v_flat, k_flat = kv_flat.chunk(2, dim=-1)
        # Upstream `CrossAttention.forward` unpacks `pre_attention()` as `q, v, k = ...` (K/V
        # swapped), AND only normalizes the first chunk of to_kv via k_norm. Trained weights
        # depend on this quirk: first chunk normalized -> V; second chunk unnormalized -> K.
        v = rearrange(v_flat, "b n (h d) -> b n h d", h=local_h, d=d)
        k = rearrange(k_flat, "b n (h d) -> b n h d", h=local_h, d=d)
        q = self.q_norm(q)
        v = self.k_norm(v)

        out = self.attn(q.contiguous(), k.contiguous(), v.contiguous(), attn_metadata=None)
        out = rearrange(out, "b n h d -> b n (h d)").contiguous()
        if tp > 1:
            out = tensor_model_parallel_all_gather(out, dim=-1)
        return out


logger = logging.getLogger(__name__)

__all__ = [
    "AudioXMMChannelLastConv1d",
    "AudioXMMConvFeedForward",
    "AudioXMMDiTSelfAttention",
    "AudioXMMDiTBlock",
    "ContinuousMMDiTTransformer",
    "MMDiffusionTransformer",
]


class AudioXMMChannelLastConv1d(nn.Conv1d):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = rearrange(x, "b n c -> b c n")
        x = super().forward(x)
        x = rearrange(x, "b c n -> b n c")
        return x


class _ColumnParallelChannelLastConv1d(AudioXMMChannelLastConv1d):
    def __init__(self, in_channels: int, out_channels_total: int, **kwargs: Any):
        tp_size = get_tensor_model_parallel_world_size()
        assert out_channels_total % tp_size == 0, (out_channels_total, tp_size)
        super().__init__(in_channels, out_channels_total // tp_size, **kwargs)
        self.weight.weight_loader = sharded_weight_loader(0)


class _RowParallelChannelLastConv1d(AudioXMMChannelLastConv1d):
    def __init__(self, in_channels_total: int, out_channels: int, **kwargs: Any):
        tp_size = get_tensor_model_parallel_world_size()
        assert in_channels_total % tp_size == 0, (in_channels_total, tp_size)
        super().__init__(in_channels_total // tp_size, out_channels, **kwargs)
        self._tp_size = tp_size
        self.weight.weight_loader = sharded_weight_loader(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = super().forward(x)
        if self._tp_size > 1:
            y = tensor_model_parallel_all_reduce(y)
        return y


class AudioXMMConvFeedForward(nn.Module):
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        multiple_of: int = 256,
        kernel_size: int = 3,
        padding: int = 1,
    ):
        super().__init__()
        hidden_dim = int(2 * hidden_dim / 3)
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)

        self.w1 = _ColumnParallelChannelLastConv1d(
            dim, hidden_dim, bias=False, kernel_size=kernel_size, padding=padding
        )
        self.w2 = _RowParallelChannelLastConv1d(hidden_dim, dim, bias=False, kernel_size=kernel_size, padding=padding)
        self.w3 = _ColumnParallelChannelLastConv1d(
            dim, hidden_dim, bias=False, kernel_size=kernel_size, padding=padding
        )

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class AudioXRMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean_sq = torch.mean(x**2, dim=-1, keepdim=True)
        scale = torch.rsqrt(mean_sq + self.eps)
        return x * scale


class AudioXMMDiTSelfAttention(nn.Module):
    def __init__(self, dim: int, nheads: int, prefix: str = ""):
        super().__init__()
        self.dim = dim
        self.nheads = nheads

        head_dim = dim // nheads
        self.head_dim = head_dim
        # Bundle weights arrive in interleaved (h, d, qkv) layout; pipeline's load_weights
        # restacks to [Q|K|V] before QKVParallelLinear's weight_loader consumes them.
        self.qkv = QKVParallelLinear(
            hidden_size=dim,
            head_size=head_dim,
            total_num_heads=nheads,
            bias=True,
            prefix=f"{prefix}.qkv",
        )
        self.q_norm = AudioXRMSNorm(head_dim)
        self.k_norm = AudioXRMSNorm(head_dim)

        self.rope = RotaryEmbedding(is_neox_style=False)
        self.attn = Attention(
            num_heads=self.qkv.num_heads,
            head_size=head_dim,
            softmax_scale=head_dim**-0.5,
            causal=False,
        )

    def apply_attention(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        out = self.attn(q.contiguous(), k.contiguous(), v.contiguous(), attn_metadata=None)
        out = rearrange(out, "b n h d -> b n (h d)").contiguous()
        # Downstream linear1/ffn are replicated and expect full hidden dim.
        if get_tensor_model_parallel_world_size() > 1:
            out = tensor_model_parallel_all_gather(out, dim=-1)
        return out

    def pre_attention(self, x: torch.Tensor, rot: tuple[torch.Tensor, torch.Tensor] | None = None):
        # ROCm GEMM dtype cast — see AudioXCrossAttention.forward.
        qkv, _ = self.qkv(x.to(self.qkv.weight.dtype))
        local_h = self.qkv.num_heads
        d = self.head_dim
        q_size = local_h * d
        q, k, v = qkv.split([q_size, q_size, q_size], dim=-1)
        q = rearrange(q, "b n (h d) -> b n h d", h=local_h, d=d)
        k = rearrange(k, "b n (h d) -> b n h d", h=local_h, d=d)
        v = rearrange(v, "b n (h d) -> b n h d", h=local_h, d=d)
        q = self.q_norm(q)
        k = self.k_norm(k)

        if rot is not None:
            cos, sin = rot
            cos = cos.to(dtype=q.dtype)
            sin = sin.to(dtype=q.dtype)
            q = self.rope(q, cos, sin)
            k = self.rope(k, cos, sin)

        return q, k, v

    def forward(
        self,
        x: torch.Tensor,
        rot: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        q, k, v = self.pre_attention(x, rot=rot)
        return self.apply_attention(q, k, v)


class AudioXMMDiTBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        nhead: int,
        mlp_ratio: float = 4.0,
        prefix: str = "",
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False)
        self.attn = AudioXMMDiTSelfAttention(dim, nhead, prefix=f"{prefix}.attn")
        self.cross_attn = AudioXCrossAttention(dim, nhead, prefix=f"{prefix}.cross_attn")
        self.linear1 = AudioXMMChannelLastConv1d(dim, dim, kernel_size=3, padding=1)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False)
        self.ffn = AudioXMMConvFeedForward(dim, int(dim * mlp_ratio), kernel_size=3, padding=1)
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(dim, 6 * dim, bias=True))

    def pre_attention(self, x: torch.Tensor, c: torch.Tensor, rot: tuple[torch.Tensor, torch.Tensor] | None):
        modulation = self.adaLN_modulation(c)
        (shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp) = modulation.chunk(6, dim=-1)
        x = self.norm1(x) * (1 + scale_msa) + shift_msa
        q, k, v = self.attn.pre_attention(x, rot)
        return (q, k, v), (gate_msa, shift_mlp, scale_mlp, gate_mlp)

    def post_attention(self, x: torch.Tensor, attn_out: torch.Tensor, c: tuple[torch.Tensor, ...], context=None):
        (gate_msa, shift_mlp, scale_mlp, gate_mlp) = c
        x = x + self.linear1(attn_out) * gate_msa

        x = x + self.cross_attn(x, context=context)

        r = self.norm2(x) * (1 + scale_mlp) + shift_mlp
        x = x + self.ffn(r) * gate_mlp
        return x

    def forward(
        self,
        x: torch.Tensor,
        cond: torch.Tensor,
        rot: tuple[torch.Tensor, torch.Tensor] | None,
        context: torch.Tensor = None,
    ) -> torch.Tensor:
        x_qkv, x_conditions = self.pre_attention(x, cond, rot)
        attn_out = self.attn.apply_attention(*x_qkv)
        x = self.post_attention(x, attn_out, x_conditions, context=context)
        return x


class MMDiffusionTransformer(nn.Module):
    """AudioX MMDiT, specialized for the published bundle (`zhangj1an/AudioX`).

    The bundle fixes patch_size=1, transformer_type="continuous_transformer",
    cond_token_dim=768 (>0, project_cond_tokens=False), and never sets
    prepend_cond_dim or input_concat_dim, so those code paths are removed.
    """

    def __init__(
        self,
        io_channels: int,
        embed_dim: int,
        cond_token_dim: int,
        global_cond_dim: int,
        depth: int,
        num_heads: int,
        project_cond_tokens: bool = False,
        project_global_cond: bool = True,
        **kwargs,
    ):
        super().__init__()
        if kwargs:
            logger.debug("MMDiffusionTransformer ignoring unused config keys: %s", sorted(kwargs.keys()))
        if project_cond_tokens:
            raise ValueError("AudioX bundle requires project_cond_tokens=False to match official checkpoints.")

        self.cond_token_dim = cond_token_dim

        timestep_features_dim = 256
        self.timestep_features = GaussianFourierProjection(
            in_features=1,
            embedding_size=timestep_features_dim // 2,
            scale=1.0,
            trainable=False,
        )
        self.to_timestep_embed = nn.Sequential(
            nn.Linear(timestep_features_dim, embed_dim, bias=True),
            nn.SiLU(),
            nn.Linear(embed_dim, embed_dim, bias=True),
        )

        cond_embed_dim = cond_token_dim if not project_cond_tokens else embed_dim
        self.to_cond_embed = nn.Sequential(
            nn.Linear(cond_token_dim, cond_embed_dim, bias=False),
            nn.SiLU(),
            nn.Linear(cond_embed_dim, cond_embed_dim, bias=False),
        )

        # ``to_global_embed`` weights live in the bundle but global conditioning is always None
        # at inference; kept so AutoWeightsLoader has a slot to load them into.
        global_embed_dim = global_cond_dim if not project_global_cond else embed_dim
        self.to_global_embed = nn.Sequential(
            nn.Linear(global_cond_dim, global_embed_dim, bias=False),
            nn.SiLU(),
            nn.Linear(global_embed_dim, global_embed_dim, bias=False),
        )

        self.transformer = ContinuousMMDiTTransformer(
            dim=embed_dim,
            depth=depth,
            dim_heads=embed_dim // num_heads,
            dim_in=io_channels,
            dim_out=io_channels,
        )

        self.preprocess_conv = nn.Conv1d(io_channels, io_channels, 1, bias=False)
        nn.init.zeros_(self.preprocess_conv.weight)
        self.postprocess_conv = nn.Conv1d(io_channels, io_channels, 1, bias=False)
        nn.init.zeros_(self.postprocess_conv.weight)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        params_dict = dict(self.named_parameters())
        loaded_params: set[str] = set()
        for name, loaded_weight in weights:
            if name not in params_dict:
                continue
            param = params_dict[name]
            weight_loader = getattr(param, "weight_loader", default_weight_loader)
            weight_loader(param, loaded_weight)
            loaded_params.add(name)
        return loaded_params

    def _forward(self, x, t, cross_attn_cond):
        timestep_embed = self.to_timestep_embed(self.timestep_features(t[:, None]))
        prepend_inputs = timestep_embed.unsqueeze(1)
        prepend_length = prepend_inputs.shape[1]

        x = self.preprocess_conv(x) + x
        x = rearrange(x, "b c n -> b n c")
        output = self.transformer(x, prepend_embeds=prepend_inputs, context=cross_attn_cond)
        output = rearrange(output, "b n c -> b c n")[:, :, prepend_length:]
        return self.postprocess_conv(output) + output

    def forward(
        self,
        x,
        t,
        cross_attn_cond,
        negative_cross_attn_cond=None,
        negative_cross_attn_mask=None,
        cfg_scale: float = 1.0,
        scale_phi: float = 0.0,
        **kwargs,
    ):
        if cfg_scale == 1.0:
            return self._forward(x, t, cross_attn_cond)

        # Classifier-free guidance: batch the conditional + unconditional pass.
        null_embed = torch.zeros_like(cross_attn_cond)
        if negative_cross_attn_cond is not None and negative_cross_attn_mask is not None:
            mask = negative_cross_attn_mask.to(torch.bool).unsqueeze(2)
            negative_cross_attn_cond = torch.where(mask, negative_cross_attn_cond, null_embed)
        uncond = negative_cross_attn_cond if negative_cross_attn_cond is not None else null_embed

        batch_output = self._forward(
            torch.cat([x, x], dim=0),
            torch.cat([t, t], dim=0),
            torch.cat([cross_attn_cond, uncond], dim=0),
        )
        cond_output, uncond_output = torch.chunk(batch_output, 2, dim=0)
        cfg_output = uncond_output + (cond_output - uncond_output) * cfg_scale

        if scale_phi == 0.0:
            return cfg_output
        cond_std = cond_output.std(dim=1, keepdim=True)
        cfg_std = cfg_output.std(dim=1, keepdim=True)
        return scale_phi * (cfg_output * (cond_std / cfg_std)) + (1 - scale_phi) * cfg_output


class ContinuousMMDiTTransformer(nn.Module):
    def __init__(
        self,
        dim,
        depth,
        *,
        dim_in=None,
        dim_out=None,
        dim_heads=64,
        _latent_seq_len=237,
    ):
        super().__init__()

        self.dim = dim
        self.depth = depth

        self.project_in = nn.Linear(dim_in, dim, bias=False) if dim_in is not None else nn.Identity()
        self.project_out = nn.Linear(dim, dim_out, bias=False) if dim_out is not None else nn.Identity()

        hidden_dim = dim
        num_heads = dim_heads
        mlp_ratio = 4.0
        self._latent_seq_len = _latent_seq_len

        self.layers = nn.ModuleList(
            [
                AudioXMMDiTBlock(
                    hidden_dim,
                    num_heads,
                    mlp_ratio=mlp_ratio,
                    prefix=f"layers.{i}",
                )
                for i in range(depth)
            ]
        )
        self.proj_mm_tokens = nn.Linear(768, hidden_dim) if dim != 768 else nn.Identity()
        self.proj_mm_seq_len = nn.Linear(384, self._latent_seq_len) if self._latent_seq_len != 384 else nn.Identity()

        # AudioX RoPE: interleaved (GPT-J) pair layout, theta=10000.
        head_dim = hidden_dim // num_heads
        pos = torch.arange(self._latent_seq_len, dtype=torch.float32, device=self.device)
        inv_freq = 1.0 / (10000 ** (torch.arange(0, head_dim, 2, dtype=torch.float32, device=self.device) / head_dim))
        ang = torch.outer(pos, inv_freq)
        self.register_buffer("latent_rope_cos", torch.cos(ang), persistent=False)
        self.register_buffer("latent_rope_sin", torch.sin(ang), persistent=False)

    @property
    def device(self):
        return next(self.parameters()).device

    def forward(
        self,
        x,
        prepend_embeds=None,
        context=None,
    ):
        x = self.project_in(x)

        if prepend_embeds is not None:
            prepend_dim = prepend_embeds.shape[-1]
            assert prepend_dim == x.shape[-1], "prepend dimension must match sequence dimension"
            x = torch.cat((prepend_embeds, x), dim=-2)

        time_cond = prepend_embeds.squeeze(1)
        mm_tokens = context

        mm_tokens = self.proj_mm_tokens(mm_tokens)
        mm_tokens = rearrange(mm_tokens, "b s d -> b d s")
        mm_tokens = self.proj_mm_seq_len(mm_tokens)
        mm_tokens = rearrange(mm_tokens, "b d s -> b s d")

        time_cond = time_cond.unsqueeze(1)
        rot = (
            self.latent_rope_cos.to(device=x.device, dtype=x.dtype),
            self.latent_rope_sin.to(device=x.device, dtype=x.dtype),
        )
        for block in self.layers:
            x = block(x, mm_tokens, rot, context=time_cond)

        x = self.project_out(x)
        return x


def __getattr__(name: str):
    if name in __all__:
        return globals()[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
