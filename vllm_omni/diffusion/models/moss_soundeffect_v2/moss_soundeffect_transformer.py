import math
from typing import Literal

import torch
import torch.nn as nn
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import (
    MergedColumnParallelLinear,
    QKVParallelLinear,
    ReplicatedLinear,
)

from vllm_omni.diffusion.attention.layer import Attention
from vllm_omni.diffusion.layers.rope import RotaryEmbeddingWan


class SelfAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int, prefix: str = "", quant_config=None, eps: float = 1e-6):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        qkv_prefix = f"{prefix}.to_qkv" if prefix else "to_qkv"
        self.to_qkv = QKVParallelLinear(
            hidden_size=self.dim,
            head_size=self.head_dim,
            total_num_heads=self.num_heads,
            total_num_kv_heads=self.num_heads,
            bias=True,
            quant_config=quant_config,
            prefix=qkv_prefix,
            disable_tp=True,
        )
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_heads * self.head_dim
        self.o = ReplicatedLinear(
            input_size=dim,
            output_size=dim,
            bias=True,
            quant_config=quant_config,
            prefix=f"{prefix}.o" if prefix else "o",
        )
        self.norm_q = RMSNorm(dim, eps=eps)
        self.norm_k = RMSNorm(dim, eps=eps)
        self.rotary_embedding = RotaryEmbeddingWan(is_neox_style=False, half_head_dim=True)

        self.attn = Attention(
            num_heads=self.num_heads,
            head_size=self.head_dim,
            softmax_scale=1.0 / (self.head_dim**0.5),
            causal=False,  # Diffusion models typically use bidirectional attention
            num_kv_heads=self.num_heads,
            prefix=prefix,
            role="self",
        )

    def forward(self, x, rotary_emb):
        qkv, _ = self.to_qkv(x)
        q, k, v = qkv.split((self.q_size, self.kv_size, self.kv_size), dim=-1)
        q = self.norm_q(q)
        k = self.norm_k(k)
        batch_size, seq_len, _ = q.shape
        q = q.reshape(batch_size, seq_len, self.num_heads, self.head_dim)
        k = k.reshape(batch_size, seq_len, self.num_heads, self.head_dim)
        v = v.reshape(batch_size, seq_len, self.num_heads, self.head_dim)
        cos, sin = rotary_emb
        output_dtype = q.dtype
        cos = cos.to(dtype=torch.float32, device=q.device)
        sin = sin.to(dtype=torch.float32, device=q.device)
        q = self.rotary_embedding(q.float(), cos, sin).to(output_dtype)
        k = self.rotary_embedding(k.float(), cos, sin).to(output_dtype)
        x = self.attn(q, k, v)
        x = x.reshape(batch_size, seq_len, self.dim)
        x, _ = self.o(x)
        return x


class CrossAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        eps: float = 1e-6,
        quant_config=None,
        prefix: str = "",
        has_image_input: bool = False,
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.q = ReplicatedLinear(
            input_size=dim,
            output_size=dim,
            bias=True,
            quant_config=quant_config,
            prefix=f"{prefix}.q" if prefix else "q",
        )
        self.kv = MergedColumnParallelLinear(
            input_size=dim,
            output_sizes=[dim, dim],
            bias=True,
            gather_output=False,
            disable_tp=True,
            quant_config=quant_config,
            prefix=f"{prefix}.kv" if prefix else "kv",
        )
        self.o = ReplicatedLinear(
            input_size=dim,
            output_size=dim,
            bias=True,
            quant_config=quant_config,
            prefix=f"{prefix}.o" if prefix else "o",
        )
        self.norm_q = RMSNorm(dim, eps=eps)
        self.norm_k = RMSNorm(dim, eps=eps)
        self.has_image_input = has_image_input
        if has_image_input:
            self.k_img = ReplicatedLinear(
                dim,
                dim,
                bias=True,
                quant_config=quant_config,
                prefix=f"{prefix}.k_img" if prefix else "k_img",
            )
            self.v_img = ReplicatedLinear(
                dim,
                dim,
                bias=True,
                quant_config=quant_config,
                prefix=f"{prefix}.v_img" if prefix else "v_img",
            )
            self.norm_k_img = RMSNorm(dim, eps=eps)

        self.attn = Attention(
            num_heads=self.num_heads,
            head_size=self.head_dim,
            softmax_scale=1.0 / (self.head_dim**0.5),
            causal=False,  # Diffusion models typically use bidirectional attention
            num_kv_heads=self.num_heads,
            prefix=prefix,
            role="cross",
        )

    def forward(self, x: torch.Tensor, y: torch.Tensor):
        if self.has_image_input:
            img = y[:, :257]
            ctx = y[:, 257:]
        else:
            ctx = y
        q, _ = self.q(x)
        kv, _ = self.kv(ctx)
        k, v = kv.chunk(2, dim=-1)
        q = self.norm_q(q)
        k = self.norm_k(k)
        batch_size, seq_len_q, _ = q.shape
        _, seq_len_kv, _ = k.shape
        q = q.reshape(batch_size, seq_len_q, self.num_heads, self.head_dim)
        k = k.reshape(batch_size, seq_len_kv, self.num_heads, self.head_dim)
        v = v.reshape(batch_size, seq_len_kv, self.num_heads, self.head_dim)
        x = self.attn(q, k, v)
        if self.has_image_input:
            k_img, _ = self.k_img(img)
            v_img, _ = self.v_img(img)
            k_img = self.norm_k_img(k_img)
            k_img = k_img.reshape(batch_size, -1, self.num_heads, self.head_dim)
            v_img = v_img.reshape(batch_size, -1, self.num_heads, self.head_dim)
            y = self.attn(q, k_img, v_img)
            x = x + y
        x = x.reshape(batch_size, seq_len_q, self.dim)
        x, _ = self.o(x)
        return x


class GateModule(nn.Module):
    def __init__(
        self,
    ):
        super().__init__()

    def forward(self, x, gate, residual):
        return torch.addcmul(x, gate, residual)


class DiTBlock(nn.Module):
    def __init__(
        self,
        has_image_input: bool,
        dim: int,
        num_heads: int,
        ffn_dim: int,
        eps: float = 1e-6,
        quant_config=None,
        prefix: str = "",
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.ffn_dim = ffn_dim

        self.self_attn = SelfAttention(dim, num_heads, prefix=f"{prefix}.self_attn", quant_config=quant_config, eps=eps)
        self.cross_attn = CrossAttention(
            dim,
            num_heads,
            eps=eps,
            quant_config=quant_config,
            prefix=f"{prefix}.cross_attn",
            has_image_input=has_image_input,
        )
        self.norm1 = nn.LayerNorm(dim, eps=eps, elementwise_affine=False)
        self.norm2 = nn.LayerNorm(dim, eps=eps, elementwise_affine=False)
        self.norm3 = nn.LayerNorm(dim, eps=eps)
        self.ffn = nn.Sequential(nn.Linear(dim, ffn_dim), nn.GELU(approximate="tanh"), nn.Linear(ffn_dim, dim))
        self.modulation = nn.Parameter(torch.randn(1, 6, dim) / dim**0.5)
        self.gate = GateModule()

    @staticmethod
    def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor):
        return x * (1 + scale) + shift

    def forward(self, x, context, t_mod, rotary_emb):
        has_seq = len(t_mod.shape) == 4
        chunk_dim = 2 if has_seq else 1
        # msa: multi-head self-attention  mlp: multi-layer perceptron
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.modulation.to(dtype=t_mod.dtype, device=t_mod.device) + t_mod
        ).chunk(6, dim=chunk_dim)
        if has_seq:
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
                shift_msa.squeeze(2),
                scale_msa.squeeze(2),
                gate_msa.squeeze(2),
                shift_mlp.squeeze(2),
                scale_mlp.squeeze(2),
                gate_mlp.squeeze(2),
            )
        input_x = self.modulate(self.norm1(x), shift_msa, scale_msa)
        x = self.gate(x, gate_msa, self.self_attn(input_x, rotary_emb))
        x = x + self.cross_attn(self.norm3(x), context)
        input_x = self.modulate(self.norm2(x), shift_mlp, scale_mlp)
        x = self.gate(x, gate_mlp, self.ffn(input_x))
        return x


def sinusoidal_embedding_1d(dim, position):
    sinusoid = torch.outer(
        position.type(torch.float64),
        torch.pow(10000, -torch.arange(dim // 2, dtype=torch.float64, device=position.device).div(dim // 2)),
    )
    x = torch.cat([torch.cos(sinusoid), torch.sin(sinusoid)], dim=1)
    return x.to(position.dtype)


def _precompute_freqs_cis_1d(dim: int, end: int = 16384, theta: float = 10000.0):
    freqs = _precompute_freqs_cis(dim, end, theta)
    return freqs.chunk(3, dim=-1)


def _legacy_precompute_freqs_cis_1d(
    dim: int,
    end: int = 16384,
    theta: float = 10000.0,
    base_tps: float = 4.0,
    target_tps: float = 44100 / 2048,
):
    scale = base_tps / target_tps
    freqs = _precompute_freqs_cis(dim - 2 * (dim // 3), end, theta, scale)
    no_freqs = torch.ones_like(_precompute_freqs_cis(dim // 3, end, theta, scale))
    return freqs, no_freqs, no_freqs


def _precompute_freqs_cis(dim: int, end: int = 16384, theta: float = 10000.0, scale: float = 1.0):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].double() / dim))
    positions = torch.arange(end, dtype=torch.float64, device=freqs.device) * scale
    freqs = torch.outer(positions, freqs)
    return torch.polar(torch.ones_like(freqs), freqs)


class MLP(nn.Module):
    def __init__(self, in_dim, out_dim, has_pos_emb=False):
        super().__init__()
        self.proj = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, in_dim),
            nn.GELU(),
            nn.Linear(in_dim, out_dim),
            nn.LayerNorm(out_dim),
        )
        self.has_pos_emb = has_pos_emb
        if has_pos_emb:
            self.emb_pos = nn.Parameter(torch.zeros((1, 514, 1280)))

    def forward(self, x):
        if self.has_pos_emb:
            x = x + self.emb_pos.to(dtype=x.dtype, device=x.device)
        return self.proj(x)


class Head(nn.Module):
    def __init__(self, dim: int, out_dim: int, patch_size: tuple[int, int, int], eps: float):
        super().__init__()
        self.norm = nn.LayerNorm(dim, eps=eps, elementwise_affine=False)
        self.head = nn.Linear(dim, out_dim * math.prod(patch_size))
        self.modulation = nn.Parameter(torch.randn(1, 2, dim) / dim**0.5)

    def forward(self, x, t_mod):
        if t_mod.ndim == 3:
            shift, scale = (
                self.modulation.unsqueeze(0).to(dtype=t_mod.dtype, device=t_mod.device) + t_mod.unsqueeze(2)
            ).chunk(2, dim=2)
            return self.head(self.norm(x) * (1 + scale.squeeze(2)) + shift.squeeze(2))

        shift, scale = (self.modulation.to(dtype=t_mod.dtype, device=t_mod.device) + t_mod.unsqueeze(1)).chunk(2, dim=1)
        return self.head(self.norm(x) * (1 + scale) + shift)


class WanAudioModel(nn.Module):
    def __init__(
        self,
        dim: int,
        in_dim: int,
        ffn_dim: int,
        out_dim: int,
        text_dim: int,
        freq_dim: int,
        eps: float,
        patch_size: tuple[int, int, int],
        num_heads: int,
        num_layers: int,
        has_image_input: bool,
        has_image_pos_emb: bool = False,
        has_ref_conv: bool = False,
        require_vae_embedding: bool = True,
        require_clip_embedding: bool = True,
        vae_type: Literal["oobleck", "dac"] = "oobleck",
        quant_config=None,
        prefix: str = "",
    ):
        super().__init__()
        self.dim = dim
        self.freq_dim = freq_dim
        self.has_image_input = has_image_input
        self.patch_size = patch_size
        self.require_vae_embedding = require_vae_embedding
        self.require_clip_embedding = require_clip_embedding
        self.vae_type = vae_type
        self.patch_embedding = nn.Conv1d(in_dim, dim, kernel_size=patch_size, stride=patch_size)
        self.text_embedding = nn.Sequential(nn.Linear(text_dim, dim), nn.GELU(approximate="tanh"), nn.Linear(dim, dim))
        self.time_embedding = nn.Sequential(nn.Linear(freq_dim, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.time_projection = nn.Sequential(nn.SiLU(), nn.Linear(dim, dim * 6))
        self.blocks = nn.ModuleList(
            [
                DiTBlock(
                    has_image_input,
                    dim,
                    num_heads,
                    ffn_dim,
                    eps,
                    quant_config=quant_config,
                    prefix=f"{prefix}.blocks.{layer_idx}" if prefix else f"blocks.{layer_idx}",
                )
                for layer_idx in range(num_layers)
            ]
        )
        self.head = Head(dim, out_dim, patch_size, eps)

        head_dim = dim // num_heads
        if vae_type == "oobleck":
            freqs = _legacy_precompute_freqs_cis_1d(head_dim)
        elif vae_type == "dac":
            freqs = _precompute_freqs_cis_1d(head_dim)
        else:
            raise ValueError(f"Invalid VAE type: {vae_type}")
        self.register_buffer(
            "rope_cos_cache",
            torch.cat([freq.real.float() for freq in freqs], dim=-1),
            persistent=False,
        )
        self.register_buffer(
            "rope_sin_cache",
            torch.cat([freq.imag.float() for freq in freqs], dim=-1),
            persistent=False,
        )

        if has_image_input:
            self.img_emb = MLP(1280, dim, has_pos_emb=has_image_pos_emb)
        if has_ref_conv:
            self.ref_conv = nn.Conv2d(16, dim, kernel_size=(2, 2), stride=(2, 2))
        self.has_image_pos_emb = has_image_pos_emb
        self.has_ref_conv = has_ref_conv
        self.control_adapter = None

    def patchify(self, x: torch.Tensor, control_camera_latents_input: torch.Tensor | None = None):
        x = self.patch_embedding(x)
        if self.control_adapter is not None and control_camera_latents_input is not None:
            y_camera = self.control_adapter(control_camera_latents_input)
            x = [u + v for u, v in zip(x, y_camera)]
            x = x[0].unsqueeze(0)
        grid_size = x.shape[2:]
        x = x.transpose(1, 2).contiguous()
        return x, grid_size

    def unpatchify(self, x: torch.Tensor, grid_size: torch.Tensor):
        batch_size, frames, channels = x.shape
        patch_frames = self.patch_size[0]
        x = x.reshape(batch_size, frames, patch_frames, channels // patch_frames)
        return x.permute(0, 3, 1, 2).reshape(batch_size, channels // patch_frames, frames * patch_frames)

    def forward(
        self,
        x: torch.Tensor,
        timestep: torch.Tensor,
        context: torch.Tensor,
        clip_feature: torch.Tensor | None = None,
        y: torch.Tensor | None = None,
        **kwargs,
    ):
        t = self.time_embedding(sinusoidal_embedding_1d(self.freq_dim, timestep))
        t_mod = self.time_projection(t).unflatten(1, (6, self.dim))
        context = self.text_embedding(context)

        if self.has_image_input:
            x = torch.cat([x, y], dim=1)
            context = torch.cat([self.img_emb(clip_feature), context], dim=1)

        x, (frames,) = self.patchify(x)
        cos = self.rope_cos_cache[:frames].reshape(frames, 1, -1)
        sin = self.rope_sin_cache[:frames].reshape(frames, 1, -1)
        for block in self.blocks:
            x = block(x, context, t_mod, (cos, sin))
        x = self.head(x, t)
        return self.unpatchify(x, (frames,))
