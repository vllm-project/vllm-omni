# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import logging
from collections.abc import Iterable

import torch
import torch.nn.functional as F
from torch import nn
from vllm.distributed import get_tensor_model_parallel_world_size
from vllm.model_executor.layers.linear import (
    ColumnParallelLinear,
    MergedColumnParallelLinear,
    QKVParallelLinear,
    RowParallelLinear,
)
from vllm.model_executor.model_loader.weight_utils import default_weight_loader

from vllm_omni.diffusion.attention.backends.abstract import AttentionMetadata
from vllm_omni.diffusion.attention.layer import Attention
from vllm_omni.diffusion.layers.fourier import GaussianFourierProjection
from vllm_omni.diffusion.layers.rope import RotaryEmbedding

logger = logging.getLogger(__name__)

__all__ = ["AudioOmniDiT", "AudioOmniTransformerBlock"]

_ROTARY_DIM = 32
_ROPE_THETA = 10000.0


class AudioOmniLayerNorm(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(dim))
        self.beta = nn.Parameter(torch.zeros(dim), requires_grad=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.layer_norm(x, x.shape[-1:], weight=self.gamma, bias=self.beta)


class AudioOmniFeedForward(nn.Module):
    def __init__(self, dim: int, inner_dim: int, prefix: str = ""):
        super().__init__()
        self.proj = MergedColumnParallelLinear(
            input_size=dim, output_sizes=[inner_dim, inner_dim], bias=True, prefix=f"{prefix}.proj"
        )
        self.out = RowParallelLinear(inner_dim, dim, bias=True, input_is_parallel=True, prefix=f"{prefix}.out")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, _ = self.proj(x)
        value, gate = h.chunk(2, dim=-1)
        y, _ = self.out(value * F.silu(gate))
        return y


class AudioOmniSelfAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int, prefix: str = ""):
        super().__init__()
        self.head_dim = dim // num_heads
        self.to_qkv = QKVParallelLinear(
            hidden_size=dim,
            head_size=self.head_dim,
            total_num_heads=num_heads,
            bias=False,
            prefix=f"{prefix}.to_qkv",
        )
        self.to_out = RowParallelLinear(dim, dim, bias=False, input_is_parallel=True, prefix=f"{prefix}.to_out")
        self.rope = RotaryEmbedding(is_neox_style=True)
        self.attn = Attention(
            num_heads=self.to_qkv.num_heads,
            head_size=self.head_dim,
            softmax_scale=self.head_dim**-0.5,
            causal=False,
            prefix=prefix,
        )

    def forward(self, x: torch.Tensor, rot: tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        local_h = self.to_qkv.num_heads
        d = self.head_dim
        qkv, _ = self.to_qkv(x)
        q, k, v = qkv.split([local_h * d, local_h * d, local_h * d], dim=-1)
        q = q.unflatten(-1, (local_h, d))
        k = k.unflatten(-1, (local_h, d))
        v = v.unflatten(-1, (local_h, d))

        cos, sin = rot
        in_dtype = q.dtype
        q = self.rope.forward_native(q.float(), cos, sin).to(in_dtype)
        k = self.rope.forward_native(k.float(), cos, sin).to(in_dtype)

        out = self.attn(q.contiguous(), k.contiguous(), v.contiguous(), attn_metadata=None)
        out, _ = self.to_out(out.flatten(2, 3).contiguous())
        return out


class AudioOmniCrossAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int, prefix: str = ""):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.to_q = ColumnParallelLinear(dim, dim, bias=False, gather_output=False, prefix=f"{prefix}.to_q")
        self.to_kv = MergedColumnParallelLinear(
            input_size=dim, output_sizes=[dim, dim], bias=False, gather_output=False, prefix=f"{prefix}.to_kv"
        )
        self.to_out = RowParallelLinear(dim, dim, bias=False, input_is_parallel=True, prefix=f"{prefix}.to_out")
        local_heads = num_heads // get_tensor_model_parallel_world_size()
        self.attn = Attention(
            num_heads=local_heads,
            head_size=self.head_dim,
            softmax_scale=self.head_dim**-0.5,
            causal=False,
            role="cross",
            prefix=prefix,
        )

    def forward(
        self,
        x: torch.Tensor,
        context: torch.Tensor,
        context_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        local_h = self.num_heads // get_tensor_model_parallel_world_size()
        d = self.head_dim
        q, _ = self.to_q(x)
        kv, _ = self.to_kv(context)
        k, v = kv.chunk(2, dim=-1)
        q = q.unflatten(-1, (local_h, d))
        k = k.unflatten(-1, (local_h, d))
        v = v.unflatten(-1, (local_h, d))

        # The caller passes None when the mask is all-True (resolved once per request in
        # the pipeline), so a non-None mask here always needs an AttentionMetadata — no
        # per-call .all() GPU->CPU sync.
        attn_metadata = None
        if context_mask is not None:
            attn_metadata = AttentionMetadata(attn_mask=context_mask.to(torch.bool))

        out = self.attn(q.contiguous(), k.contiguous(), v.contiguous(), attn_metadata=attn_metadata)
        out, _ = self.to_out(out.flatten(2, 3).contiguous())
        return out


class AudioOmniTransformerBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int, ff_mult: int = 4, prefix: str = ""):
        super().__init__()
        self.pre_norm = AudioOmniLayerNorm(dim)
        self.self_attn = AudioOmniSelfAttention(dim, num_heads, prefix=f"{prefix}.self_attn")
        self.cross_attend_norm = AudioOmniLayerNorm(dim)
        self.cross_attn = AudioOmniCrossAttention(dim, num_heads, prefix=f"{prefix}.cross_attn")
        self.ff_norm = AudioOmniLayerNorm(dim)
        self.ff = AudioOmniFeedForward(dim, dim * ff_mult, prefix=f"{prefix}.ff")

    def forward(
        self,
        x: torch.Tensor,
        rot: tuple[torch.Tensor, torch.Tensor],
        context: torch.Tensor,
        context_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        x = x + self.self_attn(self.pre_norm(x), rot)
        x = x + self.cross_attn(self.cross_attend_norm(x), context=context, context_mask=context_mask)
        x = x + self.ff(self.ff_norm(x))
        return x


class AudioOmniContinuousTransformer(nn.Module):
    def __init__(self, dim: int, depth: int, num_heads: int, dim_in: int, dim_out: int):
        super().__init__()
        self.project_in = nn.Linear(dim_in, dim, bias=False)
        self.project_out = nn.Linear(dim, dim_out, bias=False)
        self.layers = nn.ModuleList(
            [AudioOmniTransformerBlock(dim, num_heads, prefix=f"layers.{i}") for i in range(depth)]
        )
        inv_freq = 1.0 / (_ROPE_THETA ** (torch.arange(0, _ROTARY_DIM, 2, dtype=torch.float32) / _ROTARY_DIM))
        self.register_buffer("rope_inv_freq", inv_freq, persistent=False)
        # cos/sin depend only on (seq_len, device); the sequence length is constant across
        # the ~100 sampling steps, so cache to skip per-step recomputation. Bit-identical.
        self._rope_cache: dict[tuple[int, torch.device], tuple[torch.Tensor, torch.Tensor]] = {}

    def _rotary(self, seq_len: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        cached = self._rope_cache.get((seq_len, device))
        if cached is None:
            t = torch.arange(seq_len, dtype=torch.float32, device=device)
            ang = torch.outer(t, self.rope_inv_freq.to(device))
            cached = (ang.cos(), ang.sin())
            self._rope_cache[(seq_len, device)] = cached
        return cached

    def forward(
        self,
        x: torch.Tensor,
        prepend_embeds: torch.Tensor,
        context: torch.Tensor,
        context_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        x = self.project_in(x)
        x = torch.cat((prepend_embeds, x), dim=-2)
        rot = self._rotary(x.shape[1], x.device)
        for block in self.layers:
            x = block(x, rot, context=context, context_mask=context_mask)
        return self.project_out(x)


class AudioOmniDiT(nn.Module):
    """Top-level DiT (CFG double-batch inside forward)."""

    _repeated_blocks = ["AudioOmniTransformerBlock"]
    _layerwise_offload_blocks_attrs = ["transformer_blocks"]

    @staticmethod
    def _is_transformer_block(name: str, module: nn.Module) -> bool:
        return isinstance(module, AudioOmniTransformerBlock)

    _hsdp_shard_conditions = [_is_transformer_block]

    def __init__(
        self,
        io_channels: int = 64,
        embed_dim: int = 2048,
        depth: int = 36,
        num_heads: int = 32,
        cond_token_dim: int = 768,
        global_cond_dim: int = 768,
        project_cond_tokens: bool = True,
        **kwargs,
    ):
        super().__init__()
        if kwargs:
            logger.debug("AudioOmniDiT ignoring unused config keys: %s", sorted(kwargs.keys()))
        if not project_cond_tokens:
            raise ValueError("Audio-Omni checkpoint requires project_cond_tokens=True.")

        timestep_features_dim = 256
        self.timestep_features = GaussianFourierProjection(
            in_features=1, embedding_size=timestep_features_dim // 2, scale=1.0, trainable=False
        )
        self.to_timestep_embed = nn.Sequential(
            nn.Linear(timestep_features_dim, embed_dim, bias=True),
            nn.SiLU(),
            nn.Linear(embed_dim, embed_dim, bias=True),
        )
        self.to_cond_embed = nn.Sequential(
            nn.Linear(cond_token_dim, embed_dim, bias=False),
            nn.SiLU(),
            nn.Linear(embed_dim, embed_dim, bias=False),
        )
        self.to_global_embed = nn.Sequential(
            nn.Linear(global_cond_dim, embed_dim, bias=False),
            nn.SiLU(),
            nn.Linear(embed_dim, embed_dim, bias=False),
        )

        self.transformer = AudioOmniContinuousTransformer(
            dim=embed_dim, depth=depth, num_heads=num_heads, dim_in=io_channels, dim_out=io_channels
        )

        self.preprocess_conv = nn.Conv1d(io_channels, io_channels, 1, bias=False)
        self.postprocess_conv = nn.Conv1d(io_channels, io_channels, 1, bias=False)

    @property
    def transformer_blocks(self) -> nn.ModuleList:
        return self.transformer.layers

    def _forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        cross_attn_cond: torch.Tensor,
        cross_attn_mask: torch.Tensor | None,
        global_cond: torch.Tensor,
    ) -> torch.Tensor:
        cross_attn_cond = self.to_cond_embed(cross_attn_cond)

        global_embed = self.to_global_embed(global_cond)
        timestep_embed = self.to_timestep_embed(self.timestep_features(t[:, None]))
        prepend_inputs = global_embed + timestep_embed.unsqueeze(1)
        prepend_length = prepend_inputs.shape[1]

        x = self.preprocess_conv(x) + x
        x = x.transpose(1, 2)

        output = self.transformer(
            x, prepend_embeds=prepend_inputs, context=cross_attn_cond, context_mask=cross_attn_mask
        )

        output = output.transpose(1, 2)[:, :, prepend_length:]
        return self.postprocess_conv(output) + output

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        cross_attn_cond: torch.Tensor,
        cross_attn_mask: torch.Tensor | None = None,
        global_cond: torch.Tensor | None = None,
        negative_cross_attn_cond: torch.Tensor | None = None,
        negative_cross_attn_mask: torch.Tensor | None = None,
        cfg_scale: float = 1.0,
        scale_phi: float = 0.0,
        **kwargs,
    ) -> torch.Tensor:
        if cross_attn_mask is not None:
            cross_attn_mask = cross_attn_mask.bool()

        if cfg_scale == 1.0:
            return self._forward(x, t, cross_attn_cond, cross_attn_mask, global_cond)

        null_embed = torch.zeros_like(cross_attn_cond)
        if negative_cross_attn_cond is not None:
            if negative_cross_attn_mask is not None:
                neg_mask = negative_cross_attn_mask.to(torch.bool).unsqueeze(2)
                negative_cross_attn_cond = torch.where(neg_mask, negative_cross_attn_cond, null_embed)
            uncond = negative_cross_attn_cond
        else:
            uncond = null_embed

        batch_mask = torch.cat([cross_attn_mask, cross_attn_mask], dim=0) if cross_attn_mask is not None else None
        batch_output = self._forward(
            torch.cat([x, x], dim=0),
            torch.cat([t, t], dim=0),
            torch.cat([cross_attn_cond, uncond], dim=0),
            batch_mask,
            torch.cat([global_cond, global_cond], dim=0),
        )
        cond_output, uncond_output = torch.chunk(batch_output, 2, dim=0)
        cfg_output = uncond_output + (cond_output - uncond_output) * cfg_scale

        if scale_phi == 0.0:
            return cfg_output
        cond_std = cond_output.std(dim=1, keepdim=True)
        cfg_std = cfg_output.std(dim=1, keepdim=True)
        return scale_phi * (cfg_output * (cond_std / cfg_std)) + (1 - scale_phi) * cfg_output

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        params_dict = dict(self.named_parameters())
        loaded: set[str] = set()

        def _load_shards(target: str, tensor: torch.Tensor, shard_ids: list) -> None:
            param = params_dict[target]
            shards = tensor.chunk(len(shard_ids), dim=0)
            for shard_id, shard in zip(shard_ids, shards):
                param.weight_loader(param, shard, shard_id)
            loaded.add(target)

        for name, tensor in weights:
            if name == "transformer.rotary_pos_emb.inv_freq":
                continue
            if ".ff.ff.0.proj." in name:
                _load_shards(name.replace(".ff.ff.0.proj.", ".ff.proj."), tensor, [0, 1])
            elif ".ff.ff.2." in name:
                target = name.replace(".ff.ff.2.", ".ff.out.")
                param = params_dict[target]
                getattr(param, "weight_loader", default_weight_loader)(param, tensor)
                loaded.add(target)
            elif ".self_attn.to_qkv." in name:
                _load_shards(name, tensor, ["q", "k", "v"])
            elif ".cross_attn.to_kv." in name:
                _load_shards(name, tensor, [0, 1])
            elif name in params_dict:
                param = params_dict[name]
                getattr(param, "weight_loader", default_weight_loader)(param, tensor)
                loaded.add(name)
            else:
                logger.warning("AudioOmniDiT.load_weights: unexpected key %s", name)
        return loaded
