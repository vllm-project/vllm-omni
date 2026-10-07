# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""CPU-only WaveServe tiny DiT / adapter helpers for unit tests.

Production ``WaveServeWanPipeline`` / ``StageWanTransformer`` no longer ship a
silent fake-output tiny mode; these helpers keep the abstract-token path for
CPU tests that must not load Wan linear layers.
"""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn as nn
from vllm.model_executor.models.utils import PPMissingLayer
from vllm.sequence import IntermediateTensors

from vllm_omni.diffusion.models.waveserve_wan.transformer import stage_layer_range
from vllm_omni.experimental.ar_diffusion.chunk_executor import ChunkAdapter
from vllm_omni.experimental.ar_diffusion.kv_cache.paged_attention import paged_write_attn


class TinyStageWanSelfAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int, head_dim: int) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.o_proj = nn.Linear(dim, dim, bias=True)
        self.scale = head_dim**-0.5

    def forward(self, hidden: torch.Tensor, kv_ctx: Any | None) -> torch.Tensor:
        b, s, _ = hidden.shape
        qkv = self.qkv(hidden).view(b, s, 3, self.num_heads, self.head_dim)
        query, key, value = qkv.unbind(dim=2)
        if kv_ctx is not None:
            inputs = kv_ctx.to_layer_inputs() if hasattr(kv_ctx, "to_layer_inputs") else kv_ctx
            outs = [paged_write_attn(inputs, query[i], key[i], value[i], None, None, self.scale) for i in range(b)]
            attn = torch.stack(outs, dim=0)
        else:
            scores = torch.einsum("bqhd,bkhd->bhqk", query.float(), key.float()) * self.scale
            probs = torch.softmax(scores, dim=-1).to(value.dtype)
            attn = torch.einsum("bhqk,bkhd->bqhd", probs, value)
        return self.o_proj(attn.flatten(-2))


class TinyStageWanBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int, head_dim: int, ffn_dim: int) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = TinyStageWanSelfAttention(dim, num_heads, head_dim)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(nn.Linear(dim, ffn_dim), nn.GELU(), nn.Linear(ffn_dim, dim))

    def forward(self, hidden: torch.Tensor, kv_ctx: Any | None) -> torch.Tensor:
        hidden = hidden + self.attn(self.norm1(hidden), kv_ctx)
        return hidden + self.ffn(self.norm2(hidden))


class TinyStageWanTransformer(nn.Module):
    """Minimal abstract-token DiT for CPU chunk-path tests."""

    def __init__(
        self,
        *,
        num_layers: int = 2,
        dim: int = 32,
        num_heads: int = 2,
        ffn_dim: int = 64,
        in_channels: int = 16,
        patch_size: tuple[int, int, int] = (1, 2, 2),
        layer_groups: int = 1,
        pp_rank: int = 0,
    ) -> None:
        super().__init__()
        self.layer_groups = max(1, int(layer_groups))
        self.pp_rank = int(pp_rank)
        self.group = 0 if self.layer_groups == 1 else self.pp_rank % self.layer_groups
        self.is_stage_first = self.group == 0
        self.is_stage_last = self.group == self.layer_groups - 1
        self.patch_size = tuple(patch_size)
        self.dim = dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.in_features = in_channels * math.prod(self.patch_size)
        self.start_layer, self.end_layer = stage_layer_range(num_layers, self.group, self.layer_groups)
        if self.is_stage_first:
            self.patch_embed = nn.Linear(self.in_features, dim)
        else:
            self.patch_embed = PPMissingLayer()
        blocks: list[nn.Module] = []
        for idx in range(num_layers):
            if self.start_layer <= idx < self.end_layer:
                blocks.append(TinyStageWanBlock(dim, num_heads, self.head_dim, ffn_dim))
            else:
                blocks.append(PPMissingLayer())
        self.blocks = nn.ModuleList(blocks)
        if self.is_stage_last:
            self.proj_out = nn.Linear(dim, self.in_features)
        else:
            self.proj_out = PPMissingLayer()

    @property
    def local_num_layers(self) -> int:
        return max(0, self.end_layer - self.start_layer)

    def forward(
        self,
        hidden_states: torch.Tensor | IntermediateTensors | None = None,
        kv_contexts: list[Any] | None = None,
        intermediate_tensors: IntermediateTensors | None = None,
    ) -> torch.Tensor | IntermediateTensors:
        if isinstance(hidden_states, IntermediateTensors):
            intermediate_tensors = hidden_states
            hidden_states = None
        if intermediate_tensors is not None:
            hidden = intermediate_tensors["hidden_states"]
        elif hidden_states is None:
            raise RuntimeError("tiny stage transformer received no hidden states")
        elif self.is_stage_first and hidden_states.size(-1) == self.in_features:
            hidden = self.patch_embed(hidden_states)
        else:
            hidden = hidden_states
        local_count = self.local_num_layers
        for idx in range(self.start_layer, self.end_layer):
            ctx = None
            if kv_contexts is not None:
                ctx = kv_contexts[idx - self.start_layer] if len(kv_contexts) == local_count else kv_contexts[idx]
            hidden = self.blocks[idx](hidden, ctx)
        if self.is_stage_last:
            return self.proj_out(hidden)
        return IntermediateTensors({"hidden_states": hidden})


class TinyChunkAdapter(ChunkAdapter):
    """Abstract tokens through :class:`TinyStageWanTransformer`."""

    def __init__(self, transformer: TinyStageWanTransformer, seed_hidden: torch.Tensor) -> None:
        self.transformer = transformer
        self.seed_hidden = seed_hidden

    def forward(self, tasks, kv_contexts, *, hidden):
        if not tasks:
            return hidden
        outs = []
        for i, _task in enumerate(tasks):
            h = hidden
            if h is None:
                h = self.seed_hidden
            elif isinstance(h, torch.Tensor) and h.shape[0] == len(tasks):
                h = h[i : i + 1]
            elif isinstance(h, IntermediateTensors) and h["hidden_states"].shape[0] == len(tasks):
                h = IntermediateTensors({key: value[i : i + 1] for key, value in h.tensors.items()})
            ctx = kv_contexts[i] if i < len(kv_contexts) else None
            outs.append(self.transformer(h, kv_contexts=ctx))
        return self._stack(outs)

    @staticmethod
    def _stack(outs: list[Any]) -> Any:
        first = outs[0]
        if isinstance(first, IntermediateTensors):
            stacked = {key: torch.cat([out.tensors[key] for out in outs], dim=0) for key in first.tensors}
            return IntermediateTensors(stacked)
        if isinstance(first, torch.Tensor):
            return torch.cat(outs, dim=0)
        return first

    def pack_activation(self, output):
        if isinstance(output, IntermediateTensors):
            return dict(output.tensors)
        if isinstance(output, torch.Tensor):
            return {"hidden_states": output}
        raise TypeError(f"tiny adapter packs IntermediateTensors/tensor, got {type(output)}")

    def unpack_activation(self, payload: dict):
        if "hidden_states" in payload and len(payload) == 1:
            return payload["hidden_states"]
        return IntermediateTensors(dict(payload))
