# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""ZONOS3 AR modules, GDN state/kernels, Backbone and native model entry.

Modules are defined before their consumers; Zonos3ForCausalLM is the final
entry point. Forward returns Backbone hidden states; sampling owns the head.
GDN equations follow Zyphra/ZONOS3-Mini; see LICENSE.Zyphra.
"""

from __future__ import annotations

import importlib.metadata
import json
import os
import sys
from collections import deque
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch
from torch import nn
from torch.nn import functional as F
from vllm.config import VllmConfig
from vllm.config.compilation import CUDAGraphMode

from vllm_omni.model_executor.model_loader.weight_utils import download_weights_from_hf_specific
from vllm_omni.utils.device_copy import index_to_device

from .ar_config import Zonos3Architecture

if TYPE_CHECKING:
    from vllm.sequence import IntermediateTensors
    from vllm.v1.worker.gpu.mm.encoder_cache import EncoderCache

    from .model_state import Zonos3ARModelState

CHUNK = 64
HEADS = 4
HEAD_DIM = 128
INNER = HEADS * HEAD_DIM


@dataclass
class Zonos3GDNState:
    conv: torch.Tensor
    recurrent: torch.Tensor
    window: torch.Tensor
    window_length: torch.Tensor
    total_tokens: torch.Tensor
    prefill_tokens: list[int]

    def reset(self, slots: Sequence[int]) -> None:
        indices = index_to_device(slots, self.conv.device)
        for tensor in (self.conv, self.recurrent, self.window, self.window_length, self.total_tokens):
            tensor.index_fill_(0, indices, 0)
        for slot in slots:
            self.prefill_tokens[slot] = 0


@dataclass
class _Tables:
    cu: torch.Tensor
    chunks: torch.Tensor
    row_offsets: torch.Tensor


@dataclass
class _FLAKernels:
    conv: Callable[..., Any]
    norm: Callable[..., Any]
    l2norm: Callable[..., Any]
    beta: Callable[..., Any]
    delta: Callable[..., Any]


class Zonos3GDN(nn.Module):
    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.in_proj = nn.Linear(hidden_size, 4 * INNER + 2 * HEADS, bias=False)
        self.conv = nn.Conv1d(3 * INNER, 3 * INNER, 4, groups=3 * INNER, bias=False)
        self.out_weight = nn.Parameter(torch.ones(HEAD_DIM), requires_grad=False)
        self.A_log = nn.Parameter(torch.zeros(HEADS, dtype=torch.float32), requires_grad=False)
        self.dt_bias = nn.Parameter(torch.zeros(HEADS, dtype=torch.float32), requires_grad=False)
        self.wo = nn.Linear(INNER, hidden_size, bias=False)
        self._kernels: _FLAKernels | None = None
        self._tables: dict[tuple[int, int, torch.device], _Tables] = {}
        self._pinned_fla_cache: dict[int, object] = {}

    def allocate_state(self, capacity: int) -> Zonos3GDNState:
        if capacity < 1:
            raise ValueError("Zonos3 GDN request capacity must be positive")
        weight = self.in_proj.weight
        options = {"dtype": weight.dtype, "device": weight.device}
        return Zonos3GDNState(
            conv=torch.zeros((capacity, 3 * INNER, 4), **options),
            recurrent=torch.zeros((capacity, HEADS, HEAD_DIM, HEAD_DIM), dtype=torch.float32, device=weight.device),
            window=torch.zeros((capacity, CHUNK, self.hidden_size), **options),
            window_length=torch.zeros(capacity, dtype=torch.int32, device=weight.device),
            total_tokens=torch.zeros(capacity, dtype=torch.int64, device=weight.device),
            prefill_tokens=[0] * capacity,
        )

    def prepare_cuda(self, batch_buckets: Sequence[int]) -> None:
        if os.environ.get("FLA_TILELANG") != "0":
            raise ValueError("Zonos3 MRV2 currently requires explicit FLA_TILELANG=0 (the official Triton GDN path)")
        try:
            version = importlib.metadata.version("fla-core")
        except importlib.metadata.PackageNotFoundError as error:
            raise ImportError("Zonos3 CUDA GDN requires fla-core==0.5.2") from error
        if version != "0.5.2":
            raise ImportError(f"Zonos3 CUDA GDN requires fla-core==0.5.2, found {version}")
        from fla.modules.conv.triton.ops import causal_conv1d_fwd
        from fla.modules.fused_norm_gate import layer_norm_gated_fwd
        from fla.modules.l2norm import l2norm_fwd
        from fla.ops.common.gate import fused_beta_sigmoid_fwd
        from fla.ops.gated_delta_rule.chunk import chunk_gated_delta_rule_fwd

        self._kernels = _FLAKernels(
            causal_conv1d_fwd, layer_norm_gated_fwd, l2norm_fwd, fused_beta_sigmoid_fwd, chunk_gated_delta_rule_fwd
        )
        for batch in batch_buckets:
            self._get_tables(batch, CHUNK, self.in_proj.weight.device)

    def _get_tables(self, batch: int, length: int, device: torch.device) -> _Tables:
        key = batch, length, device
        tables = self._tables.get(key)
        if tables is None:
            cu = [i * length for i in range(batch + 1)]
            chunks = [value for i in range(batch) for j in range(-(-length // CHUNK)) for value in (i, j)]
            packed = index_to_device(cu + chunks, device, dtype=torch.int32)
            tables = _Tables(
                packed[: batch + 1],
                packed[batch + 1 :].view(-1, 2),
                index_to_device([i * length for i in range(batch)], device),
            )
            self._tables[key] = tables
        return tables

    def _pin_fla_tables(self, tables: _Tables) -> None:
        # FLA 0.5.2 memoizes cu_seqlens-derived tensors in small FIFO caches.
        # A captured graph keeps pointers, not Python references, to those
        # results. Hold them across later eager prefills/cache evictions, as
        # the official runtime does. This runs in warmup/capture/eager prefill;
        # native decode replay executes no Python here.
        identities = {id(tables.cu), id(tables.chunks)}
        for name, module in list(sys.modules.items()):
            if not name.startswith("fla") or module is None:
                continue
            for attribute in dir(module):
                closure = getattr(getattr(module, attribute, None), "__closure__", None)
                for cell in closure or ():
                    try:
                        cache = cell.cell_contents
                    except ValueError:
                        continue
                    if not isinstance(cache, deque):
                        continue
                    for entry in cache:
                        if len(entry) != 3:
                            continue
                        args, kwargs, result = entry
                        if any(id(value) in identities for value in (*args, *kwargs.values())):
                            self._pinned_fla_cache[id(result)] = result

    def _cuda_windows(
        self, x: torch.Tensor, conv: torch.Tensor, recurrent: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        kernels = self._kernels
        if kernels is None:
            raise RuntimeError("Zonos3 FLA kernels must be prepared before CUDA execution/capture")
        batch, length, width = x.shape
        tables = self._get_tables(batch, length, x.device)
        mixed = F.linear(x.reshape(batch * length, width), self.in_proj.weight).unsqueeze(0)
        qkv, z, a, b = mixed.split((3 * INNER, INNER, HEADS, HEADS), -1)
        qkv, conv_final = kernels.conv(
            x=qkv.contiguous(),
            weight=self.conv.weight.squeeze(1),
            bias=None,
            residual=None,
            initial_state=conv,
            output_final_state=True,
            activation="silu",
            cu_seqlens=tables.cu,
            chunk_indices=tables.chunks,
            BT=CHUNK,
        )
        q, k, v = [value.unflatten(-1, (HEADS, HEAD_DIM)).contiguous() for value in qkv.split(INNER, -1)]
        q, _ = kernels.l2norm(q)
        k, _ = kernels.l2norm(k)
        _, output, _, final, _, _ = kernels.delta(
            q=q,
            k=k,
            v=v,
            g=a.contiguous(),
            beta=kernels.beta(b.contiguous(), 1.0),
            scale=HEAD_DIM**-0.5,
            initial_state=recurrent,
            output_final_state=True,
            state_v_first=True,
            cu_seqlens=tables.cu,
            chunk_indices=tables.chunks,
            use_gate_in_kernel=True,
            A_log=self.A_log.float(),
            dt_bias=self.dt_bias.float(),
            chunk_size=CHUNK,
        )
        normalized = kernels.norm(
            x=output.to(q.dtype).reshape(-1, HEAD_DIM),
            g=z.reshape(-1, HEAD_DIM).contiguous(),
            weight=self.out_weight,
            bias=None,
            activation="swish",
            eps=1e-5,
            residual=None,
            residual_dtype=None,
            is_rms_norm=True,
        )[0]
        result = self.wo(normalized.reshape(batch * length, INNER)).view(batch, length, width)
        self._pin_fla_tables(tables)
        return result, conv_final, final.float()

    def _cpu_windows(
        self, x: torch.Tensor, conv: torch.Tensor, recurrent: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch, length, _ = x.shape
        qkv, z, a, b = self.in_proj(x).split((3 * INNER, INNER, HEADS, HEADS), -1)
        joined = torch.cat((conv[:, :, 1:], qkv.transpose(1, 2)), -1)
        filtered = F.silu(F.conv1d(joined, self.conv.weight, groups=3 * INNER).transpose(1, 2))
        q, k, v = [value.view(batch, length, HEADS, HEAD_DIM).float() for value in filtered.split(INNER, -1)]
        q = q * torch.rsqrt(q.square().sum(-1, keepdim=True) + 1e-6)
        k = k * torch.rsqrt(k.square().sum(-1, keepdim=True) + 1e-6)
        decay = torch.exp(-self.A_log.float().exp() * F.softplus(a.float() + self.dt_bias.float()))
        beta = b.float().sigmoid()
        state = recurrent
        outputs = []
        for token in range(length):
            state = state * decay[:, token, :, None, None]
            prediction = (state * k[:, token, :, None, :]).sum(-1)
            delta = beta[:, token, :, None] * (v[:, token] - prediction)
            state = state + delta[:, :, :, None] * k[:, token, :, None, :]
            outputs.append((state * (q[:, token] * HEAD_DIM**-0.5)[:, :, None, :]).sum(-1))
        y = torch.stack(outputs, dim=1).to(x.dtype)
        normalized = F.rms_norm(y, (HEAD_DIM,), weight=self.out_weight, eps=1e-5)
        gated = (normalized.float() * F.silu(z.view(batch, length, HEADS, HEAD_DIM).float())).to(x.dtype)
        return self.wo(gated.flatten(-2)), joined[:, :, -4:], state

    def _run_windows(
        self, x: torch.Tensor, conv: torch.Tensor, recurrent: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if x.device.type == "cuda":
            return self._cuda_windows(x, conv, recurrent)
        if x.device.type == "cpu":
            return self._cpu_windows(x, conv, recurrent)
        raise ValueError("Zonos3 GDN supports CUDA serving and CPU numerical checks")

    def forward_decode(self, x: torch.Tensor, slots: torch.Tensor, state: Zonos3GDNState) -> torch.Tensor:
        batch = x.shape[0]
        lengths = state.window_length.index_select(0, slots)
        offsets = self._get_tables(batch, CHUNK, x.device).row_offsets
        positions = offsets + lengths.long()
        window = state.window.index_select(0, slots).view(batch * CHUNK, self.hidden_size)
        window.index_copy_(0, positions, x.contiguous())
        conv = state.conv.index_select(0, slots)
        recurrent = state.recurrent.index_select(0, slots)
        output, conv_final, final = self._run_windows(window.view(batch, CHUNK, self.hidden_size), conv, recurrent)
        result = output.reshape(batch * CHUNK, self.hidden_size).index_select(0, positions)
        new_length = (lengths + 1) % CHUNK
        complete = new_length == 0
        state.conv.index_copy_(0, slots, torch.where(complete[:, None, None], conv_final, conv))
        state.recurrent.index_copy_(0, slots, torch.where(complete[:, None, None, None], final, recurrent))
        window = window.view(batch, CHUNK, self.hidden_size)
        state.window.index_copy_(0, slots, torch.where(complete[:, None, None], torch.zeros_like(window), window))
        state.window_length.index_copy_(0, slots, new_length)
        state.total_tokens.index_copy_(0, slots, state.total_tokens.index_select(0, slots) + 1)
        return result

    def forward_prefill(self, x: torch.Tensor, slot: int, previous_tokens: int, state: Zonos3GDNState) -> torch.Tensor:
        """Chunked prefill of one request; surrounding Backbone ops stay packed.

        The caller supplies the scheduler's CPU-known absolute offset. Prefill
        after AR decode/preemption is unsupported and must be rejected by the
        ModelState. No recurrent state is read back from the device.
        """
        if not 0 <= slot < len(state.prefill_tokens) or previous_tokens != state.prefill_tokens[slot] or x.shape[0] < 1:
            raise ValueError("Zonos3 GDN prefill offset is stale or its chunk is empty")
        indices = index_to_device([slot], x.device)
        old_tail = previous_tokens % CHUNK
        joined = torch.cat((state.window[slot, :old_tail], x), 0).unsqueeze(0)
        conv = state.conv.index_select(0, indices)
        recurrent = state.recurrent.index_select(0, indices)
        whole = joined.shape[1] // CHUNK * CHUNK
        parts = []
        if whole:
            output, conv, recurrent = self._run_windows(joined[:, :whole], conv, recurrent)
            parts.append(output)
            state.conv.index_copy_(0, indices, conv)
            state.recurrent.index_copy_(0, indices, recurrent)
        tail = joined[:, whole:]
        if tail.shape[1]:
            output, _, _ = self._run_windows(tail, conv, recurrent)
            parts.append(output)
        state.window.index_copy_(0, indices, F.pad(tail, (0, 0, 0, CHUNK - tail.shape[1])))
        state.window_length.index_fill_(0, indices, tail.shape[1])
        state.total_tokens.index_fill_(0, indices, previous_tokens + x.shape[0])
        state.prefill_tokens[slot] = previous_tokens + x.shape[0]
        return torch.cat(parts, 1)[0, old_tail:]


class Weight(nn.Module):
    def __init__(self, *shape: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.empty(shape), requires_grad=False)


class Zonos3RMSNorm(nn.Module):
    def __init__(self, width: int, eps: float) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(width), requires_grad=False)
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.rms_norm(x, (x.shape[-1],), weight=self.weight, eps=self.eps)


class MultiEmbedding(nn.Module):
    def __init__(self, config: Zonos3Architecture) -> None:
        super().__init__()
        self.embedders = nn.ModuleList(
            [nn.Embedding(config.audio_vocab, config.dim) for _ in range(config.n_codebooks)]
            + [nn.Embedding(config.text_rows, config.dim)]
        )

    def forward(self, codes: torch.Tensor) -> torch.Tensor:
        if codes.shape[-1] != len(self.embedders):
            raise ValueError("Zonos3 frame must include all audio columns and the text column")
        # Padding rows are checkpoint data, not implicitly zeroed embeddings.
        # Sum in codebook order, as in the reference and training model.
        output = self.embedders[0](codes[..., 0])
        for i in range(1, len(self.embedders)):
            output = output + self.embedders[i](codes[..., i])
        return output


def full_attnres_route(bank: torch.Tensor, query: torch.Tensor, eps: float) -> torch.Tensor:
    keys = F.rms_norm(bank, (bank.shape[-1],), eps=eps)
    scores = (keys * query).sum(-1, dtype=torch.float32)
    weights = scores.softmax(0).to(bank.dtype)
    return (bank * weights.unsqueeze(-1)).sum(0)


class FullAttnResWorkspace:
    """Static source storage for one graph bucket; never request history."""

    def __init__(
        self, sources: int, tokens: int, width: int, *, dtype: torch.dtype, device: torch.device | str
    ) -> None:
        self.bank = torch.empty((sources, tokens, width), dtype=dtype, device=device)

    def write(self, index: int, value: torch.Tensor) -> None:
        # Intentional in-place writes keep workspace addresses stable during
        # native FULL capture/replay. Every source is overwritten each forward.
        self.bank[index, : value.shape[0]].copy_(value)

    def route(self, sources: int, tokens: int, query: torch.Tensor, eps: float) -> torch.Tensor:
        return full_attnres_route(self.bank[:sources, :tokens], query, eps)


class MultiOutputHead(nn.Module):
    def __init__(self, config: Zonos3Architecture) -> None:
        super().__init__()
        self.weight = nn.Parameter(
            torch.empty(config.n_codebooks * config.audio_vocab, config.dim), requires_grad=False
        )
        self.n_codebooks = config.n_codebooks
        self.audio_vocab = config.audio_vocab
        self.softcap = config.loss_softcap

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        # The official GDN head pads B=1 to two GEMM rows to retain the training
        # arithmetic when a batch drains down to one request.
        rows = hidden.shape[0]
        padded = F.pad(hidden, (0, 0, 0, max(0, 2 - rows)))
        logits = F.linear(padded, self.weight)[:rows].view(rows, self.n_codebooks, self.audio_vocab)
        return self.softcap * torch.tanh(logits / self.softcap) if self.softcap > 0 else logits


class Zonos3DenseFFN(nn.Module):
    def __init__(self, config: Zonos3Architecture) -> None:
        super().__init__()
        self.w_in = Weight(2, config.ffn_hidden, config.dim)
        self.w_out = nn.Linear(config.ffn_hidden, config.dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        up, gate = F.linear(x, self.w_in.weight.flatten(0, 1)).chunk(2, dim=-1)
        return self.w_out(up * F.silu(gate))


class Zonos3QuantileRouter(nn.Module):
    def __init__(self, config: Zonos3Architecture) -> None:
        super().__init__()
        self.router_linear = nn.Linear(config.dim, config.n_experts, bias=False)
        self.rmsnorm_eda = Zonos3RMSNorm(config.dim, config.norm_eps)
        self.balancing_biases = nn.Parameter(torch.zeros(config.n_experts, dtype=torch.float32), requires_grad=False)
        self.top_k = config.moe_topk

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        logits = self.router_linear(self.rmsnorm_eda(x))
        # Quantile balancing affects selection only. The release server selects
        # k+1 then keeps k (including its tie behavior), and normalizes the
        # selected raw logits, never the bias-corrected scores.
        selected = torch.topk(logits.float() - self.balancing_biases.float(), self.top_k + 1, dim=-1).indices[
            :, : self.top_k
        ]
        weights = logits.float().gather(-1, selected).softmax(-1)
        return weights, selected.to(torch.int32)


FusedExperts = Callable[[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]


class Zonos3Experts(nn.Module):
    def __init__(self, config: Zonos3Architecture) -> None:
        super().__init__()
        self.gate_up_proj = nn.Parameter(
            torch.empty(config.n_experts, 2 * config.moe_hidden, config.moe_routed_dim), requires_grad=False
        )
        self.down_proj = nn.Parameter(
            torch.empty(config.n_experts, config.moe_routed_dim, config.moe_hidden), requires_grad=False
        )
        self._cuda_experts: FusedExperts | None = None

    def prepare_cuda(self) -> None:
        from vllm.model_executor.layers.fused_moe.fused_moe import fused_experts

        self._cuda_experts = fused_experts

    def forward(self, x: torch.Tensor, weights: torch.Tensor, selected: torch.Tensor) -> torch.Tensor:
        if x.device.type == "cuda":
            if self._cuda_experts is None:
                raise RuntimeError("Zonos3 native fused experts must be prepared before CUDA graph capture")
            return self._cuda_experts(x, self.gate_up_proj, self.down_proj, weights, selected)
        if x.device.type != "cpu":
            raise ValueError("Zonos3 experts support the CUDA serving path and CPU numerical checks")
        # CPU numerical oracle. All token/expert products are batched; rounding
        # of weighting and accumulation follows the reference's per-rank sum.
        rows, top_k = selected.shape
        ids = selected.to(torch.long).reshape(-1)
        expanded = x[:, None, :].expand(rows, top_k, x.shape[-1]).reshape(-1, x.shape[-1], 1)
        gate, up = torch.bmm(self.gate_up_proj.index_select(0, ids), expanded).squeeze(-1).chunk(2, -1)
        hidden = F.silu(gate) * up
        experts = torch.bmm(self.down_proj.index_select(0, ids), hidden.unsqueeze(-1)).squeeze(-1).view(rows, top_k, -1)
        output = torch.zeros_like(x)
        for rank in range(top_k):
            output = output + (experts[:, rank].float() * weights[:, rank, None]).to(x.dtype)
        return output


class Zonos3MoEFFN(nn.Module):
    def __init__(self, config: Zonos3Architecture) -> None:
        super().__init__()
        self.router = Zonos3QuantileRouter(config)
        self.latent_down = nn.Linear(config.dim, config.moe_routed_dim, bias=False)
        self.latent_up = nn.Linear(config.moe_routed_dim, config.dim, bias=False)
        self.experts = Zonos3Experts(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weights, selected = self.router(x)
        return self.latent_up(self.experts(self.latent_down(x), weights, selected))


def rope_interleaved(x: torch.Tensor, positions: torch.Tensor, theta: float) -> torch.Tensor:
    width = x.shape[-1]
    frequency = theta ** (-torch.arange(0, width, 2, dtype=torch.float32, device=x.device) / width)
    angles = positions.float()[:, None] * frequency[None, :]
    cos, sin = angles.cos()[:, None, :], angles.sin()[:, None, :]
    real, imaginary = x.float().reshape(*x.shape[:-1], width // 2, 2).unbind(-1)
    return torch.stack((real * cos - imaginary * sin, real * sin + imaginary * cos), -1).flatten(-2).to(x.dtype)


@dataclass
class Zonos3QKV:
    query: torch.Tensor
    key: torch.Tensor
    value: torch.Tensor
    gate: torch.Tensor


class Zonos3AttentionProjection(nn.Module):
    def __init__(self, config: Zonos3Architecture) -> None:
        super().__init__()
        self.config = config
        self.wq = nn.Linear(config.dim, config.n_heads * config.head_dim, bias=False)
        self.wkv = Weight(2, config.n_kv_heads * config.head_dim, config.dim)
        self.wo = nn.Linear(config.n_heads * config.head_dim, config.dim, bias=False)
        self.temp = nn.Parameter(torch.ones(1, config.n_heads, 1), requires_grad=False)
        self.gater = nn.Linear(config.dim, config.n_heads, bias=False)

    def project(self, x: torch.Tensor, positions: torch.Tensor) -> Zonos3QKV:
        c = self.config
        q = self.wq(x).view(-1, c.n_heads, c.head_dim)
        k, v = F.linear(x, self.wkv.weight.flatten(0, 1)).chunk(2, -1)
        k = k.view(-1, c.n_kv_heads, c.head_dim)
        q = F.rms_norm(q, (c.head_dim,), eps=1e-6) * self.temp.abs()
        k = F.rms_norm(k, (c.head_dim,), eps=1e-6)
        return Zonos3QKV(
            rope_interleaved(q, positions, c.rope_theta).flatten(-2),
            rope_interleaved(k, positions, c.rope_theta).flatten(-2),
            v.contiguous(),
            self.gater(x).sigmoid(),
        )

    def output(self, attention: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
        c = self.config
        return self.wo((attention.view(-1, c.n_heads, c.head_dim) * gate.unsqueeze(-1)).flatten(-2))


@dataclass(frozen=True)
class Zonos3PrefillSpan:
    start: int
    length: int
    slot: int
    previous_tokens: int


@dataclass
class Zonos3BackboneBatch:
    gdn_states: dict[int, Zonos3GDNState]
    workspace: FullAttnResWorkspace
    decode_slots: torch.Tensor
    decode_rows: torch.Tensor
    prefill_spans: tuple[Zonos3PrefillSpan, ...] = ()
    decode_only: bool = True


class Zonos3GlobalAttention(Zonos3AttentionProjection):
    def __init__(self, config: Zonos3Architecture, vllm_config: VllmConfig, prefix: str) -> None:
        super().__init__(config)
        from vllm.model_executor.layers.attention import Attention

        if vllm_config.quant_config is not None:
            raise ValueError("Zonos3 V1 does not support quantized Backbone execution")
        self.attn = Attention(
            config.n_heads,
            config.head_dim,
            scale=config.head_dim**-0.5,
            num_kv_heads=config.n_kv_heads,
            cache_config=vllm_config.cache_config,
            prefix=f"{prefix}.attn",
        )

    def forward(self, x: torch.Tensor, positions: torch.Tensor, batch: Zonos3BackboneBatch) -> torch.Tensor:
        del batch
        qkv = self.project(x, positions)
        return self.output(self.attn(qkv.query, qkv.key, qkv.value), qkv.gate)


AttentionFactory = Callable[[Zonos3Architecture, int], nn.Module]


class Zonos3TransformerBlock(nn.Module):
    def __init__(self, config: Zonos3Architecture, index: int, attention_factory: AttentionFactory) -> None:
        super().__init__()
        self.attention_norm = Zonos3RMSNorm(config.dim, config.norm_eps)
        self.ffn_norm = Zonos3RMSNorm(config.dim, config.norm_eps)
        self.attention = Zonos3GDN(config.dim) if config.uses_gdn(index) else attention_factory(config, index)
        self.feed_forward = Zonos3DenseFFN(config) if index == 0 else Zonos3MoEFFN(config)


class Zonos3Backbone(nn.Module):
    def __init__(self, config: Zonos3Architecture, attention_factory: AttentionFactory) -> None:
        super().__init__()
        self.config = config
        # Keep the release's flat module names, including conditioning modules.
        self.multi_embedder = MultiEmbedding(config)
        self.speaker_projection = nn.Linear(config.speaker_embedding_dim, config.dim, bias=True)
        self.condition_embeddings = nn.ModuleList(
            [nn.Embedding(count + 1, config.dim) for count in config.condition_bucket_counts]
        )
        self.layers = nn.ModuleList(
            [Zonos3TransformerBlock(config, i, attention_factory) for i in range(config.n_layers)]
        )
        self.block_attnres_queries = nn.Parameter(torch.empty(config.n_sources, config.dim), requires_grad=False)
        self.out_norm = Zonos3RMSNorm(config.dim, config.norm_eps)
        self.multi_output = MultiOutputHead(config)

    def allocate_gdn_states(self, capacity: int) -> dict[int, Zonos3GDNState]:
        return {
            i: layer.attention.allocate_state(capacity)
            for i, layer in enumerate(self.layers)
            if isinstance(layer.attention, Zonos3GDN)
        }

    def allocate_workspace(self, tokens: int) -> FullAttnResWorkspace:
        weight = self.block_attnres_queries
        return FullAttnResWorkspace(
            self.config.n_sources, tokens, self.config.dim, dtype=weight.dtype, device=weight.device
        )

    def prepare_cuda(self, batch_buckets: Sequence[int]) -> None:
        for layer in self.layers:
            if isinstance(layer.attention, Zonos3GDN):
                layer.attention.prepare_cuda(batch_buckets)
            if isinstance(layer.feed_forward, Zonos3MoEFFN):
                layer.feed_forward.experts.prepare_cuda()

    def load_release_state(self, weights: Mapping[str, torch.Tensor]) -> set[str]:
        # model.pth is the release's flat canonical state dict. Native
        # Attention adds runtime scale/cache buffers that are not checkpoint
        # weights; every actual parameter must still be present exactly once.
        expected = set(dict(self.named_parameters()))
        missing, unknown = expected - set(weights), set(weights) - expected
        if missing or unknown:
            raise ValueError(
                f"Zonos3 AR checkpoint keys differ: missing={sorted(missing)[:8]}, unknown={sorted(unknown)[:8]}"
            )
        result = self.load_state_dict(dict(weights), strict=False)
        if result.unexpected_keys or any(key in expected for key in result.missing_keys):
            raise RuntimeError("Zonos3 AR parameter loading was incomplete")
        self.eval().requires_grad_(False)
        return expected

    @staticmethod
    def _gdn_forward(
        module: Zonos3GDN, x: torch.Tensor, batch: Zonos3BackboneBatch, state: Zonos3GDNState
    ) -> torch.Tensor:
        if batch.decode_only:
            return module.forward_decode(x, batch.decode_slots, state)
        output = torch.zeros_like(x)
        if batch.decode_rows.numel():
            decoded = module.forward_decode(x.index_select(0, batch.decode_rows), batch.decode_slots, state)
            output.index_copy_(0, batch.decode_rows, decoded)
        for span in batch.prefill_spans:
            end = span.start + span.length
            prefilling = module.forward_prefill(x[span.start : end], span.slot, span.previous_tokens, state)
            output[span.start : end].copy_(prefilling)
        return output

    def forward(self, inputs_embeds: torch.Tensor, positions: torch.Tensor, batch: Zonos3BackboneBatch) -> torch.Tensor:
        config = self.config
        tokens = inputs_embeds.shape[0]
        workspace = batch.workspace
        if workspace.bank.shape[1] < tokens or positions.shape[0] != tokens:
            raise ValueError("Zonos3 Backbone workspace/position shape does not match the scheduled rows")
        # Speaker/condition substitution happens in ModelState before this norm.
        x = F.rms_norm(inputs_embeds, (config.dim,), eps=config.norm_eps)
        workspace.write(0, x)
        for i, layer in enumerate(self.layers):
            mixed = workspace.route(2 * i + 1, tokens, self.block_attnres_queries[2 * i], config.norm_eps)
            attention_input = layer.attention_norm(mixed)
            if isinstance(layer.attention, Zonos3GDN):
                attention_output = self._gdn_forward(layer.attention, attention_input, batch, batch.gdn_states[i])
            else:
                attention_output = layer.attention(attention_input, positions, batch)
            workspace.write(2 * i + 1, attention_output)
            mixed = workspace.route(2 * i + 2, tokens, self.block_attnres_queries[2 * i + 1], config.norm_eps)
            workspace.write(2 * i + 2, layer.feed_forward(layer.ffn_norm(mixed)))
        mixed = workspace.route(config.n_sources, tokens, self.block_attnres_queries[-1], config.norm_eps)
        return self.out_norm(mixed)


def native_attention_factory(vllm_config: VllmConfig, prefix: str = "") -> AttentionFactory:
    validate_zonos3_ar_runtime(vllm_config)

    def construct(config: Zonos3Architecture, index: int) -> nn.Module:
        name = f"layers.{index}.attention"
        return Zonos3GlobalAttention(config, vllm_config, f"{prefix}.{name}" if prefix else name)

    return construct


def validate_zonos3_ar_runtime(vllm_config: VllmConfig) -> None:
    """Reject configurations whose state/graph semantics V1 cannot preserve."""
    if vllm_config.cache_config.enable_prefix_caching:
        raise ValueError(
            "Zonos3 V1 requires prefix caching disabled: KV-only hits cannot restore GDN conv/recurrent state"
        )
    if (
        vllm_config.model_config.enforce_eager
        or vllm_config.compilation_config.cudagraph_mode is not CUDAGraphMode.FULL_DECODE_ONLY
    ):
        raise ValueError("Zonos3 AR requires FULL_DECODE_ONLY: native FULL decode graphs and eager prefill")
    parallel = vllm_config.parallel_config
    if parallel.tensor_parallel_size != 1 or parallel.pipeline_parallel_size != 1 or parallel.enable_expert_parallel:
        raise ValueError("Zonos3 V1 supports one tensor/pipeline rank per stage without expert parallelism")


class Zonos3ForCausalLM(Zonos3Backbone):
    has_preprocess = True
    has_postprocess = False
    have_multimodal_outputs = False
    requires_raw_input_tokens = False
    omni_pooler_payload_include_hidden = False

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        model_config = vllm_config.model_config
        reference = model_config.model
        subdir = getattr(model_config.hf_config, "zonos3_lm_subdir", "lm/zonos3_mini")
        root = Path(reference)
        if not root.is_dir():
            root = Path(
                download_weights_from_hf_specific(
                    reference,
                    vllm_config.load_config.download_dir,
                    [f"{subdir}/params.json"],
                    revision=model_config.revision,
                    require_all=True,
                )
            )
        direct = (root / "params.json").is_file()
        directory = root if direct else root / subdir
        with (directory / "params.json").open() as handle:
            architecture = Zonos3Architecture.from_params(json.load(handle))
        super().__init__(architecture, native_attention_factory(vllm_config, prefix))
        self.vllm_config = vllm_config
        self.model_dir = directory
        self.allow_patterns_overrides = ["model.pth" if direct else f"{subdir}/model.pth"]
        self._weights_loaded = False
        self.speaker_encoder = None

    def create_mrv2_model_state(
        self, vllm_config: VllmConfig, encoder_cache: EncoderCache | None, device: torch.device
    ) -> Zonos3ARModelState:
        from .model_state import Zonos3ARModelState

        return Zonos3ARModelState(vllm_config, self, encoder_cache, device)

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        # Logical placeholders carry length only. run_preprocess writes all
        # actual prefill/decode embeddings into the native static input buffer.
        return torch.zeros(
            (*input_ids.shape, self.config.dim), dtype=self.multi_output.weight.dtype, device=input_ids.device
        )

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # The upstream startup/prompt-logprob plumbing sees the logical control
        # vocabulary only. Real request logprobs are rejected by ModelState.
        # MultiOutputHead runs in ModelState.postprocess_model_output().
        return torch.zeros((hidden_states.shape[0], 3), dtype=hidden_states.dtype, device=hidden_states.device)

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        positions: torch.Tensor | None = None,
        intermediate_tensors: IntermediateTensors | None = None,
        *,
        inputs_embeds: torch.Tensor | None = None,
        zonos3_batch: Zonos3BackboneBatch | None = None,
        model_intermediate_buffer: list[dict[str, Any]] | None = None,
        runtime_additional_information: list[dict[str, Any]] | None = None,
        seq_token_counts: list[int] | None = None,
        sampling_metadata: object = None,
        logits_index: torch.Tensor | None = None,
        sampler: object = None,
    ) -> torch.Tensor:
        del (
            input_ids,
            intermediate_tensors,
            model_intermediate_buffer,
            runtime_additional_information,
            seq_token_counts,
            sampling_metadata,
            logits_index,
            sampler,
        )
        if not self._weights_loaded:
            raise RuntimeError("Zonos3 AR release weights have not been loaded")
        if inputs_embeds is None or positions is None or zonos3_batch is None:
            raise ValueError("Zonos3 AR requires native ModelState-prepared embeddings, positions and state")
        return super().forward(inputs_embeds, positions, zonos3_batch)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        if self._weights_loaded:
            raise RuntimeError("Zonos3 AR release weights are already loaded")
        loaded = self.load_release_state(dict(weights))
        if self.config.speaker_embedding_codec == "ownmect_s2":
            from .speaker_encoder import Zonos3SpeakerEncoder

            relative = "speaker_encoders/ownmect_s2/ckpt_0037.pth"
            root = Path(self.vllm_config.model_config.model)
            checkpoint = root / relative
            if not checkpoint.is_file():
                if root.is_dir():
                    raise FileNotFoundError(f"Zonos3 speaker encoder checkpoint missing: {checkpoint}")
                folder = download_weights_from_hf_specific(
                    str(root),
                    self.vllm_config.load_config.download_dir,
                    [relative],
                    revision=self.vllm_config.model_config.revision,
                    require_all=True,
                )
                checkpoint = Path(folder) / relative
            with torch.device("cpu"):
                self.speaker_encoder = Zonos3SpeakerEncoder(checkpoint)
            self.speaker_encoder.to(device=next(self.multi_output.parameters()).device)
            # This companion checkpoint is loaded strictly by its constructor,
            # separately from the AR checkpoint. Report its parameters too so
            # the upstream loader can account for the entire registered model.
            loaded.update(f"speaker_encoder.{name}" for name, _ in self.speaker_encoder.named_parameters())
        if next(self.parameters()).device.type == "cuda":
            buckets = self.vllm_config.compilation_config.cudagraph_capture_sizes or [
                self.vllm_config.scheduler_config.max_num_seqs
            ]
            self.prepare_cuda([size for size in buckets if size <= self.vllm_config.scheduler_config.max_num_seqs])
        self._weights_loaded = True
        return loaded
