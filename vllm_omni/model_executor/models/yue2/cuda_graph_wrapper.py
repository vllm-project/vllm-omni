# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Persistent B=1 prefill graphs over page-aligned, contiguous KV slots.

Paged FlashAttention reads a slot through an identity page table. The same
storage has a continuous token view for the existing NAR solver. Slots are
leased until the solver's GPU work completes, rather than until prefill ends.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING
from weakref import WeakKeyDictionary

import torch
from vllm.logger import init_logger
from vllm.triton_utils import HAS_TRITON, tl, triton

if TYPE_CHECKING:
    from .yue2 import Yue2ForCausalLM

from .kv_slot_pool import PAGE_SIZE, KVSlot, PrefillKVLease, round_token_capacity

logger = init_logger(__name__)


if HAS_TRITON:

    @triton.jit
    def _write_condition_kv(
        key,
        value,
        keys,
        values,
        metadata,
        q_capacity: tl.constexpr,
        heads: tl.constexpr,
        head_dim: tl.constexpr,
        key_strides: tl.constexpr,
        value_strides: tl.constexpr,
        tile: tl.constexpr,
    ):
        offsets = tl.program_id(0) * tile + tl.arange(0, tile)
        token = offsets // (heads * head_dim)
        head = offsets // head_dim % heads
        dim = offsets % head_dim
        length = tl.load(metadata + 1)
        start = tl.load(metadata + 3)
        mask = (token < length) & (token < q_capacity)
        target = ((start + token) * heads + head) * head_dim + dim
        k = tl.load(key + token * key_strides[0] + head * key_strides[1] + dim * key_strides[2], mask, other=0)
        v = tl.load(value + token * value_strides[0] + head * value_strides[1] + dim * value_strides[2], mask, other=0)
        tl.store(keys + target, k, mask)
        tl.store(values + target, v, mask)


@dataclass
class _GraphState:
    ids: torch.Tensor
    metadata: torch.Tensor  # [0, actual_q_length, actual_kv_length, first_position]
    host_ids: torch.Tensor
    host_metadata: torch.Tensor
    offsets: torch.Tensor
    output: torch.Tensor
    graph: torch.cuda.CUDAGraph | None = None


class PrefillGraphWrapper:
    """Cache prefill graphs without owning or allocating KV slots.

    Weak slot keys discard graph states when no pool/request retains the slot.
    Callers keep the lease through all GPU consumers before returning storage.
    """

    def __init__(self, model: Yue2ForCausalLM, fa_version: int):
        self.model = model
        self.fa_version = fa_version
        weight = next(model.vae2llm.parameters())
        self.device, self.dtype = weight.device, weight.dtype
        self.maximum = model.max_position_embeddings
        self.graphs: WeakKeyDictionary[KVSlot, dict[int, _GraphState]] = WeakKeyDictionary()

    @classmethod
    def create(cls, model: Yue2ForCausalLM) -> PrefillGraphWrapper | None:
        weight = next(model.vae2llm.parameters())
        if not HAS_TRITON or weight.device.type != "cuda" or weight.dtype not in (torch.float16, torch.bfloat16):
            return None
        from vllm.vllm_flash_attn import flash_attn_interface

        if not flash_attn_interface.is_fa_version_supported(2):
            logger.warning("YuE2 prefill graphs unavailable: FlashAttention 2 is not supported")
            return None
        logger.info("YuE2 B=1 prefill graphs enabled: FA2")
        return cls(model, fa_version=2)

    def _state(self, capacity: int) -> _GraphState:
        heads = int(self.model.config.num_attention_heads)
        dim = int(self.model.config.head_dim)
        return _GraphState(
            ids=torch.empty(capacity, dtype=torch.long, device=self.device),
            metadata=torch.empty(4, dtype=torch.int32, device=self.device),
            host_ids=torch.empty(capacity, dtype=torch.long, pin_memory=True),
            host_metadata=torch.empty(4, dtype=torch.int32, pin_memory=True),
            offsets=torch.arange(capacity, dtype=torch.long, device=self.device),
            output=torch.empty((capacity, heads, dim), dtype=self.dtype, device=self.device),
        )

    def _forward(self, slot: KVSlot, state: _GraphState) -> None:
        from vllm.vllm_flash_attn import flash_attn_varlen_func

        from .yue2 import _linear

        model = self.model
        capacity = state.ids.numel()
        positions = (state.offsets + state.metadata[3]).clamp(max=self.maximum - 1)
        cos, sin = model.rotary(positions[None])
        x = model.model.embed_tokens(state.ids)
        kv_heads, dim = slot.keys.shape[2:]
        last = len(model.model.layers) - 1
        for index, layer in enumerate(model.model.layers):
            q, k, v = model.ar_project(layer, layer.input_layernorm(x), cos, sin)
            _write_condition_kv[(triton.cdiv(capacity * kv_heads * dim, 256),)](
                k,
                v,
                slot.keys[index],
                slot.values[index],
                state.metadata,
                q_capacity=capacity,
                heads=kv_heads,
                head_dim=dim,
                key_strides=k.stride(),
                value_strides=v.stride(),
                tile=256,
            )
            if index == last:
                break
            # Unused Q rows stay zero and do not write to KV. All valid queries
            # use FA's bottom-right causal alignment, including the one END query.
            state.output.zero_()
            h = flash_attn_varlen_func(
                q,
                slot.keys[index].view(-1, PAGE_SIZE, kv_heads, dim),
                slot.values[index].view(-1, PAGE_SIZE, kv_heads, dim),
                max_seqlen_q=capacity,
                cu_seqlens_q=state.metadata[:2],
                max_seqlen_k=slot.capacity,
                seqused_k=state.metadata[2:3],
                block_table=slot.block_table,
                causal=True,
                fa_version=self.fa_version,
                out=state.output,
            )
            x = x + _linear(layer.self_attn.o_proj, h.flatten(1))
            x = x + _linear(layer.mlp, layer.post_attention_layernorm(x))

    @torch.inference_mode()
    def prefill(self, lease: PrefillKVLease, tokens: list[int], start: int = 0) -> None:
        slot = lease.slot
        if lease.returned or not tokens or start < 0 or start + len(tokens) > slot.capacity:
            raise ValueError("Invalid prefill input or returned KV lease")
        capacity = round_token_capacity(len(tokens), self.maximum)
        graphs = self.graphs.setdefault(slot, {})
        state = graphs.get(capacity)
        if state is None:
            # Bound retained graph workspaces. Slot reuse only happens after
            # its prior NAR consumer completes, so an old bucket is safe to drop.
            if len(graphs) >= 4:
                del graphs[next(iter(graphs))]
            state = self._state(capacity)
            graphs[capacity] = state
        state.host_ids.zero_()
        state.host_ids[: len(tokens)].copy_(torch.tensor(tokens, dtype=torch.long))
        state.host_metadata.copy_(torch.tensor([0, len(tokens), start + len(tokens), start], dtype=torch.int32))
        state.ids.copy_(state.host_ids, non_blocking=True)
        state.metadata.copy_(state.host_metadata, non_blocking=True)
        if state.graph is None:
            # Compile/lazily initialize operators before capture.
            self._forward(slot, state)
            graph = torch.cuda.CUDAGraph()
            graph.capture_begin(capture_error_mode="thread_local")
            try:
                self._forward(slot, state)
            finally:
                graph.capture_end()
            state.graph = graph
            logger.info("YuE2 captured B=1 prefill graph: Q capacity=%d, KV capacity=%d", capacity, slot.capacity)
        with torch.profiler.record_function("yue2.prefill_graph_replay"):
            state.graph.replay()
