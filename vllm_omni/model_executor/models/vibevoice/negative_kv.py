# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Model-owned fixed-slot negative KV pool for VibeVoice's CFG branch.

The negative Qwen branch shares weights with the positive backbone but keeps
independent KV, positions and progress.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import torch
from vllm.forward_context import create_forward_context, override_forward_context
from vllm.logger import init_logger
from vllm.model_executor.models.qwen2 import Qwen2Model
from vllm.sequence import IntermediateTensors
from vllm.v1.attention.backend import CommonAttentionMetadata
from vllm.v1.kv_cache_interface import FullAttentionSpec

from vllm_omni.model_executor.models.model_local_kv import (
    ModelLocalKVScope,
    ModelLocalKVSpec,
    RowDriver,
    spec_from_hf_config,
)

if TYPE_CHECKING:
    from vllm.config import VllmConfig

logger = init_logger(__name__)


@dataclass(slots=True)
class _SlotState:
    slot_id: int
    length: int = 0
    owner: str | None = None


class VibeVoiceNegativeKV:
    """Fixed request-slot paged KV pool executed with the shared Qwen weights.

    Layout per layer: ``[S*K, num_kv_heads, block_size, 2*head_dim]`` where
    ``S = max_num_seqs`` request slots each own ``K = ceil(L / block_size)``
    consecutive pages and ``L = max_model_len``. Slot ``s`` uses pages
    ``s*K .. s*K+K-1``; no dynamic allocation, hashing or sharing.
    """

    def __init__(self, *, language_model: Qwen2Model, vllm_config: VllmConfig) -> None:
        self._language_model = language_model
        self._vllm_config = vllm_config
        text_config = vllm_config.model_config.hf_config.get_text_config()
        self._hidden_size = int(text_config.hidden_size)
        self._num_layers = int(text_config.num_hidden_layers)
        self._num_kv_heads = int(text_config.num_key_value_heads)
        self._head_dim = int(
            getattr(text_config, "head_dim", None) or (text_config.hidden_size // text_config.num_attention_heads)
        )
        self._block_size = int(vllm_config.cache_config.block_size)
        self._max_seq_len = int(vllm_config.model_config.max_model_len)
        self._max_slots = int(vllm_config.scheduler_config.max_num_seqs)
        if not 1 <= self._max_slots <= 4:
            raise ValueError("VibeVoice negative KV requires 1 <= max_num_seqs <= 4.")
        self._blocks_per_slot = math.ceil(self._max_seq_len / self._block_size)
        self._slots: list[_SlotState] = [_SlotState(slot_id=i) for i in range(self._max_slots)]
        self._by_request: dict[str, int] = {}
        self._attention_layers = [layer.self_attn.attn for layer in language_model.layers]
        if len(self._attention_layers) != self._num_layers:
            raise RuntimeError(
                f"VibeVoice negative KV found {len(self._attention_layers)} attention layers, "
                f"expected {self._num_layers}."
            )
        self._kv_caches: list[torch.Tensor] | None = None
        self._block_tables: torch.Tensor | None = None  # [S, K] int32, one row per slot
        self._metadata_builder: Any = None
        self._device: torch.device | None = None
        self._dtype: torch.dtype | None = None

    # ------------------------------------------------------------------
    # allocation / reporting
    # ------------------------------------------------------------------
    def allocate(self, *, device: torch.device, dtype: torch.dtype) -> None:
        """Allocate the physical pool once the model device/dtype are known."""
        if self._kv_caches is not None:
            return
        if device.type != "cuda":
            raise ValueError("VibeVoice negative KV pool currently requires a CUDA device.")
        from vllm.v1.attention.backends.flash_attn import FlashAttentionBackend

        if dtype != torch.bfloat16:
            raise ValueError("VibeVoice negative KV currently supports BF16 only.")
        if any(
            layer.get_attn_backend() is not FlashAttentionBackend or layer.impl.vllm_flash_attn_version != 3
            for layer in self._attention_layers
        ):
            raise ValueError("VibeVoice negative KV requires the validated FlashAttention 3 backend.")
        if not FlashAttentionBackend.supports_block_size(self._block_size):
            raise ValueError(f"Unsupported VibeVoice negative KV page size: {self._block_size}.")
        spec = FullAttentionSpec(
            block_size=self._block_size,
            num_kv_heads=self._num_kv_heads,
            head_size=self._head_dim,
            dtype=dtype,
        )
        layer_names = [layer.layer_name for layer in self._attention_layers]
        builder = FlashAttentionBackend.get_builder_cls()(spec, layer_names, self._vllm_config, device)
        total_blocks = self._max_slots * self._blocks_per_slot
        shape = (total_blocks, self._num_kv_heads, self._block_size, 2 * self._head_dim)
        caches = [torch.empty(shape, dtype=dtype, device=device) for _ in range(self._num_layers)]
        tables = torch.arange(total_blocks, dtype=torch.int32, device=device).reshape(
            self._max_slots, self._blocks_per_slot
        )
        # Publish only after the entire pool and its metadata are initialized.
        self._kv_caches, self._block_tables = caches, tables
        self._metadata_builder = builder
        self._device, self._dtype = caches[0].device, dtype
        logger.info(
            "Allocated VibeVoice negative KV pool: layers=%d slots=%d blocks_per_slot=%d block_size=%d bytes=%.2f GiB",
            self._num_layers,
            self._max_slots,
            self._blocks_per_slot,
            self._block_size,
            self.model_local_kv_specs()[0].peak_bytes(self._max_slots) / 1024**3,
        )

    def model_local_kv_specs(self) -> Sequence[ModelLocalKVSpec]:
        text_config = self._vllm_config.model_config.hf_config.get_text_config()
        return [
            spec_from_hf_config(
                text_config,
                name="vibevoice_negative_kv_pool",
                physical_capacity_positions=self._blocks_per_slot * self._block_size,
                capacity_source="fixed per-request slot: ceil(max_model_len / block_size) pages",
                scope=ModelLocalKVScope.MODEL,
                dtype=self._dtype or torch.bfloat16,
                rows=RowDriver.MAX_NUM_SEQS,
                allocation_note="one physical pool allocated at load; slots are page ranges",
            )
        ]

    # ------------------------------------------------------------------
    # request lifecycle
    # ------------------------------------------------------------------
    def acquire(self, request_id: str) -> None:
        if request_id in self._by_request:
            return
        free = next((slot for slot in self._slots if slot.owner is None), None)
        if free is None:
            raise RuntimeError(f"VibeVoice negative KV pool exhausted: all {self._max_slots} slots in use.")
        free.owner = request_id
        free.length = 0
        self._by_request[request_id] = free.slot_id

    def reset_segment(self, request_id: str) -> None:
        slot_id = self._by_request.get(request_id)
        if slot_id is not None:
            self._slots[slot_id].length = 0

    def release(self, request_id: str) -> None:
        slot_id = self._by_request.pop(request_id, None)
        if slot_id is not None:
            self._slots[slot_id].owner = None
            self._slots[slot_id].length = 0

    def sequence_length(self, request_id: str) -> int:
        slot_id = self._by_request.get(request_id)
        return 0 if slot_id is None else self._slots[slot_id].length

    # ------------------------------------------------------------------
    # execution
    # ------------------------------------------------------------------
    def forward_step(
        self,
        request_ids: list[str],
        input_embeddings: list[torch.Tensor],
    ) -> list[torch.Tensor]:
        """Advance each request by one negative-branch token (eager, batched).

        Temporarily binds the pool's KV tensors to the shared attention layers
        and restores the originals in ``finally``.
        """
        if self._kv_caches is None or self._block_tables is None or self._device is None:
            raise RuntimeError("VibeVoice negative KV pool is not allocated yet.")
        if not request_ids or len(request_ids) != len(input_embeddings):
            raise ValueError("VibeVoice negative KV batch must be non-empty and aligned.")
        if len(request_ids) != len(set(request_ids)):
            raise ValueError("VibeVoice negative KV batch contains duplicate request IDs.")

        device = self._device
        for request_id, embedding in zip(request_ids, input_embeddings, strict=True):
            if request_id not in self._by_request:
                raise RuntimeError(f"VibeVoice negative KV request {request_id!r} has no slot.")
            if (
                embedding.shape != (1, self._hidden_size)
                or embedding.device != device
                or embedding.dtype != self._dtype
            ):
                raise ValueError("VibeVoice negative KV embeddings must match the pool's shape, device and dtype.")
            if self.sequence_length(request_id) >= self._max_seq_len:
                raise RuntimeError(f"VibeVoice negative KV request {request_id!r} exceeded capacity.")
        batch = len(request_ids)
        positions: list[int] = []
        seq_lens: list[int] = []
        slot_values: list[int] = []
        block_rows: list[torch.Tensor] = []
        touched: list[str] = []
        try:
            for request_id in request_ids:
                slot_id = self._by_request[request_id]
                slot = self._slots[slot_id]
                position = slot.length
                block_index = position // self._block_size
                slot_values.append(
                    (slot_id * self._blocks_per_slot + block_index) * self._block_size + position % self._block_size
                )
                positions.append(position)
                slot.length = position + 1
                seq_lens.append(slot.length)
                block_rows.append(self._block_tables[slot_id])
                touched.append(request_id)

            inputs_embeds = torch.cat(input_embeddings, dim=0)
            position_tensor = torch.tensor(positions, dtype=torch.long, device=device)
            slot_mapping = torch.tensor(slot_values, dtype=torch.int64, device=device)
            seq_lens_gpu = torch.tensor(seq_lens, dtype=torch.int32, device=device)
            query_start_cpu = torch.arange(0, batch + 1, dtype=torch.int32)
            query_start = query_start_cpu.to(device)
            block_table = torch.stack(block_rows, dim=0)

            common = CommonAttentionMetadata(
                query_start_loc=query_start,
                query_start_loc_cpu=query_start_cpu,
                seq_lens=seq_lens_gpu,
                num_reqs=batch,
                num_actual_tokens=batch,
                max_query_len=1,
                max_seq_len=max(seq_lens),
                block_table_tensor=block_table,
                slot_mapping=slot_mapping,
                causal=True,
                positions=position_tensor,
            )
            layer_names = [layer.layer_name for layer in self._attention_layers]
            metadata = self._metadata_builder.build(0, common)
            context = create_forward_context(
                {name: metadata for name in layer_names},
                self._vllm_config,
                slot_mapping={name: slot_mapping for name in layer_names},
                skip_compiled=True,
            )

            originals = [layer.kv_cache for layer in self._attention_layers]
            try:
                for layer, kv in zip(self._attention_layers, self._kv_caches, strict=True):
                    layer.kv_cache = kv
                with override_forward_context(context):
                    hidden = self._language_model(
                        input_ids=None,
                        positions=position_tensor,
                        inputs_embeds=inputs_embeds,
                    )
                if isinstance(hidden, IntermediateTensors):
                    raise RuntimeError("VibeVoice negative Qwen requires PP=1.")
                if not isinstance(hidden, torch.Tensor) or hidden.shape != (batch, self._hidden_size):
                    raise ValueError("VibeVoice negative Qwen returned an invalid hidden-state tensor.")
                owned = hidden.detach().clone(memory_format=torch.contiguous_format)
            finally:
                for layer, original in zip(self._attention_layers, originals, strict=True):
                    layer.kv_cache = original
        except Exception:
            # A partial layer write cannot be rolled back safely; drop every
            # touched request's negative state so stale KV is never reused.
            for request_id in touched:
                self.release(request_id)
            raise

        return [row.reshape(1, self._hidden_size) for row in owned.unbind(0)]


__all__ = ["VibeVoiceNegativeKV"]
