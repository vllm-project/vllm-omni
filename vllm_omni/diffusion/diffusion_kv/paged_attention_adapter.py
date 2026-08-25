# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field, replace
from typing import Any

import torch
from vllm.config import VllmConfig, set_current_vllm_config
from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase
from vllm.utils.math_utils import cdiv
from vllm.v1.attention.backend import AttentionType
from vllm.v1.attention.backends.registry import AttentionBackendEnum
from vllm.v1.attention.backends.utils import get_dcp_local_seq_lens
from vllm.v1.attention.selector import get_attn_backend
from vllm.v1.kv_cache_interface import AttentionSpec, KVCacheConfig
from vllm.v1.worker.gpu.attn_utils import build_slot_mappings_by_layer
from vllm.v1.worker.gpu.block_table import BlockTables

from vllm_omni.diffusion.attention.backends.utils.piecewise_attn import (
    PagedPiecewisePlan,
    build_paged_piecewise_plan,
)
from vllm_omni.diffusion.forward_context import override_paged_kv_adapter
from vllm_omni.platforms import current_omni_platform


@dataclass(frozen=True)
class DiffusionPagedAttentionRowBinding:
    """Worker row and logical length installed for one allocation identity."""

    row_index: int
    max_seq_len: int


DiffusionKVRowResolver = Callable[
    [str, int | None, str | None],
    DiffusionPagedAttentionRowBinding,
]


class DiffusionPagedAttentionLayerAdapter(AttentionLayerBase):
    """Register a diffusion layer with vLLM's native cache machinery.

    This object deliberately does *not* subclass ``vllm.Attention``.  The
    latter owns a second execution path and would bypass Omni's sequence
    parallel pre/post hooks.  The wrapper only supplies the small
    ``AttentionLayerBase`` contract needed by ``init_attn_backend`` and keeps
    the platform-native attention implementation/cache view available to the
    diffusion adapter.
    """

    def __init__(
        self,
        *,
        layer_name: str,
        layer: Any,
        spec: AttentionSpec,
        vllm_config: VllmConfig,
        device: torch.device,
        ulysses_degree: int = 1,
    ) -> None:
        if type(ulysses_degree) is not int or ulysses_degree <= 0:
            raise ValueError(f"ulysses_degree must be a positive integer, got {ulysses_degree!r}")
        num_heads = int(layer.num_heads)
        num_kv_heads = int(spec.num_kv_heads)
        if num_heads <= 0 or num_kv_heads <= 0 or num_heads % num_kv_heads != 0:
            raise ValueError(
                "Paged attention requires positive Q/KV heads with num_heads divisible by num_kv_heads: "
                f"num_heads={num_heads}, num_kv_heads={num_kv_heads}"
            )
        if num_heads % ulysses_degree != 0 or num_kv_heads % ulysses_degree != 0:
            raise ValueError(
                "Paged attention requires Q/KV heads divisible by ulysses_degree: "
                f"num_heads={num_heads}, num_kv_heads={num_kv_heads}, ulysses_degree={ulysses_degree}"
            )
        num_heads //= ulysses_degree
        num_kv_heads //= ulysses_degree

        attention_config = vllm_config.attention_config
        parallel_config = vllm_config.parallel_config
        previous_backend = attention_config.backend
        previous_backend_per_kind = attention_config.backend_per_kind
        previous_pcp_size = parallel_config.prefill_context_parallel_size
        try:
            # This is a portable vLLM backend request. The active platform
            # resolves it to its native implementation, such as FlashAttention
            # on CUDA or AscendAttentionBackend on NPU.
            attention_config.backend = AttentionBackendEnum.FLASH_ATTN
            attention_config.backend_per_kind = {}
            # Omni maps strict Ulysses onto vLLM's PCP group for expert
            # collectives. Paged attention runs after the Ulysses all-to-all,
            # where every rank owns the full sequence and rank-local heads, so
            # native attention must select a non-PCP kernel.
            if ulysses_degree > 1:
                parallel_config.prefill_context_parallel_size = 1
            with set_current_vllm_config(vllm_config):
                attn_backend = get_attn_backend(
                    head_size=spec.head_size,
                    dtype=vllm_config.model_config.dtype,
                    kv_cache_dtype=vllm_config.cache_config.cache_dtype,
                    attn_type=AttentionType.DECODER,
                    num_heads=num_heads,
                )
        finally:
            attention_config.backend = previous_backend
            attention_config.backend_per_kind = previous_backend_per_kind
            parallel_config.prefill_context_parallel_size = previous_pcp_size
        canonical_spec = replace(
            spec,
            num_kv_heads=num_kv_heads,
            indexes_kv_by_block_stride=attn_backend.indexes_kv_by_block_stride(),
        )
        self.layer_name = layer_name
        self.spec = canonical_spec
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_size = int(canonical_spec.head_size)
        self.head_size_v = int(getattr(canonical_spec, "head_size_v", canonical_spec.head_size))
        if self.head_size_v != self.head_size:
            raise ValueError(
                "Diffusion native paged attention requires equal key/value head sizes; "
                f"got head_size={self.head_size}, head_size_v={self.head_size_v}"
            )
        self.softmax_scale = float(layer.softmax_scale)
        self.attn_backend = attn_backend
        self.kv_cache: torch.Tensor | None = None
        # Native attention backends consume either device tensors or host
        # scalar copies of these scales. The paged diffusion path is currently
        # unquantized, so initialize both contracts to identity.
        self._q_scale = torch.ones(1, device=device, dtype=torch.float32)
        self._k_scale = torch.ones(1, device=device, dtype=torch.float32)
        self._v_scale = torch.ones(1, device=device, dtype=torch.float32)
        self._q_scale_float = 1.0
        self._k_scale_float = 1.0
        self._v_scale_float = 1.0
        with set_current_vllm_config(vllm_config):
            self.impl = self._create_native_impl(vllm_config)

    def get_attn_backend(self):
        return self.attn_backend

    def _create_native_impl(self, vllm_config: VllmConfig):
        impl_cls = self.attn_backend.get_impl_cls()
        impl = impl_cls(
            num_heads=self.num_heads,
            head_size=self.head_size,
            scale=self.softmax_scale,
            num_kv_heads=self.num_kv_heads,
            alibi_slopes=None,
            sliding_window=None,
            kv_cache_dtype=vllm_config.cache_config.cache_dtype,
            logits_soft_cap=None,
            attn_type=AttentionType.DECODER,
            kv_sharing_target_layer_name=None,
        )
        if not callable(getattr(impl, "forward", None)):
            raise RuntimeError(
                f"Native paged-attention backend {self.attn_backend.get_name()!r} has no forward implementation"
            )
        if not self.attn_backend.forward_includes_kv_cache_update and not callable(
            getattr(impl, "do_kv_cache_update", None)
        ):
            raise RuntimeError(
                f"Native paged-attention backend {self.attn_backend.get_name()!r} cannot update the KV cache"
            )
        if hasattr(impl, "vllm_flash_attn_version") and impl.vllm_flash_attn_version is None:
            raise RuntimeError(f"Native paged-attention kernel is unavailable for diffusion layer {self.layer_name!r}")
        return impl

    def get_kv_cache_spec(self, vllm_config: VllmConfig) -> AttentionSpec:
        del vllm_config
        return self.spec


@dataclass(frozen=True)
class DiffusionPagedAttentionRow:
    """One logical BlockTable row participating in a paged attention call."""

    request_id: str
    query_len: int
    seq_len: int
    kv_start_pos: int = 0
    imported_prefix_len: int = 0
    sequence_id: int | None = None
    context_id: str | None = None

    def __post_init__(self) -> None:
        if type(self.request_id) is not str or not self.request_id:
            raise ValueError("Paged attention request_id must be a non-empty string")
        if (self.sequence_id is None) == (self.context_id is None):
            raise ValueError("Paged attention row requires exactly one of sequence_id or context_id")
        if self.sequence_id is not None and (type(self.sequence_id) is not int or self.sequence_id < 0):
            raise ValueError("Paged attention sequence_id must be a non-negative integer")
        if self.context_id is not None and (type(self.context_id) is not str or not self.context_id):
            raise ValueError("Paged attention context_id must be a non-empty string")
        if type(self.query_len) is not int or self.query_len <= 0:
            raise ValueError("Paged attention query_len must be a positive integer")
        if type(self.seq_len) is not int or self.seq_len <= 0:
            raise ValueError("Paged attention seq_len must be a positive integer")
        if type(self.kv_start_pos) is not int or self.kv_start_pos < 0:
            raise ValueError("Paged attention kv_start_pos must be a non-negative integer")
        if type(self.imported_prefix_len) is not int or self.imported_prefix_len < 0:
            raise ValueError("Paged attention imported_prefix_len must be a non-negative integer")
        if self.imported_prefix_len not in (0, self.kv_start_pos):
            raise ValueError(
                "Paged attention imported prefix must fill the complete cached prefix: "
                f"imported={self.imported_prefix_len}, kv_start_pos={self.kv_start_pos}"
            )
        if self.kv_start_pos + self.query_len > self.seq_len:
            raise ValueError(
                "Paged attention write span exceeds seq_len: "
                f"start={self.kv_start_pos}, query_len={self.query_len}, seq_len={self.seq_len}"
            )

    @property
    def identity(self) -> tuple[str, int | None, str | None]:
        return (self.request_id, self.sequence_id, self.context_id)


@dataclass(frozen=True)
class PreparedDiffusionPagedAttentionBatch:
    """Native metadata shared by all paged attention layers in one forward."""

    rows: tuple[DiffusionPagedAttentionRow, ...]
    row_indices: torch.Tensor
    query_start_loc: torch.Tensor
    seq_lens: torch.Tensor
    positions: torch.Tensor
    block_tables: tuple[torch.Tensor, ...]
    slot_mappings: torch.Tensor
    attn_metadata: dict[str, Any]
    slot_mappings_by_layer: dict[str, torch.Tensor]
    imported_prefix_slot_mappings_by_layer: dict[str, torch.Tensor]
    num_tokens: int
    num_imported_prefix_tokens: int
    _owner: object = field(repr=False, compare=False)
    _generation: int = field(repr=False, compare=False)


@dataclass(frozen=True, slots=True)
class DiffusionPagedAttentionContext:
    """One layer's native inputs for an Omni paged-backend invocation."""

    layer: DiffusionPagedAttentionLayerAdapter
    query: torch.Tensor
    key_write: torch.Tensor
    value_write: torch.Tensor
    slot_mapping: torch.Tensor
    imported_prefix_key: torch.Tensor | None
    imported_prefix_value: torch.Tensor | None
    imported_prefix_slot_mapping: torch.Tensor | None
    native_metadata: Any
    piecewise_plan: PagedPiecewisePlan | None
    piecewise_native_metadata: tuple[Any, ...]
    query_token_shape: tuple[int, ...]
    query_has_head_dims: bool

    def restore_output(self, output: torch.Tensor) -> torch.Tensor:
        if self.query_has_head_dims:
            return output.reshape(*self.query_token_shape, self.layer.num_heads, self.layer.head_size_v)
        return output.reshape(*self.query_token_shape, self.layer.num_heads * self.layer.head_size_v)


class DiffusionPagedAttentionAdapter:
    """Translate diffusion rows into vLLM metadata for an Omni paged backend."""

    def __init__(
        self,
        *,
        vllm_config: VllmConfig,
        device: torch.device,
        kv_cache_config: KVCacheConfig,
        block_tables: BlockTables,
        attn_groups: list[list[Any]],
        layers: Mapping[str, DiffusionPagedAttentionLayerAdapter],
        resolve_row: DiffusionKVRowResolver,
    ) -> None:
        if not layers:
            raise ValueError("Paged attention requires at least one native attention layer")
        if len(attn_groups) != len(kv_cache_config.kv_cache_groups):
            raise ValueError(
                "Paged attention group mismatch: "
                f"builders={len(attn_groups)}, cache_groups={len(kv_cache_config.kv_cache_groups)}"
            )
        self.vllm_config = vllm_config
        self.device = torch.device(device)
        self.kv_cache_config = kv_cache_config
        self.block_tables = block_tables
        self.attn_groups = attn_groups
        self.layers = dict(layers)
        self.resolve_row = resolve_row
        self._owner = object()
        self._prepare_generation = 0
        self._active_batch: PreparedDiffusionPagedAttentionBatch | None = None
        self._active_piecewise_plan: PagedPiecewisePlan | None = None
        self._active_piecewise_native_metadata: tuple[dict[str, Any], ...] | None = None
        self._causal_by_group = self._resolve_group_causality()
        self._reorder_batch_threshold = self._resolve_reorder_batch_threshold()

    def _resolve_group_causality(self) -> dict[int, bool]:
        causal_by_group: dict[int, bool] = {}
        for group_index, cache_group in enumerate(self.kv_cache_config.kv_cache_groups):
            layer_causality = {
                not bool(getattr(self.layers[layer_name].spec, "non_causal", False))
                for layer_name in cache_group.layer_names
            }
            if len(layer_causality) != 1:
                raise ValueError(f"Paged attention cache group {group_index} mixes causal and non-causal layers")
            causal_by_group[group_index] = layer_causality.pop()
        return causal_by_group

    def _resolve_reorder_batch_threshold(self) -> int | None:
        thresholds = [
            group.get_metadata_builder(0).reorder_batch_threshold
            for group_index, groups in enumerate(self.attn_groups)
            if self._causal_by_group[group_index]
            for group in groups
        ]
        concrete_thresholds = [threshold for threshold in thresholds if threshold is not None]
        return min(concrete_thresholds, default=None)

    def _validate_native_batch_order(self, rows: tuple[DiffusionPagedAttentionRow, ...]) -> None:
        threshold = self._reorder_batch_threshold
        if threshold is None or not any(self._causal_by_group.values()):
            return

        found_long_query = False
        for row in rows:
            if row.query_len > threshold:
                found_long_query = True
            elif found_long_query:
                raise ValueError(
                    "Causal paged attention rows must place native decode/short-query rows before "
                    f"prefill/long-query rows (decode threshold={threshold})"
                )

    def _validate_row_capacity(
        self,
        rows: tuple[DiffusionPagedAttentionRow, ...],
        row_indices: list[int],
    ) -> None:
        native_num_blocks = self.block_tables.num_blocks.np
        blocks_per_kv_block = self.block_tables.blocks_per_kv_block
        for row, row_index in zip(rows, row_indices, strict=True):
            if type(row_index) is not int or not 0 <= row_index < self.block_tables.max_num_reqs:
                raise ValueError(f"Paged attention row {row.identity!r} resolved to invalid Worker row {row_index!r}")
            for group_index, (block_size, block_multiplier) in enumerate(
                zip(
                    self.block_tables.block_sizes,
                    blocks_per_kv_block,
                    strict=True,
                )
            ):
                required_blocks = cdiv(
                    row.seq_len,
                    block_size * self.block_tables.cp_size,
                )
                required_kernel_blocks = required_blocks * block_multiplier
                installed_kernel_blocks = int(native_num_blocks[group_index, row_index])
                if required_kernel_blocks > installed_kernel_blocks:
                    raise ValueError(
                        f"Paged attention row {row.identity!r} requires {required_blocks} blocks in "
                        f"cache group {group_index} for seq_len={row.seq_len}, but its installed Worker "
                        f"row contains only {installed_kernel_blocks // block_multiplier} blocks"
                    )

    def _build_native_metadata(
        self,
        *,
        query_lens: Sequence[int],
        query_start_loc: torch.Tensor,
        query_start_loc_cpu: torch.Tensor,
        seq_lens: torch.Tensor,
        seq_lens_cpu: torch.Tensor,
        positions: torch.Tensor,
        block_tables: Sequence[torch.Tensor],
        slot_mappings: torch.Tensor,
        causal: bool | Mapping[int, bool],
    ) -> dict[str, Any]:
        dcp_local_seq_lens = None
        if self.block_tables.cp_size > 1:
            dcp_local_seq_lens = get_dcp_local_seq_lens(
                seq_lens,
                dcp_size=self.block_tables.cp_size,
                dcp_rank=self.block_tables.cp_rank,
                cp_kv_cache_interleave_size=self.block_tables.cp_interleave,
            )
        with set_current_vllm_config(self.vllm_config):
            return current_omni_platform.build_diffusion_kv_attn_metadata(
                attn_groups=self.attn_groups,
                num_reqs=len(query_lens),
                num_tokens=sum(query_lens),
                query_start_loc_gpu=query_start_loc,
                query_start_loc_cpu=query_start_loc_cpu,
                max_query_len=max(query_lens),
                seq_lens=seq_lens,
                max_seq_len=max(int(seq_len) for seq_len in seq_lens_cpu),
                block_tables=block_tables,
                slot_mappings=slot_mappings,
                kv_cache_config=self.kv_cache_config,
                seq_lens_cpu=seq_lens_cpu,
                seq_lens_cpu_upper_bound=seq_lens_cpu,
                dcp_local_seq_lens=dcp_local_seq_lens,
                positions=positions,
                causal=causal,
            )

    def prepare_batch(
        self,
        rows: Sequence[DiffusionPagedAttentionRow],
    ) -> PreparedDiffusionPagedAttentionBatch:
        rows = tuple(rows)
        if not rows:
            raise ValueError("Paged attention batch must contain at least one row")
        if self._active_batch is not None:
            raise RuntimeError("Cannot prepare paged attention metadata during an active forward")
        if len(rows) > self.block_tables.max_num_reqs:
            raise ValueError(
                f"Paged attention batch has {len(rows)} rows; Worker capacity is {self.block_tables.max_num_reqs}"
            )

        identities = [row.identity for row in rows]
        if len(set(identities)) != len(identities):
            raise ValueError("Paged attention batch contains duplicate row identities")
        row_bindings = [self.resolve_row(row.request_id, row.sequence_id, row.context_id) for row in rows]
        row_indices_list = [binding.row_index for binding in row_bindings]
        if len(set(row_indices_list)) != len(row_indices_list):
            raise ValueError("Paged attention batch resolves multiple inputs to the same Worker row")
        for row, binding in zip(rows, row_bindings, strict=True):
            if type(binding.max_seq_len) is not int or binding.max_seq_len < 0:
                raise ValueError(
                    f"Paged attention row {row.identity!r} resolved to invalid logical capacity {binding.max_seq_len!r}"
                )
            if row.seq_len > binding.max_seq_len:
                raise ValueError(
                    f"Paged attention row {row.identity!r} uses seq_len={row.seq_len}, but its installed "
                    f"allocation has logical length {binding.max_seq_len}"
                )
        self._validate_native_batch_order(rows)
        self._validate_row_capacity(rows, row_indices_list)

        query_lens = [row.query_len for row in rows]
        num_tokens = sum(query_lens)
        if num_tokens > self.block_tables.max_num_batched_tokens:
            raise ValueError(
                f"Paged attention batch has {num_tokens} tokens; Worker capacity is "
                f"{self.block_tables.max_num_batched_tokens}"
            )
        query_offsets = [0]
        for query_len in query_lens:
            query_offsets.append(query_offsets[-1] + query_len)

        # Native BlockTables reuses persistent gather/slot buffers. Starting a
        # new preparation invalidates every batch prepared before it, even if a
        # later metadata builder raises.
        self._prepare_generation += 1
        generation = self._prepare_generation
        row_indices = torch.tensor(row_indices_list, dtype=torch.int32, device=self.device)
        query_start_loc_cpu = torch.tensor(query_offsets, dtype=torch.int32)
        query_start_loc = query_start_loc_cpu.to(self.device)
        seq_lens_cpu = torch.tensor([row.seq_len for row in rows], dtype=torch.int32)
        seq_lens = seq_lens_cpu.to(self.device)
        positions = torch.cat(
            [
                torch.arange(
                    row.kv_start_pos,
                    row.kv_start_pos + row.query_len,
                    dtype=torch.int64,
                    device=self.device,
                )
                for row in rows
            ]
        )

        block_tables = self.block_tables.gather_block_tables(
            row_indices,
            num_reqs_padded=len(rows),
        )
        imported_prefix_lens = [row.imported_prefix_len for row in rows]
        num_imported_prefix_tokens = sum(imported_prefix_lens)
        imported_prefix_slot_mappings_by_layer: dict[str, torch.Tensor] = {}
        if num_imported_prefix_tokens:
            imported_prefix_offsets = [0]
            for prefix_len in imported_prefix_lens:
                imported_prefix_offsets.append(imported_prefix_offsets[-1] + prefix_len)
            imported_prefix_start_loc = torch.tensor(
                imported_prefix_offsets,
                dtype=torch.int32,
                device=self.device,
            )
            imported_prefix_positions = torch.cat(
                [torch.arange(prefix_len, dtype=torch.int64, device=self.device) for prefix_len in imported_prefix_lens]
            )
            imported_prefix_slot_mappings = self.block_tables.compute_slot_mappings(
                row_indices,
                imported_prefix_start_loc,
                imported_prefix_positions,
                num_tokens_padded=num_imported_prefix_tokens,
            )
            # BlockTables owns reusable slot buffers. Preserve the imported
            # prefix before the current-write mapping reuses the same storage.
            imported_prefix_slot_mappings_by_layer = {
                layer_name: mapping.reshape(-1)[:num_imported_prefix_tokens].clone()
                for layer_name, mapping in build_slot_mappings_by_layer(
                    imported_prefix_slot_mappings,
                    self.kv_cache_config,
                ).items()
            }
        slot_mappings = self.block_tables.compute_slot_mappings(
            row_indices,
            query_start_loc,
            positions,
            num_tokens_padded=num_tokens,
        )
        causal: bool | Mapping[int, bool]
        causal_values = set(self._causal_by_group.values())
        causal = causal_values.pop() if len(causal_values) == 1 else self._causal_by_group
        attn_metadata = self._build_native_metadata(
            query_lens=query_lens,
            query_start_loc=query_start_loc,
            query_start_loc_cpu=query_start_loc_cpu,
            seq_lens=seq_lens,
            seq_lens_cpu=seq_lens_cpu,
            positions=positions,
            block_tables=block_tables,
            slot_mappings=slot_mappings,
            causal=causal,
        )
        slot_mappings_by_layer = build_slot_mappings_by_layer(
            slot_mappings,
            self.kv_cache_config,
        )
        return PreparedDiffusionPagedAttentionBatch(
            rows=rows,
            row_indices=row_indices,
            query_start_loc=query_start_loc,
            seq_lens=seq_lens,
            positions=positions,
            block_tables=tuple(block_tables),
            slot_mappings=slot_mappings,
            attn_metadata=attn_metadata,
            slot_mappings_by_layer=slot_mappings_by_layer,
            imported_prefix_slot_mappings_by_layer=imported_prefix_slot_mappings_by_layer,
            num_tokens=num_tokens,
            num_imported_prefix_tokens=num_imported_prefix_tokens,
            _owner=self._owner,
            _generation=generation,
        )

    def invalidate_prepared_batches(self) -> None:
        """Invalidate native buffer views after BlockTable state changes."""

        if self._active_batch is not None:
            raise RuntimeError("Cannot change paged attention BlockTables during an active forward")
        self._prepare_generation += 1

    @contextmanager
    def activate(
        self,
        batch: PreparedDiffusionPagedAttentionBatch,
    ) -> Iterator[DiffusionPagedAttentionAdapter]:
        if batch._owner is not self._owner:
            raise ValueError("Prepared paged attention batch belongs to a different adapter")
        if batch._generation != self._prepare_generation:
            raise ValueError("Prepared paged attention batch is stale after a newer batch preparation")
        if self._active_batch is not None:
            raise RuntimeError("Paged attention adapter already has an active forward")
        self._active_batch = batch
        self._active_piecewise_plan = None
        self._active_piecewise_native_metadata = None
        try:
            with override_paged_kv_adapter(self), set_current_vllm_config(self.vllm_config):
                yield self
        finally:
            self._active_piecewise_plan = None
            self._active_piecewise_native_metadata = None
            self._active_batch = None

    @staticmethod
    def _flatten_tensor(
        tensor: torch.Tensor,
        *,
        num_heads: int,
        head_size: int,
        name: str,
    ) -> tuple[torch.Tensor, tuple[int, ...], bool]:
        if tensor.ndim >= 3 and tensor.shape[-2:] == (num_heads, head_size):
            return tensor.reshape(-1, num_heads, head_size), tuple(tensor.shape[:-2]), True
        hidden_size = num_heads * head_size
        if tensor.ndim >= 1 and tensor.shape[-1] == hidden_size:
            return tensor.reshape(-1, num_heads, head_size), tuple(tensor.shape[:-1]), False
        raise ValueError(
            f"Paged attention {name} must end in ({num_heads}, {head_size}) or ({hidden_size},); "
            f"got shape={tuple(tensor.shape)}"
        )

    @staticmethod
    def _validate_token_layout(
        token_shape: tuple[int, ...],
        batch: PreparedDiffusionPagedAttentionBatch,
        *,
        name: str,
    ) -> None:
        if len(token_shape) == 1:
            if token_shape[0] == batch.num_tokens:
                return
            raise ValueError(
                f"Paged attention {name} token count must match the prepared write batch: "
                f"tokens={token_shape[0]}, prepared={batch.num_tokens}"
            )
        if len(token_shape) == 2:
            batch_size, tokens_per_row = token_shape
            query_lens = tuple(row.query_len for row in batch.rows)
            if batch_size == len(batch.rows) and all(query_len == tokens_per_row for query_len in query_lens):
                return
            raise ValueError(
                f"Paged attention batched {name} layout must match prepared rows for the current write: "
                f"shape={token_shape}, row_query_lens={query_lens}"
            )
        raise ValueError(
            f"Paged attention {name} supports packed [T, ...] or uniform batched [B, T, ...] token layouts; "
            f"got token shape={token_shape}"
        )

    @staticmethod
    def _validate_imported_prefix_layout(
        token_shape: tuple[int, ...],
        batch: PreparedDiffusionPagedAttentionBatch,
        *,
        name: str,
    ) -> None:
        prefix_lens = tuple(row.imported_prefix_len for row in batch.rows)
        if len(token_shape) == 1 and token_shape[0] == batch.num_imported_prefix_tokens:
            return
        if len(token_shape) == 2:
            batch_size, tokens_per_row = token_shape
            if batch_size == len(batch.rows) and all(prefix_len == tokens_per_row for prefix_len in prefix_lens):
                return
        raise ValueError(
            f"Paged attention {name} layout must match imported prefixes: "
            f"shape={token_shape}, row_prefix_lens={prefix_lens}"
        )

    def _get_piecewise_plan(
        self,
        full_attn_spans: list[list[tuple[int, int]]],
    ) -> PagedPiecewisePlan:
        batch = self._active_batch
        if batch is None:
            raise RuntimeError("Piecewise paged attention requires an active prepared batch")
        if any(row.kv_start_pos + row.query_len != row.seq_len for row in batch.rows):
            invalid_rows = [row.identity for row in batch.rows if row.kv_start_pos + row.query_len != row.seq_len]
            raise ValueError(
                "Paged piecewise attention requires each query/write span to end at seq_len; "
                f"invalid rows={invalid_rows!r}"
            )

        cached_plan = self._active_piecewise_plan
        if cached_plan is not None:
            frozen_spans = tuple(tuple(tuple(span) for span in row_spans) for row_spans in full_attn_spans)
            if cached_plan.spans != frozen_spans:
                raise ValueError("Paged piecewise attention metadata changed between layers in one active batch")
            return cached_plan

        plan = build_paged_piecewise_plan(
            full_attn_spans,
            query_offsets=[row.kv_start_pos for row in batch.rows],
            query_lens=[row.query_len for row in batch.rows],
            seq_lens=[row.seq_len for row in batch.rows],
            device=self.device,
        )
        self._active_piecewise_plan = plan
        return plan

    def _get_piecewise_native_metadata(
        self,
        plan: PagedPiecewisePlan,
    ) -> tuple[dict[str, Any], ...]:
        """Build one native metadata object per segment and cache it by batch."""

        batch = self._active_batch
        if batch is None:
            raise RuntimeError("Piecewise paged attention requires an active prepared batch")
        cached_metadata = self._active_piecewise_native_metadata
        if cached_metadata is not None:
            return cached_metadata

        metadata_by_segment: list[dict[str, Any]] = []
        for packed_segment in plan.segments:
            row_segments = packed_segment.row_segments
            query_lens = [segment.q_end - segment.q_start for segment in row_segments]
            seq_lens_values = [segment.kv_end for segment in row_segments]
            query_offsets = [0]
            for query_len in query_lens:
                query_offsets.append(query_offsets[-1] + query_len)
            query_start_loc_cpu = torch.tensor(query_offsets, dtype=torch.int32)
            seq_lens_cpu = torch.tensor(seq_lens_values, dtype=torch.int32)
            segment_metadata = self._build_native_metadata(
                query_lens=query_lens,
                query_start_loc=query_start_loc_cpu.to(self.device),
                query_start_loc_cpu=query_start_loc_cpu,
                seq_lens=seq_lens_cpu.to(self.device),
                seq_lens_cpu=seq_lens_cpu,
                positions=batch.positions.index_select(0, packed_segment.query_indices),
                block_tables=batch.block_tables,
                slot_mappings=batch.slot_mappings.index_select(-1, packed_segment.query_indices),
                causal=(row_segments[0].mode == "causal"),
            )
            # FA3's full-CUDA-graph metadata builder reuses one persistent
            # scheduler buffer across builds. Piecewise attention prepares all
            # segments before executing any of them, so each segment needs its
            # own snapshot rather than a view overwritten by the next build.
            snapshotted = set()
            for native_metadata in segment_metadata.values():
                metadata_id = id(native_metadata)
                if metadata_id in snapshotted:
                    continue
                snapshotted.add(metadata_id)
                for field_name in ("scheduler_metadata", "prefix_scheduler_metadata"):
                    scheduler_metadata = getattr(native_metadata, field_name, None)
                    if isinstance(scheduler_metadata, torch.Tensor):
                        setattr(native_metadata, field_name, scheduler_metadata.clone())
            metadata_by_segment.append(segment_metadata)
        self._active_piecewise_native_metadata = tuple(metadata_by_segment)
        return self._active_piecewise_native_metadata

    def prepare_layer_context(
        self,
        layer_name: str,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        *,
        omni_attn_metadata: Any | None = None,
    ) -> DiffusionPagedAttentionContext:
        if self._active_batch is None:
            raise RuntimeError("Paged attention forward must run inside adapter.activate(batch)")
        try:
            layer = self.layers[layer_name]
        except KeyError as exc:
            raise KeyError(f"Unknown diffusion paged attention layer {layer_name!r}") from exc
        batch = self._active_batch
        full_attn_spans = self._validate_omni_attn_metadata(omni_attn_metadata)
        imported_prefix_key = getattr(omni_attn_metadata, "paged_kv_prefix_key", None)
        imported_prefix_value = getattr(omni_attn_metadata, "paged_kv_prefix_value", None)
        if (imported_prefix_key is None) != (imported_prefix_value is None):
            raise ValueError("Paged attention imported prefix key and value must be provided together")

        query_flat, query_token_shape, query_has_head_dims = self._flatten_tensor(
            query,
            num_heads=layer.num_heads,
            head_size=layer.head_size,
            name="query",
        )
        key_flat, key_token_shape, _ = self._flatten_tensor(
            key,
            num_heads=layer.num_kv_heads,
            head_size=layer.head_size,
            name="key",
        )
        value_flat, value_token_shape, _ = self._flatten_tensor(
            value,
            num_heads=layer.num_kv_heads,
            head_size=layer.head_size_v,
            name="value",
        )
        self._validate_token_layout(query_token_shape, batch, name="query")
        self._validate_token_layout(key_token_shape, batch, name="key")
        self._validate_token_layout(value_token_shape, batch, name="value")
        imported_prefix_key_flat = imported_prefix_value_flat = None
        imported_prefix_slot_mapping = None
        if imported_prefix_key is not None:
            if batch.num_imported_prefix_tokens == 0:
                raise ValueError("Paged attention received imported prefix K/V for a batch without imported rows")
            imported_prefix_key_flat, imported_key_shape, _ = self._flatten_tensor(
                imported_prefix_key,
                num_heads=layer.num_kv_heads,
                head_size=layer.head_size,
                name="imported prefix key",
            )
            imported_prefix_value_flat, imported_value_shape, _ = self._flatten_tensor(
                imported_prefix_value,
                num_heads=layer.num_kv_heads,
                head_size=layer.head_size_v,
                name="imported prefix value",
            )
            self._validate_imported_prefix_layout(imported_key_shape, batch, name="imported prefix key")
            self._validate_imported_prefix_layout(imported_value_shape, batch, name="imported prefix value")
            imported_prefix_slot_mapping = batch.imported_prefix_slot_mappings_by_layer.get(layer_name)
            if imported_prefix_slot_mapping is None:
                raise KeyError(f"No imported-prefix slot mapping was built for diffusion layer {layer_name!r}")
        elif batch.num_imported_prefix_tokens:
            raise ValueError("Paged attention batch requires imported prefix K/V for the current layer")
        if query.device != self.device or key.device != self.device or value.device != self.device:
            raise ValueError(
                f"Paged attention Q/K/V must be on {self.device}; "
                f"got query={query.device}, key={key.device}, value={value.device}"
            )
        if query.dtype != key.dtype or query.dtype != value.dtype:
            raise ValueError(f"Paged attention Q/K/V dtypes must match; got {query.dtype}, {key.dtype}, {value.dtype}")
        expected_dtype = self.vllm_config.model_config.dtype
        if query.dtype != expected_dtype:
            raise ValueError(
                f"Paged attention Q/K/V dtype must match model activation dtype {expected_dtype}; got {query.dtype}"
            )
        if imported_prefix_key is not None:
            if imported_prefix_key.device != self.device or imported_prefix_value.device != self.device:
                raise ValueError("Paged attention imported prefix K/V must use the paged cache device")
            if imported_prefix_key.dtype != query.dtype or imported_prefix_value.dtype != query.dtype:
                raise ValueError("Paged attention imported prefix K/V dtype must match the current Q/K/V dtype")
        if not bool(getattr(layer.spec, "non_causal", False)):
            non_suffix_rows = [row.identity for row in batch.rows if row.kv_start_pos + row.query_len != row.seq_len]
            if non_suffix_rows:
                raise ValueError(
                    "Causal paged attention requires each query/write span to end at seq_len; "
                    f"invalid rows={non_suffix_rows!r}"
                )

        slot_mapping = batch.slot_mappings_by_layer.get(layer_name)
        if slot_mapping is None:
            raise KeyError(f"No native slot mapping was built for diffusion layer {layer_name!r}")
        slot_mapping = slot_mapping.reshape(-1)[: batch.num_tokens]
        if slot_mapping.numel() != batch.num_tokens:
            raise ValueError(
                f"Paged attention slot mapping for {layer_name!r} has {slot_mapping.numel()} entries; "
                f"expected {batch.num_tokens}"
            )

        # vLLM's writer consumes slot_mapping directly, so a span that starts
        # inside a physical block naturally writes only the requested suffix.
        key_flat = key_flat.contiguous()
        value_flat = value_flat.contiguous()
        try:
            native_metadata = batch.attn_metadata[layer_name]
        except KeyError as exc:
            raise KeyError(f"No native attention metadata was built for diffusion layer {layer_name!r}") from exc
        piecewise_plan: PagedPiecewisePlan | None = None
        piecewise_native_metadata: tuple[Any, ...] = ()
        if full_attn_spans is not None:
            piecewise_plan = self._get_piecewise_plan(full_attn_spans)
            native_metadata_by_segment = self._get_piecewise_native_metadata(piecewise_plan)
            try:
                piecewise_native_metadata = tuple(
                    segment_metadata[layer_name] for segment_metadata in native_metadata_by_segment
                )
            except KeyError as exc:
                raise KeyError(
                    f"No piecewise native attention metadata was built for diffusion layer {layer_name!r}"
                ) from exc
        return DiffusionPagedAttentionContext(
            layer=layer,
            query=query_flat.contiguous(),
            key_write=key_flat,
            value_write=value_flat,
            slot_mapping=slot_mapping,
            imported_prefix_key=None if imported_prefix_key_flat is None else imported_prefix_key_flat.contiguous(),
            imported_prefix_value=None
            if imported_prefix_value_flat is None
            else imported_prefix_value_flat.contiguous(),
            imported_prefix_slot_mapping=imported_prefix_slot_mapping,
            native_metadata=native_metadata,
            piecewise_plan=piecewise_plan,
            piecewise_native_metadata=piecewise_native_metadata,
            query_token_shape=query_token_shape,
            query_has_head_dims=query_has_head_dims,
        )

    @staticmethod
    def _validate_omni_attn_metadata(
        metadata: Any | None,
    ) -> list[list[tuple[int, int]]] | None:
        if metadata is None:
            return None
        full_attn_spans = getattr(metadata, "full_attn_spans", None)
        attn_mask = getattr(metadata, "attn_mask", None)
        unsupported_fields = [
            field_name
            for field_name in (
                "joint_attn_mask",
                "query_ranges",
                "video_layout",
            )
            if getattr(metadata, field_name, None) is not None
        ]
        if attn_mask is not None:
            if full_attn_spans is None:
                unsupported_fields.append("attn_mask")
            elif not isinstance(attn_mask, torch.Tensor) or attn_mask.ndim != 4:
                unsupported_fields.append("attn_mask (piecewise paging requires a 4D tensor)")
        extra = getattr(metadata, "extra", None)
        if extra:
            unsupported_fields.append(f"extra={sorted(extra)}")
        if unsupported_fields:
            raise NotImplementedError(
                "Diffusion paged attention cannot translate Omni attention metadata fields "
                f"{unsupported_fields!r} to native FlashAttention metadata"
            )
        return full_attn_spans
