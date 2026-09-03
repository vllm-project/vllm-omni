# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from __future__ import annotations

import copy
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import torch
from vllm.config import VllmConfig, set_current_vllm_config
from vllm.distributed import get_dcp_group
from vllm.v1.kv_cache_interface import AttentionSpec, KVCacheConfig, KVCacheSpec
from vllm.v1.worker import block_table as native_block_table
from vllm.v1.worker.gpu.attn_utils import init_attn_backend, init_kv_cache
from vllm.v1.worker.gpu.block_table import BlockTables

from vllm_omni.diffusion.data import OmniDiffusionConfig
from vllm_omni.diffusion.diffusion_kv.config import DiffusionKVCacheMode, is_scheduler_paged_kv_mode
from vllm_omni.diffusion.diffusion_kv.metadata import DiffusionKVMetadata
from vllm_omni.diffusion.diffusion_kv.paged_attention_adapter import (
    DiffusionPagedAttentionAdapter,
    DiffusionPagedAttentionLayerAdapter,
    DiffusionPagedAttentionMetadata,
    DiffusionPagedAttentionRowBinding,
    DiffusionPagedAttentionRuntime,
)
from vllm_omni.platforms import current_omni_platform

DiffusionKVIdentity = tuple[str, int | None, str | None]
DiffusionKVSnapshot = tuple[object, ...]


@dataclass(frozen=True)
class _DiffusionKVRowInstall:
    identity: DiffusionKVIdentity
    token_len: int
    block_ids: tuple[tuple[int, ...], ...]


@dataclass(frozen=True)
class _DiffusionKVRequestState:
    generation: int
    snapshot: DiffusionKVSnapshot
    row_installs: tuple[_DiffusionKVRowInstall, ...]


class DiffusionKVModelRunnerBackend:
    """Native paged-KV state and BlockTable operations for a model runner."""

    def __init__(
        self,
        *,
        vllm_config: VllmConfig,
        od_config: OmniDiffusionConfig,
        device: torch.device,
    ) -> None:
        self.vllm_config = vllm_config
        self.od_config = od_config
        self.device = device
        self.kv_cache_config: KVCacheConfig | None = None
        self.kv_caches: list[torch.Tensor | list[torch.Tensor]] = []
        self.block_tables: BlockTables | None = None
        self.paged_attention_adapter: DiffusionPagedAttentionAdapter | None = None
        self._kv_cache_layer_adapters: dict[str, DiffusionPagedAttentionLayerAdapter] = {}
        self._diffusion_kv_identity_to_row: dict[DiffusionKVIdentity, int] = {}
        self._diffusion_kv_free_rows: list[int] = []
        self._diffusion_kv_request_states: dict[str, _DiffusionKVRequestState] = {}

    def register_kv_cache_layers(
        self,
        layers: Mapping[str, tuple[Any, KVCacheSpec]],
    ) -> dict[str, KVCacheSpec]:
        """Register native adapters and return their canonical cache specs."""

        if self.kv_cache_config is not None:
            raise RuntimeError("Diffusion KV cache layers cannot be changed after physical initialization")

        if layers:
            current_omni_platform.get_diffusion_kv_block_tables_cls()
            if not callable(current_omni_platform.build_diffusion_kv_attn_metadata):
                raise NotImplementedError(
                    "Diffusion paged KV requires platform-native BlockTables and attention metadata hooks"
                )

        forward_context = self.vllm_config.compilation_config.static_forward_context
        previous_adapters = self._kv_cache_layer_adapters
        for layer_name, (layer, spec) in layers.items():
            if not isinstance(spec, AttentionSpec):
                raise TypeError(
                    f"Diffusion KV layer {layer_name!r} produced unsupported spec {type(spec).__name__}; "
                    "only native attention specs are supported"
                )
            if not layer.attn_backend.supports_paged_kv:
                raise NotImplementedError(
                    f"Diffusion paged KV layer {layer_name!r} requires an Omni backend with paged support; "
                    f"selected {layer.attn_backend.get_name()}"
                )
            existing = forward_context.get(layer_name)
            if existing is not None and existing is not layer and existing is not previous_adapters.get(layer_name):
                raise RuntimeError(f"Diffusion KV layer name {layer_name!r} collides with the native forward context")

        previous_use_non_causal = self.vllm_config.attention_config.use_non_causal
        self.vllm_config.attention_config.use_non_causal = any(
            bool(getattr(spec, "non_causal", False)) for _, spec in layers.values()
        )
        adapters: dict[str, DiffusionPagedAttentionLayerAdapter] = {}
        try:
            with set_current_vllm_config(self.vllm_config):
                for layer_name, (layer, spec) in layers.items():
                    adapters[layer_name] = DiffusionPagedAttentionLayerAdapter(
                        layer_name=layer_name,
                        layer=layer,
                        spec=spec,
                        vllm_config=self.vllm_config,
                        device=self.device,
                        ulysses_degree=self._get_paged_attention_ulysses_degree(layer),
                    )
        except Exception:
            self.vllm_config.attention_config.use_non_causal = previous_use_non_causal
            raise

        geometries = {(adapter.num_heads, adapter.num_kv_heads, adapter.head_size) for adapter in adapters.values()}
        if len(geometries) > 1:
            self.vllm_config.attention_config.use_non_causal = previous_use_non_causal
            raise ValueError(
                "Native paged attention metadata builders require one rank-local attention geometry; "
                f"got {sorted(geometries)!r}"
            )
        attention_geometry = next(iter(geometries)) if geometries else None
        set_attention_geometry = (
            getattr(self.vllm_config.model_config, "set_attention_geometry", None)
            if attention_geometry is not None
            else None
        )
        if attention_geometry is not None and not callable(set_attention_geometry):
            self.vllm_config.attention_config.use_non_causal = previous_use_non_causal
            raise RuntimeError(
                "Diffusion native paged attention requires a model config that exposes set_attention_geometry"
            )

        updated_forward_context = dict(forward_context)
        for layer_name in set(layers) | set(previous_adapters):
            entry = updated_forward_context.get(layer_name)
            layer_entry = layers.get(layer_name)
            if entry is previous_adapters.get(layer_name) or (layer_entry is not None and entry is layer_entry[0]):
                updated_forward_context.pop(layer_name, None)
        updated_forward_context.update(adapters)

        if attention_geometry is not None:
            num_heads, num_kv_heads, head_size = attention_geometry
            try:
                set_attention_geometry(
                    num_heads=num_heads,
                    num_kv_heads=num_kv_heads,
                    head_size=head_size,
                )
            except Exception:
                self.vllm_config.attention_config.use_non_causal = previous_use_non_causal
                raise

        forward_context.clear()
        forward_context.update(updated_forward_context)
        self._kv_cache_layer_adapters = adapters
        return {layer_name: adapter.spec for layer_name, adapter in adapters.items()}

    def _get_paged_attention_ulysses_degree(self, layer: Any) -> int:
        if bool(getattr(layer, "skip_sequence_parallel", False)):
            return 1

        parallel_config = getattr(self.od_config, "parallel_config", None)
        ulysses_degree = int(getattr(parallel_config, "ulysses_degree", 1))
        ring_degree = int(getattr(parallel_config, "ring_degree", 1))
        allgather_degree = int(getattr(parallel_config, "allgather_degree", 1))
        ulysses_mode = str(getattr(parallel_config, "ulysses_mode", "strict"))
        if ring_degree > 1:
            raise NotImplementedError(
                "Diffusion paged attention does not support Ring or hybrid Ulysses+Ring attention"
            )
        if allgather_degree > 1:
            raise NotImplementedError("Diffusion paged attention does not support AllGather-KV sequence parallelism")
        if ulysses_degree > 1 and ulysses_mode != "strict":
            raise NotImplementedError(
                f"Diffusion paged attention currently supports only strict Ulysses; got ulysses_mode={ulysses_mode!r}"
            )
        return ulysses_degree

    def _get_max_rows_per_request(self) -> int:
        max_rows = getattr(self.od_config, "diffusion_kv_max_rows_per_request", None)
        if type(max_rows) is not int or max_rows <= 0:
            raise ValueError(
                "paged_scheduler requires a positive integer diffusion_kv_max_rows_per_request adapter limit"
            )
        return max_rows

    def _get_max_model_len(self) -> int:
        max_len = getattr(self.vllm_config.model_config, "max_model_len", None)
        if type(max_len) is not int or max_len <= 0:
            raise ValueError("paged_scheduler requires a positive model_config.max_model_len")
        return max_len

    def initialize_kv_cache(self, kv_cache_config: KVCacheConfig) -> None:
        """Allocate and bind native attention KV tensors for this rank."""
        if self.kv_cache_config is not None:
            raise RuntimeError("Diffusion native KV cache is already initialized")
        if not self._kv_cache_layer_adapters:
            raise RuntimeError("Diffusion KV cache layers must be registered before physical initialization")

        configured_layers = {
            layer_name for group in kv_cache_config.kv_cache_groups for layer_name in group.layer_names
        }
        registered_layers = set(self._kv_cache_layer_adapters)
        if configured_layers != registered_layers:
            raise ValueError(
                "Rank-local Diffusion KVCacheConfig layer mismatch: "
                f"expected={sorted(registered_layers)}, configured={sorted(configured_layers)}"
            )

        kv_cache_config = copy.deepcopy(kv_cache_config)
        max_len = self._get_max_model_len()
        max_rows_per_request = self._get_max_rows_per_request()
        scheduler_config = self.vllm_config.scheduler_config
        max_num_seqs = getattr(scheduler_config, "max_num_seqs", None)
        if type(max_num_seqs) is not int or max_num_seqs <= 0:
            raise ValueError("paged_scheduler requires a positive scheduler_config.max_num_seqs")

        kv_cache_groups = kv_cache_config.kv_cache_groups
        if not kv_cache_groups:
            raise ValueError("paged_scheduler requires at least one native KV cache group")
        block_sizes = [group.kv_cache_spec.block_size for group in kv_cache_groups]
        if any(type(block_size) is not int or block_size <= 0 for block_size in block_sizes):
            raise ValueError("native diffusion KV cache groups must have positive integer block sizes")
        unaligned_max_num_blocks_per_group = [
            group.kv_cache_spec.max_num_blocks_per_req(self.vllm_config, max_len) for group in kv_cache_groups
        ]
        if any(type(capacity) is not int or capacity <= 0 for capacity in unaligned_max_num_blocks_per_group):
            raise ValueError("native diffusion KV cache groups must have positive row capacities")
        max_num_blocks_per_group = [
            native_block_table.get_block_table_width(capacity, block_size)
            for capacity, block_size in zip(unaligned_max_num_blocks_per_group, block_sizes, strict=True)
        ]

        max_num_reqs = max_num_seqs * max_rows_per_request
        max_num_batched_tokens = getattr(scheduler_config, "max_num_batched_tokens", None)
        if type(max_num_batched_tokens) is not int or max_num_batched_tokens <= 0:
            raise ValueError("scheduler_config.max_num_batched_tokens must be a positive integer")
        parallel_config = self.vllm_config.parallel_config
        cp_size = parallel_config.decode_context_parallel_size
        cp_rank = get_dcp_group().rank_in_group if cp_size > 1 else 0
        cp_interleave = parallel_config.cp_kv_cache_interleave_size

        # Native metadata builders size per-row buffers from max_num_seqs.
        # A diffusion public request can occupy several sequence/context rows.
        scheduler_config.max_num_seqs = max_num_reqs
        kv_caches: list[torch.Tensor | list[torch.Tensor]] = []
        previous_adapter_caches = {
            layer_name: adapter.kv_cache for layer_name, adapter in self._kv_cache_layer_adapters.items()
        }
        try:
            with set_current_vllm_config(self.vllm_config):
                attn_groups, _, kernel_block_sizes = init_attn_backend(
                    kv_cache_config,
                    self.vllm_config,
                    self.device,
                )
                block_tables_cls = current_omni_platform.get_diffusion_kv_block_tables_cls()
                block_tables = block_tables_cls(
                    block_sizes=block_sizes,
                    max_num_reqs=max_num_reqs,
                    max_num_batched_tokens=max_num_batched_tokens,
                    max_num_blocks_per_group=max_num_blocks_per_group,
                    device=self.device,
                    kernel_block_sizes=kernel_block_sizes,
                    cp_size=cp_size,
                    cp_rank=cp_rank,
                    cp_interleave=cp_interleave,
                )
                paged_attention_adapter = DiffusionPagedAttentionAdapter(
                    vllm_config=self.vllm_config,
                    device=self.device,
                    kv_cache_config=kv_cache_config,
                    block_tables=block_tables,
                    attn_groups=attn_groups,
                    layers=self._kv_cache_layer_adapters,
                    resolve_row=self._resolve_paged_attention_row,
                )
                init_kv_cache(
                    kv_caches,
                    self.vllm_config.compilation_config.static_forward_context,
                    kv_cache_config,
                    attn_groups,
                    self.device,
                    self.vllm_config.cache_config.cache_dtype,
                    kernel_block_sizes,
                    self.vllm_config,
                )
        except Exception:
            scheduler_config.max_num_seqs = max_num_seqs
            for layer_name, previous_cache in previous_adapter_caches.items():
                self._kv_cache_layer_adapters[layer_name].kv_cache = previous_cache
            raise

        self.kv_caches = kv_caches
        self.block_tables = block_tables
        self.paged_attention_adapter = paged_attention_adapter
        self._diffusion_kv_identity_to_row.clear()
        self._diffusion_kv_free_rows = list(range(max_num_reqs - 1, -1, -1))
        self._diffusion_kv_request_states.clear()
        self.kv_cache_config = kv_cache_config

    def _validate_row(
        self,
        *,
        identity: DiffusionKVIdentity,
        token_len: int,
        block_ids: tuple[list[int], ...],
        seen_identities: set[DiffusionKVIdentity],
    ) -> _DiffusionKVRowInstall:
        if identity in seen_identities:
            raise ValueError(f"Duplicate diffusion KV row identity: {identity!r}")
        seen_identities.add(identity)
        if type(token_len) is not int or token_len < 0:
            raise ValueError(f"Diffusion KV row {identity!r} has invalid token length {token_len!r}")

        assert self.kv_cache_config is not None
        groups = self.kv_cache_config.kv_cache_groups
        if not isinstance(block_ids, tuple) or len(block_ids) != len(groups):
            actual_groups = len(block_ids) if isinstance(block_ids, (tuple, list)) else "invalid"
            raise ValueError(f"Diffusion KV row {identity!r} has {actual_groups} block groups; expected {len(groups)}")

        max_len = self._get_max_model_len()
        normalized_groups: list[tuple[int, ...]] = []
        for group_index, (group_block_ids, kv_cache_group) in enumerate(zip(block_ids, groups, strict=True)):
            if not isinstance(group_block_ids, list):
                raise ValueError(f"Diffusion KV row {identity!r} group {group_index} block IDs must be a list")
            spec = kv_cache_group.kv_cache_spec
            expected_count = spec.max_num_blocks_per_req(self.vllm_config, token_len)
            if len(group_block_ids) != expected_count:
                raise ValueError(
                    f"Diffusion KV row {identity!r} group {group_index} has {len(group_block_ids)} blocks; "
                    f"expected {expected_count} for {token_len} tokens"
                )
            row_capacity = spec.max_num_blocks_per_req(self.vllm_config, max_len)
            if len(group_block_ids) > row_capacity:
                raise ValueError(
                    f"Diffusion KV row {identity!r} group {group_index} exceeds row capacity {row_capacity}"
                )
            normalized_ids: list[int] = []
            for block_id in group_block_ids:
                if type(block_id) is not int:
                    raise ValueError(
                        f"Diffusion KV row {identity!r} group {group_index} has non-integer block ID {block_id!r}"
                    )
                if not 0 <= block_id < self.kv_cache_config.num_blocks:
                    raise ValueError(
                        f"Diffusion KV row {identity!r} group {group_index} block ID {block_id} is outside "
                        f"[0, {self.kv_cache_config.num_blocks})"
                    )
                normalized_ids.append(block_id)
            normalized_groups.append(tuple(normalized_ids))

        return _DiffusionKVRowInstall(
            identity=identity,
            token_len=token_len,
            block_ids=tuple(normalized_groups),
        )

    def _prepare_install(
        self,
        metadata: DiffusionKVMetadata,
    ) -> tuple[list[_DiffusionKVRowInstall], DiffusionKVSnapshot]:
        if self.block_tables is None or self.kv_cache_config is None:
            raise RuntimeError("paged_scheduler native KV cache is not initialized")
        if type(metadata.request_id) is not str or not metadata.request_id:
            raise ValueError("Diffusion KV metadata request_id must be a non-empty string")
        if type(metadata.allocation_generation) is not int or metadata.allocation_generation < 0:
            raise ValueError("Diffusion KV allocation_generation must be a non-negative integer")
        if type(self.kv_cache_config.num_blocks) is not int or self.kv_cache_config.num_blocks <= 0:
            raise ValueError("native diffusion KV cache config must provide a positive num_blocks")
        if not metadata.sequences:
            raise ValueError(f"Diffusion KV request {metadata.request_id!r} must contain at least one sequence")

        seen_identities: set[DiffusionKVIdentity] = set()
        installs: list[_DiffusionKVRowInstall] = []
        sequence_snapshots: list[object] = []
        for sequence in metadata.sequences:
            if type(sequence.sequence_id) is not int or sequence.sequence_id < 0:
                raise ValueError(f"Diffusion KV sequence_id must be a non-negative integer: {sequence.sequence_id!r}")
            sequence_identity = (metadata.request_id, sequence.sequence_id, None)
            sequence_install = self._validate_row(
                identity=sequence_identity,
                token_len=sequence.seq_len,
                block_ids=sequence.block_ids,
                seen_identities=seen_identities,
            )
            installs.append(sequence_install)
            sequence_snapshots.append(
                (
                    sequence.sequence_id,
                    sequence.prefix_len,
                    sequence.target_len,
                    sequence.seq_len,
                    sequence_install.block_ids,
                    sequence.context_ids,
                )
            )

        context_snapshots: list[object] = []
        for context in metadata.contexts:
            if type(context.context_id) is not str or not context.context_id:
                raise ValueError("Diffusion KV context_id must be a non-empty string")
            context_identity = (metadata.request_id, None, context.context_id)
            context_install = self._validate_row(
                identity=context_identity,
                token_len=context.num_tokens,
                block_ids=context.block_ids,
                seen_identities=seen_identities,
            )
            installs.append(context_install)
            context_snapshots.append(
                (
                    context.context_id,
                    context.cache_role,
                    context.num_tokens,
                    context_install.block_ids,
                )
            )

        max_rows_per_request = self._get_max_rows_per_request()
        if len(installs) > max_rows_per_request:
            raise ValueError(
                f"Diffusion KV request {metadata.request_id!r} requires {len(installs)} rows; "
                f"adapter limit is {max_rows_per_request}"
            )
        snapshot: DiffusionKVSnapshot = (
            metadata.request_id,
            tuple(sequence_snapshots),
            tuple(context_snapshots),
        )
        return installs, snapshot

    def _rollback_staged_writes(self, rows: list[int], previous_num_blocks: object | None) -> None:
        block_tables = self.block_tables
        if block_tables is None:
            return
        for group_table in getattr(block_tables, "block_tables", ()):
            clear_staged_writes = getattr(group_table, "clear_staged_writes", None)
            if clear_staged_writes is not None:
                clear_staged_writes()
        if previous_num_blocks is None:
            return
        native_num_blocks = getattr(block_tables, "num_blocks", None)
        if native_num_blocks is None or not hasattr(native_num_blocks, "np"):
            return
        native_num_blocks.np[:, rows] = previous_num_blocks
        copy_to_uva = getattr(native_num_blocks, "copy_to_uva", None)
        if copy_to_uva is not None:
            copy_to_uva()

    def _apply_rows(self, rows: list[int], installs: list[_DiffusionKVRowInstall]) -> None:
        assert self.block_tables is not None
        if self.paged_attention_adapter is not None:
            self.paged_attention_adapter.invalidate_prepared_batches()
        native_num_blocks = getattr(self.block_tables, "num_blocks", None)
        previous_num_blocks = None
        if native_num_blocks is not None and hasattr(native_num_blocks, "np"):
            previous_num_blocks = native_num_blocks.np[:, rows].copy()
        try:
            for row, install in zip(rows, installs, strict=True):
                self.block_tables.append_block_ids(
                    row,
                    tuple(list(group_ids) for group_ids in install.block_ids),
                    overwrite=True,
                )
            self.block_tables.apply_staged_writes()
        except Exception:
            self._rollback_staged_writes(rows, previous_num_blocks)
            raise

    def install_diffusion_kv_metadata(self, metadata: DiffusionKVMetadata) -> bool:
        """Install one Scheduler allocation snapshot into native Worker rows."""
        if not is_scheduler_paged_kv_mode(
            getattr(self.od_config, "diffusion_kv_mode", DiffusionKVCacheMode.DENSE_LEGACY)
        ):
            raise ValueError("Dense diffusion execution must not install Diffusion KV metadata")

        installs, snapshot = self._prepare_install(metadata)
        current_state = self._diffusion_kv_request_states.get(metadata.request_id)
        if current_state is not None:
            if metadata.allocation_generation < current_state.generation:
                raise ValueError(
                    f"Stale Diffusion KV allocation generation {metadata.allocation_generation} for request "
                    f"{metadata.request_id!r}; installed generation is {current_state.generation}"
                )
            if snapshot != current_state.snapshot:
                raise ValueError(
                    f"Conflicting Diffusion KV allocation snapshot for active request {metadata.request_id!r}; "
                    "remove the request before replacing its rows"
                )
            if metadata.allocation_generation == current_state.generation:
                return False
            self._diffusion_kv_request_states[metadata.request_id] = _DiffusionKVRequestState(
                generation=metadata.allocation_generation,
                snapshot=snapshot,
                row_installs=current_state.row_installs,
            )
            return False

        existing_identities = [
            install.identity for install in installs if install.identity in self._diffusion_kv_identity_to_row
        ]
        if existing_identities:
            raise RuntimeError(f"Diffusion KV row registry contains orphaned identities: {existing_identities!r}")
        if len(installs) > len(self._diffusion_kv_free_rows):
            raise ValueError(
                f"Diffusion KV request {metadata.request_id!r} requires {len(installs)} rows, but only "
                f"{len(self._diffusion_kv_free_rows)} rows are free"
            )

        rows = list(reversed(self._diffusion_kv_free_rows[-len(installs) :])) if installs else []
        if rows:
            self._apply_rows(rows, installs)

        if installs:
            del self._diffusion_kv_free_rows[-len(installs) :]
        self._diffusion_kv_identity_to_row.update(
            (install.identity, row) for install, row in zip(installs, rows, strict=True)
        )
        self._diffusion_kv_request_states[metadata.request_id] = _DiffusionKVRequestState(
            generation=metadata.allocation_generation,
            snapshot=snapshot,
            row_installs=tuple(installs),
        )
        return True

    def get_diffusion_kv_row(
        self,
        request_id: str,
        sequence_id: int | None,
        context_id: str | None = None,
    ) -> int:
        """Resolve a Scheduler allocation identity to its native table row."""
        if (
            not is_scheduler_paged_kv_mode(
                getattr(self.od_config, "diffusion_kv_mode", DiffusionKVCacheMode.DENSE_LEGACY)
            )
            or self.block_tables is None
        ):
            raise RuntimeError("paged_scheduler native BlockTables are not initialized")
        identity = (request_id, None, context_id) if context_id is not None else (request_id, sequence_id, None)
        try:
            return self._diffusion_kv_identity_to_row[identity]
        except KeyError as exc:
            raise KeyError(f"No native diffusion KV row is installed for {identity!r}") from exc

    def get_paged_attention_adapter(self) -> DiffusionPagedAttentionAdapter:
        if self.paged_attention_adapter is None:
            raise RuntimeError("paged_scheduler native attention adapter is not initialized")
        return self.paged_attention_adapter

    def activate_paged_attention_metadata(self, metadata: DiffusionPagedAttentionMetadata):
        """Activate Runner-owned request metadata for an internal denoise loop."""

        runtime = DiffusionPagedAttentionRuntime(self.get_paged_attention_adapter(), metadata)
        return runtime, runtime.activate()

    def _resolve_paged_attention_row(
        self,
        request_id: str,
        sequence_id: int | None,
        context_id: str | None,
    ) -> DiffusionPagedAttentionRowBinding:
        row_index = self.get_diffusion_kv_row(request_id, sequence_id, context_id)
        identity = (request_id, None, context_id) if context_id is not None else (request_id, sequence_id, None)
        request_state = self._diffusion_kv_request_states.get(request_id)
        if request_state is not None:
            for install in request_state.row_installs:
                if install.identity == identity:
                    return DiffusionPagedAttentionRowBinding(
                        row_index=row_index,
                        max_seq_len=install.token_len,
                        block_ids=install.block_ids,
                    )
        raise RuntimeError(f"Diffusion KV request state is missing logical length for {identity!r}")

    def remove_diffusion_kv_requests(self, request_ids: list[str]) -> int:
        """Retire Worker rows without logically freeing Scheduler-owned blocks."""
        if (
            not is_scheduler_paged_kv_mode(
                getattr(self.od_config, "diffusion_kv_mode", DiffusionKVCacheMode.DENSE_LEGACY)
            )
            or self.block_tables is None
        ):
            return 0
        request_id_set = set(request_ids)
        identities_and_rows = [
            (identity, row)
            for identity, row in self._diffusion_kv_identity_to_row.items()
            if identity[0] in request_id_set
        ]
        rows = [row for _, row in identities_and_rows]
        if rows:
            assert self.kv_cache_config is not None
            num_groups = len(self.kv_cache_config.kv_cache_groups)
            installs = [
                _DiffusionKVRowInstall(
                    identity=identity,
                    token_len=0,
                    block_ids=tuple(tuple() for _ in range(num_groups)),
                )
                for identity, _ in identities_and_rows
            ]
            self._apply_rows(rows, installs)

        for identity, _ in identities_and_rows:
            del self._diffusion_kv_identity_to_row[identity]
        for request_id in request_id_set:
            self._diffusion_kv_request_states.pop(request_id, None)
        self._diffusion_kv_free_rows.extend(rows)
        self._diffusion_kv_free_rows.sort(reverse=True)
        return len(rows)

    def refresh_block_table_layout(self) -> None:
        """Refresh native pointer tensors after a CuMem KV-cache wake-up."""
        if (
            not is_scheduler_paged_kv_mode(
                getattr(self.od_config, "diffusion_kv_mode", DiffusionKVCacheMode.DENSE_LEGACY)
            )
            or self.block_tables is None
        ):
            return
        if self.paged_attention_adapter is not None:
            self.paged_attention_adapter.invalidate_prepared_batches()
        self.block_tables.init_block_table_layout_tensors()
