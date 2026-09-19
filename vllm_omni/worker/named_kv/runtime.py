# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Optional runner-owned causal PagedAttention KV branches.

The implementation is deliberately narrow: a fixed scheduler-bounded number
of requests, one full-attention layer group, and a fixed GPU pool with no
overcommit. Allocation and lifecycle bookkeeping run eagerly. Models may use
the eager attention context or an optional independent compiled/CUDA Graph
executor. Models that do not declare a :class:`NamedKVBranchRequest` never
construct this store.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from math import ceil
from typing import Any

import torch
from vllm.forward_context import create_forward_context, override_forward_context
from vllm.logger import init_logger
from vllm.v1.attention.backend import CommonAttentionMetadata
from vllm.v1.core.kv_cache_manager import KVCacheManager
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
    KVCacheTensor,
    KVQuantMode,
)
from vllm.v1.request import RequestStatus

from vllm_omni.platforms import current_omni_platform
from vllm_omni.worker.named_kv.types import (
    NamedKVAppendBatch,
    NamedKVBranchStep,
)

logger = init_logger(__name__)


@dataclass(frozen=True)
class NamedKVBranchRequest:
    """A model request for an optional runner-owned KV branch."""

    name: str
    memory_bytes: int
    layer_group: int = 0
    activation_margin_bytes: int = 0

    def __post_init__(self) -> None:
        if not self.name or not self.name.strip():
            raise ValueError("Named KV branch name must be non-empty.")
        if self.memory_bytes <= 0:
            raise ValueError("Named KV branch memory_bytes must be positive.")
        if self.layer_group < 0:
            raise ValueError("Named KV branch layer_group must be non-negative.")
        if self.activation_margin_bytes < 0:
            raise ValueError("Named KV branch activation_margin_bytes must be non-negative.")


class _NamedKVRequestAdapter:
    """Minimal request adapter for vLLM's KVCacheManager (caching disabled).

    Duck-types the subset of ``vllm.v1.request.Request`` that
    ``KVCacheManager.allocate_slots`` / ``free`` read. With prefix caching
    disabled, only request_id, computed/prompt/in-flight lengths and status
    are consumed; the remaining fields are provided for forward-compatibility.
    """

    __slots__ = (
        "request_id",
        "num_computed_tokens",
        "num_prompt_tokens",
        "num_in_flight_tokens",
        "status",
        "block_hashes",
        "skip_reading_prefix_cache",
        "num_preemptions",
    )

    def __init__(self, request_id: str) -> None:
        self.request_id = request_id
        self.num_computed_tokens = 0
        self.num_prompt_tokens = 0
        self.num_in_flight_tokens = 0
        self.status = RequestStatus.RUNNING
        self.block_hashes: list = []
        self.skip_reading_prefix_cache = True
        self.num_preemptions = 0

    @property
    def num_tokens(self) -> int:
        return self.num_computed_tokens


def _build_named_kv_manager(
    spec: FullAttentionSpec,
    layer_names: list[str],
    num_blocks: int,
    max_model_len: int,
) -> KVCacheManager:
    """Build a dedicated KVCacheManager for one named branch pool.

    Uses a single full-attention group with prefix caching disabled. The
    manager owns block metadata only; GPU tensors remain branch-owned.
    """
    group = KVCacheGroupSpec(layer_names=list(layer_names), kv_cache_spec=spec)
    page = spec.page_size_bytes
    # Layer-outermost layout: each layer occupies a contiguous page*num_blocks
    # region; block b of layer l sits at l*(page*num_blocks) + b*page.
    tensors = [
        KVCacheTensor(
            size=page * num_blocks * len(layer_names),
            layers=list(layer_names),
            layer_stride=page * num_blocks,
            block_stride=page,
        )
    ]
    config = KVCacheConfig(
        num_blocks=num_blocks,
        kv_cache_tensors=tensors,
        kv_cache_groups=[group],
    )
    return KVCacheManager(
        config,
        max_model_len=max_model_len,
        scheduler_block_size=spec.block_size,
        hash_block_size=spec.block_size,
        enable_caching=False,
    )


@dataclass
class _NamedKVRequestState:
    """Per-request state: a manager-backed adapter plus a GPU block-table mirror.

    Block ownership lives in the ``KVCacheManager``; this state only holds the
    request adapter (for manager calls) and the GPU block-table tensor (for
    attention dispatch). ``block_ids`` is an immutable read-only mirror refreshed
    from the manager when blocks are allocated; it never owns or frees blocks.
    """

    request_id: str
    adapter: _NamedKVRequestAdapter
    max_blocks: int
    device: torch.device
    block_table: torch.Tensor = field(init=False)
    block_ids: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        self.block_table = torch.zeros(
            (1, self.max_blocks),
            dtype=torch.int32,
            device=self.device,
        )

    @property
    def num_tokens(self) -> int:
        return self.adapter.num_computed_tokens


class NamedCausalKVBranch:
    """Fixed-pool causal PagedAttention branch owned by a model runner."""

    def __init__(self, *, runner: Any, request: NamedKVBranchRequest) -> None:
        self.request = request
        self.name = request.name
        self.vllm_config = runner.vllm_config
        self.device = torch.device(runner.device)
        self._states: dict[str, _NamedKVRequestState] = {}
        self._entered = False
        self._closed = False
        self.max_concurrent_requests = int(self.vllm_config.scheduler_config.max_num_seqs)

        self._validate_runner_contract(runner)
        kv_group = runner.kv_cache_config.kv_cache_groups[request.layer_group]
        self.kv_cache_spec = kv_group.kv_cache_spec
        assert isinstance(self.kv_cache_spec, FullAttentionSpec)
        attention_groups = runner.attn_groups[request.layer_group]
        if len(attention_groups) != 1:
            raise ValueError(
                f"Named causal KV v1 requires exactly one homogeneous attention group, got {len(attention_groups)}."
            )
        attention_group = attention_groups[0]
        self.backend = attention_group.backend
        self.layer_names = list(attention_group.layer_names)
        if not self.layer_names:
            raise ValueError("Named causal KV branch has no attention layers.")
        self.layers = {
            name: self.vllm_config.compilation_config.static_forward_context[name] for name in self.layer_names
        }

        kernel_block_size = runner._kernel_block_sizes[request.layer_group]
        if self.kv_cache_spec.block_size != kernel_block_size:
            raise ValueError(
                "Named causal KV v1 requires scheduler and kernel block sizes "
                f"to match, got {self.kv_cache_spec.block_size} and "
                f"{kernel_block_size}."
            )
        self.block_size = int(kernel_block_size)
        self.max_sequence_tokens = int(self.vllm_config.model_config.max_model_len)
        self.max_blocks_per_request = ceil(self.max_sequence_tokens / self.block_size)

        bytes_per_block = len(self.layer_names) * self.kv_cache_spec.page_size_bytes
        self.num_blocks = request.memory_bytes // bytes_per_block
        # vLLM's BlockPool reserves one null block; usable capacity is N-1.
        usable_blocks = self.num_blocks - 1
        required_blocks = self.max_concurrent_requests * self.max_blocks_per_request
        if usable_blocks < required_blocks:
            capacity_tokens = usable_blocks * self.block_size
            required_tokens = self.max_concurrent_requests * self.max_sequence_tokens
            raise ValueError(
                "Named causal KV branch cannot hold the complete fixed-concurrency set: "
                f"max_concurrent_requests={self.max_concurrent_requests}, "
                f"capacity_tokens={capacity_tokens}, "
                f"required_tokens={required_tokens}, "
                f"memory_bytes={request.memory_bytes}."
            )
        self.allocated_memory_bytes = self.num_blocks * bytes_per_block
        self._preflight_device_memory()
        self._manager = _build_named_kv_manager(
            self.kv_cache_spec,
            self.layer_names,
            self.num_blocks,
            self.max_sequence_tokens,
        )
        if self._manager.block_pool.get_num_free_blocks() != usable_blocks:
            raise RuntimeError("Named KV manager reserved-block accounting differs from the capacity guard.")
        # Every append schedules exactly one token for one
        # request, so query_start_loc is the constant [0, 1] on both sides.
        # Hoist the two allocations+H2D out of the per-step path. The dynamic
        # scalars (slot_mapping/seq_lens/position) stay per-append: pinned
        # staging would need ring-buffer hazard handling that the batched
        # batched-metadata rewrite replaces anyway.
        self._query_start_cpu = torch.tensor([0, 1], dtype=torch.int32)
        self._query_start_gpu = self._query_start_cpu.to(self.device)
        self._raw_caches: list[torch.Tensor] = []
        self.kv_caches = self._allocate_kv_caches()

        builder_spec = self.kv_cache_spec.copy_with_new_block_size(kernel_block_size)
        self._metadata_builder = self.backend.get_builder_cls()(
            builder_spec,
            self.layer_names,
            self.vllm_config,
            self.device,
        )
        # Usable token slots exclude the manager's null block. Admission also
        # requires per-request block rounding, checked separately above.
        logger.info(
            "Initialized named causal KV branch %r: layers=%d physical_blocks=%d usable_blocks=%d "
            "block_size=%d usable_token_slots=%d max_concurrent_requests=%d "
            "memory_bytes=%d",
            self.name,
            len(self.layer_names),
            self.num_blocks,
            usable_blocks,
            self.block_size,
            usable_blocks * self.block_size,
            self.max_concurrent_requests,
            self.allocated_memory_bytes,
        )

    def _validate_runner_contract(self, runner: Any) -> None:
        config = runner.vllm_config
        request = self.request
        groups = runner.kv_cache_config.kv_cache_groups
        if request.layer_group >= len(groups):
            raise ValueError(f"Named KV layer_group={request.layer_group} is out of range for {len(groups)} KV groups.")
        spec = groups[request.layer_group].kv_cache_spec
        if not isinstance(spec, FullAttentionSpec):
            raise ValueError(f"Named causal KV v1 requires FullAttentionSpec, got {type(spec).__name__}.")
        if getattr(spec, "kv_quant_mode", KVQuantMode.NONE) != KVQuantMode.NONE:
            raise ValueError("Named causal KV v1 does not support quantized KV cache.")
        if config.scheduler_config.max_num_seqs < 1:
            raise ValueError("Named causal KV requires max_num_seqs to be positive.")
        if config.cache_config.enable_prefix_caching:
            raise ValueError("Named causal KV v1 requires enable_prefix_caching=False.")
        parallel = config.parallel_config
        if parallel.pipeline_parallel_size != 1:
            raise ValueError("Named causal KV v1 requires pipeline_parallel_size=1.")
        if parallel.prefill_context_parallel_size != 1:
            raise ValueError("Named causal KV v1 requires prefill_context_parallel_size=1.")
        if parallel.decode_context_parallel_size != 1:
            raise ValueError("Named causal KV v1 requires decode_context_parallel_size=1.")
        if parallel.use_ubatching:
            raise ValueError("Named causal KV v1 does not support ubatching.")
        if config.model_config.enable_sleep_mode:
            raise ValueError("Named causal KV v1 does not support sleep mode.")
        if config.speculative_config is not None:
            raise ValueError("Named causal KV v1 does not support speculative decode.")
        if not config.model_config.enforce_eager:
            # Models must keep allocation and dynamic context management out
            # of captured forward(). An independent branch executor may also
            # compile/capture model computation; this store does not select it.
            logger.info(
                "Named causal KV branch %r running with enforce_eager=False; "
                "branch bookkeeping remains outside captured forward; "
                "branch execution mode is selected by the model.",
                self.name,
            )
        transfer = config.kv_transfer_config
        if transfer is not None and transfer.kv_connector is not None:
            raise ValueError("Named causal KV v1 does not support KV connectors.")

        usable_positive_blocks = runner.kv_cache_config.num_blocks - 1
        positive_capacity_tokens = usable_positive_blocks * spec.block_size
        required_positive_tokens = config.scheduler_config.max_num_seqs * config.model_config.max_model_len
        required_positive_blocks = config.scheduler_config.max_num_seqs * ceil(
            config.model_config.max_model_len / spec.block_size
        )
        if usable_positive_blocks < required_positive_blocks:
            raise ValueError(
                "Positive KV pool cannot hold the complete fixed-concurrency set: "
                f"max_concurrent_requests={config.scheduler_config.max_num_seqs}, "
                f"capacity_tokens={positive_capacity_tokens}, "
                f"required_tokens={required_positive_tokens}."
            )

    def _preflight_device_memory(self) -> None:
        if self.device.type != "cuda":
            raise ValueError("Named causal KV v1 currently requires a CUDA runner device.")
        free_bytes = current_omni_platform.get_free_memory(self.device)
        required_bytes = self.allocated_memory_bytes + self.request.activation_margin_bytes
        if free_bytes < required_bytes:
            raise MemoryError(
                "Insufficient free VRAM for named causal KV branch: "
                f"free={free_bytes}, branch={self.allocated_memory_bytes}, "
                f"activation_margin={self.request.activation_margin_bytes}."
            )

    def _allocate_kv_caches(self) -> dict[str, torch.Tensor]:
        if self.kv_cache_spec.page_size_padded is not None:
            raise ValueError("Named causal KV v1 does not support padded KV cache pages.")
        # Allocate one independent contiguous tensor per layer in the HND
        # logical layout [num_blocks, num_kv_heads, block_size, 2*head_size]
        # that the FA3 adapter and custom op consume. vLLM 0.29 centralized
        # allocation behind allocate_kv_cache, but that aliases all layers to
        # one backing buffer, which torch.compile rejects; independent tensors
        # preserve the capturable path.
        logical_shape = (
            self.num_blocks,
            self.kv_cache_spec.num_kv_heads,
            self.block_size,
            2 * self.kv_cache_spec.head_size,
        )
        kv_caches: dict[str, torch.Tensor] = {}
        for layer_name in self.layer_names:
            raw_cache = torch.empty(
                logical_shape,
                dtype=self.kv_cache_spec.dtype,
                device=self.device,
            )
            if raw_cache.numel() * raw_cache.element_size() != self.num_blocks * self.kv_cache_spec.page_size_bytes:
                raise AssertionError("Named causal KV allocation does not match page-size accounting.")
            self._raw_caches.append(raw_cache)
            kv_caches[layer_name] = raw_cache
        return kv_caches

    @property
    def num_free_blocks(self) -> int:
        return self._manager.block_pool.get_num_free_blocks()

    def reset(self, request_id: str) -> None:
        self._ensure_open()
        self._ensure_not_entered("reset")
        self._free_unchecked(request_id)
        self._states[request_id] = _NamedKVRequestState(
            request_id=request_id,
            adapter=_NamedKVRequestAdapter(request_id),
            max_blocks=self.max_blocks_per_request,
            device=self.device,
        )

    def free(self, request_id: str) -> None:
        if self._closed:
            return
        self._ensure_not_entered("free")
        self._free_unchecked(request_id)

    def _free_unchecked(self, request_id: str) -> None:
        """Release one request from internal cleanup paths.

        Public reset/free operations are rejected while the branch attention
        context is active. Fault cleanup intentionally bypasses that guard so
        a partial layer write can still invalidate and release the request.
        """
        state = self._states.pop(request_id, None)
        if state is not None:
            self._manager.free(state.adapter)

    def _cleanup_after_fault(self, request_id: str) -> None:
        """Best-effort cleanup that never masks the active forward exception."""
        try:
            self._free_unchecked(request_id)
        except Exception:
            logger.exception(
                "Failed to release named-KV request %r after a branch fault.",
                request_id,
            )

    def get_sequence_length(self, request_id: str) -> int:
        state = self._states.get(request_id)
        return state.num_tokens if state is not None else 0

    def _append_slots(
        self,
        request_ids: list[str],
    ) -> tuple[list[_NamedKVRequestState], list[int], list[int]]:
        """Bookkeep one causal slot per request before any context is built.

        Validates the complete batch first so a mid-batch failure cannot leave
        some requests advanced and others untouched. On any bookkeeping
        failure every touched request is fault-freed, matching the logical
        batch contract of the model-side caller.
        """
        states: list[_NamedKVRequestState] = []
        for request_id in request_ids:
            state = self._states.get(request_id)
            if state is None:
                raise RuntimeError(f"Named causal KV request {request_id!r} must be reset before append.")
            if state.num_tokens >= self.max_sequence_tokens:
                raise RuntimeError(
                    f"Named causal KV request {request_id!r} exceeded max_sequence_tokens={self.max_sequence_tokens}."
                )
            states.append(state)

        positions: list[int] = []
        slot_values: list[int] = []
        try:
            for state in states:
                position_value = state.num_tokens
                block_index = position_value // self.block_size
                if position_value % self.block_size == 0:
                    # FullAttention, no caching/lookahead/preemption: existing
                    # blocks remain valid until request free. Only boundaries
                    # need manager allocation; progress still advances per token.
                    allocated = self._manager.allocate_slots(state.adapter, num_new_tokens=1)
                    if allocated is None:
                        raise RuntimeError(
                            "Named causal KV branch exhausted its fixed GPU block pool. "
                            "This violates the startup capacity guard."
                        )
                    block_ids = tuple(self._manager.get_block_ids(state.request_id)[0])
                    state.block_table[0, block_index] = block_ids[block_index]
                    state.block_ids = block_ids
                slot_values.append(state.block_ids[block_index] * self.block_size + position_value % self.block_size)
                state.adapter.num_computed_tokens += 1
                positions.append(position_value)
        except Exception:
            for request_id in request_ids:
                self._cleanup_after_fault(request_id)
            raise
        return states, positions, slot_values

    @contextmanager
    def _bind_branch_kv_caches(self) -> Iterator[None]:
        """Restore original cache references, including partial bind failure."""
        originals = {name: layer.kv_cache for name, layer in self.layers.items()}
        try:
            for name, layer in self.layers.items():
                layer.kv_cache = self.kv_caches[name]
            yield
        finally:
            for name, layer in self.layers.items():
                layer.kv_cache = originals[name]

    @contextmanager
    def append_and_enter(
        self,
        request_id: str,
    ) -> Iterator[NamedKVBranchStep]:
        """Append one causal slot and enter its eager attention context."""
        self._ensure_open()
        if self._entered:
            raise RuntimeError("Named causal KV branch contexts cannot be re-entered.")
        states, positions, slot_values = self._append_slots([request_id])
        state = states[0]
        position_value = positions[0]
        slot_value = slot_values[0]

        try:
            slot_mapping = torch.tensor(
                [slot_value],
                dtype=torch.int64,
                device=self.device,
            )
            seq_lens = torch.tensor(
                [state.num_tokens],
                dtype=torch.int32,
                device=self.device,
            )
            position = torch.tensor(
                [position_value],
                dtype=torch.long,
                device=self.device,
            )
            common = CommonAttentionMetadata(
                query_start_loc=self._query_start_gpu,
                query_start_loc_cpu=self._query_start_cpu,
                seq_lens=seq_lens,
                num_reqs=1,
                num_actual_tokens=1,
                max_query_len=1,
                max_seq_len=state.num_tokens,
                block_table_tensor=state.block_table,
                slot_mapping=slot_mapping,
                causal=True,
                positions=position,
            )
            metadata = self._metadata_builder.build(0, common)
            context = create_forward_context(
                {name: metadata for name in self.layer_names},
                self.vllm_config,
                slot_mapping={name: slot_mapping for name in self.layer_names},
                skip_compiled=True,
            )
        except Exception:
            self._cleanup_after_fault(request_id)
            raise

        self._entered = True
        try:
            with self._bind_branch_kv_caches(), override_forward_context(context):
                yield NamedKVBranchStep(
                    position=position,
                    sequence_length=state.num_tokens,
                )
        except Exception:
            # A partial layer write cannot be rolled back safely. Drop the
            # entire request branch so stale KV is never reused. This internal
            # path must remain legal while the forward context is entered.
            self._cleanup_after_fault(request_id)
            raise
        finally:
            self._entered = False

    @contextmanager
    def append_batch(
        self,
        request_ids: list[str],
    ) -> Iterator[NamedKVAppendBatch]:
        """Append one causal slot per request without switching attention.

        Unlike :meth:`append_and_enter_batch`, this does NOT swap
        ``layer.kv_cache`` or override the forward context.  The executor
        owns compilation and attention dispatch.  This context manager only
        provides state protection and fault cleanup, located outside the
        compilation region.
        """
        self._ensure_open()
        if self._entered:
            raise RuntimeError("Named causal KV branch contexts cannot be re-entered.")
        if not request_ids or len(request_ids) != len(set(request_ids)):
            raise ValueError("Named causal KV batch append requires distinct, non-empty request IDs.")

        # _append_slots has internal try/except cleanup: on failure it
        # fault-frees every touched request and re-raises.
        states, positions, slot_values = self._append_slots(request_ids)
        seq_lens_list = [state.num_tokens for state in states]

        self._entered = True
        try:
            yield NamedKVAppendBatch(
                request_ids=tuple(request_ids),
                positions=tuple(positions),
                slot_values=tuple(slot_values),
                seq_lens=tuple(seq_lens_list),
                block_ids=tuple(tuple(s.block_ids) for s in states),
            )
        except Exception:
            for request_id in request_ids:
                self._cleanup_after_fault(request_id)
            raise
        finally:
            self._entered = False

    @contextmanager
    def append_and_enter_batch(
        self,
        request_ids: list[str],
    ) -> Iterator[NamedKVBranchStep]:
        """Append one causal slot per request and enter one shared context.

        The negative Qwen branch advances
        every active request in ONE varlen decode forward instead of B
        sequential forwards. One metadata build, one kv_cache swap, one
        forward-context override. Fault handling drops the whole logical
        batch, matching the model-side caller contract.
        """
        self._ensure_open()
        if self._entered:
            raise RuntimeError("Named causal KV branch contexts cannot be re-entered.")
        if not request_ids or len(request_ids) != len(set(request_ids)):
            raise ValueError("Named causal KV batch append requires distinct, non-empty request IDs.")

        states, positions, slot_values = self._append_slots(request_ids)
        batch_size = len(request_ids)
        seq_lens_list = [state.num_tokens for state in states]
        try:
            slot_mapping = torch.tensor(
                slot_values,
                dtype=torch.int64,
                device=self.device,
            )
            query_start_cpu = torch.arange(
                0,
                batch_size + 1,
                dtype=torch.int32,
            )
            query_start = query_start_cpu.to(self.device)
            seq_lens_cpu = torch.tensor(seq_lens_list, dtype=torch.int32)
            seq_lens = seq_lens_cpu.to(self.device)
            position = torch.tensor(
                positions,
                dtype=torch.long,
                device=self.device,
            )
            block_table = torch.cat(
                [state.block_table for state in states],
                dim=0,
            )
            common = CommonAttentionMetadata(
                query_start_loc=query_start,
                query_start_loc_cpu=query_start_cpu,
                seq_lens=seq_lens,
                num_reqs=batch_size,
                num_actual_tokens=batch_size,
                max_query_len=1,
                max_seq_len=max(seq_lens_list),
                block_table_tensor=block_table,
                slot_mapping=slot_mapping,
                causal=True,
                positions=position,
            )
            metadata = self._metadata_builder.build(0, common)
            context = create_forward_context(
                {name: metadata for name in self.layer_names},
                self.vllm_config,
                slot_mapping={name: slot_mapping for name in self.layer_names},
                skip_compiled=True,
            )
        except Exception:
            for request_id in request_ids:
                self._cleanup_after_fault(request_id)
            raise

        self._entered = True
        try:
            with self._bind_branch_kv_caches(), override_forward_context(context):
                yield NamedKVBranchStep(
                    position=position,
                    sequence_length=max(seq_lens_list),
                )
        except Exception:
            for request_id in request_ids:
                self._cleanup_after_fault(request_id)
            raise
        finally:
            self._entered = False

    def close(self) -> None:
        if self._closed:
            return
        if self._entered:
            raise RuntimeError("Cannot close a named causal KV branch in forward.")
        for request_id in list(self._states):
            self.free(request_id)
        self.kv_caches.clear()
        self._raw_caches.clear()
        self._closed = True

    def _ensure_open(self) -> None:
        if self._closed:
            raise RuntimeError(f"Named causal KV branch {self.name!r} is closed.")

    def _ensure_not_entered(self, operation: str) -> None:
        if self._entered:
            raise RuntimeError(
                f"Cannot {operation} named causal KV branch {self.name!r} while its forward context is active."
            )


__all__ = [
    "NamedCausalKVBranch",
    "NamedKVAppendBatch",
    "NamedKVBranchRequest",
    "NamedKVBranchStep",
]
