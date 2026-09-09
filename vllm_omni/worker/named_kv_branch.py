# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Optional runner-owned causal PagedAttention KV branches.

The implementation is deliberately narrow: a fixed scheduler-bounded number
of requests, one full-attention layer group, eager execution, and a fixed GPU
pool with no overcommit. Models that do not declare a
:class:`NamedKVBranchRequest` never construct this store.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from math import ceil
from typing import Any

import torch
from vllm.config import set_current_vllm_config
from vllm.forward_context import create_forward_context, override_forward_context
from vllm.logger import init_logger
from vllm.v1.attention.backend import CommonAttentionMetadata
from vllm.v1.kv_cache_interface import FullAttentionSpec, KVQuantMode

from vllm_omni.platforms import current_omni_platform

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


@dataclass(frozen=True)
class NamedKVBranchStep:
    """Metadata returned to a model while one branch append is active."""

    position: torch.Tensor
    #: Current sequence length for observability and test assertions.
    #: Production code reads only ``position``; this field is not consumed
    #: by the model forward path.
    sequence_length: int


class _FixedBlockAllocator:
    """Small deterministic allocator for a fixed, non-overcommitted block pool."""

    def __init__(self, num_blocks: int) -> None:
        if num_blocks < 1:
            raise ValueError("Fixed block allocator requires at least one block.")
        self.num_blocks = int(num_blocks)
        self._free_blocks = list(range(self.num_blocks - 1, -1, -1))
        self._allocated_blocks: set[int] = set()

    @property
    def num_free_blocks(self) -> int:
        return len(self._free_blocks)

    def allocate(self) -> int:
        if not self._free_blocks:
            raise RuntimeError(
                "Named causal KV branch exhausted its fixed GPU block pool. This violates the startup capacity guard."
            )
        block_id = self._free_blocks.pop()
        if block_id in self._allocated_blocks:
            raise AssertionError(f"KV block {block_id} was allocated twice.")
        self._allocated_blocks.add(block_id)
        return block_id

    def free(self, block_ids: list[int]) -> None:
        for block_id in reversed(block_ids):
            if block_id not in self._allocated_blocks:
                raise ValueError(f"Cannot free unallocated named-KV block {block_id}.")
            self._allocated_blocks.remove(block_id)
            self._free_blocks.append(block_id)


@dataclass
class _NamedKVRequestState:
    max_blocks: int
    device: torch.device
    block_ids: list[int] = field(default_factory=list)
    num_tokens: int = 0
    block_table: torch.Tensor = field(init=False)

    def __post_init__(self) -> None:
        self.block_table = torch.zeros(
            (1, self.max_blocks),
            dtype=torch.int32,
            device=self.device,
        )


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
        required_blocks = self.max_concurrent_requests * self.max_blocks_per_request
        if self.num_blocks < required_blocks:
            capacity_tokens = self.num_blocks * self.block_size
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
        self._allocator = _FixedBlockAllocator(self.num_blocks)
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
        logger.info(
            "Initialized named causal KV branch %r: layers=%d blocks=%d "
            "block_size=%d capacity_tokens=%d max_concurrent_requests=%d "
            "memory_bytes=%d",
            self.name,
            len(self.layer_names),
            self.num_blocks,
            self.block_size,
            self.num_blocks * self.block_size,
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
            # The negative branch itself always runs eager (dynamic metadata
            # + override_forward_context are not capturable). Models that
            # declare preprocess_finalize move the negative-branch work out of
            # forward(), so vLLM may capture the positive forward as a FULL
            # decode graph while the negative branch stays eager outside the
            # capture region. The strict enforce_eager requirement is lifted;
            # the model is responsible for keeping non-capturable work out of
            # forward().
            logger.info(
                "Named causal KV branch %r running with enforce_eager=False; "
                "the negative branch stays eager and the positive forward may "
                "use CUDA graphs.",
                self.name,
            )
        transfer = config.kv_transfer_config
        if transfer is not None and transfer.kv_connector is not None:
            raise ValueError("Named causal KV v1 does not support KV connectors.")

        positive_capacity_tokens = runner.kv_cache_config.num_blocks * spec.block_size
        required_positive_tokens = config.scheduler_config.max_num_seqs * config.model_config.max_model_len
        if positive_capacity_tokens < required_positive_tokens:
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
        cache_dtype_str = (
            getattr(self.kv_cache_spec, "cache_dtype_str", None) or self.vllm_config.cache_config.cache_dtype
        )
        cache_shape = self.backend.get_kv_cache_shape(
            self.num_blocks,
            self.block_size,
            self.kv_cache_spec.num_kv_heads,
            self.kv_cache_spec.head_size,
            cache_dtype_str=cache_dtype_str,
        )
        if self.kv_cache_spec.page_size_padded is not None:
            raise ValueError("Named causal KV v1 does not support padded KV cache pages.")
        with set_current_vllm_config(self.vllm_config):
            stride_order = self.backend.get_kv_cache_stride_order()
        permuted_shape = tuple(cache_shape[index] for index in stride_order)
        inverse_order = [stride_order.index(index) for index in range(len(stride_order))]

        kv_caches: dict[str, torch.Tensor] = {}
        for layer_name in self.layer_names:
            raw_cache = torch.empty(
                permuted_shape,
                dtype=self.kv_cache_spec.dtype,
                device=self.device,
            )
            if raw_cache.numel() * raw_cache.element_size() != self.num_blocks * self.kv_cache_spec.page_size_bytes:
                raise AssertionError("Named causal KV allocation does not match page-size accounting.")
            self._raw_caches.append(raw_cache)
            kv_caches[layer_name] = raw_cache.permute(*inverse_order)
        return kv_caches

    @property
    def num_free_blocks(self) -> int:
        return self._allocator.num_free_blocks

    def reset(self, request_id: str) -> None:
        self._ensure_open()
        self._ensure_not_entered("reset")
        self._free_unchecked(request_id)
        self._states[request_id] = _NamedKVRequestState(
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
            self._allocator.free(state.block_ids)

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
                if position_value % self.block_size == 0:
                    block_id = self._allocator.allocate()
                    state.block_ids.append(block_id)
                    state.block_table[0, len(state.block_ids) - 1] = block_id
                block_id = state.block_ids[position_value // self.block_size]
                slot_values.append(block_id * self.block_size + position_value % self.block_size)
                state.num_tokens += 1
                positions.append(position_value)
        except Exception:
            for request_id in request_ids:
                self._cleanup_after_fault(request_id)
            raise
        return states, positions, slot_values

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

        positive_caches = {name: layer.kv_cache for name, layer in self.layers.items()}
        self._entered = True
        try:
            for name, layer in self.layers.items():
                layer.kv_cache = self.kv_caches[name]
            with override_forward_context(context):
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
            for name, layer in self.layers.items():
                layer.kv_cache = positive_caches[name]
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

        positive_caches = {name: layer.kv_cache for name, layer in self.layers.items()}
        self._entered = True
        try:
            for name, layer in self.layers.items():
                layer.kv_cache = self.kv_caches[name]
            with override_forward_context(context):
                yield NamedKVBranchStep(
                    position=position,
                    sequence_length=max(seq_lens_list),
                )
        except Exception:
            for request_id in request_ids:
                self._cleanup_after_fault(request_id)
            raise
        finally:
            for name, layer in self.layers.items():
                layer.kv_cache = positive_caches[name]
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
    "NamedKVBranchRequest",
    "NamedKVBranchStep",
]
