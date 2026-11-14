# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Legacy V1 aligned Mamba cache buffers and lifecycle from vLLM 4ba787ec99.

Native 0.34 retains the shared copy kernels, but removed V1 CPU/GPU staging
and scalar-copy helpers. Reuse its kernels with the original V1 defaults;
keep the separate V1 buffer ABI here. Resumed requests now arrive as full
scheduled_new_reqs, so their old state indices must be discarded.
"""

import dataclasses
import itertools
from collections.abc import Callable
from typing import Any, NamedTuple

import torch
from vllm.config import CacheConfig
from vllm.logger import init_logger
from vllm.model_executor.layers.mamba.mamba_utils import (
    MambaStateCopyFuncsByType,
    get_conv_copy_spec,
    get_temporal_copy_spec,
    is_conv_state_dim_first,
)
from vllm.triton_utils import tl, triton
from vllm.utils.gpu_sync_debug import gpu_sync_allowed
from vllm.utils.math_utils import cdiv
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.kv_cache_interface import KVCacheConfig, MambaSpec
from vllm.v1.utils import CpuGpuBuffer
from vllm.v1.worker.mamba_utils import (
    _get_mamba_spec_for_layer,
    get_mamba_group_ids,
    get_mamba_groups,
    postprocess_mamba_fused_kernel,
    precopy_mamba_align_fused_kernel,
)
from vllm.v1.worker.mamba_utils import (
    validate_mamba_state_copy_funcs as validate_mamba_state_copy_funcs,
)

from vllm_omni.worker.v1_compat.input_batch import CachedRequestState, InputBatch

logger = init_logger(__name__)
_TEMPORAL_TILES = 16


def _reinterpret_u64_as_i64(value: int) -> int:
    """Preserve a uint64 pointer bit pattern in a torch.int64 tensor."""
    return value if value < (1 << 63) else value - (1 << 64)


@triton.jit
def batch_memcpy_kernel(src_ptrs, dst_ptrs, sizes, BLOCK_SIZE: tl.constexpr):  # noqa: N803
    pid = tl.program_id(0)

    src_ptr = tl.load(src_ptrs + pid)
    dst_ptr = tl.load(dst_ptrs + pid)
    size = tl.load(sizes + pid)
    is_left_overlap = dst_ptr < src_ptr and dst_ptr + size > src_ptr

    offsets = tl.arange(0, BLOCK_SIZE)
    for i in range(0, size, BLOCK_SIZE):
        mask = (i + offsets) < size

        curr_src_ptr = (src_ptr + i + offsets).to(tl.pointer_type(tl.uint8))
        curr_dst_ptr = (dst_ptr + i + offsets).to(tl.pointer_type(tl.uint8))

        data = tl.load(curr_src_ptr, mask=mask)
        if is_left_overlap:
            # Preserve each lane's source before a lower-address lane stores
            # over it. The condition is uniform within the program.
            tl.debug_barrier()
        tl.store(curr_dst_ptr, data, mask=mask)


def batch_memcpy(src_ptrs, dst_ptrs, sizes):
    batch = src_ptrs.shape[0]
    assert dst_ptrs.shape[0] == batch
    assert sizes.shape[0] == batch

    grid = (batch,)
    BLOCK_SIZE = 1024
    batch_memcpy_kernel[grid](src_ptrs, dst_ptrs, sizes, BLOCK_SIZE=BLOCK_SIZE)


@dataclasses.dataclass
class MambaCopyBuffers:
    src_ptrs: CpuGpuBuffer
    dst_ptrs: CpuGpuBuffer
    sizes: CpuGpuBuffer
    mamba_group_ids: list[int]
    mamba_spec: MambaSpec
    offset: int = 0

    @classmethod
    def create(
        cls,
        max_num_reqs: int,
        kv_cache_config: KVCacheConfig,
        copy_funcs: MambaStateCopyFuncsByType,
        make_buffer: Callable[..., CpuGpuBuffer],
    ) -> "MambaCopyBuffers":
        mamba_groups = get_mamba_groups(kv_cache_config)
        mamba_spec = next(iter(mamba_groups))
        assert all(
            spec.block_size == mamba_spec.block_size
            and spec.num_speculative_blocks == mamba_spec.num_speculative_blocks
            and spec.mamba_cache_mode == mamba_spec.mamba_cache_mode
            for spec in mamba_groups
        ), "all mamba groups must share cache scheduling parameters"
        mamba_group_ids = get_mamba_group_ids(mamba_groups)
        entries_per_req = sum(
            len(copy_funcs[_get_mamba_spec_for_layer(kv_cache_config.kv_cache_groups[gid], layer_name).mamba_type])
            for gid in mamba_group_ids
            for layer_name in kv_cache_config.kv_cache_groups[gid].layer_names
        )
        n = max_num_reqs * entries_per_req

        return cls(
            src_ptrs=make_buffer(n, dtype=torch.uint64),
            dst_ptrs=make_buffer(n, dtype=torch.uint64),
            sizes=make_buffer(n, dtype=torch.int32),
            mamba_group_ids=mamba_group_ids,
            mamba_spec=mamba_spec,
        )


@dataclasses.dataclass
class MambaSpecDecodeGPUContext:
    """Context for GPU-side Mamba state copy operations during the
    fused postprocess path.

    Only used when speculative decoding is enabled on a hybrid model
    (and the mamba_cache_config is in align mode).

    Precomputes memory layout metadata (base addresses, strides, element sizes)
    so the GPU kernel can perform state copies without CPU-GPU sync.

    State types are distinguished by conv_width: >0 for conv states (sliding
    window with offset-based copies), 0 for temporal states (full block copies).
    """

    # Per-state metadata tensors (shape: [num_states]). Mamba groups may have
    # different numbers of state types (for example, GDN has two while PLE
    # short-conv has one).
    # These are populated from forward_context during the first forward pass
    state_base_addrs: torch.Tensor  # int64: base address of each state tensor
    state_block_strides: torch.Tensor  # int64: bytes per block
    state_elem_sizes: torch.Tensor  # int32: element size in bytes
    state_inner_sizes: torch.Tensor  # int64: elements in inner dimensions
    state_conv_widths: torch.Tensor  # int32: conv width (0 for temporal states)
    state_group_indices: torch.Tensor  # int32: maps state_idx to group index
    # DS conv row metadata. Zero keeps the single-region copy path.
    state_dim_row_count: torch.Tensor  # int32: per-block dim row count
    state_dim_row_stride: torch.Tensor  # int64: bytes between rows

    # Configuration
    block_size: int
    num_states: int
    mamba_group_ids: list[int]
    num_groups: int

    # Output buffer for num_accepted_tokens updates
    num_accepted_tokens_out: torch.Tensor

    # Per-group block-table base addresses: int64[num_groups]. Populated in
    # initialize_from_forward_context from the persistent per-group block
    # table tensors (whose data_ptr is stable across steps). Rows are request
    # state slots for the state copies and batch rows for the aligned indices.
    block_table_ptrs: torch.Tensor
    aligned_index_block_table_ptrs: torch.Tensor
    block_table_stride_req: int = 0

    # persistent output for the once-per-step, all-group aligned-index launch.
    # shape: [num_groups, max_num_reqs, 1 + num_speculative_blocks].
    aligned_state_indices: torch.Tensor | None = None

    # Per-request staging buffers (CPU+GPU mirrors). The runner stages
    # values into the CPU view in ``_prepare_inputs`` and the fused kernel
    # reads the GPU side. These only exist when the postprocess kernel is
    # enabled (spec decode + hybrid + align mode).
    mamba_state_idx_buf: CpuGpuBuffer | None = None
    num_scheduled_tokens_buf: CpuGpuBuffer | None = None
    num_computed_tokens_buf: CpuGpuBuffer | None = None
    num_draft_tokens_buf: CpuGpuBuffer | None = None
    precopy_src_col_buf: CpuGpuBuffer | None = None
    precopy_token_bias_buf: CpuGpuBuffer | None = None

    # Flag to track if metadata has been populated
    is_initialized: bool = False

    @classmethod
    def create(
        cls,
        max_num_reqs: int,
        kv_cache_config: KVCacheConfig,
        copy_funcs: MambaStateCopyFuncsByType,
        device: torch.device,
        make_buffer: Callable[..., CpuGpuBuffer],
    ) -> "MambaSpecDecodeGPUContext":
        """Create context with allocated buffers (metadata populated later)."""
        mamba_groups = get_mamba_groups(kv_cache_config)
        mamba_group_ids = get_mamba_group_ids(mamba_groups)
        mamba_spec = next(iter(mamba_groups))
        assert all(
            spec.block_size == mamba_spec.block_size
            and spec.num_speculative_blocks == mamba_spec.num_speculative_blocks
            and spec.mamba_cache_mode == mamba_spec.mamba_cache_mode
            for spec in mamba_groups
        ), "all mamba groups must share cache scheduling parameters"
        copy_funcs_by_spec = {spec: copy_funcs[spec.mamba_type] for spec in mamba_groups}

        # Count physical state tensors across all Mamba groups. Different
        # groups may expose different state specs.
        total_states = sum(
            len(copy_funcs_by_spec[_get_mamba_spec_for_layer(kv_cache_config.kv_cache_groups[gid], layer_name)])
            for gid in mamba_group_ids
            for layer_name in kv_cache_config.kv_cache_groups[gid].layer_names
        )

        return cls(
            state_base_addrs=torch.zeros(total_states, dtype=torch.int64, device=device),
            state_block_strides=torch.zeros(total_states, dtype=torch.int64, device=device),
            state_elem_sizes=torch.zeros(total_states, dtype=torch.int32, device=device),
            state_inner_sizes=torch.zeros(total_states, dtype=torch.int64, device=device),
            state_conv_widths=torch.zeros(total_states, dtype=torch.int32, device=device),
            state_group_indices=torch.zeros(total_states, dtype=torch.int32, device=device),
            state_dim_row_count=torch.zeros(total_states, dtype=torch.int32, device=device),
            state_dim_row_stride=torch.zeros(total_states, dtype=torch.int64, device=device),
            block_size=mamba_spec.block_size,
            num_states=total_states,
            mamba_group_ids=mamba_group_ids,
            num_groups=len(mamba_group_ids),
            num_accepted_tokens_out=torch.zeros(max_num_reqs, dtype=torch.int32, device=device),
            block_table_ptrs=torch.zeros(len(mamba_group_ids), dtype=torch.int64, device=device),
            aligned_index_block_table_ptrs=torch.zeros(len(mamba_group_ids), dtype=torch.int64, device=device),
            aligned_state_indices=torch.empty(
                (
                    len(mamba_group_ids),
                    max_num_reqs,
                    1 + mamba_spec.num_speculative_blocks,
                ),
                dtype=torch.int32,
                device=device,
            ),
            mamba_state_idx_buf=make_buffer(max_num_reqs, dtype=torch.int32),
            num_scheduled_tokens_buf=make_buffer(max_num_reqs, dtype=torch.int32),
            num_computed_tokens_buf=make_buffer(max_num_reqs, dtype=torch.int32),
            num_draft_tokens_buf=make_buffer(max_num_reqs, dtype=torch.int32),
            precopy_src_col_buf=make_buffer(max_num_reqs, dtype=torch.int32),
            precopy_token_bias_buf=make_buffer(max_num_reqs, dtype=torch.int32),
            is_initialized=False,
        )

    def initialize_from_forward_context(
        self,
        kv_cache_config: KVCacheConfig,
        forward_context: dict[str, Any],
        mamba_state_copy_funcs: MambaStateCopyFuncsByType,
        block_tables: list[torch.Tensor],
        aligned_index_block_tables: list[torch.Tensor] | None = None,
    ) -> None:
        """Extract and cache memory layout metadata from Mamba state tensors.

        This method populates the pre-allocated metadata tensors with information
        needed by `postprocess_mamba_fused_kernel` to perform state copies entirely
        on the GPU without CPU-GPU synchronization.

        For each Mamba layer and state type, the following metadata is extracted:
        - state_base_addrs: GPU memory address (data_ptr) of the state tensor
        - state_block_strides: Bytes between consecutive blocks (stride * elem_size)
        - state_elem_sizes: Element size in bytes (e.g., 2 for float16)
        - state_inner_sizes: For conv states, elements per conv position (stride(1)),
          used to compute offset when slicing state[block, offset:]. For temporal
          states, this field is unused (set to 1).
        - state_conv_widths: Conv dimension size for conv states, 0 for temporal states

        The conv vs temporal state type is detected by inspecting the copy function
        name: functions containing "conv" are treated as conv states.

        This method is idempotent - it only executes once (guarded by is_initialized
        flag) since the metadata is static after model loading.

        Args:
            kv_cache_config: Configuration containing KV cache group info and
                layer name mappings.
            forward_context: Dictionary mapping layer names to attention objects,
                populated after the model is loaded. Each attention object must
                have a `kv_cache` attribute containing the list of state tensors.
            mamba_state_copy_funcs: Mapping from MambaAttentionBackendEnum to
                copy functions.
            block_tables: per-mamba-group persistent block-table tensors, in
                the same order as `mamba_group_ids`, with one row per request
                state slot. Their `data_ptr()` / `stride(0)` are captured once
                for the state-copy kernels to index into.
            aligned_index_block_tables: per-mamba-group persistent tables
                read by `compute_aligned_state_indices`, with one row per
                batch row. Defaults to `block_tables`.

        """
        if self.is_initialized:
            return
        # This only runs once per worker.
        with gpu_sync_allowed():
            self._populate_metadata(
                kv_cache_config,
                forward_context,
                mamba_state_copy_funcs,
                block_tables,
                aligned_index_block_tables or block_tables,
            )

    def _populate_metadata(
        self,
        kv_cache_config: KVCacheConfig,
        forward_context: dict[str, Any],
        mamba_state_copy_funcs: MambaStateCopyFuncsByType,
        block_tables: list[torch.Tensor],
        aligned_index_block_tables: list[torch.Tensor],
    ) -> None:
        idx = 0
        for group_local_idx, mamba_group_id in enumerate(self.mamba_group_ids):
            kv_cache_group = kv_cache_config.kv_cache_groups[mamba_group_id]
            layer_names = kv_cache_group.layer_names
            for layer_name in layer_names:
                mamba_spec = _get_mamba_spec_for_layer(kv_cache_group, layer_name)
                state_copy_funcs = mamba_state_copy_funcs[mamba_spec.mamba_type]
                attention = forward_context[layer_name]
                kv_caches: list[torch.Tensor] = attention.kv_cache
                if len(kv_caches) < len(state_copy_funcs):
                    raise ValueError(
                        f"Expected at least {len(state_copy_funcs)} Mamba state tensors, got {len(kv_caches)}"
                    )
                for state_type_idx, copy_func in enumerate(state_copy_funcs):
                    state = kv_caches[state_type_idx]
                    # Base address
                    self.state_base_addrs[idx] = _reinterpret_u64_as_i64(state.data_ptr())

                    # Block stride (bytes between consecutive blocks)
                    # state shape: [num_blocks, ...], stride(0) = elements per block
                    if state.dim() > 1:
                        block_stride_elems = state.stride(0)
                    else:
                        block_stride_elems = state.numel()
                    self.state_block_strides[idx] = block_stride_elems * state.element_size()

                    # Element size
                    self.state_elem_sizes[idx] = state.element_size()

                    assert copy_func is get_conv_copy_spec or copy_func is get_temporal_copy_spec, (
                        f"unexpected copy func: {copy_func}"
                    )
                    if copy_func is get_conv_copy_spec:
                        if state.dim() != 3:
                            raise ValueError(f"Expected 3D conv state cache, got shape {tuple(state.shape)}")
                        if is_conv_state_dim_first():
                            # DS layout: state_len is the slide axis.
                            self.state_conv_widths[idx] = state.size(2)
                            self.state_inner_sizes[idx] = 1
                            self.state_dim_row_count[idx] = state.size(1)
                            self.state_dim_row_stride[idx] = state.stride(1) * state.element_size()
                        else:
                            # SD layout: dim is contiguous.
                            self.state_conv_widths[idx] = state.size(1)
                            self.state_inner_sizes[idx] = state.stride(1)
                    else:
                        # Temporal state: inner_size = natural elements per
                        # block (prod of inner dims).  The kernel uses this
                        # to compute copy_size = inner_size * elem_size,
                        # which gives the correct byte count even when the
                        # state tensor is as_strided with padded page strides
                        # (state_block_stride would be the page size, too big).
                        self.state_conv_widths[idx] = 0
                        self.state_inner_sizes[idx] = state[0].numel() if state.dim() > 1 else 1
                        # Temporal copies vectorize with uint64 loads/stores.
                        # The kernel's head/tail handles misalignment for
                        # correctness, but unaligned base/stride costs in
                        # throughput.
                        base_addr = state.data_ptr()
                        block_stride_bytes = block_stride_elems * state.element_size()
                        if base_addr % 8 != 0:
                            logger.warning_once(
                                "layer %s: state.data_ptr() = %#x is not "
                                "8B-aligned; _memcpy_u64_tiled uint64 "
                                "vectorization will pay misaligned load cost",
                                layer_name,
                                base_addr,
                            )
                        if block_stride_bytes % 8 != 0:
                            logger.warning_once(
                                "layer %s: block stride = %dB is not "
                                "8B-aligned; _memcpy_u64_tiled uint64 "
                                "vectorization will pay misaligned load cost",
                                layer_name,
                                block_stride_bytes,
                            )

                    self.state_group_indices[idx] = group_local_idx
                    idx += 1

        assert idx == self.num_states

        # Cache per-group block-table base addresses and per-request stride.
        # `block_tables[i]` is the persistent 2D int32 block-table tensor for
        # `mamba_group_ids[i]`; `data_ptr()` / `stride(0)` are stable for the
        # engine's lifetime, so we capture them once here.
        assert len(block_tables) == len(aligned_index_block_tables) == self.num_groups
        strides = {bt.stride(0) for bt in (*block_tables, *aligned_index_block_tables)}
        assert len(strides) == 1, f"all mamba block tables must share stride(0), got {strides}"
        self.block_table_stride_req = int(next(iter(strides)))
        for i, (bt, aligned_bt) in enumerate(zip(block_tables, aligned_index_block_tables)):
            self.block_table_ptrs[i] = _reinterpret_u64_as_i64(bt.data_ptr())
            self.aligned_index_block_table_ptrs[i] = _reinterpret_u64_as_i64(aligned_bt.data_ptr())

        self.is_initialized = True

    def run_fused_postprocess(
        self,
        num_reqs: int,
        num_accepted_tokens_gpu: torch.Tensor,
        mamba_state_idx_gpu: torch.Tensor,
        num_scheduled_tokens_gpu: torch.Tensor,
        num_computed_tokens_gpu: torch.Tensor,
        num_draft_tokens_gpu: torch.Tensor,
    ) -> None:
        """Run the fused postprocess_mamba kernel on GPU.

        This computes decisions and performs mamba state copies entirely on GPU,
        eliminating the CPU-GPU sync that was previously needed.

        Args:
            num_reqs: Number of active requests
            num_accepted_tokens_gpu: [num_reqs] accepted token counts
            mamba_state_idx_gpu: [num_reqs] source block indices
            num_scheduled_tokens_gpu: [num_reqs] scheduled token counts
            num_computed_tokens_gpu: [num_reqs] computed token counts
            num_draft_tokens_gpu: [num_reqs] draft token counts

        """
        if num_reqs == 0 or not self.is_initialized:
            return

        # Initialize output to current values (unchanged unless src==dst)
        self.num_accepted_tokens_out[:num_reqs].copy_(num_accepted_tokens_gpu[:num_reqs])

        total_states = self.num_states
        grid = (num_reqs, total_states, _TEMPORAL_TILES)

        postprocess_mamba_fused_kernel[grid](
            num_accepted_tokens_gpu,
            mamba_state_idx_gpu,
            num_scheduled_tokens_gpu,
            num_computed_tokens_gpu,
            num_draft_tokens_gpu,
            self.block_table_ptrs,
            self.block_table_stride_req,
            self.state_base_addrs,
            self.state_block_strides,
            self.state_elem_sizes,
            self.state_inner_sizes,
            self.state_conv_widths,
            self.state_group_indices,
            self.state_dim_row_count,
            self.state_dim_row_stride,
            self.num_accepted_tokens_out,
            None,  # idx_mapping: V1 decision arrays are already in req order
            num_reqs,
            block_size=self.block_size,
            COPY_BLOCK_SIZE=1024,
            CONV_STATE_DIM_FIRST=is_conv_state_dim_first(),
            TEMPORAL_TILES=_TEMPORAL_TILES,
        )

    def run_fused_precopy(
        self,
        num_reqs: int,
        state_idx_gpu: torch.Tensor,
        src_col_gpu: torch.Tensor,
        token_bias_gpu: torch.Tensor,
        idx_mapping: torch.Tensor | None,
    ) -> None:
        """Pre-copy each request's previous running block into its new window
        block before the forward pass (align boundary migration).

        Args:
            num_reqs: Number of active requests (batch order).
            state_idx_gpu: [max_reqs] post-advance dst block column per req slot.
            src_col_gpu: [max_reqs] pre-advance src block column (-1 = fresh).
            token_bias_gpu: [max_reqs] accepted-token bias (num_accepted - 1).
            idx_mapping: optional [num_reqs] batch_idx -> req_state_idx.
                None means V1 batch order already equals request state order.

        """
        if num_reqs == 0 or not self.is_initialized:
            return
        total_states = self.num_states
        grid = (num_reqs, total_states, _TEMPORAL_TILES)
        precopy_mamba_align_fused_kernel[grid](
            state_idx_gpu,
            src_col_gpu,
            token_bias_gpu,
            self.block_table_ptrs,
            self.block_table_stride_req,
            self.state_base_addrs,
            self.state_block_strides,
            self.state_elem_sizes,
            self.state_inner_sizes,
            self.state_conv_widths,
            self.state_group_indices,
            self.state_dim_row_count,
            self.state_dim_row_stride,
            idx_mapping,
            num_reqs,
            COPY_BLOCK_SIZE=1024,
            CONV_STATE_DIM_FIRST=is_conv_state_dim_first(),
            HAS_IDX_MAPPING=idx_mapping is not None,
            TEMPORAL_TILES=_TEMPORAL_TILES,
        )


@dataclasses.dataclass
class MambaBuffers:
    """Single owner for all mamba-specific runner buffers.

    The two sub-objects have different gates:
    ``preprocess`` is needed whenever ``mamba_cache_mode == "align"``;
    ``postprocess_align`` is needed only when align is combined with
    speculative decoding on a hybrid model, and is ``None`` otherwise.
    """

    preprocess: MambaCopyBuffers
    postprocess_align: MambaSpecDecodeGPUContext | None

    @classmethod
    def create(
        cls,
        max_num_reqs: int,
        kv_cache_config: KVCacheConfig,
        copy_funcs: MambaStateCopyFuncsByType,
        make_buffer: Callable[..., CpuGpuBuffer],
        device: torch.device,
        with_postprocess_align: bool,
    ) -> "MambaBuffers":
        return cls(
            preprocess=MambaCopyBuffers.create(max_num_reqs, kv_cache_config, copy_funcs, make_buffer),
            postprocess_align=(
                MambaSpecDecodeGPUContext.create(
                    max_num_reqs=max_num_reqs,
                    kv_cache_config=kv_cache_config,
                    copy_funcs=copy_funcs,
                    device=device,
                    make_buffer=make_buffer,
                )
                if with_postprocess_align
                else None
            ),
        )


def collect_mamba_copy_meta(
    copy_bufs: MambaCopyBuffers,
    kv_cache_config: KVCacheConfig,
    mamba_state_copy_funcs: MambaStateCopyFuncsByType,
    mamba_group_ids: list[int],
    src_block_idx: int,
    dest_block_idx: int,
    accept_token_bias: int,
    req_state: CachedRequestState,
    forward_context: dict[str, Any],
) -> None:
    if src_block_idx == dest_block_idx and accept_token_bias == 0:
        return

    src_ptrs_np = copy_bufs.src_ptrs.np
    dst_ptrs_np = copy_bufs.dst_ptrs.np
    sizes_np = copy_bufs.sizes.np
    offset = copy_bufs.offset

    for mamba_group_id in mamba_group_ids:
        block_ids = req_state.block_ids[mamba_group_id]
        dest_block_id = block_ids[dest_block_idx]
        kv_cache_group = kv_cache_config.kv_cache_groups[mamba_group_id]
        layer_names = kv_cache_group.layer_names
        for layer_name in layer_names:
            mamba_spec = _get_mamba_spec_for_layer(kv_cache_group, layer_name)
            state_copy_funcs = mamba_state_copy_funcs[mamba_spec.mamba_type]
            attention = forward_context[layer_name]
            kv_caches: list[torch.Tensor] = attention.kv_cache
            for state, state_copy_func in zip(kv_caches, state_copy_funcs):
                copy_spec = state_copy_func(state, block_ids, src_block_idx, accept_token_bias + 1)

                src_ptrs_np[offset] = copy_spec.start_addr
                dst_ptrs_np[offset] = state[dest_block_id].data_ptr()
                sizes_np[offset] = copy_spec.num_elements * state.element_size()
                offset += 1

    copy_bufs.offset = offset


def do_mamba_copy_block(copy_bufs: MambaCopyBuffers):
    n = copy_bufs.offset
    if n == 0:
        return
    batch_memcpy(
        copy_bufs.src_ptrs.copy_to_gpu(n),
        copy_bufs.dst_ptrs.copy_to_gpu(n),
        copy_bufs.sizes.copy_to_gpu(n),
    )


def cleanup_mamba_state_idx(
    scheduler_output: SchedulerOutput,
    mamba_state_idx: dict[str, int],
) -> None:
    """Pop stale `mamba_state_idx` entries for finished/preempted/resumed reqs.

    Native 0.34 sends resumed requests through scheduled_new_reqs, including
    forced resumes without a preempted entry. Drop their previous block indices
    before looking up the new allocation, which may have fewer blocks.
    """
    finished_req_ids = scheduler_output.finished_req_ids
    preempted_req_ids = scheduler_output.preempted_req_ids or set()
    full_resend_req_ids = (req.req_id for req in scheduler_output.scheduled_new_reqs)
    for req_id in itertools.chain(finished_req_ids, preempted_req_ids, full_resend_req_ids):
        mamba_state_idx.pop(req_id, None)


class _FusedPrecopy(NamedTuple):
    """Resolved fused align pre-copy resources (all non-None once resolved)."""

    ctx: "MambaSpecDecodeGPUContext"
    state_idx: CpuGpuBuffer
    src_col: CpuGpuBuffer
    token_bias: CpuGpuBuffer


def _resolve_fused_precopy(
    align_ctx: "MambaSpecDecodeGPUContext | None",
) -> _FusedPrecopy | None:
    """Bundle the fused-path buffers, or None for the scalar path.

    Returning one non-None bundle lets callers narrow all four members with a
    single ``is not None`` check instead of re-asserting each buffer per use.
    """
    if align_ctx is None:
        return None
    assert align_ctx.mamba_state_idx_buf is not None
    assert align_ctx.precopy_src_col_buf is not None
    assert align_ctx.precopy_token_bias_buf is not None
    return _FusedPrecopy(
        align_ctx,
        align_ctx.mamba_state_idx_buf,
        align_ctx.precopy_src_col_buf,
        align_ctx.precopy_token_bias_buf,
    )


def preprocess_mamba(
    scheduler_output: SchedulerOutput,
    kv_cache_config: KVCacheConfig,
    cache_config: CacheConfig,
    mamba_state_idx: dict[str, int],
    input_batch: InputBatch,
    requests: dict[str, CachedRequestState],
    forward_context: dict[str, Any],
    mamba_state_copy_funcs: MambaStateCopyFuncsByType,
    copy_bufs: MambaCopyBuffers,
    align_ctx: MambaSpecDecodeGPUContext | None = None,
):
    """Copy the mamba state of previous step to the last
    (1 + num_speculative_blocks) block.
    """
    fused = _resolve_fused_precopy(align_ctx)
    mamba_group_ids = copy_bufs.mamba_group_ids
    mamba_spec = copy_bufs.mamba_spec
    num_speculative_blocks = mamba_spec.num_speculative_blocks
    # TODO(Chen): we need to optimize this function a lot
    assert cache_config.enable_prefix_caching
    block_size = mamba_spec.block_size
    cleanup_mamba_state_idx(scheduler_output, mamba_state_idx)

    copy_bufs.offset = 0
    num_reqs = len(input_batch.req_ids)

    if fused is not None:
        if num_reqs == 0:
            return
        if not fused.ctx.is_initialized:
            fused.ctx.initialize_from_forward_context(
                kv_cache_config,
                forward_context,
                mamba_state_copy_funcs,
                [input_batch.block_table[gid].get_device_tensor(num_reqs) for gid in fused.ctx.mamba_group_ids],
            )

        fused.src_col.np[:num_reqs] = -1
        fused.token_bias.np[:num_reqs] = 0

    for i, req_id in enumerate(input_batch.req_ids):
        req_state = requests[req_id]
        prev_state_idx = mamba_state_idx.get(req_id)
        if prev_state_idx is None:
            # New / resumed request; num_computed_tokens == 0 gives -1.
            prev_state_idx = (req_state.num_computed_tokens - 1) // block_size

        num_scheduled_tokens = scheduler_output.num_scheduled_tokens[req_id]
        num_blocks = cdiv(req_state.num_computed_tokens + num_scheduled_tokens, block_size) + num_speculative_blocks
        # We always save the current running state at the last
        # (1 + num_speculative_blocks) block.
        # A corner case worth mention here: assume we have block_size = 4 and
        # num_speculative_tokens = 2. The request is [A, B, C] and contains 2 draft
        # tokens [draft 1, draft 2]. Then we will have:
        # Block 0: [A, B, C, draft 1]
        # Block 1: [draft 2, TOFILL, TOFILL, TOFILL]
        # Block 2: speculative block
        # Block 3: speculative block
        # And use block 1 to save the running state.
        curr_state_idx = num_blocks - 1 - num_speculative_blocks
        mamba_state_idx[req_id] = curr_state_idx
        if fused is not None:
            fused.state_idx.np[i] = curr_state_idx

        if prev_state_idx != -1 and prev_state_idx != curr_state_idx:
            accept_token_bias = int(input_batch.num_accepted_tokens_cpu[i]) - 1
            if fused is not None:
                assert accept_token_bias >= 0
                fused.src_col.np[i] = prev_state_idx
                fused.token_bias.np[i] = accept_token_bias
            else:
                collect_mamba_copy_meta(
                    copy_bufs,
                    kv_cache_config,
                    mamba_state_copy_funcs,
                    mamba_group_ids,
                    prev_state_idx,
                    curr_state_idx,
                    accept_token_bias,
                    req_state,
                    forward_context,
                )
            input_batch.num_accepted_tokens_cpu[i] = 1

    if fused is not None:
        fused.state_idx.copy_to_gpu(num_reqs)
        fused.src_col.copy_to_gpu(num_reqs)
        fused.token_bias.copy_to_gpu(num_reqs)
        fused.ctx.run_fused_precopy(
            num_reqs=num_reqs,
            state_idx_gpu=fused.state_idx.gpu,
            src_col_gpu=fused.src_col.gpu,
            token_bias_gpu=fused.token_bias.gpu,
            idx_mapping=None,
        )
    else:
        do_mamba_copy_block(copy_bufs)


def postprocess_mamba_align_gpu(
    *,
    bufs: "MambaBuffers",
    num_reqs: int,
    num_accepted_tokens_gpu: torch.Tensor,
    num_accepted_tokens_cpu_tensor: torch.Tensor,
    input_batch: InputBatch,
    kv_cache_config: KVCacheConfig,
    forward_context: dict[str, Any],
    mamba_state_copy_funcs: MambaStateCopyFuncsByType,
) -> None:
    """GPU-side mamba postprocess for spec decode + hybrid + align mode.

    Lazily binds the fused-kernel context to the persistent block tables and
    forward-context state pointers on the first call, runs the fused kernel,
    and async-copies the per-request accepted-token counts back to the input
    batch's CPU tensor for the next iteration's preprocess.
    """
    ctx = bufs.postprocess_align
    # Caller is responsible for gating on spec decode + hybrid; this assert is
    # a tripwire if those gates ever drift apart.
    assert ctx is not None
    assert ctx.mamba_state_idx_buf is not None
    assert ctx.num_scheduled_tokens_buf is not None
    assert ctx.num_computed_tokens_buf is not None
    assert ctx.num_draft_tokens_buf is not None

    if not ctx.is_initialized:
        ctx.initialize_from_forward_context(
            kv_cache_config,
            forward_context,
            mamba_state_copy_funcs,
            [input_batch.block_table[gid].get_device_tensor(num_reqs) for gid in ctx.mamba_group_ids],
        )

    ctx.run_fused_postprocess(
        num_reqs=num_reqs,
        num_accepted_tokens_gpu=num_accepted_tokens_gpu,
        mamba_state_idx_gpu=ctx.mamba_state_idx_buf.gpu,
        num_scheduled_tokens_gpu=ctx.num_scheduled_tokens_buf.gpu,
        num_computed_tokens_gpu=ctx.num_computed_tokens_buf.gpu,
        num_draft_tokens_gpu=ctx.num_draft_tokens_buf.gpu,
    )

    # ``num_accepted_tokens_out`` is pre-initialized from
    # ``num_accepted_tokens_gpu``; the kernel only overwrites entries to 1
    # when src_block_idx == dest_block_idx (copy within the same block), so
    # the original count is preserved for everyone else.
    num_accepted_tokens_cpu_tensor[:num_reqs].copy_(ctx.num_accepted_tokens_out[:num_reqs], non_blocking=True)


def stage_postprocess_inputs_to_gpu(
    ctx: MambaSpecDecodeGPUContext,
    scheduler_output: SchedulerOutput,
    req_ids: list[str],
    num_reqs: int,
    requests: dict[str, CachedRequestState],
    mamba_state_idx: dict[str, int],
) -> None:
    """Stage all per-request inputs the fused mamba postprocess kernel reads.

    Walks ``req_ids[:num_reqs]`` once, writing each request's mamba block
    index and scheduled/computed/draft token counts into the matching pinned
    numpy views, then issues four non-blocking H→D copies. The fused kernel
    indexes the resulting GPU tensors by ``req_idx``. Buffers live on ``ctx``
    and only exist when the postprocess kernel is enabled.

    Invariant: ``preprocess_mamba`` must have run first for the same batch so
    that every ``req_ids[i]`` has an entry in ``mamba_state_idx``.
    """
    assert ctx.mamba_state_idx_buf is not None
    assert ctx.num_scheduled_tokens_buf is not None
    assert ctx.num_computed_tokens_buf is not None
    assert ctx.num_draft_tokens_buf is not None

    scheduled_spec_tokens = scheduler_output.scheduled_spec_decode_tokens
    num_scheduled = scheduler_output.num_scheduled_tokens
    state_idx_np = ctx.mamba_state_idx_buf.np
    scheduled_np = ctx.num_scheduled_tokens_buf.np
    computed_np = ctx.num_computed_tokens_buf.np
    draft_np = ctx.num_draft_tokens_buf.np

    for i in range(num_reqs):
        req_id = req_ids[i]
        state_idx = mamba_state_idx.get(req_id)
        assert state_idx is not None, (
            f"mamba_state_idx missing entry for {req_id!r}; "
            "preprocess_mamba must run before stage_postprocess_inputs_to_gpu"
        )
        state_idx_np[i] = state_idx
        scheduled_np[i] = num_scheduled[req_id]
        computed_np[i] = requests[req_id].num_computed_tokens
        draft_np[i] = len(scheduled_spec_tokens.get(req_id, []))

    ctx.mamba_state_idx_buf.copy_to_gpu(num_reqs)
    ctx.num_scheduled_tokens_buf.copy_to_gpu(num_reqs)
    ctx.num_computed_tokens_buf.copy_to_gpu(num_reqs)
    ctx.num_draft_tokens_buf.copy_to_gpu(num_reqs)
