# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Omni fork of small vLLM utilities deleted by aac1b2a4f3 (#63373, Remove
Model Runner V1) in vLLM 0.34, vendored from upstream
4ba787ec9959e566d3b587a796dff07673324da5:

* ``apply_grammar_bitmask`` — was vllm/v1/structured_output/utils.py:101-194.
  The pin moved grammar application into a manager method; Omni's MRv1 stage
  path needs the free-function form operating on the vendored V1 InputBatch.
* ``slice_query_start_locs`` / ``_make_metadata_with_slice`` /
  ``split_attn_metadata`` — were vllm/v1/worker/ubatch_utils.py.
  Preserve the V1 microbatch metadata slicing contract.
* ``cudagraph_capturing_enabled`` / ``set_cudagraph_capturing_enabled`` /
  ``validate_cudagraph_capturing_enabled`` — were vllm/compilation/monitor.py.
  The pin removed the global capture flag (its capture orchestration no longer
  consults it); Omni keeps the flag purely for its own capture-loop
  bookkeeping (talker-MTP graph capture).

Non-import deviations from the sources: none.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, TypeAlias

import numpy as np
import nvtx
import torch
from vllm.utils.import_utils import LazyLoader
from vllm.utils.nvtx_pytorch_hooks import construct_marker_dict_and_push
from vllm.utils.torch_utils import PIN_MEMORY, async_tensor_h2d
from vllm.v1.attention.backends.utils import AttentionMetadata, CommonAttentionMetadata

if TYPE_CHECKING:
    import xgrammar as xgr
    from vllm.v1.core.sched.output import GrammarOutput, SchedulerOutput
    from vllm.v1.worker.ubatch_utils import UBatchSlice

    from vllm_omni.worker.v1_compat.input_batch import InputBatch
else:
    xgr = LazyLoader("xgr", globals(), "xgrammar")

# --- type aliases from vllm/v1/worker/gpu_model_runner.py:246-248 ----------

AttnMetadataDict: TypeAlias = dict[str, AttentionMetadata]
PerLayerAttnMetadata: TypeAlias = list[AttnMetadataDict] | AttnMetadataDict


def apply_grammar_bitmask(
    scheduler_output: SchedulerOutput,
    grammar_output: GrammarOutput,
    input_batch: InputBatch,
    logits: torch.Tensor,
) -> None:
    """Apply grammar bitmask to output logits of the model with xgrammar function.

    Args:
        scheduler_output (SchedulerOutput): The result of engine scheduling.
        grammar_output (GrammarOutput): The grammar bitmask to apply.
        input_batch (InputBatch): The input of model runner.
        logits (torch.Tensor): The output logits of model forward.

    """
    # Serialization of np.ndarray is much more efficient than a tensor,
    # so we receive it in that format.
    grammar_bitmask = grammar_output.grammar_bitmask

    # We receive the structured output bitmask from the scheduler,
    # compacted to contain bitmasks only for structured output requests.
    # The order of the requests in the bitmask is not guaranteed to be the
    # same as the order of the requests in the gpu runner's batch. We need
    # to sort the bitmask to match the order of the requests used here.

    # Get the batch indices of the structured output requests.
    # Keep track of the number of speculative tokens scheduled for every
    # request in the batch, as the logit indices are offset by this amount.
    struct_out_req_batch_indices: dict[str, int] = {}
    cumulative_offset = 0
    spec_tokens = scheduler_output.scheduled_spec_decode_tokens
    struct_out_req_ids = set(grammar_output.structured_output_request_ids)
    for batch_index, req_id in enumerate(input_batch.req_ids):
        logit_index = batch_index + cumulative_offset
        cumulative_offset += len(spec_tokens.get(req_id, ()))
        if req_id in struct_out_req_ids:
            struct_out_req_batch_indices[req_id] = logit_index

    out_indices: list[int] = []

    # Reorder the bitmask to match the order of the requests in the batch.
    sorted_bitmask_tensor = torch.full(
        (logits.shape[0], grammar_bitmask.shape[1]),
        -1,
        dtype=torch.from_numpy(grammar_bitmask[:0]).dtype,
        pin_memory=PIN_MEMORY,
    )
    sorted_bitmask = sorted_bitmask_tensor.numpy()
    mask_stride = grammar_bitmask.shape[0] // len(grammar_output.structured_output_request_ids)
    cumulative_index = 0
    for req_id in grammar_output.structured_output_request_ids:
        num_rows = 1 + len(spec_tokens.get(req_id, ()))
        if (logit_idx := struct_out_req_batch_indices.get(req_id)) is not None:
            sorted_bitmask[logit_idx : logit_idx + num_rows] = grammar_bitmask[
                cumulative_index : cumulative_index + num_rows
            ]
            out_indices.extend(range(logit_idx, logit_idx + num_rows))
        cumulative_index += mask_stride

    # Copy async to device.
    grammar_bitmask = sorted_bitmask_tensor.to(logits.device, non_blocking=True)

    # If the length of out indices and the logits have the same shape
    # we don't need to pass indices to the kernel,
    # since the bitmask is already aligned with the logits.
    skip_out_indices = len(out_indices) == logits.shape[0]

    if not logits.is_cpu:
        index_tensor = None
        if not skip_out_indices:
            # xgrammar expects a python list of indices but it will actually work with
            # a tensor. If we copy the tensor ourselves here we can do it in a
            # non_blocking manner and there should be no cpu sync within xgrammar.
            index_tensor = async_tensor_h2d(out_indices, dtype=torch.int32, device=logits.device)

        xgr.apply_token_bitmask_inplace(logits, grammar_bitmask, indices=index_tensor)
        return

    # CPU case, use list for indices.
    indices = None if skip_out_indices else out_indices
    # Handle dtype conversion for CPU (older xgrammar CPU kernels require float32)
    # See: https://github.com/vllm-project/vllm/issues/31901
    if logits.dtype != torch.float32:
        # Convert to float32, apply bitmask, then convert back
        logits_fp32 = logits.to(torch.float32)
        xgr.apply_token_bitmask_inplace(logits_fp32, grammar_bitmask, indices=indices)
        # Copy the modified values back to the original tensor
        logits.copy_(logits_fp32.to(logits.dtype))
    else:
        xgr.apply_token_bitmask_inplace(logits, grammar_bitmask, indices=indices)


# --- vllm/compilation/monitor.py fork (flag + setter + validator) -----------

cudagraph_capturing_enabled: bool = True


def validate_cudagraph_capturing_enabled() -> None:
    # used to monitor whether a cudagraph capturing is legal at runtime.
    # should be called before any cudagraph capturing.
    # if an illegal cudagraph capturing happens, raise an error.
    global cudagraph_capturing_enabled
    if not cudagraph_capturing_enabled:
        raise RuntimeError(
            "CUDA graph capturing detected at an inappropriate time. This operation is currently disabled."
        )


def set_cudagraph_capturing_enabled(enabled: bool) -> None:
    global cudagraph_capturing_enabled
    cudagraph_capturing_enabled = enabled


class PytHooks:
    """Vendored verbatim from baseline 4ba787ec99
    vllm/utils/nvtx_pytorch_hooks.py (aac1b2a4f3 deleted the class; the
    module-level helpers it uses are still live upstream). Used by the V1
    GPUModelRunner surface's ``_register_layerwise_nvtx_hooks``.
    """

    def __init__(self):
        """Initialize module variables."""
        super().__init__()
        self.module_to_name_map = {}

    def module_fwd_hook(self, module_obj, in_tensor, out_tensor):
        """Callback function that ends the NVTX marker.
        Records the module name and tensor information.
        Called after the module executes the forward method.
        """
        nvtx.range_pop()
        module_name = self.module_to_name_map.get(module_obj, "unknown")
        construct_marker_dict_and_push(module_name, module_obj, in_tensor=None, kwargs=None, out_tensor=out_tensor)
        nvtx.range_pop()
        return

    def module_fwd_pre_hook(self, module_obj, in_tensor, kwargs):
        """Creates an NVTX marker with the module name in it.
        This function is called before the module executes.
        """
        module_name = self.module_to_name_map.get(module_obj, "unknown")
        construct_marker_dict_and_push(module_name, module_obj, in_tensor=in_tensor, kwargs=kwargs, out_tensor=None)
        return

    def register_hooks(self, network_model, module_prefix="top"):
        """User level function that activates all the hooks.
        The user needs to call this method from the network source code.
        The code descends all the modules in the network and registers their
        respective hooks.
        """
        # Module types to skip (simple operations that don't need detailed profiling)
        skip_types = (
            torch.nn.Identity,
            torch.nn.Dropout,
            torch.nn.Dropout1d,
            torch.nn.Dropout2d,
            torch.nn.Dropout3d,
        )

        for name, module in network_model.named_modules(prefix=module_prefix):
            # Skip certain module types to reduce profiling overhead
            if isinstance(module, skip_types):
                continue

            module.register_forward_pre_hook(self.module_fwd_pre_hook, with_kwargs=True)
            module.register_forward_hook(self.module_fwd_hook)
            if module not in self.module_to_name_map:
                self.module_to_name_map[module] = name
            else:
                raise ValueError(f"Module instance {module} is not unique ")
        return


# Vendored verbatim from baseline 4ba787ec99
# vllm/v1/attention/backends/utils.py (aac1b2a4f3 deleted it upstream);
# operates on the vendored V1 InputBatch, used by _may_reorder_batch.


def reorder_batch_to_split_decodes_and_prefills(
    input_batch: InputBatch,
    scheduler_output: SchedulerOutput,
    decode_threshold: int = 1,
) -> bool:
    """Reorders the batch to split into prefill and decode requests; places all
    requests with <= decode_threshold tokens at the front of the batch.

    The batch is reordered into 4 regions:
        decode:        (num_scheduled <= threshold AND is not prefilling)
        short_extend:  (num_scheduled <= threshold AND is chunked prefilling)
        long_extend:   (num_scheduled > threshold AND is chunked prefilling)
        prefill:       (num_computed == 0)   # First chunks

    Returns:
        True if the batch was modified, False otherwise.

    """
    num_reqs = len(input_batch.req_ids)
    num_scheduled_tokens = [scheduler_output.num_scheduled_tokens[id] for id in input_batch.req_ids]
    num_scheduled_tokens_np = np.array(num_scheduled_tokens)
    num_computed_tokens_np = input_batch.num_computed_tokens_cpu[:num_reqs]
    num_prompt_tokens_np = input_batch.num_prompt_tokens[:num_reqs]

    has_context = num_computed_tokens_np > 0
    is_below_threshold = num_scheduled_tokens_np <= decode_threshold
    done_prefilling = num_computed_tokens_np >= num_prompt_tokens_np

    # Mutually exclusive categories (exactly one True per request):
    # 1. No context yet -> prefill
    # 2. Has context, above threshold -> long_extend
    # 3. Has context, below threshold, still prefilling -> short_extend
    # 4. Has context, below threshold, done prefilling -> decode
    is_pure_prefill = ~has_context
    is_long_extend = has_context & ~is_below_threshold
    is_short_extend = has_context & is_below_threshold & ~done_prefilling
    is_decode = has_context & is_below_threshold & done_prefilling

    # Desired order: decode → short_extend → long_extend → prefill
    req_regions = np.zeros(num_reqs, dtype=np.int32)  # 0 = decode by default
    req_regions[is_short_extend] = 1
    req_regions[is_long_extend] = 2
    req_regions[is_pure_prefill] = 3

    num_decodes = int(is_decode.sum())
    num_short_extends = int(is_short_extend.sum())
    num_long_extends = int(is_long_extend.sum())
    num_prefills = int(is_pure_prefill.sum())

    target_regions = np.repeat(
        [0, 1, 2, 3],
        [num_decodes, num_short_extends, num_long_extends, num_prefills],
    ).astype(np.int32)

    needs_swap = req_regions != target_regions

    if not needs_swap.any():
        return False

    # Extract indices that need swapping and sort by target region
    orig_indices = np.where(needs_swap)[0]
    sorted_order = np.argsort(req_regions[needs_swap], kind="stable")
    src_indices = orig_indices[sorted_order]

    src_dest_map = {int(src): int(dst) for src, dst in zip(src_indices, orig_indices)}

    for src in src_dest_map:
        dst = src_dest_map[src]
        while src != dst:
            input_batch.swap_states(src, dst)
            # Mark dst as done by updating its destination to itself
            next_dst = src_dest_map.get(dst, dst)
            src_dest_map[dst] = dst
            dst = next_dst

    return True


# Vendored verbatim from baseline 4ba787ec99 (aac1b2a4f3 deleted them):
# get_dcp_dummy_context_len / prepare_dcp_dummy_context_metadata were
# vllm/v1/worker/cp_utils.py; split_attn_metadata was
# vllm/v1/worker/ubatch_utils.py. Used by the V1 GPUModelRunner surface
# (_dummy_run DCP path and the ubatch attention-metadata split loop).


def get_dcp_dummy_context_len(
    dcp_world_size: int,
    cp_kv_cache_interleave_size: int,
    has_kv_cache_config: bool,
    create_mixed_batch: bool,
    is_graph_capturing: bool,
    uniform_decode: bool,
) -> int:
    if (
        dcp_world_size <= 1
        or not has_kv_cache_config
        or not (create_mixed_batch or (is_graph_capturing and uniform_decode))
    ):
        return 0
    return dcp_world_size * cp_kv_cache_interleave_size


def prepare_dcp_dummy_context_metadata(
    *,
    input_batch: Any,
    kv_cache_config: Any,
    query_pos: Any,
    positions: torch.Tensor,
    query_start_loc: Any,
    num_reqs: int,
    num_tokens_unpadded: int,
    dcp_dummy_context_len: int,
) -> None:
    """Populate valid fake KV metadata for DCP CUDA graph warmup/capture."""
    if dcp_dummy_context_len == 0:
        return

    # DCP graph warmup may exercise context attention, so block-table entries
    # must point at allocated KV blocks.
    assert kv_cache_config is not None
    max_valid_block_id = kv_cache_config.num_blocks - 1
    assert max_valid_block_id > 0
    for blk_table in input_batch.block_table.block_tables:
        block_ids = [(block_idx % max_valid_block_id) + 1 for block_idx in range(blk_table.max_num_blocks_per_req)]
        for req_idx in range(num_reqs):
            blk_table.add_row(block_ids, req_idx)
        blk_table.commit_block_table(num_reqs)

    query_pos.copy_to_gpu(num_tokens_unpadded)
    positions[:num_tokens_unpadded] = query_pos.gpu[:num_tokens_unpadded] + dcp_dummy_context_len
    input_batch.block_table.compute_slot_mapping(
        num_reqs,
        query_start_loc.gpu[: num_reqs + 1],
        positions[:num_tokens_unpadded],
    )


def slice_query_start_locs(
    query_start_loc: torch.Tensor,
    request_slice: slice,
) -> torch.Tensor:
    """Creates a new query_start_loc that corresponds to the requests in
    request_slice.

    Note: This function creates a new tensor to hold the new query_start_locs.
    This will break cudagraph compatibility.
    """
    return query_start_loc[request_slice.start : request_slice.stop + 1] - query_start_loc[request_slice.start]


def _make_metadata_with_slice(
    ubatch_slice: UBatchSlice, attn_metadata: CommonAttentionMetadata
) -> CommonAttentionMetadata:
    """This function creates a new CommonAttentionMetadata that corresponds to
    the requests included in ubatch_slice
    """
    assert not ubatch_slice.is_empty(), f"Ubatch slice {ubatch_slice} is empty"

    request_slice = ubatch_slice.request_slice
    token_slice = ubatch_slice.token_slice

    start_locs = attn_metadata.query_start_loc_cpu
    first_req = request_slice.start
    first_tok = token_slice.start
    last_req = request_slice.stop - 1
    last_tok = token_slice.stop - 1

    assert start_locs[first_req] <= first_tok < start_locs[first_req + 1], "Token slice start outside of first request"
    # NOTE: last token can be outside of the last request if we have CG padding.

    # If the request is split across ubatches, we have to adjust the metadata.
    # splits_first_request: The first request in this slice is the continuation of
    #                       a request that started in a previous slice.
    # splits_last_request:  The last request in this slice continues into the
    #                       next slice.
    splits_first_request = first_tok > start_locs[first_req]
    splits_last_request = last_tok < start_locs[last_req + 1] - 1

    query_start_loc_cpu = slice_query_start_locs(start_locs, request_slice)
    query_start_loc = slice_query_start_locs(attn_metadata.query_start_loc, request_slice)

    assert len(query_start_loc) >= 2, f"query_start_loc must have at least 2 elements, got {len(query_start_loc)}"

    if splits_first_request:
        tokens_skipped = first_tok - start_locs[first_req]
        query_start_loc[1:] -= tokens_skipped
        query_start_loc_cpu[1:] -= tokens_skipped
    seq_lens = attn_metadata.seq_lens[request_slice]
    seq_lens_cpu_upper_bound = (
        attn_metadata.seq_lens_cpu_upper_bound[request_slice]
        if attn_metadata.seq_lens_cpu_upper_bound is not None
        else None
    )
    if splits_last_request:
        # NOTE: We use start_locs (the original query_start_loc_cpu) to calculate
        # the tokens skipped because query_start_loc_cpu might have been modified
        # if splits_first_request is True.
        tokens_skipped = start_locs[last_req + 1] - token_slice.stop
        query_start_loc[-1] -= tokens_skipped
        query_start_loc_cpu[-1] -= tokens_skipped

        # Make sure we don't modify the seq_lens tensors
        #  (not cudagraph compatible)
        seq_lens = seq_lens.clone()
        seq_lens[-1] -= tokens_skipped
        if seq_lens_cpu_upper_bound is not None:
            seq_lens_cpu_upper_bound = seq_lens_cpu_upper_bound.clone()
            seq_lens_cpu_upper_bound[-1] -= tokens_skipped

    assert seq_lens_cpu_upper_bound is not None
    # Preserve the max_seq_len override set during CUDA-graph capture so
    # the attention backend selects the correct kernel for SWA layers.
    max_seq_len = max(int(seq_lens_cpu_upper_bound.max()), attn_metadata.max_seq_len)

    num_requests = request_slice.stop - request_slice.start
    num_actual_tokens = token_slice.stop - token_slice.start
    max_query_len = int(torch.max(torch.abs(query_start_loc_cpu[1:] - query_start_loc_cpu[:-1])).item())

    # This is to account for the case where we are in a dummy
    # run and query_start_loc_cpu is full of 0s
    if max_query_len == 0:
        max_query_len = attn_metadata.max_query_len

    block_table_tensor = attn_metadata.block_table_tensor[request_slice]
    slot_mapping = attn_metadata.slot_mapping[token_slice]

    positions = attn_metadata.positions
    if positions is not None:
        positions = positions[token_slice]

    return CommonAttentionMetadata(
        query_start_loc=query_start_loc,
        query_start_loc_cpu=query_start_loc_cpu,
        seq_lens=seq_lens,
        num_reqs=num_requests,
        num_actual_tokens=num_actual_tokens,
        max_query_len=max_query_len,
        max_seq_len=max_seq_len,
        block_table_tensor=block_table_tensor,
        slot_mapping=slot_mapping,
        positions=positions,
        seq_lens_cpu_upper_bound=seq_lens_cpu_upper_bound,
    )


def split_attn_metadata(
    ubatch_slices: list[UBatchSlice],
    common_attn_metadata: CommonAttentionMetadata,
) -> list[CommonAttentionMetadata]:
    """Creates a new CommonAttentionMetadata instance that corresponds to the
    requests for each UBatchSlice in ubatch_slices.

    Note: This function does not modify common_attn_metadata
    """
    results = []
    for ubatch_slice in ubatch_slices:
        results.append(_make_metadata_with_slice(ubatch_slice, common_attn_metadata))

    return results
