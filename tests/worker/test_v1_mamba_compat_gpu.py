# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Real-kernel goldens for V1 Mamba state migration, without loading a model."""

from functools import partial
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from vllm.config import CacheConfig
from vllm.model_executor.layers.mamba.mamba_utils import (
    get_conv_copy_spec,
    get_conv_state_layout,
    get_temporal_copy_spec,
)
from vllm.platforms import current_platform
from vllm.v1.attention.backends.registry import MambaAttentionBackendEnum
from vllm.v1.core.sched.output import NewRequestData, SchedulerOutput
from vllm.v1.kv_cache_interface import KVCacheConfig, KVCacheGroupSpec, MambaSpec
from vllm.v1.utils import CpuGpuBuffer

from tests.helpers.mark import hardware_marks
from vllm_omni.worker.v1_compat import mamba_utils
from vllm_omni.worker.v1_compat.block_table import BlockTable, SlotMappingMode
from vllm_omni.worker.v1_compat.input_batch import CachedRequestState, InputBatch

pytestmark = [
    pytest.mark.core_model,
    *hardware_marks(res={"cuda": "L4"}, num_cards=1),
    pytest.mark.skipif(not current_platform.is_cuda(), reason="Mamba copy kernels require CUDA/Triton"),
]


def _reference_copy(states, block_ids, src_col, dst_col, bias, layout):
    conv, temporal = states
    source = conv[block_ids[src_col]]
    destination = conv[block_ids[dst_col]]
    if layout == "SD":
        destination[: source.shape[0] - bias].copy_(source[bias:].clone())
    else:
        destination[:, : source.shape[1] - bias].copy_(source[:, bias:].clone())
    temporal[block_ids[dst_col]].copy_(temporal[block_ids[src_col + bias]].clone())


@pytest.mark.parametrize("layout", ["SD", "DS"])
@pytest.mark.parametrize("with_spec", [False, True], ids=["scalar", "fused-spec"])
def test_v1_mamba_align_lifecycle_matches_torch_reference(monkeypatch, layout, with_spec):
    monkeypatch.setenv("VLLM_SSM_CONV_STATE_LAYOUT", layout)
    get_conv_state_layout.cache_clear()
    try:
        _check_lifecycle(layout, with_spec)
    finally:
        get_conv_state_layout.cache_clear()


def _check_lifecycle(layout, with_spec):
    device = torch.device("cuda")
    request_ids = ["cross", "unchanged", "same-block", "fresh"]
    computed = [7 if with_spec else 8, 4, 4, 0]
    scheduled_counts = [3 if with_spec else 2, 1, 4 if with_spec else 3, 1]
    draft_counts = [2, 0, 1, 0] if with_spec else [0] * 4
    accepted = [2 if with_spec else 1, 1, 1, 1]
    block_ids = [([8 * i + j for j in range(6)], [8 * i + j for j in (5, 3, 1, 4, 0, 2)]) for i in range(4)]
    backends = (MambaAttentionBackendEnum.GDN_ATTN, MambaAttentionBackendEnum.MAMBA2)
    conv_shape = (6, 4) if layout == "SD" else (4, 6)
    config = KVCacheConfig(
        num_blocks=32,
        kv_cache_tensors=[],
        kv_cache_groups=[
            KVCacheGroupSpec(
                layer_names=[f"layer{group}"],
                kv_cache_spec=MambaSpec(
                    block_size=4,
                    shapes=(conv_shape, (2, 2)),
                    dtypes=(torch.float32, torch.float32),
                    mamba_type=backend,
                    mamba_cache_mode="align",
                    num_speculative_blocks=2,
                ),
            )
            for group, backend in enumerate(backends)
        ],
    )
    funcs = {backend: (get_conv_copy_spec, get_temporal_copy_spec) for backend in backends}
    buffers = mamba_utils.MambaBuffers.create(
        4,
        config,
        funcs,
        partial(CpuGpuBuffer, device=device, pin_memory=False),
        device,
        with_spec,
    )
    states = [
        [
            (torch.arange(32 * 24, dtype=torch.float32, device=device) + group * 10000).reshape(32, *conv_shape),
            (torch.arange(32 * 4, dtype=torch.float32, device=device) + group * 1000).reshape(32, 2, 2),
        ]
        for group in range(2)
    ]
    reference = [[state.clone() for state in group] for group in states]
    forward = {f"layer{group}": SimpleNamespace(kv_cache=values) for group, values in enumerate(states)}
    batch = object.__new__(InputBatch)
    batch._req_ids = request_ids
    batch.req_id_to_index = {req_id: index for index, req_id in enumerate(request_ids)}
    batch.num_accepted_tokens_cpu = np.array(accepted, dtype=np.int32)
    batch.block_table = []
    for group in range(2):
        table = BlockTable(4, 4, 6, 16, False, device, 1, slot_mapping_mode=SlotMappingMode.NONE)
        table.block_table.cpu.copy_(torch.tensor([ids[group] for ids in block_ids], dtype=torch.int32))
        table.block_table.copy_to_gpu()
        batch.block_table.append(table)
    requests = {
        req_id: CachedRequestState(
            req_id=req_id,
            prompt_token_ids=[1],
            mm_features=[],
            sampling_params=None,
            generator=None,
            block_ids=block_ids[index],
            num_computed_tokens=computed[index],
            output_token_ids=[],
        )
        for index, req_id in enumerate(request_ids)
    }
    output = SchedulerOutput.make_empty()
    output.num_scheduled_tokens = dict(zip(request_ids, scheduled_counts))
    output.scheduled_spec_decode_tokens = {req_id: [0] * draft_counts[i] for i, req_id in enumerate(request_ids)}
    output.scheduled_new_reqs = [
        NewRequestData(
            req_id="fresh",
            prompt_token_ids=[1],
            prefill_token_ids=[1],
            mm_features=[],
            sampling_params=None,
            pooling_params=None,
            block_ids=block_ids[3],
            num_computed_tokens=0,
            lora_request=None,
        )
    ]
    indices = {"cross": 0, "unchanged": 1, "same-block": 1, "fresh": 99}
    expected_indices = [2, 1, 1, 0]
    for group in range(2):
        _reference_copy(reference[group], block_ids[0][group], 0, 2, accepted[0] - 1, layout)

    mamba_utils.preprocess_mamba(
        output,
        config,
        CacheConfig(enable_prefix_caching=True, mamba_cache_mode="align"),
        indices,
        batch,
        requests,
        forward,
        funcs,
        buffers.preprocess,
        align_ctx=buffers.postprocess_align,
    )

    assert [indices[req_id] for req_id in request_ids] == expected_indices
    assert batch.num_accepted_tokens_cpu.tolist() == [1, 1, 1, 1]
    for actual_group, expected_group in zip(states, reference):
        for actual, expected in zip(actual_group, expected_group):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    if not with_spec:
        return
    context = buffers.postprocess_align
    mamba_utils.stage_postprocess_inputs_to_gpu(context, output, request_ids, 4, requests, indices)
    accepted_after = [3, 1, 2, 1]
    expected_accepted = accepted_after.copy()
    for index in range(4):
        running = computed[index] + scheduled_counts[index] - draft_counts[index]
        aligned = (running + accepted_after[index] - 1) // 4 * 4
        if aligned < running:
            continue
        destination = aligned // 4 - 1
        bias = aligned - running
        source = expected_indices[index]
        if source == destination:
            expected_accepted[index] = 1
        if source != destination or bias:
            for group in range(2):
                _reference_copy(reference[group], block_ids[index][group], source, destination, bias, layout)
    accepted_gpu = torch.tensor(accepted_after, dtype=torch.int32, device=device)
    accepted_cpu = torch.empty(4, dtype=torch.int32)
    mamba_utils.postprocess_mamba_align_gpu(
        bufs=buffers,
        num_reqs=4,
        num_accepted_tokens_gpu=accepted_gpu,
        num_accepted_tokens_cpu_tensor=accepted_cpu,
        input_batch=batch,
        kv_cache_config=config,
        forward_context=forward,
        mamba_state_copy_funcs=funcs,
    )
    torch.accelerator.synchronize()
    assert accepted_cpu.tolist() == expected_accepted
    for actual_group, expected_group in zip(states, reference):
        for actual, expected in zip(actual_group, expected_group):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_v1_mamba_batch_memcpy_left_overlap_has_memmove_semantics():
    """Keep the 4ba native golden for the restored scalar kernel's barrier."""
    device = torch.device("cuda")
    batch = 128
    row_bytes = 32 * 1024
    shift = 16
    copy_size = row_bytes - shift
    pattern = (torch.arange(row_bytes, dtype=torch.int32, device=device) % 251).to(torch.uint8)
    state = pattern.expand(batch, -1).clone()
    snapshot = state.clone()
    row_stride_bytes = state.stride(0) * state.element_size()
    row_offsets = torch.arange(batch, dtype=torch.int64, device=device) * row_stride_bytes
    dst_ptrs = (row_offsets + state.data_ptr()).to(torch.uint64)
    src_ptrs = (row_offsets + state.data_ptr() + shift).to(torch.uint64)
    sizes = torch.full((batch,), copy_size, dtype=torch.int32, device=device)
    expected = snapshot.clone()
    expected[:, :copy_size].copy_(snapshot[:, shift:])
    for _ in range(10):
        state.copy_(snapshot)
        mamba_utils.batch_memcpy(src_ptrs, dst_ptrs, sizes)
        torch.accelerator.synchronize()
        torch.testing.assert_close(state, expected, rtol=0, atol=0)
