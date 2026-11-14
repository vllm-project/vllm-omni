# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""CPU contracts for the retained V1 aligned Mamba cache lifecycle."""

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
from vllm.v1.attention.backends.registry import MambaAttentionBackendEnum
from vllm.v1.core.sched.output import NewRequestData, SchedulerOutput
from vllm.v1.kv_cache_interface import KVCacheConfig, KVCacheGroupSpec, MambaSpec, UniformTypeKVCacheSpecs
from vllm.v1.utils import CpuGpuBuffer

from vllm_omni.worker.mamba_prefix_cache import prepare_mamba_prefix_cache
from vllm_omni.worker.v1_compat import gpu_model_runner as v1_runner
from vllm_omni.worker.v1_compat.input_batch import CachedRequestState, InputBatch

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]
mamba_utils = v1_runner.mamba_utils


@pytest.fixture
def mamba_case(monkeypatch):
    # Keep real CpuGpuBuffer transfers on CPU, with no pinned-host registration.
    monkeypatch.setattr("vllm.v1.utils.PIN_MEMORY", False)
    monkeypatch.setenv("VLLM_SSM_CONV_STATE_LAYOUT", "SD")
    get_conv_state_layout.cache_clear()
    gdn = MambaSpec(
        block_size=4,
        shapes=((3, 2), (2, 2)),
        dtypes=(torch.float32, torch.float32),
        mamba_type=MambaAttentionBackendEnum.GDN_ATTN,
        mamba_cache_mode="align",
        num_speculative_blocks=1,
    )
    short = MambaSpec(
        block_size=4,
        shapes=((3, 2),),
        dtypes=(torch.float32,),
        mamba_type=MambaAttentionBackendEnum.SHORT_CONV,
        mamba_cache_mode="align",
        num_speculative_blocks=1,
    )
    config = KVCacheConfig(
        num_blocks=6,
        kv_cache_tensors=[],
        kv_cache_groups=[
            KVCacheGroupSpec(layer_names=["gdn0", "gdn1"], kv_cache_spec=gdn),
            KVCacheGroupSpec(layer_names=["short"], kv_cache_spec=short),
        ],
    )
    funcs = {
        MambaAttentionBackendEnum.GDN_ATTN: (get_conv_copy_spec, get_temporal_copy_spec),
        MambaAttentionBackendEnum.SHORT_CONV: (get_conv_copy_spec,),
    }
    runner = object.__new__(v1_runner.V1GPUModelRunnerMixin)
    runner._mamba_bufs = None
    runner._mamba_state_copy_funcs = funcs
    runner.cache_config = CacheConfig(enable_prefix_caching=True, mamba_cache_mode="align")
    runner.kv_cache_config = config
    runner.device = torch.device("cpu")
    runner.max_num_reqs = 4
    runner.speculative_config = object()
    runner.model_config = SimpleNamespace(is_hybrid=True)
    runner._make_buffer = partial(CpuGpuBuffer, device=runner.device, pin_memory=False)
    states = {
        "gdn0": [torch.zeros(6, 3, 2), torch.empty(6, 8).as_strided((6, 2, 2), (8, 2, 1))],
        "gdn1": [torch.zeros(6, 3, 2), torch.zeros(6, 2, 2)],
        "short": [torch.zeros(6, 3, 2)],
    }
    forward = {name: SimpleNamespace(kv_cache=value) for name, value in states.items()}
    yield runner, states, forward, (gdn, short)
    get_conv_state_layout.cache_clear()


def _request(req_id, computed=0):
    return CachedRequestState(
        req_id=req_id,
        prompt_token_ids=[1],
        mm_features=[],
        sampling_params=None,
        generator=None,
        block_ids=([1, 2, 3], [2, 3, 4]),
        num_computed_tokens=computed,
        output_token_ids=[],
    )


@pytest.mark.parametrize("with_spec", [False, True])
def test_v1_mamba_runner_allocates_real_legacy_buffers(mamba_case, with_spec):
    runner, _, _, _ = mamba_case
    if not with_spec:
        runner.speculative_config = None
    buffers = runner._get_mamba_bufs()
    assert runner._get_mamba_bufs() is buffers
    assert buffers.preprocess.src_ptrs.cpu.shape == (4 * 5,)
    assert buffers.preprocess.mamba_group_ids == [0, 1]
    if with_spec:
        context = buffers.postprocess_align
        assert context is not None
        assert context.num_states == 5
        assert context.num_groups == 2
        assert context.mamba_state_idx_buf.cpu.shape == (4,)
    else:
        assert buffers.postprocess_align is None


@pytest.mark.parametrize("layout", ["SD", "DS"])
def test_v1_mamba_metadata_uses_real_state_addresses_and_strides(mamba_case, monkeypatch, layout):
    runner, states, forward, _ = mamba_case
    monkeypatch.setenv("VLLM_SSM_CONV_STATE_LAYOUT", layout)
    get_conv_state_layout.cache_clear()
    if layout == "DS":
        for tensors in states.values():
            tensors[0] = tensors[0].transpose(1, 2)
    context = runner._get_mamba_bufs().postprocess_align
    tables = [torch.tensor([[1, 2, 3], [2, 3, 4]], dtype=torch.int32) for _ in range(2)]
    context.initialize_from_forward_context(runner.kv_cache_config, forward, runner._mamba_state_copy_funcs, tables)
    flattened = [state for tensors in states.values() for state in tensors]
    assert context.state_base_addrs.tolist() == [state.data_ptr() for state in flattened]
    assert context.state_block_strides.tolist() == [state.stride(0) * state.element_size() for state in flattened]
    assert context.state_group_indices.tolist() == [0, 0, 0, 0, 1]
    assert context.state_conv_widths.tolist() == [3, 0, 3, 0, 3]
    assert context.state_inner_sizes.tolist() == ([2, 4, 2, 4, 2] if layout == "SD" else [1, 4, 1, 4, 1])
    assert context.state_dim_row_count.tolist() == ([0] * 5 if layout == "SD" else [2, 0, 2, 0, 2])
    assert context.block_table_ptrs.tolist() == [table.data_ptr() for table in tables]
    assert context.block_table_stride_req == 3
    assert context.is_initialized


def test_v1_mamba_mixed_uniform_group_is_not_duplicated(mamba_case):
    runner, _, _, (gdn, short) = mamba_case
    runner.kv_cache_config = KVCacheConfig(
        num_blocks=6,
        kv_cache_tensors=[],
        kv_cache_groups=[
            KVCacheGroupSpec(
                layer_names=["gdn0", "short"],
                kv_cache_spec=UniformTypeKVCacheSpecs(
                    block_size=4,
                    kv_cache_specs={"gdn0": gdn, "short": short},
                ),
            ),
        ],
    )
    buffers = runner._get_mamba_bufs()
    assert buffers.preprocess.mamba_group_ids == [0]
    assert buffers.preprocess.src_ptrs.cpu.numel() == 4 * 3
    assert buffers.postprocess_align.num_states == 3


@pytest.mark.parametrize("computed", [0, 4])
@pytest.mark.parametrize("has_output", [False, True])
def test_v1_mamba_full_resend_clears_stale_state_before_preprocess(mamba_case, monkeypatch, computed, has_output):
    runner, _, _, _ = mamba_case
    runner.speculative_config = None
    buffers = runner._get_mamba_bufs()
    scheduled = SchedulerOutput.make_empty()
    block_ids = ([2, 4, 5], [1, 3, 5])
    request = _request("resume", computed)
    request.prompt_token_ids = [1, 2, 3, 4, 5, 6]
    request.output_token_ids = [7] if has_output else []
    request.block_ids = block_ids
    scheduled.scheduled_new_reqs = [
        NewRequestData(
            req_id="resume",
            prompt_token_ids=request.prompt_token_ids,
            prefill_token_ids=request.prompt_token_ids + request.output_token_ids,
            mm_features=[],
            sampling_params=None,
            pooling_params=None,
            block_ids=block_ids,
            num_computed_tokens=computed,
            lora_request=None,
        )
    ]
    scheduled.num_scheduled_tokens = {"resume": 1}
    assert not hasattr(scheduled.scheduled_cached_reqs, "resumed_req_ids")
    batch = object.__new__(InputBatch)
    batch._req_ids = ["resume"]
    batch.num_accepted_tokens_cpu = np.array([1], dtype=np.int32)
    state_indices = {"resume": 99, "unrelated": 7, "finished": 8, "preempted": 9}
    scheduled.finished_req_ids = {"finished"}
    scheduled.preempted_req_ids = {"preempted"}
    states = {
        "gdn0": [torch.zeros(6, 3, 2), torch.zeros(6, 2, 2)],
        "gdn1": [torch.zeros(6, 3, 2), torch.zeros(6, 2, 2)],
        "short": [torch.zeros(6, 3, 2)],
    }
    forward = {name: SimpleNamespace(kv_cache=value) for name, value in states.items()}
    launches = []
    # Intercept only final device dispatch, retaining real metadata collection.
    monkeypatch.setattr(mamba_utils, "batch_memcpy", lambda *args: launches.append(args))
    mamba_utils.preprocess_mamba(
        scheduled,
        runner.kv_cache_config,
        runner.cache_config,
        state_indices,
        batch,
        {"resume": request},
        forward,
        runner._mamba_state_copy_funcs,
        buffers.preprocess,
    )
    assert state_indices == {"resume": computed // 4, "unrelated": 7}
    assert batch.num_accepted_tokens_cpu.tolist() == [1]
    if computed == 0:
        assert buffers.preprocess.offset == 0
        assert launches == []
    else:
        assert buffers.preprocess.offset == 5
        assert len(launches) == 1
        sources, destinations, sizes = launches[0]
        flattened = [state for values in states.values() for state in values]
        source_blocks = [2, 2, 2, 2, 1]
        destination_blocks = [4, 4, 4, 4, 3]
        assert sources.tolist() == [state[block].data_ptr() for state, block in zip(flattened, source_blocks)]
        assert destinations.tolist() == [state[block].data_ptr() for state, block in zip(flattened, destination_blocks)]
        assert sizes.tolist() == [24, 16, 24, 16, 24]


def test_v1_mamba_ar_delegate_consumes_deferred_correction_before_real_preprocess(mamba_case):
    runner, _, _, _ = mamba_case
    runner.speculative_config = None
    runner._omni_cache_policy = SimpleNamespace(native_kv_cache_only=True)
    runner.mamba_state_idx = {}
    runner.requests = {"fresh": _request("fresh")}
    runner.input_batch = object.__new__(InputBatch)
    runner.input_batch._req_ids = ["fresh"]
    runner.input_batch.num_accepted_tokens_cpu = np.array([1], dtype=np.int32)
    runner.num_accepted_tokens = runner._make_buffer(2, dtype=torch.int32)
    runner.compilation_config = SimpleNamespace(static_forward_context={})
    scheduled = SchedulerOutput.make_empty()
    scheduled.num_scheduled_tokens = {"fresh": 1}
    calls = []

    def correct_state():
        calls.append("deferred")
        runner.input_batch.num_accepted_tokens_cpu[0] = 2

    assert prepare_mamba_prefix_cache(runner, scheduled, 1, correct_state) is None
    assert calls == ["deferred"]
    assert runner.mamba_state_idx == {"fresh": 0}
    assert runner.num_accepted_tokens.gpu.tolist() == [2, 0]


def test_v1_mamba_copy_metadata_preserves_conv_bias_and_temporal_content_size(mamba_case):
    runner, states, forward, _ = mamba_case
    buffers = runner._get_mamba_bufs()
    request = _request("active")
    mamba_utils.collect_mamba_copy_meta(
        buffers.preprocess,
        runner.kv_cache_config,
        runner._mamba_state_copy_funcs,
        [0],
        src_block_idx=0,
        dest_block_idx=2,
        accept_token_bias=1,
        req_state=request,
        forward_context=forward,
    )
    expected_sources = [states["gdn0"][0][1, 1:], states["gdn0"][1][2], states["gdn1"][0][1, 1:], states["gdn1"][1][2]]
    expected_destinations = [state[3] for name in ("gdn0", "gdn1") for state in states[name]]
    assert buffers.preprocess.offset == 4
    assert buffers.preprocess.src_ptrs.np[:4].tolist() == [state.data_ptr() for state in expected_sources]
    assert buffers.preprocess.dst_ptrs.np[:4].tolist() == [state.data_ptr() for state in expected_destinations]
    assert buffers.preprocess.sizes.np[:4].tolist() == [
        state.numel() * state.element_size() for state in expected_sources
    ]
    # The padded temporal page has 32B stride but only 16B of actual state.
    assert buffers.preprocess.sizes.np[1] == 16


def test_v1_mamba_staging_keeps_request_order_and_unused_tail(mamba_case):
    runner, _, _, _ = mamba_case
    context = runner._get_mamba_bufs().postprocess_align
    scheduled = SchedulerOutput.make_empty()
    scheduled.num_scheduled_tokens = {"a": 5, "b": 2, "c": 1}
    scheduled.scheduled_spec_decode_tokens = {"b": [7, 8], "a": [6]}
    requests = {name: _request(name, index * 4) for index, name in enumerate(("a", "b", "c"))}
    indices = {"a": 2, "b": 3, "c": 4}
    for buffer in (
        context.mamba_state_idx_buf,
        context.num_scheduled_tokens_buf,
        context.num_computed_tokens_buf,
        context.num_draft_tokens_buf,
    ):
        buffer.cpu.fill_(-99)
        buffer.gpu.fill_(-99)
    mamba_utils.stage_postprocess_inputs_to_gpu(context, scheduled, ["b", "a", "c"], 3, requests, indices)
    for buffer, expected in (
        (context.mamba_state_idx_buf, [3, 2, 4, -99]),
        (context.num_scheduled_tokens_buf, [2, 5, 1, -99]),
        (context.num_computed_tokens_buf, [4, 0, 8, -99]),
        (context.num_draft_tokens_buf, [2, 1, 0, -99]),
    ):
        assert buffer.cpu.tolist() == expected
        assert buffer.gpu.tolist() == expected
    with pytest.raises(AssertionError, match="missing entry"):
        mamba_utils.stage_postprocess_inputs_to_gpu(context, scheduled, ["b"], 1, requests, {})
