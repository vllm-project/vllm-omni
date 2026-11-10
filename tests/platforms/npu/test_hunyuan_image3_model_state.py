# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""HunyuanImage3 MRv2 must use Ascend attention metadata on NPU."""

import numpy as np
import pytest
import torch

pytest.importorskip("vllm_ascend")
pytest.importorskip("torch_npu")

import torch_npu
from vllm.config import CUDAGraphMode
from vllm_ascend.worker.v2.input_batch import AscendInputBatch
from vllm_ascend.worker.v2.model_states.default import AscendModelState

from vllm_omni.model_executor.models.hunyuan_image3.hunyuan_image3 import HunyuanImage3ForConditionalGeneration
from vllm_omni.platforms import current_omni_platform
from vllm_omni.platforms.npu.attention.static_shape_backend import OmniStaticShapeAttentionBackendImpl
from vllm_omni.platforms.npu.models import hunyuan_image3_model_state
from vllm_omni.platforms.npu.models.hunyuan_image3_model_state import (
    NPUHunyuanImage3AttentionImpl,
    NPUHunyuanImage3ModelState,
    _prefix_queries,
)
from vllm_omni.worker_v2.model_states import init_omni_model_state
from vllm_omni.worker_v2.model_states.omni_model_state import OmniModelState

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


@pytest.mark.parametrize("npu", [False, True])
def test_factory_selects_platform_state(monkeypatch, mocker, npu):
    monkeypatch.setattr(current_omni_platform, "is_npu", lambda: npu)
    init = mocker.patch.object(OmniModelState, "__init__", return_value=None)
    model = HunyuanImage3ForConditionalGeneration.__new__(HunyuanImage3ForConditionalGeneration)
    config, cache = object(), object()
    device = torch.device("cpu")
    state = init_omni_model_state(config, model, cache, device)
    assert type(state) is (NPUHunyuanImage3ModelState if npu else OmniModelState)
    init.assert_called_once_with(config, model, cache, device)


def test_ascend_attention_preserves_omni_inputs_and_sampler(mocker):
    state = NPUHunyuanImage3ModelState.__new__(NPUHunyuanImage3ModelState)
    prepare = mocker.patch.object(AscendModelState, "prepare_attn", return_value={})
    batch = AscendInputBatch.__new__(AscendInputBatch)
    args = (batch, CUDAGraphMode.NONE, (), None, mocker.sentinel.attn_groups, None)
    assert state.prepare_attn(*args) == {}
    prepare.assert_called_once_with(*args, for_capture=False, ubatch_idx=0, model_specific_attn_metadata=None)
    assert state.uses_ascend_attn_metadata
    assert NPUHunyuanImage3ModelState.prepare_inputs is OmniModelState.prepare_inputs
    assert NPUHunyuanImage3ModelState.custom_sampler is OmniModelState.custom_sampler


@pytest.mark.parametrize(
    ("starts", "lengths", "ranges", "expected"),
    [
        ([0, 8], [8], {0: [(2, 4)]}, [(0, 0, 2, 5, 5)]),
        ([0, 4], [8], {0: [(2, 6)]}, [(0, 0, 0, 3, 7)]),
        ([0, 4], [4], {0: [(2, 6)]}, [(0, 0, 2, 4, 4)]),
        ([0, 4], [8], {0: [(0, 3), (8, 10)]}, []),
        ([0, 8], [8], {0: [(1, 1)]}, []),
        ([0, 4, 7], [8, 6], {0: [(2, 6)], 1: [(3, 4)]}, [(0, 0, 0, 3, 7), (1, 4, 4, 6, 5)]),
        ([0, 8], [8], {}, []),
    ],
)
def test_prefix_queries_preserve_inclusive_ranges_and_chunk_offsets(starts, lengths, ranges, expected):
    assert _prefix_queries(starts, lengths, ranges) == expected


def _attention(query, key, value, allowed):
    repeat = query.shape[1] // key.shape[1]
    key = key.repeat_interleave(repeat, dim=1)
    value = value.repeat_interleave(repeat, dim=1)
    scores = torch.einsum("qhd,khd->hqk", query, key) * 0.5
    scores.masked_fill_(~allowed[None], float("-inf"))
    return torch.einsum("hqk,khd->qhd", scores.softmax(-1), value)


@pytest.mark.parametrize("cached", [False, True])
def test_prefix_attention_matches_dense_mask_without_changing_text_queries(mocker, cached):
    generator = torch.Generator().manual_seed(0)
    key = torch.randn(8, 1, 4, generator=generator)
    value = torch.randn(8, 1, 4, generator=generator)
    past = 4 if cached else 0
    query = torch.randn(8 - past, 2, 4, generator=generator)
    positions = torch.arange(past, 8)[:, None]
    keys = torch.arange(8)[None, :]
    left, right = (2, 6) if cached else (2, 4)
    causal = keys <= positions
    allowed = causal | ((positions >= left) & (positions <= right) & (keys >= left) & (keys <= right))
    metadata = mocker.Mock()
    metadata._omni_prefix_queries = _prefix_queries([0, len(query)], [8], {0: [(left, right)]})
    output = torch.empty_like(query)
    impl = NPUHunyuanImage3AttentionImpl.__new__(NPUHunyuanImage3AttentionImpl)
    impl.num_heads, impl.num_kv_heads, impl.head_size, impl.scale = 2, 1, 4, 0.5
    table = torch.tensor([[0, 1]], dtype=torch.int32) if cached else None
    mocker.patch.object(impl, "_get_fia_params", return_value=(key, value, 4, table, [8]))

    def base_forward(query, key, value, metadata, output, kv_cache=None):
        output.copy_(_attention(query, key, value, causal))
        return output

    base = mocker.patch.object(
        OmniStaticShapeAttentionBackendImpl, "forward_fused_infer_attention", side_effect=base_forward
    )

    def fused(**kwargs):
        assert "atten_mask" not in kwargs
        assert kwargs["sparse_mode"] == 0
        assert kwargs["actual_seq_lengths_kv"] == [right + 1]
        if cached:
            torch.testing.assert_close(kwargs["block_table"], table)
        else:
            assert kwargs["block_table"] is None
        length = right + 1
        selected_query = kwargs["query"]
        return _attention(
            selected_query,
            kwargs["key"][:length],
            kwargs["value"][:length],
            torch.ones(len(selected_query), length, dtype=torch.bool),
        ), None

    call = mocker.patch.object(torch_npu, "npu_fused_infer_attention_score", side_effect=fused)
    assert impl.forward_fused_infer_attention(query, key, value, metadata, output) is output
    torch.testing.assert_close(output, _attention(query, key, value, allowed))
    base.assert_called_once()
    call.assert_called_once()


def test_prefix_attention_delegates_plain_decode(mocker):
    impl = NPUHunyuanImage3AttentionImpl.__new__(NPUHunyuanImage3AttentionImpl)
    metadata = mocker.Mock(_omni_prefix_queries=[])
    base = mocker.patch.object(
        OmniStaticShapeAttentionBackendImpl, "forward_fused_infer_attention", return_value=mocker.sentinel.output
    )
    fia = mocker.patch.object(torch_npu, "npu_fused_infer_attention_score")
    query = torch.zeros(1, 1, 1)
    output = torch.empty_like(query)
    assert impl.forward_fused_infer_attention(query, None, None, metadata, output) is mocker.sentinel.output
    base.assert_called_once_with(query, None, None, metadata, output, None)
    fia.assert_not_called()


def test_prepare_attn_binds_local_backend_and_clears_prefix_queries_for_decode(mocker):
    state = NPUHunyuanImage3ModelState.__new__(NPUHunyuanImage3ModelState)
    state.encoder_cache = mocker.Mock(mm_features={"request": [mocker.sentinel.image]})
    state.model_config = mocker.Mock()
    state.model_config.get_sliding_window.return_value = None
    state.vllm_config = mocker.Mock()
    state.vllm_config.parallel_config.decode_context_parallel_size = 1
    state.pcp_context = None
    impl = OmniStaticShapeAttentionBackendImpl.__new__(OmniStaticShapeAttentionBackendImpl)
    state.vllm_config.compilation_config.static_forward_context = {"attention": mocker.Mock(impl=impl)}
    batch = mocker.Mock(spec=AscendInputBatch)
    batch.is_dummy, batch.has_prefill, batch.num_reqs = False, True, 1
    batch.req_ids = ["request"]
    batch.query_start_loc_np = np.array([0, 6], dtype=np.int32)
    batch.seq_lens_np = np.array([6], dtype=np.int32)
    metadata = mocker.Mock()
    mocker.patch.object(AscendModelState, "prepare_attn", return_value={"attention": metadata})
    ranges = mocker.patch.object(hunyuan_image3_model_state, "compute_mm_prefix_ranges", return_value={0: [(1, 3)]})
    args = (batch, CUDAGraphMode.NONE, (), None, mocker.sentinel.attn_groups, None)
    state.prepare_attn(*args)
    assert type(impl) is NPUHunyuanImage3AttentionImpl
    assert metadata._omni_prefix_queries == [(0, 0, 1, 4, 4)]
    ranges.assert_called_once_with([[mocker.sentinel.image]], sliding_window=None)
    batch.has_prefill = False
    state.prepare_attn(*args)
    assert metadata._omni_prefix_queries == []
    assert ranges.call_count == 1
