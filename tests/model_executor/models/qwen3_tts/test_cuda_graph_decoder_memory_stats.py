# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import pytest
import torch

from tests.model_executor.models.qwen3_tts.test_qwen3_tts_code2wav import _make_model
from tests.model_executor.models.qwen3_tts.test_qwen3_tts_incremental_decode import _make_small_decoder
from vllm_omni.model_executor.models.qwen3_tts.segmented_graph_wrapper import CUDAGraphDecoderWrapper

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


@pytest.fixture
def decoder_graph_capture(monkeypatch):
    # Keep production wrapper selection and warmup; replace only GPU capture
    # and cache allocation so both warmup paths can run on a CPU host.
    def capture_icl(wrapper, batch_size, device, dtype):
        wrapper.icl_prefix_states[batch_size] = {}

    def capture_xvec(wrapper, batch_size, device, dtype):
        wrapper.xvec_prefix_states[batch_size] = {}

    def capture_suffix(wrapper, mode, batch_size, size, caches, device, dtype):
        wrapper.combined_states[(mode, batch_size, size)] = {}

    def capture_stateless(wrapper, batch_size, size, device, dtype):
        wrapper.stateless_states[(batch_size, size)] = {}

    monkeypatch.setattr(CUDAGraphDecoderWrapper, "_capture_icl_prefix", capture_icl)
    monkeypatch.setattr(CUDAGraphDecoderWrapper, "_capture_xvec_prefix", capture_xvec)
    monkeypatch.setattr(CUDAGraphDecoderWrapper, "_capture_combined_suffix", capture_suffix)
    monkeypatch.setattr(CUDAGraphDecoderWrapper, "_capture_stateless", capture_stateless)
    monkeypatch.setattr(CUDAGraphDecoderWrapper, "_make_dummy_icl_cache", lambda *_: {})
    monkeypatch.setattr(CUDAGraphDecoderWrapper, "_make_dummy_xvec_cache", lambda *_: {})


def _enable_graphs(model):
    model._maybe_enable_decoder_cudagraph(
        device=torch.device("cuda"),
        codec_chunk_frames=25,
        codec_left_context_frames=72,
        initial_codec_chunk_frames=1,
        codec_chunk_ramp=None,
        decode_cudagraph_batch_sizes=[1],
        decode_cudagraph_capture_sizes=[4],
    )


@pytest.mark.parametrize("async_chunk", [False, True])
def test_live_decoder_warmup_retains_memory_snapshot(decoder_graph_capture, mocker, async_chunk):
    model = _make_model(async_chunk=async_chunk)
    model.decoder = _make_small_decoder()
    allocated = mocker.patch.object(torch.accelerator, "memory_allocated", return_value=17)
    reserved = mocker.patch.object(torch.accelerator, "memory_reserved", return_value=23)

    assert model.get_stage_post_warmup_memory_stats() is None
    _enable_graphs(model)

    wrapper = model.decoder._cudagraph_wrapper
    assert isinstance(wrapper, CUDAGraphDecoderWrapper)
    assert wrapper._warmed_up
    # RPCs read the retained snapshot, even if allocator values later change.
    allocated.return_value = 99
    reserved.return_value = 101
    for _ in range(2):
        stats = model.get_stage_post_warmup_memory_stats()
        assert stats is not None
        assert (stats.allocated_bytes, stats.reserved_bytes) == (17, 23)
    wrapper.warmup(torch.device("cuda"))
    allocated.assert_called_once_with(torch.device("cuda"))
    reserved.assert_called_once_with(torch.device("cuda"))


@pytest.mark.parametrize("async_chunk", [False, True])
def test_failed_graph_capture_omits_snapshot(decoder_graph_capture, mocker, async_chunk):
    model = _make_model(async_chunk=async_chunk)
    model.decoder = _make_small_decoder()
    for name in ("_capture_icl_prefix", "_capture_xvec_prefix", "_capture_combined_suffix", "_capture_stateless"):
        mocker.patch.object(CUDAGraphDecoderWrapper, name, side_effect=RuntimeError("capture failed"))
    allocated = mocker.patch.object(torch.accelerator, "memory_allocated")
    reserved = mocker.patch.object(torch.accelerator, "memory_reserved")

    _enable_graphs(model)

    assert not model.decoder._cudagraph_wrapper._warmed_up
    assert model.get_stage_post_warmup_memory_stats() is None
    allocated.assert_not_called()
    reserved.assert_not_called()


@pytest.mark.parametrize("async_chunk", [False, True])
def test_unavailable_memory_stats_do_not_fail_warmup(decoder_graph_capture, mocker, async_chunk):
    model = _make_model(async_chunk=async_chunk)
    model.decoder = _make_small_decoder()
    mocker.patch.object(torch.accelerator, "memory_allocated", side_effect=RuntimeError("stats unavailable"))

    _enable_graphs(model)

    assert model.decoder._cudagraph_wrapper._warmed_up
    assert model.get_stage_post_warmup_memory_stats() is None


def test_eager_decoder_omits_snapshot(decoder_graph_capture, mocker):
    model = _make_model()
    model.decoder = _make_small_decoder()
    model.vllm_config.model_config.enforce_eager = True
    allocated = mocker.patch.object(torch.accelerator, "memory_allocated")
    reserved = mocker.patch.object(torch.accelerator, "memory_reserved")

    _enable_graphs(model)

    assert model.get_stage_post_warmup_memory_stats() is None
    allocated.assert_not_called()
    reserved.assert_not_called()
