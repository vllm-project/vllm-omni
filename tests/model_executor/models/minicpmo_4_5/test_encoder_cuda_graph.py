# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import pytest
import torch

from vllm_omni.model_executor.models.minicpmo_4_5.encoder_cuda_graph import EncoderCudaGraph
from vllm_omni.model_executor.models.minicpmo_4_5.minicpmo_4_5_omni_llm import (
    SiglipVisionConfig,
    SiglipVisionTransformer,
)
from vllm_omni.platforms import current_omni_platform

pytestmark = [pytest.mark.core_model]


def _make_graph(forward, **kwargs):
    from vllm.config import ModelConfig, VllmConfig
    from vllm.config.multimodal import MultiModalConfig

    config = VllmConfig()
    # The manager only needs multimodal limits; no checkpoint is loaded here.
    config.model_config = ModelConfig.__new__(ModelConfig)
    config.model_config.multimodal_config = MultiModalConfig()
    return EncoderCudaGraph(forward, config, **kwargs)


@pytest.mark.cpu
def test_cpu_and_grad_paths_remain_eager():
    graph = _make_graph(lambda x, mask: x.sin() if mask is None else x.sin() + mask)
    x = torch.randn(2, 3, requires_grad=True)
    graph(x, None).sum().backward()
    torch.testing.assert_close(x.grad, x.detach().cos())
    assert not graph.graphs
    assert not graph._seen


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@torch.inference_mode()
def test_replay_refreshes_inputs_preserves_outputs_and_bounds_shapes():
    graph = _make_graph(lambda x, mask: x.sin() if mask is None else x.sin() + mask, max_graphs=2)
    x = torch.randn(2, 8, device="cuda")
    mask = torch.randn_like(x)
    graph(x, mask)
    old = graph(x, mask)
    expected_old = old.clone()
    x.mul_(2)
    mask.add_(3)
    torch.testing.assert_close(graph(x, mask), x.sin() + mask)
    torch.testing.assert_close(old, expected_old)
    graph(x, None)
    torch.testing.assert_close(graph(x, None), x.sin())
    assert len(graph.graphs) == 2
    from vllm.v1.worker.encoder_cudagraph import EncoderCudaGraphManager

    assert all(isinstance(entry, EncoderCudaGraphManager) for entry in graph.graphs.values())
    assert sum(entry.graph_hits for entry in graph.graphs.values()) == 3
    assert graph.vllm_config.compilation_config.encoder_cudagraph_token_budgets == []
    for size in range(3, 24):
        other = torch.randn(size, 8, device="cuda")
        torch.testing.assert_close(graph(other, None), other.sin())
    assert len(graph.graphs) == 2
    assert len(graph._seen) <= 8


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@torch.inference_mode()
def test_vision_graph_handles_changed_mask_and_retained_embeddings():
    config = SiglipVisionConfig(
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        image_size=28,
        patch_size=14,
    )
    config._attn_implementation = "eager"
    model = SiglipVisionTransformer(config).eval().cuda()
    pixels = torch.randn(2, 3, 14, 56, device="cuda")
    sizes = torch.tensor([[2, 2], [1, 2]], dtype=torch.int32)
    mask = torch.tensor([[[1, 1, 1, 1]], [[1, 1, 0, 0]]], device="cuda", dtype=torch.bool)
    reference = model(pixels, mask, sizes).last_hidden_state
    model._encoder_graph = _make_graph(model._encode_last_hidden_state)
    model(pixels, mask, sizes)
    output = model(pixels, mask, sizes).last_hidden_state
    torch.testing.assert_close(output, reference)
    assert len(model._encoder_graph.graphs) == 1
    # Same padded shape, different valid positions and image content.
    pixels.add_(1)
    sizes[1] = torch.tensor([1, 3])
    mask[1, 0, 2] = True
    actual = model(pixels, mask, sizes).last_hidden_state
    graph = model._encoder_graph
    model._encoder_graph = None
    expected = model(pixels, mask, sizes).last_hidden_state
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(output, reference)
    model._encoder_graph = graph
    detailed = model(pixels, mask, sizes, output_hidden_states=True)
    assert len(detailed.hidden_states) == 3
    assert len(graph.graphs) == 1


def _audio_model(device):
    from types import SimpleNamespace

    from transformers import WhisperConfig

    from vllm_omni.model_executor.models.minicpmo_4_5.minicpmo_4_5_omni_llm import (
        MiniCPMO45OmniLLMForConditionalGeneration,
        MiniCPMWhisperEncoder,
        MultiModalProjector,
    )

    config = WhisperConfig(
        d_model=32,
        encoder_layers=2,
        encoder_attention_heads=4,
        encoder_ffn_dim=64,
        num_mel_bins=80,
        max_source_positions=1500,
    )
    config._attn_implementation = "sdpa"
    model = MiniCPMO45OmniLLMForConditionalGeneration.__new__(MiniCPMO45OmniLLMForConditionalGeneration)
    torch.nn.Module.__init__(model)
    model.config = SimpleNamespace(audio_chunk_length=1.0, audio_pool_step=5)
    model.apm = MiniCPMWhisperEncoder(config)
    model.audio_projection_layer = MultiModalProjector(32, 48)
    model.audio_avg_pooler = torch.nn.AvgPool1d(5, stride=5)
    model.audio_encoder_layer = -1
    model.audio_past_key_values = None
    return model.eval().to(device)


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@torch.inference_mode()
def test_audio_graph_refreshes_mask_and_matches_eager():
    model = _audio_model("cuda")
    data = {
        "audio_features": torch.randn(2, 80, 100, device="cuda"),
        "audio_feature_lens": torch.tensor([[100], [80]], device="cuda"),
    }
    expected = model.get_audio_hidden_states(data)
    graph = _make_graph(model._encode_audio_features)
    model._audio_encoder_graph = graph
    model.get_audio_hidden_states(data)
    actual = model.get_audio_hidden_states(data)
    for a, b in zip(actual, expected, strict=True):
        torch.testing.assert_close(a, b)
    assert len(graph.graphs) == 1
    data["audio_features"].add_(1)
    data["audio_feature_lens"][1, 0] = 30
    changed = model.get_audio_hidden_states(data)
    model._audio_encoder_graph = None
    for a, b in zip(changed, model.get_audio_hidden_states(data), strict=True):
        torch.testing.assert_close(a, b)
    for a, b in zip(actual, expected, strict=True):
        torch.testing.assert_close(a, b)


@pytest.mark.cpu
@torch.inference_mode()
def test_streaming_audio_does_not_replay_stateless_graph():
    model = _audio_model("cpu")

    def forbidden(*args):
        raise AssertionError("streaming KV must not use the stateless graph")

    model._audio_encoder_graph = forbidden
    data = {"audio_features": torch.randn(1, 80, 100), "audio_feature_lens": torch.tensor([[100]])}
    for _ in range(2):
        output = model.get_audio_embedding_streaming(data)
        assert output[0][0].shape == (10, 48)
    assert model.audio_past_key_values.get_seq_length() == 100


@pytest.mark.cpu
@torch.inference_mode()
def test_fp16_audio_keeps_host_overflow_check_eager():
    model = _audio_model("cpu").half()

    def forbidden(*args):
        raise AssertionError("FP16 overflow guard cannot run inside capture")

    model._audio_encoder_graph = forbidden
    data = {
        "audio_features": torch.randn(1, 80, 100, dtype=torch.float16),
        "audio_feature_lens": torch.tensor([[100]]),
    }
    assert model.get_audio_hidden_states(data)[0].shape == (10, 48)


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@torch.inference_mode()
def test_one_off_shapes_do_not_exhaust_capture_admission():
    graph = _make_graph(torch.sin, max_graphs=1)
    for size in range(1, 20):
        graph(torch.zeros(size, device="cuda"))
    assert len(graph._seen) == 4
    assert not graph.graphs
    x = torch.randn(32, device="cuda")
    graph(x)
    torch.testing.assert_close(graph(x), x.sin())
    assert len(graph.graphs) == 1


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@torch.inference_mode()
def test_managers_do_not_share_buffers_between_streams():
    graph = _make_graph(torch.sin)
    streams = [torch.cuda.Stream(), torch.cuda.Stream()]
    outputs = []
    for i, stream in enumerate(streams):
        with torch.cuda.stream(stream):
            value = torch.full((4, 8), float(i + 1), device="cuda")
            graph(value)
            output = graph(value)
            graph(value + 1)
            outputs.append((output, value.sin()))
    for stream in streams:
        torch.cuda.current_stream().wait_stream(stream)
    assert len(graph.graphs) == 2
    for actual, expected in outputs:
        torch.testing.assert_close(actual, expected)
    assert not graph.vllm_config.compilation_config.encoder_cudagraph_token_budgets


@pytest.mark.cpu
@pytest.mark.parametrize(
    "options",
    [
        {"max_graphs": -1},
        {"max_graphs": True},
        {"min_capture_calls": 1},
        {"min_capture_calls": 2.5},
        {"min_free_bytes": -1},
    ],
)
def test_invalid_admission_options(options):
    with pytest.raises(ValueError, match="must be an integer"):
        _make_graph(torch.sin, **options)


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@torch.inference_mode()
def test_configurable_admission_and_capacity_miss_accounting():
    graph = _make_graph(torch.sin, max_graphs=1, min_capture_calls=3)
    x = torch.randn(2, 8, device="cuda")
    for _ in range(2):
        torch.testing.assert_close(graph(x), x.sin())
        assert not graph.graphs
    torch.testing.assert_close(graph(x), x.sin())
    torch.testing.assert_close(graph(x + 1), (x + 1).sin())
    for _ in range(5):
        other = torch.randn(3, 8, device="cuda")
        torch.testing.assert_close(graph(other), other.sin())
    stats = graph.get_cumulative_stats()
    assert stats["num_graphs"] == 1
    assert stats["graph_hits"] == 2
    assert stats["graph_misses"] == 7
    assert stats["capacity_misses"] == 5
    assert stats["warmup_misses"] == 2
    assert stats["hit_rate"] == pytest.approx(2 / 9)


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@torch.inference_mode()
def test_memory_admission_defers_capture_but_preserves_existing_replay(monkeypatch):
    graph = _make_graph(torch.sin, min_free_bytes=1024)
    x = torch.randn(2, 8, device="cuda")
    graph(x)
    monkeypatch.setattr(current_omni_platform, "get_free_memory", lambda device: 1023)
    for _ in range(3):
        torch.testing.assert_close(graph(x), x.sin())
    assert not graph.graphs
    assert graph.get_cumulative_stats()["memory_misses"] == 3
    monkeypatch.setattr(current_omni_platform, "get_free_memory", lambda device: 1024)
    old = graph(x)
    assert len(graph.graphs) == 1
    monkeypatch.setattr(current_omni_platform, "get_free_memory", lambda device: 0)
    torch.testing.assert_close(graph(x + 1), (x + 1).sin())
    torch.testing.assert_close(old, x.sin())
    assert graph.get_cumulative_stats()["graph_hits"] == 2


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@torch.inference_mode()
def test_capture_failure_is_fatal_without_retry_or_eager_fallback(monkeypatch):
    from unittest.mock import Mock

    forward = Mock(side_effect=torch.sin)
    graph = _make_graph(forward)
    x = torch.randn(2, 8, device="cuda")
    graph(x)
    capture = Mock(side_effect=RuntimeError("injected capture failure"))
    monkeypatch.setattr(graph, "_capture", capture)
    with pytest.raises(RuntimeError, match="injected capture failure"):
        graph(x)
    for value in (x, torch.randn(3, 8, device="cuda"), torch.randn(2, 8)):
        with pytest.raises(RuntimeError, match="restart the worker"):
            graph(value)
    capture.assert_called_once()
    forward.assert_called_once()
    assert not graph.graphs
    assert not graph._seen
    assert not graph._seen_calls
    assert graph.get_cumulative_stats()["capture_failures"] == 1


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@torch.inference_mode()
def test_zero_capacity_disables_capture():
    graph = _make_graph(torch.sin, max_graphs=0)
    x = torch.randn(2, 8, device="cuda")
    for _ in range(3):
        torch.testing.assert_close(graph(x), x.sin())
    assert not graph.graphs
    assert not graph._seen
    assert graph.get_cumulative_stats()["capacity_misses"] == 3


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("share_pools", [False, True])
@torch.inference_mode()
def test_pool_reuse_preserves_outputs_across_late_capture_and_arbitrary_replay(share_pools):
    graph = _make_graph(lambda x: (x.sin() + 2).cos(), share_pools=share_pools)
    retained = []
    for size in (64, 128, 32, 96):
        x = torch.randn(size, 64, device="cuda")
        graph(x)
        retained.append((graph(x), (x.sin() + 2).cos()))
        # Replay older shapes between captures, including smaller -> larger.
        for previous in (64, size, 64):
            value = torch.randn(previous, 64, device="cuda")
            torch.testing.assert_close(graph(value), (value.sin() + 2).cos())
        for actual, expected in retained:
            torch.testing.assert_close(actual, expected)
    pools = {entry.graph_pool for entry in graph.graphs.values()}
    assert len(pools) == (1 if share_pools else 4)
    assert len(graph._capture_pools) == (1 if share_pools else 0)
    for size in (96, 32, 128, 64, 128, 32):
        value = torch.randn(size, 64, device="cuda")
        torch.testing.assert_close(graph(value), (value.sin() + 2).cos())
    for actual, expected in retained:
        torch.testing.assert_close(actual, expected)


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@torch.inference_mode()
def test_shared_pools_are_isolated_by_encoder_and_replay_stream():
    graphs = [_make_graph(torch.sin), _make_graph(torch.cos)]
    streams = [torch.cuda.Stream(), torch.cuda.Stream()]
    retained = []
    for graph in graphs:
        for stream in streams:
            with torch.cuda.stream(stream):
                for size in (16, 32):
                    x = torch.randn(size, 64, device="cuda")
                    graph(x)
                    retained.append((graph(x), graph.forward(x)))
                    graph(x + 1)
    for stream in streams:
        torch.cuda.current_stream().wait_stream(stream)
    pools = {entry.graph_pool for graph in graphs for entry in graph.graphs.values()}
    assert len(pools) == 4
    assert all(len(graph._capture_pools) == 2 for graph in graphs)
    for actual, expected in retained:
        torch.testing.assert_close(actual, expected)
