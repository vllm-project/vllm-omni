# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Phase routing and independently owned recurrent state need no CUDA."""

import pytest
import torch
from omegaconf import DictConfig
from torch import nn
from vllm.config import CompilationConfig, CUDAGraphMode, ModelConfig, VllmConfig
from vllm.v1.worker.encoder_cudagraph import EncoderCudaGraphManager

from vllm_omni.model_executor.models.nemotron_voicechat import perception_cudagraph as graphs
from vllm_omni.model_executor.models.nemotron_voicechat.nemo_vendored.perception import (
    AudioPerceptionModule,
    IdentityConnector,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


class _Encoder(nn.Module):
    _feat_in = 2

    def cache_aware_stream_step(
        self,
        *,
        processed_signal,
        processed_signal_length,
        cache_last_channel,
        cache_last_time,
        cache_last_channel_len,
        keep_all_outputs,
        drop_extra_pre_encoded,
    ):
        signal = processed_signal[:, :, -1:] + cache_last_channel[:, :, -1, :].transpose(1, 2)
        value = signal.mean()
        return (
            signal,
            torch.ones_like(processed_signal_length),
            cache_last_channel + value,
            cache_last_time + value,
            (cache_last_channel_len + 1).clamp(max=2),
        )


@pytest.fixture
def perception():
    model = AudioPerceptionModule.__new__(AudioPerceptionModule)
    nn.Module.__init__(model)
    model.cfg = DictConfig({"output_dim": 2})
    model.encoder = _Encoder()
    model.modality_adapter = IdentityConnector()
    model.proj = nn.Identity()
    return model.eval()


def _caches():
    return torch.zeros(1, 1, 2, 2), torch.zeros(1, 1, 2, 1), torch.zeros(1, dtype=torch.long)


class _CPUManager(EncoderCudaGraphManager):
    """Replace only CUDA replay while exercising the adapter's real protocol."""

    def __init__(self, adapter):
        self.adapter = adapter
        self.calls = 0
        self.outputs: dict[str, torch.Tensor] = {}
        self.buffers = {
            path: adapter.prepare_encoder_cudagraph_capture_inputs(
                1, 1, 0, torch.device("cpu"), torch.float32, path
            ).values
            for path in adapter.paths
        }

    def execute(self, inputs):
        self.calls += 1
        phase = self.adapter.phase(inputs)
        path = next(path for path, value in self.adapter.paths.items() if value == phase)
        buffers = self.buffers[path]
        replay = self.adapter.prepare_encoder_cudagraph_replay_buffers(inputs, 1, 0).values
        for key, tensor in replay.items():
            buffers[key].copy_(tensor)
        encoded = self.adapter.encoder_cudagraph_forward(buffers, path)
        if path not in self.outputs:
            self.outputs[path] = torch.empty_like(encoded)
        self.outputs[path].copy_(encoded)
        result: dict[int, torch.Tensor] = {}
        self.adapter.postprocess_encoder_output({path: self.outputs[path]}, [0], [1], result, clone=True)
        return [result[0]]


def test_graph_returned_state_is_private_across_replays(perception, monkeypatch):
    graph = graphs.PerceptionCudaGraph.__new__(graphs.PerceptionCudaGraph)
    graph.perception = perception
    graph.device, graph.dtype = torch.device("cpu"), torch.float32
    graph.adapter = graphs._PerceptionEncoder(perception, ((1, 0), (3, 1)), _caches())
    graph.manager = _CPUManager(graph.adapter)
    graph.stream = object()
    monkeypatch.setattr(graphs, "current_stream", lambda: graph.stream)
    with torch.inference_mode():
        first = graph.run(torch.ones(1, 2, 1), _caches(), 0)
        snapshot = tuple(value.clone() for value in first)
        graph.run(torch.full((1, 2, 1), 9.0), _caches(), 0)
        assert all(torch.equal(a, b) for a, b in zip(first, snapshot))
        steady = graph.run(torch.full((1, 2, 3), 2.0), first[2:], 1)
        expected = graphs.perception_step(perception, torch.full((1, 2, 3), 2.0), first[2:], 1)
        assert all(torch.equal(a, b) for a, b in zip(steady, expected))
    assert graph.manager.calls == 3


def test_equal_mel_widths_still_select_by_drop(perception):
    adapter = graphs._PerceptionEncoder(perception, ((3, 0), (3, 1)), _caches())
    for drop, path in ((0, "default"), (1, "steady")):
        spec = adapter.get_encoder_cudagraph_item_specs({"mel": torch.zeros(1, 2, 3), "drop": drop})[0]
        assert spec.path_output_tokens[path] == 1
        assert sum(spec.path_output_tokens.values()) == 1


@pytest.mark.parametrize("reason", ["shape", "stream", "autocast"])
def test_runtime_ineligible_frame_stays_eager(perception, monkeypatch, reason):
    graph = graphs.PerceptionCudaGraph.__new__(graphs.PerceptionCudaGraph)
    graph.perception = perception
    graph.device, graph.dtype = torch.device("cpu"), torch.float32
    graph.adapter = graphs._PerceptionEncoder(perception, ((1, 0), (3, 1)), _caches())
    graph.manager = _CPUManager(graph.adapter)
    graph.stream = object()
    monkeypatch.setattr(graphs, "current_stream", lambda: object() if reason == "stream" else graph.stream)
    monkeypatch.setattr(torch, "is_autocast_enabled", lambda device: reason == "autocast")
    mel = torch.ones(1, 2, 2 if reason == "shape" else 1)
    caches = _caches()
    with torch.inference_mode():
        actual = graph.run(mel, caches, 0)
        expected = graphs.perception_step(perception, mel, caches, 0)
    assert all(torch.equal(a, b) for a, b in zip(actual, expected))
    assert graph.manager.calls == 0


@pytest.mark.parametrize("reason", ["disabled", "enforce_eager", "mode_none", "cpu", "multi_rank"])
def test_ineligible_configuration_never_constructs_manager(perception, monkeypatch, reason):
    config = VllmConfig.__new__(VllmConfig)
    config.compilation_config = CompilationConfig(cudagraph_mm_encoder=True, cudagraph_mode=CUDAGraphMode.FULL)
    config.model_config = ModelConfig.__new__(ModelConfig)
    config.model_config.enforce_eager = False
    monkeypatch.setattr(graphs.current_omni_platform, "is_cuda", lambda: True)
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: False)
    if reason == "disabled":
        config.compilation_config.cudagraph_mm_encoder = False
    elif reason == "enforce_eager":
        config.model_config.enforce_eager = True
    elif reason == "mode_none":
        config.compilation_config.cudagraph_mode = CUDAGraphMode.NONE
    elif reason == "cpu":
        monkeypatch.setattr(graphs.current_omni_platform, "is_cuda", lambda: False)
    else:
        monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
        monkeypatch.setattr(torch.distributed, "get_world_size", lambda: 2)

    def unexpected(*args):
        raise AssertionError("ineligible configurations must remain eager")

    monkeypatch.setattr(graphs, "PerceptionCudaGraph", unexpected)
    assert graphs.make_perception_graph(perception, config, ((1, 0), (3, 1))) is None
