# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""CPU routing contracts and exact CUDA replay of independent recurrent state."""

from collections import Counter

import pytest
import torch
from omegaconf import DictConfig
from torch import nn
from vllm.config import CompilationConfig, CUDAGraphMode, DeviceConfig, ModelConfig, ParallelConfig, VllmConfig
from vllm.utils import torch_utils
from vllm.v1.worker.encoder_cudagraph import EncoderCudaGraphManager

from tests.helpers.mark import hardware_test
from vllm_omni.model_executor.models.nemotron_voicechat import nemotron_voicechat_thinker as thinker_module
from vllm_omni.model_executor.models.nemotron_voicechat import perception_cudagraph as graphs
from vllm_omni.model_executor.models.nemotron_voicechat.nemo_vendored.asr.conformer_encoder import ConformerEncoder
from vllm_omni.model_executor.models.nemotron_voicechat.nemo_vendored.perception import (
    AudioPerceptionModule,
    IdentityConnector,
)

pytestmark = pytest.mark.core_model


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


def _caches(batch_size=1):
    return torch.zeros(1, batch_size, 2, 2), torch.zeros(1, batch_size, 2, 1), torch.zeros(batch_size, dtype=torch.long)


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

    def get_cumulative_stats(self):
        return {"graph_hits": self.calls, "graph_misses": 0, "hit_rate": 1.0 if self.calls else 0.0}


@pytest.mark.cpu
def test_graph_returned_state_is_private_across_replays(perception, monkeypatch):
    graph = graphs.PerceptionCudaGraph.__new__(graphs.PerceptionCudaGraph)
    graph.perception = perception
    graph.device, graph.dtype = torch.device("cpu"), torch.float32
    graph.adapter = graphs._PerceptionEncoder(perception, ((1, 0), (3, 1)), _caches())
    graph.manager = _CPUManager(graph.adapter)
    graph._fallback_reasons = Counter()
    graph.stream = object()
    monkeypatch.setattr(torch.cuda, "current_stream", lambda device=None: graph.stream)
    with torch.inference_mode():
        first = graph.run(torch.ones(1, 2, 1), _caches(), 0)
        snapshot = tuple(value.clone() for value in first)
        graph.run(torch.full((1, 2, 1), 9.0), _caches(), 0)
        assert all(torch.equal(a, b) for a, b in zip(first, snapshot))
        steady = graph.run(torch.full((1, 2, 3), 2.0), first[2:], 1)
        expected = graphs.perception_step(perception, torch.full((1, 2, 3), 2.0), first[2:], 1)
        assert all(torch.equal(a, b) for a, b in zip(steady, expected))
    assert graph.manager.calls == 3
    assert graph.get_cumulative_stats()["ineligible_misses"] == 0
    monkeypatch.setattr(torch.cuda, "current_stream", lambda device=None: object())
    with torch.inference_mode():
        graph.run(torch.ones(1, 2, 1), _caches(), 0)
    stats = graph.get_cumulative_stats()
    assert stats["graph_hits"] == 3 and stats["graph_misses"] == 1
    assert stats["hit_rate"] == 0.75


@pytest.mark.cpu
def test_equal_mel_widths_still_select_by_drop(perception):
    adapter = graphs._PerceptionEncoder(perception, ((3, 0), (3, 1)), _caches())
    for drop, path in ((0, "default"), (1, "steady")):
        spec = adapter.get_encoder_cudagraph_item_specs({"mel": torch.zeros(1, 2, 3), "drop": drop})[0]
        assert spec.path_output_tokens[path] == 1
        assert sum(spec.path_output_tokens.values()) == 1


@pytest.mark.cpu
@pytest.mark.parametrize("phase", [(1, 0), (3, 1)])
def test_manager_eager_without_capture_owns_and_reuses_side_buffers(perception, phase, monkeypatch):
    adapter = graphs._PerceptionEncoder(perception, ((1, 0), (3, 1)), _caches())
    width, drop = phase
    inputs = {
        "mel": torch.ones(1, 2, width),
        "drop": drop,
        **dict(zip(("channel", "time", "valid"), _caches())),
    }
    frame = graphs.perception_step(perception, inputs["mel"], _caches(), drop)
    monkeypatch.setattr(graphs, "perception_step", lambda *args: frame)
    assert torch.equal(adapter.encoder_eager_forward(inputs), frame[0])
    buffers = adapter.outputs[phase]
    assert all(torch.equal(target, source) for target, source in zip(buffers, frame[1:]))
    snapshot = tuple(value.clone() for value in buffers)
    for value in frame[1:]:
        value.add_(1)
    assert all(torch.equal(value, saved) for value, saved in zip(buffers, snapshot))
    assert torch.equal(adapter.encoder_eager_forward(inputs), frame[0])
    assert all(target is saved for target, saved in zip(adapter.outputs[phase], buffers))
    assert all(torch.equal(target, source) for target, source in zip(buffers, frame[1:]))


@pytest.mark.cpu
def test_manager_eager_does_not_broadcast_between_batch_sizes(perception):
    adapter = graphs._PerceptionEncoder(perception, ((1, 0), (3, 1)), _caches())
    for batch in (2, 1):
        inputs = {
            "mel": torch.ones(batch, 2, 1),
            "drop": 0,
            **dict(zip(("channel", "time", "valid"), _caches(batch))),
        }
        if batch == 2:
            adapter.encoder_eager_forward(inputs)
        else:
            with pytest.raises(ValueError, match="shape"):
                adapter.encoder_eager_forward(inputs)


@pytest.mark.cpu
@pytest.mark.parametrize(
    "reason", ["phase", "drop", "shape", "stream", "autocast", "training", "device", "dtype", "cache", "grad_enabled"]
)
def test_runtime_ineligible_frame_stays_eager(perception, monkeypatch, reason):
    warnings = []
    monkeypatch.setattr(graphs.logger, "warning", lambda *args: warnings.append(args))
    graph = graphs.PerceptionCudaGraph.__new__(graphs.PerceptionCudaGraph)
    graph.perception = perception
    graph.device, graph.dtype = torch.device("cpu"), torch.float32
    graph.adapter = graphs._PerceptionEncoder(perception, ((1, 0), (3, 1)), _caches())
    graph.manager = _CPUManager(graph.adapter)
    graph._fallback_reasons = Counter()
    graph.stream = object()
    monkeypatch.setattr(
        torch.cuda, "current_stream", lambda device=None: object() if reason == "stream" else graph.stream
    )
    monkeypatch.setattr(torch, "is_autocast_enabled", lambda device: reason == "autocast")
    mel = torch.ones(1, 2, 2 if reason == "phase" else 1)
    drop = 2 if reason == "drop" else 0
    caches = _caches()
    if reason == "training":
        perception.train()
    elif reason == "shape":
        mel = mel.expand(2, -1, -1).contiguous()
        caches = tuple(torch.cat([cache, cache], dim=0 if index == 2 else 1) for index, cache in enumerate(caches))
    elif reason == "device":
        graph.device = torch.device("meta")
    elif reason == "dtype":
        graph.dtype = torch.float64
    elif reason == "cache":
        graph.adapter.caches = tuple(cache.to(torch.float64) for cache in caches)
    with torch.set_grad_enabled(reason == "grad_enabled"):
        actual = graph.run(mel, caches, drop)
        graph.run(mel, caches, drop)
        expected = graphs.perception_step(perception, mel, caches, drop)
    assert all(torch.equal(a, b) for a, b in zip(actual, expected))
    assert graph.manager.calls == 0
    stats = graph.get_cumulative_stats()
    assert stats["graph_hits"] == 0
    assert stats["graph_misses"] == stats["ineligible_misses"] == 2 * mel.shape[0]
    fallback_reason = "phase" if reason == "drop" else reason
    assert stats["fallback_reasons"] == {fallback_reason: 2 * mel.shape[0]}
    assert stats["hit_rate"] == 0.0
    assert len(warnings) == 1


@pytest.mark.cpu
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


def _small_perception():
    model = AudioPerceptionModule.__new__(AudioPerceptionModule)
    nn.Module.__init__(model)
    model.cfg = DictConfig({"output_dim": 8})
    model.encoder = ConformerEncoder(
        feat_in=8,
        n_layers=2,
        d_model=16,
        n_heads=2,
        subsampling="dw_striding",
        subsampling_factor=8,
        subsampling_conv_channels=8,
        causal_downsampling=True,
        att_context_size=[5, 0],
        att_context_style="chunked_limited",
        conv_kernel_size=9,
        conv_context_size="causal",
        conv_norm_type="layer_norm",
        dropout=0.0,
        dropout_att=0.0,
        dropout_emb=0.0,
        dropout_pre_encoder=0.0,
    )
    model.modality_adapter = IdentityConnector()
    model.proj = nn.Linear(16, 8)
    return model.eval()


def _config(device):
    config = VllmConfig.__new__(VllmConfig)
    config.compilation_config = CompilationConfig(cudagraph_mm_encoder=True, cudagraph_mode=CUDAGraphMode.FULL)
    config.model_config = ModelConfig.__new__(ModelConfig)
    config.model_config.enforce_eager = False
    config.model_config.multimodal_config = None
    config.parallel_config = ParallelConfig()
    config.device_config = DeviceConfig(device=device)
    return config


@pytest.mark.cpu
def test_worker_capture_uses_loaded_weights_and_actual_streaming_phases(monkeypatch):
    model = thinker_module.NemotronVoiceChatThinkerForConditionalGeneration.__new__(
        thinker_module.NemotronVoiceChatThinkerForConditionalGeneration
    )
    nn.Module.__init__(model)
    model.perception = _small_perception()
    model._streaming_preprocessor = nn.Identity()
    model.embed_tokens = nn.Embedding(4, 8)
    model.function_head = nn.Linear(8, 4, bias=False)
    model.language_model = nn.Linear(8, 8)
    model._dtype, model._use_function_head = torch.float32, True
    model._w_text = model._w_audio = model._w_func = 1.0
    model.vllm_config = _config("cpu")
    model._perception_graph = None
    reload_warnings: list[str] = []
    monkeypatch.setattr(
        thinker_module.logger,
        "warning_once",
        lambda message, *args, **kwargs: reload_warnings.append(message % args if args else message),
    )
    monkeypatch.setattr(
        model.language_model, "load_weights", lambda weights: {key for key, _ in weights}, raising=False
    )
    captured = []
    sentinel = object()

    def capture(perception, config, phases):
        assert perception is model.perception and config is model.vllm_config
        captured.append(phases)
        return sentinel

    monkeypatch.setattr(thinker_module, "make_perception_graph", capture)
    weights = [
        ("stt_model.embed_tokens.weight", model.embed_tokens.weight.detach().clone()),
        ("stt_model.function_head.weight", model.function_head.weight.detach().clone()),
        ("stt_model.llm.test", torch.zeros(1)),
        *[(f"stt_model.perception.{key}", value.clone()) for key, value in model.perception.state_dict().items()],
    ]
    model.load_weights(iter(weights))
    assert model._perception_graph is None and not captured
    assert not reload_warnings
    model.capture_auxiliary_graphs()
    assert model._perception_graph is sentinel
    model.capture_auxiliary_graphs()
    assert len(captured) == 1
    for index in (0, 1, 8):
        mel = torch.zeros(1, 8, 8 * (index + 1) + 1)
        chunk, drop = thinker_module.slice_perception_streaming_mel(mel, index, model.perception.encoder.streaming_cfg)
        assert captured[0][int(index > 0)] == (chunk.shape[-1], drop)
    model.release_auxiliary_graphs()
    assert model._perception_graph is None
    model.capture_auxiliary_graphs()
    assert model._perception_graph is sentinel and len(captured) == 2
    model.load_weights(iter(weights))
    assert model._perception_graph is None
    assert len(reload_warnings) == 1 and "capture_auxiliary_graphs" in reload_warnings[0]
    model.capture_auxiliary_graphs()
    assert model._perception_graph is sentinel and len(captured) == 3


@hardware_test(res={"cuda": "L4"}, num_cards=1)
@pytest.mark.nvidia_only
@pytest.mark.parametrize("force_eager", [False, True])
@torch.inference_mode()
def test_cuda_manager_replays_interleaved_sessions_exactly(monkeypatch, force_eager):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(42)
        perception = _small_perception().to("cuda")
    streaming = perception.encoder.streaming_cfg
    phases = []
    for index in (0, 1):
        mel = torch.zeros(1, 8, 8 * (index + 1) + 1)
        chunk, drop = thinker_module.slice_perception_streaming_mel(mel, index, streaming)
        phases.append((chunk.shape[-1], drop))
    replay_stream = torch.cuda.current_stream()
    # Reproduce weight loading before vLLM has cached a stream on this thread.
    monkeypatch.setattr(torch_utils._current_stream_tls, "value", None, raising=False)
    with monkeypatch.context() as patch:
        if force_eager:
            patch.setattr(EncoderCudaGraphManager, "capture", lambda *args, **kwargs: None)
        graph = graphs.make_perception_graph(perception, _config("cuda"), tuple(phases))
    assert graph is not None
    assert torch.cuda.current_stream() == replay_stream
    assert graph.stream == replay_stream
    assert graph.manager.path_token_budgets == {"default": [0, 1], "steady": [0, 1]}
    assert bool(graph.adapter.outputs) is not force_eager
    states = [perception.encoder.get_initial_cache_state(1, torch.float32, torch.device("cuda")) for _ in range(2)]
    references = [tuple(value.clone() for value in caches) for caches in states]
    generator = torch.Generator(device="cuda").manual_seed(123)
    frame_count = states[0][0].shape[2] + 3
    saved = None
    saved_copy = None
    for index in range(frame_count):
        width, drop = phases[int(index > 0)]
        for row in (0, 1) if index % 2 else (1, 0):
            mel = torch.randn(1, 8, width, generator=generator, device="cuda")
            with monkeypatch.context() as patch:
                if force_eager and (index < 2 or (index == 2 and row == 0)):
                    patch.setattr(graph.manager, "_path_fitting_budget", lambda path, tokens: None)
                actual = graph.run(mel, states[row], drop)
            assert torch.cuda.current_stream() == replay_stream
            expected = graphs.perception_step(perception, mel, references[row], drop)
            for result, reference in zip(actual, expected):
                assert torch.equal(result, reference)
            states[row], references[row] = actual[2:], expected[2:]
            if saved is None:
                saved, saved_copy = actual, tuple(value.clone() for value in actual)
            assert all(torch.equal(value, copy) for value, copy in zip(saved, saved_copy))
        if force_eager and index == 1:
            # Eager first/steady phases must also survive subsequent real capture.
            capture_stream = torch.cuda.Stream()
            capture_stream.wait_stream(replay_stream)
            with torch.cuda.stream(capture_stream):
                graph.manager.capture(graph_pool=None)
            replay_stream.wait_stream(capture_stream)
    assert all(bool((state[2] == state[0].shape[2]).all()) for state in states)
    stats = graph.get_cumulative_stats()
    # Two streams at first/steady startup (2 + 2), then one miss after capture.
    misses = 5 if force_eager else 0
    assert stats["graph_hits"] == 2 * frame_count - misses
    assert stats["graph_misses"] == misses
    assert stats["ineligible_misses"] == 0
