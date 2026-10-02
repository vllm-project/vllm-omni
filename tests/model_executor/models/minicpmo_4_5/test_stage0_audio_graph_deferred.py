# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""The Stage-0 streaming audio encoder's CUDA graphs are captured after the weights load, not at construction.

The duplex runtime is built in the model's ``__init__`` (#8146), where the
encoder is still in training mode with its initial weights; capturing there was
refused on every deployment ("stays per-session: training mode").

The stand-in tests run on CPU; ``test_post_load_capture_matches_eager_batched``
captures real (tiny) encoder graphs and needs a GPU.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from vllm_omni.model_executor.models.minicpmo_4_5.duplex.stage0 import MiniCPMO45Stage0DuplexRuntime
from vllm_omni.model_executor.models.minicpmo_4_5.minicpmo_4_5_omni import MiniCPMO45OmniForConditionalGeneration
from vllm_omni.model_executor.models.minicpmo_4_5.minicpmo_4_5_omni_llm import (
    MiniCPMO45OmniLLMForConditionalGeneration,
    MiniCPMWhisperEncoder,
    MultiModalProjector,
)
from vllm_omni.model_executor.models.minicpmo_4_5.streaming_audio_encoder import (
    StreamingAudioChunk,
    StreamingAudioKVCache,
    encode_streaming_audio_batch,
)

pytestmark = [pytest.mark.core_model]

_cpu = pytest.mark.cpu
_cuda = pytest.mark.cuda


class _Processor:
    """Streaming processor stand-in: 10 ms mel hop; unconfigured it streams 100 ms chunks."""

    def __init__(self) -> None:
        self._streaming_mel_processor = SimpleNamespace(buffer=np.zeros(0, dtype=np.float32), sample_rate=16000)
        self.mode: dict | None = None
        self.units = 0

    def set_streaming_mode(self, **kwargs) -> None:
        self.mode = kwargs

    def reset_streaming(self) -> None:
        self.units = 0

    def get_streaming_chunk_size(self) -> int:
        if self.mode is None:
            return 1600
        return self.mode["first_chunk_ms"] * 16 if self.units == 0 else self.mode["chunk_ms"] * 16

    def process_audio_streaming(self, audio_chunk, *, reset: bool, return_batch_feature: bool = False):
        self.units += 1
        return {"audio_features": np.zeros((1, 80, len(audio_chunk) // 160), dtype=np.float32)}


class _Thinker:
    def __init__(self, *, cuda_graph: bool | None = None) -> None:
        self.apm = torch.nn.Linear(1, 1)
        self.apm.train()
        self.calls: list[tuple[int, bool]] = []
        if cuda_graph is not None:
            self.config = SimpleNamespace(duplex_audio_encoder_cuda_graph=cuda_graph)

    def build_streaming_audio_graph_encoder(self, *, unit_frames: int) -> bool:
        self.calls.append((unit_frames, self.apm.training))
        return True


def _runtime(*, cuda_graph: bool | None = None) -> tuple[MiniCPMO45Stage0DuplexRuntime, _Thinker]:
    thinker = _Thinker(cuda_graph=cuda_graph)
    stage_model = SimpleNamespace(config=SimpleNamespace(), processor=_Processor(), thinker=thinker)
    return MiniCPMO45Stage0DuplexRuntime(stage_model, device="cpu"), thinker


@_cpu
def test_construction_does_not_capture() -> None:
    _, thinker = _runtime()
    assert thinker.calls == []


@_cpu
def test_capture_runs_once_after_load_in_eval_mode() -> None:
    runtime, thinker = _runtime()
    assert runtime.build_audio_cuda_graph() is True
    assert thinker.calls == [(100, False)]
    assert runtime.build_audio_cuda_graph() is False
    assert len(thinker.calls) == 1


@_cpu
def test_probe_uses_a_session_configured_copy_of_the_processor() -> None:
    # The steady unit is 1000 ms (100 mel frames), not the unconfigured 100 ms default.
    runtime, thinker = _runtime()
    runtime.build_audio_cuda_graph()
    assert thinker.calls[0][0] == 100
    assert runtime.processor.mode is None  # the shared processor stays untouched


@_cpu
def test_model_post_load_hook_builds_the_runtime_graph() -> None:
    runtime, thinker = _runtime()
    model = SimpleNamespace(
        _minicpmo45_duplex_data_plane_helper=runtime,
        thinker=thinker.apm,
        _module_device=lambda module: torch.device("cpu"),
    )
    MiniCPMO45OmniForConditionalGeneration.omni_post_load(model)
    assert thinker.calls == [(100, False)]
    # Stages without the duplex runtime (Talker, turn mode) have nothing to do.
    MiniCPMO45OmniForConditionalGeneration.omni_post_load(SimpleNamespace(thinker=None))


@_cpu
def test_disabled_switch_skips_the_probe_and_the_capture(monkeypatch: pytest.MonkeyPatch) -> None:
    # --hf-overrides '{"duplex_audio_encoder_cuda_graph": false}': nothing is probed or captured.
    runtime, thinker = _runtime(cuda_graph=False)
    probes: list[object] = []
    monkeypatch.setattr(runtime, "_configure_streaming_processor", lambda state=None: probes.append(state))
    assert runtime.build_audio_cuda_graph() is False
    assert thinker.calls == []
    assert probes == []


@_cpu
def test_runner_runs_post_load_once_at_the_start_of_profile_run(monkeypatch: pytest.MonkeyPatch) -> None:
    # Not inside load_model: vLLM's Worker.load_model scopes max_split_size_mb=20
    # around it, under which the graph pool cannot reuse blocks across captures.
    from vllm.v1.worker.gpu_model_runner import GPUModelRunner

    from vllm_omni.worker.gpu_model_runner import OmniGPUModelRunner

    events: list[object] = []

    class _Model(torch.nn.Module):
        def omni_post_load(self) -> None:
            events.append(("post_load", self.training))

    model = _Model()
    assert model.training

    def load_model(runner, *args, **kwargs) -> None:
        events.append("load")
        runner.model = model.eval()  # vLLM's loader returns ``model.eval()``

    monkeypatch.setattr(GPUModelRunner, "load_model", load_model)
    monkeypatch.setattr(GPUModelRunner, "profile_run", lambda runner: events.append("profile"))
    for name in (
        "_snapshot_prefix_cache_model_policy",
        "_maybe_enable_output_token_ids_for_model_sampler",
        "_init_talker_mtp",
        "_prewarm_attention_capture_workspaces",
        "_report_model_local_kv",
        "_warn_unexposed_stage_hooks",
    ):
        monkeypatch.setattr(OmniGPUModelRunner, name, lambda *args, **kwargs: None)
    runner = object.__new__(OmniGPUModelRunner)
    runner.load_model()
    assert events == ["load"]
    runner.profile_run()
    runner.profile_run()
    assert events == ["load", ("post_load", False), "profile", "profile"]


@_cpu
def test_generation_runner_profile_run_runs_post_load_first(monkeypatch: pytest.MonkeyPatch) -> None:
    # GPUGenerationModelRunner replaces profile_run without calling up.
    from vllm_omni.worker.gpu_generation_model_runner import GPUGenerationModelRunner

    events: list[str] = []
    runner = object.__new__(GPUGenerationModelRunner)
    runner._omni_post_load = lambda: events.append("post_load")
    runner.supports_mm_inputs = False

    def dummy_run(*args, **kwargs):
        events.append("dummy")
        raise RuntimeError("stop after the first dummy run")

    monkeypatch.setattr(GPUGenerationModelRunner, "_dummy_run", dummy_run, raising=False)
    runner.max_num_tokens = 1
    with pytest.raises(RuntimeError, match="stop after the first dummy run"):
        runner.profile_run()
    assert events[:2] == ["post_load", "dummy"]


# --- CUDA: the post-load capture of a real (tiny) encoder vs. the eager batched encoder ----------

N_MELS = 16
POOL = 5
FRAMES = 24  # steady unit: unit_length 10 (see test_streaming_audio_graph.py)


class _GraphThinker:
    """The audio half of the thinker, built like the model: in training mode until the post-load hook."""

    supports_streaming_audio_batch = MiniCPMO45OmniLLMForConditionalGeneration.supports_streaming_audio_batch
    build_streaming_audio_graph_encoder = MiniCPMO45OmniLLMForConditionalGeneration.build_streaming_audio_graph_encoder
    get_audio_embedding_streaming_batch = MiniCPMO45OmniLLMForConditionalGeneration.get_audio_embedding_streaming_batch

    def __init__(self) -> None:
        from transformers.models.whisper.modeling_whisper import WhisperConfig

        torch.manual_seed(0)
        config = WhisperConfig(
            num_mel_bins=N_MELS,
            d_model=32,
            encoder_layers=2,
            encoder_attention_heads=2,
            encoder_ffn_dim=128,
            max_source_positions=40,
            dropout=0.0,
            attention_dropout=0.0,
        )
        config._attn_implementation = "sdpa"
        self.apm = MiniCPMWhisperEncoder(config).to(device="cuda", dtype=torch.bfloat16)
        self.apm.train()
        self.audio_projection_layer = MultiModalProjector(in_dim=config.encoder_ffn_dim // 4, out_dim=24).to(
            device="cuda", dtype=torch.bfloat16
        )
        self.audio_avg_pooler = torch.nn.AvgPool1d(POOL, stride=POOL)
        self.audio_encoder_layer = -1
        self.config = SimpleNamespace(
            audio_pool_step=POOL,
            duplex_audio_kv_page_positions=16,
            duplex_audio_encoder_cuda_graph_batch_sizes=[1, 2, 4],
            duplex_audio_encoder_cuda_graph_cache_buckets=[16, 32],
        )


def _mel(seed: int) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    return torch.randn((1, N_MELS, FRAMES), generator=generator).to("cuda")


def _clone_cache(cache: StreamingAudioKVCache) -> StreamingAudioKVCache:
    clone = StreamingAudioKVCache(
        num_layers=cache.num_layers,
        embed_dim=cache.embed_dim,
        num_heads=cache.num_heads,
        max_positions=cache.max_positions,
        page_positions=cache.page_positions,
    )
    history = cache.history(cache.length)
    clone.reserve(cache.length, dtype=history.dtype, device=history.device)
    clone.commit(0, history)
    clone.length = cache.length
    return clone


@_cuda
def test_post_load_capture_matches_eager_batched() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA graph capture needs a GPU")
    thinker = _GraphThinker()
    # The stage's chunk sizes make a steady unit FRAMES mel frames (10 ms hop).
    stage_model = SimpleNamespace(
        config=SimpleNamespace(),
        processor=_Processor(),
        thinker=thinker,
        chunk_ms=FRAMES * 10,
        first_chunk_ms=FRAMES * 10,
    )
    runtime = MiniCPMO45Stage0DuplexRuntime(stage_model, device="cuda")
    assert thinker.apm.training
    assert getattr(thinker, "_duplex_audio_cuda_graph_encoder", None) is None
    model = SimpleNamespace(
        _minicpmo45_duplex_data_plane_helper=runtime,
        thinker=thinker.apm,
        _module_device=MiniCPMO45OmniForConditionalGeneration._module_device,
    )
    MiniCPMO45OmniForConditionalGeneration.omni_post_load(model)
    graph = thinker._duplex_audio_cuda_graph_encoder
    assert graph is not None and not thinker.apm.training
    assert graph.unit_frames == FRAMES
    assert sorted(graph._graphs) == [(1, 16), (1, 32), (2, 16), (2, 32), (4, 16), (4, 32)]
    replays: list[int] = []
    replay = graph._replay

    def counting_replay(group, *args):
        replays.append(len(group))
        return replay(group, *args)

    graph._replay = counting_replay

    for batch in (1, 3, 4):
        with torch.inference_mode():
            first = [
                StreamingAudioChunk(
                    features=_mel(10 * batch + s), cache=None, prefix_extra_frames=0, suffix_extra_frames=2
                )
                for s in range(batch)
            ]
            _, eager_caches = encode_streaming_audio_batch(
                thinker.apm,
                thinker.audio_projection_layer,
                thinker.audio_avg_pooler,
                first,
                pool_step=POOL,
                page_positions=16,
            )
        graph_caches = [_clone_cache(cache) for cache in eager_caches]
        replays.clear()
        # Two steady rounds: past 11 (bucket 16) then 21 (bucket 32).
        for round_index in range(2):
            seeds = [1000 * (round_index + 1) + 10 * batch + s for s in range(batch)]
            with torch.inference_mode():
                eager_outputs, eager_caches = encode_streaming_audio_batch(
                    thinker.apm,
                    thinker.audio_projection_layer,
                    thinker.audio_avg_pooler,
                    [
                        StreamingAudioChunk(
                            features=_mel(seed), cache=cache, prefix_extra_frames=2, suffix_extra_frames=2
                        )
                        for seed, cache in zip(seeds, eager_caches, strict=True)
                    ],
                    pool_step=POOL,
                    page_positions=16,
                )
                graph_outputs, graph_caches = thinker.get_audio_embedding_streaming_batch(
                    [
                        StreamingAudioChunk(
                            features=_mel(seed), cache=cache, prefix_extra_frames=2, suffix_extra_frames=2
                        )
                        for seed, cache in zip(seeds, graph_caches, strict=True)
                    ]
                )
            for eager_out, graph_out in zip(eager_outputs, graph_outputs, strict=True):
                assert eager_out is not None and graph_out is not None
                # The tolerance tier of test_streaming_audio_graph.py's padded-bucket cases.
                torch.testing.assert_close(graph_out.float(), eager_out.float(), rtol=2e-2, atol=2e-2)
            for eager_cache, graph_cache in zip(eager_caches, graph_caches, strict=True):
                assert graph_cache.length == eager_cache.length
                torch.testing.assert_close(
                    graph_cache.history(graph_cache.length).float(),
                    eager_cache.history(eager_cache.length).float(),
                    rtol=2e-2,
                    atol=2e-2,
                )
        assert replays == [batch, batch]
