# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""The Talker decode graph must reset its resident cache between requests."""

from threading import Event, Thread

import pytest
import torch
from transformers import Qwen2Config, Qwen2Model, StaticCache

from tests.helpers.mark import hardware_test
from vllm_omni.model_executor.models.ming_flash_omni.talker_llm_graph import Qwen2DecodeGraph

pytestmark = pytest.mark.core_model


def _model(device="cpu"):
    config = Qwen2Config(
        vocab_size=32,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=64,
    )
    config._attn_implementation = "sdpa"
    return Qwen2Model(config).eval().to(device)


@pytest.mark.cpu
@torch.inference_mode()
def test_cpu_falls_back_without_allocating_resident_cache():
    model = _model()
    graph = Qwen2DecodeGraph(model, max_cache_len=32)
    prefix = torch.randn(1, 3, 32)
    with graph.request(prefix):
        assert graph.cache is None
        assert graph.decode(prefix[:, :1], position=3) is None
        actual = model(inputs_embeds=prefix).last_hidden_state
    assert torch.isfinite(actual).all()
    assert graph.cache is None


@hardware_test(res={"cuda": "H100"})
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA graph requires CUDA")
@torch.inference_mode()
def test_decode_matches_eager_across_prefixes_and_request_failure():
    model = _model("cuda")
    graph = Qwen2DecodeGraph(model, max_cache_len=32)
    cache_addresses = None
    for length in (2, 7, 13, 3):
        prefix = torch.randn(1, length, 32, device="cuda")
        eager = StaticCache(config=model.config, max_cache_len=32)
        with graph.request(prefix):
            assert graph.cache is not None
            addresses = [(layer.keys.data_ptr(), layer.values.data_ptr()) for layer in graph.cache.layers]
            if cache_addresses is not None:
                assert addresses == cache_addresses
            cache_addresses = addresses
            model(inputs_embeds=prefix, past_key_values=eager, use_cache=True)
            model(inputs_embeds=prefix, past_key_values=graph.cache, use_cache=True)
            previous = None
            saved = None
            for offset in range(4):
                embed = torch.randn(1, 1, 32, device="cuda")
                expected = model(inputs_embeds=embed, past_key_values=eager, use_cache=True).last_hidden_state
                actual = graph.decode(embed, position=length + offset)
                assert actual is not None
                torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)
                # A replay must not mutate a previously returned result.
                if previous is not None:
                    torch.testing.assert_close(previous, saved)
                previous, saved = actual, actual.clone()
        assert int(graph.cache.get_seq_length()) == 0
    with pytest.raises(RuntimeError, match="cancelled"):
        with graph.request(prefix):
            model(inputs_embeds=prefix, past_key_values=graph.cache, use_cache=True)
            raise RuntimeError("cancelled")
    with graph.request(prefix):
        assert int(graph.cache.get_seq_length()) == 0
        assert graph.decode(torch.randn(2, 1, 32, device="cuda"), position=1) is None
        assert not graph.active
    with graph.request(prefix):
        assert graph.active
        assert graph.decode(torch.randn(1, 1, 32, device="cuda"), position=32) is None


@pytest.mark.cpu
def test_request_lock_covers_whole_sequence_and_releases_after_error():
    graph = Qwen2DecodeGraph(_model(), max_cache_len=32)
    prefix = torch.randn(1, 2, 32)
    trying, entered = Event(), Event()

    def concurrent_request():
        trying.set()
        with graph.request(prefix):
            entered.set()

    worker = Thread(target=concurrent_request)
    try:
        with pytest.raises(RuntimeError, match="cancelled"):
            with graph.request(prefix):
                worker.start()
                assert trying.wait(timeout=2)
                assert not entered.wait(timeout=0.05)
                raise RuntimeError("cancelled")
        assert entered.wait(timeout=2)
    finally:
        worker.join(timeout=2)
    assert not worker.is_alive()


@pytest.mark.cpu
@torch.inference_mode()
def test_replay_error_disables_graph_instead_of_retrying_partial_kv():
    graph = Qwen2DecodeGraph(_model(), max_cache_len=32)
    graph.cache = StaticCache(config=graph.model.config, max_cache_len=32)
    graph.active = True
    graph._embed = torch.zeros(1, 1, 32)
    graph._position = torch.zeros(1, 1, dtype=torch.long)

    class FailedReplay:
        def replay(self):
            raise RuntimeError("partial replay")

    graph._graph = FailedReplay()
    embed = graph._embed
    with pytest.raises(RuntimeError, match="partial replay"):
        graph.decode(embed, position=2)
    assert graph.cache is None
    assert not graph.active
    assert graph.decode(embed, position=2) is None
    with graph.request(torch.zeros(1, 3, 32)):
        assert graph.cache is None


@hardware_test(res={"cuda": "H100"})
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA graph requires CUDA")
def test_capture_failure_is_disabled_for_later_requests(monkeypatch):
    graph = Qwen2DecodeGraph(_model("cuda"), max_cache_len=32)
    prefix = torch.zeros(1, 3, 32, device="cuda")
    attempts = []

    def fail_capture(inputs_embeds):
        attempts.append(inputs_embeds.shape)
        raise RuntimeError("unsupported capture")

    monkeypatch.setattr(graph, "_capture", fail_capture)
    for _ in range(2):
        with graph.request(prefix):
            assert graph.cache is None
            assert not graph.active
    assert len(attempts) == 1


@pytest.mark.cpu
@torch.inference_mode()
def test_generator_cpu_fallback_and_enforce_eager_match(monkeypatch):
    pytest.importorskip("vllm")
    from types import SimpleNamespace

    from vllm_omni.model_executor.models.ming_flash_omni.talker_module import MingAudioGenerator

    model = _model()
    prefix = torch.randn(1, 3, 32)
    results = []
    for use_graphs in (False, True):
        generator = MingAudioGenerator(
            config=SimpleNamespace(max_cache_len=16),
            llm_config=model.config,
            model=model,
            cfm=None,
            aggregator=None,
            stop_head=None,
            audio_vae=None,
            patch_size=1,
            his_patch_size=1,
            latent_dim=4,
            cfg_strength=2.0,
            use_cuda_graphs=use_graphs,
            enable_llm_cuda_graph=True,
        )

        def sample(hidden, history, **kwargs):
            return hidden[..., :4].clone(), hidden.clone(), torch.tensor([[1.0, 0.0]])

        monkeypatch.setattr(generator, "cfm_sample_step", sample)
        results.append(generator.generate_latents(prefix, max_steps=4))
        if not use_graphs:
            assert generator._llm_graph is None
        else:
            assert generator._llm_graph.cache is None
    assert len(results[0]) == len(results[1]) == 4
    for eager, fallback in zip(*results):
        torch.testing.assert_close(eager, fallback)


@pytest.mark.cpu
def test_request_orders_stream_wait_reset_and_completion(monkeypatch):
    from types import SimpleNamespace

    graph = Qwen2DecodeGraph(_model(), max_cache_len=32)
    calls = []

    class Stream:
        def wait_event(self, event):
            calls.append("wait")

    class Completion:
        ready = False

        def query(self):
            return self.ready

        def record(self, stream):
            assert graph._lock.locked()
            calls.append("record")

    class Cache:
        def reset(self):
            calls.append("reset")

    monkeypatch.setattr(graph.model, "parameters", lambda: iter([SimpleNamespace(device=torch.device("cuda"))]))
    monkeypatch.setattr(torch.cuda, "current_stream", lambda device: Stream())
    graph._completion = Completion()
    graph.cache = Cache()
    with graph.request(torch.zeros(1, 3, 32)):
        calls.append("body")
    assert calls == ["wait", "reset", "body", "reset", "record"]
    graph._disable()
    with graph.request(torch.zeros(1, 3, 32)):
        assert graph._retired_resources is not None
    graph._completion.ready = True
    with graph.request(torch.zeros(1, 3, 32)):
        assert graph._retired_resources is None


@hardware_test(res={"cuda": "H100"})
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA graph requires CUDA")
@torch.inference_mode()
def test_new_stream_waits_for_previous_reset_including_cancel(monkeypatch):
    model = _model("cuda")
    graph = Qwen2DecodeGraph(model, max_cache_len=32)
    cases = []
    for length in (2, 7, 13, 3):
        prefix = torch.randn(1, length, 32, device="cuda")
        embed = torch.randn(1, 1, 32, device="cuda")
        cache = StaticCache(config=model.config, max_cache_len=32)
        model(inputs_embeds=prefix, past_key_values=cache, use_cache=True)
        expected = model(inputs_embeds=embed, past_key_values=cache, use_cache=True).last_hidden_state
        cases.append((prefix, embed, expected))
    with graph.request(cases[0][0]):
        pass
    streams = [torch.cuda.Stream(), torch.cuda.Stream()]
    for stream in streams:
        stream.wait_stream(torch.cuda.current_stream())
    reset = graph.cache.reset
    delayed_reset = False

    def reset_with_delay():
        nonlocal delayed_reset
        if delayed_reset:
            delayed_reset = False
            torch.cuda._sleep(100_000_000)
        reset()

    monkeypatch.setattr(graph.cache, "reset", reset_with_delay)
    actual = []
    for index, (prefix, embed, _) in enumerate(cases):
        with torch.cuda.stream(streams[index % 2]):
            try:
                with graph.request(prefix):
                    model(inputs_embeds=prefix, past_key_values=graph.cache, use_cache=True)
                    # Without the completion dependency, the previous stream's
                    # delayed reset erases this prefill before decode executes.
                    if index % 2:
                        torch.cuda._sleep(200_000_000)
                    actual.append(graph.decode(embed, position=prefix.shape[1]))
                    delayed_reset = index % 2 == 0
                    if index == 2:
                        raise RuntimeError("cancelled")
            except RuntimeError as exc:
                assert index == 2 and str(exc) == "cancelled"
    # No host read / assert_close between requests: it would hide the race.
    for stream in streams:
        stream.synchronize()
    for output, (_, _, expected) in zip(actual, cases):
        torch.testing.assert_close(output, expected, atol=1e-5, rtol=1e-5)
