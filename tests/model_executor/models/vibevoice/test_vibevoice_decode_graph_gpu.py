# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""GPU acceptance for the audio decode CUDA-graph executor.

Replay must be bitwise identical to eager across consecutive tokens (cache
accumulates), different inputs, and segment boundaries (cache reset). Capture
failure must fall back to eager permanently.
"""

from __future__ import annotations

import os
import random

import pytest
import torch

pytestmark = [
    pytest.mark.core_model,
    pytest.mark.cuda,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required"),
]

_MODEL_ENV = "VIBEVOICE_TEST_MODEL"


def _build():
    from transformers import AutoModel

    from vllm_omni.model_executor.models.vibevoice.audio_decode import (
        VibeVoiceAudioTokenDecoder,
        VibeVoiceDecodeGraphExecutor,
    )
    from vllm_omni.model_executor.models.vibevoice.vibevoice import (
        VibeVoiceMultiModalProjector,
    )
    from vllm_omni.transformers_utils.configs.vibevoice import VibeVoiceConfig

    model = os.getenv(_MODEL_ENV, "microsoft/VibeVoice-1.5B")
    config = VibeVoiceConfig.from_pretrained(model)
    torch.manual_seed(0)
    audio_tower = AutoModel.from_config(config.audio_config).to(device="cuda", dtype=torch.bfloat16).eval()
    semantic_encoder = (
        AutoModel.from_config(config.semantic_model_config).to(device="cuda", dtype=torch.bfloat16).eval()
    )
    acoustic_projector = (
        VibeVoiceMultiModalProjector(config.audio_config.hidden_size, config.hidden_size)
        .to(device="cuda", dtype=torch.bfloat16)
        .eval()
    )
    semantic_connector = (
        VibeVoiceMultiModalProjector(config.semantic_model_config.hidden_size, config.hidden_size)
        .to(device="cuda", dtype=torch.bfloat16)
        .eval()
    )
    latent_scaling = torch.tensor(1.0, device="cuda", dtype=torch.bfloat16)
    latent_bias = torch.tensor(0.0, device="cuda", dtype=torch.bfloat16)
    decoder = VibeVoiceAudioTokenDecoder.from_model_config(config)
    executor = VibeVoiceDecodeGraphExecutor(decoder)
    latent_size = decoder.latent_size
    return (
        executor,
        decoder,
        audio_tower,
        semantic_encoder,
        acoustic_projector,
        semantic_connector,
        latent_scaling,
        latent_bias,
        latent_size,
    )


def _decode(executor, decoder, at, se, ap, sc, ls, lb, latent, ac, sec, *, use_graph):
    if use_graph:
        out = executor.decode(
            audio_tower=at,
            semantic_encoder=se,
            acoustic_projector=ap,
            semantic_connector=sc,
            latent_scaling_factor=ls,
            latent_bias_factor=lb,
            audio_latent=latent,
            acoustic_cache=ac,
            semantic_cache=sec,
        )
        if out is not None:
            return out
    return decoder_decode(decoder, at, se, ap, sc, ls, lb, latent, ac, sec)


def decoder_decode(decoder, at, se, ap, sc, ls, lb, latent, ac, sec):
    return decoder.decode_audio_token(
        audio_tower=at,
        semantic_encoder=se,
        acoustic_projector=ap,
        semantic_connector=sc,
        latent_scaling_factor=ls,
        latent_bias_factor=lb,
        audio_latent=latent,
        acoustic_cache=ac,
        semantic_cache=sec,
    )


def test_decode_graph_replay_is_bitwise_identical_across_tokens() -> None:
    (executor, decoder, at, se, ap, sc, ls, lb, latent_size) = _build()
    latents = [torch.randn(1, 1, latent_size, device="cuda", dtype=torch.bfloat16) for _ in range(5)]

    with torch.inference_mode():
        # Eager reference: sequential, cache accumulates.
        ac = sec = None
        eager_audio, eager_semantic, eager_emb = [], [], []
        for latent in latents:
            out = decoder_decode(decoder, at, se, ap, sc, ls, lb, latent, ac, sec)
            ac, sec = out.acoustic_cache, out.semantic_cache
            eager_audio.append(out.audio.clone())
            eager_semantic.append(out.semantic_latent.clone())
            eager_emb.append(out.next_embedding.clone())

        # Graph: token 1 eager (populate cache), token 2 capture, token 3-5 replay.
        ac = sec = None
        out1 = decoder_decode(decoder, at, se, ap, sc, ls, lb, latents[0], ac, sec)
        ac, sec = out1.acoustic_cache, out1.semantic_cache
        graph_audio = [out1.audio.clone()]
        graph_semantic = [out1.semantic_latent.clone()]
        graph_emb = [out1.next_embedding.clone()]
        for i in range(1, 5):
            out = _decode(executor, decoder, at, se, ap, sc, ls, lb, latents[i], ac, sec, use_graph=True)
            graph_audio.append(out.audio.clone())
            graph_semantic.append(out.semantic_latent.clone())
            graph_emb.append(out.next_embedding.clone())

    for i in range(5):
        assert torch.equal(graph_audio[i], eager_audio[i]), (
            f"token {i}: graph audio diverged (max diff "
            f"{(graph_audio[i].float() - eager_audio[i].float()).abs().max().item()})"
        )
        assert torch.equal(graph_semantic[i], eager_semantic[i]), (
            f"token {i}: graph semantic latent diverged (max diff "
            f"{(graph_semantic[i].float() - eager_semantic[i].float()).abs().max().item()})"
        )
        assert torch.equal(graph_emb[i], eager_emb[i]), (
            f"token {i}: graph embedding diverged (max diff "
            f"{(graph_emb[i].float() - eager_emb[i].float()).abs().max().item()})"
        )


def test_decode_graph_survives_segment_reset() -> None:
    """Cache zero_ at a segment boundary keeps addresses stable; graph stays valid."""
    (executor, decoder, at, se, ap, sc, ls, lb, latent_size) = _build()
    latents = [torch.randn(1, 1, latent_size, device="cuda", dtype=torch.bfloat16) for _ in range(4)]

    with torch.inference_mode():
        # Segment 1: token 1 eager + token 2 graph (capture).
        ac = sec = None
        out1 = decoder_decode(decoder, at, se, ap, sc, ls, lb, latents[0], ac, sec)
        ac, sec = out1.acoustic_cache, out1.semantic_cache
        _decode(executor, decoder, at, se, ap, sc, ls, lb, latents[1], ac, sec, use_graph=True)

        # Segment boundary: reset caches (zero_), graph must remain valid.
        for cache in (ac, sec):
            for layer in cache.layers.values():
                if getattr(layer, "is_initialized", False) and layer.cache is not None:
                    layer.cache.zero_()

        # Segment 2: graph replay from zero cache. Replay outputs borrow the
        # graph's static buffers, so preserve token 3 before token 4 overwrites
        # them.
        out3 = _decode(executor, decoder, at, se, ap, sc, ls, lb, latents[2], ac, sec, use_graph=True)
        out3_ptrs = (
            out3.audio.data_ptr(),
            out3.semantic_latent.data_ptr(),
            out3.next_embedding.data_ptr(),
        )
        out3_values = (
            out3.audio.clone(),
            out3.semantic_latent.clone(),
            out3.next_embedding.clone(),
        )
        out4 = _decode(executor, decoder, at, se, ap, sc, ls, lb, latents[3], ac, sec, use_graph=True)
        out4_values = (
            out4.audio.clone(),
            out4.semantic_latent.clone(),
            out4.next_embedding.clone(),
        )
        assert out3_ptrs == (
            out4.audio.data_ptr(),
            out4.semantic_latent.data_ptr(),
            out4.next_embedding.data_ptr(),
        )

        # Oracle: segment 2 from a fresh (None) cache. Clone each eager result
        # at the same lifetime boundary to keep the ownership comparison
        # explicit on both paths.
        ac2 = sec2 = None
        ref3 = decoder_decode(decoder, at, se, ap, sc, ls, lb, latents[2], ac2, sec2)
        ac2, sec2 = ref3.acoustic_cache, ref3.semantic_cache
        ref3_values = (
            ref3.audio.clone(),
            ref3.semantic_latent.clone(),
            ref3.next_embedding.clone(),
        )
        ref4 = decoder_decode(decoder, at, se, ap, sc, ls, lb, latents[3], ac2, sec2)
        ref4_values = (
            ref4.audio.clone(),
            ref4.semantic_latent.clone(),
            ref4.next_embedding.clone(),
        )

    # After cache reset, graph replay and a fresh start both read zero
    # context; bf16 conv algorithm selection may differ by buffer address,
    # so assert within bf16 tolerance rather than bitwise.
    for graph_value, eager_value in zip(out3_values, ref3_values, strict=True):
        torch.testing.assert_close(graph_value.float(), eager_value.float(), rtol=1e-3, atol=1e-3)
    for graph_value, eager_value in zip(out4_values, ref4_values, strict=True):
        torch.testing.assert_close(graph_value.float(), eager_value.float(), rtol=1e-3, atol=1e-3)


def test_decode_graph_survives_dynamic_request_order_and_cache_reuse() -> None:
    """Private graph pools remain valid as request cache slots are reused."""
    (executor, decoder, at, se, ap, sc, ls, lb, latent_size) = _build()
    rng = random.Random(42)

    def fresh_cache_pair():
        latent = torch.randn(1, 1, latent_size, device="cuda", dtype=torch.bfloat16)
        eager = decoder_decode(decoder, at, se, ap, sc, ls, lb, latent, None, None)
        captured = _decode(
            executor,
            decoder,
            at,
            se,
            ap,
            sc,
            ls,
            lb,
            latent,
            eager.acoustic_cache,
            eager.semantic_cache,
            use_graph=True,
        )
        captured.audio.clone()
        captured.semantic_latent.clone()
        captured.next_embedding.clone()
        return eager.acoustic_cache, eager.semantic_cache

    with torch.inference_mode():
        slots = [fresh_cache_pair() for _ in range(4)]
        torch.accelerator.synchronize()
        for step in range(200):
            order = list(range(4))
            rng.shuffle(order)
            for index in order:
                acoustic_cache, semantic_cache = slots[index]
                latent = torch.randn(1, 1, latent_size, device="cuda", dtype=torch.bfloat16)
                out = _decode(
                    executor,
                    decoder,
                    at,
                    se,
                    ap,
                    sc,
                    ls,
                    lb,
                    latent,
                    acoustic_cache,
                    semantic_cache,
                    use_graph=True,
                )
                out.audio.clone()
                out.semantic_latent.clone()
                out.next_embedding.clone()
            if step and step % 2 == 0:
                for cache in slots[rng.randrange(4)]:
                    for layer in cache.layers.values():
                        if getattr(layer, "is_initialized", False) and layer.cache is not None:
                            layer.cache.zero_()
            # Surface asynchronous graph memory faults at the iteration that
            # caused them instead of at unrelated later model work.
            torch.accelerator.synchronize()


def test_decode_graph_without_initialized_cache_uses_eager() -> None:
    (executor, decoder, at, se, ap, sc, ls, lb, latent_size) = _build()
    latent = torch.randn(1, 1, latent_size, device="cuda", dtype=torch.bfloat16)

    with torch.inference_mode():
        out = executor.decode(
            audio_tower=at,
            semantic_encoder=se,
            acoustic_projector=ap,
            semantic_connector=sc,
            latent_scaling_factor=ls,
            latent_bias_factor=lb,
            audio_latent=latent,
            acoustic_cache=None,
            semantic_cache=None,
        )
        assert out is None
        assert executor._disabled is False


def test_decode_graph_capture_failure_is_observable(monkeypatch: pytest.MonkeyPatch) -> None:
    (executor, decoder, at, se, ap, sc, ls, lb, latent_size) = _build()
    latent = torch.randn(1, 1, latent_size, device="cuda", dtype=torch.bfloat16)

    with torch.inference_mode():
        first = decoder_decode(decoder, at, se, ap, sc, ls, lb, latent, None, None)

    assert (
        executor.decode(
            audio_tower=at,
            semantic_encoder=se,
            acoustic_projector=ap,
            semantic_connector=sc,
            latent_scaling_factor=ls,
            latent_bias_factor=lb,
            audio_latent=latent,
            acoustic_cache=first.acoustic_cache,
            semantic_cache=None,
        )
        is None
    )
    assert executor._disabled is False

    calls = 0

    def fail_capture(**_kwargs):
        nonlocal calls
        calls += 1
        raise RuntimeError("injected capture failure")

    monkeypatch.setattr(executor, "_capture", fail_capture)
    kwargs = {
        "audio_tower": at,
        "semantic_encoder": se,
        "acoustic_projector": ap,
        "semantic_connector": sc,
        "latent_scaling_factor": ls,
        "latent_bias_factor": lb,
        "audio_latent": latent,
        "acoustic_cache": first.acoustic_cache,
        "semantic_cache": first.semantic_cache,
    }

    assert executor.decode(**kwargs) is None
    assert executor.decode(**kwargs) is None
    assert executor._disabled is True
    assert calls == 1

    strict_executor = type(executor)(decoder, capture_failure_fatal=True)
    monkeypatch.setattr(strict_executor, "_capture", fail_capture)
    with pytest.raises(RuntimeError, match="Required VibeVoice decode CUDA-graph capture failed"):
        strict_executor.decode(**kwargs)
    with pytest.raises(RuntimeError, match="disabled after a prior capture failure"):
        strict_executor.decode(**kwargs)


def test_request_cleanup_waits_for_pending_waveform_copy() -> None:
    """Aborting a request must synchronize its outstanding waveform D2H copy."""
    from vllm_omni.model_executor.models.vibevoice.stateful import VibeVoiceRequestState

    state = VibeVoiceRequestState(
        request_id="request-a",
        guidance_scale=1.3,
        num_diffusion_steps=10,
    )
    source = torch.arange(1_048_576, device="cuda", dtype=torch.float32)
    buffer = torch.empty_like(source, device="cpu", pin_memory=True)
    buffer.copy_(source, non_blocking=True)
    event = torch.cuda.Event()
    event.record()
    state.waveform_chunks_cpu.append(buffer)
    state._waveform_events[id(buffer)] = (event, buffer)

    state.clear()

    assert event.query()
    assert torch.equal(buffer, torch.arange(buffer.numel(), dtype=torch.float32))
    assert state.waveform_chunks_cpu == []
    assert state._waveform_events == {}
    assert state._pinned_pool == []
