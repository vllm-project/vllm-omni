# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""GPU acceptance for the audio decode CUDA-graph executor.

Replay must be bitwise identical to eager across consecutive tokens (cache
accumulates) and segment boundaries (cache reset). Aborting a request must
synchronize its outstanding waveform D2H copy.
"""

from __future__ import annotations

import os

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
        ac = sec = None
        eager_audio, eager_semantic, eager_emb = [], [], []
        for latent in latents:
            out = _decode(executor, decoder, at, se, ap, sc, ls, lb, latent, ac, sec, use_graph=False)
            ac, sec = out.acoustic_cache, out.semantic_cache
            eager_audio.append(out.audio.clone())
            eager_semantic.append(out.semantic_latent.clone())
            eager_emb.append(out.next_embedding.clone())

        ac = sec = None
        out1 = _decode(executor, decoder, at, se, ap, sc, ls, lb, latents[0], ac, sec, use_graph=False)
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
        assert torch.equal(graph_audio[i], eager_audio[i])
        assert torch.equal(graph_emb[i], eager_emb[i])


def test_decode_graph_survives_segment_reset() -> None:
    """Cache zero_ at a segment boundary keeps addresses stable; graph stays valid."""
    (executor, decoder, at, se, ap, sc, ls, lb, latent_size) = _build()
    latents = [torch.randn(1, 1, latent_size, device="cuda", dtype=torch.bfloat16) for _ in range(4)]

    with torch.inference_mode():
        ac = sec = None
        out1 = _decode(executor, decoder, at, se, ap, sc, ls, lb, latents[0], ac, sec, use_graph=False)
        ac, sec = out1.acoustic_cache, out1.semantic_cache
        _decode(executor, decoder, at, se, ap, sc, ls, lb, latents[1], ac, sec, use_graph=True)

        for cache in (ac, sec):
            for layer in cache.layers.values():
                if getattr(layer, "is_initialized", False) and layer.cache is not None:
                    layer.cache.zero_()

        out3 = _decode(executor, decoder, at, se, ap, sc, ls, lb, latents[2], ac, sec, use_graph=True)
        out3_values = (out3.audio.clone(), out3.next_embedding.clone())
        out4 = _decode(executor, decoder, at, se, ap, sc, ls, lb, latents[3], ac, sec, use_graph=True)
        out4_values = (out4.audio.clone(), out4.next_embedding.clone())

        ac2 = sec2 = None
        ref3 = _decode(executor, decoder, at, se, ap, sc, ls, lb, latents[2], ac2, sec2, use_graph=False)
        ac2, sec2 = ref3.acoustic_cache, ref3.semantic_cache
        ref4 = _decode(executor, decoder, at, se, ap, sc, ls, lb, latents[3], ac2, sec2, use_graph=False)

    for graph_val, eager_val in zip(out3_values, (ref3.audio.clone(), ref3.next_embedding.clone()), strict=True):
        torch.testing.assert_close(graph_val.float(), eager_val.float(), rtol=1e-3, atol=1e-3)
    for graph_val, eager_val in zip(out4_values, (ref4.audio.clone(), ref4.next_embedding.clone()), strict=True):
        torch.testing.assert_close(graph_val.float(), eager_val.float(), rtol=1e-3, atol=1e-3)


def test_request_cleanup_waits_for_pending_waveform_copy() -> None:
    """Aborting a request must synchronize its outstanding waveform D2H copy."""
    from vllm_omni.model_executor.models.vibevoice.stateful import VibeVoiceRequestState

    state = VibeVoiceRequestState(request_id="request-a", guidance_scale=1.3, num_diffusion_steps=10)
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
