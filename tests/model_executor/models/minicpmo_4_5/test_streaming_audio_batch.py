# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""The cross-session batched streaming audio encoder against the per-session path."""

from types import SimpleNamespace
from typing import Any

import pytest
import torch
from torch import nn
from transformers.models.whisper.modeling_whisper import WhisperConfig

from vllm_omni.model_executor.models.minicpmo_4_5.minicpmo_4_5_omni import MiniCPMO45OmniForConditionalGeneration
from vllm_omni.model_executor.models.minicpmo_4_5.minicpmo_4_5_omni_llm import (
    MiniCPMO45OmniLLMForConditionalGeneration,
    MiniCPMWhisperEncoder,
    MultiModalProjector,
)
from vllm_omni.model_executor.models.minicpmo_4_5.streaming_audio_encoder import StreamingAudioChunk

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

N_MELS, POOL = 16, 5
# Per round, each session's unit frames (None: sits the round out): 22 / 24 frames mirror 1020 / 1040 ms
# at 1/10 scale, odd lengths pad conv2, 25 frames leave the pooler a remainder, and 40 positions force resets.
_SCHEDULE: list[list[int | None]] = [
    [22, 22, 23, 21],
    [24, None, 24, 25],
    [24, 24, None, 24],
    [24, 23, 24, 24],
    [24, 24, 24, None],
    [25, 24, 23, 24],
]


def _thinker() -> Any:
    """The audio half of the thinker: real modules and methods, toy sizes."""
    torch.manual_seed(0)
    config = WhisperConfig(num_mel_bins=N_MELS, d_model=32, encoder_layers=2, encoder_attention_heads=2)
    config.update(dict(encoder_ffn_dim=128, max_source_positions=40, dropout=0.0, attention_dropout=0.0))
    config._attn_implementation = "sdpa"
    thinker = MiniCPMO45OmniLLMForConditionalGeneration.__new__(MiniCPMO45OmniLLMForConditionalGeneration)
    nn.Module.__init__(thinker)
    thinker.apm = MiniCPMWhisperEncoder(config).eval()
    thinker.audio_projection_layer = MultiModalProjector(in_dim=32, out_dim=24)  # takes ffn // 4, as the checkpoint
    thinker.audio_avg_pooler = nn.AvgPool1d(POOL, stride=POOL)
    thinker.audio_encoder_layer, thinker.audio_past_key_values = -1, None
    thinker.config = SimpleNamespace(audio_pool_step=POOL, duplex_audio_kv_page_positions=16)
    return thinker


def test_batched_rounds_match_sequential_sessions() -> None:
    thinker = _thinker()
    legacy: list[Any] = [None] * 4
    batched: list[Any] = [None] * 4
    resets = 0
    for round_index, frames_per_session in enumerate(_SCHEDULE):
        chunks, active, expected = [], [], []
        for session, frames in enumerate(frames_per_session):
            if frames is None:
                continue
            mel = torch.randn((1, N_MELS, frames), generator=torch.Generator().manual_seed(100 * round_index + session))
            extra = dict(prefix_extra_frames=0 if legacy[session] is None else 2, suffix_extra_frames=2)
            features = {"audio_features": mel, "audio_feature_lens": [torch.tensor([frames])]}
            thinker.audio_past_key_values = legacy[session]
            with torch.no_grad():
                nested = thinker.get_audio_embedding_streaming(features, use_extra_context=True, **extra)
            legacy[session] = thinker.audio_past_key_values
            expected.append(torch.cat([t for row in nested for t in row]))
            chunks.append(StreamingAudioChunk(mel, batched[session], **extra))
            active.append(session)
        idle = {s: (cache, cache.length if cache else 0) for s, cache in enumerate(batched) if s not in active}
        outputs, caches = thinker.get_audio_embedding_streaming_batch(chunks)
        for session, output, cache, reference in zip(active, outputs, caches, expected, strict=True):
            torch.testing.assert_close(output, reference, rtol=1e-5, atol=1e-5)
            resets += batched[session] is not None and cache is not batched[session]
            batched[session] = cache
            want = legacy[session].self_attention_cache
            assert cache.length == want.get_seq_length()
            for got, ref in zip(cache.to_legacy_cache().self_attention_cache.layers, want.layers, strict=True):
                torch.testing.assert_close((got.keys, got.values), (ref.keys, ref.values))
        # Not in the batch: the same cache object, the same committed length.
        assert all(batched[s] is c and (c.length if c else 0) == n for s, (c, n) in idle.items())
    assert resets >= 2, "the schedule must cross the max_source_positions reset"


def test_graph_build_honors_eager_without_engine_config() -> None:
    # The thinker keeps no vllm_config: the gate reads the flag __init__ stored.
    thinker = _thinker()
    thinker._enforce_eager = True
    assert thinker.build_streaming_audio_graph_encoder(unit_frames=24) is False
    assert thinker._duplex_audio_cuda_graph_encoder is None


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_graph_build_captures_on_cuda() -> None:
    thinker = _thinker().cuda()
    thinker._enforce_eager = False
    thinker.config.duplex_audio_encoder_cuda_graph_batch_sizes = [1, 2]
    thinker.config.duplex_audio_encoder_cuda_graph_cache_buckets = [8]
    assert thinker.build_streaming_audio_graph_encoder(unit_frames=24) is True
    assert thinker._duplex_audio_cuda_graph_encoder is not None


@pytest.mark.parametrize(
    ("session_mode", "device", "captures"),
    [("duplex", "cuda", 1), ("turn", "cuda", 0), ("duplex", "cpu", 0)],
)
def test_duplex_audio_graphs_capture_while_the_model_loads(monkeypatch, session_mode, device, captures) -> None:
    # Captured in load_weights, the graphs count against the KV budget instead of past it.
    model = MiniCPMO45OmniForConditionalGeneration.__new__(MiniCPMO45OmniForConditionalGeneration)
    nn.Module.__init__(model)
    model.vllm_config = SimpleNamespace(model_config=SimpleNamespace(session_mode=session_mode))
    model.thinker = nn.Linear(1, 1)
    calls: list[None] = []
    model._minicpmo45_duplex_data_plane_helper = SimpleNamespace(build_audio_cuda_graph=lambda: calls.append(None))
    monkeypatch.setattr(
        MiniCPMO45OmniForConditionalGeneration, "_module_device", staticmethod(lambda module: torch.device(device))
    )
    model._capture_duplex_audio_graphs()
    assert len(calls) == captures
