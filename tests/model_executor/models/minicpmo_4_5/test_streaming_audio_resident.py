# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Resident mode of the streaming audio encoder graphs (``resident_slots > 0``).

Sessions keep their cache in a slot pool the graphs read and write in place.
The parity tests replay the same session schedule through the copy-in graph
encoder (the default) and the resident one and require identical bits: the
resident graph gathers the history into a temporary laid out exactly like the
shared storage, so attention sees the same values at every unmasked position.
"""

from __future__ import annotations

import gc

import pytest
import torch
from torch import nn

from vllm_omni.model_executor.models.minicpmo_4_5.minicpmo_4_5_omni_llm import (
    MiniCPMWhisperEncoder,
    MultiModalProjector,
)
from vllm_omni.model_executor.models.minicpmo_4_5.streaming_audio_encoder import (
    StreamingAudioChunk,
    StreamingAudioKVCache,
    encode_streaming_audio_batch,
)
from vllm_omni.model_executor.models.minicpmo_4_5.streaming_audio_encoder_graph import (
    PooledStreamingAudioKVCache,
    StreamingAudioGraphEncoder,
    StreamingAudioKVSlotPool,
)

pytestmark = [pytest.mark.core_model]

N_MELS = 16
POOL = 5
FRAMES = 24  # steady unit_length 10; a first unit (prefix 0) is 11
MAX_POSITIONS = 40


class _Audio:
    def __init__(self, implementation: str, device: str = "cuda") -> None:
        from transformers.models.whisper.modeling_whisper import WhisperConfig

        torch.manual_seed(0)
        config = WhisperConfig(
            num_mel_bins=N_MELS,
            d_model=32,
            encoder_layers=2,
            encoder_attention_heads=2,
            encoder_ffn_dim=128,
            max_source_positions=MAX_POSITIONS,
            dropout=0.0,
            attention_dropout=0.0,
        )
        config._attn_implementation = implementation
        self.apm = MiniCPMWhisperEncoder(config).eval().to(device=device, dtype=torch.bfloat16)
        self.proj = MultiModalProjector(in_dim=config.encoder_ffn_dim // 4, out_dim=24).to(
            device=device, dtype=torch.bfloat16
        )
        self.pooler = nn.AvgPool1d(POOL, stride=POOL)

    def graph(self, *, resident_slots: int, batch_sizes, cache_buckets) -> StreamingAudioGraphEncoder:
        graph = StreamingAudioGraphEncoder(
            self.apm,
            self.proj,
            self.pooler,
            unit_frames=FRAMES,
            pool_step=POOL,
            batch_sizes=batch_sizes,
            cache_buckets=cache_buckets,
            page_positions=16,
            resident_slots=resident_slots,
        )
        with torch.inference_mode():
            graph.capture()
        return graph


def _mel(frames: int, seed: int) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    return torch.randn((1, N_MELS, frames), generator=generator)


def _chunk(cache, *, seed: int, frames: int = FRAMES) -> StreamingAudioChunk:
    first = cache is None
    return StreamingAudioChunk(
        features=_mel(frames, seed),
        cache=cache,
        prefix_extra_frames=0 if first else 2,
        suffix_extra_frames=2,
    )


def _history(cache: StreamingAudioKVCache) -> torch.Tensor:
    return cache.history(cache.length)


# Rounds of (session, frames): sessions drop in and out (batch padding), a
# session starts late (first unit in a later round), one round carries a
# non-steady unit (frames + 2: eager; round 7's, at past 30, resets eagerly),
# and enough rounds pass for every session to cross the cache buckets, go past
# the largest one (eager in place) and reset (past back to 0) in the graph.
_SCHEDULE: list[list[tuple[int, int]]] = [
    [(0, FRAMES), (1, FRAMES), (2, FRAMES)],
    [(0, FRAMES), (2, FRAMES)],
    [(0, FRAMES), (1, FRAMES), (2, FRAMES), (3, FRAMES)],
    [(0, FRAMES), (1, FRAMES + 2), (2, FRAMES), (3, FRAMES)],
    [(0, FRAMES), (1, FRAMES), (3, FRAMES)],
    [(0, FRAMES), (1, FRAMES), (2, FRAMES), (3, FRAMES)],
    [(1, FRAMES + 2), (2, FRAMES), (3, FRAMES)],
    [(0, FRAMES), (1, FRAMES), (2, FRAMES + 2), (3, FRAMES)],
    [(0, FRAMES), (1, FRAMES), (2, FRAMES), (3, FRAMES)],
    [(0, FRAMES), (2, FRAMES), (3, FRAMES)],
]


def _run(graph: StreamingAudioGraphEncoder, sessions: int = 4):
    caches: list[StreamingAudioKVCache | None] = [None] * sessions
    rounds = []
    with torch.inference_mode():
        for round_index, present in enumerate(_SCHEDULE):
            chunks = [_chunk(caches[s], seed=1000 * round_index + s, frames=frames) for s, frames in present]
            outputs, updated = graph.encode(chunks)
            for (s, _), cache in zip(present, updated, strict=True):
                caches[s] = cache
            rounds.append(
                (
                    [out.clone() for out in outputs],
                    {s: (caches[s].length, _history(caches[s]).clone()) for s, _ in present},
                )
            )
    torch.accelerator.synchronize()
    return rounds, caches


@pytest.fixture(params=["sdpa", "eager"])
def audio(request):
    if not torch.cuda.is_available():
        pytest.skip("CUDA graph capture needs a GPU")
    return _Audio(request.param)


@pytest.mark.cuda
@pytest.mark.parametrize(
    "cache_buckets",
    [(16, 32), (16,)],
    ids=["two-buckets", "past-the-largest-bucket-goes-eager"],
)
def test_resident_graph_is_bitwise_identical_to_the_copy_in_graph(audio, cache_buckets) -> None:
    batch_sizes = (1, 2, 4)
    copy_in = audio.graph(resident_slots=0, batch_sizes=batch_sizes, cache_buckets=cache_buckets)
    resident = audio.graph(resident_slots=4, batch_sizes=batch_sizes, cache_buckets=cache_buckets)
    assert copy_in._cache_storage is not None and resident._cache_storage is None

    expected, _ = _run(copy_in)
    got, caches = _run(resident)
    assert all(isinstance(cache, PooledStreamingAudioKVCache) for cache in caches)
    lengths = set()
    for (exp_out, exp_caches), (got_out, got_caches) in zip(expected, got, strict=True):
        for e, g in zip(exp_out, got_out, strict=True):
            assert torch.equal(e, g)
        assert exp_caches.keys() == got_caches.keys()
        for s, (length, history) in exp_caches.items():
            assert got_caches[s][0] == length
            assert torch.equal(got_caches[s][1], history)
            lengths.add(length)
    # Graph resets (a steady unit from past 0: length 10) and eager resets (a
    # non-steady unit from past 0: 11, also every first unit) both happened.
    assert 10 in lengths and 11 in lengths


@pytest.mark.cuda
def test_resident_slots_are_released_with_their_session(audio) -> None:
    resident = audio.graph(resident_slots=3, batch_sizes=(1, 2, 4), cache_buckets=(16, 32))
    pool = resident._pool
    assert isinstance(pool, StreamingAudioKVSlotPool) and pool.free_slots == 3
    _, caches = _run(resident, sessions=4)
    # Four sessions, three slots: the fourth to start got a paged cache.
    pooled = [c for c in caches if isinstance(c, PooledStreamingAudioKVCache)]
    assert len(pooled) == 3 and pool.free_slots == 0
    assert sorted(c.slot for c in pooled) == [0, 1, 2]
    assert sum(type(c) is StreamingAudioKVCache for c in caches) == 1
    del pooled, caches
    gc.collect()
    assert pool.free_slots == 3


@pytest.mark.cuda
def test_paged_rows_under_a_full_pool_match_the_eager_batch(audio) -> None:
    """A session that found the pool full runs eagerly, as encode_streaming_audio_batch would."""
    resident = audio.graph(resident_slots=1, batch_sizes=(1, 2), cache_buckets=(16, 32))
    with torch.inference_mode():
        (_, _), (held, paged) = resident.encode([_chunk(None, seed=1), _chunk(None, seed=2)])
        assert isinstance(held, PooledStreamingAudioKVCache) and type(paged) is StreamingAudioKVCache
        reference = StreamingAudioKVCache(
            num_layers=paged.num_layers,
            embed_dim=paged.embed_dim,
            num_heads=paged.num_heads,
            max_positions=paged.max_positions,
            page_positions=16,
        )
        reference.reserve(paged.length, dtype=torch.bfloat16, device=torch.device("cuda"))
        reference.commit(0, _history(paged))
        reference.length = paged.length
        (got,), _ = resident.encode([_chunk(paged, seed=3)])
        (want,), _ = encode_streaming_audio_batch(
            audio.apm, audio.proj, audio.pooler, [_chunk(reference, seed=3)], pool_step=POOL
        )
    assert torch.equal(got, want)
    assert torch.equal(_history(paged), _history(reference))


@pytest.mark.cuda
def test_resident_storage_replaces_the_shared_history_storage(audio) -> None:
    resident = audio.graph(resident_slots=4, batch_sizes=(1, 2, 4), cache_buckets=(16, 32))
    storage = resident._pool.storage
    # [slots, layers, 2, max_positions + unit_length tail, d]
    assert tuple(storage.shape) == (4, 2, 2, MAX_POSITIONS + resident.unit_length, 32)
    assert resident._cache_storage is None
