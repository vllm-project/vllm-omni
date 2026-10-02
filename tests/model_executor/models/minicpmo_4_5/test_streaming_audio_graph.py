# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""CUDA-graph capture of the streaming Whisper encoder's steady unit shape.

The bucket/eligibility helpers are pure Python and run on CPU; the graph
itself needs CUDA (``torch.cuda.graph``), so ``test_graph_matches_eager_*``
are skipped without a GPU. Both halves check the same thing the module
promises: batch and history padding never change a real row's result.
"""

from __future__ import annotations

import pytest
import torch
from torch import nn

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
from vllm_omni.model_executor.models.minicpmo_4_5.streaming_audio_encoder_graph import (
    StreamingAudioGraphEncoder,
    normalize_buckets,
    row_is_steady,
    select_bucket,
    steady_pooled_length,
    steady_unit_length,
)

pytestmark = [pytest.mark.core_model]

_cpu = pytest.mark.cpu
_cuda = pytest.mark.cuda

N_MELS = 16
POOL = 5
FRAMES = 24  # steady unit: conv_length 12, trimmed by 1 + 1 -> unit_length 10 (POOL-aligned)


# --- pure bucket/eligibility logic: CPU, no CUDA needed -----------------------------------------


@_cpu
def test_select_bucket_picks_the_smallest_covering_value() -> None:
    assert select_bucket(0, (500, 1000, 1500)) == 500
    assert select_bucket(500, (500, 1000, 1500)) == 500
    assert select_bucket(501, (500, 1000, 1500)) == 1000
    assert select_bucket(1500, (500, 1000, 1500)) == 1500
    assert select_bucket(1501, (500, 1000, 1500)) is None


@_cpu
def test_normalize_buckets_dedupes_sorts_and_drops_non_positive() -> None:
    assert normalize_buckets([1000, 500, 500, 0, -5, 1000]) == (500, 1000)


@_cpu
def test_steady_unit_length_matches_the_conv_and_trim_arithmetic() -> None:
    # frames=24 -> conv_length (24-1)//2+1 = 12; trim(2) = (2+1)//2 = 1 each side.
    assert steady_unit_length(24) == 10
    # frames=22 -> conv_length 11; still trimmed by 1 + 1.
    assert steady_unit_length(22) == 9


def _chunk(
    *,
    frames: int = FRAMES,
    cache=object(),
    prefix: int = 2,
    suffix: int = 2,
    use_extra_context=True,
    feature_length=None,
):
    return StreamingAudioChunk(
        features=torch.zeros(1, N_MELS, frames),
        cache=cache,
        prefix_extra_frames=prefix,
        suffix_extra_frames=suffix,
        use_extra_context=use_extra_context,
        feature_length=feature_length,
    )


@_cpu
def test_row_is_steady_accepts_only_the_stage0_steady_shape() -> None:
    assert row_is_steady(_chunk(), expected_frames=FRAMES)
    # A session's first unit: no cache yet, and prefix_extra_frames=0.
    assert not row_is_steady(_chunk(cache=None), expected_frames=FRAMES)
    assert not row_is_steady(_chunk(prefix=0), expected_frames=FRAMES)
    # Any other trim, frame count, or a short final unit falls back too.
    assert not row_is_steady(_chunk(suffix=0), expected_frames=FRAMES)
    assert not row_is_steady(_chunk(use_extra_context=False), expected_frames=FRAMES)
    assert not row_is_steady(_chunk(frames=FRAMES + 1), expected_frames=FRAMES)
    assert not row_is_steady(_chunk(feature_length=FRAMES - 3), expected_frames=FRAMES)
    assert row_is_steady(_chunk(feature_length=FRAMES), expected_frames=FRAMES)


@_cpu
def test_steady_pooled_length_takes_the_smaller_of_cap_and_actual() -> None:
    # frames=24 -> unit_length 10: cap ((24-1)//2+1-5)//5+1=2, actual 10//5=2.
    assert steady_pooled_length(24, 10, 5) == 2
    # Contrived numbers where the feature_length-derived cap binds below the
    # unit_length-derived actual pooled count: still take the min, not actual.
    assert steady_pooled_length(unit_frames=10, unit_length=100, pool_step=3) == 1


# --- CUDA: graph replay vs. eager, bucketed and padded ----------------------------------------


class _Thinker:
    """The audio half of the thinker: real modules, the real methods under test."""

    get_audio_embedding_streaming_batch = MiniCPMO45OmniLLMForConditionalGeneration.get_audio_embedding_streaming_batch
    supports_streaming_audio_batch = MiniCPMO45OmniLLMForConditionalGeneration.supports_streaming_audio_batch

    def __init__(self, *, max_positions: int, dtype=torch.bfloat16, device="cuda") -> None:
        from types import SimpleNamespace

        from transformers.models.whisper.modeling_whisper import WhisperConfig

        torch.manual_seed(0)
        config = WhisperConfig(
            num_mel_bins=N_MELS,
            d_model=32,
            encoder_layers=2,
            encoder_attention_heads=2,
            encoder_ffn_dim=128,
            max_source_positions=max_positions,
            dropout=0.0,
            attention_dropout=0.0,
        )
        config._attn_implementation = "sdpa"
        self.apm = MiniCPMWhisperEncoder(config).eval().to(device=device, dtype=dtype)
        self.audio_projection_layer = MultiModalProjector(in_dim=config.encoder_ffn_dim // 4, out_dim=24).to(
            device=device, dtype=dtype
        )
        self.audio_avg_pooler = nn.AvgPool1d(POOL, stride=POOL)
        self.audio_encoder_layer = -1
        self.config = SimpleNamespace(audio_pool_step=POOL, duplex_audio_kv_page_positions=16)


def _mel(frames: int, seed: int, device: str) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    return torch.randn((1, N_MELS, frames), generator=generator).to(device)


def _clone_cache(cache: StreamingAudioKVCache) -> StreamingAudioKVCache:
    clone = StreamingAudioKVCache(
        num_layers=cache.num_layers,
        embed_dim=cache.embed_dim,
        num_heads=cache.num_heads,
        max_positions=cache.max_positions,
        page_positions=cache.page_positions,
    )
    if cache.length:
        history = cache.history(cache.length)
        clone.reserve(cache.length, dtype=history.dtype, device=history.device)
        clone.commit(0, history)
        clone.length = cache.length
    return clone


def _start_session(thinker: _Thinker, seed: int, *, frames: int = FRAMES) -> StreamingAudioKVCache:
    """One session's first (non-steady) unit, eager: the real cache-in state."""
    (embeds,), (cache,) = encode_streaming_audio_batch(
        thinker.apm,
        thinker.audio_projection_layer,
        thinker.audio_avg_pooler,
        [
            StreamingAudioChunk(
                features=_mel(frames, seed, "cuda"), cache=None, prefix_extra_frames=0, suffix_extra_frames=2
            )
        ],
        pool_step=POOL,
    )
    assert embeds is not None
    return cache


@pytest.fixture
def cuda_thinker():
    if not torch.cuda.is_available():
        pytest.skip("CUDA graph capture needs a GPU")
    return _Thinker(max_positions=40)


@_cuda
def test_graph_matches_eager_across_batch_padding_and_a_cache_bucket_boundary(cuda_thinker) -> None:
    thinker = cuda_thinker
    graph = StreamingAudioGraphEncoder(
        thinker.apm,
        thinker.audio_projection_layer,
        thinker.audio_avg_pooler,
        unit_frames=FRAMES,
        pool_step=POOL,
        batch_sizes=(1, 2, 4),
        cache_buckets=(16, 32, 40),
        page_positions=16,
    )
    with torch.inference_mode():
        graph.capture()
    assert graph.unit_length == steady_unit_length(FRAMES) == 10

    sessions = 3
    eager_caches = [_start_session(thinker, seed=session) for session in range(sessions)]
    graph_caches = [_clone_cache(cache) for cache in eager_caches]

    # 4 rounds: past grows 11 -> 21 -> 31 (crossing the 16 bucket into 32);
    # by round 3 some sessions' past + unit_length (>= 40) trips the eager
    # reset path (past back to 0) on both sides, at different rounds for
    # different sessions. Round 1 drops a session (batch padding: 2 of 3
    # sessions ready that round).
    present_by_round = [[0, 1, 2], [0, 2], [0, 1, 2], [0, 1, 2]]
    with torch.inference_mode():
        for round_index, present in enumerate(present_by_round):
            eager_chunks = [
                StreamingAudioChunk(
                    features=_mel(FRAMES, 100 * round_index + s, "cuda"),
                    cache=eager_caches[s],
                    prefix_extra_frames=2,
                    suffix_extra_frames=2,
                )
                for s in present
            ]
            eager_outputs, eager_updated = encode_streaming_audio_batch(
                thinker.apm, thinker.audio_projection_layer, thinker.audio_avg_pooler, eager_chunks, pool_step=POOL
            )
            for s, cache in zip(present, eager_updated, strict=True):
                eager_caches[s] = cache

            graph_chunks = [
                StreamingAudioChunk(
                    features=_mel(FRAMES, 100 * round_index + s, "cuda"),
                    cache=graph_caches[s],
                    prefix_extra_frames=2,
                    suffix_extra_frames=2,
                )
                for s in present
            ]
            graph_outputs, graph_updated = graph.encode(graph_chunks)
            for s, cache in zip(present, graph_updated, strict=True):
                graph_caches[s] = cache

            for eager_out, graph_out in zip(eager_outputs, graph_outputs, strict=True):
                assert eager_out is not None and graph_out is not None
                torch.testing.assert_close(graph_out.float(), eager_out.float(), rtol=2e-2, atol=2e-2)
            for s in present:
                assert graph_caches[s].length == eager_caches[s].length
                torch.testing.assert_close(
                    graph_caches[s].history(graph_caches[s].length).float(),
                    eager_caches[s].history(eager_caches[s].length).float(),
                    rtol=2e-2,
                    atol=2e-2,
                )


def _cache_with_history(thinker: _Thinker, *, past: int, seed: int) -> StreamingAudioKVCache:
    """A cache whose committed history is exactly ``past`` (a multiple of the
    steady unit_length, 10) positions of real, reproducible content.

    Starts from an empty (not ``None``) cache and only ever runs steady
    (``prefix_extra_frames=2``) units through it -- ``encode_streaming_audio_batch``
    resets on ``cache is None``, not on an empty-but-real cache, so this never
    takes the session's-first-unit path and each unit adds exactly 10.
    """
    assert past % 10 == 0
    cache = StreamingAudioKVCache(
        num_layers=len(thinker.apm.layers),
        embed_dim=int(thinker.apm.config.d_model),
        num_heads=int(thinker.apm.config.encoder_attention_heads),
        max_positions=int(thinker.apm.embed_positions.weight.shape[0]),
        page_positions=16,
    )
    for step in range(past // 10):
        mel = _mel(FRAMES, seed * 1000 + step, "cuda")
        with torch.inference_mode():
            (_out,), (cache,) = encode_streaming_audio_batch(
                thinker.apm,
                thinker.audio_projection_layer,
                thinker.audio_avg_pooler,
                [StreamingAudioChunk(features=mel, cache=cache, prefix_extra_frames=2, suffix_extra_frames=2)],
                pool_step=POOL,
            )
    assert cache.length == past, f"expected past={past}, got {cache.length}"
    return cache


@_cuda
def test_alternating_buckets_share_storage_without_leaking(cuda_thinker) -> None:
    """Every graph takes a prefix view of one shared storage (see the module
    docstring): a small-batch/small-cache-bucket graph and a large one
    physically overlap in that storage. Replay them interleaved, out of
    capture order, and check neither ever reads the other's leftovers."""
    thinker = cuda_thinker
    graph = StreamingAudioGraphEncoder(
        thinker.apm,
        thinker.audio_projection_layer,
        thinker.audio_avg_pooler,
        unit_frames=FRAMES,
        pool_step=POOL,
        batch_sizes=(1, 4),
        cache_buckets=(16, 32),
        page_positions=16,
    )
    with torch.inference_mode():
        graph.capture()
    # (1, 16) and (4, 32) are both prefixes of the same shared storage; (1,
    # 16)'s whole footprint sits inside (4, 32)'s (same base pointer).
    assert graph._views(1, 16)[1].data_ptr() == graph._views(4, 32)[1].data_ptr()

    # Fixed pasts (10 and 20: both multiples of the unit_length-10 steady
    # step, so they land exactly, no drift) that resolve to distinct keys:
    # 1 session at past 10 -> (1, 16); 4 sessions at past 20 -> (4, 32).
    eager_small = [_cache_with_history(thinker, past=10, seed=11)]
    eager_big = [_cache_with_history(thinker, past=20, seed=200 + i) for i in range(4)]

    def replay_round(base_caches: list[StreamingAudioKVCache], *, seed_base: int):
        # Fresh clones every round, on both sides: the point is to replay the
        # *same* (batch, cache bucket) key repeatedly, interleaved with the
        # other key, so past must not drift between rounds.
        eager_caches = [_clone_cache(c) for c in base_caches]
        graph_caches = [_clone_cache(c) for c in base_caches]

        def chunks(caches: list[StreamingAudioKVCache]) -> list[StreamingAudioChunk]:
            return [
                StreamingAudioChunk(
                    features=_mel(FRAMES, seed_base + s, "cuda"),
                    cache=caches[s],
                    prefix_extra_frames=2,
                    suffix_extra_frames=2,
                )
                for s in range(len(caches))
            ]

        with torch.inference_mode():
            eager_outputs, eager_updated = encode_streaming_audio_batch(
                thinker.apm,
                thinker.audio_projection_layer,
                thinker.audio_avg_pooler,
                chunks(eager_caches),
                pool_step=POOL,
            )
            graph_outputs, graph_updated = graph.encode(chunks(graph_caches))
        for s in range(len(base_caches)):
            torch.testing.assert_close(graph_outputs[s].float(), eager_outputs[s].float(), rtol=2e-2, atol=2e-2)
            assert graph_updated[s].length == eager_updated[s].length
            torch.testing.assert_close(
                graph_updated[s].history(graph_updated[s].length).float(),
                eager_updated[s].history(eager_updated[s].length).float(),
                rtol=2e-2,
                atol=2e-2,
            )

    # Interleave (1, 16) and (4, 32) replays, out of the order they were
    # captured in, several times: if shared storage leaked between them, one
    # of these rounds would see the other key's stale content instead of its
    # own freshly refilled inputs.
    for round_index in range(4):
        replay_round(eager_small, seed_base=1000 + round_index)
        replay_round(eager_big, seed_base=2000 + round_index)
        replay_round(eager_big, seed_base=3000 + round_index)
        replay_round(eager_small, seed_base=4000 + round_index)


@_cuda
def test_graph_batch_size_one_is_near_bit_exact_with_eager(cuda_thinker) -> None:
    """No batch padding and no history padding (cache bucket == true past): tightest case."""
    thinker = cuda_thinker
    graph = StreamingAudioGraphEncoder(
        thinker.apm,
        thinker.audio_projection_layer,
        thinker.audio_avg_pooler,
        unit_frames=FRAMES,
        pool_step=POOL,
        batch_sizes=(1,),
        cache_buckets=(11, 21, 31),
        page_positions=16,
    )
    with torch.inference_mode():
        graph.capture()
    eager_cache = _start_session(thinker, seed=0)  # length 11
    graph_cache = _clone_cache(eager_cache)

    with torch.inference_mode():
        eager_chunk = StreamingAudioChunk(
            features=_mel(FRAMES, 1, "cuda"), cache=eager_cache, prefix_extra_frames=2, suffix_extra_frames=2
        )
        (eager_out,), (eager_cache,) = encode_streaming_audio_batch(
            thinker.apm, thinker.audio_projection_layer, thinker.audio_avg_pooler, [eager_chunk], pool_step=POOL
        )
        graph_chunk = StreamingAudioChunk(
            features=_mel(FRAMES, 1, "cuda"), cache=graph_cache, prefix_extra_frames=2, suffix_extra_frames=2
        )
        (graph_out,), (graph_cache,) = graph.encode([graph_chunk])

    torch.testing.assert_close(graph_out, eager_out)
    assert graph_cache.length == eager_cache.length
    torch.testing.assert_close(graph_cache.history(graph_cache.length), eager_cache.history(eager_cache.length))


@_cuda
def test_graph_leaves_ineligible_rows_to_the_eager_fallback(cuda_thinker) -> None:
    thinker = cuda_thinker
    graph = StreamingAudioGraphEncoder(
        thinker.apm,
        thinker.audio_projection_layer,
        thinker.audio_avg_pooler,
        unit_frames=FRAMES,
        pool_step=POOL,
        batch_sizes=(2,),
        cache_buckets=(32,),
        page_positions=16,
    )
    with torch.inference_mode():
        graph.capture()
    with torch.inference_mode():
        # A session's first unit (cache=None, prefix=0): not steady, must
        # still succeed via the eager fallback inside encode().
        first_chunk = StreamingAudioChunk(
            features=_mel(FRAMES, 5, "cuda"), cache=None, prefix_extra_frames=0, suffix_extra_frames=2
        )
        (out,), (cache,) = graph.encode([first_chunk])
    assert out is not None
    assert isinstance(cache, StreamingAudioKVCache) and cache.length > 0


@_cuda
def test_graph_matches_eager_with_a_non_pool_step_aligned_unit_length(cuda_thinker) -> None:
    """Real checkpoints do not guarantee unit_length % pool_step == 0 (see
    ``steady_pooled_length``'s cap/actual ``min``); this checks the graph
    still matches eager when it is not (unit_frames=25 -> unit_length 11)."""
    thinker = cuda_thinker
    odd_frames = FRAMES + 1
    graph = StreamingAudioGraphEncoder(
        thinker.apm,
        thinker.audio_projection_layer,
        thinker.audio_avg_pooler,
        unit_frames=odd_frames,
        pool_step=POOL,
        batch_sizes=(1,),
        cache_buckets=(16,),
        page_positions=16,
    )
    with torch.inference_mode():
        graph.capture()
    assert graph.unit_length == steady_unit_length(odd_frames) == 11
    assert graph.pooled_length == steady_pooled_length(odd_frames, 11, POOL) == 2

    eager_cache = _start_session(thinker, seed=0, frames=odd_frames)
    graph_cache = _clone_cache(eager_cache)
    with torch.inference_mode():
        eager_chunk = StreamingAudioChunk(
            features=_mel(odd_frames, 1, "cuda"), cache=eager_cache, prefix_extra_frames=2, suffix_extra_frames=2
        )
        (eager_out,), _ = encode_streaming_audio_batch(
            thinker.apm, thinker.audio_projection_layer, thinker.audio_avg_pooler, [eager_chunk], pool_step=POOL
        )
        graph_chunk = StreamingAudioChunk(
            features=_mel(odd_frames, 1, "cuda"), cache=graph_cache, prefix_extra_frames=2, suffix_extra_frames=2
        )
        (graph_out,), _ = graph.encode([graph_chunk])
    assert eager_out.shape[0] == 2
    torch.testing.assert_close(graph_out, eager_out)


def _steady_round(thinker: _Thinker, graph: StreamingAudioGraphEncoder, cache, seed: int):
    chunk = StreamingAudioChunk(
        features=_mel(FRAMES, seed, "cpu"), cache=cache, prefix_extra_frames=2, suffix_extra_frames=2
    )
    with torch.inference_mode():
        (out,), (cache,) = graph.encode([chunk])
    return out, cache


@_cuda
def test_pinned_h2d_is_bitwise_identical_and_does_not_block(cuda_thinker) -> None:
    """Host mel staged through pinned memory: same bits, and no blocking copy in the replay."""
    thinker = cuda_thinker
    graphs = []
    for pinned in (False, True):
        graph = StreamingAudioGraphEncoder(
            thinker.apm,
            thinker.audio_projection_layer,
            thinker.audio_avg_pooler,
            unit_frames=FRAMES,
            pool_step=POOL,
            batch_sizes=(1,),
            cache_buckets=(11, 21, 31),
            page_positions=16,
            pinned_h2d=pinned,
        )
        with torch.inference_mode():
            graph.capture()
        graphs.append(graph)
    base_cache = _start_session(thinker, seed=0)
    blocking_out, blocking_cache = _steady_round(thinker, graphs[0], _clone_cache(base_cache), seed=7)
    torch.accelerator.synchronize()
    torch.cuda.set_sync_debug_mode("error")
    try:
        pinned_out, pinned_cache = _steady_round(thinker, graphs[1], _clone_cache(base_cache), seed=7)
    finally:
        torch.cuda.set_sync_debug_mode("default")
    torch.accelerator.synchronize()
    assert torch.equal(pinned_out, blocking_out)
    assert torch.equal(pinned_cache.history(pinned_cache.length), blocking_cache.history(blocking_cache.length))
