# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Cross-session batched streaming audio encoder against the per-session path.

A tiny random Whisper-like encoder runs several sessions through both paths:
the per-session ``get_audio_embedding_streaming`` (``EncoderDecoderCache``) and
one ``encode_streaming_audio_batch`` call per round. Ragged mel lengths, rows
that sit a round out, units the pooler cannot divide, the 30 s position-bound
reset and the Stage-0 prefill staging are all covered.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn
from transformers.models.whisper.modeling_whisper import WhisperConfig

from vllm_omni.model_executor.models.minicpmo_4_5.minicpmo_4_5_omni_llm import (
    MiniCPMO45OmniLLMForConditionalGeneration,
    MiniCPMWhisperEncoder,
    MultiModalProjector,
)
from vllm_omni.model_executor.models.minicpmo_4_5.streaming_audio_encoder import (
    StreamingAudioChunk,
    StreamingAudioKVCache,
    streaming_batch_unsupported_reason,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

N_MELS = 16
POOL = 5


class _Thinker:
    """The audio half of the thinker: real modules, the real methods under test."""

    get_audio_embedding_streaming = MiniCPMO45OmniLLMForConditionalGeneration.get_audio_embedding_streaming
    get_audio_embedding_streaming_batch = MiniCPMO45OmniLLMForConditionalGeneration.get_audio_embedding_streaming_batch
    supports_streaming_audio_batch = MiniCPMO45OmniLLMForConditionalGeneration.supports_streaming_audio_batch
    _get_feat_extract_output_lengths = MiniCPMO45OmniLLMForConditionalGeneration._get_feat_extract_output_lengths

    def __init__(self, *, attn_implementation: str = "sdpa", max_positions: int = 40, dtype=torch.float32) -> None:
        torch.manual_seed(0)
        config = WhisperConfig(
            num_mel_bins=N_MELS,
            d_model=32,
            encoder_layers=2,
            encoder_attention_heads=2,
            encoder_ffn_dim=128,  # the projection takes ffn // 4 == d_model, as in the checkpoint
            max_source_positions=max_positions,
            dropout=0.0,
            attention_dropout=0.0,
        )
        config._attn_implementation = attn_implementation
        self.apm = MiniCPMWhisperEncoder(config).eval().to(dtype)
        self.audio_projection_layer = MultiModalProjector(in_dim=config.encoder_ffn_dim // 4, out_dim=24).to(dtype)
        self.audio_avg_pooler = nn.AvgPool1d(POOL, stride=POOL)
        self.audio_encoder_layer = -1
        self.audio_past_key_values = None
        self.config = SimpleNamespace(audio_pool_step=POOL, duplex_audio_kv_page_positions=16)


def _mel(frames: int, seed: int) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    return torch.randn((1, N_MELS, frames), generator=generator)


def _sequential(thinker: _Thinker, cache, mel: torch.Tensor, prefix: int, *, use_extra_context: bool = True):
    """One unit through the per-session path, with that session's cache swapped in."""
    thinker.audio_past_key_values = cache
    with torch.no_grad():
        nested = thinker.get_audio_embedding_streaming(
            {"audio_features": mel, "audio_feature_lens": [torch.tensor([mel.shape[-1]])]},
            use_extra_context=use_extra_context,
            prefix_extra_frames=prefix,
            suffix_extra_frames=2,
        )
    embeds = torch.cat([t for row in nested for t in row]) if nested else None
    return embeds, thinker.audio_past_key_values


def _legacy_layer(cache, layer: int) -> tuple[torch.Tensor, torch.Tensor]:
    layer_cache = cache.self_attention_cache.layers[layer]
    return layer_cache.keys, layer_cache.values


# Per round, the mel frames of each session's unit (None: sits the round out).
# 22 frames at chunk 0 and 24 later mirror 1020 ms / 1040 ms at 1/10 scale; the
# odd lengths make conv2 read a padded column and 25 frames give an 11-position
# unit that the pooler cannot divide. max_positions=40 forces a reset per cycle.
_SCHEDULE = [
    [22, 22, 23, 21],
    [24, None, 24, 25],
    [24, 24, None, 24],
    [24, 23, 24, 24],
    [24, 24, 24, None],
    [25, 24, 23, 24],
]


@pytest.mark.parametrize("attn_implementation", ["eager", "sdpa"])
def test_batched_rounds_match_sequential_sessions(attn_implementation: str) -> None:
    thinker = _Thinker(attn_implementation=attn_implementation)
    sessions = len(_SCHEDULE[0])
    legacy_caches: list[object | None] = [None] * sessions
    batched_caches: list[StreamingAudioKVCache | None] = [None] * sessions
    chunk_index = [0] * sessions
    resets = 0

    for round_index, frames_per_session in enumerate(_SCHEDULE):
        chunks, active, expected = [], [], []
        for session, frames in enumerate(frames_per_session):
            if frames is None:
                continue
            mel = _mel(frames, seed=100 * round_index + session)
            prefix = 0 if chunk_index[session] == 0 else 2
            before = batched_caches[session]
            embeds, legacy_caches[session] = _sequential(thinker, legacy_caches[session], mel, prefix)
            expected.append(embeds)
            chunks.append(
                StreamingAudioChunk(features=mel, cache=before, prefix_extra_frames=prefix, suffix_extra_frames=2)
            )
            active.append(session)
            chunk_index[session] += 1

        idle = {s: (batched_caches[s], batched_caches[s].length if batched_caches[s] else 0) for s in range(sessions)}
        outputs, caches = thinker.get_audio_embedding_streaming_batch(chunks)

        for session, output, cache, reference in zip(active, outputs, caches, expected, strict=True):
            assert output is not None and reference is not None
            torch.testing.assert_close(output, reference, rtol=1e-5, atol=1e-5)
            if cache is not batched_caches[session] and batched_caches[session] is not None:
                resets += 1
            batched_caches[session] = cache
            legacy_length = legacy_caches[session].self_attention_cache.get_seq_length()
            assert cache.length == legacy_length
            for layer in range(len(thinker.apm.layers)):
                keys, values = _legacy_layer(legacy_caches[session], layer)
                torch.testing.assert_close(cache.head_view(cache.keys(layer)[: cache.length]), keys)
                torch.testing.assert_close(cache.head_view(cache.values(layer)[: cache.length]), values)
        for session in set(range(sessions)) - set(active):
            # Not in the batch: the same cache object, the same committed length.
            cache, length = idle[session]
            assert batched_caches[session] is cache
            assert (cache.length if cache is not None else 0) == length

    assert resets >= 2, "the schedule must cross the max_source_positions reset"


def test_ragged_rows_without_extra_context_see_zeros_past_their_end() -> None:
    """Untrimmed odd-length rows: conv2's last column reads past the row's frames."""
    thinker = _Thinker()
    legacy: list[object | None] = [None] * 3
    batched: list[StreamingAudioKVCache | None] = [None] * 3
    for index, lengths in enumerate([(21, 25, 23), (19, 25, 24)]):
        chunks, expected = [], []
        for session, frames in enumerate(lengths):
            mel = _mel(frames, seed=20 * index + session)
            reference, legacy[session] = _sequential(thinker, legacy[session], mel, 0, use_extra_context=False)
            expected.append(reference)
            chunks.append(StreamingAudioChunk(features=mel, cache=batched[session], use_extra_context=False))
        outputs, batched = thinker.get_audio_embedding_streaming_batch(chunks)
        for output, reference in zip(outputs, expected, strict=True):
            torch.testing.assert_close(output, reference, rtol=1e-5, atol=1e-5)


def test_single_row_matches_the_per_session_path_bit_for_bit() -> None:
    thinker = _Thinker(attn_implementation="eager")
    legacy_cache, cache = None, None
    for index, frames in enumerate([22, 24, 24]):
        mel = _mel(frames, seed=index)
        prefix = 0 if index == 0 else 2
        reference, legacy_cache = _sequential(thinker, legacy_cache, mel, prefix)
        (output,), (cache,) = thinker.get_audio_embedding_streaming_batch(
            [StreamingAudioChunk(features=mel, cache=cache, prefix_extra_frames=prefix, suffix_extra_frames=2)]
        )
        # Same shapes, same kernels on CPU: batch size one is exact.
        assert torch.equal(output, reference)


def test_bf16_batch_stays_within_bf16_rounding_of_the_per_session_path() -> None:
    thinker = _Thinker(attn_implementation="sdpa", dtype=torch.bfloat16)
    legacy = [None, None, None]
    batched = [None, None, None]
    for index in range(3):
        chunks, expected = [], []
        for session, frames in enumerate([22 if index == 0 else 24, 23 if index == 0 else 24, 21]):
            mel = _mel(frames, seed=10 * index + session)
            prefix = 0 if index == 0 else 2
            reference, legacy[session] = _sequential(thinker, legacy[session], mel, prefix)
            expected.append(reference)
            chunks.append(
                StreamingAudioChunk(
                    features=mel, cache=batched[session], prefix_extra_frames=prefix, suffix_extra_frames=2
                )
            )
        outputs, batched = thinker.get_audio_embedding_streaming_batch(chunks)
        for output, reference in zip(outputs, expected, strict=True):
            torch.testing.assert_close(output.float(), reference.float(), rtol=3e-2, atol=3e-2)


def test_unit_empty_after_trimming_returns_none_and_keeps_the_cache() -> None:
    thinker = _Thinker()
    (first,), (cache,) = thinker.get_audio_embedding_streaming_batch(
        [StreamingAudioChunk(features=_mel(22, 0), cache=None, prefix_extra_frames=0, suffix_extra_frames=2)]
    )
    assert first is not None
    length = cache.length
    # 4 frames -> 2 CNN positions, both trimmed as extra context.
    (empty, full), (same, other) = thinker.get_audio_embedding_streaming_batch(
        [
            StreamingAudioChunk(features=_mel(4, 1), cache=cache, prefix_extra_frames=2, suffix_extra_frames=2),
            StreamingAudioChunk(features=_mel(24, 2), cache=None, prefix_extra_frames=0, suffix_extra_frames=2),
        ]
    )
    assert empty is None and full is not None
    assert same is cache and cache.length == length
    assert other is not None and other.length > 0


def test_unit_longer_than_the_position_table_repeats_its_last_row() -> None:
    thinker = _Thinker(max_positions=8)
    mel = _mel(24, 3)  # 12 CNN positions -> 10 after trimming, past the 8-row table.
    reference, _ = _sequential(thinker, None, mel, prefix=2)
    (output,), _ = thinker.get_audio_embedding_streaming_batch(
        [StreamingAudioChunk(features=mel, cache=None, prefix_extra_frames=2, suffix_extra_frames=2)]
    )
    torch.testing.assert_close(output, reference)


def test_kv_cache_grows_by_pages_up_to_the_position_bound_and_keeps_history() -> None:
    cache = StreamingAudioKVCache(num_layers=2, embed_dim=8, num_heads=2, max_positions=40, page_positions=16)
    cache.reserve(10, dtype=torch.float32, device=torch.device("cpu"))
    assert cache.capacity == 16
    cache.keys(1)[:10] = torch.arange(80, dtype=torch.float32).reshape(10, 8)
    cache.length = 10
    cache.reserve(20, dtype=torch.float32, device=torch.device("cpu"))
    assert cache.capacity == 32
    torch.testing.assert_close(cache.keys(1)[:10], torch.arange(80, dtype=torch.float32).reshape(10, 8))
    cache.reserve(33, dtype=torch.float32, device=torch.device("cpu"))
    assert cache.capacity == 40
    assert cache.head_view(cache.keys(0)[:5]).shape == (1, 2, 5, 4)


def test_legacy_cache_conversion_continues_the_same_history() -> None:
    thinker = _Thinker()
    legacy_cache, cache = None, None
    for index in range(2):
        mel = _mel(22 if index == 0 else 24, index)
        _, legacy_cache = _sequential(thinker, legacy_cache, mel, 0 if index == 0 else 2)
        _, (cache,) = thinker.get_audio_embedding_streaming_batch(
            [
                StreamingAudioChunk(
                    features=mel, cache=cache, prefix_extra_frames=0 if index == 0 else 2, suffix_extra_frames=2
                )
            ]
        )
    mel = _mel(24, 7)
    reference, _ = _sequential(thinker, legacy_cache, mel, 2)
    converted, _ = _sequential(thinker, cache.to_legacy_cache(), mel, 2)
    torch.testing.assert_close(converted, reference)


def test_support_gate_names_the_configurations_it_leaves_per_session() -> None:
    thinker = _Thinker()
    assert streaming_batch_unsupported_reason(thinker.apm, audio_encoder_layer=-1) is None
    assert thinker.supports_streaming_audio_batch()
    assert "final encoder layer" in streaming_batch_unsupported_reason(thinker.apm, audio_encoder_layer=3)
    thinker.config.duplex_audio_encoder_batching = False
    assert not thinker.supports_streaming_audio_batch()
    half = _Thinker(dtype=torch.float16)
    assert "dtype" in streaming_batch_unsupported_reason(half.apm, audio_encoder_layer=-1)
    flash = _Thinker()
    flash.apm.config._attn_implementation = "flash_attention_2"
    assert "flash_attention_2" in streaming_batch_unsupported_reason(flash.apm, audio_encoder_layer=-1)


# --- Stage-0 prefill staging -----------------------------------------------------------------


class _Feature(dict):
    """A BatchFeature-like mapping that takes the chunk attributes Stage 0 sets."""


class _MelProcessor:
    """Deterministic stand-in for the remote-code streaming mel processor (copied per session)."""

    def __init__(self) -> None:
        self.calls = 0

    def get_streaming_chunk_size(self) -> int:
        return 240

    def process_audio_streaming(self, audio_chunk, reset=False, return_batch_feature=True):
        del reset, return_batch_feature
        self.calls += 1
        frames = 22 if self.calls == 1 else 24
        seed = int(abs(float(np.asarray(audio_chunk).sum())) * 1000) % 10_000 + self.calls
        mel = _mel(frames, seed)
        return _Feature(audio_features=mel, audio_feature_lens=[torch.tensor([frames])])


class _StageModel(nn.Module):
    def __init__(self, thinker: _Thinker) -> None:
        super().__init__()
        self.embed = nn.Embedding(256, 24)
        self._thinker = thinker

    def get_input_embeddings(self):
        return self.embed


def _runtime(batching: bool):
    from vllm_omni.model_executor.models.minicpmo_4_5.duplex.stage0 import MiniCPMO45Stage0DuplexRuntime

    thinker = _Thinker()
    thinker.config.duplex_audio_encoder_batching = batching
    runtime = MiniCPMO45Stage0DuplexRuntime.__new__(MiniCPMO45Stage0DuplexRuntime)
    runtime.stage_model = _StageModel(thinker)
    runtime.thinker = thinker
    thinker.llm = SimpleNamespace(model=SimpleNamespace(embed_tokens=runtime.stage_model.embed))
    runtime.tokenizer = SimpleNamespace(
        unk_token_id=0,
        convert_tokens_to_ids=lambda token: {
            "<unit>": 1,
            "</unit>": 2,
            "<|listen|>": 3,
            "<|speak|>": 4,
            "<|tts_bos|>": 5,
            "<|tts_eos|>": 6,
            "<|tts_pad|>": 7,
            "<|chunk_eos|>": 8,
            "<|chunk_tts_eos|>": 9,
            "<|turn_eos|>": 10,
            "<|audio|>": 11,
        }.get(token, 0),
        encode=lambda text, add_special_tokens=False: [],
    )
    runtime.processor = _MelProcessor()
    runtime.device = "cpu"
    runtime._init_token_ids()
    return runtime


def _audio(seed: int, samples: int) -> np.ndarray:
    return np.random.default_rng(seed).standard_normal(samples).astype(np.float32)


@pytest.mark.parametrize("units_in_first_append", [1, 2])
def test_stage0_batched_prefill_matches_per_session_prefill(units_in_first_append: int) -> None:
    from vllm_omni.model_executor.models.minicpmo_4_5.duplex.stage0 import _MiniCPMO45Stage0SessionState

    sequential, batched = _runtime(batching=False), _runtime(batching=True)
    assert sequential._batched_audio_encoder() is None
    assert batched._batched_audio_encoder() is not None
    states = {
        name: [_MiniCPMO45Stage0SessionState(session_id=f"s{i}") for i in range(3)]
        for name in ("sequential", "batched")
    }
    # The runner's preprocess runs under inference mode; so do both paths here.
    with torch.inference_mode():
        for seq in range(1, 5):
            appends = [
                (index, _audio(10 * seq + index, 240 * (units_in_first_append if seq == 1 else 1)))
                for index in range(3)
                if not (seq == 3 and index == 1)  # session 1 sends nothing this step
            ]
            expected = [
                sequential._stage_prefill_embeddings_only(states["sequential"][index], audio, seq=seq, epoch=0)
                for index, audio in appends
            ]
            batch = [(states["batched"][index], audio, {"seq": seq, "epoch": 0}) for index, audio in appends]
            batched.stage_prefill_batch(batch)
            for (index, audio), reference in zip(appends, expected, strict=True):
                state = states["batched"][index]
                assert state.staged_prefill is not None
                result = batched._stage_prefill_embeddings_only(state, audio, seq=seq, epoch=0)
                assert state.staged_prefill is None
                assert result["success"] is reference["success"] is True
                assert result["input_token_ids"] == reference["input_token_ids"]
                torch.testing.assert_close(result["inputs_embeds"], reference["inputs_embeds"], rtol=1e-5, atol=1e-5)
                assert isinstance(state.audio_past_key_values, StreamingAudioKVCache)
                assert not batched.needs_prefill(state, 0, seq)
    # Every append's processor ran exactly as often on both paths.
    for mine, theirs in zip(states["batched"], states["sequential"], strict=True):
        assert mine.streaming_processor.calls == theirs.streaming_processor.calls > 0


def test_stage0_staged_result_is_only_taken_by_its_own_append() -> None:
    from vllm_omni.model_executor.models.minicpmo_4_5.duplex.stage0 import _MiniCPMO45Stage0SessionState

    runtime = _runtime(batching=True)
    a, b = (_MiniCPMO45Stage0SessionState(session_id=name) for name in "ab")
    runtime.stage_prefill_batch(
        [(a, _audio(1, 240), {"seq": 1, "epoch": 0}), (b, _audio(2, 240), {"seq": 1, "epoch": 0})]
    )
    staged = a.staged_prefill[1]
    assert runtime._stage_prefill_embeddings_only(a, _audio(1, 240), seq=1, epoch=0) is staged
    # A stale staged result (another identity) is dropped, not returned.
    b.staged_prefill = ((0, 9), {"success": False})
    result = runtime._stage_prefill_embeddings_only(b, _audio(3, 240), seq=2, epoch=0)
    assert result["success"] is True and b.staged_prefill is None
    with pytest.raises(ValueError, match="one append per session"):
        runtime.stage_prefill_batch([(a, _audio(4, 240), {"seq": 3}), (a, _audio(5, 240), {"seq": 4})])


def test_preprocess_batch_stages_only_new_appends_of_distinct_sessions() -> None:
    from vllm_omni.model_executor.models.minicpmo_4_5.minicpmo_4_5_omni import (
        MiniCPMO45OmniForConditionalGeneration,
    )

    staged: list[list[tuple[object, dict]]] = []
    prepared = {("done", 0, 1)}
    sessions: dict[str, object] = {}
    helper = SimpleNamespace(
        sessions=sessions,
        batches_audio_encoder=lambda: True,
        needs_prefill=lambda state, epoch, seq: (getattr(state, "session_id", None), epoch, seq) not in prepared,
        stage_prefill_batch=lambda appends: staged.append([(state, kwargs) for state, _, kwargs in appends]),
        _decode_audio_payload=lambda payload: np.zeros(4, dtype=np.float32),
        _decode_video_frames_payload=lambda payload: [],
        prefetch_vision=lambda appends: None,
        take_prefetched_vision=lambda duplex: None,
    )
    model = MiniCPMO45OmniForConditionalGeneration.__new__(MiniCPMO45OmniForConditionalGeneration)
    model.model_stage = "llm"
    model._minicpmo45_duplex_data_plane_helper = helper
    model._minicpmo45_duplex_session_state = lambda helper, session_id, duplex: sessions.setdefault(
        session_id, SimpleNamespace(session_id=session_id)
    )
    sessions["done"] = SimpleNamespace(session_id="done")

    def append(session_id: str, seq: object) -> dict:
        return {"duplex": {"data_plane": True, "session_id": session_id, "epoch": 0, "seq": seq, "payload": {}}}

    buffer = {
        "r1": append("a", 1),
        "r2": append("b", 5),
        "r3": append("a", 2),  # second append of "a": left to preprocess
        "r4": append("done", 1),  # already built
        "r5": append("c", None),  # no identity
        "r6": {"duplex": {"data_plane": False}},
        "r7": append("d", 1),
    }
    model.preprocess_batch(req_ids=list(buffer), model_intermediate_buffer=buffer, device=torch.device("cpu"))
    assert [[(state.session_id, kwargs["seq"]) for state, kwargs in call] for call in staged] == [
        [("a", 1), ("b", 5), ("d", 1)]
    ]

    staged.clear()
    single = {"r1": append("e", 1), "r2": append("done", 1)}
    model.preprocess_batch(req_ids=list(single), model_intermediate_buffer=single, device=torch.device("cpu"))
    assert staged == []  # one new append: preprocess builds it itself


def test_stage0_hands_a_batched_history_to_the_per_session_path() -> None:
    from vllm_omni.model_executor.models.minicpmo_4_5.duplex.stage0 import (
        _MiniCPMO45AudioEncodeRequest,
        _MiniCPMO45Stage0SessionState,
    )

    runtime = _runtime(batching=True)
    reference = _runtime(batching=False)
    state = _MiniCPMO45Stage0SessionState(session_id="s")
    expected = _MiniCPMO45Stage0SessionState(session_id="s")
    with torch.inference_mode():
        for index, frames in enumerate([22, 24]):
            mel = _mel(frames, index)
            feature = _Feature(audio_features=mel, audio_feature_lens=[torch.tensor([frames])])
            feature.chunk_idx = index
            (embeds,) = runtime._stage_audio_embeddings_batch([_MiniCPMO45AudioEncodeRequest(feature, state)])
            (reference_embeds,) = reference._stage_audio_embeddings_batch(
                [_MiniCPMO45AudioEncodeRequest(feature, expected)]
            )
            torch.testing.assert_close(embeds, reference_embeds, rtol=1e-5, atol=1e-5)
        assert isinstance(state.audio_past_key_values, StreamingAudioKVCache)
        # The per-session path taking a unit of a batched session continues its history.
        feature = _Feature(audio_features=_mel(24, 9), audio_feature_lens=torch.tensor([[24]]))
        feature.chunk_idx = 2
        converted = runtime._stage_audio_embeddings(feature, state=state)
        continued = reference._stage_audio_embeddings(feature, state=expected)
    torch.testing.assert_close(converted, continued, rtol=1e-5, atol=1e-5)
    assert not isinstance(state.audio_past_key_values, StreamingAudioKVCache)


def test_preprocess_batch_leaves_everything_per_request_without_a_batched_encoder() -> None:
    from vllm_omni.model_executor.models.minicpmo_4_5.minicpmo_4_5_omni import (
        MiniCPMO45OmniForConditionalGeneration,
    )

    def unexpected(*args, **kwargs):
        raise AssertionError("preprocess_batch staged work with batching disabled")

    model = MiniCPMO45OmniForConditionalGeneration.__new__(MiniCPMO45OmniForConditionalGeneration)
    model.model_stage = "llm"
    model._minicpmo45_duplex_data_plane_helper = SimpleNamespace(
        batches_audio_encoder=lambda: False,
        needs_prefill=unexpected,
        stage_prefill_batch=unexpected,
        prefetch_vision=lambda appends: None,
    )
    buffer = {
        rid: {"duplex": {"data_plane": True, "session_id": rid, "epoch": 0, "seq": 1, "payload": {}}}
        for rid in ("r1", "r2")
    }
    model.preprocess_batch(req_ids=["r1", "r2"], model_intermediate_buffer=buffer, device=torch.device("cpu"))

    runtime = _runtime(batching=False)
    assert not runtime.batches_audio_encoder()


def test_preprocess_batch_hands_prefetched_frames_to_the_batched_prefill() -> None:
    from vllm_omni.model_executor.models.minicpmo_4_5.minicpmo_4_5_omni import (
        MiniCPMO45OmniForConditionalGeneration,
    )

    staged: list[list[dict]] = []
    prefetched: dict[str, list] = {}
    sessions: dict[str, object] = {}

    def undecodable(payload):
        raise AssertionError("frames prefetched this step were decoded again")

    helper = SimpleNamespace(
        sessions=sessions,
        batches_audio_encoder=lambda: True,
        needs_prefill=lambda state, epoch, seq: True,
        stage_prefill_batch=lambda appends: staged.append([kwargs for _, _, kwargs in appends]),
        _decode_audio_payload=lambda payload: np.zeros(4, dtype=np.float32),
        _decode_video_frames_payload=undecodable,
        prefetch_vision=lambda appends: prefetched.update(
            {a["session_id"]: [["emb", a["session_id"]]] for a in appends}
        ),
        take_prefetched_vision=lambda duplex: prefetched.pop(duplex["session_id"], None),
    )
    model = MiniCPMO45OmniForConditionalGeneration.__new__(MiniCPMO45OmniForConditionalGeneration)
    model.model_stage = "llm"
    model._minicpmo45_duplex_data_plane_helper = helper
    model._duplex_data_plane_helper = lambda: helper
    model._minicpmo45_duplex_session_state = lambda helper, session_id, duplex: sessions.setdefault(
        session_id, SimpleNamespace(session_id=session_id)
    )
    buffer = {
        rid: {
            "duplex": {
                "data_plane": True,
                "session_id": rid,
                "epoch": 0,
                "seq": 1,
                "payload": {"video_frames": ["jpeg"]},
            }
        }
        for rid in ("a", "b")
    }
    model.preprocess_batch(req_ids=list(buffer), model_intermediate_buffer=buffer, device=torch.device("cpu"))
    assert [[kwargs.get("encoded_frames") for kwargs in call] for call in staged] == [[[["emb", "a"]], [["emb", "b"]]]]
    assert all("video_frames" not in kwargs for kwargs in staged[0])
    assert prefetched == {}
