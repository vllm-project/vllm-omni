# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
import re
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import yaml

from tests.helpers.stage_config import get_deploy_config_path
from vllm_omni.model_executor.models.chatterbox.chatterbox_s3gen import (
    LEFT_CONTEXT_TOKENS,
    SAMPLES_PER_FRAME,
    SOURCE_CACHE_SAMPLES,
    ChatterboxS3Gen,
    Chunk,
    Reference,
    S3GenDecoder,
    StreamState,
    fade_in_out,
    flow_mels,
    window,
)
from vllm_omni.transformers_utils.configs.chatterbox import ChatterboxConfig

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

LOOKAHEAD = 3
HOPS = [20, 30, 60]


def chunk_plan(total: int) -> list[tuple[int, int, bool]]:
    """(cumulative prefix length, token offset, finalize) per chunk.

    What CosyVoice3's async-chunk processor emits for ``total`` tokens with
    the deploy file's chunk sizes and a 250-token reference (first hop 20).
    """
    plan = []
    emitted = 0
    hops = iter(HOPS)
    hop = next(hops)
    while True:
        needed = emitted + hop + LOOKAHEAD
        if needed >= total:
            plan.append((total, emitted, True))
            return plan
        plan.append((needed, emitted, False))
        emitted += hop
        hop = next(hops, 60)


def reference(prompt_tokens: int) -> Reference:
    return Reference(
        prompt_token=torch.randint(0, 6561, (1, prompt_tokens)),
        prompt_feat=torch.randn(1, 2 * prompt_tokens, 80),
        embedding=torch.randn(1, 192),
    )


def stream_payload(ref: Reference, finished: bool, offset: int) -> dict:
    """One request's merged payload as the runner hands it to stage 1."""
    return {
        "meta": {
            "finished": torch.tensor(finished),
            "stream_finished": torch.tensor(finished),
            "req_id": ["external-id"],
            "left_context_size": offset,
        },
        "embed": {"speech_token": ref.prompt_token, "speech_feat": ref.prompt_feat, "embedding": ref.embedding},
        "generated_len": 0,
    }


@pytest.fixture(scope="module")
def decoder() -> S3GenDecoder:
    torch.manual_seed(0)
    return S3GenDecoder(ChatterboxConfig()).eval()


@pytest.fixture(scope="module")
def references() -> list[Reference]:
    torch.manual_seed(1)
    # Unequal lengths on purpose: a clip shorter than ten seconds gives a
    # shorter prompt, and rows of one batch need not agree.
    return [reference(20), reference(32)]


def test_fade_in_out_blends_the_overlap_001() -> None:
    win = torch.hamming_window(8, periodic=False)
    blended = fade_in_out(torch.ones(1, 8), torch.zeros(1, 4), win)
    assert torch.allclose(blended[0, :4], win[:4])
    assert torch.equal(blended[0, 4:], torch.ones(4))


def test_window_keeps_the_left_context_and_rebases_the_offset_001() -> None:
    tokens = torch.arange(200)
    kept, offset = window(tokens, 120)
    assert torch.equal(kept, tokens[120 - LEFT_CONTEXT_TOKENS :])
    assert offset == LEFT_CONTEXT_TOKENS
    kept, offset = window(tokens, 10)
    assert torch.equal(kept, tokens)
    assert offset == 10


def test_batch_rows_match_single_rows_with_different_references_001(
    decoder: S3GenDecoder, references: list[Reference]
) -> None:
    """Three requests in one flow call, two voices, unequal lengths.

    Each row must decode as it does alone. The noise is pinned over the whole
    input, reference frames included, because the flow draws whatever the
    caller leaves unpinned and the two calls would then differ by the draw.
    """
    torch.manual_seed(2)
    tokens = [torch.randint(0, 6561, (n,)) for n in (23, 38, 17)]
    refs = [references[0], references[1], references[0]]
    finalize = [False, False, True]
    lengths = [ref.prompt_token.shape[1] + row.numel() for ref, row in zip(refs, tokens, strict=True)]
    noise = torch.randn(3, 80, 2 * max(lengths))

    batched = flow_mels(decoder.flow, tokens, refs, finalize, 2, True, noise)
    for i in range(3):
        (alone,) = flow_mels(
            decoder.flow, [tokens[i]], [refs[i]], [finalize[i]], 2, True, noise[i : i + 1, :, : 2 * lengths[i]]
        )
        frames = 2 * tokens[i].numel() - (0 if finalize[i] else 2 * LOOKAHEAD)
        assert batched[i].shape == alone.shape == (1, 80, frames)
        assert torch.allclose(batched[i], alone, atol=1e-3, rtol=1e-3), (batched[i] - alone).abs().max()

    # Not vacuous: the same tokens and noise under another voice of the
    # same length give another mel.
    (other_voice,) = flow_mels(
        decoder.flow, [tokens[0]], [reference(20)], [False], 2, True, noise[:1, :, : 2 * lengths[0]]
    )
    assert other_voice.shape == batched[0].shape
    assert not torch.allclose(other_voice, batched[0], atol=1e-2)


def test_non_final_chunk_holds_back_the_overlap_001(decoder: S3GenDecoder, references: list[Reference]) -> None:
    tokens = torch.randint(0, 6561, (23,))
    ((piece, state),) = decoder.chunked_decode_streaming([Chunk(tokens, 0, references[0], None, False)])
    assert piece.shape == (1, 40 * SAMPLES_PER_FRAME - SOURCE_CACHE_SAMPLES)
    assert state is not None
    assert state.mel.shape == (1, 80, 8)
    assert state.source.shape == (1, 1, SOURCE_CACHE_SAMPLES)
    assert state.speech.shape == (1, SOURCE_CACHE_SAMPLES)


def test_chunked_and_one_shot_agree_in_length_001(decoder: S3GenDecoder, references: list[Reference]) -> None:
    """Streaming must neither drop nor repeat a frame, whatever the chunking."""
    tokens = torch.randint(0, 6561, (140,))
    ((whole, final_state),) = decoder.chunked_decode_streaming([Chunk(tokens, 0, references[0], None, True)])
    assert final_state is None
    assert whole.shape == (1, 2 * (140 + 3) * SAMPLES_PER_FRAME)

    pieces = []
    state: StreamState | None = None
    for prefix, offset, finalize in chunk_plan(140):
        ((piece, state),) = decoder.chunked_decode_streaming(
            [Chunk(tokens[:prefix], offset, references[0], state, finalize)]
        )
        pieces.append(piece)
    streamed = torch.cat(pieces, dim=1)

    assert state is None
    assert streamed.shape == whole.shape
    assert torch.isfinite(streamed).all()


def test_decode_step_keeps_state_until_the_stream_finishes_001(
    decoder: S3GenDecoder, references: list[Reference]
) -> None:
    tokens = torch.randint(0, 6561, (60,))
    (first,) = decoder.decode_step(tokens[:23], [23], [stream_payload(references[0], False, 0)], ["scheduler-id"])
    assert first.shape == (40 * SAMPLES_PER_FRAME - SOURCE_CACHE_SAMPLES,)
    assert first.dtype == torch.float32
    # Keyed by the scheduler's id, the one on_requests_finished is given, not
    # by the external id in the payload.
    assert set(decoder.streams) == {"scheduler-id"}

    (last,) = decoder.decode_step(tokens, [60], [stream_payload(references[0], True, 20)], ["scheduler-id"])
    assert first.numel() + last.numel() == 2 * (60 + 3) * SAMPLES_PER_FRAME
    assert decoder.streams == {}


def test_abort_frees_stream_state_001(decoder: S3GenDecoder, references: list[Reference]) -> None:
    """A client that disconnects never sends a final chunk."""
    tokens = torch.randint(0, 6561, (23,))
    decoder.decode_step(tokens, [23], [stream_payload(references[0], False, 0)], ["aborted"])
    assert "aborted" in decoder.streams
    decoder.on_requests_finished({"aborted", "never-seen"})
    assert decoder.streams == {}


def test_two_requests_in_one_step_keep_separate_state_001(decoder: S3GenDecoder, references: list[Reference]) -> None:
    tokens = torch.randint(0, 6561, (23 + 30,))
    audio = decoder.decode_step(
        tokens,
        [23, 30],
        [stream_payload(references[0], False, 0), stream_payload(references[1], True, 0)],
        ["a", "b"],
    )
    assert audio[0].shape == (40 * SAMPLES_PER_FRAME - SOURCE_CACHE_SAMPLES,)
    assert audio[1].shape == (2 * (30 + 3) * SAMPLES_PER_FRAME,)
    assert set(decoder.streams) == {"a"}
    decoder.on_requests_finished({"a"})


def test_spans_without_a_payload_or_tokens_give_empty_audio_001(
    decoder: S3GenDecoder, references: list[Reference]
) -> None:
    """The profiling run has no payloads; the terminal payload has no tokens."""
    (profiling,) = decoder.decode_step(torch.zeros(16, dtype=torch.long), [16], None, None)
    assert profiling.shape == (0,)
    (terminal,) = decoder.decode_step(
        torch.zeros(0, dtype=torch.long), [0], [{"meta": {"finished": torch.tensor(True)}}], ["done"]
    )
    assert terminal.shape == (0,)


def test_tokens_without_stream_metadata_are_refused_001(decoder: S3GenDecoder, references: list[Reference]) -> None:
    """Decoding them as a whole utterance would play wrong audio with no error."""
    payload = stream_payload(references[0], False, 0)
    del payload["meta"]
    with pytest.raises(RuntimeError, match="stream metadata"):
        decoder.decode_step(torch.randint(0, 6561, (23,)), [23], [payload], ["x"])


def test_batch_mode_payload_decodes_the_whole_utterance_001(decoder: S3GenDecoder, references: list[Reference]) -> None:
    """What the batch-mode processor sends: a stream of one final chunk."""
    payload = {
        "embed": stream_payload(references[1], True, 0)["embed"],
        "meta": {"stream_finished": torch.tensor(True), "left_context_size": 0},
    }
    (audio,) = decoder.decode_step(torch.randint(0, 6561, (40,)), [40], [payload], ["x"])
    assert audio.shape == (2 * (40 + 3) * SAMPLES_PER_FRAME,)
    assert decoder.streams == {}


def test_decoding_a_chunk_twice_gives_the_same_mel_001(decoder: S3GenDecoder, references: list[Reference]) -> None:
    """The flow draws nothing: its noise comes by position from a fixed buffer.

    The audio is the same only once the vocoder's own draw is pinned: HiFT's
    source module draws phase and noise on every call, as upstream's does.
    """
    chunk = Chunk(torch.randint(0, 6561, (53,)), 20, references[0], None, False)
    (first,) = decoder.chunk_mels([chunk])
    (again,) = decoder.chunk_mels([chunk])
    assert first.shape == (1, 80, 60)
    assert torch.equal(first, again)

    audio = []
    for _ in range(2):
        torch.manual_seed(3)
        ((piece, _),) = decoder.chunked_decode_streaming([chunk])
        audio.append(piece)
    assert torch.equal(audio[0], audio[1])


def test_a_row_does_not_depend_on_its_batch_001(decoder: S3GenDecoder, references: list[Reference]) -> None:
    """Nothing is pinned here: each row's noise is fixed by its own positions."""
    tokens = torch.randint(0, 6561, (113,))
    chunks = [
        Chunk(tokens[:53], 20, references[0], None, False),
        Chunk(tokens, 50, references[1], None, True),
        Chunk(tokens[:23], 0, references[0], None, False),
    ]
    batched = decoder.chunk_mels(chunks)
    assert [mel.shape[2] for mel in batched] == [60, 2 * (63 + 3), 40]
    for chunk, mel in zip(chunks, batched, strict=True):
        (alone,) = decoder.chunk_mels([chunk])
        assert torch.allclose(mel, alone, atol=1e-3, rtol=1e-3), (mel - alone).abs().max()


def test_noise_is_taken_by_position_for_every_prompt_length_001(
    decoder: S3GenDecoder, references: list[Reference]
) -> None:
    """A row is the prompt region from frame 0, then the tokens' own frames."""
    tokens = torch.randint(0, 6561, (113,))
    # Seventy tokens played: the window keeps tokens[20:], fifty of them context.
    kept = tokens[20:]
    start = decoder.prompt_noise_frames + 2 * 20
    for ref in references:
        prompt = 2 * ref.prompt_token.shape[1]
        (mel,) = decoder.chunk_mels([Chunk(tokens, 70, ref, None, False)])

        def decoded_from(utterance_start: int) -> torch.Tensor:
            noise = torch.cat(
                [
                    decoder.flow_noise[:, :, :prompt],
                    decoder.flow_noise[:, :, utterance_start : utterance_start + 2 * kept.numel()],
                ],
                dim=2,
            )
            (whole,) = flow_mels(decoder.flow, [kept], [ref], [False], 2, True, noise)
            return whole[:, :, 2 * LEFT_CONTEXT_TOKENS :]

        assert torch.equal(mel, decoded_from(start))
        # Not vacuous: the same row placed at the start of the utterance differs.
        assert not torch.allclose(mel, decoded_from(decoder.prompt_noise_frames), atol=1e-2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_noise_is_the_same_draw_when_built_on_a_default_device_001(decoder: S3GenDecoder) -> None:
    """vLLM builds the stage inside a default-device context."""
    with torch.device("cuda"):
        built = S3GenDecoder(ChatterboxConfig())
    assert built.flow_noise.device.type == built.trim_fade.device.type == "cuda"
    assert torch.equal(built.flow_noise.cpu(), decoder.flow_noise)


def test_chunked_mel_is_closer_to_one_chunk_than_a_noise_redraw_001(
    decoder: S3GenDecoder, references: list[Reference]
) -> None:
    """Chunks and one final chunk share noise, so they differ by context only."""
    tokens = torch.randint(0, 6561, (140,))
    (whole,) = decoder.chunk_mels([Chunk(tokens, 0, references[0], None, True)])
    stitched = torch.cat(
        [
            decoder.chunk_mels([Chunk(tokens[:prefix], offset, references[0], None, finalize)])[0]
            for prefix, offset, finalize in chunk_plan(140)
        ],
        dim=2,
    )
    assert stitched.shape == whole.shape == (1, 80, 2 * (140 + 3))

    (redrawn,) = flow_mels(
        decoder.flow,
        [torch.cat([tokens, decoder.silence])],
        [references[0]],
        [True],
        2,
        True,
        torch.randn(1, 80, 2 * (references[0].prompt_token.shape[1] + 140 + 3)),
    )
    assert (stitched - whole).abs().mean() < (redrawn - whole).abs().mean()


def test_a_non_final_chunk_shorter_than_the_mel_cache_is_refused_001(
    decoder: S3GenDecoder, references: list[Reference]
) -> None:
    """Three new tokens are six mel frames, and the cross-fade caches eight."""
    tokens = torch.randint(0, 6561, (3 + LOOKAHEAD,))
    with pytest.raises(RuntimeError, match="codec_chunk_frames must be at least 4"):
        decoder.chunked_decode_streaming([Chunk(tokens, 0, references[0], None, False)])


def test_tokens_past_the_noise_buffer_are_refused_001(decoder: S3GenDecoder, references: list[Reference]) -> None:
    """The buffer covers ``max_new_tokens`` and the silence; nothing wraps."""
    tokens = torch.randint(0, 6561, (1001,))
    with pytest.raises(RuntimeError, match="1004 speech tokens after a 20-token reference.*holds 1003 and 250"):
        decoder.chunked_decode_streaming([Chunk(tokens, 960, references[0], None, True)])


def test_a_row_is_vocoded_as_it_is_alone_whatever_shares_the_step_001(decoder: S3GenDecoder) -> None:
    """Rows of unequal length never share a vocoder call, so no row is padded.

    Seeded because HiFT's source module draws on every call. The first row's
    call comes first either way, so it sees the same draw.
    """
    short, longer = torch.randn(1, 80, 40), torch.randn(1, 80, 68)
    torch.manual_seed(5)
    (alone,), (alone_source,) = decoder.vocode([short], torch.zeros(1, 1, 0))
    torch.manual_seed(5)
    wavs, sources = decoder.vocode([short, longer], torch.zeros(2, 1, 0))
    assert torch.equal(wavs[0], alone)
    assert torch.equal(sources[0], alone_source)
    assert wavs[1].shape == (1, 68 * SAMPLES_PER_FRAME)
    assert sources[1].shape == (1, 1, 68 * SAMPLES_PER_FRAME)


def test_rows_of_equal_length_share_one_vocoder_call_001(decoder: S3GenDecoder) -> None:
    """Nothing is padded between them, so batching them costs no independence."""
    mels = [torch.randn(1, 80, 68), torch.randn(1, 80, 40), torch.randn(1, 80, 68)]
    cache = torch.randn(3, 1, SOURCE_CACHE_SAMPLES)
    torch.manual_seed(5)
    wavs, sources = decoder.vocode(mels, cache)
    torch.manual_seed(5)
    speech, source = decoder.mel2wav.inference(speech_feat=torch.cat([mels[0], mels[2]]), cache_source=cache[[0, 2]])
    assert torch.equal(torch.cat([wavs[0], wavs[2]]), speech)
    assert torch.equal(torch.cat([sources[0], sources[2]]), source)
    # Each row keeps its own source cache.
    assert all(torch.equal(sources[i][:, :, :SOURCE_CACHE_SAMPLES], cache[i : i + 1]) for i in range(3))


def test_two_live_streams_in_one_step_use_their_own_state_001(
    decoder: S3GenDecoder, references: list[Reference]
) -> None:
    """Cached mel, source cache and cross-fade tail are paired with their row.

    The tails are constants of opposite sign: a cross-fade starts at 0.08 of
    the new audio, which HiFT clamps below 1, plus all of the tail, so a
    row's first sample has its own tail's sign. Then one row's cached mel
    and source are changed under a pinned vocoder draw: that row's audio
    must change and the other row's must not.
    """
    tokens = [torch.randint(0, 6561, (53,)) for _ in range(2)]
    refs = [references[0], references[1]]
    states = []
    for row, ref, level in zip(tokens, refs, (0.5, -0.5), strict=True):
        ((_, state),) = decoder.chunked_decode_streaming([Chunk(row[:23], 0, ref, None, False)])
        states.append(replace(state, speech=torch.full((1, SOURCE_CACHE_SAMPLES), level)))

    def step(state_a: StreamState, state_b: StreamState) -> list[torch.Tensor]:
        torch.manual_seed(7)
        results = decoder.chunked_decode_streaming(
            [Chunk(tokens[0], 20, refs[0], state_a, False), Chunk(tokens[1], 20, refs[1], state_b, False)]
        )
        return [piece for piece, _ in results]

    a, b = step(states[0], states[1])
    assert a.shape == b.shape == (1, (8 + 60) * SAMPLES_PER_FRAME - SOURCE_CACHE_SAMPLES)
    assert a[0, 0] > 0.4
    assert b[0, 0] < -0.4

    other = [replace(state, mel=state.mel.flip(2), source=-state.source) for state in states]
    changed_a, same_b = step(other[0], states[1])
    assert torch.equal(same_b, b)
    assert not torch.allclose(changed_a[:, :SOURCE_CACHE_SAMPLES], a[:, :SOURCE_CACHE_SAMPLES], atol=1e-3)
    same_a, changed_b = step(states[0], other[1])
    assert torch.equal(same_a, a)
    assert not torch.allclose(changed_b[:, :SOURCE_CACHE_SAMPLES], b[:, :SOURCE_CACHE_SAMPLES], atol=1e-3)


def test_a_live_stream_and_a_first_chunk_in_one_step_001(decoder: S3GenDecoder, references: list[Reference]) -> None:
    """The first chunk gets the fade-in and no cross-fade; the live stream the reverse."""
    tokens = torch.randint(0, 6561, (53,))
    ((_, state),) = decoder.chunked_decode_streaming([Chunk(tokens[:23], 0, references[0], None, False)])
    state = replace(state, speech=torch.full((1, SOURCE_CACHE_SAMPLES), 0.5))
    (first, first_state), (live, live_state) = decoder.chunked_decode_streaming(
        [Chunk(tokens[:23], 0, references[1], None, False), Chunk(tokens, 20, references[0], state, False)]
    )
    assert first.shape == (1, 40 * SAMPLES_PER_FRAME - SOURCE_CACHE_SAMPLES)
    assert live.shape == (1, (8 + 60) * SAMPLES_PER_FRAME - SOURCE_CACHE_SAMPLES)
    # Upstream's trim: the first 20 ms of an utterance are silenced.
    assert torch.count_nonzero(first[0, :SAMPLES_PER_FRAME]) == 0
    assert live[0, 0] > 0.4
    assert first_state is not None and live_state is not None
    assert first_state.speech.shape == live_state.speech.shape == (1, SOURCE_CACHE_SAMPLES)


def test_tokens_with_an_empty_payload_are_refused_001(decoder: S3GenDecoder) -> None:
    """What the runner passes for a request it holds no payload for."""
    with pytest.raises(RuntimeError, match="23 tokens for request x without a reference or stream metadata"):
        decoder.decode_step(torch.randint(0, 6561, (23,)), [23], [{}], ["x"])


def test_tokens_without_a_payload_list_are_refused_001(decoder: S3GenDecoder) -> None:
    """Only the profiling run, which has no request ids, comes without payloads."""
    with pytest.raises(RuntimeError, match="23 tokens for request x without a reference or stream metadata"):
        decoder.decode_step(torch.randint(0, 6561, (23,)), [23], None, ["x"])


def test_token_offset_and_stream_state_must_agree_001(decoder: S3GenDecoder, references: list[Reference]) -> None:
    """Either mismatch would play a chunk with the wrong seam and no error."""
    tokens = torch.randint(0, 6561, (53,))
    with pytest.raises(RuntimeError, match="token offset 20 for request lost, which has no earlier chunk"):
        decoder.decode_step(tokens, [53], [stream_payload(references[0], False, 20)], ["lost"])

    decoder.decode_step(tokens[:23], [23], [stream_payload(references[0], False, 0)], ["restarted"])
    with pytest.raises(RuntimeError, match="token offset 0 for request restarted, which already has an earlier chunk"):
        decoder.decode_step(tokens[:23], [23], [stream_payload(references[0], False, 0)], ["restarted"])
    decoder.on_requests_finished({"restarted"})
    assert decoder.streams == {}


def stage_config(
    max_num_seqs: int, max_num_batched_tokens: int, max_num_scheduled_tokens: int | None = None
) -> SimpleNamespace:
    """The fields of the engine's config that the stage reads."""
    return SimpleNamespace(
        model_config=SimpleNamespace(hf_config=ChatterboxConfig(), max_model_len=2048),
        scheduler_config=SimpleNamespace(
            max_num_seqs=max_num_seqs,
            max_num_batched_tokens=max_num_batched_tokens,
            max_num_scheduled_tokens=max_num_scheduled_tokens,
        ),
    )


@pytest.mark.parametrize(
    ("max_num_batched_tokens", "max_num_scheduled_tokens", "names"),
    [
        (16383, None, "max_num_batched_tokens >= max_num_seqs (8) * max_model_len (2048) = 16384, got 16383"),
        (16384, 8000, "max_num_scheduled_tokens >= max_num_seqs (8) * max_model_len (2048) = 16384, got 8000"),
        (16384, 0, "max_num_scheduled_tokens >= max_num_seqs (8) * max_model_len (2048) = 16384, got 0"),
    ],
    ids=["batched tokens", "scheduled tokens", "scheduled tokens set to 0"],
)
def test_a_step_budget_that_can_split_an_utterance_is_refused_at_startup_001(
    max_num_batched_tokens: int, max_num_scheduled_tokens: int | None, names: str
) -> None:
    """A new request the step's budget cannot cover is scheduled with part of its tokens.

    The decoder would take that part for the whole utterance. An utterance is
    at most the stage's own context long, so eight slots of a 2048-token
    context need 16384, whatever ``max_tokens`` a request asks for. The
    scheduler's budget is its scheduled-token limit whenever one is set, 0
    included, and the message names the field it read.
    """
    with pytest.raises(ValueError, match=re.escape(names)):
        ChatterboxS3Gen(vllm_config=stage_config(8, max_num_batched_tokens, max_num_scheduled_tokens))


def test_the_stage_builds_with_the_deploy_files_budget_001() -> None:
    decoder = yaml.safe_load(Path(get_deploy_config_path("chatterbox_turbo.yaml")).read_text())["stages"][1]
    config = stage_config(decoder["max_num_seqs"], decoder["max_num_batched_tokens"])
    config.model_config.max_model_len = decoder["max_model_len"]

    stage = ChatterboxS3Gen(vllm_config=config)

    assert stage.allow_patterns_overrides == [ChatterboxConfig().s3gen_weights]
