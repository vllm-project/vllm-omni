# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
import re
from collections.abc import Iterator
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import yaml
from torch import nn
from vllm.sampling_params import SamplingParams

from tests.helpers.stage_config import get_deploy_config_path
from vllm_omni.config.stage_config import resolve_deploy_yaml
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
# Stage 1's context in the deploy file: the longest utterance a decoder is built for.
CONTEXT = 2048
# Batched against alone, CPU float32. Comparable models use 1e-5 on CPU.
TOLERANCE = 1e-5


def chunk_plan(total: int) -> list[tuple[int, int, bool]]:
    """(cumulative prefix length, token offset, finalize) per chunk.

    What the chunk processor emits for ``total`` tokens with the deploy
    file's chunk sizes and a 250-token reference (first hop 20).
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


def stream_payload(ref: Reference | None, finished: bool, offset: int) -> dict:
    """One request's payload as the runner hands it to stage 1, with or without the voice.

    The runner's own per-request keys share the dict, as on Model Runner V2.
    """
    payload: dict = {
        "meta": {
            "finished": torch.tensor(finished),
            "stream_finished": torch.tensor(finished),
            "left_context_size": offset,
        },
        "generated_len": 0,
        "req_id": "scheduler-id",
        "global_request_id": ["external-id"],
        "sampling_params": SamplingParams(max_tokens=16),
    }
    if ref is not None:
        payload["embed"] = {
            "speech_token": ref.prompt_token,
            "speech_feat": ref.prompt_feat,
            "embedding": ref.embedding,
        }
    return payload


@pytest.fixture(scope="module")
def decoder() -> S3GenDecoder:
    torch.manual_seed(0)
    return S3GenDecoder(ChatterboxConfig(), CONTEXT).eval()


@pytest.fixture(autouse=True)
def no_stream_left_behind(decoder: S3GenDecoder) -> Iterator[None]:
    """A test that fails mid-stream must not leave its state to the next one."""
    yield
    decoder.streams.clear()


@pytest.fixture(scope="module")
def references() -> list[Reference]:
    torch.manual_seed(1)
    # Unequal lengths on purpose: a clip shorter than ten seconds gives a
    # shorter prompt, and rows of one batch need not agree.
    return [reference(20), reference(32)]


def test_fade_in_out_blends_the_overlap() -> None:
    """The new audio rises over the window's first half while the tail's last samples fall over its second."""
    win = torch.tensor([0.0, 0.25, 0.5, 0.75, 1.0, 0.75, 0.5, 0.25])
    # A tail longer than the overlap: only its last four samples are faded out.
    tail = torch.arange(6.0).reshape(1, 6)

    blended = fade_in_out(torch.full((1, 8), 2.0), tail, win)

    # 2 * win[i] + tail[0, 2 + i] * win[4 + i]
    assert blended.tolist() == [[2.0, 2.75, 3.0, 2.75, 2.0, 2.0, 2.0, 2.0]]


def test_bfloat16_flow_keeps_vocoder_and_audio_in_float32(decoder, monkeypatch):
    """Actual flow and HiFT calls enforce the mixed-precision boundary."""
    monkeypatch.setattr(decoder.config, "flow_dtype", "bfloat16")
    flow_dtypes, vocoder_dtypes = [], []
    flow_hook = decoder.flow.decoder.register_forward_hook(
        lambda module, inputs, output: flow_dtypes.append(output[0].dtype)
    )
    vocoder_hook = decoder.mel2wav.conv_pre.register_forward_pre_hook(
        lambda module, inputs: vocoder_dtypes.append(inputs[0].dtype)
    )
    try:
        chunks = [Chunk(torch.arange(15), 0, reference(2), None, True)]
        ((audio, state),) = decoder.chunked_decode_streaming(chunks)
        assert flow_dtypes == [torch.bfloat16]
        assert vocoder_dtypes == [torch.float32]
        assert audio.dtype == torch.float32
        assert torch.isfinite(audio).all()
        assert state is None
        assert next(decoder.flow.parameters()).dtype == torch.float32
    finally:
        flow_hook.remove()
        vocoder_hook.remove()


def test_cudnn_attention_rejects_cpu_execution(decoder, monkeypatch):
    monkeypatch.setattr(decoder.config, "flow_dtype", "bfloat16")
    monkeypatch.setattr(decoder.config, "flow_attention_backend", "cudnn")
    with pytest.raises(ValueError, match="requires CUDA"):
        decoder.chunk_mels([Chunk(torch.arange(15), 0, reference(2), None, True)])


def test_window_keeps_the_left_context_and_rebases_the_offset() -> None:
    tokens = torch.arange(200)
    kept, offset = window(tokens, 120)
    assert torch.equal(kept, tokens[120 - LEFT_CONTEXT_TOKENS :])
    assert offset == LEFT_CONTEXT_TOKENS
    kept, offset = window(tokens, 10)
    assert torch.equal(kept, tokens)
    assert offset == 10


def test_batch_rows_match_single_rows_with_different_references(
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
        # Measured on CPU float32: at most 5.3e-6 on values up to 5.1.
        torch.testing.assert_close(batched[i], alone, atol=TOLERANCE, rtol=TOLERANCE)

    # Not vacuous: the same tokens and noise under another voice of the
    # same length give another mel (measured: 1.1 apart).
    (other_voice,) = flow_mels(
        decoder.flow, [tokens[0]], [reference(20)], [False], 2, True, noise[:1, :, : 2 * lengths[0]]
    )
    assert other_voice.shape == batched[0].shape
    assert (other_voice - batched[0]).abs().max() > 0.1


def test_one_final_row_is_upstreams_flow_inference(decoder: S3GenDecoder, references: list[Reference]) -> None:
    """For one row with nothing to pad, ``flow_mels`` is ``CausalMaskedDiffWithXvec.inference`` (0.1.7).

    The reference is upstream's method written out over the same modules:
    prompt and tokens embedded under an all-ones mask, encoded, projected;
    the prompt's mel as the condition over its own frames; the generated
    frames after them returned. The non-final case differs from upstream on
    purpose (upstream crops the encoder output and keeps the uncropped mask)
    and has no reference.
    """
    flow, ref = decoder.flow, references[1]
    torch.manual_seed(12)
    tokens = torch.randint(0, 6561, (17,))
    length = ref.prompt_token.shape[1] + tokens.numel()
    noise = torch.randn(1, 80, 2 * length)

    with torch.inference_mode():
        (mel,) = flow_mels(flow, [tokens], [ref], [True], 2, True, noise)

        token = torch.cat([ref.prompt_token, tokens[None]], dim=1)
        embedding = flow.spk_embed_affine_layer(nn.functional.normalize(ref.embedding, dim=1))
        h = flow.encoder_proj(
            flow.encoder(flow.input_embedding(token) * torch.ones(1, length, 1), torch.tensor([length]))
        )
        prompt_frames = ref.prompt_feat.shape[1]
        conds = torch.zeros(1, 2 * length, 80)
        conds[:, :prompt_frames] = ref.prompt_feat
        feat, _ = flow.decoder(
            mu=h.transpose(1, 2).contiguous(),
            mask=torch.ones(1, 1, 2 * length),
            spks=embedding,
            cond=conds.transpose(1, 2),
            n_timesteps=2,
            noised_mels=noise,
            meanflow=True,
        )

    assert mel.shape == (1, 80, 2 * tokens.numel())
    assert torch.equal(mel, feat[:, :, prompt_frames:])


def test_the_encoder_is_upstreams_when_nothing_is_padded(decoder: S3GenDecoder) -> None:
    """``UpsampleConformerEncoder.forward`` zeroes padding after its input layer; without padding that is a no-op.

    The reference is upstream's ``forward`` (0.1.7) for this configuration,
    written out over the same layers: no padding zeroed, the padding masks
    as chunk masks.
    """
    encoder = decoder.flow.encoder
    torch.manual_seed(13)
    xs, lengths = torch.randn(2, 30, 512), torch.tensor([30, 30])

    with torch.inference_mode():
        encoded = encoder(xs, lengths)

        masks = torch.ones(2, 1, 30, dtype=torch.bool)
        h, pos_emb, masks = encoder.embed(xs, masks)
        h = encoder.pre_lookahead_layer(h)
        for layer in encoder.encoders:
            h, masks, _, _ = layer(h, masks, pos_emb, masks)
        h, _ = encoder.up_layer(h.transpose(1, 2).contiguous(), lengths)
        masks = torch.ones(2, 1, 60, dtype=torch.bool)
        h, pos_emb, masks = encoder.up_embed(h.transpose(1, 2).contiguous(), masks)
        for layer in encoder.up_encoders:
            h, masks, _, _ = layer(h, masks, pos_emb, masks)
        expected = encoder.after_norm(h)

    assert encoded.shape == (2, 60, 512)
    assert torch.equal(encoded, expected)


def test_a_padded_encoder_row_is_encoded_as_it_is_alone(decoder: S3GenDecoder) -> None:
    """The shorter rows' padding reaches none of their frames, through the look-ahead convolution included."""
    encoder = decoder.flow.encoder
    torch.manual_seed(14)
    lengths = torch.tensor([40, 31, 17])
    xs = torch.randn(3, 40, 512) * (torch.arange(40) < lengths[:, None]).unsqueeze(-1)

    with torch.inference_mode():
        batched = encoder(xs, lengths)
        for row, length in enumerate(lengths.tolist()):
            alone = encoder(xs[row : row + 1, :length], lengths[row : row + 1])
            # Measured on CPU float32: at most 4.3e-6 on values up to 4.
            torch.testing.assert_close(batched[row : row + 1, : 2 * length], alone, atol=TOLERANCE, rtol=TOLERANCE)


def test_non_final_chunk_holds_back_the_overlap(decoder: S3GenDecoder, references: list[Reference]) -> None:
    """The state is the chunk's last eight frames, their source and their samples, and they are not played.

    The flow is deterministic and the vocoder's own draw is pinned, so the
    chunk's mel and audio can be computed again and compared by value.
    """
    torch.manual_seed(4)
    chunk = Chunk(torch.randint(0, 6561, (23,)), 0, references[0], None, False)

    torch.manual_seed(5)
    ((piece, state),) = decoder.chunked_decode_streaming([chunk])
    torch.manual_seed(5)
    (mel,) = decoder.chunk_mels([chunk])
    (wav,), (source,) = decoder.vocode([mel], torch.zeros(1, 1, 0))

    assert mel.shape == (1, 80, 40)
    assert state is not None
    assert torch.equal(state.mel, mel[:, :, -8:])
    assert torch.equal(state.source, source[:, :, -SOURCE_CACHE_SAMPLES:])
    assert torch.equal(state.speech, wav[:, -SOURCE_CACHE_SAMPLES:])
    # Played: everything before the held-back samples, with upstream's 20 ms trim and fade-in.
    trim = decoder.trim_fade.numel()
    assert piece.shape == (1, 40 * SAMPLES_PER_FRAME - SOURCE_CACHE_SAMPLES)
    assert torch.equal(piece[:, :trim], wav[:, :trim] * decoder.trim_fade)
    assert torch.equal(piece[:, trim:], wav[:, trim:-SOURCE_CACHE_SAMPLES])


def test_chunked_and_one_shot_agree_in_length(decoder: S3GenDecoder, references: list[Reference]) -> None:
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


def test_decode_step_keeps_state_until_the_stream_finishes(decoder: S3GenDecoder, references: list[Reference]) -> None:
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


def decode_stream(
    decoder: S3GenDecoder, tokens: torch.Tensor, request_id: str, ref: Reference, later: Reference | None
) -> list[torch.Tensor]:
    """One request's chunks through ``decode_step``, one step each.

    The first chunk's payload carries ``ref``, every later one ``later``
    (or no voice). The vocoder's own draw is pinned per chunk, so two
    decodes of the same stream can be compared sample for sample.
    """
    pieces = []
    for step, (prefix, offset, finalize) in enumerate(chunk_plan(tokens.numel())):
        torch.manual_seed(step)
        voice = ref if offset == 0 else later
        (piece,) = decoder.decode_step(
            tokens[:prefix], [prefix], [stream_payload(voice, finalize, offset)], [request_id]
        )
        pieces.append(piece)
    return pieces


def test_the_voice_arrives_once_and_is_kept_for_the_stream(decoder: S3GenDecoder, references: list[Reference]) -> None:
    """A stream decodes in the voice of its first chunk, whatever a later payload holds.

    The runner-side transport hands stage 1 each chunk's own payload, so a
    later chunk may have no voice at all; the scheduler-side one keeps
    merging payloads, so a later chunk may have one. A decoder that took a
    later payload's voice when it found one would fail the second
    comparison: there the later payloads hold a different voice.
    """
    tokens = torch.randint(0, 6561, (70,))
    assert [offset for _, offset, _ in chunk_plan(70)] == [0, 20, 50]

    once = decode_stream(decoder, tokens, "once", references[0], later=None)
    other_later = decode_stream(decoder, tokens, "other-later", references[0], later=references[1])

    assert sum(piece.numel() for piece in once) == 2 * (70 + 3) * SAMPLES_PER_FRAME
    assert all(torch.equal(a, b) for a, b in zip(once, other_later, strict=True))
    assert decoder.streams == {}
    # Not vacuous: the other voice from the first chunk on gives other audio in the later chunks.
    other = decode_stream(decoder, tokens, "other", references[1], later=None)
    assert not torch.allclose(other[1], once[1], atol=1e-3)
    assert not torch.allclose(other[2], once[2], atol=1e-3)


def test_interleaved_streams_keep_their_own_voices(decoder: S3GenDecoder, references: list[Reference]) -> None:
    """Two live streams, a step each in turn; neither later chunk carries a voice."""
    tokens = [torch.randint(0, 6561, (70,)) for _ in range(2)]
    alone = [decode_stream(decoder, row, "alone", ref, None) for row, ref in zip(tokens, references, strict=True)]

    interleaved: list[list[torch.Tensor]] = [[], []]
    for step, (prefix, offset, finalize) in enumerate(chunk_plan(70)):
        for stream in (0, 1):
            torch.manual_seed(step)
            voice = references[stream] if offset == 0 else None
            (piece,) = decoder.decode_step(
                tokens[stream][:prefix], [prefix], [stream_payload(voice, finalize, offset)], [f"stream-{stream}"]
            )
            interleaved[stream].append(piece)
        assert set(decoder.streams) == ({"stream-0", "stream-1"} if not finalize else set())

    for stream in (0, 1):
        assert all(torch.equal(a, b) for a, b in zip(interleaved[stream], alone[stream], strict=True))


def test_abort_frees_stream_state(decoder: S3GenDecoder, references: list[Reference]) -> None:
    """A client that disconnects never sends a final chunk."""
    tokens = torch.randint(0, 6561, (23,))
    decoder.decode_step(tokens, [23], [stream_payload(references[0], False, 0)], ["aborted"])
    assert "aborted" in decoder.streams
    decoder.on_requests_finished({"aborted", "never-seen"})
    assert decoder.streams == {}


def test_two_requests_in_one_step_keep_separate_state(
    decoder: S3GenDecoder, references: list[Reference], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Each row of a step is decoded from its own tokens, offset, end flag and voice.

    What ``decode_step`` hands the decoder is recorded instead of decoded, so
    every field is compared by value; the second step shows a live stream
    keeps the voice it began with beside a new request with another.
    """
    seen: list[list[Chunk]] = []

    def record(chunks: list[Chunk]) -> list[tuple[torch.Tensor, StreamState | None]]:
        seen.append(chunks)
        return [
            (torch.zeros(1, 1), None if chunk.finalize else StreamState(chunk.reference, *[torch.zeros(1)] * 3))
            for chunk in chunks
        ]

    monkeypatch.setattr(decoder, "chunked_decode_streaming", record)
    torch.manual_seed(9)
    a, b, c = (torch.randint(0, 6561, (n,)) for n in (53, 30, 53))

    decoder.decode_step(
        torch.cat([a[:23], b]),
        [23, 30],
        [stream_payload(references[0], False, 0), stream_payload(references[1], True, 0)],
        ["a", "b"],
    )
    decoder.decode_step(
        torch.cat([a, c]),
        [53, 53],
        [stream_payload(None, False, 20), stream_payload(references[1], False, 0)],
        ["a", "c"],
    )

    def voice(ref: Reference) -> list:
        return [ref.prompt_token.tolist(), ref.prompt_feat.tolist(), ref.embedding.tolist()]

    def described(chunk: Chunk) -> tuple:
        return (chunk.tokens.tolist(), chunk.token_offset, chunk.finalize, chunk.state is None, voice(chunk.reference))

    first, second = seen
    assert [described(chunk) for chunk in first] == [
        (a[:23].tolist(), 0, False, True, voice(references[0])),
        (b.tolist(), 0, True, True, voice(references[1])),
    ]
    assert [described(chunk) for chunk in second] == [
        (a.tolist(), 20, False, False, voice(references[0])),
        (c.tolist(), 0, False, True, voice(references[1])),
    ]
    assert set(decoder.streams) == {"a", "c"}
    decoder.on_requests_finished({"a", "c"})


def test_spans_without_a_payload_or_tokens_give_empty_audio(decoder: S3GenDecoder, references: list[Reference]) -> None:
    """The profiling run has no payloads; the terminal payload has no tokens."""
    (profiling,) = decoder.decode_step(torch.zeros(16, dtype=torch.long), [16], None, None)
    assert profiling.shape == (0,)
    (terminal,) = decoder.decode_step(
        torch.zeros(0, dtype=torch.long), [0], [{"meta": {"finished": torch.tensor(True)}}], ["done"]
    )
    assert terminal.shape == (0,)

    # A token-less row beside a live one, on either side: each gets its own audio.
    tokens = torch.randint(0, 6561, (23,))
    for order in ((0, 1), (1, 0)):
        rows = [
            ({"meta": {"finished": torch.tensor(True)}}, "done", 0),
            (stream_payload(references[0], False, 0), "live", 23),
        ]
        payloads, ids, counts = zip(*[rows[i] for i in order], strict=True)
        audio = decoder.decode_step(tokens, list(counts), list(payloads), list(ids))
        assert [row.shape for row in audio] == [
            (40 * SAMPLES_PER_FRAME - SOURCE_CACHE_SAMPLES,) if count else (0,) for count in counts
        ]
        assert set(decoder.streams) == {"live"}
        decoder.on_requests_finished({"live"})


@pytest.mark.parametrize(
    "payloads",
    [[{"embed": {}}], [{}], None],
    ids=["payload without stream metadata", "empty payload", "no payload list"],
)
def test_tokens_without_stream_metadata_are_refused(decoder: S3GenDecoder, payloads: list[dict] | None) -> None:
    """Decoding them as a whole utterance would play wrong audio with no error.

    The runner passes an empty payload for a request it holds none for, and
    no list at all only on the profiling run, which has no request ids.
    """
    with pytest.raises(RuntimeError, match="23 tokens for request x without stream metadata"):
        decoder.decode_step(torch.randint(0, 6561, (23,)), [23], payloads, ["x"])


def test_decoding_a_chunk_twice_gives_the_same_mel(decoder: S3GenDecoder, references: list[Reference]) -> None:
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


def test_a_row_does_not_depend_on_its_batch(decoder: S3GenDecoder, references: list[Reference]) -> None:
    """Nothing is pinned here: each row's noise is fixed by its own positions."""
    torch.manual_seed(6)
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
        # Measured on CPU float32: at most 5.2e-6.
        torch.testing.assert_close(mel, alone, atol=TOLERANCE, rtol=TOLERANCE)


def test_noise_is_taken_by_position_for_every_prompt_length(decoder: S3GenDecoder, references: list[Reference]) -> None:
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
def test_noise_is_the_same_draw_when_built_on_a_default_device(decoder: S3GenDecoder) -> None:
    """vLLM builds the stage inside a default-device context."""
    with torch.device("cuda"):
        built = S3GenDecoder(ChatterboxConfig(), CONTEXT)
    assert built.flow_noise.device.type == built.trim_fade.device.type == "cuda"
    assert torch.equal(built.flow_noise.cpu(), decoder.flow_noise)


def test_chunked_mel_is_closer_to_one_chunk_than_a_noise_redraw(
    decoder: S3GenDecoder, references: list[Reference]
) -> None:
    """Chunks and one final chunk share noise, so they differ by context only."""
    torch.manual_seed(8)
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
    # Measured: 0.10 against 1.17. Noise drawn per chunk would put the
    # stitched mel as far from the one-chunk mel as a redraw is.
    assert (stitched - whole).abs().mean() < 0.25 * (redrawn - whole).abs().mean()


def test_a_non_final_chunk_shorter_than_the_mel_cache_is_refused(
    decoder: S3GenDecoder, references: list[Reference]
) -> None:
    """Three new tokens are six mel frames, and the cross-fade caches eight."""
    tokens = torch.randint(0, 6561, (3 + LOOKAHEAD,))
    with pytest.raises(RuntimeError, match="codec_chunk_frames must be at least 4"):
        decoder.chunked_decode_streaming([Chunk(tokens, 0, references[0], None, False)])


def test_a_longer_context_keeps_the_noise_of_the_default_cap():
    """The buffer of a longer context begins with the buffer sized for the default output cap.

    A buffer redrawn at the longer size would give every position other
    noise, and with it every utterance other audio than before the context
    could be longer than the cap. The first part must also be the one draw
    it has always been.
    """
    config = ChatterboxConfig()
    capped = S3GenDecoder(config, config.max_new_tokens).flow_noise
    longer = S3GenDecoder(config, CONTEXT).flow_noise

    frames = 2 * (250 + config.max_new_tokens + config.n_silence_tokens)
    assert capped.shape == (1, 80, frames)
    assert torch.equal(capped, torch.randn(1, 80, frames, generator=torch.Generator().manual_seed(0)))
    assert longer.shape == (1, 80, 2 * (250 + CONTEXT + config.n_silence_tokens))
    assert torch.equal(longer[:, :, :frames], capped)
    # A context under the cap still gets the capped buffer.
    assert torch.equal(S3GenDecoder(config, 300).flow_noise, capped)


def test_an_utterance_at_the_context_limit_has_noise_for_its_last_frame(
    decoder: S3GenDecoder, references: list[Reference]
) -> None:
    """The last chunk of the longest utterance, silence included, ends exactly at the end of the buffer.

    A buffer one frame short would leave the flow to draw the rest itself,
    and that chunk would no longer decode the same way twice.
    """
    tokens = torch.randint(0, 6561, (CONTEXT,))
    kept = torch.cat([tokens[CONTEXT - 48 - LEFT_CONTEXT_TOKENS :], decoder.silence])
    prompt = 2 * references[0].prompt_token.shape[1]
    start = decoder.prompt_noise_frames + 2 * (CONTEXT - 48 - LEFT_CONTEXT_TOKENS)
    assert start + 2 * kept.numel() == decoder.flow_noise.shape[2]

    (mel,) = decoder.chunk_mels([Chunk(tokens, CONTEXT - 48, references[0], None, True)])

    noise = torch.cat([decoder.flow_noise[:, :, :prompt], decoder.flow_noise[:, :, start:]], dim=2)
    (whole,) = flow_mels(decoder.flow, [kept], [references[0]], [True], 2, True, noise)
    assert mel.shape == (1, 80, 2 * (48 + 3))
    assert torch.equal(mel, whole[:, :, 2 * LEFT_CONTEXT_TOKENS :])


@pytest.mark.parametrize("length", [1, 2, 3])
def test_an_utterance_of_a_few_tokens_decodes_as_one_final_chunk(
    decoder: S3GenDecoder, references: list[Reference], length: int
) -> None:
    """A request can end after one token (``max_tokens=1``, an early stop token).

    Its only chunk is final, shorter than the lookahead and far shorter than
    the cross-fade's cache; an exception here would stop the stage for every
    request.
    """
    tokens = torch.randint(0, 6561, (length,))

    (audio,) = decoder.decode_step(tokens, [length], [stream_payload(references[0], True, 0)], ["short"])

    assert audio.shape == (2 * (length + 3) * SAMPLES_PER_FRAME,)
    assert torch.isfinite(audio).all() and audio.abs().max() > 0
    # Upstream's trim: the first 20 ms are silenced.
    assert torch.count_nonzero(audio[:SAMPLES_PER_FRAME]) == 0
    assert decoder.streams == {}


def test_a_row_is_vocoded_as_it_is_alone_whatever_shares_the_step(decoder: S3GenDecoder) -> None:
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


def test_rows_of_equal_length_share_one_vocoder_call(decoder: S3GenDecoder, monkeypatch: pytest.MonkeyPatch) -> None:
    """Nothing is padded between them, so batching them costs no independence."""
    torch.manual_seed(10)
    mels = [torch.randn(1, 80, 68), torch.randn(1, 80, 40), torch.randn(1, 80, 68)]
    cache = torch.randn(3, 1, SOURCE_CACHE_SAMPLES)
    inference = decoder.mel2wav.inference
    calls: list[int] = []

    def counted(speech_feat: torch.Tensor, cache_source: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        calls.append(speech_feat.shape[0])
        return inference(speech_feat=speech_feat, cache_source=cache_source)

    monkeypatch.setattr(decoder.mel2wav, "inference", counted)
    wavs, sources = decoder.vocode(mels, cache)

    assert calls == [2, 1]
    assert [wav.shape[1] for wav in wavs] == [68 * SAMPLES_PER_FRAME, 40 * SAMPLES_PER_FRAME, 68 * SAMPLES_PER_FRAME]
    # Each row keeps its own source cache.
    assert all(torch.equal(sources[i][:, :, :SOURCE_CACHE_SAMPLES], cache[i : i + 1]) for i in range(3))


def test_two_live_streams_in_one_step_use_their_own_state(decoder: S3GenDecoder, references: list[Reference]) -> None:
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


def test_a_live_stream_and_a_first_chunk_in_one_step(decoder: S3GenDecoder, references: list[Reference]) -> None:
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
def test_a_step_budget_that_can_split_an_utterance_is_refused_at_startup(
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


@pytest.fixture(scope="module")
def stage() -> ChatterboxS3Gen:
    """Stage 1 as the engine builds it with the deploy file's numbers."""
    decoder = yaml.safe_load(Path(get_deploy_config_path("chatterbox_turbo.yaml")).read_text())["stages"][1]
    config = stage_config(decoder["max_num_seqs"], decoder["max_num_batched_tokens"])
    config.model_config.max_model_len = decoder["max_model_len"]
    return ChatterboxS3Gen(vllm_config=config).eval()


def test_the_stage_builds_with_the_deploy_files_budget(stage: ChatterboxS3Gen, decoder: S3GenDecoder) -> None:
    """The deploy file passes the startup check, and every process draws the same flow noise."""
    assert stage.allow_patterns_overrides == [ChatterboxConfig().s3gen_weights]
    assert torch.equal(stage.flow_noise, decoder.flow_noise)


def test_forward_returns_each_requests_new_audio_and_rate(stage: ChatterboxS3Gen, references: list[Reference]) -> None:
    """The runner's contract: per request, 1-D float32 delta audio and an int32 rate, in batch order."""
    torch.manual_seed(11)
    tokens = torch.randint(0, 6561, (23,))

    output = stage(
        tokens,
        torch.arange(23),
        seq_token_counts=[23, 0],
        model_intermediate_buffer=[
            stream_payload(references[0], False, 0),
            {"meta": {"finished": torch.tensor(True)}},
        ],
        request_ids=["live", "done"],
        sampling_metadata=None,
    )

    assert output.text_hidden_states is None
    assert set(output.multimodal_outputs) == {"audio", "sr"}
    live, done = output.multimodal_outputs["audio"]
    assert live.shape == (40 * SAMPLES_PER_FRAME - SOURCE_CACHE_SAMPLES,) and live.dtype == torch.float32
    assert done.shape == (0,)
    rates = output.multimodal_outputs["sr"]
    assert [(rate.dtype, rate.shape, int(rate)) for rate in rates] == [(torch.int32, (), 24000)] * 2
    assert set(stage.streams) == {"live"}
    stage.on_requests_finished({"live"})


def test_load_weights_fills_every_parameter_and_skips_the_reference_encoders(
    stage: ChatterboxS3Gen, decoder: S3GenDecoder
) -> None:
    """The checkpoint also holds the tokenizer and speaker encoder, which run in the API process."""
    checkpoint = decoder.state_dict()
    encoders = {"tokenizer.conv1.weight": torch.zeros(1), "speaker_encoder.head.weight": torch.zeros(1)}

    loaded = stage.load_weights([*checkpoint.items(), *encoders.items()])

    assert loaded == set(checkpoint) == set(stage.state_dict())
    assert all(torch.equal(stage.state_dict()[name], tensor) for name, tensor in checkpoint.items())


@pytest.mark.parametrize(
    ("change", "match"),
    [("foreign", "Unexpected key"), ("missing", "Missing key")],
    ids=["a key the stage does not own", "a parameter the file does not fill"],
)
def test_load_weights_refuses_a_checkpoint_of_another_layout(
    stage: ChatterboxS3Gen, decoder: S3GenDecoder, change: str, match: str
) -> None:
    """Strict: a partial load would leave random weights behind with no error."""
    checkpoint = dict(decoder.state_dict())
    if change == "foreign":
        checkpoint["flow.extra.weight"] = torch.zeros(1)
    else:
        del checkpoint["mel2wav.conv_pre.bias"]
    with pytest.raises(RuntimeError, match=match):
        stage.load_weights(checkpoint.items())


def test_flow_batches_all_scheduled_rows_and_keeps_distinct_voices(decoder, monkeypatch):
    """All 64 rows retain their own prompt and position-dependent noise."""
    import vllm_omni.model_executor.models.chatterbox.chatterbox_s3gen as module

    refs = [reference(2 + i % 3) for i in range(64)]
    chunks = [Chunk(torch.full((15,), i, dtype=torch.long), 0, ref, None, True) for i, ref in enumerate(refs)]
    calls = []

    def flow(flow, tokens, references, finalize, n_timesteps, meanflow, noise):
        calls.append(len(tokens))
        outputs = []
        for row, ref, final, row_noise in zip(tokens, references, finalize, noise, strict=True):
            i = int(row[0])
            assert ref is refs[i]
            assert final
            prompt_frames = ref.prompt_token.shape[1] * 2
            assert torch.equal(row_noise[:, :prompt_frames], decoder.flow_noise[0, :, :prompt_frames])
            outputs.append(torch.full((1, 80, row.numel() * 2), float(i)))
        return outputs

    monkeypatch.setattr(module, "flow_mels", flow)
    mels = decoder.chunk_mels(chunks)
    assert calls == [64]
    assert [float(mel[0, 0, 0]) for mel in mels] == list(range(64))


def test_equal_length_vocoder_groups_batch_all_rows_and_keep_source_rows(decoder, monkeypatch):
    """Each mel length forms one batch, preserving its source-cache rows."""
    calls = []
    mels = [torch.full((1, 80, 12 + i % 2), float(i)) for i in range(64)]
    cache = torch.arange(64, dtype=torch.float32).reshape(64, 1, 1)

    def inference(speech_feat, cache_source):
        calls.append(speech_feat.shape[0])
        owners = speech_feat[:, :1, :1]
        assert torch.equal(owners, cache_source)
        return owners[:, 0], owners

    monkeypatch.setattr(decoder.mel2wav, "inference", inference)
    wavs, sources = decoder.vocode(mels, cache)
    assert calls == [32, 32]
    assert [float(wav[0, 0]) for wav in wavs] == list(range(64))
    assert [float(source[0, 0, 0]) for source in sources] == list(range(64))


def test_profile_with_64_slots_skips_neural_decode(decoder, monkeypatch):
    def unexpected(chunks):
        pytest.fail("startup profiling must not create neural activation batches")

    monkeypatch.setattr(decoder, "chunked_decode_streaming", unexpected)
    audios = decoder.decode_step(torch.zeros(64 * CONTEXT, dtype=torch.long), [CONTEXT] * 64, None, None)
    assert len(audios) == 64
    assert all(audio.numel() == 0 for audio in audios)


@pytest.mark.parametrize(
    "name,stage0_slots,suffix",
    [
        ("chatterbox", 128, ""),
        ("chatterbox_turbo", 64, ""),
        ("chatterbox", 128, "_bf16_mps"),
        ("chatterbox_turbo", 64, "_bf16_mps"),
    ],
)
def test_high_concurrency_profiles_admit_64_complete_requests(name, stage0_slots, suffix):
    config = resolve_deploy_yaml(get_deploy_config_path(f"{name}_high_concurrency{suffix}.yaml"))
    generator, decoder = config["stages"]
    assert generator["max_num_seqs"] == stage0_slots
    assert decoder["max_num_seqs"] == 64
    assert decoder["max_model_len"] >= generator["max_model_len"]
    assert decoder["max_num_batched_tokens"] >= 64 * decoder["max_model_len"]
    if name == "chatterbox":
        assert generator["hf_overrides"]["variant"] == "original"
        assert decoder["hf_overrides"]["variant"] == "original"
    if suffix:
        assert config["cuda_mps"] is True
        assert decoder["dtype"] == "float32"
        assert decoder["hf_overrides"]["flow_dtype"] == "bfloat16"
        assert decoder["hf_overrides"]["flow_attention_backend"] == "cudnn"
        chunking = config["connectors"]["connector_of_shared_memory"]["extra"]
        assert chunking["codec_chunk_frames"] == (30 if name == "chatterbox" else 15)
        assert chunking["codec_max_chunk_frames"] == 60
        assert chunking["codec_stream_scale_factor"] == 2
    baseline = yaml.safe_load(Path(get_deploy_config_path(f"{name}.yaml")).read_text())
    assert [stage["max_num_seqs"] for stage in baseline["stages"]] == [8, 8]


def test_streaming_batch_preserves_request_state(decoder, monkeypatch):
    refs = [reference(2) for _ in range(64)]
    chunks = []
    for i, ref in enumerate(refs):
        state = (
            None
            if i % 2
            else StreamState(
                ref,
                torch.full((1, 80, 8), float(i)),
                torch.full((1, 1, SOURCE_CACHE_SAMPLES), float(i)),
                torch.full((1, SOURCE_CACHE_SAMPLES), float(i)),
            )
        )
        chunks.append(Chunk(torch.tensor([i]), 0, ref, state, i % 3 == 0))
    calls = []

    def chunk_mels(group):
        calls.append(len(group))
        return [torch.full((1, 80, 12), float(chunk.tokens[0])) for chunk in group]

    def vocode(mels, cache):
        assert len(mels) == 32
        audio, sources = [], []
        for index, mel in enumerate(mels):
            owner = float(mel[0, 0, -1])
            if cache.shape[2]:
                assert float(cache[index, 0, 0]) == owner
            audio.append(torch.full((1, mel.shape[2] * SAMPLES_PER_FRAME), owner))
            sources.append(torch.full((1, 1, mel.shape[2] * SAMPLES_PER_FRAME), owner))
        return audio, sources

    monkeypatch.setattr(decoder, "chunk_mels", chunk_mels)
    monkeypatch.setattr(decoder, "vocode", vocode)
    results = decoder.chunked_decode_streaming(chunks)
    assert calls == [64]
    for i, (audio, state) in enumerate(results):
        assert float(audio[0, -1]) == i
        if chunks[i].finalize:
            assert state is None
        else:
            assert state.reference is refs[i]
            assert torch.all(state.mel == i)
            assert torch.all(state.source == i)
            assert torch.all(state.speech == i)
