# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
import re
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path

import pytest
import torch
from torch import nn
from vllm.sampling_params import SamplingParams

from tests.helpers.stage_config import get_deploy_config_path, modify_stage_config
from tests.model_executor.models.chatterbox.helpers import (
    DEPLOY,
    LEFT_CONTEXT,
    SEED,
    build_stage,
    load_stage_weights,
    random_voice,
    stage_vllm_config,
    write_model_dir,
    write_s3gen_checkpoint,
)
from vllm_omni.model_executor.models.chatterbox.chatterbox import ChatterboxForConditionalGeneration
from vllm_omni.model_executor.models.chatterbox.chatterbox_s3gen import (
    SAMPLES_PER_FRAME,
    SOURCE_CACHE_SAMPLES,
    ChatterboxS3Gen,
    Chunk,
    Reference,
    S3GenDecoder,
    StreamState,
    fade_in_out,
    flow_mels,
)
from vllm_omni.transformers_utils.configs.chatterbox import ChatterboxConfig

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

# (cumulative prefix length, token offset, finalize) per chunk: what the
# chunk processor sends for 70 and for 140 tokens under the deploy file's
# schedule and a 250-token reference (hops of 20, 30 and 60, each chunk but
# the last three lookahead tokens past its hop).
PLAN_70 = [(23, 0, False), (53, 20, False), (70, 50, True)]
PLAN_140 = [(23, 0, False), (53, 20, False), (113, 50, False), (140, 110, True)]
# Stage 1's context in the deploy file: the longest utterance a decoder is built for.
CONTEXT = 2048
# Batched against alone, CPU float32. Comparable models use 1e-5 on CPU.
TOLERANCE = 1e-5


def reference(prompt_tokens: int) -> Reference:
    """Stage 1's part of a random voice."""
    voice = random_voice(1, prompt_tokens)
    return Reference(prompt_token=voice.prompt_token, prompt_feat=voice.prompt_feat, embedding=voice.embedding)


def stream_payload(ref: Reference | None, finished: bool, offset: int) -> dict:
    """One request's payload as the runner hands it to stage 1, with or without the voice.

    The runner's own per-request keys share the dict, as on Model Runner V2.
    """
    payload: dict = {
        "meta": {
            "finished": torch.tensor(finished),
            "stream_finished": torch.tensor(finished),
            "left_context_size": offset,
            "num_processed_tokens": 0,
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
    """A float32 flow: the value-level checks below compare float32 computations."""
    torch.manual_seed(0)
    return S3GenDecoder(ChatterboxConfig(flow_dtype="float32"), CONTEXT, SEED).eval()


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


def test_bfloat16_flow_keeps_vocoder_and_audio_in_float32(decoder: S3GenDecoder) -> None:
    """Actual flow and HiFT calls enforce the mixed-precision boundary.

    The decoder is built from the default config, whose flow runs in
    bfloat16, with the float32 decoder's weights.
    """
    bf16 = S3GenDecoder(ChatterboxConfig(), CONTEXT, SEED).eval()
    bf16.load_state_dict(decoder.state_dict())
    flow_dtypes, vocoder_dtypes = [], []

    def flow_output(module: nn.Module, inputs: tuple, output: tuple) -> None:
        flow_dtypes.append(output[0].dtype)

    def vocoder_input(module: nn.Module, inputs: tuple) -> None:
        vocoder_dtypes.append(inputs[0].dtype)

    hooks = [
        bf16.flow.decoder.register_forward_hook(flow_output),
        bf16.mel2wav.conv_pre.register_forward_pre_hook(vocoder_input),
    ]
    try:
        chunks = [Chunk(torch.arange(15), 0, reference(2), None, True)]
        ((audio, state),) = bf16.chunked_decode_streaming(chunks)
    finally:
        for hook in hooks:
            hook.remove()

    assert flow_dtypes == [torch.bfloat16]
    assert vocoder_dtypes == [torch.float32]
    assert audio.dtype == torch.float32
    assert torch.isfinite(audio).all()
    assert state is None
    assert next(bf16.flow.parameters()).dtype == torch.float32


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
        frames = 2 * tokens[i].numel() - (0 if finalize[i] else 2 * decoder.config.pre_lookahead_len)
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
    for prefix, offset, finalize in PLAN_140:
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
    """One request's 70 tokens through ``decode_step`` in ``PLAN_70``'s chunks, one step each.

    The first chunk's payload carries ``ref``, every later one ``later``
    (or no voice). The vocoder's own draw is pinned per chunk, so two
    decodes of the same stream can be compared sample for sample.
    """
    pieces = []
    for step, (prefix, offset, finalize) in enumerate(PLAN_70):
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

    once = decode_stream(decoder, tokens, "once", references[0], later=None)
    other_later = decode_stream(decoder, tokens, "other-later", references[0], later=references[1])

    assert sum(piece.numel() for piece in once) == 2 * (70 + 3) * SAMPLES_PER_FRAME
    assert all(torch.equal(a, b) for a, b in zip(once, other_later, strict=True))
    assert decoder.streams == {}
    # Not vacuous: the other voice from the first chunk on gives other audio in the later chunks.
    other = decode_stream(decoder, tokens, "other", references[1], later=None)
    # Measured on initial weights: about 5e-4 apart.
    assert not torch.allclose(other[1], once[1], atol=1e-4)
    assert not torch.allclose(other[2], once[2], atol=1e-4)


def test_interleaved_streams_keep_their_own_voices(decoder: S3GenDecoder, references: list[Reference]) -> None:
    """Two live streams, a step each in turn; neither later chunk carries a voice."""
    tokens = [torch.randint(0, 6561, (70,)) for _ in range(2)]
    alone = [decode_stream(decoder, row, "alone", ref, None) for row, ref in zip(tokens, references, strict=True)]

    interleaved: list[list[torch.Tensor]] = [[], []]
    for step, (prefix, offset, finalize) in enumerate(PLAN_70):
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


def test_two_requests_in_one_step_keep_separate_state(decoder: S3GenDecoder, references: list[Reference]) -> None:
    """Each row of a step is decoded from its own tokens, offset, end flag and voice.

    ``decode_step`` must form the chunks written out below from the flat ids
    and the payloads: decoding those directly under the same vocoder draw
    gives the same audio sample for sample, and the streams it keeps hold
    the voices they began with. The second step shows a live stream keeps
    its voice beside a new request with another.
    """
    torch.manual_seed(9)
    a, b, c = (torch.randint(0, 6561, (n,)) for n in (53, 30, 53))

    torch.manual_seed(20)
    first = decoder.decode_step(
        torch.cat([a[:23], b]),
        [23, 30],
        [stream_payload(references[0], False, 0), stream_payload(references[1], True, 0)],
        ["a", "b"],
    )
    assert set(decoder.streams) == {"a"}
    torch.manual_seed(21)
    second = decoder.decode_step(
        torch.cat([a, c]),
        [53, 53],
        [stream_payload(None, False, 20), stream_payload(references[1], False, 0)],
        ["a", "c"],
    )
    kept = {request_id: decoder.streams[request_id].reference for request_id in ("a", "c")}
    decoder.on_requests_finished({"a", "c"})

    torch.manual_seed(20)
    (first_a, state_a), (first_b, _) = decoder.chunked_decode_streaming(
        [Chunk(a[:23], 0, references[0], None, False), Chunk(b, 0, references[1], None, True)]
    )
    torch.manual_seed(21)
    (second_a, _), (second_c, _) = decoder.chunked_decode_streaming(
        [Chunk(a, 20, references[0], state_a, False), Chunk(c, 0, references[1], None, False)]
    )

    expected = [first_a, first_b, second_a, second_c]
    assert all(torch.equal(got, want.reshape(-1)) for got, want in zip([*first, *second], expected, strict=True))

    def voice(ref: Reference) -> list:
        return [ref.prompt_token.tolist(), ref.prompt_feat.tolist(), ref.embedding.tolist()]

    assert voice(kept["a"]) == voice(references[0])
    assert voice(kept["c"]) == voice(references[1])


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
    """Decoding them as a whole utterance, or skipping them, would play wrong audio with no error.

    The runner passes an empty payload for a request it holds none for, and
    no list at all only on the profiling run, which has no request ids; the
    default runner also goes on without one when building it failed.
    """
    with pytest.raises(RuntimeError, match="request.*x.* without stream metadata"):
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
    # Seventy tokens played: the chunk processor sends tokens[20:], fifty of them context.
    kept = tokens[20:]
    start = decoder.prompt_noise_frames + 2 * 20
    for ref in references:
        prompt = 2 * ref.prompt_token.shape[1]
        (mel,) = decoder.chunk_mels([Chunk(kept, LEFT_CONTEXT, ref, None, False, start=20)])

        def decoded_from(utterance_start: int) -> torch.Tensor:
            noise = torch.cat(
                [
                    decoder.flow_noise[:, :, :prompt],
                    decoder.flow_noise[:, :, utterance_start : utterance_start + 2 * kept.numel()],
                ],
                dim=2,
            )
            (whole,) = flow_mels(decoder.flow, [kept], [ref], [False], 2, True, noise)
            return whole[:, :, 2 * LEFT_CONTEXT :]

        assert torch.equal(mel, decoded_from(start))
        # Not vacuous: the same row placed at the start of the utterance differs.
        assert not torch.allclose(mel, decoded_from(decoder.prompt_noise_frames), atol=1e-2)


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
            for prefix, offset, finalize in PLAN_140
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
    tokens = torch.randint(0, 6561, (3 + decoder.config.pre_lookahead_len,))
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
    capped = S3GenDecoder(config, config.max_new_tokens, SEED).flow_noise
    longer = S3GenDecoder(config, CONTEXT, SEED).flow_noise

    frames = 2 * (250 + config.max_new_tokens + config.n_silence_tokens)
    assert capped.shape == (1, 80, frames)
    assert torch.equal(capped, torch.randn(1, 80, frames, generator=torch.Generator().manual_seed(SEED)))
    assert longer.shape == (1, 80, 2 * (250 + CONTEXT + config.n_silence_tokens))
    assert torch.equal(longer[:, :, :frames], capped)
    # A context under the cap still gets the capped buffer.
    assert torch.equal(S3GenDecoder(config, 300, SEED).flow_noise, capped)


def test_an_utterance_at_the_context_limit_has_noise_for_its_last_frame(
    decoder: S3GenDecoder, references: list[Reference]
) -> None:
    """The last chunk of the longest utterance, silence included, ends exactly at the end of the buffer.

    A buffer one frame short would leave the flow to draw the rest itself,
    and that chunk would no longer decode the same way twice.
    """
    tokens = torch.randint(0, 6561, (CONTEXT,))
    kept = torch.cat([tokens[CONTEXT - 48 - LEFT_CONTEXT :], decoder.silence])
    prompt = 2 * references[0].prompt_token.shape[1]
    start = decoder.prompt_noise_frames + 2 * (CONTEXT - 48 - LEFT_CONTEXT)
    assert start + 2 * kept.numel() == decoder.flow_noise.shape[2]

    first = CONTEXT - 48 - LEFT_CONTEXT
    (mel,) = decoder.chunk_mels([Chunk(tokens[first:], LEFT_CONTEXT, references[0], None, True, start=first)])

    noise = torch.cat([decoder.flow_noise[:, :, :prompt], decoder.flow_noise[:, :, start:]], dim=2)
    (whole,) = flow_mels(decoder.flow, [kept], [references[0]], [True], 2, True, noise)
    assert mel.shape == (1, 80, 2 * (48 + 3))
    assert torch.equal(mel, whole[:, :, 2 * LEFT_CONTEXT :])


def test_a_chunk_past_the_context_has_no_noise_and_is_an_error(
    decoder: S3GenDecoder, references: list[Reference]
) -> None:
    """One token later than the last chunk of the longest utterance, the noise slice would come back short."""
    first = CONTEXT - 48 - LEFT_CONTEXT
    tokens = torch.randint(0, 6561, (48 + LEFT_CONTEXT,))
    with pytest.raises(RuntimeError, match=f"utterance position {first + 1}, past the"):
        decoder.chunk_mels([Chunk(tokens, LEFT_CONTEXT, references[0], None, True, start=first + 1)])


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


def vocoder_calls(decoder: S3GenDecoder) -> tuple[list[int], torch.utils.hooks.RemovableHandle]:
    """The batch size of every HiFT call, from its F0 predictor, which every call runs once."""
    calls: list[int] = []

    def f0_input(module: nn.Module, inputs: tuple) -> None:
        calls.append(inputs[0].shape[0])

    return calls, decoder.mel2wav.f0_predictor.register_forward_pre_hook(f0_input)


def test_rows_of_equal_length_share_one_vocoder_call(decoder: S3GenDecoder) -> None:
    """Nothing is padded between them, so batching them costs no independence."""
    torch.manual_seed(10)
    mels = [torch.randn(1, 80, 68), torch.randn(1, 80, 40), torch.randn(1, 80, 68)]
    cache = torch.randn(3, 1, SOURCE_CACHE_SAMPLES)
    calls, hook = vocoder_calls(decoder)
    try:
        wavs, sources = decoder.vocode(mels, cache)
    finally:
        hook.remove()

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


@pytest.mark.parametrize(
    ("max_num_batched_tokens", "max_num_scheduled_tokens", "names"),
    [
        (131071, None, "max_num_batched_tokens >= max_num_seqs (64) * max_model_len (2048) = 131072, got 131071"),
        (131072, 8000, "max_num_scheduled_tokens >= max_num_seqs (64) * max_model_len (2048) = 131072, got 8000"),
    ],
    ids=["batched tokens", "scheduled tokens"],
)
def test_a_step_budget_that_can_split_an_utterance_is_refused_at_startup(
    model_dir: str, max_num_batched_tokens: int, max_num_scheduled_tokens: int | None, names: str
) -> None:
    """A new request the step's budget cannot cover is scheduled with part of its tokens.

    The decoder would take that part for the whole utterance. An utterance is
    at most the stage's own context long, so 32 slots of a 2048-token
    context need 65536, whatever ``max_tokens`` a request asks for. The
    scheduler's budget is its scheduled-token limit whenever one is set, and
    the message names the field it read. The deploy file's
    stage 1 gets the budget as ``--stage-overrides`` would give it.
    """
    config = stage_vllm_config(
        model_dir,
        DEPLOY["turbo"],
        1,
        max_num_batched_tokens=max_num_batched_tokens,
        max_num_scheduled_tokens=max_num_scheduled_tokens,
    )
    assert (config.scheduler_config.max_num_seqs, config.model_config.max_model_len) == (64, 2048)

    with pytest.raises(ValueError, match=re.escape(names)):
        build_stage(config, "cpu")


@pytest.fixture(scope="module")
def stage(built_stage: Callable[[str, int], ChatterboxForConditionalGeneration]) -> ChatterboxS3Gen:
    """Stage 1 as the engine builds it from the deploy file."""
    return built_stage("turbo", 1).model


def test_the_deploy_files_stage_loads_its_own_weights_and_draws_the_shared_noise(
    stage: ChatterboxS3Gen, decoder: S3GenDecoder
) -> None:
    """The deploy file passes the startup check, and every process draws the same flow noise."""
    assert stage.allow_patterns_overrides == [ChatterboxConfig().s3gen_weights]
    assert torch.equal(stage.flow_noise, decoder.flow_noise)


def test_a_largest_chunk_below_the_first_is_refused_at_startup(model_dir: str) -> None:
    """The schedule grows from codec_chunk_frames to codec_max_chunk_frames; a cap below the start is a typo."""
    deploy = modify_stage_config(
        get_deploy_config_path(DEPLOY["turbo"]),
        updates={"connectors.connector_of_shared_memory.extra.codec_max_chunk_frames": 10},
    )
    with pytest.raises(ValueError, match="codec_max_chunk_frames"):
        build_stage(stage_vllm_config(model_dir, deploy, 1), "cpu")


def test_the_stage_seeds_its_flow_noise_with_its_own_seed(model_dir: str) -> None:
    """The flow noise is the draw of stage 1's vLLM ``seed``, as ``--stage-overrides`` gives it."""
    config = stage_vllm_config(model_dir, DEPLOY["turbo"], 1, seed=7)
    stage = build_stage(config, "cpu")

    expected = S3GenDecoder(config.model_config.hf_config, config.model_config.max_model_len, 7)
    assert torch.equal(stage.flow_noise, expected.flow_noise)


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


def test_load_weights_fills_every_parameter_and_skips_the_reference_encoders(tmp_path: Path) -> None:
    """The checkpoint also holds the tokenizer and speaker encoder, which run in the API process.

    The file is written in the released layout and loaded through vLLM's
    loader, which fails on any parameter the stage reports it did not fill.
    """
    directory = write_model_dir(tmp_path)
    config = stage_vllm_config(directory, DEPLOY["turbo"], 1)
    unified = build_stage(config, "cpu")
    checkpoint = write_s3gen_checkpoint(directory, config.model_config.hf_config, seed=0)

    load_stage_weights(unified, config)

    stage = unified.model
    own = {
        name: tensor for name, tensor in checkpoint.items() if not name.startswith(("tokenizer.", "speaker_encoder."))
    }
    assert len(own) < len(checkpoint)
    assert set(stage.state_dict()) == set(own)
    assert all(torch.equal(stage.state_dict()[name], tensor) for name, tensor in own.items())
    assert stage.load_weights(checkpoint.items()) == set(own)
    assert unified.load_weights(checkpoint.items()) == {name for name, _ in unified.named_parameters()}


@pytest.mark.xfail(
    strict=True,
    reason=(
        "Production defect in models/common/snake_activation.py: Snake.precompute_exp_cache assigns "
        "self.alpha.contiguous(), which is the alpha Parameter itself, to _exp_alpha, so a stage's first decode "
        "registers each of HiFT's 72 Snake alphas a second time as a parameter (_exp_alpha); reloading the "
        "released checkpoint into a stage that has decoded then fails the strict load on those names"
    ),
)
def test_a_stage_that_has_decoded_reloads_its_checkpoint(
    stage: ChatterboxS3Gen, references: list[Reference], tmp_path: Path
) -> None:
    """vLLM reloads weights into a running stage (weight updates, sleep mode); decoding must not change its parameters."""
    before = {name for name, _ in stage.named_parameters()}
    stage.decode_step(torch.randint(0, 6561, (23,)), [23], [stream_payload(references[0], True, 0)], ["reload"])
    checkpoint = write_s3gen_checkpoint(write_model_dir(tmp_path), stage.config, seed=1)

    assert {name for name, _ in stage.named_parameters()} == before
    assert stage.load_weights(checkpoint.items()) == before


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


def flow_calls(decoder: S3GenDecoder) -> tuple[list[int], torch.utils.hooks.RemovableHandle]:
    """The batch size of every flow call."""
    calls: list[int] = []

    def flow_input(module: nn.Module, args: tuple, kwargs: dict) -> None:
        calls.append(kwargs["mu"].shape[0])

    return calls, decoder.flow.decoder.register_forward_pre_hook(flow_input, with_kwargs=True)


def test_flow_batches_all_scheduled_rows_and_keeps_distinct_voices(decoder: S3GenDecoder) -> None:
    """All 64 rows go through one flow call and each decodes as it does alone.

    Alone, a row has only its own prompt and its own position-dependent
    noise; the batch must give it the same mel.
    """
    torch.manual_seed(15)
    refs = [reference(2 + i % 3) for i in range(64)]
    chunks = [Chunk(torch.randint(0, 6561, (15,)), 0, ref, None, True) for ref in refs]
    calls, hook = flow_calls(decoder)
    try:
        mels = decoder.chunk_mels(chunks)
    finally:
        hook.remove()

    assert calls == [64]
    assert [mel.shape for mel in mels] == [(1, 80, 2 * (15 + 3))] * 64
    for chunk, mel in zip(chunks, mels, strict=True):
        (alone,) = decoder.chunk_mels([chunk])
        torch.testing.assert_close(mel, alone, atol=TOLERANCE, rtol=TOLERANCE)


def test_equal_length_vocoder_groups_batch_all_rows_and_keep_source_rows(decoder: S3GenDecoder) -> None:
    """Each mel length forms one batch; every row keeps its own source cache and its own audio.

    The audio is compared with HiFT called on each group directly, under the
    same draw.
    """
    torch.manual_seed(16)
    mels = [torch.randn(1, 80, 12 + i % 2) for i in range(64)]
    cache = torch.randn(64, 1, SOURCE_CACHE_SAMPLES)
    calls, hook = vocoder_calls(decoder)
    try:
        torch.manual_seed(17)
        wavs, sources = decoder.vocode(mels, cache)
    finally:
        hook.remove()

    torch.manual_seed(17)
    groups = [list(range(0, 64, 2)), list(range(1, 64, 2))]
    expected: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
    for rows in groups:
        group_wavs, group_sources = decoder.mel2wav.inference(
            speech_feat=torch.cat([mels[i] for i in rows]), cache_source=cache[rows]
        )
        expected |= {i: (group_wavs[k : k + 1], group_sources[k : k + 1]) for k, i in enumerate(rows)}

    assert calls == [32, 32]
    assert all(torch.equal(sources[i][:, :, :SOURCE_CACHE_SAMPLES], cache[i : i + 1]) for i in range(64))
    assert all(torch.equal(wavs[i], expected[i][0]) and torch.equal(sources[i], expected[i][1]) for i in range(64))


def test_profile_with_64_slots_skips_neural_decode(decoder: S3GenDecoder) -> None:
    """The engine's profiling run has no request ids: neither the flow nor the vocoder runs."""
    calls: list[str] = []

    def refuse(module: nn.Module, inputs: tuple) -> None:
        calls.append(type(module).__name__)
        raise AssertionError("startup profiling must not create neural activation batches")

    hooks = [module.register_forward_pre_hook(refuse) for module in (decoder.flow.decoder, decoder.mel2wav)]
    try:
        audios = decoder.decode_step(torch.zeros(64 * CONTEXT, dtype=torch.long), [CONTEXT] * 64, None, None)
    finally:
        for hook in hooks:
            hook.remove()

    assert calls == []
    assert len(audios) == 64
    assert all(audio.numel() == 0 for audio in audios)


@pytest.mark.parametrize("variant", ["turbo", "original"])
def test_each_profiles_stage_1_passes_its_startup_budget_check(model_dir: str, variant: str) -> None:
    """Stage 1 refuses at startup a step budget that could split an utterance; each profile's must pass.

    Stage 0 admits a request per stage 1 slot, Original's guided pairs taking two sequences each.
    """
    stage1 = stage_vllm_config(model_dir, DEPLOY[variant], 1)
    stage0 = stage_vllm_config(model_dir, DEPLOY[variant], 0)

    build_stage(stage1, "cpu")

    pairs = 2 if stage1.model_config.hf_config.variant == "original" else 1
    assert stage0.scheduler_config.max_num_seqs == pairs * stage1.scheduler_config.max_num_seqs


def test_streaming_batch_preserves_request_state(decoder: S3GenDecoder) -> None:
    """64 rows in one step, half continuing a stream, a third final: each keeps its own state.

    Every continuing row's held-back tail is a constant, its row number, so
    its first played sample shows whose tail it was faded with. Changing one
    row's cached mel and source under the same draw changes that row's audio
    and no other. The flow runs once for all rows, and the vocoder once per
    mel length within the continuing and the first chunks.
    """
    torch.manual_seed(18)
    refs = [reference(2) for _ in range(64)]
    chunks = []
    for i, ref in enumerate(refs):
        state = (
            None
            if i % 2
            else StreamState(
                ref,
                torch.randn(1, 80, 8),
                torch.randn(1, 1, SOURCE_CACHE_SAMPLES) * 0.1,
                torch.full((1, SOURCE_CACHE_SAMPLES), float(i)),
            )
        )
        chunks.append(Chunk(torch.randint(0, 6561, (9,)), 0, ref, state, i % 3 == 0))
    mels = decoder.chunk_mels(chunks)

    flows, flow_hook = flow_calls(decoder)
    vocoder, vocoder_hook = vocoder_calls(decoder)
    try:
        torch.manual_seed(19)
        results = decoder.chunked_decode_streaming(chunks)
    finally:
        flow_hook.remove()
        vocoder_hook.remove()
    other = list(chunks)
    other[10] = replace(
        chunks[10], state=replace(chunks[10].state, mel=-chunks[10].state.mel, source=-chunks[10].state.source)
    )
    torch.manual_seed(19)
    changed = decoder.chunked_decode_streaming(other)

    continuing = [chunk.state is not None for chunk in chunks]
    # Continuing rows first, then first chunks; within each, one call per mel
    # length in order of first appearance, and a final row's mel is longer.
    expected_calls = []
    for rows in ([i for i in range(64) if continuing[i]], [i for i in range(64) if not continuing[i]]):
        finals = [chunks[i].finalize for i in rows]
        expected_calls += [finals.count(final) for final in dict.fromkeys(finals)]
    assert flows == [64]
    assert vocoder == expected_calls
    for i, ((audio, state), (changed_audio, changed_state)) in enumerate(zip(results, changed, strict=True)):
        if continuing[i]:
            assert abs(float(audio[0, 0]) - i) < 0.1
        assert torch.equal(changed_audio, audio) is (i != 10)
        if chunks[i].finalize:
            assert state is None
        else:
            assert state.reference is refs[i]
            assert torch.equal(state.mel, mels[i][:, :, -8:])
            if i != 10:
                assert torch.equal(changed_state.source, state.source)
                assert torch.equal(changed_state.speech, state.speech)
