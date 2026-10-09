# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import yaml

from tests.helpers.stage_config import get_deploy_config_path
from vllm_omni.data_entry_keys import OmniPayloadStruct, to_dict, to_struct
from vllm_omni.model_executor.models.chatterbox.chatterbox_s3gen import SAMPLES_PER_FRAME, S3GenDecoder
from vllm_omni.model_executor.models.chatterbox.conditioning import VoiceConditioning, build_prompt
from vllm_omni.model_executor.stage_input_processors.chatterbox import t3_to_s3gen
from vllm_omni.model_executor.stage_input_processors.cosyvoice3 import talker2code2wav_async_chunk
from vllm_omni.transformers_utils.configs.chatterbox import ChatterboxConfig

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def request_prompt(prompt_tokens: int) -> dict:
    conditioning = VoiceConditioning(
        cond_tokens=torch.randint(0, 6561, (1, 375)),
        speaker_emb=torch.randn(1, 256),
        prompt_token=torch.randint(0, 6561, (1, prompt_tokens)),
        prompt_feat=torch.randn(1, 2 * prompt_tokens, 80),
        embedding=torch.randn(1, 192),
    )
    return build_prompt([5, 6, 7], conditioning, ChatterboxConfig())


def talker_output(token_ids: list[int], finished: bool = True) -> SimpleNamespace:
    return SimpleNamespace(finished=finished, request_id="r", outputs=[SimpleNamespace(cumulative_token_ids=token_ids)])


def test_codes_are_the_valid_speech_tokens_001() -> None:
    """The stop token, and any placeholder echoed from the prompt, never reach the decoder."""
    (stage_input,) = t3_to_s3gen([talker_output([6561, 6561, 10, 20, 6560, 6562])], request_prompt(250))
    assert stage_input["prompt_token_ids"] == [10, 20, 6560]


def test_payload_is_one_final_chunk_in_the_requests_own_voice_001() -> None:
    prompt = request_prompt(202)
    (stage_input,) = t3_to_s3gen([talker_output([10, 20, 6562])], prompt)
    payload = to_struct(stage_input["additional_information"])

    assert bool(payload.meta.stream_finished) is True
    assert payload.meta.left_context_size == 0
    reference = prompt["additional_information"]["embed"]
    assert payload.embed.speech_token is reference["speech_token"]
    assert payload.embed.speech_feat is reference["speech_feat"]
    assert payload.embed.embedding is reference["embedding"]
    # Stage 0's own conditioning stays behind.
    assert payload.embed.voice is None and payload.ids is None


def test_unfinished_outputs_are_skipped_001() -> None:
    assert t3_to_s3gen([talker_output([10, 20], finished=False)], request_prompt(250)) == []


def test_an_utterance_with_no_speech_tokens_is_an_error_001() -> None:
    with pytest.raises(RuntimeError, match="no speech tokens"):
        t3_to_s3gen([talker_output([6562])], request_prompt(250))


@pytest.fixture(scope="module")
def connector_extra() -> dict:
    """The deploy file's connector settings, which are the chunk processor's configuration."""
    deploy = yaml.safe_load(Path(get_deploy_config_path("chatterbox_turbo.yaml")).read_text())
    return deploy["connectors"]["connector_of_shared_memory"]["extra"]


@pytest.fixture(scope="module")
def decoder() -> S3GenDecoder:
    return S3GenDecoder(ChatterboxConfig()).eval()


def streamed_payloads(extra: dict, prompt: dict, tokens: list[int], capped: bool = False) -> list[OmniPayloadStruct]:
    """Everything CosyVoice3's chunk processor emits for one utterance.

    Stage 0 is played one sampled token at a time. It finishes on the stop
    token it samples next or, when ``capped``, on the last token itself, as a
    request that reaches ``max_tokens`` does. The processor is then called
    once more for the finished request, as a late scheduler step would call it.
    """
    transfer_manager = SimpleNamespace(
        code_prompt_token_ids=defaultdict(list),
        request_payload={},
        connector=SimpleNamespace(config={"extra": extra}),
    )
    steps = [(tokens[:produced], False) for produced in range(1, len(tokens))]
    steps += [(tokens, True)] if capped else [(tokens, False), (tokens + [6562], True)]
    payloads = []
    for output_token_ids, finished in [*steps, steps[-1]]:
        request = SimpleNamespace(
            external_req_id="external-id",
            output_token_ids=output_token_ids,
            additional_information=prompt["additional_information"],
            is_finished=lambda finished=finished: finished,
        )
        payload = talker2code2wav_async_chunk(
            transfer_manager=transfer_manager, multimodal_output=None, request=request, is_finished=finished
        )
        if payload is not None:
            payloads.append(payload)
    return payloads


def decoded_samples(decoder: S3GenDecoder, payloads: list[OmniPayloadStruct]) -> int:
    """How many samples stage 1 returns for one request's payloads."""
    merged: dict = {}
    samples = 0
    for payload in payloads:
        # The runner merges each update into the request's payload; the
        # reference arrives once, on the first chunk.
        merged.update(to_dict(payload))
        codes = payload.codes.audio
        (audio,) = decoder.decode_step(codes, [codes.numel()], [merged], ["scheduler-id"])
        samples += audio.numel()
    return samples


def test_streamed_chunks_from_the_shared_processor_decode_to_the_whole_utterance_001(
    connector_extra: dict, decoder: S3GenDecoder
) -> None:
    """Streaming reuses CosyVoice3's chunk processor.

    It must find the reference under the names build_prompt writes, chunk as
    the deploy file's connector settings say, and emit payloads stage 1
    accepts, one token of talker output at a time.
    """
    tokens = torch.randint(0, 6561, (60,)).tolist()

    payloads = streamed_payloads(connector_extra, request_prompt(250), tokens)

    # First hop 15 plus 5 to align the 250-token reference, then 30; every
    # chunk but the last waits for the 3-token lookahead.
    assert [
        (payload.codes.audio.numel(), payload.meta.left_context_size, bool(payload.meta.stream_finished))
        for payload in payloads
    ] == [(23, 0, False), (53, 20, False), (60, 50, True)]
    assert decoded_samples(decoder, payloads) == 2 * (60 + 3) * SAMPLES_PER_FRAME
    assert decoder.streams == {}


@pytest.mark.parametrize("capped", [False, True], ids=["stop token", "length cap"])
@pytest.mark.parametrize("prompt_tokens", [250, 202])
def test_every_utterance_length_is_sent_once_and_finished_once_001(
    connector_extra: dict, prompt_tokens: int, capped: bool
) -> None:
    """No length makes the processor send a token's audio twice, skip one or finish twice.

    The processor sends a chunk as soon as a hop of unsent tokens and the
    lookahead after it exist, as the prefix through that lookahead, and moves
    its cursor by the hop alone. The lengths to fear are those where the
    utterance ends exactly there, so the stop token finds nothing unsent but
    the lookahead: 23 and 53 tokens for the 250-token reference (hops of 20
    and 30), 26 and 56 for the 202-token one (23 and 30).
    """
    chunk, lookahead = connector_extra["codec_chunk_frames"], connector_extra["codec_pre_lookahead_frames"]
    # The first hop also pads the reference prompt to a multiple of the chunk; later hops double up to the cap.
    hops = [chunk + -prompt_tokens % chunk]
    while sum(hops) < 240:
        hops.append(
            min(
                connector_extra["codec_max_chunk_frames"],
                chunk * connector_extra["codec_stream_scale_factor"] ** len(hops),
            )
        )
    prompt = request_prompt(prompt_tokens)
    tokens = torch.randint(0, 6561, (240,)).tolist()

    ended_on_a_boundary = []
    for length in range(1, 241):
        payloads = streamed_payloads(connector_extra, prompt, tokens[:length], capped)
        codes = [payload.codes.audio.tolist() for payload in payloads]
        offsets = [payload.meta.left_context_size for payload in payloads]

        # One final payload, the last, holding the whole utterance; nothing after it.
        finals = [bool(payload.meta.stream_finished) for payload in payloads]
        assert finals == [False] * (len(payloads) - 1) + [True], length
        assert codes[-1] == tokens[:length], length
        # Each payload is a prefix with something past its offset to decode.
        assert all(sent == tokens[: len(sent)] for sent in codes), length
        assert all(len(sent) > offset for sent, offset in zip(codes, offsets, strict=True)), length
        # The cursor starts at 0 and moves by the hops. A chunk that is not
        # the last ends exactly the lookahead past the next one's offset: the
        # next chunk decodes from where this one stopped, no token twice.
        assert offsets == [sum(hops[:sent]) for sent in range(len(payloads))], length
        assert [len(sent) - lookahead for sent in codes[:-1]] == offsets[1:], length
        # The reference travels once.
        assert [payload.embed is not None for payload in payloads] == [True] + [False] * (len(payloads) - 1), length

        if len(payloads) > 1 and codes[-1] == codes[-2]:
            ended_on_a_boundary.append(length)

    # After a stop token on a boundary the last payload repeats the codes
    # with a new offset, and only the lookahead is left to decode. A request
    # cut off there by its token cap ends in the same step as its last token.
    boundaries = [sum(hops[:sent]) + lookahead for sent in range(1, len(hops)) if sum(hops[:sent]) + lookahead <= 240]
    assert ended_on_a_boundary == ([] if capped else boundaries)


@pytest.mark.parametrize(("prompt_tokens", "length"), [(250, 53), (202, 56)])
def test_an_utterance_ending_on_a_chunk_boundary_decodes_to_its_own_length_001(
    connector_extra: dict, decoder: S3GenDecoder, prompt_tokens: int, length: int
) -> None:
    """The stop token's payload adds the lookahead's audio and nothing that was already played."""
    tokens = torch.randint(0, 6561, (length,)).tolist()

    payloads = streamed_payloads(connector_extra, request_prompt(prompt_tokens), tokens)

    assert payloads[-1].codes.audio.tolist() == payloads[-2].codes.audio.tolist() == tokens
    assert decoded_samples(decoder, payloads) == 2 * (length + 3) * SAMPLES_PER_FRAME
    assert decoder.streams == {}
