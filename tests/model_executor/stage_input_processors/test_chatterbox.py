# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
from collections import defaultdict
from collections.abc import Iterator
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import yaml

from tests.helpers.stage_config import get_deploy_config_path
from vllm_omni.core.sched.omni_scheduling_coordinator import OmniSchedulingCoordinator
from vllm_omni.data_entry_keys import OmniPayloadStruct, to_dict, to_struct
from vllm_omni.distributed.omni_connectors.model_runner.omni_connector_payload_transport import (
    _OmniConnectorPayloadTransportMixin as Transport,
)
from vllm_omni.engine.serialization import serialize_additional_information
from vllm_omni.model_executor.models.chatterbox.chatterbox_s3gen import SAMPLES_PER_FRAME, S3GenDecoder
from vllm_omni.model_executor.models.chatterbox.conditioning import VoiceConditioning, build_prompt
from vllm_omni.model_executor.models.chatterbox.pipeline import CHATTERBOX_TURBO_PIPELINE
from vllm_omni.model_executor.stage_input_processors.chatterbox import t3_to_s3gen, t3_to_s3gen_async_chunk
from vllm_omni.transformers_utils.configs.chatterbox import ChatterboxConfig
from vllm_omni.worker_v2.omni_data_plane import OmniRunnerDataPlane

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

STOP = ChatterboxConfig().stop_speech_token
PLANES = ["scheduler-side", "runner-side"]


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
    """The deploy file's connector settings, which are the connector's config and the chunk processor's."""
    deploy = yaml.safe_load(Path(get_deploy_config_path("chatterbox_turbo.yaml")).read_text())
    return deploy["connectors"]["connector_of_shared_memory"]["extra"]


@pytest.fixture(scope="module")
def decoder() -> S3GenDecoder:
    return S3GenDecoder(ChatterboxConfig()).eval()


@pytest.fixture
def plane(connector_extra: dict) -> Iterator[OmniRunnerDataPlane]:
    """Stage 0's runner-side transport as the engine builds it, short of the connector.

    The processor is loaded from the pipeline's own path by the transport's
    own loader. Only the hand-over to the connector's send queue is replaced:
    it records what would be sent and keeps the two books the real one keeps
    (chunks sent per request, and the external id cleanup later frees).
    """
    plane = OmniRunnerDataPlane(
        vllm_config=None,
        model_config=SimpleNamespace(
            async_chunk=True,
            worker_type="ar",
            stage_id=0,
            custom_process_next_stage_input_func=CHATTERBOX_TURBO_PIPELINE.stages[
                0
            ].async_chunk_process_next_stage_input_func,
        ),
    )
    plane._omni_connector = SimpleNamespace(config=connector_extra)
    plane.sent = []

    def enqueue(request: SimpleNamespace, payload: OmniPayloadStruct, *, wait_for_delivery: bool) -> tuple[bool, None]:
        plane.sent.append(payload)
        plane._request_ids_mapping.setdefault(request.request_id, request.external_req_id)
        plane.put_req_chunk[request.external_req_id] += 1
        return True, None

    plane._enqueue_chunk_payload = enqueue
    yield plane
    plane._stop_output_worker()


def scheduler_side_payloads(extra: dict, prompt: dict, tokens: list[int], capped: bool) -> list[OmniPayloadStruct]:
    """Everything the processor emits for one utterance on the scheduler-side transport.

    Stage 0 is played one sampled token at a time; every call sees the whole
    output history. It finishes on the stop token it samples next or, when
    ``capped``, on the last token itself, as a request that reaches
    ``max_tokens`` does, and that call is the finished one. The processor is
    then called once more for the finished request, which must send nothing.
    """
    transfer_manager = SimpleNamespace(
        code_prompt_token_ids=defaultdict(list),
        request_payload={},
        connector=SimpleNamespace(config=extra),
    )
    steps = [(tokens[:produced], False) for produced in range(1, len(tokens))]
    steps += [(tokens, True)] if capped else [(tokens, False), ([*tokens, STOP], True)]
    payloads = []
    for output_token_ids, finished in [*steps, steps[-1]]:
        # No ``last_output_token_id``: this transport's snapshot has none.
        request = SimpleNamespace(
            external_req_id="external-id",
            output_token_ids=output_token_ids,
            output_token_count=len(output_token_ids),
            additional_information=prompt["additional_information"],
        )
        payload = t3_to_s3gen_async_chunk(
            transfer_manager=transfer_manager, multimodal_output=None, request=request, is_finished=finished
        )
        if payload is not None:
            payloads.append(payload)
    assert payload is None
    # All state is in the two entries the transport frees with the request.
    assert set(vars(transfer_manager)) == {"code_prompt_token_ids", "request_payload", "connector"}
    assert set(transfer_manager.request_payload) == set(transfer_manager.code_prompt_token_ids) == {"external-id"}
    return payloads


def runner_side_payloads(
    plane: OmniRunnerDataPlane, prompt: dict, tokens: list[int], capped: bool
) -> list[OmniPayloadStruct]:
    """Everything the processor emits for one utterance on the runner-side transport.

    Each sampled token is one completed model step: the transport fences the
    stream at the stop token or the length cap, builds the request snapshot
    (the token history only until the first chunk was sent) and calls the
    processor. The finished call comes on its own afterwards, then cleanup.
    """
    attributes = set(vars(plane))
    plane.sent.clear()
    plane.register_request(
        SimpleNamespace(
            req_id="internal-id",
            external_req_id="external-id",
            prompt_token_ids=prompt["prompt_token_ids"],
            # As the scheduler hands it to the runner.
            additional_information=serialize_additional_information(prompt["additional_information"]),
            sampling_params=SimpleNamespace(stop_token_ids=[STOP], max_tokens=len(tokens) if capped else None),
        )
    )
    # A step's payload for the request: any key at all, see ``make_omni_output``.
    step_payload = {"meta.codec_streaming": torch.ones(1, dtype=torch.bool)}
    for token in tokens if capped else [*tokens, STOP]:
        plane.complete_outputs(req_ids=["internal-id"], inter_stage_outputs=[step_payload], sampled_token_ids=[[token]])
    # A step already in flight when the request stopped: its token is not the request's.
    plane.complete_outputs(req_ids=["internal-id"], inter_stage_outputs=[step_payload], sampled_token_ids=[[17]])
    plane.request_terminal({"internal-id"})
    # Nothing was added to the transport, and its cleanup freed both entries.
    assert set(vars(plane)) == attributes
    assert plane.request_payload == {} and plane.code_prompt_token_ids == {}
    return list(plane.sent)


def streamed_payloads(
    side: str, plane: OmniRunnerDataPlane, extra: dict, prompt: dict, tokens: list[int], capped: bool = False
) -> list[OmniPayloadStruct]:
    if side == "scheduler-side":
        return scheduler_side_payloads(extra, prompt, tokens, capped)
    return runner_side_payloads(plane, prompt, tokens, capped)


def chunk_sequence(payloads: list[OmniPayloadStruct]) -> list[tuple]:
    """What stage 1 is sent, in a form two runs can be compared in."""
    return [
        (
            payload.codes.audio.tolist(),
            payload.meta.left_context_size,
            bool(payload.meta.finished),
            None if payload.meta.stream_finished is None else bool(payload.meta.stream_finished),
            sorted(to_dict(payload).get("embed", {})),
        )
        for payload in payloads
    ]


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


@pytest.mark.parametrize("side", PLANES)
def test_streamed_chunks_decode_to_the_whole_utterance_001(
    side: str, plane: OmniRunnerDataPlane, connector_extra: dict, decoder: S3GenDecoder
) -> None:
    """The processor chunks as the deploy file says and emits payloads stage 1 accepts."""
    tokens = torch.randint(0, 6561, (60,)).tolist()

    payloads = streamed_payloads(side, plane, connector_extra, request_prompt(250), tokens)

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
    plane: OmniRunnerDataPlane, connector_extra: dict, prompt_tokens: int, capped: bool
) -> None:
    """No length makes the processor send a token's audio twice, skip one or finish twice.

    The processor sends a chunk as soon as a hop of unsent tokens and the
    lookahead after it exist, as the prefix through that lookahead, and moves
    its cursor by the hop alone. The lengths to fear are those where the
    utterance ends exactly there, so the finished call finds nothing unsent
    but the lookahead: 23 and 53 tokens for the 250-token reference (hops of
    20 and 30), 26 and 56 for the 202-token one (23 and 30).

    Every length is played through both transports. The scheduler-side one
    shows the processor the whole token history on every call and finishes
    on the call that brings the request's last token; the runner-side one
    shows the history only until the first chunk was sent, then one id per
    call, and finishes on a call of its own. They give the same chunks,
    except for a request cut off by its token cap exactly on a boundary:
    the scheduler-side transport learns of the end with the last token and
    sends one final chunk, the runner-side one has already sent that chunk
    as an ordinary one when it learns.
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
    boundaries = [sum(hops[:sent]) + lookahead for sent in range(1, len(hops)) if sum(hops[:sent]) + lookahead <= 240]
    prompt = request_prompt(prompt_tokens)
    tokens = torch.randint(0, 6561, (240,)).tolist()

    ended_on_a_boundary: dict[str, list[int]] = {side: [] for side in PLANES}
    for length in range(1, 241):
        sequences = []
        for side in PLANES:
            payloads = streamed_payloads(side, plane, connector_extra, prompt, tokens[:length], capped)
            sequences.append(chunk_sequence(payloads))
            codes = [payload.codes.audio.tolist() for payload in payloads]
            offsets = [payload.meta.left_context_size for payload in payloads]

            # One final payload, the last, holding the whole utterance; nothing after it.
            finals = [bool(payload.meta.stream_finished) for payload in payloads]
            assert finals == [bool(payload.meta.finished) for payload in payloads], (side, length)
            assert finals == [False] * (len(payloads) - 1) + [True], (side, length)
            assert codes[-1] == tokens[:length], (side, length)
            # Each payload is a prefix with something past its offset to decode.
            assert all(sent == tokens[: len(sent)] for sent in codes), (side, length)
            assert all(len(sent) > offset for sent, offset in zip(codes, offsets, strict=True)), (side, length)
            # The cursor starts at 0 and moves by the hops. A chunk that is not
            # the last ends exactly the lookahead past the next one's offset: the
            # next chunk decodes from where this one stopped, no token twice.
            assert offsets == [sum(hops[:sent]) for sent in range(len(payloads))], (side, length)
            assert [len(sent) - lookahead for sent in codes[:-1]] == offsets[1:], (side, length)
            # The reference travels once.
            assert [payload.embed is not None for payload in payloads] == [True] + [False] * (len(payloads) - 1), (
                side,
                length,
            )

            if len(payloads) > 1 and codes[-1] == codes[-2]:
                ended_on_a_boundary[side].append(length)
        assert (sequences[0] == sequences[1]) != (capped and length in boundaries), length

    # After an end on a boundary the last payload repeats the codes with a
    # new offset, and only the lookahead is left to decode.
    assert ended_on_a_boundary == {"scheduler-side": [] if capped else boundaries, "runner-side": boundaries}


@pytest.mark.parametrize("side", PLANES)
@pytest.mark.parametrize(("prompt_tokens", "length"), [(250, 53), (202, 56)])
def test_an_utterance_ending_on_a_chunk_boundary_decodes_to_its_own_length_001(
    side: str, plane: OmniRunnerDataPlane, connector_extra: dict, decoder: S3GenDecoder, prompt_tokens: int, length: int
) -> None:
    """The stop token's payload adds the lookahead's audio and nothing that was already played."""
    tokens = torch.randint(0, 6561, (length,)).tolist()

    payloads = streamed_payloads(side, plane, connector_extra, request_prompt(prompt_tokens), tokens)

    assert payloads[-1].codes.audio.tolist() == payloads[-2].codes.audio.tolist() == tokens
    assert decoded_samples(decoder, payloads) == 2 * (length + 3) * SAMPLES_PER_FRAME
    assert decoder.streams == {}


@pytest.mark.parametrize("side", PLANES)
def test_an_utterance_with_no_speech_tokens_ends_with_explicit_empty_codes_001(
    side: str, plane: OmniRunnerDataPlane, connector_extra: dict
) -> None:
    """Stage 1 was started on a placeholder prompt, and only an explicit empty ``codes.audio`` clears it.

    The payload is put through the receiving transport's own metadata
    extraction and the receiving scheduler's update, as a chunk that arrives
    is.
    """
    (terminal,) = streamed_payloads(side, plane, connector_extra, request_prompt(250), [])
    payload = to_dict(terminal)

    assert payload["codes"]["audio"].numel() == 0 and bool(payload["meta"]["finished"])
    assert not Transport._payload_is_consumable(payload)
    receiver = SimpleNamespace(
        request_id="r",
        external_req_id="r",
        prompt_token_ids=[0] * 88,
        num_prompt_tokens=88,
        _all_token_ids=[0] * 88,
        _output_token_ids=[],
        num_computed_tokens=0,
    )
    coordinator = OmniSchedulingCoordinator(stage_id=1)
    coordinator.update_request_metadata(
        {"r": receiver}, {"r": Transport._extract_scheduling_metadata(payload)}, model_mode="generation"
    )
    assert receiver.prompt_token_ids == [] and receiver.num_prompt_tokens == 0
    assert "r" in coordinator.input_terminal_req_ids


def test_a_stream_aborted_between_chunks_is_flushed_and_freed_001(
    plane: OmniRunnerDataPlane, connector_extra: dict
) -> None:
    """An abort reaches the processor as the same finished call, and cleanup follows it."""
    prompt = request_prompt(250)
    tokens = torch.randint(0, 6561, (40,)).tolist()
    plane.register_request(
        SimpleNamespace(
            req_id="internal-id",
            external_req_id="external-id",
            prompt_token_ids=prompt["prompt_token_ids"],
            additional_information=prompt["additional_information"],
            sampling_params=SimpleNamespace(stop_token_ids=[STOP], max_tokens=None),
        )
    )
    for token in tokens:
        plane.complete_outputs(
            req_ids=["internal-id"],
            inter_stage_outputs=[{"meta.codec_streaming": torch.ones(1, dtype=torch.bool)}],
            sampled_token_ids=[[token]],
        )
    assert set(plane.request_payload) == set(plane.code_prompt_token_ids) == {"external-id"}

    plane.abort_requests({"internal-id"})

    assert chunk_sequence(plane.sent) == [
        (tokens[:23], 0, False, False, ["embedding", "speech_feat", "speech_token"]),
        (tokens, 20, True, True, []),
    ]
    assert plane.request_payload == {} and plane.code_prompt_token_ids == {}


def test_tokens_sampled_but_never_shown_are_an_error_001(connector_extra: dict) -> None:
    """The count ran ahead of the calls and the history is gone: their audio would be missing."""
    transfer_manager = SimpleNamespace(
        code_prompt_token_ids=defaultdict(list),
        request_payload={},
        connector=SimpleNamespace(config=connector_extra),
    )
    info = request_prompt(250)["additional_information"]

    def call(output_token_ids: list[int], count: int, last: int) -> OmniPayloadStruct | None:
        request = SimpleNamespace(
            external_req_id="external-id",
            output_token_ids=output_token_ids,
            output_token_count=count,
            last_output_token_id=last,
            additional_information=info,
        )
        return t3_to_s3gen_async_chunk(transfer_manager, None, request)

    assert call([10, 11, 12], 3, 12) is None
    assert call([], 4, 13) is None
    assert transfer_manager.code_prompt_token_ids["external-id"] == [10, 11, 12, 13]
    with pytest.raises(RuntimeError, match="external-id has sampled 6 tokens, its chunk processor took 4"):
        call([], 6, 15)


@pytest.mark.parametrize(
    "key",
    ["codec_chunk_frames", "codec_pre_lookahead_frames", "codec_max_chunk_frames", "codec_stream_scale_factor"],
)
def test_a_deploy_file_without_a_chunk_setting_is_an_error_001(connector_extra: dict, key: str) -> None:
    """The schedule is the deploy file's; no setting has a default to fall back on."""
    extra = {name: value for name, value in connector_extra.items() if name != key}
    transfer_manager = SimpleNamespace(
        code_prompt_token_ids=defaultdict(list), request_payload={}, connector=SimpleNamespace(config=extra)
    )
    request = SimpleNamespace(
        external_req_id="external-id",
        output_token_ids=[10],
        output_token_count=1,
        additional_information=request_prompt(250)["additional_information"],
    )
    with pytest.raises(KeyError, match=key):
        t3_to_s3gen_async_chunk(transfer_manager, None, request)
