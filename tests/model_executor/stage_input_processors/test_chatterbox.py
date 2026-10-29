# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Chatterbox's stage handoff on both transports, as the engine builds them.

Stage 0's two transports are built from the deploy file's stage 0 config:
the scheduler-side ``OmniChunkTransferAdapter`` and Model Runner V2's
``OmniRunnerDataPlane``. Each sends through its own shared-memory
connector; stage 1's connector, built from stage 1's config, reads what
arrives. Requests are the engine's own request objects.
"""

import time
import uuid
from collections.abc import Iterator

import pytest
import torch
from vllm.outputs import CompletionOutput, RequestOutput
from vllm.sampling_params import SamplingParams
from vllm.v1.request import RequestStatus

from tests.helpers.stage_config import get_deploy_config_path, modify_stage_config
from tests.model_executor.models.chatterbox.helpers import (
    DEPLOY,
    LEFT_CONTEXT,
    SEED,
    random_voice,
    stage_vllm_config,
    write_model_dir,
)
from vllm_omni.core.sched.omni_scheduling_coordinator import OmniSchedulingCoordinator
from vllm_omni.core.sched.output import OmniNewRequestData
from vllm_omni.data_entry_keys import OmniPayloadStruct, to_dict, to_struct
from vllm_omni.distributed.omni_connectors.connectors.shm_connector import SharedMemoryConnector
from vllm_omni.distributed.omni_connectors.transfer_adapter.chunk_transfer_adapter import OmniChunkTransferAdapter
from vllm_omni.engine.serialization import serialize_additional_information
from vllm_omni.model_executor.models.chatterbox.chatterbox_s3gen import SAMPLES_PER_FRAME, S3GenDecoder
from vllm_omni.model_executor.models.chatterbox.conditioning import build_prompt
from vllm_omni.model_executor.stage_input_processors.chatterbox import t3_to_s3gen, t3_to_s3gen_async_chunk
from vllm_omni.request import OmniRequest
from vllm_omni.transformers_utils.configs.chatterbox import ChatterboxConfig
from vllm_omni.worker_v2.omni_data_plane import OmniRunnerDataPlane

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

STOP = ChatterboxConfig().stop_speech_token
PLANES = ["scheduler-side", "runner-side"]
# Long enough for every utterance below to end on its stop token or its own cap.
MAX_TOKENS = 1000
# How long the scheduler-side transport's sender thread may take for one chunk.
DELIVERY_SECONDS = 10.0


def request_prompt(prompt_tokens: int) -> dict:
    return build_prompt([5, 6, 7], random_voice(375, prompt_tokens), ChatterboxConfig())


def talker_output(token_ids: list[int], finished: bool = True) -> RequestOutput:
    """Stage 0's output as the orchestrator hands it over.

    vLLM-Omni's output processor gives every completion the request's
    ``cumulative_token_ids``; stage 0 has no detokenizer, so they are the
    completion's own ids.
    """
    completion = CompletionOutput(
        index=0,
        text="",
        token_ids=token_ids,
        cumulative_logprob=None,
        logprobs=None,
        finish_reason="stop" if finished else None,
    )
    completion.cumulative_token_ids = list(token_ids)
    return RequestOutput(
        request_id="r", prompt=None, prompt_token_ids=[], prompt_logprobs=None, outputs=[completion], finished=finished
    )


def stage0_request(prompt: dict, external_id: str, max_tokens: int = MAX_TOKENS) -> OmniRequest:
    """Stage 0's request as the engine core holds it."""
    return OmniRequest(
        request_id="internal-id",
        prompt_token_ids=prompt["prompt_token_ids"],
        sampling_params=SamplingParams(stop_token_ids=[STOP], max_tokens=max_tokens),
        pooling_params=None,
        external_req_id=external_id,
        additional_information=serialize_additional_information(prompt["additional_information"]),
    )


def test_codes_are_the_valid_speech_tokens() -> None:
    """The stop token, and any placeholder echoed from the prompt, never reach the decoder."""
    (stage_input,) = t3_to_s3gen([talker_output([6561, 6561, 10, 20, 6560, 6562])], request_prompt(250))
    assert stage_input["prompt_token_ids"] == [10, 20, 6560]


def test_payload_is_one_final_chunk_in_the_requests_own_voice() -> None:
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


def test_unfinished_outputs_are_skipped() -> None:
    assert t3_to_s3gen([talker_output([10, 20], finished=False)], request_prompt(250)) == []


def test_an_utterance_with_no_speech_tokens_gets_no_stage_1_request() -> None:
    """The orchestrator ends a request that has no next-stage input with empty audio.

    Raising here instead stops the orchestrator, and with it every other
    request; streaming ends the same utterance with an empty final chunk.
    """
    prompt = request_prompt(250)
    assert t3_to_s3gen([talker_output([6562])], prompt) == []
    # Only that utterance is left out.
    (stage_input,) = t3_to_s3gen([talker_output([6562]), talker_output([10, 6562])], prompt)
    assert stage_input["prompt_token_ids"] == [10]


@pytest.fixture(scope="module")
def stage_configs(tmp_path_factory: pytest.TempPathFactory) -> list:
    """Both stages' engine configs, from the deploy file, for a model directory of their own."""
    model_dir = write_model_dir(tmp_path_factory.mktemp("chatterbox"))
    return [stage_vllm_config(model_dir, DEPLOY["turbo"], stage_id) for stage_id in (0, 1)]


@pytest.fixture(scope="module")
def decoder() -> S3GenDecoder:
    return S3GenDecoder(ChatterboxConfig(), 2048, LEFT_CONTEXT, SEED).eval()


@pytest.fixture
def plane(stage_configs: list) -> Iterator[OmniRunnerDataPlane]:
    """Stage 0's runner-side transport, as Model Runner V2 builds it from stage 0's config."""
    plane = OmniRunnerDataPlane(vllm_config=stage_configs[0], model_config=stage_configs[0].model_config)
    yield plane
    plane.close()


@pytest.fixture
def adapter(stage_configs: list) -> Iterator[OmniChunkTransferAdapter]:
    """Stage 0's scheduler-side transport, as the scheduler builds it from stage 0's config."""
    adapter = OmniChunkTransferAdapter(stage_configs[0])
    yield adapter
    adapter.shutdown()


@pytest.fixture
def receiver(stage_configs: list) -> Iterator[SharedMemoryConnector]:
    """Stage 1's connector, which reads what stage 0 sends."""
    connector = OmniChunkTransferAdapter.create_connector(stage_configs[1].model_config)
    yield connector
    connector.close()


def received(receiver: SharedMemoryConnector, external_id: str, first: int) -> list[OmniPayloadStruct]:
    """Every chunk of the request that has arrived, from chunk ``first`` on, as stage 1 reads them."""
    payloads: list[OmniPayloadStruct] = []
    while (chunk := receiver.get("0", "1", f"{external_id}_0_{first + len(payloads)}")) is not None:
        payloads.append(to_struct(chunk[0]))
    return payloads


def receive_through_final(receiver: SharedMemoryConnector, external_id: str) -> list[OmniPayloadStruct]:
    """Stage 1's reads until the request's final chunk arrives from a sender thread."""
    payloads: list[OmniPayloadStruct] = []
    deadline = time.monotonic() + DELIVERY_SECONDS
    while not payloads or not bool(payloads[-1].meta.finished):
        assert time.monotonic() < deadline, f"no final chunk after {len(payloads)} chunks"
        arrived = received(receiver, external_id, len(payloads))
        payloads += arrived
        if not arrived:
            time.sleep(0.001)
    return payloads


def test_the_whole_utterance_decodes_through_stage_1(decoder: S3GenDecoder) -> None:
    """With ``async_chunk: false`` stage 1 decodes the processor's prompt as a stream of one final chunk."""
    torch.manual_seed(15)
    tokens = torch.randint(0, 6561, (40,)).tolist()

    (stage_input,) = t3_to_s3gen([talker_output([*tokens, STOP])], request_prompt(202))
    (audio,) = decoder.decode_step(
        torch.tensor(stage_input["prompt_token_ids"]),
        [len(stage_input["prompt_token_ids"])],
        [stage_input["additional_information"]],
        ["x"],
    )

    assert audio.shape == (2 * (40 + 3) * SAMPLES_PER_FRAME,)
    assert decoder.streams == {}


def scheduler_side_payloads(
    adapter: OmniChunkTransferAdapter, receiver: SharedMemoryConnector, prompt: dict, tokens: list[int], capped: bool
) -> list[OmniPayloadStruct]:
    """Everything stage 1 receives for one utterance from the scheduler-side transport.

    The scheduler hands the transport the request after every sampled token,
    with the token appended. It finishes on the stop token it samples next
    or, when ``capped``, on the last token itself, as a request that reaches
    ``max_tokens`` does; that step's call is the finished one.
    """
    external_id = f"external-{uuid.uuid4().hex}"
    attributes = set(vars(adapter))
    request = stage0_request(prompt, external_id, len(tokens) if capped else MAX_TOKENS)
    for token in tokens if capped else [*tokens, STOP]:
        request.append_output_token_ids(token)
        request.num_computed_tokens = request.num_prompt_tokens + request.num_output_tokens - 1
        if token == STOP:
            request.status = RequestStatus.FINISHED_STOPPED
        elif request.num_output_tokens == request.max_tokens:
            request.status = RequestStatus.FINISHED_LENGTH_CAPPED
        adapter.save_async(None, request, new_token_ids=[token])
    payloads = receive_through_final(receiver, external_id)
    # Nothing was added to the transport; its cleanup after the final chunk freed both entries.
    assert set(vars(adapter)) == attributes
    assert external_id not in adapter.request_payload and external_id not in adapter.code_prompt_token_ids
    assert received(receiver, external_id, len(payloads)) == []
    return payloads


def runner_side_payloads(
    plane: OmniRunnerDataPlane, receiver: SharedMemoryConnector, prompt: dict, tokens: list[int], capped: bool
) -> list[OmniPayloadStruct]:
    """Everything stage 1 receives for one utterance from the runner-side transport.

    Each sampled token is one completed model step: the transport fences the
    stream at the stop token or the length cap, builds the request snapshot
    (the token history only until the first chunk was sent) and calls the
    processor. The finished call comes on its own afterwards, then cleanup.
    """
    external_id = f"external-{uuid.uuid4().hex}"
    attributes = set(vars(plane))
    request = stage0_request(prompt, external_id, len(tokens) if capped else MAX_TOKENS)
    plane.register_request(
        OmniNewRequestData.from_request(request, ([],), prefill_token_ids=prompt["prompt_token_ids"])
    )
    # A decode step's payload for the request, as stage 0's ``make_omni_output`` leaves the runner.
    step_payload = {"meta.codec_frame_valid": torch.zeros(1, dtype=torch.bool)}
    payloads: list[OmniPayloadStruct] = []
    for token in tokens if capped else [*tokens, STOP]:
        plane.complete_outputs(req_ids=["internal-id"], inter_stage_outputs=[step_payload], sampled_token_ids=[[token]])
        payloads += received(receiver, external_id, len(payloads))
    # A step already in flight when the request stopped: its token is not the request's.
    plane.complete_outputs(req_ids=["internal-id"], inter_stage_outputs=[step_payload], sampled_token_ids=[[17]])
    plane.request_terminal({"internal-id"})
    payloads += received(receiver, external_id, len(payloads))
    # Nothing was added to the transport, and its cleanup freed both entries.
    assert set(vars(plane)) == attributes
    assert plane.request_payload == {} and plane.code_prompt_token_ids == {}
    return payloads


def streamed_payloads(
    side: str,
    adapter: OmniChunkTransferAdapter,
    plane: OmniRunnerDataPlane,
    receiver: SharedMemoryConnector,
    prompt: dict,
    tokens: list[int],
    capped: bool = False,
) -> list[OmniPayloadStruct]:
    if side == "scheduler-side":
        return scheduler_side_payloads(adapter, receiver, prompt, tokens, capped)
    return runner_side_payloads(plane, receiver, prompt, tokens, capped)


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


def decoded_samples(side: str, decoder: S3GenDecoder, payloads: list[OmniPayloadStruct]) -> int:
    """How many samples stage 1 returns for one request's payloads.

    The scheduler-side transport merges each chunk into the request's
    payload, so stage 1 sees the first chunk's voice again on every later
    one. The runner-side transport hands over each chunk's payload alone.
    """
    merged: dict = {}
    samples = 0
    for payload in payloads:
        merged = {**merged, **to_dict(payload)} if side == "scheduler-side" else to_dict(payload)
        codes = payload.codes.audio
        (audio,) = decoder.decode_step(codes, [codes.numel()], [merged], ["scheduler-id"])
        samples += audio.numel()
    return samples


@pytest.mark.parametrize("side", PLANES)
def test_streamed_chunks_decode_to_the_whole_utterance(
    side: str,
    adapter: OmniChunkTransferAdapter,
    plane: OmniRunnerDataPlane,
    receiver: SharedMemoryConnector,
    decoder: S3GenDecoder,
) -> None:
    """The processor chunks as the deploy file says and emits payloads stage 1 accepts."""
    tokens = torch.randint(0, 6561, (60,)).tolist()

    payloads = streamed_payloads(side, adapter, plane, receiver, request_prompt(250), tokens)

    # First hop 15 plus 5 to align the 250-token reference, then 30; every
    # chunk but the last waits for the 3-token lookahead.
    assert [
        (payload.codes.audio.numel(), payload.meta.left_context_size, bool(payload.meta.stream_finished))
        for payload in payloads
    ] == [(23, 0, False), (53, 20, False), (60, 50, True)]
    assert decoded_samples(side, decoder, payloads) == 2 * (60 + 3) * SAMPLES_PER_FRAME
    assert decoder.streams == {}


# Per reference length, the deploy file's hops (15, growing twice over to 60,
# the first one padded to align the reference to 15) and the utterance
# lengths up to 240 that end exactly the lookahead past a hop.
SCHEDULES = {
    250: ([20, 30, 60, 60, 60, 60], [23, 53, 113, 173, 233]),
    202: ([23, 30, 60, 60, 60, 60], [26, 56, 116, 176, 236]),
}


@pytest.mark.parametrize("capped", [False, True], ids=["stop token", "length cap"])
@pytest.mark.parametrize("prompt_tokens", [250, 202])
def test_every_utterance_length_is_sent_once_and_finished_once(
    adapter: OmniChunkTransferAdapter,
    plane: OmniRunnerDataPlane,
    receiver: SharedMemoryConnector,
    prompt_tokens: int,
    capped: bool,
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
    lookahead = ChatterboxConfig().pre_lookahead_len
    hops, boundaries = SCHEDULES[prompt_tokens]
    prompt = request_prompt(prompt_tokens)
    tokens = torch.randint(0, 6561, (240,)).tolist()

    ended_on_a_boundary: dict[str, list[int]] = {side: [] for side in PLANES}
    for length in range(1, 241):
        sequences = []
        for side in PLANES:
            payloads = streamed_payloads(side, adapter, plane, receiver, prompt, tokens[:length], capped)
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
def test_an_utterance_ending_on_a_chunk_boundary_decodes_to_its_own_length(
    side: str,
    adapter: OmniChunkTransferAdapter,
    plane: OmniRunnerDataPlane,
    receiver: SharedMemoryConnector,
    decoder: S3GenDecoder,
    prompt_tokens: int,
    length: int,
) -> None:
    """The stop token's payload adds the lookahead's audio and nothing that was already played."""
    tokens = torch.randint(0, 6561, (length,)).tolist()

    payloads = streamed_payloads(side, adapter, plane, receiver, request_prompt(prompt_tokens), tokens)

    assert payloads[-1].codes.audio.tolist() == payloads[-2].codes.audio.tolist() == tokens
    assert decoded_samples(side, decoder, payloads) == 2 * (length + 3) * SAMPLES_PER_FRAME
    assert decoder.streams == {}


def receive_on_the_runner_side(terminal: OmniPayloadStruct, stage1_request: OmniRequest, stage_configs: list) -> None:
    """Model Runner V2's native data plane: stage 1's metadata extraction and the scheduler's update."""
    payload = to_dict(terminal)
    plane = OmniRunnerDataPlane(vllm_config=stage_configs[1], model_config=stage_configs[1].model_config)
    try:
        assert not plane._payload_is_consumable(payload)
        update = plane._extract_scheduling_metadata_update(payload)
    finally:
        plane.close()
    coordinator = OmniSchedulingCoordinator(stage_id=1)
    coordinator.update_request_metadata({"r": stage1_request}, {"r": update})
    assert "r" in coordinator.input_terminal_req_ids


def receive_on_the_scheduler_side(
    terminal: OmniPayloadStruct, stage1_request: OmniRequest, stage_configs: list, receiver: SharedMemoryConnector
) -> None:
    """The default runner: stage 1's OmniChunkTransferAdapter reads the chunk from its connector."""
    stage1 = OmniChunkTransferAdapter(stage_configs[1])
    try:
        receiver.put("0", "1", f"{stage1_request.external_req_id}_0_0", to_dict(terminal))
        stage1.load_async(stage1_request)
        deadline = time.monotonic() + DELIVERY_SECONDS
        while not stage1.is_done_receiving_chunks("r"):
            assert time.monotonic() < deadline, "stage 1 did not receive the terminal chunk"
            time.sleep(0.001)
    finally:
        stage1.shutdown()


@pytest.mark.parametrize("receiving_side", PLANES)
@pytest.mark.parametrize("side", PLANES)
def test_an_utterance_with_no_speech_tokens_ends_with_explicit_empty_codes(
    side: str,
    receiving_side: str,
    adapter: OmniChunkTransferAdapter,
    plane: OmniRunnerDataPlane,
    receiver: SharedMemoryConnector,
    stage_configs: list,
) -> None:
    """Stage 1 was started on a placeholder prompt, and only an explicit empty ``codes.audio`` clears it.

    The terminal payload is received as stage 1 receives any chunk, on
    either data plane: the request's prompt becomes empty and its upstream
    is done.
    """
    (terminal,) = streamed_payloads(side, adapter, plane, receiver, request_prompt(250), [])
    assert terminal.codes.audio.numel() == 0 and bool(terminal.meta.finished)
    stage1_request = OmniRequest(
        request_id="r",
        prompt_token_ids=[0] * 88,
        sampling_params=SamplingParams(max_tokens=1),
        pooling_params=None,
        external_req_id=f"stage-1-{uuid.uuid4().hex}",
    )

    if receiving_side == "runner-side":
        receive_on_the_runner_side(terminal, stage1_request, stage_configs)
    else:
        receive_on_the_scheduler_side(terminal, stage1_request, stage_configs, receiver)

    assert stage1_request.prompt_token_ids == [] and stage1_request.num_prompt_tokens == 0


def test_a_stream_aborted_between_chunks_is_flushed_and_freed(
    plane: OmniRunnerDataPlane, receiver: SharedMemoryConnector
) -> None:
    """An abort reaches the processor as the same finished call, and cleanup follows it."""
    prompt = request_prompt(250)
    tokens = torch.randint(0, 6561, (40,)).tolist()
    external_id = f"external-{uuid.uuid4().hex}"
    plane.register_request(
        OmniNewRequestData.from_request(
            stage0_request(prompt, external_id), ([],), prefill_token_ids=prompt["prompt_token_ids"]
        )
    )
    for token in tokens:
        plane.complete_outputs(
            req_ids=["internal-id"],
            inter_stage_outputs=[{"meta.codec_frame_valid": torch.zeros(1, dtype=torch.bool)}],
            sampled_token_ids=[[token]],
        )
    assert set(plane.request_payload) == set(plane.code_prompt_token_ids) == {external_id}

    plane.abort_requests({"internal-id"})

    assert chunk_sequence(received(receiver, external_id, 0)) == [
        (tokens[:23], 0, False, False, ["embedding", "speech_feat", "speech_token"]),
        (tokens, 20, True, True, []),
    ]
    assert plane.request_payload == {} and plane.code_prompt_token_ids == {}


def test_a_finished_request_is_sent_its_terminal_chunk_once(adapter: OmniChunkTransferAdapter) -> None:
    """Called again for a request it has finished, the processor sends nothing more.

    The processor is called the way the scheduler-side transport calls it,
    with the transport itself and the engine's request.
    """
    external_id = f"external-{uuid.uuid4().hex}"
    request = stage0_request(request_prompt(250), external_id)
    for token in (10, 11, STOP):
        request.append_output_token_ids(token)
    request.status = RequestStatus.FINISHED_STOPPED

    def call() -> OmniPayloadStruct | None:
        return t3_to_s3gen_async_chunk(
            transfer_manager=adapter, multimodal_output=None, request=request, is_finished=True
        )

    terminal = call()

    assert terminal is not None and terminal.codes is not None and terminal.meta is not None
    assert terminal.codes.audio.tolist() == [10, 11] and bool(terminal.meta.finished)
    assert call() is None
    adapter.cleanup(request.request_id, external_id)


def test_tokens_sampled_but_never_shown_are_an_error(plane: OmniRunnerDataPlane) -> None:
    """The count ran ahead of the calls and the history is gone: their audio would be missing.

    The snapshots are the runner-side transport's own: with the token history
    until the first chunk is sent and without it afterwards. The transport
    accepts tokens without calling the processor in between, as a skipped
    call would leave it.
    """
    prompt = request_prompt(250)
    external_id = f"external-{uuid.uuid4().hex}"
    plane.register_request(
        OmniNewRequestData.from_request(
            stage0_request(prompt, external_id), ([],), prefill_token_ids=prompt["prompt_token_ids"]
        )
    )
    state = plane._native_requests["internal-id"]

    def call(accepted: list[int], history: bool) -> OmniPayloadStruct | None:
        state.accept_tokens(accepted)
        snapshot = state.snapshot(include_token_history=history)
        return t3_to_s3gen_async_chunk(plane, None, snapshot)

    assert call([10, 11, 12], history=True) is None
    assert call([13], history=False) is None
    assert plane.code_prompt_token_ids[external_id] == [10, 11, 12, 13]
    with pytest.raises(RuntimeError, match=f"{external_id} has sampled 6 tokens, its chunk processor took 4"):
        call([14, 15], history=False)


@pytest.mark.parametrize("key", ["codec_chunk_frames", "codec_max_chunk_frames", "codec_stream_scale_factor"])
def test_a_deploy_file_without_a_chunk_setting_is_an_error(stage_configs: list, key: str) -> None:
    """The schedule is the deploy file's; no setting has a default to fall back on.

    The deploy file lacks the setting, and stage 0's scheduler-side
    transport is built from it.
    """
    deploy = modify_stage_config(
        get_deploy_config_path(DEPLOY["turbo"]), deletes={f"connectors.connector_of_shared_memory.extra.{key}": None}
    )
    config = stage_vllm_config(stage_configs[0].model_config.model, deploy, 0)
    assert key not in config.model_config.stage_connector_config["extra"]
    adapter = OmniChunkTransferAdapter(config)
    request = stage0_request(request_prompt(250), f"external-{uuid.uuid4().hex}")
    request.append_output_token_ids(10)
    try:
        with pytest.raises(KeyError, match=key):
            t3_to_s3gen_async_chunk(adapter, None, request)
    finally:
        adapter.shutdown()


def test_two_requests_interleaved_keep_their_own_schedule_and_voice(
    adapter: OmniChunkTransferAdapter, receiver: SharedMemoryConnector
) -> None:
    """Two requests with references of different lengths, their tokens arriving in turn.

    Each request's first hop aligns its own reference (20 and 23 tokens),
    each sends its own voice once, and finishing one leaves the other's
    state and schedule as they were. The short request reaches its 40-token
    cap while the long one is mid-stream.
    """
    torch.manual_seed(16)
    prompts = {"long": request_prompt(250), "short": request_prompt(202)}
    tokens = {name: torch.randint(0, 6561, (60,)).tolist() for name in prompts}
    requests = {
        name: stage0_request(prompt, f"{name}-{uuid.uuid4().hex}", max_tokens)
        for (name, prompt), max_tokens in zip(prompts.items(), (60, 40), strict=True)
    }

    def feed(name: str) -> None:
        request = requests[name]
        token = tokens[name][request.num_output_tokens]
        request.append_output_token_ids(token)
        request.num_computed_tokens = request.num_prompt_tokens + request.num_output_tokens - 1
        if request.num_output_tokens == request.max_tokens:
            request.status = RequestStatus.FINISHED_LENGTH_CAPPED
        adapter.save_async(None, request, new_token_ids=[token])

    for _ in range(40):
        for name in prompts:
            feed(name)
    sent = {"short": receive_through_final(receiver, requests["short"].external_req_id)}
    for _ in range(20):
        feed("long")
    sent["long"] = receive_through_final(receiver, requests["long"].external_req_id)

    assert [(len(payload.codes.audio), payload.meta.left_context_size) for payload in sent["long"]] == [
        (23, 0),
        (53, 20),
        (60, 50),
    ]
    assert [(len(payload.codes.audio), payload.meta.left_context_size) for payload in sent["short"]] == [
        (26, 0),
        (40, 23),
    ]
    for name, payloads in sent.items():
        assert all(payload.codes.audio.tolist() == tokens[name][: len(payload.codes.audio)] for payload in payloads)
        reference = prompts[name]["additional_information"]["embed"]
        assert torch.equal(payloads[0].embed.speech_token, reference["speech_token"])
        assert torch.equal(payloads[0].embed.embedding, reference["embedding"])
        assert [payload.embed is None for payload in payloads[1:]] == [True] * (len(payloads) - 1)
