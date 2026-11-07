# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Predictor ownership, bounded windows, exact delta intervals and cleanup."""

from dataclasses import dataclass, field

import pytest
import torch
from torch import nn

from vllm_omni.model_executor.models.soprano.soprano_decoder import SopranoDecoderForGeneration
from vllm_omni.model_executor.models.soprano.streaming import CONTEXT, DecoderWindow
from vllm_omni.model_executor.models.soprano.vocos import SopranoDecoder
from vllm_omni.model_executor.stage_input_processors.soprano import ar_to_decoder_async_chunk

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


class IntervalDecoder(nn.Module):
    """Independent oracle: each interval is tagged with its left input value."""

    def __init__(self):
        super().__init__()
        self.head = nn.Module()
        self.head.out = nn.Linear(1, 1)

    def forward(self, hidden):
        return hidden[:, 0, :-1].repeat_interleave(2048, dim=-1)


def states(length, offset=0):
    return (torch.arange(length * 512, dtype=torch.float32).reshape(length, 512) + 1 + offset) / 1024


@pytest.mark.parametrize("length", [0, 1, 2, 3, 4, 5, 8, 9, 37])
@pytest.mark.parametrize("chunk_size", [1, 2, 4, 7])
@pytest.mark.parametrize("empty_flush", [False, True])
def test_delta_intervals_have_no_replay_or_missing_tail(length, chunk_size, empty_flush):
    hidden = states(length)
    state = DecoderWindow(chunk_size)
    chunks = []
    if length < 2:
        with pytest.raises(ValueError, match="at least two"):
            state.push(IntervalDecoder(), hidden, finished=True)
        return
    decoder = IntervalDecoder()
    for start in range(0, length, chunk_size):
        finished = not empty_flush and start + chunk_size >= length
        chunks.append(state.push(decoder, hidden[start : start + chunk_size], finished=finished))
        if not finished:
            assert state.hidden.shape[0] <= 2 * CONTEXT + chunk_size
    if empty_flush:
        chunks.append(state.push(decoder, hidden[:0], finished=True))
    assert torch.equal(torch.cat(chunks), hidden[:-1, 0].repeat_interleave(2048))
    assert state.emitted == (length - 1) * 2048
    assert state.closed and state.hidden is None
    with pytest.raises(ValueError, match="already finished"):
        state.push(decoder, hidden[:0], finished=True)


def wrapper(decoder):
    result = object.__new__(SopranoDecoderForGeneration)
    nn.Module.__init__(result)
    result.decoder = decoder
    result._streams = {}
    result._failed_streams = set()
    return result


def payload(hidden, *, finished=False, chunk_size=1, error=None):
    return {
        "hidden_states": {"output": hidden},
        "meta": {"stream_finished": finished},
        "additional_information": {
            "soprano_stream": True,
            "soprano_chunk_size": chunk_size,
            "soprano_error": error,
        },
    }


def run_isolated(decoder, hidden):
    state = DecoderWindow(1)
    return [state.push(decoder, row[None], finished=i == len(hidden) - 1) for i, row in enumerate(hidden)]


@pytest.mark.usefixtures("single_threaded_cpu")
def test_nonzero_real_decoder_interleaving_and_batch_row_swaps():
    # Compare identical operations, so no invented floating-point tolerance.
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(12)
        decoder = SopranoDecoder().eval()
    first = states(9)
    second = -states(9, offset=731)
    with torch.inference_mode():
        expected = {"a": run_isolated(decoder, first), "b": run_isolated(decoder, second)}
        model = wrapper(decoder)
        for i in range(9):
            ids = ["a", "b"] if i % 2 else ["b", "a"]
            rows = {"a": first[i : i + 1], "b": second[i : i + 1]}
            output = model(
                runtime_additional_information=[payload(rows[k], finished=i == 8) for k in ids], request_ids=ids
            ).multimodal_outputs
            for row, key in enumerate(ids):
                assert torch.equal(output["model_outputs"][row], expected[key][i])
                assert not bool(output["audio_synthesis_error"][row])
        assert not model._streams
        assert not torch.equal(torch.cat(expected["a"]), torch.cat(expected["b"]))


def test_abort_error_finish_and_request_slot_reuse():
    model = wrapper(IntervalDecoder())
    model(runtime_additional_information=[payload(states(1))], request_ids=["same-slot"])
    assert "same-slot" in model._streams
    model.on_requests_finished(["same-slot"])
    assert not model._streams
    recovered = []
    for i, hidden in enumerate(states(5, offset=10000)):
        output = model(
            runtime_additional_information=[payload(hidden[None], finished=i == 4)], request_ids=["same-slot"]
        ).multimodal_outputs
        recovered.append(output["model_outputs"][0])
    assert torch.equal(torch.cat(recovered), states(5, offset=10000)[:-1, 0].repeat_interleave(2048))
    assert not model._streams
    output = model(runtime_additional_information=[payload(states(1), finished=True)], request_ids=["short"])
    assert bool(output.multimodal_outputs["audio_synthesis_error"][0])
    assert not model._streams
    output = model(runtime_additional_information=[payload(states(0), error="bad handoff")], request_ids=["bad"])
    assert bool(output.multimodal_outputs["audio_synthesis_error"][0])
    assert not model._streams


@pytest.mark.parametrize("error_finished", [False, True])
def test_failed_stream_stays_failed_until_request_cleanup(error_finished):
    model = wrapper(IntervalDecoder())
    output = model(
        runtime_additional_information=[payload(states(4)[:, :511], chunk_size=4, finished=error_finished)],
        request_ids=["failed"],
    ).multimodal_outputs
    assert bool(output["audio_synthesis_error"][0])
    assert output["model_outputs"][0].numel() == 0
    assert not model._streams

    peer = states(8, offset=10000)
    peer_audio = []
    for step in range(2):
        ids = ["failed", "peer"] if step == 0 else ["peer", "failed"]
        rows = {
            "failed": payload(states(4), chunk_size=4, finished=step == 1),
            "peer": payload(peer[4 * step : 4 * (step + 1)], chunk_size=4, finished=step == 1),
        }
        output = model(runtime_additional_information=[rows[key] for key in ids], request_ids=ids).multimodal_outputs
        failed_row, peer_row = ids.index("failed"), ids.index("peer")
        assert bool(output["audio_synthesis_error"][failed_row])
        assert output["model_outputs"][failed_row].numel() == 0
        assert not bool(output["audio_synthesis_error"][peer_row])
        peer_audio.append(output["model_outputs"][peer_row])
        assert "failed" not in model._streams
    assert torch.equal(torch.cat(peer_audio), peer[:-1, 0].repeat_interleave(2048))
    assert model._failed_streams == {"failed"}

    model.on_requests_finished(["failed", "peer"])
    assert not model._streams and not model._failed_streams
    recovered = []
    for step in range(2):
        output = model(
            runtime_additional_information=[payload(peer[4 * step : 4 * (step + 1)], chunk_size=4, finished=step == 1)],
            request_ids=["failed"],
        ).multimodal_outputs
        assert not bool(output["audio_synthesis_error"][0])
        recovered.append(output["model_outputs"][0])
    assert torch.equal(torch.cat(recovered), peer[:-1, 0].repeat_interleave(2048))
    assert not model._streams and not model._failed_streams


@dataclass
class Connector:
    config: dict = field(default_factory=lambda: {"extra": {"soprano_chunk_size": 4}})


@dataclass
class Transfer:
    request_payload: dict = field(default_factory=dict)
    connector: Connector = field(default_factory=Connector)


@dataclass
class Request:
    external_req_id: str = "request"
    output_token_ids: list[int] = field(default_factory=list)
    finished: bool = False

    def is_finished(self):
        return self.finished


@pytest.mark.parametrize("eos", [True, False])
def test_prefill_decode_and_terminal_predictor_ownership(eos):
    transfer = Transfer()
    request = Request(output_token_ids=[20])
    prefill = states(3)
    assert ar_to_decoder_async_chunk(transfer, {"hidden": prefill}, request, new_token_ids=(20,)) is None
    request.output_token_ids.append(21)
    request.finished = not eos
    result = ar_to_decoder_async_chunk(transfer, {"hidden": states(1, 90)}, request, new_token_ids=(21,))
    if eos:
        assert result is None
        request.output_token_ids.append(3)
        request.finished = True
        result = ar_to_decoder_async_chunk(transfer, {"hidden": states(1, 999)}, request, new_token_ids=(3,))
    assert torch.equal(result.hidden_states.output, torch.stack([prefill[-1], states(1, 90)[0]]))
    assert bool(result.meta.stream_finished)
    assert result.additional_information["soprano_truncated"] is not eos
    assert ar_to_decoder_async_chunk(transfer, None, request, new_token_ids=()) is None


def test_invalid_hidden_handoff_finishes_explicitly_instead_of_hanging():
    transfer = Transfer()
    request = Request(output_token_ids=[20], finished=True)
    result = ar_to_decoder_async_chunk(transfer, {"hidden": states(1).to(torch.bfloat16)}, request, new_token_ids=(20,))
    assert result.additional_information["soprano_error"]
    assert result.hidden_states.output.shape == (0, 512)
    assert bool(result.meta.stream_finished)


@pytest.mark.parametrize("chunk_size", [1, 2, 4, 7])
def test_first_emit_uses_reference_cadence_even_with_small_transport_deltas(chunk_size):
    state = DecoderWindow(chunk_size)
    decoder = IntervalDecoder()
    for length in range(1, CONTEXT + chunk_size + 1):
        wave = state.push(decoder, states(1, offset=length), finished=False)
        assert wave.numel() == (chunk_size * 2048 if length == CONTEXT + chunk_size else 0)


def test_window_owns_context_after_borrowed_input_is_reused():
    state = DecoderWindow(4)
    source = states(4)
    state.push(IntervalDecoder(), source, finished=False)
    source.fill_(-999)
    result = state.push(IntervalDecoder(), states(4, offset=4 * 512), finished=True)
    assert torch.equal(result, states(8)[:-1, 0].repeat_interleave(2048))


def test_unhandled_decoder_error_still_reclaims_request_state():
    class BrokenDecoder(IntervalDecoder):
        def forward(self, hidden):
            raise RuntimeError("simulated decoder failure")

    model = wrapper(BrokenDecoder())
    model(runtime_additional_information=[payload(states(4), chunk_size=4)], request_ids=["failure"])
    with pytest.raises(RuntimeError, match="simulated decoder failure"):
        model(runtime_additional_information=[payload(states(4), chunk_size=4)], request_ids=["failure"])
    assert not model._streams
    assert model._failed_streams == {"failure"}

    # A device/runtime failure propagates, but late chunks still cannot restart
    # the failed synthesis unit, including a terminal chunk.
    late = model(
        runtime_additional_information=[payload(states(8), chunk_size=8, finished=True)], request_ids=["failure"]
    ).multimodal_outputs
    assert bool(late["audio_synthesis_error"][0])
    assert late["model_outputs"][0].numel() == 0
    assert model._failed_streams == {"failure"}

    model.on_requests_finished(["failure"])
    assert not model._streams and not model._failed_streams
    model.decoder = IntervalDecoder()
    recovered = model(
        runtime_additional_information=[payload(states(8), chunk_size=8, finished=True)], request_ids=["failure"]
    ).multimodal_outputs
    assert not bool(recovered["audio_synthesis_error"][0])
    assert torch.equal(recovered["model_outputs"][0], states(8)[:-1, 0].repeat_interleave(2048))


def test_pending_state_remains_bounded_after_a_handoff_error():
    transfer = Transfer()
    request = Request(output_token_ids=[20])
    assert ar_to_decoder_async_chunk(transfer, None, request, new_token_ids=(20,)) is None
    for _ in range(20):
        assert ar_to_decoder_async_chunk(transfer, {"hidden": states(1)}, request, new_token_ids=(20,)) is None
    assert transfer.request_payload["request"]["soprano"]["pending"] == []
    request.finished = True
    result = ar_to_decoder_async_chunk(transfer, {"hidden": states(1)}, request, new_token_ids=(20,))
    assert result.additional_information["soprano_error"]
    assert result.hidden_states.output.shape[0] == 0


@pytest.mark.parametrize("nested", [True, False])
def test_chunk_size_survives_real_connector_config_flattening(nested):
    transfer = Transfer()
    config = {"soprano_chunk_size": 1}
    transfer.connector.config = {"extra": config} if nested else config
    request = Request(output_token_ids=[20])
    result = ar_to_decoder_async_chunk(transfer, {"hidden": states(1)}, request, new_token_ids=(20,))
    assert result is not None
    assert result.additional_information["soprano_chunk_size"] == 1
    assert result.hidden_states.output.shape == (1, 512)


def test_graph_or_speculative_streaming_is_rejected_before_weight_loading(mocker):
    from transformers import Qwen3Config

    from vllm_omni.model_executor.models.soprano.soprano_ar import SopranoForCausalLM

    config = mocker.Mock()
    config.model_config.hf_config = Qwen3Config(hidden_size=512, vocab_size=8192, eos_token_id=3)
    config.model_config.async_chunk = True
    config.model_config.enforce_eager = False
    with pytest.raises(ValueError, match="eager execution"):
        SopranoForCausalLM(vllm_config=config)
    config.model_config.enforce_eager = True
    config.speculative_config = mocker.Mock()
    with pytest.raises(ValueError, match="speculative"):
        SopranoForCausalLM(vllm_config=config)


@pytest.mark.parametrize("valid_tokens", [0, 1, 2])
def test_immediate_or_early_eos_reaches_decoder_and_releases_state(valid_tokens):
    transfer = Transfer()
    request = Request()
    for step in range(valid_tokens):
        request.output_token_ids.append(20 + step)
        assert (
            ar_to_decoder_async_chunk(transfer, {"hidden": states(1, step * 512)}, request, new_token_ids=(20 + step,))
            is None
        )
    request.output_token_ids.append(3)
    request.finished = True
    result = ar_to_decoder_async_chunk(transfer, {"hidden": states(1, 9000)}, request, new_token_ids=(3,))
    assert result.hidden_states.output.shape == (valid_tokens, 512)
    assert not result.additional_information["soprano_truncated"]
    model = wrapper(IntervalDecoder())
    output = model(
        runtime_additional_information=[
            {
                "hidden_states": {"output": result.hidden_states.output},
                "meta": {"stream_finished": True},
                "additional_information": result.additional_information,
            }
        ],
        request_ids=["early-eos"],
    ).multimodal_outputs
    assert bool(output["audio_synthesis_error"][0]) == (valid_tokens < 2)
    assert output["model_outputs"][0].numel() == (2048 if valid_tokens == 2 else 0)
    assert not model._streams
