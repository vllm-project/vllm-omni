# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Exercise the real output accumulator and PersonaPlex consumer together."""

import numpy as np
import pytest
import torch
from vllm.outputs import PoolingRequestOutput, RequestOutput
from vllm.sampling_params import RequestOutputKind, SamplingParams
from vllm.v1.engine import FinishReason

from vllm_omni.engine import OmniEngineCoreOutput
from vllm_omni.model_executor.models.personaplex.duplex.data_plane import PersonaPlexDataPlaneSession
from vllm_omni.model_executor.models.personaplex.duplex.runtime_extension import PersonaPlexDuplexRuntimeExtension
from vllm_omni.outputs.output_modality import OutputModality
from vllm_omni.outputs.output_processor import MultimodalOutputProcessor, OmniRequestState

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _state(kind: RequestOutputKind, request_id: str = "session") -> OmniRequestState:
    return OmniRequestState(
        request_id=request_id,
        external_req_id=request_id,
        parent_req=None,
        request_index=0,
        lora_request=None,
        prompt=None,
        prompt_token_ids=[0],
        prompt_embeds=None,
        logprobs_processor=None,
        detokenizer=None,
        max_tokens_param=None,
        arrival_time=0.0,
        queue=None,
        log_stats=False,
        stream_interval=1,
        output_kind=kind,
    )


class _AudioSink:
    def __init__(self) -> None:
        self.chunks: list[np.ndarray] = []

    def encode(self, audio: object, sample_rate: int, response_format: str, speed: float | None) -> str | None:
        assert sample_rate == 24000
        if audio is None:
            return None
        samples = np.asarray(audio, dtype=np.float32).reshape(-1)
        if not samples.size:
            return None
        self.chunks.append(samples.copy())
        return f"chunk-{len(self.chunks)}"


def _emit(state: OmniRequestState, audio: torch.Tensor, *, final: bool = False) -> RequestOutput:
    state.add_multimodal_tensor({"model_outputs": audio, "sr": torch.tensor(24000)}, "audio")
    output = state.make_request_output([], None, FinishReason.STOP if final else None, None)
    assert output is not None and not isinstance(output, PoolingRequestOutput)
    return output


@pytest.mark.parametrize("kind", list(RequestOutputKind))
@pytest.mark.parametrize("skip_clone", [False, True])
def test_stage1_delta_configuration_does_not_mutate_defaults(kind, skip_clone):
    extension = PersonaPlexDuplexRuntimeExtension()
    defaults = (SamplingParams(temperature=0.7), SamplingParams(output_kind=kind, skip_clone=skip_clone))
    extra_stage = object()
    configured = extension.configure_sampling_params(runtime_config={}, defaults=(*defaults, extra_stage))
    assert configured[1].output_kind == RequestOutputKind.DELTA
    assert configured[1] is not defaults[1]
    assert defaults[1].output_kind == kind
    assert defaults[0].temperature == 0.7
    assert configured[0].temperature == 0.0
    assert configured[0].max_tokens == 1
    assert configured[0].output_kind == defaults[0].output_kind
    assert configured[2] is extra_stage


@pytest.mark.parametrize("defaults", [(), (object(),), (object(), object())])
def test_incomplete_or_non_sampling_defaults_are_preserved(defaults):
    assert (
        PersonaPlexDuplexRuntimeExtension().configure_sampling_params(runtime_config={}, defaults=defaults) == defaults
    )


@pytest.mark.parametrize("sizes", [(1920, 1920, 1920), (960, 2880, 1920), (0, 1920, 0, 1920, 960)])
def test_equal_variable_and_empty_audio_deltas_reconstruct_exact_pcm(sizes):
    # Fixed DELTA producer isolates consumer correctness from configuration.
    state = _state(RequestOutputKind.DELTA)
    sink = _AudioSink()
    projector = PersonaPlexDataPlaneSession(sink.encode)
    expected = []
    for index, size in enumerate(sizes):
        chunk = torch.full((size,), index / 10, dtype=torch.float32)
        expected.append(chunk.numpy())
        output = _emit(state, chunk, final=index == len(sizes) - 1)
        events = list(projector.project({"data_plane_outputs": [output]}))
        assert len(events) == int(size > 0)
        if events:
            assert events[0]["audio_duration_ms"] == round(size / 24)
            assert events[0]["end_of_turn"] is False
    np.testing.assert_array_equal(np.concatenate(sink.chunks), np.concatenate(expected))


def test_identical_consecutive_chunks_are_not_deduplicated():
    state = _state(RequestOutputKind.DELTA)
    sink = _AudioSink()
    projector = PersonaPlexDataPlaneSession(sink.encode)
    chunk = torch.full((1920,), 0.125)
    for _ in range(4):
        assert len(list(projector.project({"data_plane_outputs": [_emit(state, chunk)]}))) == 1
    assert len(sink.chunks) == 4


def test_multiple_accumulations_form_one_delta_without_history():
    state = _state(RequestOutputKind.DELTA)
    sink = _AudioSink()
    projector = PersonaPlexDataPlaneSession(sink.encode)
    for _ in range(2):
        state.add_multimodal_tensor({"model_outputs": torch.ones(960), "sr": 24000}, "audio")
        output = _emit(state, torch.full((1920,), 0.25))
        events = list(projector.project({"data_plane_outputs": [output]}))
        assert len(events) == 1
        np.testing.assert_array_equal(sink.chunks[-1], np.r_[np.ones(960), np.full(1920, 0.25)])


@pytest.mark.parametrize("sessions", [1, 2])
def test_long_stream_has_constant_per_chunk_payload_and_independent_sessions(sessions):
    params = PersonaPlexDuplexRuntimeExtension().configure_sampling_params(
        runtime_config={}, defaults=(SamplingParams(), SamplingParams())
    )
    states = [_state(params[1].output_kind, f"request-{i}") for i in range(sessions)]
    sink = _AudioSink()
    projector = PersonaPlexDataPlaneSession(sink.encode)
    total = 0
    for index in range(1000):
        for session, state in enumerate(states):
            chunk = torch.full((1920,), (index % 17 + session) / 32)
            output = _emit(state, chunk)
            audio = output.outputs[0].multimodal_output["audio"]
            assert isinstance(audio, torch.Tensor)
            assert audio.numel() == 1920
            rate = output.outputs[0].multimodal_output["sr"]
            assert isinstance(rate, torch.Tensor) and rate.numel() == 1
            total += audio.numel() * audio.element_size()
            assert "audio" not in state.mm_accumulated
            events = list(projector.project({"data_plane_outputs": [output]}))
            assert len(events) == 1
            assert events[0]["data_plane_request_id"] == state.request_id
            np.testing.assert_array_equal(sink.chunks[-1], chunk.numpy())
    assert total == sessions * 1000 * 1920 * 4
    for state in states:
        output = _emit(state, torch.empty(0), final=True)
        assert output.finished is True
        assert list(projector.project({"data_plane_outputs": [output]})) == []
        projector.mark_terminal(state.request_id)
        assert projector.is_terminal(state.request_id)
        projector.close_stream(state.request_id)
        assert not projector.is_terminal(state.request_id)


def test_actual_multimodal_output_processor_retains_stream_then_finishes():
    state = _state(RequestOutputKind.DELTA)
    processor = MultimodalOutputProcessor(tokenizer=None, log_stats=False, output_modality=OutputModality.AUDIO)
    processor.request_states[state.request_id] = state
    processor.external_req_ids[state.external_req_id] = [state.request_id]
    sink = _AudioSink()
    projector = PersonaPlexDataPlaneSession(sink.encode)
    for index in range(3):
        output = OmniEngineCoreOutput(
            request_id=state.request_id,
            new_token_ids=[],
            finish_reason=FinishReason.STOP,
            is_segment_finished=index < 2,
            multimodal_output={"model_outputs": torch.full((1920,), index / 10), "sr": torch.tensor(24000)},
        )
        processed = processor.process_outputs([output])
        assert len(processed.request_outputs) == 1
        assert len(list(projector.project({"data_plane_outputs": processed.request_outputs}))) == 1
        assert (state.request_id in processor.request_states) is (index < 2)
    assert len(sink.chunks) == 3


def test_cumulative_output_for_other_callers_is_unchanged():
    state = _state(RequestOutputKind.CUMULATIVE)
    for size in range(1, 5):
        output = _emit(state, torch.ones(1920))
        assert output.outputs[0].multimodal_output["audio"].numel() == size * 1920


@pytest.mark.parametrize("pending_samples", [0, 960])
def test_abort_emits_only_pending_audio_and_does_not_replay_delivered_chunks(pending_samples):
    state = _state(RequestOutputKind.DELTA)
    processor = MultimodalOutputProcessor(tokenizer=None, log_stats=False, output_modality=OutputModality.AUDIO)
    processor.request_states[state.request_id] = state
    processor.external_req_ids[state.external_req_id] = [state.request_id]
    sink = _AudioSink()
    projector = PersonaPlexDataPlaneSession(sink.encode)
    list(projector.project({"data_plane_outputs": [_emit(state, torch.full((1920,), 0.25))]}))
    assert len(sink.chunks) == 1
    if pending_samples:
        state.add_multimodal_tensor({"model_outputs": torch.ones(pending_samples)}, "audio")
    aborted, outputs = processor.abort_requests_collecting_outputs([state.request_id], internal=True)
    assert aborted == [state.request_id]
    assert len(outputs) == 1 and outputs[0].finished
    events = list(projector.project({"data_plane_outputs": outputs}))
    assert len(events) == int(pending_samples > 0)
    assert sum(chunk.size for chunk in sink.chunks) == 1920 + pending_samples
    assert state.output_kind == RequestOutputKind.DELTA
    assert state.request_id not in processor.request_states


@pytest.mark.parametrize("failure", ["empty", "exception"])
def test_encode_failure_does_not_consume_text_or_corrupt_another_request(failure):
    class RecoveringSink(_AudioSink):
        fail_next = True

        def encode(self, audio, sample_rate, response_format, speed):
            if self.fail_next:
                self.fail_next = False
                if failure == "exception":
                    raise ValueError("encoding failed")
                return None
            return super().encode(audio, sample_rate, response_format, speed)

    state = _state(RequestOutputKind.DELTA, "failed")
    output = _emit(state, torch.ones(1920))
    output.outputs[0].text = "first transcript"
    sink = RecoveringSink()
    projector = PersonaPlexDataPlaneSession(sink.encode)
    with pytest.raises((RuntimeError, ValueError)):
        list(projector.project({"data_plane_outputs": [output]}))
    other = _emit(_state(RequestOutputKind.DELTA, "other"), torch.full((1920,), 0.25))
    assert len(list(projector.project({"data_plane_outputs": [other]}))) == 1
    retried = list(projector.project({"data_plane_outputs": [output]}))
    assert len(retried) == 1 and retried[0]["text"] == "first transcript"
    assert len(sink.chunks) == 2
    np.testing.assert_array_equal(sink.chunks[-1], np.ones(1920))
