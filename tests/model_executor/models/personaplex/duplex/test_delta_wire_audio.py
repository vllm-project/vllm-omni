# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Exercise wire serialization and the production PCM/WAV encoder, without weights."""

import io
import wave

import pybase64 as base64
import pytest
import torch
from vllm.outputs import RequestOutput
from vllm.sampling_params import RequestOutputKind
from vllm.v1.engine import FinishReason
from vllm.v1.serial_utils import MsgpackDecoder, MsgpackEncoder

from tests.model_executor.models.personaplex.duplex.test_delta_output import _state
from vllm_omni.engine import OmniEngineCoreOutput, OmniEngineCoreOutputs
from vllm_omni.entrypoints.duplex.runtime_bridge import NativeRuntimeBridgeMixin
from vllm_omni.entrypoints.openai.audio_utils_mixin import AudioMixin
from vllm_omni.model_executor.models.personaplex.duplex.data_plane import (
    PersonaPlexDataPlaneContext,
    PersonaPlexDataPlaneSession,
)
from vllm_omni.outputs.output_modality import OutputModality
from vllm_omni.outputs.output_processor import MultimodalOutputProcessor

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


class _EncodingBridge(NativeRuntimeBridgeMixin):
    def __init__(self) -> None:
        # Only bind the real audio service needed by this production method;
        # no model, HTTP handler or mocked encoder is constructed.
        self._chat_service = AudioMixin()


def _pcm_bytes(event: dict[str, object], response_format: str) -> bytes:
    encoded = event["audio_data"]
    assert isinstance(encoded, str)
    raw = base64.b64decode(encoded, validate=True)
    if response_format == "wav":
        with wave.open(io.BytesIO(raw), "rb") as stream:
            assert stream.getnchannels() == 1
            assert stream.getsampwidth() == 2
            assert stream.getframerate() == 24000
            return stream.readframes(stream.getnframes())
    return raw


@pytest.mark.parametrize("sessions", [1, 2])
@pytest.mark.parametrize("response_format", ["pcm", "wav"])
@pytest.mark.parametrize("finish", ["stop", "abort"])
@pytest.mark.parametrize("layout", ["flat", "strided_row"])
def test_wire_to_encoded_audio_preserves_interleaved_streams(sessions, response_format, finish, layout):
    processor = MultimodalOutputProcessor(tokenizer=None, log_stats=False, output_modality=OutputModality.AUDIO)
    states = [_state(RequestOutputKind.DELTA, f"session-{i}") for i in range(sessions)]
    for state in states:
        processor.request_states[state.request_id] = state
        processor.external_req_ids[state.external_req_id] = [state.request_id]
    bridge = _EncodingBridge()
    projector = PersonaPlexDataPlaneSession(bridge._encode_native_data_plane_audio)
    context = PersonaPlexDataPlaneContext(response_format=response_format)
    generator = torch.Generator().manual_seed(127)
    expected: dict[str, list[bytes]] = {state.request_id: [] for state in states}
    actual: dict[str, list[bytes]] = {state.request_id: [] for state in states}
    encoder = MsgpackEncoder()
    decoder = MsgpackDecoder(OmniEngineCoreOutputs)

    for size in (1920, 0, 1920, 3840):
        batch = []
        for state in states:
            # Exact PCM16 grid values give an independent byte reference rather
            # than comparing two invocations of the same production encoder.
            pcm16 = torch.randint(-20000, 20001, (size,), generator=generator, dtype=torch.int16)
            expected[state.request_id].append(pcm16.numpy().astype("<i2").tobytes())
            audio = pcm16.float() / 32768
            if layout == "strided_row":
                audio = torch.stack((audio, audio), dim=-1).reshape(1, -1)[:, ::2]
            batch.append(
                OmniEngineCoreOutput(
                    request_id=state.request_id,
                    new_token_ids=[],
                    finish_reason=FinishReason.STOP,
                    is_segment_finished=True,
                    multimodal_output={"model_outputs": audio, "sr": torch.tensor(24000)},
                )
            )
        decoded = decoder.decode(encoder.encode(OmniEngineCoreOutputs(outputs=batch)))
        processed = processor.process_outputs(decoded.outputs)
        events = list(projector.project({"data_plane_outputs": processed.request_outputs}, context=context))
        # A header-only WAV must not masquerade as a nonempty audio delta.
        assert len(events) == (sessions if size else 0)
        for event in events:
            request_id = event["data_plane_request_id"]
            assert isinstance(request_id, str) and request_id in actual
            assert event["sample_rate_hz"] == 24000
            assert event["audio_duration_ms"] == round(size / 24)
            actual[request_id].append(_pcm_bytes(event, response_format))
        assert all(state.request_id in processor.request_states for state in states)

    for state in states:
        if finish == "abort":
            aborted, outputs = processor.abort_requests_collecting_outputs([state.request_id], internal=True)
            assert aborted == [state.request_id]
        else:
            terminal = OmniEngineCoreOutput(
                request_id=state.request_id,
                new_token_ids=[],
                finish_reason=FinishReason.STOP,
            )
            decoded = decoder.decode(encoder.encode(OmniEngineCoreOutputs(outputs=[terminal])))
            outputs = processor.process_outputs(decoded.outputs).request_outputs
        assert state.request_id not in processor.request_states
        terminal_events = list(projector.project({"data_plane_outputs": outputs}, context=context))
        assert terminal_events == []
        assert b"".join(actual[state.request_id]) == b"".join(expected[state.request_id])
        projector.close_stream(state.request_id)


@pytest.mark.parametrize("response_format", ["pcm", "wav"])
@pytest.mark.parametrize("text", ["", "first transcript"])
def test_empty_audio_does_not_invoke_encoder_or_lose_text(response_format, text, mocker):
    bridge = _EncodingBridge()
    encode_spy = mocker.spy(bridge._chat_service, "create_audio")
    projector = PersonaPlexDataPlaneSession(bridge._encode_native_data_plane_audio)
    context = PersonaPlexDataPlaneContext(response_format=response_format)
    state = _state(RequestOutputKind.DELTA)
    state.add_multimodal_tensor({"model_outputs": torch.empty(0), "sr": torch.tensor(24000)}, "audio")
    output = state.make_request_output([], None, None, None)
    assert isinstance(output, RequestOutput)
    output.outputs[0].text = text

    events = list(projector.project({"data_plane_outputs": [output]}, context=context))
    encode_spy.assert_not_called()
    assert len(events) == int(bool(text))
    if text:
        assert events[0]["text"] == text
        assert events[0]["audio_data"] == ""
        assert events[0]["audio_duration_ms"] == 0
    assert list(projector.project({"data_plane_outputs": [output]}, context=context)) == []
    encode_spy.assert_not_called()


def test_production_encoding_failure_is_not_a_successful_projection(monkeypatch):
    bridge = _EncodingBridge()
    projector = PersonaPlexDataPlaneSession(bridge._encode_native_data_plane_audio)
    output = _state(RequestOutputKind.DELTA)
    output.add_multimodal_tensor({"model_outputs": torch.ones(1920), "sr": torch.tensor(24000)}, "audio")
    emitted = output.make_request_output([], None, None, None)
    assert emitted is not None

    def fail_encoding(_audio_obj):
        raise ValueError("test codec failure")

    monkeypatch.setattr(bridge._chat_service, "create_audio", fail_encoding)
    # The production bridge catches codec exceptions and returns None; the
    # model projector must still reject nonempty audio rather than lose it.
    with pytest.raises(RuntimeError, match="could not encode a nonempty audio delta"):
        list(projector.project({"data_plane_outputs": [emitted]}))
