# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Contracts for the unified PersonaPlex plugin's delta-audio projection."""

import numpy as np
import pytest
from vllm.outputs import RequestOutput
from vllm.sampling_params import RequestOutputKind, SamplingParams

from vllm_omni.engine.duplex.plugin import DuplexDataPlaneContext
from vllm_omni.model_executor.common.duplex.data_plane import CumulativeAudioTextDataPlane
from vllm_omni.model_executor.models.personaplex.duplex.data_plane import PersonaPlexDataPlaneSession
from vllm_omni.model_executor.models.personaplex.duplex.plugin import PersonaPlexDuplexPlugin
from vllm_omni.outputs.mm_outputs import MultimodalCompletionOutput, MultimodalPayload

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


class _Sink:
    def __init__(self):
        self.chunks = []

    def encode(self, audio, rate, response_format, speed):
        assert rate == 24000
        assert response_format == "wav" and speed is None
        if audio is None:
            return None
        self.chunks.append(np.asarray(audio, dtype=np.float32).reshape(-1).copy())
        return f"chunk-{len(self.chunks)}"


def _output(request_id: str, audio: object, text: str = "") -> RequestOutput:
    completion = MultimodalCompletionOutput(
        index=0,
        text=text,
        token_ids=[],
        cumulative_logprob=None,
        logprobs=None,
        multimodal_output=MultimodalPayload.from_dict({"audio": audio, "sr": 24000}),
    )
    return RequestOutput(
        request_id=request_id,
        prompt=None,
        prompt_token_ids=[],
        prompt_logprobs=None,
        outputs=[completion],
        finished=False,
    )


def _project(plane, output):
    return list(plane.project({"data_plane_outputs": [output]}, context=DuplexDataPlaneContext()))


@pytest.mark.parametrize("kind", list(RequestOutputKind))
@pytest.mark.parametrize("skip_clone", [False, True])
def test_stage1_delta_selection_isolated_from_sampling_defaults(kind, skip_clone):
    plugin = PersonaPlexDuplexPlugin(_Sink().encode)
    defaults = (SamplingParams(), SamplingParams(output_kind=kind, skip_clone=skip_clone), object())
    configured = plugin.configure_sampling_params(runtime_config={}, defaults=defaults)
    assert configured[1].output_kind == RequestOutputKind.DELTA
    assert configured[1] is not defaults[1]
    assert defaults[1].output_kind == kind
    assert configured[2] is defaults[2]


@pytest.mark.parametrize("defaults", [(), (object(),), (object(), object())])
def test_non_sampling_defaults_are_preserved(defaults):
    assert (
        PersonaPlexDuplexPlugin(_Sink().encode).configure_sampling_params(runtime_config={}, defaults=defaults)
        == defaults
    )


@pytest.mark.parametrize("sizes", [(1920, 1920, 1920), (960, 2880, 1920), (0, 1920, 0, 1920, 960)])
def test_equal_variable_and_empty_deltas_reconstruct_exact_pcm(sizes):
    sink = _Sink()
    plane = PersonaPlexDataPlaneSession(sink.encode)
    expected = []
    for index, size in enumerate(sizes):
        chunk = np.full(size, index / 10, dtype=np.float32)
        expected.append(chunk)
        events = _project(plane, _output("req", chunk))
        assert len(events) == int(size > 0)
        if events:
            assert events[0]["audio_duration_ms"] == round(size / 24)
    np.testing.assert_array_equal(np.concatenate(sink.chunks), np.concatenate(expected))


def test_identical_chunks_are_distinct_emissions_with_cumulative_text():
    sink = _Sink()
    plane = PersonaPlexDataPlaneSession(sink.encode)
    chunk = np.full(1920, 0.125, dtype=np.float32)
    transcripts = []
    for text in ("he", "hello", "hello", "hello!"):
        [event] = _project(plane, _output("req", chunk, text))
        transcripts.append(event["text"])
    assert transcripts == ["he", "llo", "", "!"]
    assert len(sink.chunks) == 4
    for actual in sink.chunks:
        np.testing.assert_array_equal(actual, chunk)


def test_empty_pcm_does_not_encode_header_only_wav_and_retains_text():
    calls = []

    def encode(*args):
        calls.append(args)
        return "wav-header"

    plane = PersonaPlexDataPlaneSession(encode)
    assert _project(plane, _output("req", np.empty(0))) == []
    [event] = _project(plane, _output("req", np.empty(0), "hello"))
    assert event["text"] == "hello" and event["audio_data"] == ""
    assert event["audio_duration_ms"] == 0
    assert calls == []


def test_deferred_audio_chunks_are_coalesced_within_an_emission_only():
    sink = _Sink()
    plane = PersonaPlexDataPlaneSession(sink.encode)
    chunks = [np.full(960, 0.25, dtype=np.float32), np.ones(1920, dtype=np.float32)]
    for _ in range(2):
        [event] = _project(plane, _output("req", chunks))
        assert event["audio_duration_ms"] == 120
        np.testing.assert_array_equal(sink.chunks[-1], np.concatenate(chunks))
    assert len(sink.chunks) == 2


@pytest.mark.parametrize("failure", ["empty", "exception"])
def test_failed_encoding_preserves_cursors_and_other_requests(failure):
    sink = _Sink()
    fail_next = True

    def encode(*args):
        nonlocal fail_next
        if fail_next:
            fail_next = False
            if failure == "exception":
                raise ValueError("encoding failed")
            return None
        return sink.encode(*args)

    plane = PersonaPlexDataPlaneSession(encode)
    output = _output("failed", np.ones(1920), "first transcript")
    with pytest.raises((RuntimeError, ValueError)):
        _project(plane, output)
    assert plane._requests["failed"].audio_samples == 0
    assert plane._requests["failed"].text == ""
    assert len(_project(plane, _output("other", np.zeros(1920)))) == 1
    [retried] = _project(plane, output)
    assert retried["text"] == "first transcript"
    assert plane._requests["failed"].audio_samples == 1920


@pytest.mark.parametrize("sessions", [1, 2])
def test_long_stream_retains_counts_not_audio_history_and_cleans_up(sessions):
    sink = _Sink()
    plane = PersonaPlexDataPlaneSession(sink.encode)
    for frame in range(100):
        for session in range(sessions):
            request_id = f"req-{session}"
            chunk = np.full(1920, (frame % 17 + session) / 32, dtype=np.float32)
            [event] = _project(plane, _output(request_id, chunk))
            assert event["data_plane_request_id"] == request_id
            assert event["audio_duration_ms"] == 80
            np.testing.assert_array_equal(sink.chunks[-1], chunk)
    for session in range(sessions):
        request_id = f"req-{session}"
        assert plane._requests[request_id].audio_samples == 100 * 1920
        plane.mark_terminal(request_id)
        assert plane.is_terminal(request_id)
        plane.close_stream(request_id)
        assert request_id not in plane._requests


def test_generic_cumulative_projector_is_unchanged():
    sink = _Sink()
    plane = CumulativeAudioTextDataPlane(sink.encode)
    for samples in (1920, 3840, 5760):
        [event] = _project(plane, _output("req", np.arange(samples, dtype=np.float32)))
        assert event["audio_duration_ms"] == 80
    assert [chunk.size for chunk in sink.chunks] == [1920, 1920, 1920]
