# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import asyncio
import base64
import io
import time
import wave
from types import SimpleNamespace

import pytest
from vllm.benchmarks.lib.endpoint_request_func import RequestFuncInput

from vllm_omni.benchmarks import realtime_seed_tts as realtime

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

_REAL_SLEEP = asyncio.sleep

#: The fixture clip: 400 ms of content plus the 1000 ms silent tail the dataset
#: appends. Both are exact multiples of the 200 ms upload chunk.
_CONTENT_MS = 400
_CONTENT_BYTES = 24_000 * _CONTENT_MS // 1000 * 2
_TAIL_BYTES = 24_000 * realtime.SEED_TTS_SILENT_TAIL_MS // 1000 * 2


def _wav() -> str:
    result = io.BytesIO()
    with wave.open(result, "wb") as audio:
        audio.setparams((1, 2, 24_000, 0, "NONE", "not compressed"))
        audio.writeframes(b"\x01\x00" * (_CONTENT_BYTES // 2) + bytes(_TAIL_BYTES))
    return base64.b64encode(result.getvalue()).decode()


class _FakeClock:
    """Virtual monotonic clock advanced by the patched ``asyncio.sleep``.

    The paced upload models 1.4 s; sleeping that for real would make the suite
    slow, but dropping the sleeps outright would desynchronize the pacing
    schedule from event timestamps and turn every latency negative.
    """

    def __init__(self) -> None:
        self.now = 1_000.0

    def monotonic(self) -> float:
        return self.now

    async def sleep(self, seconds: float = 0.0, *args: object) -> None:
        # Yield before advancing: events already queued arrive at the start of
        # the sleep window, not its end. Advancing first would back-date every
        # arrival to the far edge and hide latencies shorter than one chunk.
        await _REAL_SLEEP(0)
        self.now += max(0.0, float(seconds))


@pytest.fixture
def client(monkeypatch):
    class Client:
        instances = []
        speech_events = True
        status = "completed"
        server_error = False
        early = False
        split_on_last_content = False

        def __init__(self, url, *, config, **kwargs):
            assert url.startswith("ws://") and "duplex=1" in url
            self.config = config
            self.session_id = "session-1"
            self.sent = []
            self.pcm = bytearray()
            self.queue: asyncio.Queue = asyncio.Queue()
            self.closed = False
            self.answered = False
            self.instances.append(self)

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            pass

        async def events(self):
            while True:
                yield SimpleNamespace(raw=await self.queue.get())

        def answer(self):
            self.answered = True
            if self.server_error:
                self.queue.put_nowait({"type": "error", "error": {"message": "engine failed"}})
                return
            # Measured against a live server: speech_started / speech_stopped
            # arrive even with turn_detection off, where they merely follow the
            # client's own commit. Emitting them unconditionally is what lets
            # this mock catch a client that reports them as VAD timings.
            if self.speech_events:
                self.queue.put_nowait({"type": "input_audio_buffer.speech_started"})
                self.queue.put_nowait({"type": "input_audio_buffer.speech_stopped"})
            for event in [
                {"type": "response.created", "response": {"id": "r1"}},
                {"type": "response.output_text.delta", "response_id": "r1", "delta": "target text"},
                {
                    "type": "response.output_audio.delta",
                    "response_id": "r1",
                    "delta": base64.b64encode(bytes(48000)).decode(),
                },
                {
                    "type": "response.done",
                    "response": {"id": "r1", "status": self.status},
                    "response_request_metrics": {"ttft_ms": 999999, "ttfp_ms": 999999},
                },
            ]:
                self.queue.put_nowait(event)

        async def send(self, event):
            self.sent.append(event)
            if event["type"] == "response.create":
                self.answer()

        async def append_audio(self, pcm):
            self.pcm.extend(pcm)
            # Server VAD only settles once the silent tail has arrived.
            if self.config.turn_detection and not self.answered:
                if self.early:
                    self.answer()
                elif self.split_on_last_content and len(self.pcm) >= _CONTENT_BYTES:
                    # Splits exactly on the final content chunk. The in-loop
                    # guard checks at the top of the next iteration, which is
                    # already a tail chunk, so only the negative-TTFP check
                    # can catch this.
                    self.answer()
                elif len(self.pcm) >= _CONTENT_BYTES + _TAIL_BYTES:
                    self.answer()

        async def commit(self, **kwargs):
            self.sent.append({"type": "input_audio_buffer.commit", **kwargs})

        async def close(self, **kwargs):
            self.closed = True

    clock = _FakeClock()

    async def ack(*args):
        pass

    monkeypatch.setattr(time, "monotonic", clock.monotonic)
    monkeypatch.setattr(realtime, "DuplexClient", Client)
    monkeypatch.setattr(realtime.asyncio, "sleep", clock.sleep)
    monkeypatch.setattr(realtime, "acknowledge_collected_playback", ack)
    return Client


def _request(trigger):
    """Build the real RequestFuncInput the backend receives, extras attached
    the same way ``_attach_seed_tts_to_request_func_input`` attaches them."""
    request = RequestFuncInput(
        prompt="target text",
        api_url="http://localhost/v1/realtime",
        prompt_len=3,
        output_len=256,
        model="qwen",
        model_name="qwen",
        extra_body={"realtime_trigger": trigger},
    )
    request.seed_tts_input_audio = _wav()
    request.seed_tts_system_prompt = "Read exactly."
    request.seed_tts_utterance_id = "utt0"
    return request


def test_vad_silence_threshold_fits_inside_the_dataset_tail():
    # The module asserts this at import; pin it so lowering the dataset tail
    # cannot silently leave VAD unable to ever detect the endpoint.
    assert realtime._VAD_SILENCE_MS < realtime.SEED_TTS_SILENT_TAIL_MS


@pytest.mark.asyncio
@pytest.mark.parametrize("trigger", ["explicit", "vad"])
async def test_matched_audio_and_trigger_contract(client, trigger):
    request = _request(trigger)
    output = await realtime.run_realtime_seed_tts(request)
    instance = client.instances[-1]
    full_pcm = realtime._input_pcm(request.seed_tts_input_audio)
    # The silent tail exists only so server VAD can find the endpoint; an
    # explicit client commits at the end of speech and must not be charged
    # for streaming a second of silence first.
    assert bytes(instance.pcm) == (full_pcm if trigger == "vad" else full_pcm[:_CONTENT_BYTES])
    assert instance.sent[0]["item"]["content"] == [{"type": "input_text", "text": "target text"}]
    triggers = [event["type"] for event in instance.sent[1:]]
    assert triggers == (["input_audio_buffer.commit", "response.create"] if trigger == "explicit" else [])
    payload = instance.config.to_session_payload(model="qwen")
    from vllm_omni.engine.duplex.turn_detection import validate_realtime_turn_detection
    from vllm_omni.model_executor.models.qwen3_omni.duplex.plugin import Qwen3OmniDuplexPlugin

    assert validate_realtime_turn_detection(payload) is None
    Qwen3OmniDuplexPlugin(lambda *args: None).validate_client_extra_body(payload["extra_body"])
    assert payload["extra_body"]["auto_response"] is False
    assert payload["temperature"] == 0
    assert payload["max_output_tokens"] == 256
    if trigger == "vad":
        assert payload["turn_detection"]["interrupt_response"] is True
    assert output["generated_text"] == "target text"
    assert output["audio_duration"] == 1.0
    # Client timings must not silently switch to server-only timings.
    assert 0 <= output["audio_ttfp"] < 100
    assert output["duplex_session_metrics"]["audio_turn_count"] == 1
    metrics = output["duplex_request_metrics"][0]
    assert metrics["utterance_id"] == "utt0"
    assert metrics["input_content_ms"] == _CONTENT_MS
    assert ("vad_stop_to_first_audio_ms" in metrics) == (trigger == "vad")
    assert instance.closed


@pytest.mark.asyncio
@pytest.mark.parametrize("trigger", ["explicit", "vad"])
async def test_latency_origin_is_end_of_reference_speech(client, trigger):
    """TTFT/TTFP/RTF must exclude the caller's own real-time audio upload."""
    output = await realtime.run_realtime_seed_tts(_request(trigger))
    metrics = output["duplex_request_metrics"][0]
    assert metrics["measurement_origin"]["ttfp"].startswith("end of reference speech")
    # Every bound below is stated in absolute milliseconds against the fixture's
    # known pacing, not as a relation between two fields derived from the same
    # number: timing from session start would fold the _CONTENT_MS upload into
    # TTFP and push each one over its limit.
    if trigger == "explicit":
        # Commit lands at the end of speech, so the wait is the model's alone
        # and must be far shorter than the upload that preceded it.
        assert metrics["ttfp_ms"] < _CONTENT_MS
        assert metrics["audio_generation_ms"] < metrics["input_upload_ms"]
    else:
        # VAD must still spend its silence threshold before it can fire — that
        # wait is real user-visible latency, not a measurement artifact — but
        # it cannot exceed the tail it is listening to plus one upload chunk.
        assert realtime._VAD_SILENCE_MS <= metrics["ttfp_ms"] < realtime.SEED_TTS_SILENT_TAIL_MS + realtime._CHUNK_MS
        assert metrics["vad_stop_received_ms"] >= realtime._VAD_SILENCE_MS
        assert metrics["vad_stop_to_first_audio_ms"] < _CONTENT_MS
    # E2EL must share TTFT's origin: upstream derives TPOT from their
    # difference whenever the backend reports no engine token count, so an
    # E2EL still anchored at session start would inflate TPOT by the upload.
    assert output["latency"] * 1000 < metrics["session_start_to_response_done_ms"]
    assert output["latency"] >= output["ttft"]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "fault,match",
    [
        ("missing_vad", "speech-start"),
        ("failed", "did not complete"),
        ("error", "server error"),
        ("early", "split the reference"),
        ("split_on_last_content", "before the reference speech ended"),
    ],
)
async def test_invalid_comparison_releases_session(client, fault, match):
    client.speech_events = fault != "missing_vad"
    client.status = "failed" if fault == "failed" else "completed"
    client.server_error = fault == "error"
    client.early = fault == "early"
    client.split_on_last_content = fault == "split_on_last_content"
    with pytest.raises(RuntimeError, match=match):
        await realtime.run_realtime_seed_tts(_request("vad"))
    assert client.instances[-1].closed


def test_reference_audio_replaces_voice_clone_fields_and_stays_off_chat_content(tmp_path, monkeypatch):
    from vllm_omni.benchmarks.data_modules.seed_tts_dataset import SeedTTSDataset
    from vllm_omni.benchmarks.patch.patch import _attach_seed_tts_to_request_func_input

    (tmp_path / "en").mkdir()
    (tmp_path / "en" / "prompt.wav").write_bytes(base64.b64decode(_wav()))
    (tmp_path / "en" / "meta.lst").write_text("utt0|reference|prompt.wav|target text\n")
    dataset = SeedTTSDataset(dataset_path=str(tmp_path), disable_shuffle=True)
    from unittest.mock import MagicMock

    tokenizer = MagicMock()
    tokenizer.encode.return_value = [1, 2]
    monkeypatch.setattr("vllm_omni.benchmarks.data_modules.seed_tts_dataset.get_cached_tokenizer", lambda value: value)
    requests = dataset.sample(tokenizer=tokenizer, num_requests=1, reference_as_input=True)
    request = SimpleNamespace(extra_body={})
    _attach_seed_tts_to_request_func_input(requests[0], request)
    assert request.seed_tts_utterance_id == "utt0"
    # ref_audio / ref_text must not also be sent: the speech is the input now.
    assert request.extra_body == {}
    # The reference speech travels as PCM for the socket, never as chat content.
    assert request.omni_chat_messages[1]["content"] == [{"type": "text", "text": "target text"}]
    assert realtime._input_pcm(request.seed_tts_input_audio).endswith(bytes(_TAIL_BYTES))
    with pytest.raises(ValueError, match="one utterance"):
        dataset.sample(tokenizer=tokenizer, num_requests=1, reference_as_input=True, turns_per_session=4)


@pytest.mark.parametrize(
    "backend,dataset_name,match",
    [
        ("openai-chat-omni", "seed-tts", "openai-realtime-chat"),
        ("openai-realtime-chat", "seed-tts-text", "openai-realtime-chat"),
    ],
)
def test_reference_as_input_is_realtime_only(backend, dataset_name, match):
    from vllm_omni.benchmarks.patch.patch import get_samples

    args = SimpleNamespace(
        dataset_name=dataset_name,
        backend=backend,
        dataset_path="some/repo",
        hf_name=None,
        seed_tts_reference_as_input=True,
    )
    with pytest.raises(ValueError, match=match):
        get_samples(args, tokenizer=None)
