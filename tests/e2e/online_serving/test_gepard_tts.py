# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""E2E online serving tests for Gepard-1.0 via ``/v1/audio/speech``.

Zero-shot default voice. Needs a GPU and the NeMo NanoCodec; the weekly
``TTS · Gepard-1.0 · L4`` step installs NeMo then runs this file.
"""

from __future__ import annotations

import array
import os
import statistics
import struct
import time
import wave
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from io import BytesIO

os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

import pytest
import requests
import soundfile as sf

from tests.helpers.mark import hardware_test
from tests.helpers.media import convert_audio_bytes_to_text
from tests.helpers.runtime import OmniServerParams
from tests.helpers.stage_config import get_deploy_config_path

pytest.importorskip("nemo.collections.tts.models")

pytestmark = [pytest.mark.slow, pytest.mark.tts]

MODEL = "nineninesix/gepard-1.0"
SAMPLE_RATE = 22050
SAMPLES_PER_FRAME = 1024
PCM16_BYTES_PER_FRAME = SAMPLES_PER_FRAME * 2
DEFAULT_TIMEOUT_S = 180.0

_DISTINGUISHABLE_PROMPTS = {
    "He drinks coffee every morning.": "coffee",
    "Machine learning is interesting.": "learning",
    "Please close the window before leaving.": "window",
    "My favorite color is purple.": "purple",
}

tts_server_params = [
    pytest.param(
        OmniServerParams(
            model=MODEL,
            stage_config_path=get_deploy_config_path("gepard.yaml"),
            server_args=["--trust-remote-code", "--disable-log-stats"],
            stage_init_timeout=900,
        ),
        id="gepard",
    )
]


def _base_config(omni_server, text: str, **extra) -> dict:
    cfg = {
        "model": omni_server.model,
        "input": text,
        "voice": "default",
        "timeout": DEFAULT_TIMEOUT_S,
        "seed": 7,
        # NanoCodec is quieter than the 24 kHz models the PCM HNR helper was
        # calibrated on; keep the catastrophic-failure check, not a quality gate.
        "min_hnr_db": -2.0,
        "transcript_escalation_model": "large-v3",
    }
    cfg.update(extra)
    return cfg


def _wav_sample_rate(wav_bytes: bytes) -> int:
    return struct.unpack_from("<I", wav_bytes, 24)[0]


def _wav_pcm_payload_len(wav_bytes: bytes) -> int:
    data, _sr = sf.read(BytesIO(wav_bytes), dtype="int16")
    return int(data.size) * 2


def _pcm16le_to_wav(pcm: bytes, sample_rate: int = SAMPLE_RATE) -> bytes:
    """Whisper helpers expect a container; streamed speech is raw s16le."""
    buf = BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm)
    return buf.getvalue()


def _pcm16_samples(pcm: bytes) -> array.array:
    samples = array.array("h")
    samples.frombytes(pcm)
    return samples


def _pcm16_diag(label: str, left: bytes, right: bytes) -> None:
    extra_frames = (len(left) - len(right)) // PCM16_BYTES_PER_FRAME
    overlap = min(len(left), len(right))
    first_frame_match = (
        overlap >= PCM16_BYTES_PER_FRAME and left[:PCM16_BYTES_PER_FRAME] == right[:PCM16_BYTES_PER_FRAME]
    )
    worst = None
    corr = None
    first_diff_sample = None
    if overlap >= 2 and overlap % 2 == 0:
        a = _pcm16_samples(left[:overlap])
        b = _pcm16_samples(right[:overlap])
        worst = max(abs(x - y) for x, y in zip(a, b, strict=True))
        for i, (x, y) in enumerate(zip(a, b, strict=True)):
            if x != y:
                first_diff_sample = i
                break
        try:
            corr = round(statistics.correlation(a, b), 6)
        except statistics.StatisticsError:
            corr = None
    print(
        f"[Gepard pcm-diag] {label} left={len(left)} right={len(right)} "
        f"extra_frames={extra_frames} first_frame_match={first_frame_match} "
        f"first_diff_sample={first_diff_sample} max_abs={worst} corr={corr}"
    )


def _scan_logs_for_preemption(omni_server) -> None:
    log_paths = getattr(omni_server, "_stage_log_paths", {}) or {}
    hits: list[str] = []
    for path in log_paths.values():
        if path is None or not path.exists():
            continue
        text = path.read_text(encoding="utf-8", errors="replace").lower()
        if "preempt" in text:
            hits.append(str(path))
    assert not hits, f"preemption mentioned in stage logs: {hits}"


@hardware_test(res={"cuda": "L4"}, num_cards=1)
@pytest.mark.parametrize("omni_server", tts_server_params, indirect=True)
@pytest.mark.parametrize("response_format", ["wav", "pcm"])
def test_text_to_audio_basic(omni_server, online_client, response_format: str, run_level: str) -> None:
    text = "Hello, this is Gepard speaking."
    # whisper-small/large-v3 mishear 22.05 kHz WAV of this prompt as
    # "Jeb Ard" / "Jeff Bard" (cosine 0.80 / 0.75). Offline uses keyword
    # containment; the PCM twin below still goes through the 0.9 ASR gate.
    if response_format == "wav":
        url = f"http://{omni_server.host}:{omni_server.port}/v1/audio/speech"
        payload = {
            "model": omni_server.model,
            "input": text,
            "voice": "default",
            "seed": 7,
            "stream": False,
            "response_format": "wav",
        }
        r = requests.post(url, json=payload, timeout=DEFAULT_TIMEOUT_S)
        r.raise_for_status()
        audio = r.content
        assert _wav_sample_rate(audio) == SAMPLE_RATE
        pcm_len = _wav_pcm_payload_len(audio)
    else:
        [resp] = online_client.send_audio_speech_request(
            _base_config(omni_server, text, stream=False, response_format=response_format)
        )
        assert resp.audio_bytes
        pcm_len = len(resp.audio_bytes)
    assert pcm_len % PCM16_BYTES_PER_FRAME == 0
    if run_level in {"advanced_model", "full_model"}:
        duration = pcm_len / 2 / SAMPLE_RATE
        assert 0.5 < duration < 30.0, f"implausible duration {duration:.2f}s"


@hardware_test(res={"cuda": "L4"}, num_cards=1)
@pytest.mark.parametrize("omni_server", tts_server_params, indirect=True)
def test_seed_determinism(omni_server, online_client) -> None:
    text = "Hello, this is Gepard speaking."
    [a] = online_client.send_audio_speech_request(_base_config(omni_server, text, seed=7, response_format="pcm"))
    [b] = online_client.send_audio_speech_request(_base_config(omni_server, text, seed=7, response_format="pcm"))
    [c] = online_client.send_audio_speech_request(_base_config(omni_server, text, seed=11, response_format="pcm"))
    assert a.audio_bytes == b.audio_bytes
    assert a.audio_bytes != c.audio_bytes


@hardware_test(res={"cuda": "L4"}, num_cards=1)
@pytest.mark.parametrize("omni_server", tts_server_params, indirect=True)
def test_streaming_pcm_incremental(omni_server, online_client) -> None:
    text = "Hello, this is Gepard speaking."
    url = f"http://{omni_server.host}:{omni_server.port}/v1/audio/speech"
    payload = {
        "model": omni_server.model,
        "input": text,
        "voice": "default",
        "seed": 7,
        "stream": True,
        "stream_format": "audio",
        "response_format": "pcm",
    }
    start = time.perf_counter()
    arrivals: list[tuple[float, int]] = []
    streamed = bytearray()
    with requests.post(url, json=payload, stream=True, timeout=DEFAULT_TIMEOUT_S) as resp:
        resp.raise_for_status()
        content_type = resp.headers.get("content-type") or ""
        assert "pcm" in content_type or "octet-stream" in content_type
        for chunk in resp.iter_content(chunk_size=None):
            if not chunk:
                continue
            arrivals.append((time.perf_counter(), len(chunk)))
            streamed.extend(chunk)
    assert len(arrivals) >= 2, f"expected incremental chunks, got {len(arrivals)}"
    assert arrivals[1][0] > arrivals[0][0]
    ttfa_ms = (arrivals[0][0] - start) * 1000.0
    cadence_ms = [(arrivals[i][0] - arrivals[i - 1][0]) * 1000.0 for i in range(1, len(arrivals))]
    print(f"[Gepard streaming] TTFA_ms={ttfa_ms:.1f} chunk_cadence_ms={cadence_ms}")

    [non_stream] = online_client.send_audio_speech_request(
        _base_config(omni_server, text, seed=7, stream=False, response_format="pcm")
    )
    assert bytes(streamed) == non_stream.audio_bytes


@hardware_test(res={"cuda": "L4"}, num_cards=1)
@pytest.mark.parametrize("omni_server", tts_server_params, indirect=True)
def test_seeded_streaming_stops(omni_server) -> None:
    url = f"http://{omni_server.host}:{omni_server.port}/v1/audio/speech"
    payload = {
        "model": omni_server.model,
        "input": "Hello, this is Gepard speaking.",
        "voice": "default",
        "seed": 7,
        "stream": True,
        "stream_format": "audio",
        "response_format": "pcm",
    }
    with requests.post(url, json=payload, stream=True, timeout=DEFAULT_TIMEOUT_S) as resp:
        resp.raise_for_status()
        audio = b"".join(chunk for chunk in resp.iter_content(chunk_size=None) if chunk)
    duration = len(audio) / 2 / SAMPLE_RATE
    assert duration < 30.0, f"seeded stream ran {duration:.1f}s — stop constraint dropped?"


@hardware_test(res={"cuda": "L4"}, num_cards=1)
@pytest.mark.parametrize("omni_server", tts_server_params, indirect=True)
def test_sse_stream_emits_delta_then_done(omni_server) -> None:
    url = f"http://{omni_server.host}:{omni_server.port}/v1/audio/speech"
    payload = {
        "model": omni_server.model,
        "input": "Hello, this is Gepard speaking.",
        "voice": "default",
        "seed": 7,
        "stream": True,
        "stream_format": "sse",
        "response_format": "pcm",
    }
    with requests.post(url, json=payload, stream=True, timeout=DEFAULT_TIMEOUT_S) as resp:
        resp.raise_for_status()
        assert resp.headers.get("content-type", "").startswith("text/event-stream")
        body = b"".join(resp.iter_content(chunk_size=None)).decode("utf-8", errors="replace")
    assert "speech.audio.delta" in body
    assert "speech.audio.done" in body


@hardware_test(res={"cuda": "L4"}, num_cards=1)
@pytest.mark.parametrize("omni_server", tts_server_params, indirect=True)
def test_concurrent_requests_stay_isolated(omni_server, online_client, run_level: str) -> None:
    def _one(text: str):
        return online_client.send_audio_speech_request(_base_config(omni_server, text, seed=7, response_format="wav"))[
            0
        ]

    texts = list(_DISTINGUISHABLE_PROMPTS)
    with ThreadPoolExecutor(max_workers=4) as pool:
        responses = list(pool.map(_one, texts))

    assert all(r.success for r in responses)
    for i in range(len(responses)):
        for j in range(i + 1, len(responses)):
            assert responses[i].audio_bytes != responses[j].audio_bytes

    if run_level in {"advanced_model", "full_model"}:
        for text, resp in zip(texts, responses, strict=True):
            keyword = _DISTINGUISHABLE_PROMPTS[text]
            transcript = convert_audio_bytes_to_text(resp.audio_bytes, language="en").lower()
            assert keyword in transcript, f"expected {keyword!r} in {transcript!r} for {text!r}"

    _scan_logs_for_preemption(omni_server)


def _speech_url(omni_server) -> str:
    return f"http://{omni_server.host}:{omni_server.port}/v1/audio/speech"


def _stream_payload(omni_server, text: str, **extra) -> dict:
    payload = {
        "model": omni_server.model,
        "input": text,
        "voice": "default",
        "seed": 7,
        "stream": True,
        "stream_format": "audio",
        "response_format": "pcm",
    }
    payload.update(extra)
    return payload


@dataclass
class _StreamCapture:
    pcm: bytes
    arrivals: list[tuple[float, int]]

    @property
    def duration_s(self) -> float:
        return len(self.pcm) / 2 / SAMPLE_RATE


def _capture_stream(
    url: str,
    payload: dict,
    timeout_s: float,
    *,
    sleep_after_first_s: float = 0.0,
) -> _StreamCapture:
    start = time.perf_counter()
    arrivals: list[tuple[float, int]] = []
    pcm = bytearray()
    with requests.post(url, json=payload, stream=True, timeout=timeout_s) as resp:
        resp.raise_for_status()
        first = True
        for chunk in resp.iter_content(chunk_size=None):
            if not chunk:
                continue
            arrivals.append((time.perf_counter() - start, len(chunk)))
            pcm.extend(chunk)
            if first and sleep_after_first_s > 0:
                time.sleep(sleep_after_first_s)
            first = False
    return _StreamCapture(bytes(pcm), arrivals)


@hardware_test(res={"cuda": "L4"}, num_cards=1)
@pytest.mark.parametrize("omni_server", tts_server_params, indirect=True)
def test_concurrent_streaming_stays_isolated(omni_server, run_level: str) -> None:
    url = _speech_url(omni_server)
    texts = list(_DISTINGUISHABLE_PROMPTS)

    def _one(text: str) -> tuple[str, _StreamCapture]:
        return text, _capture_stream(url, _stream_payload(omni_server, text), DEFAULT_TIMEOUT_S)

    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(_one, texts))

    captures = [cap for _, cap in results]
    assert all(len(c.arrivals) >= 2 for c in captures)
    for i in range(len(captures)):
        for j in range(i + 1, len(captures)):
            assert captures[i].pcm != captures[j].pcm

    if run_level in {"advanced_model", "full_model"}:
        for text, cap in results:
            keyword = _DISTINGUISHABLE_PROMPTS[text]
            transcript = convert_audio_bytes_to_text(_pcm16le_to_wav(cap.pcm), language="en").lower()
            assert keyword in transcript, f"expected {keyword!r} in {transcript!r} for {text!r}"

    _scan_logs_for_preemption(omni_server)


@hardware_test(res={"cuda": "L4"}, num_cards=1)
@pytest.mark.parametrize("omni_server", tts_server_params, indirect=True)
def test_slow_client_still_completes(omni_server, online_client) -> None:
    """Slow reader must finish, and seed=7 must match a prior non-stream.

    Runs before ``test_client_disconnect_releases_capacity`` so an abort
    storm cannot contaminate this comparison. Prints overlap diagnostics
    before the bit-exact asserts.
    """
    text = "Hello, this is Gepard speaking."
    [baseline] = online_client.send_audio_speech_request(
        _base_config(omni_server, text, seed=7, stream=False, response_format="pcm")
    )
    cap = _capture_stream(
        url=_speech_url(omni_server),
        payload=_stream_payload(omni_server, text),
        timeout_s=DEFAULT_TIMEOUT_S,
        sleep_after_first_s=3.0,
    )
    assert len(cap.arrivals) >= 2
    assert len(cap.pcm) % PCM16_BYTES_PER_FRAME == 0
    assert 0.5 < cap.duration_s < 30.0
    print(f"[Gepard slow-client] chunks={len(cap.arrivals)} duration_s={cap.duration_s:.3f}")
    _pcm16_diag("slow-vs-prior-nonstream", cap.pcm, baseline.audio_bytes)
    [after] = online_client.send_audio_speech_request(
        _base_config(omni_server, text, seed=7, stream=False, response_format="pcm")
    )
    _pcm16_diag("after-nonstream-vs-prior", after.audio_bytes, baseline.audio_bytes)
    assert cap.pcm == baseline.audio_bytes
    assert after.audio_bytes == baseline.audio_bytes


@hardware_test(res={"cuda": "L4"}, num_cards=1)
@pytest.mark.parametrize("omni_server", tts_server_params, indirect=True)
def test_client_disconnect_releases_capacity(omni_server, online_client) -> None:
    """A follow-up request must complete after in-flight streams are closed.

    Logs PCM diagnostics against a pre-abort ``seed=7`` baseline. Bit-exact
    equality after abort is not asserted here: a dying stream can still share
    a GPU batch with the next generate after ``abort()`` returns, and Gepard's
    in-model stop head / bf16 GEMM are not row-stable in that batch.
    """
    url = _speech_url(omni_server)
    text = "Hello, this is Gepard speaking."
    [baseline] = online_client.send_audio_speech_request(
        _base_config(omni_server, text, seed=7, stream=False, response_format="pcm")
    )
    long_payload = _stream_payload(
        omni_server,
        "Please keep talking about the weather, the window, coffee, and purple flowers.",
        max_new_tokens=80,
    )
    for _ in range(4):
        resp = requests.post(url, json=long_payload, stream=True, timeout=DEFAULT_TIMEOUT_S)
        resp.raise_for_status()
        try:
            chunk = next(c for c in resp.iter_content(chunk_size=None) if c)
            assert chunk
        finally:
            resp.close()

    t0 = time.perf_counter()
    follow = requests.post(
        url,
        json={
            "model": omni_server.model,
            "input": text,
            "voice": "default",
            "seed": 7,
            "stream": False,
            "response_format": "pcm",
        },
        timeout=DEFAULT_TIMEOUT_S,
    )
    follow.raise_for_status()
    elapsed = time.perf_counter() - t0
    pcm_len = len(follow.content)
    assert pcm_len % PCM16_BYTES_PER_FRAME == 0
    print(f"[Gepard disconnect] followup_s={elapsed:.2f} pcm_bytes={pcm_len}")
    _pcm16_diag("disconnect-followup-vs-prior", follow.content, baseline.audio_bytes)


@hardware_test(res={"cuda": "L4"}, num_cards=1)
@pytest.mark.parametrize("omni_server", tts_server_params, indirect=True)
def test_mixed_stream_and_nonstream_isolated(omni_server, online_client, run_level: str) -> None:
    url = _speech_url(omni_server)
    stream_text = "He drinks coffee every morning."
    nonstream_text = "My favorite color is purple."

    def _stream():
        return _capture_stream(url, _stream_payload(omni_server, stream_text), DEFAULT_TIMEOUT_S)

    def _nonstream():
        return online_client.send_audio_speech_request(
            _base_config(omni_server, nonstream_text, seed=7, response_format="wav")
        )[0]

    with ThreadPoolExecutor(max_workers=2) as pool:
        stream_future = pool.submit(_stream)
        nonstream_future = pool.submit(_nonstream)
        streamed = stream_future.result()
        nonstream = nonstream_future.result()

    assert streamed.pcm
    assert nonstream.success
    assert streamed.pcm != nonstream.audio_bytes
    if run_level in {"advanced_model", "full_model"}:
        stream_tr = convert_audio_bytes_to_text(_pcm16le_to_wav(streamed.pcm), language="en").lower()
        nonstream_tr = convert_audio_bytes_to_text(nonstream.audio_bytes, language="en").lower()
        assert "coffee" in stream_tr, stream_tr
        assert "purple" in nonstream_tr, nonstream_tr
    _scan_logs_for_preemption(omni_server)


@hardware_test(res={"cuda": "L4"}, num_cards=1)
@pytest.mark.parametrize("omni_server", tts_server_params, indirect=True)
def test_max_new_tokens_caps_whole_frames(omni_server) -> None:
    frames = 16
    url = _speech_url(omni_server)
    r = requests.post(
        url,
        json={
            "model": omni_server.model,
            "input": "Hello, this is Gepard speaking.",
            "voice": "default",
            "seed": 7,
            "stream": False,
            "response_format": "pcm",
            "max_new_tokens": frames,
        },
        timeout=DEFAULT_TIMEOUT_S,
    )
    assert r.status_code == 200, r.text
    pcm_len = len(r.content)
    assert pcm_len % PCM16_BYTES_PER_FRAME == 0
    assert pcm_len == frames * PCM16_BYTES_PER_FRAME, f"got {pcm_len} bytes, want {frames} frames"
    duration = pcm_len / 2 / SAMPLE_RATE
    assert duration < 1.0, f"budget cap should stop near 16 frames, got {duration:.2f}s"

    over = requests.post(
        url,
        json={
            "model": omni_server.model,
            "input": "Hello, this is Gepard speaking.",
            "voice": "default",
            "max_new_tokens": 4097,
        },
        timeout=DEFAULT_TIMEOUT_S,
    )
    assert over.status_code == 400
    assert "max_new_tokens" in over.text
