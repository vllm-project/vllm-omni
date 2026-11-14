# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""SGLang-Omni wire adapter for the shared duplex dataset replay."""

from __future__ import annotations

import asyncio
import math
import time
from collections.abc import AsyncIterator, Mapping
from contextlib import asynccontextmanager
from typing import Protocol
from urllib.parse import urlsplit, urlunsplit
from uuid import uuid4

import pybase64 as base64

from vllm_omni.benchmarks.duplex.omni_duplex_eval_media import iter_timed_av_units
from vllm_omni.benchmarks.duplex.omni_duplex_eval_runner import (
    DuplexReplaySession,
    ReplayDrainResult,
)
from vllm_omni.benchmarks.duplex_session_inputs import (
    FRAME_TRANSPORT_SGLANG_IMAGE,
    TURN_DETECTION_NONE,
)
from vllm_omni.experimental.fullduplex.client import (
    PCM16_BYTES_PER_SAMPLE,
    PCM16_SAMPLE_RATE,
    RealtimeDuplexClient,
    RealtimeEventCollector,
    wait_for,
)

_HANDSHAKE_TIMEOUT_S = 60.0


class _RealtimeClient(Protocol):
    events: RealtimeEventCollector

    async def send(self, event: dict[str, object]) -> None: ...

    def advance_media_clock(self, media_clock_ms: int) -> None: ...

    def raise_if_reader_stopped(self) -> None: ...


def build_sglang_realtime_url(url: str) -> str:
    """Convert an HTTP Realtime endpoint to its WebSocket equivalent."""
    parts = urlsplit(url)
    if parts.scheme in {"http", "https"}:
        parts = parts._replace(scheme="ws" if parts.scheme == "http" else "wss")
    if parts.scheme not in {"ws", "wss"} or not parts.netloc:
        raise ValueError(f"Unsupported SGLang Realtime URL: {url!r}")
    path = parts.path or "/v1/realtime"
    return urlunsplit((parts.scheme, parts.netloc, path, parts.query, parts.fragment))


def _related_event_ids(event: Mapping[str, object]) -> tuple[str, ...]:
    related: list[str] = []
    for key in ("client_event_id", "event_id", "related_event_id"):
        value = event.get(key)
        if isinstance(value, str):
            related.append(value)
    error = event.get("error")
    if isinstance(error, Mapping):
        for key in ("event_id", "related_event_id"):
            value = error.get(key)
            if isinstance(value, str):
                related.append(value)
    return tuple(related)


def _error_message(event: Mapping[str, object]) -> str:
    error = event.get("error")
    if isinstance(error, Mapping):
        message = error.get("message")
        if isinstance(message, str) and message:
            return message
    return str(event)


def _reference_audio_payload(ref_audio: str) -> dict[str, str]:
    prefix, separator, encoded = ref_audio.partition(",")
    if separator != "," or prefix.lower() != "data:audio/wav;base64":
        raise ValueError("SGLang reference audio must be a base64 audio/wav data URL")
    try:
        base64.b64decode(encoded, validate=True)
    except (TypeError, ValueError) as exc:
        raise ValueError("SGLang reference audio contains invalid base64") from exc
    return {"media_type": "audio/wav", "data": encoded}


def _sglang_extension(ref_audio: str | None, extra_body: dict[str, object] | None) -> dict[str, object]:
    options = dict(extra_body or {})
    # This flag selects vLLM's route and is not a model sampling parameter.
    options.pop("native_duplex", None)
    nested = options.pop("sglang", {})
    if not isinstance(nested, Mapping):
        raise ValueError("extra_body.sglang must be an object")
    extension = dict(nested)
    nested_sampling = extension.pop("sampling", {})
    if not isinstance(nested_sampling, Mapping):
        raise ValueError("extra_body.sglang.sampling must be an object")
    sampling = dict(nested_sampling)
    top_level_sampling = options.pop("sampling", {})
    if not isinstance(top_level_sampling, Mapping):
        raise ValueError("extra_body.sampling must be an object")
    sampling.update(top_level_sampling)
    sampling.update(options)
    # vLLM's native replay client uses zero forced-listen turns. Match that
    # default while preserving an explicit benchmark override.
    sampling.setdefault("force_listen_count", 0)
    if sampling:
        extension["sampling"] = sampling
    if ref_audio is not None:
        extension["reference_audio"] = _reference_audio_payload(ref_audio)
    return extension


def _positive_int(value: object, *, label: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"SGLang session did not grant a positive {label}")
    return value


class SGLangReplaySession:
    """Translate shared replay operations to SGLang's native duplex protocol."""

    frame_transport = FRAME_TRANSPORT_SGLANG_IMAGE
    supports_session_stats = False

    def __init__(self, client: _RealtimeClient) -> None:
        self._client = client
        self.events: RealtimeEventCollector = client.events
        self._native_unit_ms = 0
        self._input_modalities: set[str] = set()
        self.session_configuration: dict[str, object] = {}

    async def _send_and_wait(
        self,
        event: dict[str, object],
        *,
        response_type: str,
        timeout_s: float,
    ) -> dict[str, object]:
        event_id = event.get("event_id")
        if not isinstance(event_id, str):
            raise ValueError("correlated SGLang commands require event_id")
        from_index = len(self.events.events)
        deadline = time.monotonic() + timeout_s
        await asyncio.wait_for(self._client.send(event), timeout=timeout_s)

        def arrived() -> bool:
            for received in self.events.events[from_index:]:
                if received.get("type") == "error":
                    return True
                if received.get("type") == response_type and event_id in _related_event_ids(received):
                    return True
            self._client.raise_if_reader_stopped()
            return False

        remaining_s = deadline - time.monotonic()
        if remaining_s <= 0:
            raise TimeoutError(f"Timed out waiting for {response_type}")
        await wait_for(arrived, timeout_s=remaining_s, label=response_type)
        for received in self.events.events[from_index:]:
            if received.get("type") == "error":
                raise ConnectionError(_error_message(received))
            if received.get("type") == response_type and event_id in _related_event_ids(received):
                return received
        raise RuntimeError(f"SGLang {response_type} waiter completed without an event")

    async def configure(
        self,
        model: str,
        *,
        ref_audio: str | None,
        instructions: str | None,
        extra_body: dict[str, object] | None,
        turn_detection: str,
    ) -> None:
        del model
        if turn_detection != TURN_DETECTION_NONE:
            raise ValueError("SGLang native duplex replay supports turn_detection='none' only")

        def session_created() -> bool:
            if self.events.count("session.created"):
                return True
            self._client.raise_if_reader_stopped()
            return False

        await wait_for(session_created, timeout_s=_HANDSHAKE_TIMEOUT_S, label="session.created")
        session: dict[str, object] = {"output_modalities": ["audio"]}
        if instructions is not None:
            session["instructions"] = instructions
        extension = _sglang_extension(ref_audio, extra_body)
        if extension:
            session["sglang"] = extension
        updated = await self._send_and_wait(
            {
                "type": "session.update",
                "event_id": f"evt_{uuid4().hex}",
                "session": session,
            },
            response_type="session.updated",
            timeout_s=_HANDSHAKE_TIMEOUT_S,
        )
        updated_session = updated.get("session")
        sglang = updated_session.get("sglang") if isinstance(updated_session, Mapping) else None
        granted = sglang.get("granted") if isinstance(sglang, Mapping) else None
        if not isinstance(granted, Mapping) or granted.get("native_full_duplex") is not True:
            raise ValueError("SGLang session did not grant native full duplex")
        if granted.get("client_commit") is not False:
            raise ValueError("SGLang session did not grant server-owned native turn control")
        if granted.get("tail_policy") != "pad":
            raise ValueError("SGLang session must grant tail_policy='pad' for complete dataset replay")
        input_modalities = granted.get("input_modalities")
        if not isinstance(input_modalities, list) or "audio" not in input_modalities:
            raise ValueError("SGLang session did not grant audio input")
        self._input_modalities = {str(modality) for modality in input_modalities}
        output_modalities = granted.get("output_modalities")
        if not isinstance(output_modalities, list) or "audio" not in output_modalities:
            raise ValueError("SGLang session did not grant audio output")
        self._native_unit_ms = _positive_int(granted.get("native_unit_ms"), label="native_unit_ms")
        input_format = granted.get("input_audio_format")
        input_rate = input_format.get("rate") if isinstance(input_format, Mapping) else None
        if input_rate != PCM16_SAMPLE_RATE:
            raise ValueError(f"SGLang session requires {input_rate!r} Hz input, expected {PCM16_SAMPLE_RATE}")
        output_format = granted.get("output_audio_format")
        output_rate = output_format.get("rate") if isinstance(output_format, Mapping) else None
        self.events.output_sample_rate_hz = _positive_int(output_rate, label="output audio rate")
        effective_extension = updated_session.get("sglang") if isinstance(updated_session, Mapping) else None
        effective_sampling = effective_extension.get("sampling") if isinstance(effective_extension, Mapping) else None
        requested_sampling = extension.get("sampling")
        supported_sampling = granted.get("sampling_parameters")
        self.session_configuration = {
            "backend": "sglang",
            "requested_sampling": dict(requested_sampling) if isinstance(requested_sampling, Mapping) else {},
            "effective_sampling": dict(effective_sampling) if isinstance(effective_sampling, Mapping) else {},
            "sampling_source": "server_echo",
            "supported_sampling_parameters": list(supported_sampling) if isinstance(supported_sampling, list) else [],
        }

    def prepare_input(self, pcm: bytes, frames: tuple[tuple[float, bytes], ...]) -> bytes:
        del frames
        return pcm

    async def stream_input(
        self,
        pcm: bytes,
        frames: tuple[tuple[float, bytes], ...],
        *,
        sample_id: str,
        unit_ms: int,
        realtime: bool,
    ) -> None:
        del sample_id
        if unit_ms != self._native_unit_ms:
            raise ValueError(
                f"SGLang granted {self._native_unit_ms} ms native units, but the replay requested {unit_ms} ms"
            )
        if frames and "image" not in self._input_modalities:
            raise ValueError("SGLang session did not grant image input required by this sample")
        bytes_per_second = PCM16_SAMPLE_RATE * PCM16_BYTES_PER_SAMPLE
        audio_end_ms = 0.0
        started_at = time.monotonic()
        for sequence, (chunk, timed_frames) in enumerate(iter_timed_av_units(pcm, iter(frames), unit_ms=unit_ms)):
            if not chunk:
                continue
            duration_ms = len(chunk) * 1000 / bytes_per_second
            start_ms = audio_end_ms
            audio_end_ms += duration_ms
            media_clock_ms = math.floor(audio_end_ms)
            self._client.advance_media_clock(media_clock_ms)
            for timestamp_s, frame in timed_frames:
                await self._client.send(
                    {
                        "type": "sglang.input_image.append",
                        "event_id": f"evt_{uuid4().hex}",
                        "image": base64.b64encode(frame).decode("ascii"),
                        "sglang": {"t_ms": timestamp_s * 1000},
                    }
                )
            await self._client.send(
                {
                    "type": "input_audio_buffer.append",
                    "event_id": f"evt_{uuid4().hex}",
                    "audio": base64.b64encode(chunk).decode("ascii"),
                    "sglang": {"seq": sequence, "t_start_ms": start_ms},
                }
            )
            if realtime:
                await asyncio.sleep(max(0.0, started_at + audio_end_ms / 1000 - time.monotonic()))

    async def finish_input(self, *, timeout_s: float) -> ReplayDrainResult:
        try:
            await self._send_and_wait(
                {"type": "sglang.input_audio.end", "event_id": f"evt_{uuid4().hex}"},
                response_type="sglang.input_audio.drained",
                timeout_s=timeout_s,
            )
        except TimeoutError as exc:
            return ReplayDrainResult(
                drain_timeout=str(exc) or "SGLang input drain timed out",
                drain_timeout_state={
                    "response_ids": list(self.events.response_ids),
                    "unfinished_response_ids": [
                        response_id
                        for response_id in self.events.response_ids
                        if not self.events.response_is_done(response_id)
                    ],
                },
            )
        return ReplayDrainResult(
            response_done=all(self.events.response_is_done(response_id) for response_id in self.events.response_ids),
            input_committed=True,
        )

    async def acknowledge_playback(self) -> None:
        # SGLang's native protocol does not require playback acknowledgements.
        return None

    async def get_session_stats(self, *, timeout_s: float, event_id: str) -> dict[str, object]:
        del timeout_s, event_id
        raise NotImplementedError("SGLang does not expose vLLM session.stats")

    async def close(self, *, timeout_s: float) -> None:
        if self.events.count("session.closed"):
            return
        await self._send_and_wait(
            {"type": "session.close", "event_id": f"evt_{uuid4().hex}"},
            response_type="session.closed",
            timeout_s=timeout_s,
        )


@asynccontextmanager
async def open_sglang_replay_session(
    url: str, model: str, additional_headers: dict[str, str] | None
) -> AsyncIterator[DuplexReplaySession]:
    del model
    async with RealtimeDuplexClient(build_sglang_realtime_url(url), additional_headers=additional_headers) as client:
        yield SGLangReplaySession(client)
