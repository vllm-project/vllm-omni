# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Matched Seed-TTS audio + text requests through the public duplex API.

Unlike native MiniCPM TTS, Qwen consumes an ordinary audio turn. Explicit
mode ends the turn itself at the end of the reference speech; VAD mode streams
the silent tail as well and must receive a real speech-stop event, never
sending commit/response.create. Both stream the same content PCM at real time.

Every latency here is measured from the end of the reference speech, because
that is the instant a live speaker stops talking and starts waiting. Timing
from session start instead would fold the caller's own real-time upload (the
whole reference clip) into TTFT/TTFP/RTF: the numbers would be dominated by
how long each dataset clip happens to be, RTF could not be compared against
its ``< 1`` SLO, and a model regression would be invisible inside them.
"""

from __future__ import annotations

import asyncio
import contextlib
import io
import time
import wave
from typing import Any

import pybase64 as base64
from vllm.benchmarks.lib.endpoint_request_func import RequestFuncInput

from vllm_omni.benchmarks.data_modules.seed_tts_dataset import (
    SEED_TTS_INPUT_SAMPLE_RATE_HZ,
    SEED_TTS_SILENT_TAIL_MS,
)
from vllm_omni.clients.duplex import (
    AudioFormat,
    DuplexClient,
    EventCollector,
    SessionConfig,
    acknowledge_collected_playback,
    build_realtime_url,
    summarize_session_request_metrics,
    wait_for_condition,
)
from vllm_omni.metrics.definitions import compute_audio_rtf

#: Input is appended in chunks of this size, paced at real time.
_CHUNK_MS = 200

#: Server-VAD silence threshold. It must stay below the tail the dataset
#: appends, or VAD never sees enough silence and every request stalls until
#: the response timeout.
_VAD_SILENCE_MS = 800
assert _VAD_SILENCE_MS < SEED_TTS_SILENT_TAIL_MS, (
    f"server-VAD silence threshold {_VAD_SILENCE_MS}ms must be shorter than the "
    f"{SEED_TTS_SILENT_TAIL_MS}ms silent tail appended to each reference clip"
)


def _input_pcm(encoded_wav: str) -> bytes:
    with wave.open(io.BytesIO(base64.b64decode(encoded_wav)), "rb") as audio:
        if (audio.getframerate(), audio.getnchannels(), audio.getsampwidth()) != (
            SEED_TTS_INPUT_SAMPLE_RATE_HZ,
            1,
            2,
        ):
            raise ValueError("Seed-TTS Realtime input must be normalized 24 kHz mono PCM16")
        return audio.readframes(audio.getnframes())


def _as_float(value: object, *, field: str) -> float:
    """Narrow one raw measurement out of the client's ``dict[str, object]``."""
    if not isinstance(value, int | float) or isinstance(value, bool):
        raise RuntimeError(f"Realtime timing summary returned a non-numeric {field}: {value!r}")
    return float(value)


async def run_realtime_seed_tts(request: RequestFuncInput) -> dict[str, Any]:
    """Run one independent utterance, excluding session setup/teardown from latency.

    The ``seed_tts_*`` attributes are attached at runtime by
    ``_attach_seed_tts_to_request_func_input``, so they are read with
    ``getattr`` like every other Seed-TTS backend in this package.
    """
    encoded_audio = getattr(request, "seed_tts_input_audio", None)
    if not encoded_audio:
        raise ValueError("Realtime chat requires --seed-tts-reference-as-input")
    pcm = _input_pcm(encoded_audio)
    trigger = (request.extra_body or {}).get("realtime_trigger", "explicit")
    if trigger not in {"explicit", "vad"}:
        raise ValueError(f"Unknown realtime_trigger: {trigger}")
    vad = trigger == "vad"
    config = SessionConfig(
        input_audio=AudioFormat("pcm16", SEED_TTS_INPUT_SAMPLE_RATE_HZ),
        output_audio=AudioFormat("pcm16", SEED_TTS_INPUT_SAMPLE_RATE_HZ),
        instructions=getattr(request, "seed_tts_system_prompt", None),
        temperature=0.0,
        max_output_tokens=request.output_len,
        # Native model-owned auto_response is not Qwen server-VAD triggering.
        auto_response=False,
        overlap_policy="barge_in_on_speech" if vad else "listen_only",
        playback_commit_policy="ack_only",
        turn_detection=(
            {
                "type": "server_vad",
                "threshold": 0.5,
                "prefix_padding_ms": 300,
                "silence_duration_ms": _VAD_SILENCE_MS,
                "create_response": True,
                # Required by the public server-VAD contract; this workload
                # never supplies overlapping speech while a response is active.
                "interrupt_response": True,
            }
            if vad
            else None
        ),
        # No extra_body: the server returns `stage_metrics: {}` on this path
        # (measured against Qwen3-Omni), so asking for stage metrics buys
        # nothing and only adds server work inside a latency measurement.
        extra_body={},
    )
    events = EventCollector()
    setup_started = time.monotonic()
    async with DuplexClient(
        build_realtime_url(request.api_url, request.model_name or request.model),
        model=request.model_name or request.model,
        config=config,
        reconnect=None,
        heartbeat_interval_s=None,
        handshake_timeout_s=120.0,
    ) as client:
        consumer = asyncio.create_task(events.consume(client))

        def reraise_consumer_failure() -> None:
            """Surface a dead event reader instead of stalling until timeout."""
            if consumer.done() and not consumer.cancelled():
                exc = consumer.exception()
                if exc is not None:
                    raise RuntimeError("Realtime event stream stopped") from exc

        try:
            started = time.monotonic()
            request_started_perf = time.perf_counter()
            await client.send(
                {
                    "type": "conversation.item.create",
                    "item": {
                        "type": "message",
                        "role": "user",
                        "content": [{"type": "input_text", "text": request.prompt}],
                    },
                }
            )
            chunk_bytes = config.input_audio.byte_count(_CHUNK_MS)
            tail_bytes = config.input_audio.byte_count(SEED_TTS_SILENT_TAIL_MS)
            content_bytes = max(0, len(pcm) - tail_bytes)
            # The silent tail exists so server VAD can find the endpoint. An
            # explicit client ends the turn when the speaker stops, so making
            # it stream a second of silence first would charge it for latency
            # no real caller would pay.
            upload = pcm if vad else pcm[:content_bytes]
            # Pace against absolute deadlines: sleeping for each chunk's
            # duration instead lets per-iteration overhead accumulate, and that
            # drift lands directly in the measured latency.
            deadline = started
            content_end = started
            for offset in range(0, len(upload), chunk_bytes):
                reraise_consumer_failure()
                if events.errors():
                    raise RuntimeError(f"Realtime server error: {events.errors()[-1]}")
                if vad and offset < content_bytes and events.count("response.created"):
                    raise RuntimeError("VAD split the reference audio before its end; comparison requires one turn")
                chunk = upload[offset : offset + chunk_bytes]
                await client.append_audio(chunk)
                deadline += config.input_audio.duration_ms(len(chunk)) / 1000.0
                if offset < content_bytes:
                    # Wall clock at which this chunk's samples finish arriving:
                    # the instant a live speaker would have stopped talking.
                    content_end = max(deadline, time.monotonic())
                await asyncio.sleep(max(0.0, deadline - time.monotonic()))
            upload_finished = time.monotonic()
            explicit_trigger_sent = None
            if not vad:
                explicit_trigger_sent = time.monotonic()
                await client.commit(create_response=False)
                await client.send({"type": "response.create"})
            await wait_for_condition(
                lambda: events.count("response.done") > 0 or bool(events.errors()) or consumer.done(),
                timeout_s=180.0,
                label="Seed-TTS Realtime response.done",
            )
            reraise_consumer_failure()
            if events.errors():
                raise RuntimeError(f"Realtime server error: {events.errors()[-1]}")
            if vad and not (
                events.count("input_audio_buffer.speech_started") and events.count("input_audio_buffer.speech_stopped")
            ):
                raise RuntimeError("VAD benchmark did not observe a speech-start / speech-stop pair")
            if len(events.response_ids) != 1:
                raise RuntimeError(f"Expected one response, got {len(events.response_ids)}")
            response_id = events.response_ids[0]
            done = next(event for event in events.events if event.get("type") == "response.done")
            response = done.get("response")
            status = response.get("status") if isinstance(response, dict) else None
            if status != "completed":
                raise RuntimeError(f"Realtime response did not complete: {done}")
            metrics = events.global_timing_summary(
                after_s=started,
                window_started_at_s=content_end,
                response_ids=[response_id],
                measurement_origin={
                    "ttft": "end of reference speech to first non-empty text delta",
                    "ttfp": "end of reference speech to first audio packet",
                    "rtf": "end of reference speech to last audio packet divided by output audio duration",
                },
            )
            if not metrics or not metrics.get("audio_duration_ms") or metrics.get("ttft_ms") is None:
                raise RuntimeError("Realtime response omitted text or audio")
            ttft_ms = _as_float(metrics["ttft_ms"], field="ttft_ms")
            ttfp_ms = _as_float(metrics["ttfp_ms"], field="ttfp_ms")
            audio_generation_ms = _as_float(metrics["audio_generation_ms"], field="audio_generation_ms")
            audio_duration_ms = _as_float(metrics["audio_duration_ms"], field="audio_duration_ms")
            if ttfp_ms < 0:
                # Audio before the reference speech ended means the turn was
                # split early; the in-loop VAD guard cannot see a split that
                # happens on the final content chunk.
                raise RuntimeError(f"Response started {-ttfp_ms:.0f}ms before the reference speech ended")
            metrics.update(
                response_id=response_id,
                session_id=client.session_id,
                request_index=0,
                utterance_id=getattr(request, "seed_tts_utterance_id", None),
                trigger=trigger,
                session_setup_ms=(started - setup_started) * 1000,
                input_audio_ms=config.input_audio.duration_ms(len(pcm)),
                input_content_ms=config.input_audio.duration_ms(content_bytes),
                input_uploaded_ms=config.input_audio.duration_ms(len(upload)),
                input_upload_ms=(upload_finished - started) * 1000,
                # Kept for diagnosis only: this is the old session-start origin,
                # dominated by the paced upload rather than by the model.
                session_start_to_first_audio_ms=ttfp_ms + (content_end - started) * 1000,
                rtf=compute_audio_rtf(audio_generation_ms / 1000, audio_duration_ms / 1000),
            )
            if explicit_trigger_sent is not None:
                metrics["explicit_commit_to_first_audio_ms"] = ttfp_ms - (explicit_trigger_sent - content_end) * 1000
            # The server emits speech_started/speech_stopped even with
            # turn_detection off, where they follow the client's own commit and
            # say nothing about endpoint detection. Only report them as VAD
            # timings when VAD is what actually ended the turn.
            stopped = events.first_received_at("input_audio_buffer.speech_stopped", after_s=started) if vad else None
            if stopped is not None:
                vad_stop_received_ms = (stopped - content_end) * 1000
                metrics["vad_stop_received_ms"] = vad_stop_received_ms
                metrics["vad_stop_to_first_audio_ms"] = ttfp_ms - vad_stop_received_ms
            finished = events.last_received_at("response.done")
            assert finished is not None
            metrics["session_start_to_response_done_ms"] = (finished - started) * 1000
            output = {
                # E2EL shares TTFT's origin on purpose: ``calculate_metrics``
                # derives TPOT as ``(latency - ttft) / (output_tokens - 1)``
                # whenever the backend reports no engine token count, and
                # mixing origins there would inflate it by the whole upload.
                # ``start_time`` moves with it so ``start_time + latency``
                # stays a real wall-clock interval.
                "start_time": request_started_perf + (content_end - started),
                "generated_text": events.response_text(response_id),
                "ttft": ttft_ms / 1000,
                "audio_ttfp": ttfp_ms / 1000,
                "audio_rtf": metrics["rtf"],
                "audio_duration": audio_duration_ms / 1000,
                "audio_frames": len(events.audio_bytes(response_id)) // 2,
                "latency": finished - content_end,
                "duplex_request_metrics": [metrics],
                "duplex_session_metrics": summarize_session_request_metrics([metrics], session_id=client.session_id),
            }
            await acknowledge_collected_playback(client, events)
            return output
        finally:
            try:
                # Transport close alone leaves a resumable session holding a slot.
                await client.close(timeout_s=30.0)
            finally:
                consumer.cancel()
                # Suppress everything: a consumer that already died carries its
                # own exception, and re-raising it here would replace whatever
                # failure is actually propagating.
                with contextlib.suppress(BaseException):
                    await consumer
