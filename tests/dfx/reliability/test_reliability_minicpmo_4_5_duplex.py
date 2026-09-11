# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Real-process reliability coverage for MiniCPM-o scheduler-native duplex KV."""

from __future__ import annotations

import asyncio
import base64
import json
import os
import time
import uuid
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit
from urllib.request import urlopen

import pytest
import websockets
from websockets.exceptions import ConnectionClosed

from tests.dfx.reliability.helpers import make_process_kill_fault_injector
from tests.e2e.online_serving.helpers.minicpmo_4_5_duplex import (
    DEPLOY_CONFIG,
    MODEL,
    SERVER_PARAMS,
    realtime_url,
    resolve_ref_audio,
    validated_input_wav,
)
from tests.e2e.online_serving.helpers.minicpmo_realtime_duplex_scenarios import (
    _ref_audio_data_url,
)
from tests.helpers.mark import hardware_test
from tests.helpers.runtime import OmniServerParams
from vllm_omni.experimental.fullduplex.client import (
    PCM16_BYTES_PER_SAMPLE,
    PCM16_SAMPLE_RATE,
    build_realtime_url,
    read_pcm16_wav,
)

pytestmark = [pytest.mark.slow, pytest.mark.full_model, pytest.mark.omni]

_CONTEXT_LIMIT = 256
_CONTEXT_LIMIT_SERVER_PARAMS = [
    pytest.param(
        OmniServerParams(
            model=MODEL,
            stage_config_path=DEPLOY_CONFIG,
            use_stage_cli=False,
            server_args=[
                "--trust-remote-code",
                "--stage-overrides",
                json.dumps({"0": {"max_model_len": _CONTEXT_LIMIT}}),
            ],
            deploy_config_overrides={
                # Keep this E2E deterministic even when a vLLM build omits
                # context metrics from an early parked append: the bounded
                # replay journal independently forces the same planned
                # context-rollover path before the 256-token hard ceiling.
                "duplex_session": {
                    # MiniCPM materializes one probe unit to about 81 model
                    # tokens.  The journal limit must admit one complete unit;
                    # 192 tokens still forces rollover on the third append.
                    "kv_recovery_max_replay_tokens_per_session": 192,
                    "kv_rollover_retain_tokens": 96,
                }
            },
        ),
        id="three-stage-stage0-context-256",
    )
]
_LOST_REPLY_SITECUSTOMIZE_DIR = Path(__file__).parent / "fault_injection" / "lost_duplex_append_reply"
_INHERITED_PYTHONPATH = os.environ.get("PYTHONPATH")
_INPUT_FAULT_DIR = Path(__file__).parent / "fault_injection" / "native_input_safety"


def _input_fault_server(mode: str, *, chunked_prefill: bool = False):
    args = ["--trust-remote-code"]
    if chunked_prefill:
        args += ["--stage-overrides", json.dumps({"0": {"max_num_batched_tokens": 8}})]
    return [
        pytest.param(
            OmniServerParams(
                model=MODEL,
                stage_config_path=DEPLOY_CONFIG,
                use_stage_cli=False,
                server_args=args,
                env_dict={
                    "PYTHONPATH": os.pathsep.join(p for p in (str(_INPUT_FAULT_DIR), _INHERITED_PYTHONPATH) if p),
                    "VLLM_OMNI_TEST_NATIVE_INPUT_FAULT": mode,
                },
            ),
            id=f"native-input-{mode}",
        )
    ]


@hardware_test(res={"cuda": "H100"}, num_cards=1)
@pytest.mark.parametrize("omni_server_function", _input_fault_server("preempt", chunked_prefill=True), indirect=True)
def test_native_duplex_kv_preemption_recomputes_history_and_finishes(
    omni_server_function,
    openai_client_function,
    tmp_path,
):
    pcm = read_pcm16_wav(validated_input_wav())[: 1400 * PCM16_SAMPLE_RATE * PCM16_BYTES_PER_SAMPLE // 1000]
    result = openai_client_function.send_realtime_duplex_audio_request(
        {
            "model": omni_server_function.model,
            "session_id": f"duplex-preempt-{uuid.uuid4().hex}",
            "ref_audio": _ref_audio_data_url(str(resolve_ref_audio())),
            "pcm16": pcm,
            "artifact_dir": str(tmp_path / "preempt"),
        }
    )
    observations = [
        _native_append_metrics(append)
        for event in result["events"]
        if (append := _native_append_result(event)) is not None
    ]
    assert any(
        item and isinstance((count := item.get("omni_num_preemptions", 0)), int) and count >= 1 for item in observations
    )


@hardware_test(res={"cuda": "H100"}, num_cards=1)
@pytest.mark.parametrize("omni_server_function", _input_fault_server("empty_encoder"), indirect=True)
def test_native_duplex_failed_input_isolated_from_concurrent_peer(
    omni_server_function,
    openai_client_function,
    tmp_path,
):
    pcm = read_pcm16_wav(validated_input_wav())[: 1400 * PCM16_SAMPLE_RATE * PCM16_BYTES_PER_SAMPLE // 1000]
    common = {
        "model": omni_server_function.model,
        "ref_audio": _ref_audio_data_url(str(resolve_ref_audio())),
        "pcm16": pcm,
    }
    bad = {
        **common,
        "session_id": f"duplex-input-failure-{uuid.uuid4().hex}",
        "expected_error": "native_duplex_prefill_failed",
        "artifact_dir": str(tmp_path / "failed"),
    }
    good = {**common, "session_id": f"duplex-input-peer-{uuid.uuid4().hex}", "artifact_dir": str(tmp_path / "peer")}
    with ThreadPoolExecutor(max_workers=2) as pool:
        failed = pool.submit(openai_client_function.send_realtime_duplex_audio_request, bad)
        peer = pool.submit(openai_client_function.send_realtime_duplex_audio_request, good)
        assert failed.result()["audio_bytes"] == 0
        assert peer.result()["audio_bytes"] > 0


_STAGE0_RECOVERY_OVERRIDES = {
    "0": {
        # Two independent EngineCore processes share the one GPU exposed by
        # this test.  Duplicate logical placement is intentional; the reduced
        # utilization leaves enough room for both Stage0 replicas and the
        # colocated Talker/Code2Wav stages on an H100/H200-class card.
        "num_replicas": 2,
        "devices": "0,0",
        "gpu_memory_utilization": 0.28,
    }
}
_REPLICA_RECOVERY_SERVER_PARAMS = [
    pytest.param(
        OmniServerParams(
            model=MODEL,
            stage_config_path=DEPLOY_CONFIG,
            use_stage_cli=False,
            server_args=[
                "--trust-remote-code",
                "--stage-overrides",
                json.dumps(_STAGE0_RECOVERY_OVERRIDES),
            ],
        ),
        id="three-stage-two-stage0-replicas",
    )
]
_LOST_REPLY_SERVER_PARAMS = [
    pytest.param(
        OmniServerParams(
            model=MODEL,
            stage_config_path=DEPLOY_CONFIG,
            use_stage_cli=False,
            server_args=[
                "--trust-remote-code",
                "--stage-overrides",
                json.dumps(_STAGE0_RECOVERY_OVERRIDES),
            ],
            env_dict={
                "PYTHONPATH": os.pathsep.join(
                    part for part in (str(_LOST_REPLY_SITECUSTOMIZE_DIR), _INHERITED_PYTHONPATH) if part
                )
            },
        ),
        id="three-stage-drop-second-committed-append-reply",
    )
]


def _native_control_result(
    event: dict[str, object],
    operation: str,
) -> dict[str, object] | None:
    nested = event.get("event") if event.get("type") == "duplex.runtime.control" else event
    if not isinstance(nested, dict) or nested.get("type") != "runtime.control":
        return None
    result = nested.get("result")
    if not isinstance(result, dict) or result.get("operation") != operation:
        return None
    return result


def _native_append_result(event: dict[str, object]) -> dict[str, object] | None:
    return _native_control_result(event, "append")


def _native_append_metrics(result: dict[str, object]) -> dict[str, object] | None:
    stage_results = result.get("stage_results")
    if not isinstance(stage_results, list):
        return None
    for stage_result in stage_results:
        if not isinstance(stage_result, dict):
            continue
        native_result = stage_result.get("result")
        if not isinstance(native_result, dict):
            continue
        append_metrics = native_result.get("append_metrics")
        if isinstance(append_metrics, dict):
            return append_metrics
    return None


def _native_append_request_id(result: dict[str, object]) -> str | None:
    stage_results = result.get("stage_results")
    if not isinstance(stage_results, list):
        return None
    for stage_result in stage_results:
        if not isinstance(stage_result, dict):
            continue
        native_result = stage_result.get("result")
        if isinstance(native_result, dict) and isinstance(native_result.get("request_id"), str):
            return native_result["request_id"]
    return None


def _native_append_replica_id(result: dict[str, object]) -> int | None:
    stage_results = result.get("stage_results")
    if not isinstance(stage_results, list):
        return None
    for stage_result in stage_results:
        if isinstance(stage_result, dict) and isinstance(stage_result.get("replica_id"), int):
            return stage_result["replica_id"]
    return None


def _scrape_metrics(url: str) -> str:
    parts = urlsplit(url)
    metrics_url = urlunsplit(("http", parts.netloc, "/metrics", "", ""))
    with urlopen(metrics_url, timeout=15) as response:  # noqa: S310 - loopback E2E server.
        return response.read().decode("utf-8")


def _metric_values(metrics: str, name: str, *required_labels: str) -> list[float]:
    values: list[float] = []
    for line in metrics.splitlines():
        if not line.startswith(name) or any(label not in line for label in required_labels):
            continue
        try:
            values.append(float(line.rsplit(" ", 1)[-1]))
        except ValueError:
            continue
    return values


async def _receive_until(
    ws,
    predicate: Callable[[dict[str, object]], bool],
    *,
    timeout_s: float,
) -> list[dict[str, object]]:
    events: list[dict[str, object]] = []

    async def receive() -> None:
        while True:
            raw = await ws.recv()
            if not isinstance(raw, str):
                continue
            event = json.loads(raw)
            if not isinstance(event, dict):
                continue
            events.append(event)
            if predicate(event):
                return

    await asyncio.wait_for(receive(), timeout=timeout_s)
    return events


async def _receive_for(ws, *, duration_s: float) -> list[dict[str, object]]:
    events: list[dict[str, object]] = []
    deadline = asyncio.get_running_loop().time() + duration_s
    while (remaining := deadline - asyncio.get_running_loop().time()) > 0:
        try:
            raw = await asyncio.wait_for(ws.recv(), timeout=remaining)
        except TimeoutError:
            break
        if not isinstance(raw, str):
            continue
        event = json.loads(raw)
        if isinstance(event, dict):
            events.append(event)
    return events


async def _run_stage0_replica_loss(
    *,
    url: str,
    model: str,
    ref_audio: Path,
    input_wav: Path,
    inject_fault: Callable[[], None],
) -> dict[str, object]:
    session_id = f"duplex-replica-loss-{uuid.uuid4().hex}"
    websocket_url = build_realtime_url(url, model, autostart=False, session_id=session_id)
    pcm16 = read_pcm16_wav(input_wav)
    unit_bytes = PCM16_SAMPLE_RATE * PCM16_BYTES_PER_SAMPLE
    chunk_bytes = unit_bytes // 5
    events: list[dict[str, object]] = []
    connection_closed = False
    failure_started_at = 0.0

    async with websockets.connect(websocket_url, max_size=64 * 1024 * 1024) as ws:
        await ws.send(
            json.dumps(
                {
                    "type": "session.update",
                    "session": {
                        "session_id": session_id,
                        "model": model,
                        "modalities": ["audio", "text"],
                        "ref_audio": _ref_audio_data_url(str(ref_audio)),
                        "extra_body": {
                            "auto_response": True,
                            "minicpmo45_native_duplex": True,
                            "emit_duplex_control_results": True,
                        },
                    },
                }
            )
        )
        events.extend(
            await _receive_until(
                ws,
                lambda event: event.get("type") == "session.updated",
                timeout_s=60,
            )
        )

        audio_end_ms = 0
        for offset in range(0, min(len(pcm16), unit_bytes * 6 // 5), chunk_bytes):
            chunk = pcm16[offset : offset + chunk_bytes]
            duration_ms = len(chunk) * 1000 // unit_bytes
            audio_end_ms += duration_ms
            await ws.send(
                json.dumps(
                    {
                        "type": "input_audio_buffer.append",
                        "audio": base64.b64encode(chunk).decode("ascii"),
                        "input_audio_format": "pcm16",
                        "sample_rate_hz": PCM16_SAMPLE_RATE,
                        "duration_ms": duration_ms,
                        "audio_end_ms": audio_end_ms,
                        "transcript": "duplex replica loss probe",
                    }
                )
            )

        events.extend(
            await _receive_until(
                ws,
                lambda event: _native_append_result(event) is not None,
                timeout_s=60,
            )
        )
        append_result = next(result for event in events if (result := _native_append_result(event)) is not None)
        stage_results = append_result.get("stage_results")
        assert isinstance(stage_results, list) and stage_results

        first_append = append_result
        inject_fault()
        failure_started_at = time.monotonic()
        try:
            previous_append_count = sum(_native_append_result(event) is not None for event in events)
            for offset in range(0, unit_bytes, chunk_bytes):
                chunk = pcm16[offset : offset + chunk_bytes]
                duration_ms = len(chunk) * 1000 // unit_bytes
                audio_end_ms += duration_ms
                await ws.send(
                    json.dumps(
                        {
                            "type": "input_audio_buffer.append",
                            "audio": base64.b64encode(chunk).decode("ascii"),
                            "input_audio_format": "pcm16",
                            "sample_rate_hz": PCM16_SAMPLE_RATE,
                            "duration_ms": duration_ms,
                            "audio_end_ms": audio_end_ms,
                            "transcript": "duplex replica recovery probe",
                        }
                    )
                )
            events.extend(
                await _receive_until(
                    ws,
                    lambda event: (
                        event.get("type") == "error"
                        or sum(_native_append_result(candidate) is not None for candidate in events + [event])
                        > previous_append_count
                    ),
                    timeout_s=60,
                )
            )
        except ConnectionClosed:
            connection_closed = True
        finally:
            metrics = await asyncio.to_thread(_scrape_metrics, url)
            try:
                await ws.send(json.dumps({"type": "session.close"}))
                events.extend(
                    await _receive_until(
                        ws,
                        lambda event: event.get("type") == "session.closed",
                        timeout_s=30,
                    )
                )
            except ConnectionClosed:
                connection_closed = True

    errors = [event for event in events if event.get("type") == "error"]
    return {
        "events": events,
        "errors": errors,
        "first_append": first_append,
        "append_results": [result for event in events if (result := _native_append_result(event)) is not None],
        "metrics": metrics,
        "connection_closed": connection_closed,
        "failure_latency_s": time.monotonic() - failure_started_at,
    }


async def _run_context_ceiling(
    *,
    url: str,
    model: str,
    ref_audio: Path,
    input_wav: Path,
) -> dict[str, object]:
    session_id = f"duplex-context-ceiling-{uuid.uuid4().hex}"
    websocket_url = build_realtime_url(url, model, autostart=False, session_id=session_id)
    pcm16 = read_pcm16_wav(input_wav)
    unit_bytes = PCM16_SAMPLE_RATE * PCM16_BYTES_PER_SAMPLE
    chunk_bytes = unit_bytes // 5
    probe_pcm = pcm16[: unit_bytes * 6 // 5]
    events: list[dict[str, object]] = []
    error: dict[str, object] | None = None

    async with websockets.connect(websocket_url, max_size=64 * 1024 * 1024) as ws:
        await ws.send(
            json.dumps(
                {
                    "type": "session.update",
                    "session": {
                        "session_id": session_id,
                        "model": model,
                        "modalities": ["audio", "text"],
                        "ref_audio": _ref_audio_data_url(str(ref_audio)),
                        "extra_body": {
                            "auto_response": True,
                            "minicpmo45_native_duplex": True,
                            "emit_duplex_control_results": True,
                        },
                    },
                }
            )
        )
        events.extend(
            await _receive_until(
                ws,
                lambda event: event.get("type") == "session.updated",
                timeout_s=60,
            )
        )

        audio_end_ms = 0
        initial_request_id: str | None = None
        rollover_seen = False
        for _ in range(64):
            for offset in range(0, len(probe_pcm), chunk_bytes):
                chunk = probe_pcm[offset : offset + chunk_bytes]
                duration_ms = len(chunk) * 1000 // unit_bytes
                audio_end_ms += duration_ms
                await ws.send(
                    json.dumps(
                        {
                            "type": "input_audio_buffer.append",
                            "audio": base64.b64encode(chunk).decode("ascii"),
                            "input_audio_format": "pcm16",
                            "sample_rate_hz": PCM16_SAMPLE_RATE,
                            "duration_ms": duration_ms,
                            "audio_end_ms": audio_end_ms,
                            "transcript": "duplex context ceiling probe",
                        }
                    )
                )

            batch_events = await _receive_until(
                ws,
                lambda event: event.get("type") == "error" or _native_append_result(event) is not None,
                timeout_s=60,
            )
            events.extend(batch_events)
            error = next((event for event in batch_events if event.get("type") == "error"), None)
            if error is not None:
                break
            batch_request_ids = [
                request_id
                for event in batch_events
                if (append_result := _native_append_result(event)) is not None
                if (request_id := _native_append_request_id(append_result)) is not None
            ]
            if initial_request_id is None and batch_request_ids:
                initial_request_id = batch_request_ids[0]
            if initial_request_id is not None and any(
                request_id != initial_request_id for request_id in batch_request_ids
            ):
                rollover_seen = True
                break

        assert rollover_seen, "context probe exhausted 64 append units before rollover"

        metrics = await asyncio.to_thread(_scrape_metrics, url)
        await ws.send(json.dumps({"type": "session.close"}))
        events.extend(
            await _receive_until(
                ws,
                lambda event: event.get("type") == "session.closed",
                timeout_s=30,
            )
        )

    return {"events": events, "error": error, "metrics": metrics}


async def _run_lost_append_reply_retry(
    *,
    url: str,
    model: str,
    ref_audio: Path,
    input_wav: Path,
    inject_fault: Callable[[], None] | None = None,
) -> list[dict[str, object]]:
    session_id = f"duplex-lost-reply-{uuid.uuid4().hex}"
    websocket_url = build_realtime_url(url, model, autostart=False, session_id=session_id)
    pcm16 = read_pcm16_wav(input_wav)
    unit_bytes = PCM16_SAMPLE_RATE * PCM16_BYTES_PER_SAMPLE
    chunk_bytes = unit_bytes // 5
    events: list[dict[str, object]] = []

    async with websockets.connect(websocket_url, max_size=64 * 1024 * 1024) as ws:
        await ws.send(
            json.dumps(
                {
                    "type": "session.update",
                    "session": {
                        "session_id": session_id,
                        "model": model,
                        "modalities": ["audio", "text"],
                        "ref_audio": _ref_audio_data_url(str(ref_audio)),
                        "extra_body": {
                            "auto_response": True,
                            "minicpmo45_native_duplex": True,
                            "emit_duplex_control_results": True,
                        },
                    },
                }
            )
        )
        events.extend(
            await _receive_until(
                ws,
                lambda event: event.get("type") == "session.updated",
                timeout_s=60,
            )
        )

        audio_end_ms = 0
        for _ in range(2):
            for offset in range(0, unit_bytes, chunk_bytes):
                chunk = pcm16[offset : offset + chunk_bytes]
                duration_ms = len(chunk) * 1000 // unit_bytes
                audio_end_ms += duration_ms
                await ws.send(
                    json.dumps(
                        {
                            "type": "input_audio_buffer.append",
                            "audio": base64.b64encode(chunk).decode("ascii"),
                            "input_audio_format": "pcm16",
                            "sample_rate_hz": PCM16_SAMPLE_RATE,
                            "duration_ms": duration_ms,
                            "audio_end_ms": audio_end_ms,
                            "transcript": "duplex lost reply retry probe",
                        }
                    )
                )
            previous_append_count = sum(_native_append_result(event) is not None for event in events)
            events.extend(
                await _receive_until(
                    ws,
                    lambda event: (
                        sum(_native_append_result(candidate) is not None for candidate in events + [event])
                        > previous_append_count
                    ),
                    timeout_s=60,
                )
            )
        if inject_fault is not None:
            inject_fault()
            previous_append_count = sum(_native_append_result(event) is not None for event in events)
            for offset in range(0, unit_bytes, chunk_bytes):
                chunk = pcm16[offset : offset + chunk_bytes]
                duration_ms = len(chunk) * 1000 // unit_bytes
                audio_end_ms += duration_ms
                await ws.send(
                    json.dumps(
                        {
                            "type": "input_audio_buffer.append",
                            "audio": base64.b64encode(chunk).decode("ascii"),
                            "input_audio_format": "pcm16",
                            "sample_rate_hz": PCM16_SAMPLE_RATE,
                            "duration_ms": duration_ms,
                            "audio_end_ms": audio_end_ms,
                            "transcript": "duplex lost reply then recovery probe",
                        }
                    )
                )
            events.extend(
                await _receive_until(
                    ws,
                    lambda event: (
                        event.get("type") == "error"
                        or sum(_native_append_result(candidate) is not None for candidate in events + [event])
                        > previous_append_count
                    ),
                    timeout_s=60,
                )
            )
        await ws.send(json.dumps({"type": "session.close"}))
        events.extend(
            await _receive_until(
                ws,
                lambda event: event.get("type") == "session.closed",
                timeout_s=30,
            )
        )

    return events


async def _run_pending_append_terminal_case(
    *,
    url: str,
    model: str,
    ref_audio: Path,
    input_wav: Path,
    terminal_event: str,
    pause_stage0: Callable[[], None],
    resume_stage0: Callable[[], None],
) -> dict[str, object]:
    session_id = f"duplex-pending-{terminal_event.replace('.', '-')}-{uuid.uuid4().hex}"
    websocket_url = build_realtime_url(url, model, autostart=False, session_id=session_id)
    pcm16 = read_pcm16_wav(input_wav)
    unit_bytes = PCM16_SAMPLE_RATE * PCM16_BYTES_PER_SAMPLE
    chunk_bytes = unit_bytes // 5
    audio_unit = pcm16[:unit_bytes]
    events: list[dict[str, object]] = []
    pending_window_events: list[dict[str, object]] = []
    terminal_started_at = 0.0
    terminal_latency_s = 0.0

    async with websockets.connect(websocket_url, max_size=64 * 1024 * 1024) as ws:
        await ws.send(
            json.dumps(
                {
                    "type": "session.update",
                    "session": {
                        "session_id": session_id,
                        "model": model,
                        "modalities": ["audio", "text"],
                        "ref_audio": _ref_audio_data_url(str(ref_audio)),
                        "extra_body": {
                            "auto_response": True,
                            "minicpmo45_native_duplex": True,
                            "emit_duplex_control_results": True,
                        },
                    },
                }
            )
        )
        events.extend(
            await _receive_until(
                ws,
                lambda event: event.get("type") == "session.updated",
                timeout_s=60,
            )
        )

        audio_end_ms = 0

        async def send_audio_unit() -> None:
            nonlocal audio_end_ms
            for offset in range(0, len(audio_unit), chunk_bytes):
                chunk = audio_unit[offset : offset + chunk_bytes]
                duration_ms = len(chunk) * 1000 // unit_bytes
                audio_end_ms += duration_ms
                await ws.send(
                    json.dumps(
                        {
                            "type": "input_audio_buffer.append",
                            "audio": base64.b64encode(chunk).decode("ascii"),
                            "input_audio_format": "pcm16",
                            "sample_rate_hz": PCM16_SAMPLE_RATE,
                            "duration_ms": duration_ms,
                            "audio_end_ms": audio_end_ms,
                            "transcript": "duplex pending append terminal probe",
                        }
                    )
                )

        await send_audio_unit()
        events.extend(
            await _receive_until(
                ws,
                lambda event: _native_append_result(event) is not None,
                timeout_s=60,
            )
        )

        pause_stage0()
        try:
            await send_audio_unit()
            pending_window_events = await _receive_for(ws, duration_s=1.0)
            events.extend(pending_window_events)
            terminal_started_at = time.monotonic()
            await ws.send(json.dumps({"type": terminal_event, "reason": "pending_append_probe"}))
            terminal_operation = "close" if terminal_event == "session.close" else "signal"
            events.extend(
                await _receive_until(
                    ws,
                    lambda event: _native_control_result(event, terminal_operation) is not None,
                    timeout_s=15,
                )
            )
            terminal_latency_s = time.monotonic() - terminal_started_at
        finally:
            resume_stage0()

        if terminal_event != "session.close":
            await ws.send(json.dumps({"type": "session.close"}))
        if not any(event.get("type") == "session.closed" for event in events):
            events.extend(
                await _receive_until(
                    ws,
                    lambda event: event.get("type") == "session.closed",
                    timeout_s=30,
                )
            )

    return {
        "events": events,
        "pending_window_events": pending_window_events,
        "terminal_latency_s": terminal_latency_s,
        "terminal_operation": "close" if terminal_event == "session.close" else "signal",
    }


@pytest.mark.skipif(os.name == "nt", reason="process-kill injection helper is POSIX-only")
@hardware_test(res={"cuda": "H100"}, num_cards=1)
@pytest.mark.parametrize("omni_server_function", _REPLICA_RECOVERY_SERVER_PARAMS, indirect=True)
def test_native_duplex_committed_kv_recovers_on_surviving_replica(
    omni_server_function,
) -> None:
    injector = make_process_kill_fault_injector(
        grep_patterns=(
            "VLLM::StageEngineCoreProc_stage0_replica0",
            "StageEngineCoreProc_stage0_replica0",
        ),
        signal_name="SIGKILL",
        limit=1,
        post_kill_wait_seconds=0.5,
    )
    result = asyncio.run(
        _run_stage0_replica_loss(
            url=realtime_url(omni_server_function),
            model=omni_server_function.model,
            ref_audio=resolve_ref_audio(),
            input_wav=validated_input_wav(),
            inject_fault=lambda: injector(omni_server_function),
        )
    )

    assert isinstance(result["failure_latency_s"], int | float)
    assert isinstance(result["metrics"], str)
    assert isinstance(result["events"], list)
    assert result["failure_latency_s"] < 91
    assert result["errors"] == []
    assert result["connection_closed"] is False
    append_results = result["append_results"]
    assert isinstance(append_results, list)
    assert len(append_results) >= 2
    first_append, recovered_append = append_results[0], append_results[-1]
    first_request_id = _native_append_request_id(first_append)
    recovered_request_id = _native_append_request_id(recovered_append)
    assert _native_append_replica_id(first_append) == 0
    assert _native_append_replica_id(recovered_append) == 1
    assert isinstance(first_request_id, str) and isinstance(recovered_request_id, str)
    assert first_request_id != recovered_request_id
    assert "stage0g1" in recovered_request_id
    assert (
        _metric_values(
            result["metrics"],
            "vllm_omni:duplex_kv_rebuilds_total",
            'reason="replica_loss"',
            'outcome="success"',
        )[-1]
        >= 1
    )
    assert any(event.get("type") == "session.closed" for event in result["events"])


@hardware_test(res={"cuda": "H100"}, num_cards=1)
@pytest.mark.parametrize("omni_server_function", _CONTEXT_LIMIT_SERVER_PARAMS, indirect=True)
def test_native_duplex_context_rollover_bounds_kv_growth_and_continues(
    omni_server_function,
) -> None:
    result = asyncio.run(
        _run_context_ceiling(
            url=realtime_url(omni_server_function),
            model=omni_server_function.model,
            ref_audio=resolve_ref_audio(),
            input_wav=validated_input_wav(),
        )
    )

    events = result["events"]
    assert isinstance(events, list)
    append_observations = [
        (_native_append_request_id(append_result), metrics)
        for event in events
        if (append_result := _native_append_result(event)) is not None
        if (metrics := _native_append_metrics(append_result)) is not None
    ]
    assert append_observations, "the context probe must establish resident KV"
    assert result["error"] is None, json.dumps(result["error"], ensure_ascii=False)
    request_ids = [request_id for request_id, _ in append_observations]
    assert all(isinstance(request_id, str) for request_id in request_ids)
    assert len(set(request_ids)) >= 2, "the reduced Stage0 context limit did not trigger a new KV generation"
    assert any("stage0g" in request_id for request_id in request_ids if request_id is not None)
    for request_id in dict.fromkeys(request_ids):
        generation_tokens: list[int] = []
        for observed_request_id, metrics in append_observations:
            if observed_request_id == request_id:
                tokens = metrics["omni_context_tokens"]
                assert isinstance(tokens, int)
                generation_tokens.append(tokens)
        assert generation_tokens == sorted(generation_tokens)
        assert generation_tokens[-1] <= _CONTEXT_LIMIT
    assert all(metrics["omni_context_limit"] == _CONTEXT_LIMIT for _, metrics in append_observations)

    scraped_metrics = result["metrics"]
    assert isinstance(scraped_metrics, str)
    assert (
        _metric_values(
            scraped_metrics,
            "vllm_omni:duplex_kv_rebuilds_total",
            'reason="context_rollover"',
            'outcome="success"',
        )[-1]
        >= 1
    )
    assert _metric_values(scraped_metrics, "vllm_omni:duplex_replay_journal_tokens")[-1] > 0
    assert _metric_values(scraped_metrics, "vllm_omni:duplex_replay_journal_bytes")[-1] > 0
    assert _metric_values(scraped_metrics, "vllm_omni:duplex_resource_generation")[-1] >= 1
    assert any(event.get("type") == "session.closed" for event in events)


@hardware_test(res={"cuda": "H100"}, num_cards=1)
@pytest.mark.parametrize("omni_server_function", _LOST_REPLY_SERVER_PARAMS, indirect=True)
def test_native_duplex_committed_append_lost_reply_retries_exactly_once(
    omni_server_function,
) -> None:
    injector = make_process_kill_fault_injector(
        grep_patterns=(
            "VLLM::StageEngineCoreProc_stage0_replica0",
            "StageEngineCoreProc_stage0_replica0",
        ),
        signal_name="SIGKILL",
        limit=1,
        post_kill_wait_seconds=1.0,
    )
    events = asyncio.run(
        _run_lost_append_reply_retry(
            url=realtime_url(omni_server_function),
            model=omni_server_function.model,
            ref_audio=resolve_ref_audio(),
            input_wav=validated_input_wav(),
            inject_fault=lambda: injector(omni_server_function),
        )
    )

    append_results = [result for event in events if (result := _native_append_result(event)) is not None]
    assert len(append_results) == 3
    metrics = [_native_append_metrics(result) for result in append_results]
    assert all(isinstance(item, dict) for item in metrics)
    first_metrics, retry_metrics, recovered_metrics = metrics
    assert first_metrics is not None and retry_metrics is not None and recovered_metrics is not None
    assert isinstance(first_metrics["omni_context_tokens"], int)
    assert isinstance(retry_metrics["omni_context_tokens"], int)
    # Initial admission reports context metrics before any scheduler receipt
    # exists, so older matching vLLM builds legitimately omit this optional
    # flag. Only the retried scheduler append must state deduplication.
    assert first_metrics.get("deduplicated", False) is False
    assert retry_metrics["deduplicated"] is True
    assert retry_metrics["omni_append_receipt_count"] == 1
    assert retry_metrics["omni_context_tokens"] > first_metrics["omni_context_tokens"]
    assert recovered_metrics.get("deduplicated", False) is False
    assert _native_append_replica_id(append_results[0]) == 0
    assert _native_append_replica_id(append_results[-1]) == 1
    first_request_id = _native_append_request_id(append_results[0])
    recovered_request_id = _native_append_request_id(append_results[-1])
    assert isinstance(first_request_id, str) and isinstance(recovered_request_id, str)
    assert first_request_id != recovered_request_id
    assert "stage0g1" in recovered_request_id
    assert not any(event.get("type") == "error" for event in events)
    assert any(event.get("type") == "session.closed" for event in events)


@pytest.mark.skipif(os.name == "nt", reason="process-stop injection helper is POSIX-only")
@hardware_test(res={"cuda": "H100"}, num_cards=1)
@pytest.mark.parametrize("omni_server_function", SERVER_PARAMS, indirect=True)
def test_native_duplex_pending_append_cancel_and_close_preempt_bounded(
    omni_server_function,
) -> None:
    process_patterns = (
        "VLLM::StageEngineCoreProc_stage0_replica0",
        "StageEngineCoreProc_stage0_replica0",
    )
    pause = make_process_kill_fault_injector(
        grep_patterns=process_patterns,
        signal_name="SIGSTOP",
        limit=1,
        post_kill_wait_seconds=0.2,
    )
    resume = make_process_kill_fault_injector(
        grep_patterns=process_patterns,
        signal_name="SIGCONT",
        limit=1,
        post_kill_wait_seconds=0.2,
    )

    async def run_cases() -> list[dict[str, object]]:
        results: list[dict[str, object]] = []
        for terminal_event in ("input.cancel", "session.close"):
            results.append(
                await _run_pending_append_terminal_case(
                    url=realtime_url(omni_server_function),
                    model=omni_server_function.model,
                    ref_audio=resolve_ref_audio(),
                    input_wav=validated_input_wav(),
                    terminal_event=terminal_event,
                    pause_stage0=lambda: pause(omni_server_function),
                    resume_stage0=lambda: resume(omni_server_function),
                )
            )
        return results

    for result in asyncio.run(run_cases()):
        assert isinstance(result["pending_window_events"], list)
        assert isinstance(result["events"], list)
        assert isinstance(result["terminal_latency_s"], int | float)
        assert not any(_native_append_result(event) is not None for event in result["pending_window_events"]), (
            "the append unexpectedly completed while its EngineCore was stopped"
        )
        assert result["terminal_latency_s"] < 15
        terminal_results = [
            control_result
            for event in result["events"]
            if (
                control_result := _native_control_result(
                    event,
                    str(result["terminal_operation"]),
                )
            )
            is not None
        ]
        assert terminal_results and terminal_results[-1]["ok"] is True
        assert any(event.get("type") == "session.closed" for event in result["events"])
