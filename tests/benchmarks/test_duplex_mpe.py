# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""L1 CPU unit and loopback WebSocket/HTTP tests for benchmarks/duplex_mpe."""

from __future__ import annotations

import asyncio
import base64
import io
import json
import sys
from collections.abc import AsyncIterator
from dataclasses import dataclass
from pathlib import Path
from types import TracebackType

import httpx
import numpy as np
import pytest
import soundfile
import torch
import websockets
from numpy.typing import NDArray
from pydantic import JsonValue

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarks.duplex_mpe.dataset import (  # noqa: E402
    DUPLEX_MPE_DUTY_PREAMBLE_RELATIVE_PATH,
    compute_file_sha256,
    load_duplex_mpe_scenarios,
    resolve_dataset_audio_path,
    validate_duplex_mpe_dataset,
    write_json_atomic,
)
from benchmarks.duplex_mpe.evaluate import (  # noqa: E402
    DuplexMpeEvalConfig,
    run_duplex_mpe_evaluation,
    run_generate_phase,
    sanitize_launch_command,
    verify_selected_human_audio_inventory,
)
from benchmarks.duplex_mpe.evaluate import main as evaluate_main  # noqa: E402
from benchmarks.duplex_mpe.metrics import (  # noqa: E402
    count_n4_windows,
    format_markdown_score_table,
    score_scenario_reports,
    validate_scenario_reports,
)
from benchmarks.duplex_mpe.prompts import (  # noqa: E402
    compute_prompt_sha256,
    render_judge_prompt,
)
from benchmarks.duplex_mpe.protocol import (  # noqa: E402
    TARGET_CONTINUATION_OUTCOME,
    TARGET_HIT_OUTCOME,
    TARGET_MISS_OUTCOME,
    TARGET_OVERLONG_OUTCOME,
    ClockedAudioAdapter,
    ScenarioTurnInput,
    load_reference_seeds,
    run_scenario_schedule,
)
from benchmarks.duplex_mpe.realtime_session import (  # noqa: E402
    NEUTRAL_NEMOTRON_INSTRUCTIONS,
    PlaybackTimelineBuffer,
    ProtocolSessionError,
    RealtimeSessionOptions,
    StatefulWireResampler,
    build_duplex_session_payload,
    build_openai_turn_session_payload,
    compute_effective_scenario_timeout_seconds,
    decode_output_audio_delta_to_float32,
    encode_audio_file_as_data_url,
    encode_float32_to_pcm16_base64,
    infer_model_audio_defaults,
    negotiate_realtime_endpoint,
    run_realtime_scenario,
)
from benchmarks.duplex_mpe.semantic import (  # noqa: E402
    JudgeEndpointConfig,
    JudgeTaskDecodingConfig,
    build_semantic_tasks,
    judge_tasks_via_http,
    parse_judge_verdict_json,
    transcribe_tasks_via_http,
    validate_transcribed_against_tasks,
)
from benchmarks.duplex_mpe.vad import (  # noqa: E402
    SILERO_UNINSTALLED_SENTINEL,
    SileroActivityConfig,
    SileroActivityDetector,
    SileroVadModel,
    silero_package_version,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


class EnergyThresholdVadStub:
    """Deterministic fake SileroVadModel keyed on RMS amplitude."""

    def __init__(self, rms_threshold: float = 0.05) -> None:
        self.rms_threshold: float = rms_threshold
        self.resets: int = 0

    def reset_states(self) -> None:
        self.resets += 1

    def __call__(self, audio_tensor: torch.Tensor, sample_rate_hz: int) -> torch.Tensor:
        del sample_rate_hz
        rms = float(torch.sqrt(torch.mean(audio_tensor.float() ** 2)).item())
        probability = 0.85 if rms >= self.rms_threshold else 0.05
        return torch.tensor(probability, dtype=torch.float32)


@dataclass
class ScheduledReplyAdapter:
    """Synthetic sample-preserving adapter emitting voiced tone intervals on schedule."""

    input_sample_rate_hz: int
    output_sample_rate_hz: int
    voiced_intervals_seconds: list[tuple[float, float]]
    cursor_samples: int = 0

    def feed(self, input_samples: NDArray[np.float32]) -> NDArray[np.float32]:
        expected_end_samples = int(
            round(
                (self.cursor_samples / self.output_sample_rate_hz + len(input_samples) / self.input_sample_rate_hz)
                * self.output_sample_rate_hz
            )
        )
        chunk_len = max(0, expected_end_samples - self.cursor_samples)
        output = np.zeros(chunk_len, dtype=np.float32)
        if chunk_len > 0:
            sample_indices = np.arange(self.cursor_samples, expected_end_samples, dtype=np.float64)
            times_s = sample_indices / self.output_sample_rate_hz
            voiced_mask = np.zeros(chunk_len, dtype=bool)
            for start_s, end_s in self.voiced_intervals_seconds:
                voiced_mask |= (times_s >= start_s) & (times_s < end_s)
            if np.any(voiced_mask):
                output[voiced_mask] = (0.3 * np.sin(2.0 * np.pi * 440.0 * times_s[voiced_mask])).astype(np.float32)
        self.cursor_samples = expected_end_samples
        return output


def write_synthetic_wav(path: Path, duration_seconds: float, sample_rate_hz: int = 16000) -> dict[str, JsonValue]:
    path.parent.mkdir(parents=True, exist_ok=True)
    frames = max(1, int(round(duration_seconds * sample_rate_hz)))
    t_axis = np.arange(frames, dtype=np.float64) / sample_rate_hz
    samples = (0.25 * np.sin(2.0 * np.pi * 330.0 * t_axis)).astype(np.float32)
    soundfile.write(str(path), samples, sample_rate_hz, format="WAV", subtype="PCM_16")
    return {
        "path": "",
        "sha256": compute_file_sha256(path),
        "bytes": path.stat().st_size,
        "sample_rate": sample_rate_hz,
        "frames": frames,
        "channels": 1,
        "duration_s": round(frames / sample_rate_hz, 4),
    }


def create_synthetic_duplex_mpe_dataset(dataset_root: Path) -> Path:
    """Create a self-contained paired explicit+implicit Duplex-MPE dataset fixture."""
    seeds = load_reference_seeds()
    explicit_ids = sorted(sid for sid in seeds if sid.endswith("_explicit"))[:2]
    assert len(explicit_ids) == 2

    inventory: list[dict[str, JsonValue]] = []
    preamble_file = dataset_root / DUPLEX_MPE_DUTY_PREAMBLE_RELATIVE_PATH
    preamble_meta = write_synthetic_wav(preamble_file, duration_seconds=0.16)
    preamble_meta["path"] = DUPLEX_MPE_DUTY_PREAMBLE_RELATIVE_PATH
    inventory.append(preamble_meta)

    explicit_rows: list[dict[str, JsonValue]] = []
    implicit_rows: list[dict[str, JsonValue]] = []
    for pair_offset, exp_id in enumerate(explicit_ids):
        base_prefix = exp_id.removesuffix("_explicit")
        imp_id = f"{base_prefix}_implicit"
        assert imp_id in seeds
        pair_id = f"pair_{pair_offset:04d}"
        for condition, scenario_id in (("explicit", exp_id), ("implicit", imp_id)):
            turns: list[JsonValue] = []
            turn_specs = [
                (0, "N1", "Alice", "Flight 42 departs from Gate B12."),
                (1, "N4_Q", "Bob", "Did anyone bring the boarding passes?"),
                (2, "N4_R", "Alice", "Yes, I have both passes right here."),
                (
                    3,
                    "T",
                    "Bob",
                    "Aria, which gate does Flight 42 depart from?"
                    if condition == "explicit"
                    else "Wait, which gate does Flight 42 depart from?",
                ),
            ]
            for turn_idx, event_type, speaker, text in turn_specs:
                rel_audio = f"audio/{scenario_id}_t{turn_idx}.wav"
                clip_meta = write_synthetic_wav(dataset_root / rel_audio, duration_seconds=0.16)
                clip_meta["path"] = rel_audio
                inventory.append(clip_meta)
                turns.append(
                    {
                        "turn_index": turn_idx,
                        "event_type": event_type,
                        "speaker": speaker,
                        "text": text,
                        "audio_path": rel_audio,
                    }
                )
            row: dict[str, JsonValue] = {
                "scenario_id": scenario_id,
                "pair_id": pair_id,
                "addressing": condition,
                "trigger_index": 3,
                "supporting_indices": [0],
                "gold_answer": "Gate B12",
                "turns": turns,
            }
            if condition == "explicit":
                explicit_rows.append(row)
            else:
                implicit_rows.append(row)

    write_json_atomic(dataset_root / "dataset.json", {"name": "synthetic_duplex_mpe", "pairs": 2})
    write_json_atomic(dataset_root / "manifests" / "explicit.json", explicit_rows)
    write_json_atomic(dataset_root / "manifests" / "implicit.json", implicit_rows)
    write_json_atomic(dataset_root / "audio_inventory.json", inventory)
    return dataset_root


def test_dataset_validation_and_path_traversal_guards(tmp_path: Path) -> None:
    dataset_root = create_synthetic_duplex_mpe_dataset(tmp_path / "dataset")
    summary = validate_duplex_mpe_dataset(dataset_root, mini=True, verify_audio_checksums=True)
    assert summary["scenarios"] == 4
    assert summary["pairs"] == 2
    assert summary["all_audio_checksums_verified"] is True

    scenarios = load_duplex_mpe_scenarios(dataset_root, addressing="all", mini=True, max_scenarios=2)
    assert len(scenarios) == 2

    with pytest.raises(ValueError, match="Unsafe"):
        resolve_dataset_audio_path(dataset_root, "../escape.wav")
    with pytest.raises(ValueError, match="Unsafe"):
        resolve_dataset_audio_path(dataset_root, "audio/../../escape.wav")


@pytest.mark.parametrize("sample_rate_hz", [16000, 22050, 24000, 48000])
def test_silero_detector_streaming_hysteresis_across_sample_rates(sample_rate_hz: int) -> None:
    stub = EnergyThresholdVadStub()
    detector = SileroActivityDetector(
        vad_config=SileroActivityConfig(),
        sample_rate_hz=sample_rate_hz,
        vad_model=stub,
    )
    silence_half_sec = np.zeros(sample_rate_hz // 2, dtype=np.float32)
    t_axis = np.arange(sample_rate_hz // 2, dtype=np.float64) / sample_rate_hz
    tone_half_sec = (0.35 * np.sin(2.0 * np.pi * 400.0 * t_axis)).astype(np.float32)
    silence_tail = np.zeros(int(sample_rate_hz * 1.2), dtype=np.float32)

    stream = np.concatenate([silence_half_sec, tone_half_sec, silence_tail])
    step = max(1, sample_rate_hz // 12)
    for offset in range(0, len(stream), step):
        detector.push(stream[offset : offset + step])
    detector.finalize()

    assert len(detector.episodes) == 1
    ep_start, ep_end = detector.episodes[0]
    assert 0.45 <= ep_start <= 0.60
    assert 0.92 <= ep_end <= 1.08
    assert detector.is_endpoint_confirmed(800.0) is True


def test_reactive_protocol_target_outcomes_and_clock_guard() -> None:
    stub = EnergyThresholdVadStub()
    turns = [
        ScenarioTurnInput(
            turn_index=0,
            event_type="N1",
            speaker="Alice",
            waveform_samples=np.zeros(3200, dtype=np.float32),
        ),
        ScenarioTurnInput(
            turn_index=1,
            event_type="T",
            speaker="Bob",
            waveform_samples=np.zeros(3200, dtype=np.float32),
        ),
    ]
    hit_adapter = ClockedAudioAdapter(
        ScheduledReplyAdapter(
            input_sample_rate_hz=16000,
            output_sample_rate_hz=24000,
            voiced_intervals_seconds=[(1.2, 1.8)],
        )
    )
    hit_result = run_scenario_schedule(
        hit_adapter,
        turns,
        vad_model=stub,
        onset_deadline_seconds=2.0,
        maximum_response_seconds=4.0,
        target_endpoint_hangover_milliseconds=500.0,
        collect_audio=True,
    )
    assert hit_result.target_outcome == TARGET_HIT_OUTCOME
    assert hit_result.target_onset_latency_seconds is not None
    assert 0.2 <= hit_result.target_onset_latency_seconds <= 0.6

    miss_adapter = ClockedAudioAdapter(
        ScheduledReplyAdapter(
            input_sample_rate_hz=16000,
            output_sample_rate_hz=24000,
            voiced_intervals_seconds=[],
        )
    )
    miss_result = run_scenario_schedule(
        miss_adapter,
        turns,
        vad_model=stub,
        onset_deadline_seconds=1.0,
        maximum_response_seconds=3.0,
        target_endpoint_hangover_milliseconds=500.0,
    )
    assert miss_result.target_outcome == TARGET_MISS_OUTCOME

    cont_adapter = ClockedAudioAdapter(
        ScheduledReplyAdapter(
            input_sample_rate_hz=16000,
            output_sample_rate_hz=24000,
            voiced_intervals_seconds=[(0.6, 1.3)],
        )
    )
    cont_result = run_scenario_schedule(
        cont_adapter,
        turns,
        vad_model=stub,
        onset_deadline_seconds=1.5,
        maximum_response_seconds=3.0,
        target_endpoint_hangover_milliseconds=400.0,
    )
    assert cont_result.target_outcome == TARGET_CONTINUATION_OUTCOME

    overlong_adapter = ClockedAudioAdapter(
        ScheduledReplyAdapter(
            input_sample_rate_hz=16000,
            output_sample_rate_hz=24000,
            voiced_intervals_seconds=[(1.0, 10.0)],
        )
    )
    overlong_result = run_scenario_schedule(
        overlong_adapter,
        turns,
        vad_model=stub,
        onset_deadline_seconds=1.5,
        maximum_response_seconds=1.2,
        target_endpoint_hangover_milliseconds=400.0,
    )
    assert overlong_result.target_outcome == TARGET_OVERLONG_OUTCOME
    assert overlong_result.target_forced_floor_return is True


def test_playback_timeline_buffer_acks_and_has_audio_gated_truncation() -> None:
    buffer = PlaybackTimelineBuffer(sample_rate_hz=24000)
    # Cancelling an item that never produced audio must NOT emit a truncate record.
    empty_truncations = buffer.cancel_unplayed(
        0.5,
        target_response_id="resp_empty",
        target_item_id="item_resp_empty",
    )
    assert empty_truncations == []

    # Append 1 second (24,000 samples) starting at t=0.0s, play 0.5s (12,000 samples), then barge-in.
    samples = np.full(24000, 0.2, dtype=np.float32)
    buffer.append_audio(0.0, samples, response_id="resp_01", item_id="item_resp_01")
    rendered, acks = buffer.render_interval(0, 12000)
    assert len(rendered) == 12000
    assert len(acks) == 1
    assert acks[0].response_id == "resp_01"
    assert acks[0].item_id == "item_resp_01"
    assert acks[0].played_milliseconds == 500

    truncations = buffer.cancel_unplayed(0.5, target_response_id="resp_01", target_item_id="item_resp_01")
    assert len(truncations) == 1
    assert truncations[0].item_id == "item_resp_01"
    assert truncations[0].audio_end_milliseconds == 500
    assert buffer.cancelled_unplayed_samples == 12000


def test_n4_yielding_and_scored_vs_probe_metrics(tmp_path: Path) -> None:
    dataset_root = create_synthetic_duplex_mpe_dataset(tmp_path / "dataset")
    scenarios = load_duplex_mpe_scenarios(dataset_root, addressing="all")
    exp_scenario = scenarios[0]
    scenarios_by_id = {exp_scenario.scenario_id: exp_scenario}

    # N4_Q win: [1.0, 5.0] (turn [1.0, 2.0] + 3s pause [2.0, 5.0]).
    # Assistant starts at 2.5s (inside 3s pause), crosses N4_R turn_start=5.0s, yields at 6.2s (< 5.0 + 2.0 = 7.0s).
    report: dict[str, JsonValue] = {
        "protocol_id": "mpeval.unified.v3",
        "total_duration_s": 12.0,
        "assistant_audio_sha256": "abc123",
        "aed_config": SileroActivityConfig().to_report_dict(custom_vad_injected=True),
        "assistant_speech_segments": [[0.2, 0.5], [2.5, 6.2], [9.2, 10.2]],
        "T": {
            "outcome": "T_HIT_5S",
            "t_end_s": 9.0,
            "onset_s": 9.2,
            "onset_latency_s": 0.2,
            "endpoint_s": 10.2,
            "response_duration_s": 1.0,
            "forced_floor_return": False,
            "speaking_at_t_end": False,
        },
        "rows": [
            {
                "turn_index": 0,
                "event_type": "N1",
                "speaker": "Alice",
                "turn_start": 0.0,
                "turn_end": 0.5,
                "win_start": 0.0,
                "win_end": 1.0,
            },
            {
                "turn_index": 1,
                "event_type": "N4_Q",
                "speaker": "Bob",
                "turn_start": 1.0,
                "turn_end": 2.0,
                "win_start": 1.0,
                "win_end": 5.0,
            },
            {
                "turn_index": 2,
                "event_type": "N4_R",
                "speaker": "Alice",
                "turn_start": 5.0,
                "turn_end": 7.5,
                "win_start": 5.0,
                "win_end": 8.0,
            },
            {
                "turn_index": 3,
                "event_type": "T",
                "speaker": "Bob",
                "turn_start": 8.0,
                "turn_end": 9.0,
                "win_start": 8.0,
                "win_end": 12.0,
            },
        ],
    }
    validate_scenario_reports({exp_scenario.scenario_id: report}, scenarios_by_id)
    n4_counts = count_n4_windows(report)
    assert n4_counts.windows == 1
    assert n4_counts.window_hits == 1
    assert n4_counts.eligible == 1
    assert n4_counts.yielded == 1

    verdicts: list[dict[str, JsonValue]] = [
        {
            "scenario_id": exp_scenario.scenario_id,
            "turn_index": 0,
            "kind": "silence",
            "label": "BACKCHANNEL",
            "assistant_audio_sha256": "abc123",
        },
        {
            "scenario_id": exp_scenario.scenario_id,
            "turn_index": 3,
            "kind": "answer",
            "correct": True,
            "assistant_audio_sha256": "abc123",
        },
    ]
    scored = score_scenario_reports(
        {exp_scenario.scenario_id: report},
        scenarios_by_id,
        verdicts,
        probe_only=False,
    )
    pooled = scored["overall_pooled"]
    assert pooled["evaluation_mode"] == "scored_full_duplex"
    assert isinstance(pooled["fresh_onset_response_rate"], dict)
    assert pooled["fresh_onset_response_rate"]["value"] == 1.0
    assert isinstance(pooled["answering_window_yield"], dict)
    assert pooled["answering_window_yield"]["value"] == 1.0

    probe_scored = score_scenario_reports(
        {exp_scenario.scenario_id: report},
        scenarios_by_id,
        verdicts,
        probe_only=True,
    )
    probe_pooled = probe_scored["overall_pooled"]
    assert probe_pooled["evaluation_mode"] == "unscored_turn_based_probe"
    assert probe_pooled["fresh_onset_response_rate"] is None
    assert probe_pooled["answering_window_yield"] is None
    md_table = format_markdown_score_table(
        probe_scored,
        interaction_protocol="turn_based",
        turn_detection={"type": "server_vad", "silence_duration_ms": 500},
    )
    assert "n/a (probe)" in md_table


@pytest.mark.parametrize(
    ("raw_cmd", "expected_sanitized"),
    [
        (
            "CUDA_VISIBLE_DEVICES=0 vllm serve openbmb/MiniCPM-o-4_5 --omni "
            "--deploy-config vllm_omni/deploy/minicpmo_4_5.yaml --trust-remote-code --port 8091",
            "CUDA_VISIBLE_DEVICES=0 vllm serve openbmb/MiniCPM-o-4_5 --omni "
            "--deploy-config vllm_omni/deploy/minicpmo_4_5.yaml --trust-remote-code --port 8091",
        ),
        (
            "HF_TOKEN=hf_secret12345678 vllm-omni serve nvidia/personaplex-7b-v1 --omni --api-key sk-secret12345678",
            "'HF_TOKEN=<REDACTED>' vllm-omni serve nvidia/personaplex-7b-v1 --omni --api-key '<REDACTED>'",
        ),
    ],
)
def test_launch_command_sanitization_accepts_vllm_serve_and_redacts_secrets(
    raw_cmd: str,
    expected_sanitized: str,
) -> None:
    assert sanitize_launch_command(raw_cmd) == expected_sanitized
    with pytest.raises(ValueError):
        sanitize_launch_command("vllm serve openbmb/MiniCPM-o-4_5 && cat /etc/passwd")


def test_long_scenario_timeout_scaling() -> None:
    effective = compute_effective_scenario_timeout_seconds(
        configured_timeout_seconds=120.0,
        maximum_schedule_duration_seconds=292.4,
    )
    assert effective == pytest.approx(352.4)


async def test_loopback_native_duplex_session_with_admission_retry_and_playback_ack(tmp_path: Path) -> None:
    dataset_root = create_synthetic_duplex_mpe_dataset(tmp_path / "dataset")
    scenario = load_duplex_mpe_scenarios(dataset_root, addressing="explicit", max_scenarios=1)[0]
    stub_vad = EnergyThresholdVadStub()

    connection_attempts = 0
    received_acks: list[dict[str, JsonValue]] = []
    received_formats: set[str] = set()
    opening_extra_bodies: list[dict[str, JsonValue]] = []

    async def handle_ws(websocket: websockets.ServerConnection) -> None:
        nonlocal connection_attempts
        connection_attempts += 1
        assert "autostart=0" in (websocket.request.path if websocket.request else "")
        first_raw = await websocket.recv()
        first_msg = json.loads(first_raw)
        assert first_msg["type"] == "session.update"
        if connection_attempts == 1:
            # Simulate capacity overflow (`resource_exhausted`, `retryable: true`) followed by socket close.
            await websocket.send(
                json.dumps(
                    {
                        "type": "error",
                        "error": {
                            "code": "resource_exhausted",
                            "message": "duplex_session_capacity_exhausted",
                            "retryable": True,
                        },
                    }
                )
            )
            await websocket.close()
            return

        session_cfg = first_msg["session"]
        if isinstance(session_cfg.get("extra_body"), dict):
            opening_extra_bodies.append(session_cfg["extra_body"])
        session_obj: dict[str, JsonValue] = {
            "id": "duplex-test-01",
            "model": "nvidia/personaplex-7b-v1",
            "turn_detection": None,
            "capabilities": {
                "implementation_level": "model_native_duplex",
                "supports_model_native_turn_policy": True,
                "supports_barge_in": False,
                "supports_playback_ack": True,
                "supports_audio_truncate": False,
                "supports_client_commit": False,
                "chunk_period_ms": 80,
            },
        }
        await websocket.send(json.dumps({"type": "session.created", "session": session_obj}))
        await websocket.send(json.dumps({"type": "session.updated", "session": session_obj}))

        append_count = 0
        emitted_reply = False
        async for raw_frame in websocket:
            msg = json.loads(raw_frame)
            msg_type = msg.get("type")
            if msg_type == "input_audio_buffer.append":
                append_count += 1
                fmt = msg.get("format")
                if isinstance(fmt, str):
                    received_formats.add(fmt)
                raw_bytes = base64.b64decode(msg["audio"])
                # 24 kHz * 0.08s = 1920 float32 samples = 7680 bytes per packet (except short turn tails).
                assert len(raw_bytes) % 4 == 0
                assert len(raw_bytes) <= 1920 * 4
                # Emit a 0.35s spoken reply right after the T turn finishes (~append 60).
                if not emitted_reply and append_count >= 58:
                    emitted_reply = True
                    resp_id = "resp_test_01"
                    item_id = f"item_{resp_id}"
                    tone_t = np.arange(int(24000 * 0.35), dtype=np.float64) / 24000.0
                    tone = (0.35 * np.sin(2.0 * np.pi * 440.0 * tone_t)).astype(np.float32)
                    await websocket.send(
                        json.dumps(
                            {
                                "type": "response.created",
                                "response_id": resp_id,
                                "response": {"id": resp_id, "status": "in_progress"},
                            }
                        )
                    )
                    await websocket.send(
                        json.dumps(
                            {
                                "type": "response.output_audio.delta",
                                "response_id": resp_id,
                                "item_id": item_id,
                                "sample_rate_hz": 24000,
                                "delta": encode_float32_to_pcm16_base64(tone),
                            }
                        )
                    )
                    await websocket.send(
                        json.dumps(
                            {
                                "type": "response.done",
                                "response_id": resp_id,
                                "response": {"id": resp_id, "status": "completed", "output": [{"id": item_id}]},
                            }
                        )
                    )
            elif msg_type == "playback.ack":
                received_acks.append(msg)
            elif msg_type == "session.close":
                await websocket.send(
                    json.dumps({"type": "session.closed", "session_id": "duplex-test-01", "reason": "client_close"})
                )
                break

    async with websockets.serve(handle_ws, "127.0.0.1", 0) as server:
        port = server.sockets[0].getsockname()[1]
        spec = await negotiate_realtime_endpoint(
            f"http://127.0.0.1:{port}",
            protocol_selection="auto",
            session_options=RealtimeSessionOptions(),
        )
        assert spec.protocol == "native"
        assert spec.input_audio_encoding == "pcm_f32le"
        assert spec.input_sample_rate_hz == 24000
        assert spec.output_sample_rate_hz == 24000

        connection_attempts = 0
        output_wav = tmp_path / "out" / f"{scenario.scenario_id}.wav"
        report = await run_realtime_scenario(
            scenario,
            dataset_root=dataset_root,
            endpoint_spec=spec,
            output_audio_path=output_wav,
            vad_config=SileroActivityConfig(target_endpoint_hangover_milliseconds=400.0),
            vad_model=stub_vad,
            realtime_pacing=False,
        )

    assert connection_attempts == 2
    assert opening_extra_bodies[-1].get("silence_continuation") is False
    assert opening_extra_bodies[-1].get("auto_response") is True
    assert received_formats == {"pcm_f32le"}
    assert len(received_acks) >= 1
    assert received_acks[-1]["item_id"] == "item_resp_test_01"
    assert report["realtime_protocol"] == "native"
    assert output_wav.is_file()


async def test_loopback_turn_based_duplex_qwen_omits_auto_response_and_gates_truncate(tmp_path: Path) -> None:
    dataset_root = create_synthetic_duplex_mpe_dataset(tmp_path / "dataset")
    scenario = load_duplex_mpe_scenarios(dataset_root, addressing="explicit", max_scenarios=1)[0]
    stub_vad = EnergyThresholdVadStub()
    truncates_received: list[dict[str, JsonValue]] = []

    async def handle_qwen_ws(websocket: websockets.ServerConnection) -> None:
        first_msg = json.loads(await websocket.recv())
        assert first_msg["type"] == "session.update"
        session_cfg = first_msg["session"]
        extra_body = session_cfg.get("extra_body", {})
        # Qwen3OmniDuplexPlugin forbids extra_body.auto_response=True!
        if isinstance(extra_body, dict) and extra_body.get("auto_response") is True:
            await websocket.send(
                json.dumps(
                    {
                        "type": "error",
                        "error": {"code": "invalid_duplex_runtime_config", "message": "auto_response forbidden"},
                    }
                )
            )
            await websocket.close()
            return

        turn_det = session_cfg.get("turn_detection")
        session_obj: dict[str, JsonValue] = {
            "id": "duplex-qwen-01",
            "model": "Qwen/Qwen3-Omni-30B-A3B-Instruct",
            "turn_detection": turn_det
            if isinstance(turn_det, dict)
            else {"type": "server_vad", "silence_duration_ms": 500},
            "capabilities": {
                "implementation_level": "turn_based_duplex",
                "supports_model_native_turn_policy": False,
                "supports_barge_in": True,
                "supports_playback_ack": True,
                "supports_audio_truncate": True,
                "supports_client_commit": True,
            },
        }
        await websocket.send(json.dumps({"type": "session.created", "session": session_obj}))
        await websocket.send(json.dumps({"type": "session.updated", "session": session_obj}))

        append_idx = 0
        async for raw_frame in websocket:
            msg = json.loads(raw_frame)
            mtype = msg.get("type")
            if mtype == "input_audio_buffer.append":
                append_idx += 1
                if append_idx == 5:
                    # Cancelled response WITH NO AUDIO: client must NOT send conversation.item.truncate!
                    await websocket.send(
                        json.dumps(
                            {
                                "type": "response.created",
                                "response_id": "resp_no_audio",
                                "response": {"id": "resp_no_audio", "status": "in_progress"},
                            }
                        )
                    )
                    await websocket.send(
                        json.dumps(
                            {
                                "type": "response.done",
                                "response_id": "resp_no_audio",
                                "response": {
                                    "id": "resp_no_audio",
                                    "status": "cancelled",
                                    "output": [{"id": "item_resp_no_audio"}],
                                },
                            }
                        )
                    )
                elif append_idx == 10:
                    # Response WITH 1.0s audio that gets cancelled after 0.08s: client MUST send truncate.
                    tone = (0.3 * np.ones(24000, dtype=np.float32)).astype(np.float32)
                    await websocket.send(
                        json.dumps(
                            {
                                "type": "response.created",
                                "response_id": "resp_with_audio",
                                "response": {"id": "resp_with_audio", "status": "in_progress"},
                            }
                        )
                    )
                    await websocket.send(
                        json.dumps(
                            {
                                "type": "response.output_audio.delta",
                                "response_id": "resp_with_audio",
                                "item_id": "item_resp_with_audio",
                                "sample_rate_hz": 24000,
                                "delta": encode_float32_to_pcm16_base64(tone),
                            }
                        )
                    )
                elif append_idx == 12:
                    await websocket.send(
                        json.dumps(
                            {
                                "type": "response.done",
                                "response_id": "resp_with_audio",
                                "response": {
                                    "id": "resp_with_audio",
                                    "status": "cancelled",
                                    "output": [{"id": "item_resp_with_audio"}],
                                },
                            }
                        )
                    )
            elif mtype == "conversation.item.truncate":
                truncates_received.append(msg)
            elif mtype == "session.close":
                await websocket.send(
                    json.dumps({"type": "session.closed", "session_id": "duplex-qwen-01", "reason": "client_close"})
                )
                break

    async with websockets.serve(handle_qwen_ws, "127.0.0.1", 0) as server:
        port = server.sockets[0].getsockname()[1]
        spec = await negotiate_realtime_endpoint(f"http://127.0.0.1:{port}", protocol_selection="auto")
        assert spec.protocol == "turn_based"
        assert spec.turn_detection == {"type": "server_vad", "silence_duration_ms": 500}

        report = await run_realtime_scenario(
            scenario,
            dataset_root=dataset_root,
            endpoint_spec=spec,
            output_audio_path=tmp_path / "qwen_out" / f"{scenario.scenario_id}.wav",
            vad_model=stub_vad,
            realtime_pacing=False,
        )

    assert report["realtime_protocol"] == "turn_based"
    assert [t["item_id"] for t in truncates_received] == ["item_resp_with_audio"]


async def test_semantic_asr_judge_and_partial_score_cli_exit_code(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset_root = create_synthetic_duplex_mpe_dataset(tmp_path / "dataset")
    output_dir = tmp_path / "eval_out"
    scenarios = load_duplex_mpe_scenarios(dataset_root, addressing="explicit")
    first_scenario = scenarios[0]

    # Write retained 24kHz WAV with voiced segments for N1 and T.
    audio_path = output_dir / "audio" / f"{first_scenario.scenario_id}.wav"
    audio_path.parent.mkdir(parents=True, exist_ok=True)
    wav_samples = (0.25 * np.ones(24000 * 6, dtype=np.float32)).astype(np.float32)
    soundfile.write(str(audio_path), wav_samples, 24000, format="WAV", subtype="FLOAT")
    audio_sha = compute_file_sha256(audio_path)

    report: dict[str, JsonValue] = {
        "protocol_id": "mpeval.unified.v3",
        "realtime_protocol": "native",
        "total_duration_s": 5.5,
        "assistant_audio_sha256": audio_sha,
        "aed_config": SileroActivityConfig().to_report_dict(custom_vad_injected=True),
        "assistant_speech_segments": [[0.1, 0.5], [4.2, 5.0]],
        "T": {
            "outcome": "T_HIT_5S",
            "t_end_s": 4.0,
            "onset_s": 4.2,
            "onset_latency_s": 0.2,
            "endpoint_s": 5.0,
            "response_duration_s": 0.8,
            "forced_floor_return": False,
            "speaking_at_t_end": False,
        },
        "rows": [
            {
                "turn_index": 0,
                "event_type": "N1",
                "speaker": "Alice",
                "turn_start": 0.0,
                "turn_end": 0.4,
                "win_start": 0.0,
                "win_end": 0.8,
            },
            {
                "turn_index": 1,
                "event_type": "N4_Q",
                "speaker": "Bob",
                "turn_start": 1.0,
                "turn_end": 1.5,
                "win_start": 1.0,
                "win_end": 2.5,
            },
            {
                "turn_index": 2,
                "event_type": "N4_R",
                "speaker": "Alice",
                "turn_start": 2.5,
                "turn_end": 3.0,
                "win_start": 2.5,
                "win_end": 3.5,
            },
            {
                "turn_index": 3,
                "event_type": "T",
                "speaker": "Bob",
                "turn_start": 3.5,
                "turn_end": 4.0,
                "win_start": 3.5,
                "win_end": 5.5,
            },
        ],
    }
    write_json_atomic(output_dir / "report.json", {first_scenario.scenario_id: report})
    write_json_atomic(
        output_dir / "run.json",
        {
            "model": "openbmb/MiniCPM-o-4_5",
            "protocol": "native",
            "turn_detection": None,
            "max_concurrency": 1,
            "warmup_sessions": 0,
            "session_timeout_seconds": 420.0,
        },
    )

    tasks = build_semantic_tasks(
        {first_scenario.scenario_id: report},
        {first_scenario.scenario_id: first_scenario},
    )
    assert len(tasks) == 2

    # Partial ASR checkpoint (only 1 of 2 tasks present) is rejected normally, accepted with allow_partial=True.
    partial_row = {**tasks[0], "hyp": "mm-hmm", "asr_model": "Qwen/Qwen3-ASR-1.7B"}
    with pytest.raises(ValueError, match="do not match"):
        validate_transcribed_against_tasks(tasks, [partial_row], allow_partial=False)
    accepted_partial = validate_transcribed_against_tasks(tasks, [partial_row], allow_partial=True)
    assert len(accepted_partial) == 1

    # Score phase with missing scenario #2 must return incomplete (exit code 1) while still writing RESULTS.md.
    write_json_atomic(output_dir / "transcribed.json", [partial_row])
    rc = await asyncio.to_thread(
        evaluate_main,
        [
            "run",
            "--dataset-root",
            str(dataset_root),
            "--output-dir",
            str(output_dir),
            "--phase",
            "score",
            "--addressing",
            "explicit",
        ],
    )
    assert rc == 1
    assert (output_dir / "RESULTS.md").is_file()
    summary = json.loads((output_dir / "duplex_mpe_results.json").read_text(encoding="utf-8"))
    assert summary["status"] == "incomplete"

    # Mock httpx.AsyncClient via monkeypatch to test transcribe_tasks_via_http + judge_tasks_via_http end-to-end.
    def mock_handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/audio/transcriptions"):
            return httpx.Response(200, json={"text": "Flight 42 leaves from Gate B12."})
        body = json.loads(request.content.decode("utf-8"))
        user_prompt = body["messages"][0]["content"]
        if "BACKCHANNEL" in user_prompt:
            return httpx.Response(
                200,
                json={
                    "choices": [
                        {
                            "finish_reason": "stop",
                            "message": {"content": '{"label": "BACKCHANNEL", "reason": "short nod"}'},
                        }
                    ]
                },
            )
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "finish_reason": "stop",
                        "message": {"content": '{"correct": true, "reason": "names Gate B12"}'},
                    }
                ]
            },
        )

    mock_transport = httpx.MockTransport(mock_handler)
    real_async_client = httpx.AsyncClient

    def build_mock_async_client(
        *,
        timeout: float = 60.0,
        trust_env: bool = False,
    ) -> httpx.AsyncClient:
        return real_async_client(transport=mock_transport, timeout=timeout, trust_env=trust_env)

    monkeypatch.setattr(httpx, "AsyncClient", build_mock_async_client)

    single_out = tmp_path / "single_complete_out"
    single_audio = single_out / "audio" / f"{first_scenario.scenario_id}.wav"
    single_audio.parent.mkdir(parents=True, exist_ok=True)
    single_audio.write_bytes(audio_path.read_bytes())
    write_json_atomic(single_out / "report.json", {first_scenario.scenario_id: report})
    write_json_atomic(
        single_out / "run.json",
        {
            "model": "openbmb/MiniCPM-o-4_5",
            "protocol": "native",
            "turn_detection": None,
            "max_concurrency": 1,
            "warmup_sessions": 0,
            "session_timeout_seconds": 420.0,
        },
    )

    transcribed = await transcribe_tasks_via_http(
        tasks,
        audio_directory=single_out / "audio",
        output_path=single_out / "transcribed.json",
        asr_base_url="http://127.0.0.1:18090/v1",
    )
    assert len(transcribed) == 2
    assert all(compute_prompt_sha256(row) for row in transcribed)
    assert "Gate B12" in render_judge_prompt(transcribed[1])

    judge_cfg = JudgeEndpointConfig(
        model="Qwen/Qwen3.8-27B",
        base_url="http://127.0.0.1:18091/v1",
        enable_thinking=False,
    )
    judge_cfg_path = tmp_path / "judge.json"
    write_json_atomic(judge_cfg_path, json.loads(judge_cfg.model_dump_json()))

    judge_res = await judge_tasks_via_http(
        transcribed,
        judge_config=judge_cfg,
        output_path=single_out / "labels.json",
    )
    assert len(judge_res.verdicts) == 2
    assert judge_res.failed_task_keys == ()

    final_summary = await run_duplex_mpe_evaluation(
        DuplexMpeEvalConfig(
            dataset_root=dataset_root,
            output_dir=single_out,
            phase="score",
            addressing="explicit",
            max_scenarios=1,
            judge_config_path=judge_cfg_path,
        )
    )
    assert final_summary["status"] == "complete"
    assert final_summary["evaluation_mode"] == "scored_full_duplex"


def test_capability_audio_discovery_without_model_name_and_server_session_defaults() -> None:
    # Local anonymous checkpoint `/models/ckpt-100` for PersonaPlex (supports_client_commit=False, chunk_period_ms=80)
    # and Nemotron VoiceChat (requires_auto_response_on_open=True, chunk_period_ms=80) must infer exact native rates!
    persona_enc, persona_in_sr, persona_out_sr = infer_model_audio_defaults(
        "/models/ckpt-100",
        is_native_duplex=True,
        chunk_period_ms=80,
        supports_client_commit=False,
        requires_auto_response_on_open=False,
    )
    assert (persona_enc, persona_in_sr, persona_out_sr) == ("pcm_f32le", 24000, 24000)

    nemotron_enc, nemotron_in_sr, nemotron_out_sr = infer_model_audio_defaults(
        "/models/ckpt-200",
        is_native_duplex=True,
        chunk_period_ms=80,
        supports_client_commit=True,
        requires_auto_response_on_open=True,
    )
    assert (nemotron_enc, nemotron_in_sr, nemotron_out_sr) == ("pcm_f32le", 16000, 22050)

    nemotron_payload = build_duplex_session_payload(
        model="/models/ckpt-200",
        protocol="native",
        input_audio_encoding="pcm_f32le",
        input_sample_rate_hz=16000,
        output_sample_rate_hz=22050,
        turn_detection=None,
        supports_barge_in=False,
        supports_playback_ack=True,
        requires_auto_response_on_open=True,
        session_options=RealtimeSessionOptions(),
    )
    assert nemotron_payload["instructions"] == NEUTRAL_NEMOTRON_INSTRUCTIONS
    assert "output_audio_format" not in nemotron_payload
    audio_block = nemotron_payload["audio"]
    assert isinstance(audio_block, dict)
    out_block = audio_block["output"]
    assert isinstance(out_block, dict)
    assert out_block["sample_rate_hz"] == 22050
    assert out_block["format"] == {"type": "audio/pcm", "rate": 22050}


def test_stateful_wire_resampler_and_delta_formats() -> None:
    source_tone = np.sin(2.0 * np.pi * 440.0 * np.arange(22050, dtype=np.float32) / 22050.0).astype(np.float32)
    whole_resampler = StatefulWireResampler(24000)
    whole_waveform = np.concatenate(
        [whole_resampler.push(source_tone, 22050, stream_key="whole"), whole_resampler.flush(stream_key="whole")]
    )
    assert len(whole_waveform) == 24000

    # Irregular non-GCD chunk boundaries (317, 1000, 4093, ...) must produce the EXACT same waveform!
    chunked_resampler = StatefulWireResampler(24000)
    cuts = [0, 317, 1317, 5410, 5411, 12345, 22050]
    pieces = [
        chunked_resampler.push(source_tone[cuts[i] : cuts[i + 1]], 22050, stream_key="chunked")
        for i in range(len(cuts) - 1)
    ]
    pieces.append(chunked_resampler.flush(stream_key="chunked"))
    joined = np.concatenate(pieces)
    assert len(joined) == 24000
    assert np.allclose(joined, whole_waveform, atol=1e-6)

    # PCM_16, signed PCM_32 (0x20000000 -> +0.25), and IEEE FLOAT WAV container decoding
    for subtype in ("PCM_16", "PCM_32", "FLOAT"):
        wav_buf = io.BytesIO()
        soundfile.write(wav_buf, np.full(441, 0.25, dtype=np.float32), 22050, format="WAV", subtype=subtype)
        wav_b64 = base64.b64encode(wav_buf.getvalue()).decode("ascii")
        decoded_wav, detected_sr = decode_output_audio_delta_to_float32(wav_b64, declared_format="wav")
        assert detected_sr == 22050
        assert len(decoded_wav) == 441
        assert np.allclose(decoded_wav, 0.25, atol=1e-4)

    with pytest.raises(ProtocolSessionError, match="Unsupported output audio delta format"):
        decode_output_audio_delta_to_float32(wav_b64, declared_format="mp3")


def test_short_n4_r_subsequent_turn_does_not_corrupt_ay_and_judge_think_stripping() -> None:
    # Exact boundary onset at N4_R.win_end == 6.88 belongs to the next T turn and must NOT penalize AY!
    short_n4_report: dict[str, JsonValue] = {
        "assistant_speech_segments": [[3.2, 5.3], [6.88, 7.8]],
        "rows": [
            {
                "turn_index": 0,
                "event_type": "N4_Q",
                "speaker": "A",
                "turn_start": 0.0,
                "turn_end": 1.5,
                "win_start": 0.0,
                "win_end": 4.5,
            },
            {
                "turn_index": 1,
                "event_type": "N4_R",
                "speaker": "B",
                "turn_start": 5.0,
                "turn_end": 5.88,
                "win_start": 5.0,
                "win_end": 6.88,
            },
            {
                "turn_index": 2,
                "event_type": "T",
                "speaker": "A",
                "turn_start": 6.88,
                "turn_end": 6.92,
                "win_start": 6.88,
                "win_end": 8.5,
            },
        ],
    }
    counts = count_n4_windows(short_n4_report)
    assert counts.eligible == 1
    assert counts.yielded == 1

    think_wrapped = (
        '<think>Example schema {"correct": false} while reasoning...</think>\n'
        '{"correct": true, "reason": "exact match"}'
    )
    parsed = parse_judge_verdict_json(think_wrapped, "answer")
    assert parsed is not None and parsed["correct"] is True
    assert parse_judge_verdict_json('<think>{"correct": true}', "answer") is None

    dec = JudgeTaskDecodingConfig(temperature=0.0, max_tokens=2048, token_limit_param="max_completion_tokens")
    assert dec.to_metadata_dict()["token_limit_param"] == "max_completion_tokens"


async def test_completed_response_speech_started_interruption_and_early_close_failure(tmp_path: Path) -> None:
    dataset_root = create_synthetic_duplex_mpe_dataset(tmp_path / "dataset")
    scenario = load_duplex_mpe_scenarios(dataset_root, addressing="explicit", max_scenarios=1)[0]
    stub_vad = EnergyThresholdVadStub()
    truncates_received: list[dict[str, JsonValue]] = []

    async def handle_completed_barge_in(websocket: websockets.ServerConnection) -> None:
        first_msg = json.loads(await websocket.recv())
        turn_det = first_msg["session"].get("turn_detection")
        session_obj: dict[str, JsonValue] = {
            "id": "duplex-qwen-barge",
            "model": "Qwen/Qwen3-Omni-30B-A3B-Instruct",
            "input_audio_format": "pcm_f32le",
            "turn_detection": turn_det,
            "capabilities": {
                "implementation_level": "turn_based_duplex",
                "supports_model_native_turn_policy": False,
                "supports_multi_session": False,
                "supports_barge_in": True,
                "supports_playback_ack": True,
                "supports_audio_truncate": True,
                "supports_client_commit": True,
            },
        }
        await websocket.send(json.dumps({"type": "session.created", "session": session_obj}))
        await websocket.send(json.dumps({"type": "session.updated", "session": session_obj}))
        step = 0
        async for raw_frame in websocket:
            msg = json.loads(raw_frame)
            mtype = msg.get("type")
            if mtype == "input_audio_buffer.append":
                step += 1
                if step == 4:
                    long_tone = (0.35 * np.ones(24000 * 3, dtype=np.float32)).astype(np.float32)
                    await websocket.send(
                        json.dumps(
                            {
                                "type": "response.created",
                                "response_id": "resp_fast_done",
                                "response": {"id": "resp_fast_done", "status": "in_progress"},
                            }
                        )
                    )
                    await websocket.send(
                        json.dumps(
                            {
                                "type": "response.output_audio.delta",
                                "response_id": "resp_fast_done",
                                "item_id": "custom_item_99",
                                "sample_rate_hz": 24000,
                                "delta": encode_float32_to_pcm16_base64(long_tone),
                            }
                        )
                    )
                    await websocket.send(
                        json.dumps(
                            {
                                "type": "response.done",
                                "response_id": "resp_fast_done",
                                "response": {
                                    "id": "resp_fast_done",
                                    "status": "completed",
                                    "output": [{"id": "custom_item_99"}],
                                },
                            }
                        )
                    )
                elif step == 7:
                    await websocket.send(json.dumps({"type": "input_audio_buffer.speech_started"}))
            elif mtype == "conversation.item.truncate":
                truncates_received.append(msg)
            elif mtype == "session.close":
                await websocket.send(json.dumps({"type": "session.closed", "session_id": "duplex-qwen-barge"}))
                break

    async with websockets.serve(handle_completed_barge_in, "127.0.0.1", 0) as server:
        port = server.sockets[0].getsockname()[1]
        spec = await negotiate_realtime_endpoint(f"http://127.0.0.1:{port}", protocol_selection="auto")
        assert spec.max_sessions == 1
        assert spec.supports_multi_session is False
        barge_report = await run_realtime_scenario(
            scenario,
            dataset_root=dataset_root,
            endpoint_spec=spec,
            output_audio_path=tmp_path / "barge" / f"{scenario.scenario_id}.wav",
            vad_model=stub_vad,
            realtime_pacing=False,
        )
    assert [t["item_id"] for t in truncates_received] == ["custom_item_99"]
    cancelled_samples = barge_report["cancelled_unplayed_samples"]
    assert isinstance(cancelled_samples, int) and cancelled_samples > 24000

    async def handle_early_close(websocket: websockets.ServerConnection) -> None:
        _ = json.loads(await websocket.recv())
        session_obj: dict[str, JsonValue] = {
            "id": "early-close-01",
            "model": "openbmb/MiniCPM-o-4_5",
            "input_audio_format": "pcm_f32le",
            "turn_detection": None,
            "capabilities": {
                "implementation_level": "model_native_duplex",
                "supports_model_native_turn_policy": True,
                "supports_barge_in": True,
                "supports_playback_ack": True,
                "supports_audio_truncate": True,
                "supports_client_commit": True,
                "chunk_period_ms": 1000,
            },
        }
        await websocket.send(json.dumps({"type": "session.created", "session": session_obj}))
        await websocket.send(json.dumps({"type": "session.updated", "session": session_obj}))
        seen = 0
        async for raw_frame in websocket:
            msg = json.loads(raw_frame)
            mtype = msg.get("type")
            if mtype == "session.close":
                await websocket.send(json.dumps({"type": "session.closed", "session_id": "early-close-01"}))
                return
            if mtype == "input_audio_buffer.append":
                seen += 1
                if seen == 2:
                    await websocket.send(json.dumps({"type": "session.closed", "session_id": "early-close-01"}))

    async with websockets.serve(handle_early_close, "127.0.0.1", 0) as server:
        port = server.sockets[0].getsockname()[1]
        early_spec = await negotiate_realtime_endpoint(f"http://127.0.0.1:{port}", protocol_selection="auto")
        with pytest.raises(RuntimeError, match="Server closed session before scenario completion"):
            await run_realtime_scenario(
                scenario,
                dataset_root=dataset_root,
                endpoint_spec=early_spec,
                output_audio_path=tmp_path / "early" / f"{scenario.scenario_id}.wav",
                vad_model=stub_vad,
                realtime_pacing=False,
            )


def test_symlink_ref_audio_rejection_and_empty_cli_help(tmp_path: Path) -> None:
    real_wav = tmp_path / "voice.wav"
    soundfile.write(str(real_wav), np.zeros(1600, dtype=np.float32), 16000, format="WAV", subtype="PCM_16")
    symlink_wav = tmp_path / "voice_link.wav"
    symlink_wav.symlink_to(real_wav)
    with pytest.raises(ValueError, match="Symlink reference audio"):
        encode_audio_file_as_data_url(symlink_wav)

    with pytest.raises(SystemExit) as exc_info:
        evaluate_main([])
    assert exc_info.value.code == 0

    # Missing human-turn WAV on disk must report False rather than claiming verified dataset inventory!
    ds_root = create_synthetic_duplex_mpe_dataset(tmp_path / "missing_human_ds")
    loaded_scenarios = load_duplex_mpe_scenarios(ds_root, addressing="explicit", max_scenarios=1)
    assert verify_selected_human_audio_inventory(ds_root, loaded_scenarios) is True
    missing_clip = resolve_dataset_audio_path(ds_root, loaded_scenarios[0].turns[0].audio_path)
    missing_clip.unlink()
    assert verify_selected_human_audio_inventory(ds_root, loaded_scenarios) is False


def test_resampler_source_rate_invariance_and_double_flush_safety() -> None:
    resampler = StatefulWireResampler(24000)
    short_chunk = np.full(100, 0.25, dtype=np.float32)
    _ = resampler.push(short_chunk, 22050, stream_key="resp_1")
    # Switching to another non-target rate or to target rate mid-response must fail fast!
    with pytest.raises(ProtocolSessionError, match="changed sample rate mid-response"):
        resampler.push(short_chunk, 16000, stream_key="resp_1")
    with pytest.raises(ProtocolSessionError, match="changed sample rate mid-response"):
        resampler.push(short_chunk, 24000, stream_key="resp_1")
    tail_first = resampler.flush(stream_key="resp_1")
    assert len(tail_first) > 0
    # Second flush (e.g., response.done following response.output_audio.done) must be a clean no-op!
    tail_second = resampler.flush(stream_key="resp_1")
    assert len(tail_second) == 0


async def test_local_cancellation_discards_resampler_tail_before_first_stable_output(tmp_path: Path) -> None:
    dataset_root = create_synthetic_duplex_mpe_dataset(tmp_path / "dataset")
    scenario = load_duplex_mpe_scenarios(dataset_root, addressing="explicit", max_scenarios=1)[0]
    stub_vad = EnergyThresholdVadStub()

    async def handle_short_resampled_cancel(websocket: websockets.ServerConnection) -> None:
        first_msg = json.loads(await websocket.recv())
        turn_det = first_msg["session"].get("turn_detection")
        session_obj: dict[str, JsonValue] = {
            "id": "duplex-cancel-tail",
            "model": "Qwen/Qwen3-Omni-30B-A3B-Instruct",
            "input_audio_format": "pcm_f32le",
            "turn_detection": turn_det,
            "capabilities": {
                "implementation_level": "turn_based_duplex",
                "supports_model_native_turn_policy": False,
                "supports_barge_in": True,
                "supports_playback_ack": False,
                "supports_audio_truncate": True,
                "supports_client_commit": True,
            },
        }
        await websocket.send(json.dumps({"type": "session.created", "session": session_obj}))
        await websocket.send(json.dumps({"type": "session.updated", "session": session_obj}))
        step = 0
        async for raw_frame in websocket:
            msg = json.loads(raw_frame)
            mtype = msg.get("type")
            if mtype == "input_audio_buffer.append":
                step += 1
                if step == 3:
                    # 5 samples at 22050 -> 24000 is shorter than the polyphase filter half-width (10),
                    # so push() emits 0 stable samples while holding energy in filter history!
                    tiny_burst = np.full(5, 0.8, dtype=np.float32)
                    await websocket.send(
                        json.dumps(
                            {
                                "type": "response.created",
                                "response_id": "resp_tiny",
                                "response": {"id": "resp_tiny", "status": "in_progress"},
                            }
                        )
                    )
                    await websocket.send(
                        json.dumps(
                            {
                                "type": "response.output_audio.delta",
                                "response_id": "resp_tiny",
                                "item_id": "item_tiny",
                                "sample_rate_hz": 22050,
                                "delta": encode_float32_to_pcm16_base64(tiny_burst),
                            }
                        )
                    )
                    # Local barge-in followed by late response.done(status="completed"):
                    # the cancelled filter tail MUST NOT be flushed into the output waveform!
                    await websocket.send(json.dumps({"type": "input_audio_buffer.speech_started"}))
                    await websocket.send(
                        json.dumps(
                            {
                                "type": "response.done",
                                "response_id": "resp_tiny",
                                "response": {
                                    "id": "resp_tiny",
                                    "status": "completed",
                                    "output": [{"id": "item_tiny"}],
                                },
                            }
                        )
                    )
            elif mtype == "session.close":
                await websocket.send(json.dumps({"type": "session.closed", "session_id": "duplex-cancel-tail"}))
                break

    async with websockets.serve(handle_short_resampled_cancel, "127.0.0.1", 0) as server:
        port = server.sockets[0].getsockname()[1]
        spec = await negotiate_realtime_endpoint(f"http://127.0.0.1:{port}", protocol_selection="auto")
        out_wav = tmp_path / "cancel_tail" / f"{scenario.scenario_id}.wav"
        await run_realtime_scenario(
            scenario,
            dataset_root=dataset_root,
            endpoint_spec=spec,
            output_audio_path=out_wav,
            vad_model=stub_vad,
            realtime_pacing=False,
        )
    captured_audio, _ = soundfile.read(str(out_wav), dtype="float32")
    assert float(np.max(np.abs(captured_audio))) == 0.0


async def test_control_sender_failure_wakes_drain_and_close_timeout_cancels_tasks(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import benchmarks.duplex_mpe.realtime_session as rt_mod

    dataset_root = create_synthetic_duplex_mpe_dataset(tmp_path / "dataset")
    scenario = load_duplex_mpe_scenarios(dataset_root, addressing="explicit", max_scenarios=1)[0]
    stub_vad = EnergyThresholdVadStub()

    # 1. Deterministic drain wakeup when control sender fails with unfinished queue items at the end of streaming!
    ack_gate = asyncio.Event()
    entered_drain_join = asyncio.Event()

    async def handle_ack_drain_fail(websocket: websockets.ServerConnection) -> None:
        _ = json.loads(await websocket.recv())
        session_obj: dict[str, JsonValue] = {
            "id": "ack-drain-01",
            "model": "nvidia/personaplex-7b-v1",
            "turn_detection": None,
            "capabilities": {
                "implementation_level": "model_native_duplex",
                "supports_model_native_turn_policy": True,
                "supports_barge_in": False,
                "supports_playback_ack": True,
                "supports_audio_truncate": False,
                "supports_client_commit": False,
                "chunk_period_ms": 80,
            },
        }
        await websocket.send(json.dumps({"type": "session.created", "session": session_obj}))
        await websocket.send(json.dumps({"type": "session.updated", "session": session_obj}))
        seen_appends = 0
        async for raw_frame in websocket:
            msg = json.loads(raw_frame)
            mtype = msg.get("type")
            if mtype == "input_audio_buffer.append":
                seen_appends += 1
                # Emit deltas on step 2 so acks are queued well before end of schedule!
                if seen_appends == 2:
                    for idx in (1, 2):
                        tone = np.full(1920, 0.25, dtype=np.float32)
                        await websocket.send(
                            json.dumps(
                                {
                                    "type": "response.output_audio.delta",
                                    "response_id": f"resp_ack_{idx}",
                                    "item_id": f"item_ack_{idx}",
                                    "sample_rate_hz": 24000,
                                    "delta": encode_float32_to_pcm16_base64(tone),
                                }
                            )
                        )
            elif mtype == "session.close":
                await websocket.send(json.dumps({"type": "session.closed", "session_id": "ack-drain-01"}))
                return

    async with websockets.serve(handle_ack_drain_fail, "127.0.0.1", 0) as server:
        port = server.sockets[0].getsockname()[1]
        spec = await negotiate_realtime_endpoint(f"http://127.0.0.1:{port}", protocol_selection="auto")

        real_open_ws = rt_mod.open_websocket_with_admission_state
        real_advance = rt_mod.advance_schedule_step_sync

        def tracking_advance(
            schedule: rt_mod.Generator[NDArray[np.float32], NDArray[np.float32], rt_mod.ScenarioScheduleResult],
            rendered_chunk: NDArray[np.float32],
        ) -> tuple[NDArray[np.float32] | None, rt_mod.ScenarioScheduleResult | None]:
            nxt, done_res = real_advance(schedule, rendered_chunk)
            if nxt is None and done_res is not None:
                entered_drain_join.set()
            return nxt, done_res

        class AckFailingSocketProxy:
            def __init__(self, inner: websockets.ClientConnection) -> None:
                self._inner = inner

            async def __aenter__(self) -> AckFailingSocketProxy:
                await self._inner.__aenter__()
                return self

            async def __aexit__(
                self,
                exc_type: type[BaseException] | None,
                exc: BaseException | None,
                tb: TracebackType | None,
            ) -> None:
                await self._inner.__aexit__(exc_type, exc, tb)

            def __aiter__(self) -> AsyncIterator[str | bytes]:
                return self._inner.__aiter__()

            async def recv(self) -> str | bytes:
                return await self._inner.recv()

            async def send(self, message: str) -> None:
                parsed = json.loads(message)
                if parsed.get("type") == "playback.ack":
                    ack_gate.set()
                    raise OSError("simulated control sender socket failure during drain")
                await self._inner.send(message)

        # Delay pump_control_messages via Queue.get wrapper until stream_measured_schedule finishes (`entered_drain_join`),
        # guaranteeing >=2 unfinished tasks are in control_queue when control_queue.join() begins!
        real_queue_cls = asyncio.Queue

        class DrainDelayedControlQueue(
            asyncio.Queue[tuple[str, dict[str, JsonValue], rt_mod.PlaybackAckRecord | None] | None]
        ):
            async def get(self) -> tuple[str, dict[str, JsonValue], rt_mod.PlaybackAckRecord | None] | None:
                item = await super().get()
                if item is not None:
                    await asyncio.wait_for(entered_drain_join.wait(), timeout=5.0)
                return item

        async def open_ack_failing_ws(
            websocket_url: str,
            *,
            admission_state: rt_mod.AdmissionState,
            open_timeout_seconds: float = rt_mod.HANDSHAKE_TIMEOUT_SECONDS,
            additional_headers: dict[str, str] | None = None,
            trust_env: bool = False,
        ) -> AckFailingSocketProxy:
            conn = await real_open_ws(
                websocket_url,
                admission_state=admission_state,
                open_timeout_seconds=open_timeout_seconds,
                additional_headers=additional_headers,
                trust_env=trust_env,
            )
            return AckFailingSocketProxy(conn)

        monkeypatch.setattr(rt_mod, "advance_schedule_step_sync", tracking_advance)
        monkeypatch.setattr(rt_mod.asyncio, "Queue", DrainDelayedControlQueue)
        monkeypatch.setattr(rt_mod, "open_websocket_with_admission_state", open_ack_failing_ws)
        with pytest.raises(RuntimeError, match="Realtime control sender failed during drain: OSError"):
            await run_realtime_scenario(
                scenario,
                dataset_root=dataset_root,
                endpoint_spec=spec,
                output_audio_path=tmp_path / "ack_drop" / f"{scenario.scenario_id}.wav",
                vad_model=stub_vad,
                realtime_pacing=False,
            )
        assert entered_drain_join.is_set()
        assert ack_gate.is_set()
        monkeypatch.setattr(rt_mod, "advance_schedule_step_sync", real_advance)
        monkeypatch.setattr(rt_mod.asyncio, "Queue", real_queue_cls)
        monkeypatch.setattr(rt_mod, "open_websocket_with_admission_state", real_open_ws)

    # 2. Hung/timing-out `websocket.send(session.close)` in `finally` + Python 3.10 `asyncio.TimeoutError`!
    close_send_attempted = asyncio.Event()

    async def handle_hang_forever(websocket: websockets.ServerConnection) -> None:
        first_raw = await websocket.recv()
        first_msg = json.loads(first_raw)
        assert first_msg["type"] == "session.update"
        session_obj: dict[str, JsonValue] = {
            "id": "hang-01",
            "model": "nvidia/personaplex-7b-v1",
            "turn_detection": None,
            "capabilities": {
                "implementation_level": "model_native_duplex",
                "supports_model_native_turn_policy": True,
                "supports_barge_in": False,
                "supports_playback_ack": False,
                "supports_audio_truncate": False,
                "supports_client_commit": False,
                "chunk_period_ms": 80,
            },
        }
        await websocket.send(json.dumps({"type": "session.created", "session": session_obj}))
        await websocket.send(json.dumps({"type": "session.updated", "session": session_obj}))
        async for raw_frame in websocket:
            msg = json.loads(raw_frame)
            if msg.get("type") == "session.close":
                await websocket.send(json.dumps({"type": "session.closed", "session_id": "hang-01"}))
                return

    async with websockets.serve(handle_hang_forever, "127.0.0.1", 0) as hang_server:
        hang_port = hang_server.sockets[0].getsockname()[1]
        hang_spec = await negotiate_realtime_endpoint(f"http://127.0.0.1:{hang_port}", protocol_selection="auto")

        class CloseTimeoutSocketProxy:
            def __init__(self, inner: websockets.ClientConnection) -> None:
                self._inner = inner

            async def __aenter__(self) -> CloseTimeoutSocketProxy:
                await self._inner.__aenter__()
                return self

            async def __aexit__(
                self,
                exc_type: type[BaseException] | None,
                exc: BaseException | None,
                tb: TracebackType | None,
            ) -> None:
                await self._inner.__aexit__(exc_type, exc, tb)

            def __aiter__(self) -> AsyncIterator[str | bytes]:
                return self._inner.__aiter__()

            async def recv(self) -> str | bytes:
                return await self._inner.recv()

            async def send(self, message: str) -> None:
                parsed = json.loads(message)
                if parsed.get("type") == "session.close":
                    close_send_attempted.set()
                    raise asyncio.TimeoutError("simulated hung close frame send")
                await self._inner.send(message)

        baseline_tasks = set(asyncio.all_tasks())
        monkeypatch.setattr(
            rt_mod,
            "compute_effective_scenario_timeout_seconds",
            real_compute_timeout := rt_mod.compute_effective_scenario_timeout_seconds,
        )
        monkeypatch.setattr(
            rt_mod,
            "compute_effective_scenario_timeout_seconds",
            lambda *, configured_timeout_seconds, maximum_schedule_duration_seconds: configured_timeout_seconds,
        )

        # Probe runs cleanly on real_open_ws; wrap ONLY the measured scenario connection so close send raises TimeoutError!
        call_index = 0

        async def open_close_timeout_ws(
            websocket_url: str,
            *,
            admission_state: rt_mod.AdmissionState,
            open_timeout_seconds: float = rt_mod.HANDSHAKE_TIMEOUT_SECONDS,
            additional_headers: dict[str, str] | None = None,
            trust_env: bool = False,
        ) -> websockets.ClientConnection | CloseTimeoutSocketProxy:
            nonlocal call_index
            call_index += 1
            conn = await real_open_ws(
                websocket_url,
                admission_state=admission_state,
                open_timeout_seconds=open_timeout_seconds,
                additional_headers=additional_headers,
                trust_env=trust_env,
            )
            if call_index >= 2:
                return CloseTimeoutSocketProxy(conn)
            return conn

        monkeypatch.setattr(rt_mod, "open_websocket_with_admission_state", open_close_timeout_ws)
        eval_cfg = DuplexMpeEvalConfig(
            dataset_root=dataset_root,
            output_dir=tmp_path / "timeout_eval_out",
            phase="generate",
            addressing="explicit",
            max_scenarios=1,
            base_url=f"http://127.0.0.1:{hang_port}",
            warmup_sessions=0,
            session_timeout_seconds=0.05,
        )
        gen_res = await run_generate_phase(eval_cfg, [scenario], vad_model_factory=EnergyThresholdVadStub)
        assert close_send_attempted.is_set()
        assert scenario.scenario_id in gen_res.failed_scenarios
        assert (tmp_path / "timeout_eval_out" / "failures.json").is_file()
        # Verify zero leaked receiver_task / control_task after hung-close cleanup!
        leaked_tasks = [
            task
            for task in (set(asyncio.all_tasks()) - baseline_tasks)
            if not task.done() and ("receive_server_frames" in repr(task) or "pump_control_messages" in repr(task))
        ]
        assert leaked_tasks == []
        monkeypatch.setattr(rt_mod, "open_websocket_with_admission_state", real_open_ws)
        monkeypatch.setattr(rt_mod, "compute_effective_scenario_timeout_seconds", real_compute_timeout)

        # 3. Unexpected programming bugs (`TypeError` in receiver, `AssertionError` in control sender) MUST propagate!
        real_decode = rt_mod.decode_json_frame

        def buggy_decode(raw_frame: str | bytes, context_label: str) -> dict[str, JsonValue]:
            if context_label == "/v1/realtime stream":
                raise TypeError("injected receiver programming bug")
            return real_decode(raw_frame, context_label)

        monkeypatch.setattr(rt_mod, "decode_json_frame", buggy_decode)
        with pytest.raises(TypeError, match="injected receiver programming bug"):
            await run_realtime_scenario(
                scenario,
                dataset_root=dataset_root,
                endpoint_spec=hang_spec,
                output_audio_path=tmp_path / "rx_bug" / f"{scenario.scenario_id}.wav",
                vad_model=stub_vad,
                realtime_pacing=False,
            )
        monkeypatch.setattr(rt_mod, "decode_json_frame", real_decode)

    async with websockets.serve(handle_ack_drain_fail, "127.0.0.1", 0) as ctrl_bug_server:
        ctrl_port = ctrl_bug_server.sockets[0].getsockname()[1]
        ctrl_spec = await negotiate_realtime_endpoint(f"http://127.0.0.1:{ctrl_port}", protocol_selection="auto")

        class ControlBugSocketProxy(CloseTimeoutSocketProxy):
            async def send(self, message: str) -> None:
                parsed = json.loads(message)
                if parsed.get("type") == "playback.ack":
                    raise AssertionError("injected control sender programming bug")
                await self._inner.send(message)

        async def open_ctrl_bug_ws(
            websocket_url: str,
            *,
            admission_state: rt_mod.AdmissionState,
            open_timeout_seconds: float = rt_mod.HANDSHAKE_TIMEOUT_SECONDS,
            additional_headers: dict[str, str] | None = None,
            trust_env: bool = False,
        ) -> ControlBugSocketProxy:
            conn = await real_open_ws(
                websocket_url,
                admission_state=admission_state,
                open_timeout_seconds=open_timeout_seconds,
                additional_headers=additional_headers,
                trust_env=trust_env,
            )
            return ControlBugSocketProxy(conn)

        monkeypatch.setattr(rt_mod, "open_websocket_with_admission_state", open_ctrl_bug_ws)
        with pytest.raises(AssertionError, match="injected control sender programming bug"):
            await run_realtime_scenario(
                scenario,
                dataset_root=dataset_root,
                endpoint_spec=ctrl_spec,
                output_audio_path=tmp_path / "ctrl_bug" / f"{scenario.scenario_id}.wav",
                vad_model=stub_vad,
                realtime_pacing=False,
            )
        monkeypatch.setattr(rt_mod, "open_websocket_with_admission_state", real_open_ws)


async def test_openai_turn_model_forwarding_and_temperature_rejection() -> None:
    seen_queries: list[str] = []

    async def handle_openai_turn(websocket: websockets.ServerConnection) -> None:
        req_path = websocket.request.path if websocket.request else ""
        if "duplex=1" in req_path:
            await websocket.send(json.dumps({"type": "error", "code": "unsupported", "error": "no duplex"}))
            await websocket.close()
            return
        seen_queries.append(req_path)
        if "model=wrong-model" in req_path:
            await websocket.send(
                json.dumps(
                    {
                        "type": "error",
                        "error": {"code": "unsupported_model", "message": "Model not available"},
                    }
                )
            )
            await websocket.close(code=1008)
            return
        await websocket.send(
            json.dumps(
                {
                    "type": "session.created",
                    "session": {"id": "sess_oa_1", "model": "Qwen/Qwen3-Omni-30B-A3B-Instruct"},
                }
            )
        )
        await websocket.close()

    async with websockets.serve(handle_openai_turn, "127.0.0.1", 0) as server:
        port = server.sockets[0].getsockname()[1]
        spec = await negotiate_realtime_endpoint(
            f"http://127.0.0.1:{port}",
            protocol_selection="openai_turn",
            requested_model="Qwen/Qwen3-Omni-30B-A3B-Instruct",
        )
        assert "model=Qwen%2FQwen3-Omni-30B-A3B-Instruct" in spec.websocket_url
        assert any("model=Qwen%2FQwen3-Omni-30B-A3B-Instruct" in q for q in seen_queries)

        with pytest.raises(ProtocolSessionError, match="unsupported_model"):
            await negotiate_realtime_endpoint(
                f"http://127.0.0.1:{port}",
                protocol_selection="openai_turn",
                requested_model="wrong-model",
            )

        with pytest.raises(ValueError, match="do not support --temperature"):
            await negotiate_realtime_endpoint(
                f"http://127.0.0.1:{port}",
                protocol_selection="openai_turn",
                session_options=RealtimeSessionOptions(temperature=0.7),
            )

    with pytest.raises(ValueError, match="do not support --temperature"):
        build_openai_turn_session_payload(
            model="Qwen/Qwen3-Omni-30B-A3B-Instruct",
            turn_detection={"type": "server_vad", "silence_duration_ms": 500},
            session_options=RealtimeSessionOptions(temperature=0.7),
        )


def test_validate_scenario_reports_rejects_forged_overlong_and_inconsistent_target_timestamps(
    tmp_path: Path,
) -> None:
    dataset_root = create_synthetic_duplex_mpe_dataset(tmp_path / "dataset")
    scenario = load_duplex_mpe_scenarios(dataset_root, addressing="explicit", max_scenarios=1)[0]
    scenarios_by_id = {scenario.scenario_id: scenario}

    def make_report(t_block: dict[str, JsonValue]) -> dict[str, JsonValue]:
        return {
            "protocol_id": "mpeval.unified.v3",
            "total_duration_s": 5.5,
            "aed_config": SileroActivityConfig().to_report_dict(custom_vad_injected=True),
            "assistant_speech_segments": [[4.2, 5.0]],
            "T": t_block,
            "rows": [
                {
                    "turn_index": 0,
                    "event_type": "N1",
                    "speaker": "Alice",
                    "turn_start": 0.0,
                    "turn_end": 0.4,
                    "win_start": 0.0,
                    "win_end": 0.8,
                },
                {
                    "turn_index": 1,
                    "event_type": "N4_Q",
                    "speaker": "Bob",
                    "turn_start": 1.0,
                    "turn_end": 1.5,
                    "win_start": 1.0,
                    "win_end": 2.5,
                },
                {
                    "turn_index": 2,
                    "event_type": "N4_R",
                    "speaker": "Alice",
                    "turn_start": 2.5,
                    "turn_end": 3.0,
                    "win_start": 2.5,
                    "win_end": 3.5,
                },
                {
                    "turn_index": 3,
                    "event_type": "T",
                    "speaker": "Bob",
                    "turn_start": 3.5,
                    "turn_end": 4.0,
                    "win_start": 3.5,
                    "win_end": 5.5,
                },
            ],
        }

    # Valid fresh-onset T_OVERLONG_60S (onset within 5 s of t_end_s=4.0) passes:
    valid_overlong = make_report(
        {
            "outcome": "T_OVERLONG_60S",
            "t_end_s": 4.0,
            "onset_s": 4.2,
            "onset_latency_s": 0.2,
            "forced_floor_return": True,
            "speaking_at_t_end": False,
        }
    )
    validate_scenario_reports({scenario.scenario_id: valid_overlong}, scenarios_by_id)

    # Forged T_OVERLONG_60S with onset_s=999 / onset_latency_s=999 must be rejected!
    forged_overlong = make_report(
        {
            "outcome": "T_OVERLONG_60S",
            "t_end_s": 4.0,
            "onset_s": 999.0,
            "onset_latency_s": 999.0,
            "speaking_at_t_end": False,
        }
    )
    with pytest.raises(ValueError, match="outside audio timeline"):
        validate_scenario_reports({scenario.scenario_id: forged_overlong}, scenarios_by_id)

    # Inconsistent onset_s - t_end_s != onset_latency_s on T_HIT_5S must be rejected!
    inconsistent_hit = make_report(
        {
            "outcome": "T_HIT_5S",
            "t_end_s": 4.0,
            "onset_s": 5.1,
            "onset_latency_s": 0.2,
            "speaking_at_t_end": False,
        }
    )
    with pytest.raises(ValueError, match="Inconsistent T.onset_s"):
        validate_scenario_reports({scenario.scenario_id: inconsistent_hit}, scenarios_by_id)

    # Contradictory T_HIT_5S with speaking_at_t_end=True or missing speaking_at_t_end on T_OVERLONG_60S must fail!
    contradictory_hit = make_report(
        {
            "outcome": "T_HIT_5S",
            "t_end_s": 4.0,
            "onset_s": 4.2,
            "onset_latency_s": 0.2,
            "speaking_at_t_end": True,
        }
    )
    with pytest.raises(ValueError, match="Contradictory speaking_at_t_end=True"):
        validate_scenario_reports({scenario.scenario_id: contradictory_hit}, scenarios_by_id)

    missing_speaking_overlong = make_report(
        {
            "outcome": "T_OVERLONG_60S",
            "t_end_s": 4.0,
            "onset_s": 4.2,
            "onset_latency_s": 0.2,
        }
    )
    with pytest.raises(ValueError, match="requires boolean speaking_at_t_end"):
        validate_scenario_reports({scenario.scenario_id: missing_speaking_overlong}, scenarios_by_id)

    # Valid overlong continuation (speaking_at_t_end=True with null onset_s/onset_latency_s) passes:
    valid_overlong_cont = make_report(
        {
            "outcome": "T_OVERLONG_60S",
            "t_end_s": 4.0,
            "onset_s": None,
            "onset_latency_s": None,
            "forced_floor_return": True,
            "speaking_at_t_end": True,
        }
    )
    validate_scenario_reports({scenario.scenario_id: valid_overlong_cont}, scenarios_by_id)

    # Contradictory overlong continuation populating fresh-onset fields (`onset_s=999.0`) MUST fail!
    contradictory_overlong_cont = make_report(
        {
            "outcome": "T_OVERLONG_60S",
            "t_end_s": 4.0,
            "onset_s": 999.0,
            "onset_latency_s": 999.0,
            "speaking_at_t_end": True,
        }
    )
    with pytest.raises(ValueError, match="cannot populate onset_s/onset_latency_s"):
        validate_scenario_reports({scenario.scenario_id: contradictory_overlong_cont}, scenarios_by_id)


async def test_offline_score_sanitizes_saved_manifest_credentials_and_base_url(tmp_path: Path) -> None:
    dataset_root = create_synthetic_duplex_mpe_dataset(tmp_path / "dataset")
    scenario = load_duplex_mpe_scenarios(dataset_root, addressing="explicit", max_scenarios=1)[0]
    out_dir = tmp_path / "dirty_manifest_out"
    audio_path = out_dir / "audio" / f"{scenario.scenario_id}.wav"
    audio_path.parent.mkdir(parents=True, exist_ok=True)
    wav_samples = np.zeros(24000 * 6, dtype=np.float32)
    soundfile.write(str(audio_path), wav_samples, 24000, format="WAV", subtype="FLOAT")
    audio_sha = compute_file_sha256(audio_path)

    report: dict[str, JsonValue] = {
        "protocol_id": "mpeval.unified.v3",
        "realtime_protocol": "native",
        "total_duration_s": 5.5,
        "assistant_audio_sha256": audio_sha,
        "aed_config": SileroActivityConfig().to_report_dict(custom_vad_injected=True),
        "assistant_speech_segments": [],
        "T": {
            "outcome": "T_MISS_5S",
            "t_end_s": 4.0,
            "onset_s": None,
            "onset_latency_s": None,
            "endpoint_s": None,
            "response_duration_s": None,
            "forced_floor_return": False,
            "speaking_at_t_end": False,
        },
        "rows": [
            {
                "turn_index": 0,
                "event_type": "N1",
                "speaker": "Alice",
                "turn_start": 0.0,
                "turn_end": 0.4,
                "win_start": 0.0,
                "win_end": 0.8,
            },
            {
                "turn_index": 1,
                "event_type": "N4_Q",
                "speaker": "Bob",
                "turn_start": 1.0,
                "turn_end": 1.5,
                "win_start": 1.0,
                "win_end": 2.5,
            },
            {
                "turn_index": 2,
                "event_type": "N4_R",
                "speaker": "Alice",
                "turn_start": 2.5,
                "turn_end": 3.0,
                "win_start": 2.5,
                "win_end": 3.5,
            },
            {
                "turn_index": 3,
                "event_type": "T",
                "speaker": "Bob",
                "turn_start": 3.5,
                "turn_end": 4.0,
                "win_start": 3.5,
                "win_end": 5.5,
            },
        ],
    }
    write_json_atomic(out_dir / "report.json", {scenario.scenario_id: report})
    write_json_atomic(
        out_dir / "run.json",
        {
            "model": "openbmb/MiniCPM-o-4_5",
            "protocol": "native",
            "turn_detection": None,
            "launch_command": "HF_TOKEN=hf_syntheticsecret999 vllm serve openbmb/MiniCPM-o-4_5 --omni --api-key sk-syntheticsecret999",
            "base_url": "https://user:pass@127.0.0.1:18220/v1/realtime?duplex=1&token=leak",
            "max_concurrency": 1,
            "warmup_sessions": 0,
            "session_timeout_seconds": 420.0,
        },
    )

    summary = await run_duplex_mpe_evaluation(
        DuplexMpeEvalConfig(
            dataset_root=dataset_root,
            output_dir=out_dir,
            phase="score",
            addressing="explicit",
            max_scenarios=1,
        )
    )
    serialized = json.dumps(summary)
    assert "hf_syntheticsecret999" not in serialized
    assert "sk-syntheticsecret999" not in serialized
    assert "user:pass" not in serialized
    assert "token=leak" not in serialized
    prov = summary["provenance"]
    assert isinstance(prov, dict)
    assert "<REDACTED>" in str(prov["launch_command"])
    srv = prov["declared_server_config"]
    assert isinstance(srv, dict)
    assert srv["base_url"] == "https://127.0.0.1:18220"
    assert srv["retained_audio_verified"] is True
    assert srv["retained_audio_verified_scenarios"] == 1

    # Imported legacy report without assistant_audio_sha256 must NOT claim retained_audio_verified=True!
    unverified_dir = tmp_path / "unverified_legacy_out"
    unverified_dir.mkdir(parents=True, exist_ok=True)
    legacy_report = {k: v for k, v in report.items() if k != "assistant_audio_sha256"}
    write_json_atomic(unverified_dir / "report.json", {scenario.scenario_id: legacy_report})
    legacy_summary = await run_duplex_mpe_evaluation(
        DuplexMpeEvalConfig(
            dataset_root=dataset_root,
            output_dir=unverified_dir,
            phase="score",
            addressing="explicit",
            max_scenarios=1,
            protocol="native",
        )
    )
    legacy_prov = legacy_summary["provenance"]
    assert isinstance(legacy_prov, dict)
    legacy_srv = legacy_prov["declared_server_config"]
    assert isinstance(legacy_srv, dict)
    assert legacy_srv["retained_audio_verified"] is False
    assert legacy_srv["retained_audio_verified_scenarios"] == 0


def test_silero_vad_optional_import_and_uninstalled_metadata_fallback(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import importlib.metadata as importlib_metadata

    import benchmarks.duplex_mpe.vad as vad_mod

    dataset_root = create_synthetic_duplex_mpe_dataset(tmp_path / "dataset")
    scenario = load_duplex_mpe_scenarios(dataset_root, addressing="explicit", max_scenarios=1)[0]

    real_version = importlib_metadata.version

    def fake_missing_version(distribution_name: str) -> str:
        if distribution_name == "silero-vad":
            raise importlib_metadata.PackageNotFoundError("silero-vad")
        return real_version(distribution_name)

    def fake_missing_import(module_name: str, package: str | None = None) -> object:
        if module_name == "silero_vad":
            raise ModuleNotFoundError("No module named 'silero_vad'", name="silero_vad")
        raise AssertionError(f"Unexpected dynamic import: {module_name}")

    monkeypatch.setattr(importlib_metadata, "version", fake_missing_version)
    monkeypatch.setattr(vad_mod.importlib, "import_module", fake_missing_import)

    assert silero_package_version() == SILERO_UNINSTALLED_SENTINEL
    stub_cfg = SileroActivityConfig().to_report_dict(custom_vad_injected=True)
    assert stub_cfg["silero_vad_version"] == SILERO_UNINSTALLED_SENTINEL
    assert stub_cfg["custom_vad_injected"] is True

    # Default detector without injected stub raises a clear actionable error when silero-vad is absent,
    # while a broken transitive dependency inside silero_vad propagates its own ModuleNotFoundError name!
    with pytest.raises(ModuleNotFoundError, match="pip install"):
        SileroActivityDetector(sample_rate_hz=16000, vad_model=None)

    def fake_broken_subdep_import(module_name: str, package: str | None = None) -> object:
        if module_name == "silero_vad":
            raise ModuleNotFoundError("No module named 'onnxruntime'", name="onnxruntime")
        raise AssertionError(f"Unexpected dynamic import: {module_name}")

    monkeypatch.setattr(vad_mod.importlib, "import_module", fake_broken_subdep_import)
    with pytest.raises(ModuleNotFoundError, match="onnxruntime"):
        SileroActivityDetector(sample_rate_hz=16000, vad_model=None)
    monkeypatch.setattr(vad_mod.importlib, "import_module", fake_missing_import)

    # Injected deterministic stub works end-to-end without silero-vad installed, while strict mode still rejects it!
    turns = [
        ScenarioTurnInput(
            turn_index=t.turn_index,
            event_type=t.event_type,
            speaker=t.speaker,
            waveform_samples=np.zeros(4000, dtype=np.float32),
        )
        for t in scenario.turns
    ]
    adapter = ClockedAudioAdapter(
        ScheduledReplyAdapter(
            input_sample_rate_hz=16000,
            output_sample_rate_hz=24000,
            voiced_intervals_seconds=[],
        )
    )
    stub_report = run_scenario_schedule(
        adapter,
        turns,
        vad_model=EnergyThresholdVadStub(),
        onset_deadline_seconds=0.5,
        maximum_response_seconds=1.0,
        target_endpoint_hangover_milliseconds=2000.0,
    ).to_report_dict()
    validate_scenario_reports({scenario.scenario_id: stub_report}, {scenario.scenario_id: scenario})
    with pytest.raises(ValueError, match="Invalid or unsupported silero_vad_version"):
        validate_scenario_reports(
            {scenario.scenario_id: stub_report},
            {scenario.scenario_id: scenario},
            strict_reference_vad_version=True,
        )


async def test_phased_generation_vs_scoring_environment_provenance_separation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import benchmarks.duplex_mpe.evaluate as eval_mod

    dataset_root = create_synthetic_duplex_mpe_dataset(tmp_path / "dataset")
    out_dir = tmp_path / "phased_prov_out"

    gen_snapshot: dict[str, JsonValue] = {
        "repository": {"commit": "111111111111", "branch": "gen-branch", "dirty": False},
        "host": {
            "platform": "Linux-gen-gpu-host",
            "cpu_model": "GenCPU",
            "logical_cpus": 128,
            "memory_total_kib": "1024",
        },
        "gpu": {
            "nvidia_smi_csv": "0, NVIDIA H20, GPU-gen, 97871, 535.0, P0, 500.0, 1980, 2619, 9.0",
            "cuda_visible_devices": "0",
            "torch_cuda_build": "13.0",
            "cudnn_version": 92501,
        },
        "packages": {"torch": "2.14.0+cu130", "vllm": "0.34.0", "silero-vad": "6.2.1"},
        "dependency_inventory": {"vllm": "0.34.0"},
        "dependency_freeze_sha256": "deadbeef" * 8,
    }
    score_snapshot: dict[str, JsonValue] = {
        "repository": {"commit": "222222222222", "branch": "score-branch", "dirty": True},
        "host": {
            "platform": "Linux-score-cpu-host",
            "cpu_model": "ScoreCPU",
            "logical_cpus": 16,
            "memory_total_kib": "512",
        },
        "gpu": {"nvidia_smi_csv": None, "cuda_visible_devices": None, "torch_cuda_build": None, "cudnn_version": None},
        "packages": {"torch": "2.14.0+cpu", "vllm": None, "silero-vad": None},
        "dependency_inventory": {"numpy": "2.0.0"},
        "dependency_freeze_sha256": "cafebabe" * 8,
    }

    async def handle_ws(websocket: websockets.ServerConnection) -> None:
        _ = json.loads(await websocket.recv())
        session_obj: dict[str, JsonValue] = {
            "id": "prov-gen-01",
            "model": "nvidia/personaplex-7b-v1",
            "turn_detection": None,
            "capabilities": {
                "implementation_level": "model_native_duplex",
                "supports_model_native_turn_policy": True,
                "supports_barge_in": False,
                "supports_playback_ack": False,
                "supports_audio_truncate": False,
                "supports_client_commit": False,
                "chunk_period_ms": 80,
            },
        }
        await websocket.send(json.dumps({"type": "session.created", "session": session_obj}))
        await websocket.send(json.dumps({"type": "session.updated", "session": session_obj}))
        async for raw_frame in websocket:
            msg = json.loads(raw_frame)
            if msg.get("type") == "session.close":
                await websocket.send(json.dumps({"type": "session.closed", "session_id": "prov-gen-01"}))
                return

    real_run_scenario = eval_mod.run_realtime_scenario

    async def fast_stub_run_scenario(
        scen: eval_mod.DuplexMpeScenarioRecord,
        *,
        dataset_root: Path,
        endpoint_spec: eval_mod.RealtimeEndpointSpec,
        output_audio_path: Path,
        vad_model: SileroVadModel | None = None,
        session_timeout_seconds: float,
        seed_mapping: dict[str, int],
    ) -> dict[str, JsonValue]:
        return await real_run_scenario(
            scen,
            dataset_root=dataset_root,
            endpoint_spec=endpoint_spec,
            output_audio_path=output_audio_path,
            vad_model=vad_model,
            session_timeout_seconds=session_timeout_seconds,
            seed_mapping=seed_mapping,
            realtime_pacing=False,
        )

    monkeypatch.setattr(eval_mod, "run_realtime_scenario", fast_stub_run_scenario)
    monkeypatch.setattr(eval_mod, "collect_runtime_environment_snapshot", lambda: gen_snapshot)

    async with websockets.serve(handle_ws, "127.0.0.1", 0) as server:
        port = server.sockets[0].getsockname()[1]
        # Verify default generation WITHOUT silero-vad fails BEFORE freezing run.json,
        # so installing silero-vad and retrying the SAME --output-dir succeeds cleanly!
        import importlib.metadata as importlib_metadata

        import benchmarks.duplex_mpe.vad as vad_mod

        real_meta_version = importlib_metadata.version
        real_import_mod = vad_mod.importlib.import_module
        monkeypatch.setattr(
            importlib_metadata,
            "version",
            lambda dist_name: (_ for _ in ()).throw(importlib_metadata.PackageNotFoundError(dist_name))
            if dist_name == "silero-vad"
            else real_meta_version(dist_name),
        )
        monkeypatch.setattr(
            vad_mod.importlib,
            "import_module",
            lambda mod_name, pkg=None: (_ for _ in ()).throw(
                ModuleNotFoundError("No module named 'silero_vad'", name="silero_vad")
            )
            if mod_name == "silero_vad"
            else real_import_mod(mod_name, pkg),
        )
        with pytest.raises(ModuleNotFoundError, match="pip install"):
            await run_duplex_mpe_evaluation(
                DuplexMpeEvalConfig(
                    dataset_root=dataset_root,
                    output_dir=out_dir,
                    phase="generate",
                    addressing="explicit",
                    max_scenarios=1,
                    base_url=f"http://127.0.0.1:{port}",
                    warmup_sessions=0,
                )
            )
        assert not (out_dir / "run.json").exists()

        # Simulate installing silero-vad==6.2.1 and retrying on the EXACT SAME out_dir via the default loader path!
        class FakeInstalledSileroModule:
            @staticmethod
            def load_silero_vad() -> EnergyThresholdVadStub:
                return EnergyThresholdVadStub()

        fake_silero_mod = FakeInstalledSileroModule()
        monkeypatch.setattr(
            importlib_metadata,
            "version",
            lambda dist_name: "6.2.1" if dist_name == "silero-vad" else real_meta_version(dist_name),
        )
        monkeypatch.setattr(
            vad_mod.importlib,
            "import_module",
            lambda mod_name, pkg=None: fake_silero_mod if mod_name == "silero_vad" else real_import_mod(mod_name, pkg),
        )

        # Reject strict generation when a custom vad_model_factory is injected, even if silero-vad==6.2.1 is installed!
        with pytest.raises(ValueError, match="not permitted with --strict-reference-vad-version"):
            await run_duplex_mpe_evaluation(
                DuplexMpeEvalConfig(
                    dataset_root=dataset_root,
                    output_dir=out_dir,
                    phase="generate",
                    addressing="explicit",
                    max_scenarios=1,
                    base_url=f"http://127.0.0.1:{port}",
                    warmup_sessions=0,
                    strict_reference_vad_version=True,
                ),
                vad_model_factory=EnergyThresholdVadStub,
            )
        assert not (out_dir / "run.json").exists()

        # Retry default generation (no vad_model_factory!) on the same out_dir — must succeed cleanly!
        gen_summary = await run_duplex_mpe_evaluation(
            DuplexMpeEvalConfig(
                dataset_root=dataset_root,
                output_dir=out_dir,
                phase="generate",
                addressing="explicit",
                max_scenarios=1,
                base_url=f"http://127.0.0.1:{port}",
                warmup_sessions=0,
                strict_reference_vad_version=True,
            )
        )
    assert gen_summary["status"] == "complete"
    saved_manifest = json.loads((out_dir / "run.json").read_text(encoding="utf-8"))
    assert saved_manifest["generation_environment"] == gen_snapshot

    # Now switch to a completely different CPU scoring host (`score_snapshot`) and run `--phase score`!
    monkeypatch.setattr(eval_mod, "collect_runtime_environment_snapshot", lambda: score_snapshot)
    score_summary = await run_duplex_mpe_evaluation(
        DuplexMpeEvalConfig(
            dataset_root=dataset_root,
            output_dir=out_dir,
            phase="score",
            addressing="explicit",
            max_scenarios=1,
        )
    )
    prov = score_summary["provenance"]
    assert isinstance(prov, dict)
    assert prov["environment_source"] == "saved_generation_manifest"
    # Top-level provenance preserves the generation host/GPU/commit rather than overwriting with the score host!
    assert prov["repository"] == gen_snapshot["repository"]
    assert prov["gpu"] == gen_snapshot["gpu"]
    assert prov["packages"] == gen_snapshot["packages"]
    assert prov["generation_environment"] == gen_snapshot
    assert prov["scoring_environment"] == score_snapshot


async def test_custom_vad_factory_isolation_resume_guards_and_partial_unknown_provenance(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import importlib.metadata as importlib_metadata

    import benchmarks.duplex_mpe.evaluate as eval_mod
    import benchmarks.duplex_mpe.vad as vad_mod

    dataset_root = create_synthetic_duplex_mpe_dataset(tmp_path / "dataset")
    real_meta_version = importlib_metadata.version
    monkeypatch.setattr(
        importlib_metadata,
        "version",
        lambda dist_name: "6.2.1" if dist_name == "silero-vad" else real_meta_version(dist_name),
    )

    class FakeInstalledSileroModule:
        @staticmethod
        def load_silero_vad() -> EnergyThresholdVadStub:
            return EnergyThresholdVadStub()

    fake_silero_mod = FakeInstalledSileroModule()
    real_import_mod = vad_mod.importlib.import_module
    monkeypatch.setattr(
        vad_mod.importlib,
        "import_module",
        lambda mod_name, pkg=None: fake_silero_mod if mod_name == "silero_vad" else real_import_mod(mod_name, pkg),
    )

    # 1. Per-session stateful VAD factory creates distinct instances across concurrent scenarios!
    created_instances: list[object] = []

    class StatefulIsolatedVadStub:
        def __init__(self) -> None:
            self.resets: int = 0
            self.frames_seen: int = 0
            created_instances.append(self)

        def reset_states(self) -> None:
            self.resets += 1
            self.frames_seen = 0

        def __call__(self, audio_tensor: torch.Tensor, sample_rate_hz: int) -> torch.Tensor:
            del audio_tensor, sample_rate_hz
            self.frames_seen += 1
            return torch.tensor([[0.05]], dtype=torch.float32)

    async def handle_ws(websocket: websockets.ServerConnection) -> None:
        _ = json.loads(await websocket.recv())
        session_obj: dict[str, JsonValue] = {
            "id": "iso-01",
            "model": "nvidia/personaplex-7b-v1",
            "turn_detection": None,
            "capabilities": {
                "implementation_level": "model_native_duplex",
                "supports_model_native_turn_policy": True,
                "supports_barge_in": False,
                "supports_playback_ack": False,
                "supports_audio_truncate": False,
                "supports_client_commit": False,
                "chunk_period_ms": 80,
            },
        }
        await websocket.send(json.dumps({"type": "session.created", "session": session_obj}))
        await websocket.send(json.dumps({"type": "session.updated", "session": session_obj}))
        async for raw_frame in websocket:
            msg = json.loads(raw_frame)
            if msg.get("type") == "session.close":
                await websocket.send(json.dumps({"type": "session.closed", "session_id": "iso-01"}))
                return

    real_run_scenario = eval_mod.run_realtime_scenario

    async def fast_run_scenario(
        scen: eval_mod.DuplexMpeScenarioRecord,
        *,
        dataset_root: Path,
        endpoint_spec: eval_mod.RealtimeEndpointSpec,
        output_audio_path: Path,
        vad_model: SileroVadModel | None = None,
        session_timeout_seconds: float,
        seed_mapping: dict[str, int],
    ) -> dict[str, JsonValue]:
        return await real_run_scenario(
            scen,
            dataset_root=dataset_root,
            endpoint_spec=endpoint_spec,
            output_audio_path=output_audio_path,
            vad_model=vad_model,
            session_timeout_seconds=session_timeout_seconds,
            seed_mapping=seed_mapping,
            realtime_pacing=False,
        )

    monkeypatch.setattr(eval_mod, "run_realtime_scenario", fast_run_scenario)

    inj_dir = tmp_path / "injected_partial_out"
    async with websockets.serve(handle_ws, "127.0.0.1", 0) as server:
        port = server.sockets[0].getsockname()[1]
        await run_duplex_mpe_evaluation(
            DuplexMpeEvalConfig(
                dataset_root=dataset_root,
                output_dir=inj_dir,
                phase="generate",
                addressing="explicit",
                max_scenarios=2,
                max_concurrency=2,
                base_url=f"http://127.0.0.1:{port}",
                warmup_sessions=1,
            ),
            vad_model_factory=StatefulIsolatedVadStub,
        )
        # 1 warmup + 2 concurrent measured scenarios -> 3 independent instances!
        assert len(created_instances) == 3
        assert len({id(inst) for inst in created_instances}) == 3

        # 2. Resuming an injected-VAD output directory (either with default VAD or another factory) MUST be rejected,
        # AND a no-op strict resume over custom-VAD reports must ALSO fail!
        with pytest.raises(ValueError, match="Run configuration"):
            await run_duplex_mpe_evaluation(
                DuplexMpeEvalConfig(
                    dataset_root=dataset_root,
                    output_dir=inj_dir,
                    phase="generate",
                    addressing="explicit",
                    max_scenarios=2,
                    max_concurrency=2,
                    base_url=f"http://127.0.0.1:{port}",
                    warmup_sessions=1,
                )
            )
        with pytest.raises(ValueError, match="Run configuration"):
            await run_duplex_mpe_evaluation(
                DuplexMpeEvalConfig(
                    dataset_root=dataset_root,
                    output_dir=inj_dir,
                    phase="generate",
                    addressing="explicit",
                    max_scenarios=2,
                    max_concurrency=2,
                    base_url=f"http://127.0.0.1:{port}",
                    warmup_sessions=1,
                ),
                vad_model_factory=StatefulIsolatedVadStub,
            )

        # 3. Partial resume when initial run.json lacks generation_environment MUST keep generation_environment=None
        # and record the resume snapshot in generation_environment_history!
        partial_dir = tmp_path / "partial_no_init_env"
        await run_duplex_mpe_evaluation(
            DuplexMpeEvalConfig(
                dataset_root=dataset_root,
                output_dir=partial_dir,
                phase="generate",
                addressing="explicit",
                max_scenarios=1,
                base_url=f"http://127.0.0.1:{port}",
                warmup_sessions=0,
            )
        )
        # Strip generation_environment from saved run.json (simulating a pre-provenance partial checkpoint)
        # and expand max_scenarios=1 -> missing scenario #2 via a compatible 2-scenario manifest!
        # Specifically, delete scenario #2 from a 2-scenario run while stripping generation_environment:
        two_dir = tmp_path / "two_partial_no_env"
        await run_duplex_mpe_evaluation(
            DuplexMpeEvalConfig(
                dataset_root=dataset_root,
                output_dir=two_dir,
                phase="generate",
                addressing="explicit",
                max_scenarios=2,
                base_url=f"http://127.0.0.1:{port}",
                warmup_sessions=0,
            )
        )
        saved_two_manifest = json.loads((two_dir / "run.json").read_text(encoding="utf-8"))
        saved_two_manifest.pop("generation_environment", None)
        saved_two_manifest.pop("generation_environment_history", None)
        write_json_atomic(two_dir / "run.json", saved_two_manifest)

        saved_two_reports = json.loads((two_dir / "report.json").read_text(encoding="utf-8"))
        second_sid = sorted(saved_two_reports.keys())[1]
        del saved_two_reports[second_sid]
        write_json_atomic(two_dir / "report.json", saved_two_reports)

        resume_env_snapshot: dict[str, JsonValue] = {
            "repository": {"commit": "333333333333", "branch": "resume-branch", "dirty": False},
            "host": {
                "platform": "Linux-resume-host",
                "cpu_model": "ResumeCPU",
                "logical_cpus": 64,
                "memory_total_kib": "2048",
            },
            "gpu": {
                "nvidia_smi_csv": None,
                "cuda_visible_devices": None,
                "torch_cuda_build": None,
                "cudnn_version": None,
            },
            "packages": {"torch": "2.14.0", "vllm": "0.34.0", "silero-vad": "6.2.1"},
            "dependency_inventory": {"vllm": "0.34.0"},
            "dependency_freeze_sha256": "feedface" * 8,
        }
        monkeypatch.setattr(eval_mod, "collect_runtime_environment_snapshot", lambda: resume_env_snapshot)

        await run_duplex_mpe_evaluation(
            DuplexMpeEvalConfig(
                dataset_root=dataset_root,
                output_dir=two_dir,
                phase="generate",
                addressing="explicit",
                max_scenarios=2,
                base_url=f"http://127.0.0.1:{port}",
                warmup_sessions=0,
            )
        )
        resumed_manifest = json.loads((two_dir / "run.json").read_text(encoding="utf-8"))
        assert resumed_manifest["generation_environment"] is None
        assert resumed_manifest["generation_environment_history"] == [resume_env_snapshot]

        scored_partial = await run_duplex_mpe_evaluation(
            DuplexMpeEvalConfig(
                dataset_root=dataset_root,
                output_dir=two_dir,
                phase="score",
                addressing="explicit",
                max_scenarios=2,
            )
        )
        partial_prov = scored_partial["provenance"]
        assert isinstance(partial_prov, dict)
        assert partial_prov["generation_environment"] is None
        assert partial_prov["environment_source"] == "partial_resume_without_initial_environment"
        assert partial_prov["generation_environment_history"] == [resume_env_snapshot]

        # 4. Importable silero_vad module WITHOUT distribution metadata (`silero_package_version() == "uninstalled"`)
        # MUST fail preflight BEFORE freezing `run.json` on default generation!
        unpacked_dir = tmp_path / "unpacked_no_dist_meta"
        monkeypatch.setattr(
            importlib_metadata,
            "version",
            lambda dist_name: (_ for _ in ()).throw(importlib_metadata.PackageNotFoundError(dist_name))
            if dist_name == "silero-vad"
            else real_meta_version(dist_name),
        )
        with pytest.raises(ValueError, match="distribution metadata >= 5.1"):
            await run_duplex_mpe_evaluation(
                DuplexMpeEvalConfig(
                    dataset_root=dataset_root,
                    output_dir=unpacked_dir,
                    phase="generate",
                    addressing="explicit",
                    max_scenarios=1,
                    base_url=f"http://127.0.0.1:{port}",
                    warmup_sessions=0,
                )
            )
        assert not (unpacked_dir / "run.json").exists()
        monkeypatch.setattr(
            importlib_metadata,
            "version",
            lambda dist_name: "6.2.1" if dist_name == "silero-vad" else real_meta_version(dist_name),
        )

        # 5. Saved reports with `custom_vad_injected=True` MUST block no-op and partial generation resume
        # EVEN IF `custom_vad_injected` was stripped from `run.json`!
        stripped_flag_manifest = json.loads((inj_dir / "run.json").read_text(encoding="utf-8"))
        stripped_flag_manifest.pop("custom_vad_injected", None)
        write_json_atomic(inj_dir / "run.json", stripped_flag_manifest)
        with pytest.raises(ValueError, match="custom-VAD report"):
            await run_duplex_mpe_evaluation(
                DuplexMpeEvalConfig(
                    dataset_root=dataset_root,
                    output_dir=inj_dir,
                    phase="generate",
                    addressing="explicit",
                    max_scenarios=2,
                    max_concurrency=2,
                    base_url=f"http://127.0.0.1:{port}",
                    warmup_sessions=1,
                )
            )
        # Also delete 1 of the 2 custom-VAD reports to verify partial resume is equally blocked:
        inj_reports = json.loads((inj_dir / "report.json").read_text(encoding="utf-8"))
        del inj_reports[sorted(inj_reports.keys())[1]]
        write_json_atomic(inj_dir / "report.json", inj_reports)
        with pytest.raises(ValueError, match="custom-VAD report"):
            await run_duplex_mpe_evaluation(
                DuplexMpeEvalConfig(
                    dataset_root=dataset_root,
                    output_dir=inj_dir,
                    phase="generate",
                    addressing="explicit",
                    max_scenarios=2,
                    max_concurrency=2,
                    base_url=f"http://127.0.0.1:{port}",
                    warmup_sessions=1,
                )
            )

        # 6. Contradictory `run.json` (`custom_vad_injected=True`) vs `report.json` (`custom_vad_injected` absent)
        # MUST be rejected during offline `--phase score` (both strict and non-strict)!
        contradictory_manifest = json.loads((two_dir / "run.json").read_text(encoding="utf-8"))
        contradictory_manifest["custom_vad_injected"] = True
        write_json_atomic(two_dir / "run.json", contradictory_manifest)
        with pytest.raises(ValueError, match="not permitted with --strict-reference-vad-version"):
            await run_duplex_mpe_evaluation(
                DuplexMpeEvalConfig(
                    dataset_root=dataset_root,
                    output_dir=two_dir,
                    phase="score",
                    addressing="explicit",
                    max_scenarios=2,
                    strict_reference_vad_version=True,
                )
            )
        with pytest.raises(ValueError, match="custom_vad_injected mismatch"):
            await run_duplex_mpe_evaluation(
                DuplexMpeEvalConfig(
                    dataset_root=dataset_root,
                    output_dir=two_dir,
                    phase="score",
                    addressing="explicit",
                    max_scenarios=2,
                )
            )
        # Restore valid custom_vad_injected=False so we can test silero_vad_version mismatch & no-op env validation:
        contradictory_manifest["custom_vad_injected"] = False
        write_json_atomic(two_dir / "run.json", contradictory_manifest)

        # 7. Contradictory `run.json` (`silero_vad_version="5.1"`) vs `report.json` (`silero_vad_version="6.2.1"`)
        # MUST be rejected in both strict and non-strict `--phase score`, AND contradictory report version MUST
        # fail partial resume before generating additional reports!
        version_mismatch_manifest = json.loads((two_dir / "run.json").read_text(encoding="utf-8"))
        version_mismatch_manifest["silero_vad_version"] = "5.1"
        write_json_atomic(two_dir / "run.json", version_mismatch_manifest)
        with pytest.raises(ValueError, match="silero_vad_version"):
            await run_duplex_mpe_evaluation(
                DuplexMpeEvalConfig(
                    dataset_root=dataset_root,
                    output_dir=two_dir,
                    phase="score",
                    addressing="explicit",
                    max_scenarios=2,
                    strict_reference_vad_version=True,
                )
            )
        with pytest.raises(ValueError, match="silero_vad_version mismatch"):
            await run_duplex_mpe_evaluation(
                DuplexMpeEvalConfig(
                    dataset_root=dataset_root,
                    output_dir=two_dir,
                    phase="score",
                    addressing="explicit",
                    max_scenarios=2,
                )
            )
        # Restore manifest silero_vad_version="6.2.1", corrupt saved report #1 to "5.1" and delete report #2 ->
        # partial resume MUST reject the report/manifest version mismatch before generating!
        version_mismatch_manifest["silero_vad_version"] = "6.2.1"
        write_json_atomic(two_dir / "run.json", version_mismatch_manifest)
        full_two_reports_text = (two_dir / "report.json").read_text(encoding="utf-8")
        partial_version_reports = json.loads(full_two_reports_text)
        first_sid, second_sid_again = sorted(partial_version_reports.keys())
        del partial_version_reports[second_sid_again]
        partial_version_reports[first_sid]["aed_config"]["silero_vad_version"] = "5.1"
        write_json_atomic(two_dir / "report.json", partial_version_reports)
        with pytest.raises(ValueError, match="silero_vad_version"):
            await run_duplex_mpe_evaluation(
                DuplexMpeEvalConfig(
                    dataset_root=dataset_root,
                    output_dir=two_dir,
                    phase="generate",
                    addressing="explicit",
                    max_scenarios=2,
                    base_url=f"http://127.0.0.1:{port}",
                    warmup_sessions=0,
                )
            )
        (two_dir / "report.json").write_text(full_two_reports_text, encoding="utf-8")

        # 8. Strict offline scoring (`--strict-reference-vad-version --phase score`) of valid 6.2.1 reports
        # MUST succeed even when `silero-vad` is NOT installed on the scoring host!
        monkeypatch.setattr(
            importlib_metadata,
            "version",
            lambda dist_name: (_ for _ in ()).throw(importlib_metadata.PackageNotFoundError(dist_name))
            if dist_name == "silero-vad"
            else real_meta_version(dist_name),
        )
        strict_offline_no_pkg = await run_duplex_mpe_evaluation(
            DuplexMpeEvalConfig(
                dataset_root=dataset_root,
                output_dir=two_dir,
                phase="score",
                addressing="explicit",
                max_scenarios=2,
                strict_reference_vad_version=True,
            )
        )
        assert strict_offline_no_pkg["status"] == "complete"
        monkeypatch.setattr(
            importlib_metadata,
            "version",
            lambda dist_name: "6.2.1" if dist_name == "silero-vad" else real_meta_version(dist_name),
        )

        # 9. Completed no-op `--phase generate` resume MUST still validate `generation_environment` and
        # `generation_environment_history` in `run.json` rather than silently bypassing malformed snapshots!
        bad_env_manifest = json.loads((two_dir / "run.json").read_text(encoding="utf-8"))
        bad_env_manifest["generation_environment"] = 42
        write_json_atomic(two_dir / "run.json", bad_env_manifest)
        with pytest.raises(ValueError, match="Malformed generation_environment"):
            await run_duplex_mpe_evaluation(
                DuplexMpeEvalConfig(
                    dataset_root=dataset_root,
                    output_dir=two_dir,
                    phase="generate",
                    addressing="explicit",
                    max_scenarios=2,
                    base_url=f"http://127.0.0.1:{port}",
                    warmup_sessions=0,
                )
            )
        bad_env_manifest["generation_environment"] = None
        bad_env_manifest["generation_environment_history"] = "not-a-list"
        write_json_atomic(two_dir / "run.json", bad_env_manifest)
        with pytest.raises(ValueError, match="Malformed generation_environment_history"):
            await run_duplex_mpe_evaluation(
                DuplexMpeEvalConfig(
                    dataset_root=dataset_root,
                    output_dir=two_dir,
                    phase="generate",
                    addressing="explicit",
                    max_scenarios=2,
                    base_url=f"http://127.0.0.1:{port}",
                    warmup_sessions=0,
                )
            )
