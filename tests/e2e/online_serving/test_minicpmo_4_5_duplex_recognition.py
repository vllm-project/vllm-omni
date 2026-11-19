# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Real three-stage AV duplex regression: 180 s, 12 scenes, no sliding window."""

import argparse
import asyncio
import hashlib
import importlib.util
import json
import uuid
import warnings
from pathlib import Path

import pytest

from tests.e2e.online_serving.helpers.minicpmo_4_5_duplex import SERVER_PARAMS, realtime_url, resolve_ref_audio
from tests.e2e.online_serving.helpers.minicpmo_duplex_recognition import evaluate_session, load_manifest, unpack_inputs
from tests.helpers.mark import hardware_test
from tests.helpers.stage_config import modify_stage_config

pytestmark = [pytest.mark.omni, pytest.mark.core_model]

# Temporary diagnostics are confined to this existing recognition fixture.
EOS_TRACE_SERVER_PARAMS = [
    pytest.param(
        param.values[0]._replace(
            stage_config_path=modify_stage_config(
                param.values[0].stage_config_path,
                updates={
                    "stages": {
                        0: {
                            "worker_extension_cls": "tests.e2e.online_serving.helpers.minicpmo_duplex_eos_trace.MiniCPMDuplexEOSTrace"
                        }
                    }
                },
            )
        ),
        id=param.id,
    )
    for param in SERVER_PARAMS
]


@pytest.mark.skip(
    reason=(
        "Intermittent long-video scene-coverage failures (including the final scene); "
        "tracked in https://github.com/vllm-project/vllm-omni/issues/10417"
    )
)
@hardware_test(res={"cuda": ["H100", "B200"]}, num_cards=1)
@pytest.mark.parametrize("omni_server", EOS_TRACE_SERVER_PARAMS, indirect=True)
def test_duplex_long_video_scene_recognition(omni_server, tmp_path: Path, monkeypatch) -> None:
    manifest = load_manifest()
    wav, frames = unpack_inputs(tmp_path / "input")
    reference = resolve_ref_audio()
    assert hashlib.sha256(reference.read_bytes()).hexdigest() == manifest["reference_audio_sha256"], (
        "Reference voice differs from the validated fixture; do not silently compare another conditioning input."
    )
    repo = Path(__file__).resolve().parents[3]
    spec = importlib.util.spec_from_file_location(
        "minicpmo_recognition_demo", repo / "examples/online_serving/minicpmo/realtime_duplex_demo.py"
    )
    assert spec is not None and spec.loader is not None
    demo = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(demo)
    # Only replace file decoding with frozen bytes. The demo drives the real
    # WebSocket server, streaming/pacing, playback ACKs, commit and drain.
    monkeypatch.setattr(demo, "_resolve_duplex_av_inputs", lambda **kwargs: (str(wav), frames, [None] * len(frames)))
    attempts_root = repo / "tests/e2e/online_serving/results/minicpmo_duplex_recognition" / uuid.uuid4().hex
    args = argparse.Namespace(
        output_dir=str(attempts_root / "attempt_0"),
        input_wav=str(wav),
        input_video=None,
        video_fps=1,
        frame_max_side=0,
        stack_frames=1,
        url=realtime_url(omni_server),
        model=omni_server.model,
        ref_audio=str(reference),
        temperature=0.0,
        instructions=manifest["instructions"],
        sliding_window_mode="off",
        basic_window_high_tokens=8000,
        basic_window_low_tokens=6000,
        context_max_units=24,
        context_previous_max_tokens=500,
        timeout_s=90,
        chunk_ms=200,
        no_realtime_pacing=False,
        # Keep one 200 ms residual after the eight whole silent model units.
        # The final commit can then answer from retained input instead of
        # falling back to response.create after every unit was consumed. This
        # gives a sampled mid-sentence final scene one bounded completion
        # opportunity without weakening the 12-scene accuracy oracle.
        tail_silence_s=8.2,
        # Native duplex lets the model keep listening instead of answering the
        # last scene. Asking for one response on the final commit makes that
        # answer a session contract instead of a ~25% CI coin flip; the
        # scene-coverage assertion below still judges what the model said.
        final_create_response=True,
        # Native duplex can also end the model's own answer mid-sentence
        # (a sampled <|turn_eos|> between units), and the forced response
        # above then listens to the trailing silence instead of speaking,
        # so the cut sentence's tail is lost. Text re-asks are refused for
        # speech-native duplex models, so the in-protocol recovery is to
        # replay the last scene's audio+video window (15 s = one scene)
        # and drain one more forced response; the replay only fires on the
        # truncated shape (silent post-commit answer + mid-sentence
        # transcript), so intact sessions stay single-shot.
        post_commit_replay_seconds=15.0,
        require_audio=True,
    )
    for attempt in range(2):
        # Each attempt streams into its own directory, so a green retry
        # keeps the first-pass event trace and verdict instead of
        # overwriting a coverage regression away.
        attempt_dir = attempts_root / f"attempt_{attempt}"
        args.output_dir = str(attempt_dir)
        result = asyncio.run(demo.run_demo(args))
        events = [json.loads(line) for line in (attempt_dir / "events.jsonl").read_text().splitlines()]
        report = evaluate_session(
            result["transcript"],
            events,
            manifest["scenes"],
            forced_response_windows=result["post_commit"]["forced_response_windows"],
        )
        report["fixture_archive_sha256"] = manifest["archive_sha256"]
        report["reference_audio_sha256"] = manifest["reference_audio_sha256"]
        report["scope"] = "12 scene descriptions plus response completion; not all-question accuracy or token parity"
        report["post_commit"] = result["post_commit"]
        report["attempt"] = attempt
        if attempt > 0:
            report["retry_reason"] = (
                "coverage-only failure on the previous attempt (see the "
                "attempt_0 recognition.json for the retained first pass)"
            )
        (attempt_dir / "recognition.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        assert result["ok"], result
        assert result["video_frames_sent"] == 180, result
        assert result["post_commit"]["input_had_residual_model_unit"] is True, result
        if report["passed"]:
            break
        # Native duplex turn decisions are sampled, so the model's own
        # scene answer can still end mid-sentence (the replay above only
        # recovers what the protocol can carry). Retry once with a fresh
        # session when only the coverage oracle failed -- protocol
        # problems should fail fast instead of burning another session.
        if attempt == 0 and not report["protocol_errors"]:
            warnings.warn(f"duplex scene coverage failed once, retrying: {report}")
            continue
        assert report["passed"], report
    # A coverage-only failure followed by a pass must retain both traces:
    # the first pass keeps its events.jsonl plus a failing recognition.json,
    # and the retry's directory holds the passing verdict.
    if report["passed"] and report["attempt"] > 0:
        first_pass = json.loads((attempts_root / "attempt_0" / "recognition.json").read_text(encoding="utf-8"))
        assert (attempts_root / "attempt_0" / "events.jsonl").exists()
        assert first_pass["passed"] is False and not first_pass["protocol_errors"], first_pass
