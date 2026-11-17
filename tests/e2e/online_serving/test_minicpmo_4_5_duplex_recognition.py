# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Real three-stage AV duplex regression: 180 s, 12 scenes, no sliding window."""

import argparse
import asyncio
import hashlib
import importlib.util
import json
import uuid
from pathlib import Path

import pytest

from tests.e2e.online_serving.helpers.minicpmo_4_5_duplex import SERVER_PARAMS, realtime_url, resolve_ref_audio
from tests.e2e.online_serving.helpers.minicpmo_duplex_recognition import evaluate_session, load_manifest, unpack_inputs
from tests.helpers.mark import hardware_test

pytestmark = [pytest.mark.omni, pytest.mark.full_model]


@hardware_test(res={"cuda": ["H100", "B200"]}, num_cards=1)
@pytest.mark.parametrize("omni_server", SERVER_PARAMS, indirect=True)
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
    output = repo / "tests/e2e/online_serving/results/minicpmo_duplex_recognition" / uuid.uuid4().hex
    args = argparse.Namespace(
        output_dir=str(output),
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
        tail_silence_s=8,
        require_audio=True,
    )
    result = asyncio.run(demo.run_demo(args))
    events = [json.loads(line) for line in (output / "events.jsonl").read_text().splitlines()]
    report = evaluate_session(result["transcript"], events, manifest["scenes"])
    report["fixture_archive_sha256"] = manifest["archive_sha256"]
    report["reference_audio_sha256"] = manifest["reference_audio_sha256"]
    report["scope"] = "12 scene descriptions plus response completion; not all-question accuracy or token parity"
    (output / "recognition.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    assert result["ok"], result
    assert result["video_frames_sent"] == 180, result
    assert report["passed"], report
