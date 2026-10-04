# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""P5-02 real HTTP WAV/raw/SSE with inline, vendored reference audio."""

import json
import os
import time
import uuid
from pathlib import Path

import numpy as np
import pytest

from tests.e2e.zonos2.audio_checks import check_audio, check_lifecycle, data_uri, decode_sse, decode_wav, read_trace
from tests.e2e.zonos2.process import speech_server
from tests.e2e.zonos2.runtime import reference_audio
from tests.helpers.mark import hardware_test

pytestmark = [pytest.mark.tts]


@pytest.mark.advanced_model
@hardware_test(res={"cuda": ["H100", "B200"]}, num_cards=1)
def test_real_http_nonstreaming_and_streaming(tmp_path):
    directory = Path(os.environ.get("ZONOS2_TEST_OUTPUT_DIR", str(tmp_path))) / f"online-{uuid.uuid4().hex[:8]}"
    reference = data_uri(reference_audio().read_bytes())
    reports = []
    with speech_server(directory) as client:
        for index, mode in enumerate(("wav", "raw", "sse")):
            seed = 211 + index
            body = {
                "model": "zonos2-e2e",
                "input": "This sentence should sound like the reference speaker.",
                "language": "English",
                "ref_audio": reference,
                "seed": seed,
                "response_format": "wav" if mode == "wav" else "pcm",
                "sample_rate": 44100,
            }
            if mode != "wav":
                body.update(stream=True, stream_format="audio" if mode == "raw" else "sse")
            pieces = []
            with client.stream("POST", "/v1/audio/speech", json=body) as response:
                assert response.status_code == 200, response.read().decode(errors="replace")
                for piece in response.iter_bytes():
                    if piece:
                        pieces.append(piece)
            content = b"".join(pieces)
            chunks = len(pieces)
            if mode == "wav":
                audio, sr = decode_wav(content)
            else:
                if mode == "sse":
                    assert "text/event-stream" in response.headers["content-type"]
                    content, chunks = decode_sse(content)
                else:
                    assert chunks > 1, "Raw HTTP audio was not streamed"
                assert len(content) % 2 == 0
                audio, sr = np.frombuffer(content, dtype="<i2").astype(np.float32) / 32768, 44100
            for _ in range(100):
                rows = read_trace(directory / "trace")
                finishes = [row for row in rows if row["kind"] == "finish" and row["seed"] == seed]
                try:
                    assert len(finishes) == 1
                    life = check_lifecycle(rows, finishes[0]["request"])
                    break
                except AssertionError:
                    time.sleep(0.05)
            else:
                raise AssertionError(f"Missing completion/cleanup for HTTP seed {seed}")
            report = check_audio(audio, sr, frames=life["decoded_samples"] // 512)
            report.update(mode=mode, chunks=chunks, seed=seed, lifecycle=life)
            reports.append(report)
            (directory / f"{mode}.bin").write_bytes(content)
            print("REAL_HTTP_PASS", mode, report, flush=True)
        # Ensure invalid conditioning receives a client error without generation.
        invalid = client.post(
            "/v1/audio/speech", json={"model": "zonos2-e2e", "input": "Test", "extra_params": {"emotion_cfg_scale": 2}}
        )
        assert invalid.status_code == 400
        capped = client.post(
            "/v1/audio/speech",
            json={
                "model": "zonos2-e2e",
                "input": "A sentence deliberately constrained by a tiny codec budget.",
                "max_new_tokens": 16,
                "seed": 42,
                "response_format": "wav",
            },
        )
        assert capped.status_code >= 400
        assert "incomplete" in capped.text

    (directory / "summary.json").write_text(
        json.dumps(
            {"status": "pass", "real_weights": True, "reference_source": "vendored data URI", "cases": reports},
            indent=2,
        )
    )
