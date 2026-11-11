# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Qualification against a running consumer recipe; opt in with FASTH3_TEST_URL."""

import hashlib
import io
import json
import os
import time
from pathlib import Path

import av
import numpy as np
import pytest

from tests.helpers.mark import hardware_marks
from tests.helpers.runtime import OnlineOmniClient

pytestmark = [pytest.mark.advanced_model, pytest.mark.diffusion, *hardware_marks(res={"cuda": ["L4", "RTX5090"]})]


def _decode_media(body: bytes, width: int, height: int) -> dict[str, int | str]:
    video_hash, audio_hash = hashlib.sha256(), hashlib.sha256()
    with av.open(io.BytesIO(body)) as media:
        stream = media.streams.video[0]
        assert stream.codec_context.name == "h264"
        assert (stream.width, stream.height, float(stream.average_rate)) == (width, height, 24)
        frames = 0
        for frame in media.decode(video=0):
            video_hash.update(frame.to_ndarray(format="rgb24").tobytes())
            frames += 1
        assert frames == 124
    with av.open(io.BytesIO(body)) as media:
        stream = media.streams.audio[0]
        assert stream.sample_rate == 32000 and len(stream.layout.channels) == 2
        samples, peak = 0, 0.0
        for frame in media.decode(audio=0):
            values = frame.to_ndarray()
            assert np.isfinite(values).all()
            samples += frame.samples
            peak = max(peak, float(np.abs(values).max()))
            audio_hash.update(values.tobytes())
        assert samples > 0 and peak > 0, "audio is empty or silent"
    return {
        "frames": frames,
        "audio_samples": samples,
        "video_sha256": video_hash.hexdigest(),
        "audio_sha256": audio_hash.hexdigest(),
        "mp4_sha256": hashlib.sha256(body).hexdigest(),
    }


@pytest.mark.skipif(not os.environ.get("FASTH3_TEST_URL"), reason="start a recipe and set FASTH3_TEST_URL")
@pytest.mark.parametrize("width,height", [(832, 480), (1344, 768)])
def test_fasth3_omni_repeated_http_requests(width, height, tmp_path):
    client = OnlineOmniClient(port=8000)
    client.base_url = os.environ["FASTH3_TEST_URL"].rstrip("/")
    reference = Path(os.environ["FASTH3_TEST_REFERENCE"]).read_bytes()
    output = Path(os.environ.get("FASTH3_TEST_OUTPUT", str(tmp_path))) / f"{width}x{height}"
    output.mkdir(parents=True, exist_ok=True)
    prompt = os.environ.get(
        "FASTH3_TEST_PROMPT",
        "A family enjoys dinner together around a wooden table. They smile and talk naturally. "
        "The camera slowly pushes toward the table. Warm indoor lighting, quiet conversation, soft clinking of dishes.",
    )
    results = []
    for phase in ("first", "repeat"):
        start = time.perf_counter()
        response = client.send_videos_sync_http_request(
            {
                "data": {"prompt": prompt, "size": f"{width}x{height}", "seconds": "5", "seed": "1234"},
                "files": {"input_reference": ("reference.png", reference, "image/png")},
                "timeout": 3600,
            }
        )[0]
        elapsed = time.perf_counter() - start
        assert response.status_code == 200, response.error_message
        (output / f"{phase}.mp4").write_bytes(response.content)
        results.append({"phase": phase, "http_seconds": elapsed, **_decode_media(response.content, width, height)})
        (output / "results.json").write_text(json.dumps(results, indent=2) + "\n")
    identical = {key: results[0][key] == results[1][key] for key in ("video_sha256", "audio_sha256", "mp4_sha256")}
    (output / "repeatability.json").write_text(json.dumps(identical, indent=2) + "\n")
    # Lossless-change qualification is stricter than a capacity smoke.
    if os.environ.get("FASTH3_REQUIRE_IDENTICAL") == "1":
        assert all(identical.values()), f"Cold/warm output differs: {identical}"
