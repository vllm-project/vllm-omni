# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""E2E online tests for Audex (Nemotron-Labs-Audex-2B) text-to-audio.

Covers /v1/audio/speech through the ``audex_tta`` deployment:
the ``audex_tta`` adapter builds the caption prompt and the serving layer
injects the RVQ phase contract plus the official default guidance
(cfg_scale 3.0) server-side.
"""

import os

os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

import pytest

from tests.helpers.mark import hardware_test
from tests.helpers.runtime import OmniServerParams
from tests.helpers.stage_config import get_deploy_config_path

pytestmark = [pytest.mark.slow, pytest.mark.tts]

MODEL = "nvidia/Nemotron-Labs-Audex-2B"
CAPTION = "Heavy rain falling on a tin roof."
SAMPLE_RATE = 16_000

audex_tta_server_params = [
    pytest.param(
        OmniServerParams(
            model=MODEL,
            stage_config_path=get_deploy_config_path("audex_tta.yaml"),
            server_args=["--trust-remote-code"],
        ),
        id="audex_tta",
    )
]


@hardware_test(res={"cuda": "H100"}, num_cards=1)
@pytest.mark.parametrize("omni_server", audex_tta_server_params, indirect=True)
def test_audex_tta_online_smoke(omni_server, openai_client, run_level) -> None:
    """A caption returns a valid finite non-silent WAV at the default cfg 3.0."""
    import io
    import wave

    import numpy as np
    import requests

    url = f"{openai_client.base_url.rstrip('/')}/v1/audio/speech"
    resp = requests.post(
        url,
        json={"model": omni_server.model, "input": CAPTION, "response_format": "wav"},
        timeout=600,
    )
    assert resp.status_code == 200, resp.text[:200]
    assert resp.content[:4] == b"RIFF", "not a WAV response"

    with wave.open(io.BytesIO(resp.content)) as wav:
        assert wav.getframerate() == SAMPLE_RATE, f"unexpected sample rate {wav.getframerate()}"
        frames = wav.readframes(wav.getnframes())
        duration_s = wav.getnframes() / wav.getframerate()

    audio = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
    assert audio.size > 0, "empty audio body"
    assert np.isfinite(audio).all(), "audio contains NaN/Inf"

    if run_level in {"advanced_model", "full_model"}:
        # The decode cap is 500 frames = 10 s at 50 fps.
        assert 0.1 <= duration_s <= 10.5, f"unexpected duration={duration_s:.3f}s"
        rms = float(np.sqrt(np.mean(np.square(audio))))
        assert rms > 1e-4, f"audio is near-silent (rms={rms})"


@hardware_test(res={"cuda": "H100"}, num_cards=1)
@pytest.mark.parametrize("omni_server", audex_tta_server_params, indirect=True)
def test_audex_tta_online_policies(omni_server, openai_client) -> None:
    """Malformed guidance and unsupported knobs are rejected with 400."""
    import requests

    url = f"{openai_client.base_url.rstrip('/')}/v1/audio/speech"
    for bad_payload, needle in (
        ({"input": CAPTION, "extra_params": {"cfg_scale": 99}}, "cfg_scale"),
        ({"input": CAPTION, "extra_params": {"cfg_scale": "big"}}, "cfg_scale"),
        ({"input": CAPTION, "extra_params": {"tta_rvq": {}}}, "tta_rvq"),
        ({"input": "   "}, "caption"),
        ({"input": CAPTION, "voice": "alloy"}, "voice"),
        ({"input": CAPTION, "ref_audio": "https://example.com/ref.wav"}, "reference audio"),
    ):
        resp = requests.post(url, json={"model": omni_server.model, **bad_payload}, timeout=120)
        assert resp.status_code == 400, f"{bad_payload} -> {resp.status_code}: {resp.text[:200]}"
        assert needle in resp.text, f"{bad_payload} error missing {needle!r}: {resp.text[:200]}"
