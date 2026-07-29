# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""E2E online serving tests for Vevo2.

The deploy yaml is sync (``async_chunk: false``) so streaming is not
exercised here -- batch-only is the MVP scope per #3391.

Reference audio is fetched once per session and inlined as a
``data:audio/wav;base64,...`` URL so the server never reaches the
network for fixtures (CI rule from
``docs/contributing/model/adding_tts_model.md``).

One ``OmniServerParams`` set per file, per the same guide -- the
``omni_server`` fixture is module-scoped and a second id forces a
mid-module restart.
"""

from __future__ import annotations

import base64
import os
import urllib.request

os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
os.environ["VLLM_TEST_CLEAN_GPU_MEMORY"] = "0"

import pytest

from tests.helpers.mark import hardware_test
from tests.helpers.runtime import OmniServerParams
from tests.helpers.stage_config import get_deploy_config_path


# See the offline suite: the published repo lacks a root config.json, so set
# VEVO2_MODEL_PATH to a local checkpoint initialised by init_vevo2_checkpoint.py.
# When it is absent we skip the whole suite with a clear message instead of
# starting a server that can't load the model.
def _resolve_vevo2_model() -> str:
    model = os.environ.get("VEVO2_MODEL_PATH")
    if model and os.path.isdir(model) and os.path.exists(os.path.join(model, "config.json")):
        return model
    reason = (
        "Vevo2 e2e tests require an init_vevo2_checkpoint.py-prepared local "
        "checkpoint (the bare RMSnow/Vevo2 repo ships no root config.json). "
        "Download the repo, run init_vevo2_checkpoint.py on it, and set "
        "VEVO2_MODEL_PATH=/path/to/Vevo2."
    )
    if model:
        reason = (
            f"VEVO2_MODEL_PATH={model!r} is not an initialized checkpoint dir (missing root config.json). " + reason
        )
    pytest.skip(reason, allow_module_level=True)


MODEL = _resolve_vevo2_model()
REF_AUDIO_URL = "https://raw.githubusercontent.com/open-mmlab/Amphion/main/models/vc/vevo/wav/arabic_male.wav"
REF_AUDIO_TRANSCRIPT = "Philip stood undecided, his ears strained to catch the slightest sound."


@pytest.fixture(scope="session")
def ref_audio_data_url() -> str:
    """Fetch the upstream reference clip and return as a base64 data URL.

    Hard-fails if the network fetch fails so a broken path does not
    silently mask regressions. ``VEVO2_SKIP_ON_NET_FAIL=1`` lets
    air-gapped CI opt into skipping; ``VEVO2_LOCAL_REF=/path/to/wav``
    bypasses the network entirely.
    """
    local = os.environ.get("VEVO2_LOCAL_REF")
    if local and os.path.exists(local):
        with open(local, "rb") as f:
            data = f.read()
        return f"data:audio/wav;base64,{base64.b64encode(data).decode('ascii')}"
    try:
        with urllib.request.urlopen(REF_AUDIO_URL, timeout=30) as resp:
            data = resp.read()
    except Exception as e:
        msg = f"Cannot fetch upstream reference clip {REF_AUDIO_URL}: {e}"
        if os.environ.get("VEVO2_SKIP_ON_NET_FAIL"):
            pytest.skip(msg)
        pytest.fail(msg)
    if not data:
        pytest.fail(f"Reference clip empty: {REF_AUDIO_URL}")
    return f"data:audio/wav;base64,{base64.b64encode(data).decode('ascii')}"


tts_server_params = [
    pytest.param(
        OmniServerParams(
            model=MODEL,
            stage_config_path=get_deploy_config_path("vevo2.yaml"),
            server_args=["--disable-log-stats"],
        ),
        id="vevo2",
    )
]


@pytest.mark.slow
@pytest.mark.tts
@hardware_test(res={"cuda": "L4"}, num_cards=1)
@pytest.mark.parametrize("omni_server", tts_server_params, indirect=True)
def test_vevo2_basic(omni_server, openai_client, ref_audio_data_url) -> None:
    """Basic English TTS with timbre reference -> 24 kHz WAV."""
    request_config = {
        "model": omni_server.model,
        # Avoid the coined word "Vevo2" in the *spoken* text: Whisper mishears
        # it ("Vavor 2" / "Fable 2"), which drags the transcript-similarity
        # check below its 0.9 threshold even though the audio is correct. Use
        # ordinary words and let ASR escalation rule out a whisper-small mishear.
        "input": "Hello, this is a short voice cloning demo for testing.",
        "stream": False,
        "response_format": "wav",
        "ref_audio": ref_audio_data_url,
        "ref_text": REF_AUDIO_TRANSCRIPT,
        "transcript_escalation_model": "large-v3",
    }

    openai_client.send_audio_speech_request(request_config)


@pytest.mark.slow
@pytest.mark.tts
@hardware_test(res={"cuda": "L4"}, num_cards=1)
@pytest.mark.parametrize("omni_server", tts_server_params, indirect=True)
def test_vevo2_chinese(omni_server, openai_client, ref_audio_data_url) -> None:
    """Chinese TTS with timbre reference -> 24 kHz WAV."""
    request_config = {
        "model": omni_server.model,
        "input": "今天天气很好，我们一起去公园散步吧。",
        "stream": False,
        "response_format": "wav",
        "ref_audio": ref_audio_data_url,
        "ref_text": REF_AUDIO_TRANSCRIPT,
        "transcript_escalation_model": "large-v3",
    }

    openai_client.send_audio_speech_request(request_config)


@pytest.mark.slow
@pytest.mark.tts
@hardware_test(res={"cuda": "L4"}, num_cards=1)
@pytest.mark.parametrize("omni_server", tts_server_params, indirect=True)
def test_vevo2_no_ref_text(omni_server, openai_client, ref_audio_data_url) -> None:
    """``ref_text`` is recommended but not strictly required."""
    request_config = {
        "model": omni_server.model,
        "input": "This is a simple text to speech example without a reference transcript.",
        "stream": False,
        "response_format": "wav",
        "ref_audio": ref_audio_data_url,
        "transcript_escalation_model": "large-v3",
    }

    openai_client.send_audio_speech_request(request_config)


@pytest.mark.slow
@pytest.mark.tts
@hardware_test(res={"cuda": "L4"}, num_cards=1)
@pytest.mark.parametrize("omni_server", tts_server_params, indirect=True)
def test_vevo2_missing_ref_audio_rejected(omni_server, openai_client) -> None:
    """``ref_audio`` is required by Vevo2; the server should reject the request."""
    import httpx

    request_config = {
        "model": omni_server.model,
        "input": "This request should fail because ref_audio is missing.",
        "stream": False,
        "response_format": "wav",
        "voice": "default",
    }

    base_url = openai_client.base_url.rstrip("/")
    with httpx.Client(timeout=60) as client:
        resp = client.post(
            f"{base_url}/v1/audio/speech",
            json=request_config,
            headers={"Authorization": f"Bearer {openai_client.client.api_key}"},
        )
    assert resp.status_code >= 400, f"Expected 4xx, got {resp.status_code}: {resp.text[:200]}"
