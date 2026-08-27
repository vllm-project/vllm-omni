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
import io
import os
import urllib.request
import wave

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

# Vevo2 emits 24 kHz mono. Every prompt in this file is one short sentence, so a
# handful of seconds is the expected output; tens of seconds means the AR loop
# never stopped. The offline suite pins this with its own MAX_REASONABLE_DURATION_S
# and without an equivalent here a runaway generation would still satisfy the
# transcript-similarity checks.
SAMPLE_RATE = 24000
MAX_REASONABLE_DURATION_S = 60.0


def _audio_bytes(responses) -> bytes:
    """Pull the WAV payload out of a ``send_audio_speech_request`` result."""
    items = responses if isinstance(responses, list) else [responses]
    assert items, "Expected at least one response"
    audio = getattr(items[0], "audio_bytes", None)
    assert audio, "Expected WAV bytes on the response"
    return audio


def _assert_reasonable_duration(audio: bytes) -> None:
    """Assert the WAV is real 24 kHz audio of a plausible length."""
    with wave.open(io.BytesIO(audio)) as wav:
        frames = wav.getnframes()
        rate = wav.getframerate()
    assert rate == SAMPLE_RATE, f"Expected {SAMPLE_RATE} Hz, got {rate}"
    duration_s = frames / float(rate)
    assert duration_s > 0.0, "Synthesized zero frames of audio"
    assert duration_s <= MAX_REASONABLE_DURATION_S, (
        f"Synthesized {duration_s:.0f}s for a single short prompt "
        f"(> {MAX_REASONABLE_DURATION_S}s) — likely a runaway generation bug"
    )


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

    responses = openai_client.send_audio_speech_request(request_config)
    _assert_reasonable_duration(_audio_bytes(responses))


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

    responses = openai_client.send_audio_speech_request(request_config)
    _assert_reasonable_duration(_audio_bytes(responses))


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

    responses = openai_client.send_audio_speech_request(request_config)
    _assert_reasonable_duration(_audio_bytes(responses))


@pytest.mark.slow
@pytest.mark.tts
@hardware_test(res={"cuda": "L4"}, num_cards=1)
@pytest.mark.parametrize("omni_server", tts_server_params, indirect=True)
def test_vevo2_missing_ref_audio_rejected(omni_server, openai_client) -> None:
    """``ref_audio`` is required by Vevo2; the request must be rejected with a 400.

    Pinned to exactly 400 rather than "any status >= 400": a 5xx would mean the
    server crashed on the missing reference instead of validating it, which is
    a different (and worse) outcome that must not pass as a rejection.
    """
    request_config = {
        "model": omni_server.model,
        "input": "This request should fail because ref_audio is missing.",
        "stream": False,
        "response_format": "wav",
        "voice": "default",
        "status_code": 400,
        "err_message": "ref_audio",
    }

    openai_client.send_audio_speech_request(request_config)


@pytest.mark.slow
@pytest.mark.tts
@hardware_test(res={"cuda": "L4"}, num_cards=1)
@pytest.mark.parametrize("omni_server", tts_server_params, indirect=True)
def test_vevo2_same_seed_is_reproducible(omni_server, openai_client, ref_audio_data_url) -> None:
    """Two identical seeded requests must return identical audio.

    Vevo2 samples inside Amphion's ``inference_ar_and_fm``, which reads its
    seed from ``additional_information``; the ``SamplingParams`` the dummy AR
    scheduler carries never reach it. An adapter that dropped ``seed`` on the
    floor would leave the offline path reproducible and the online path not,
    which is exactly the regression this pins.
    """

    def synthesize(seed: int) -> bytes:
        responses = openai_client.send_audio_speech_request(
            {
                "model": omni_server.model,
                # Plain dictionary words only: Whisper mishears "seed" as
                # "seat", which drops transcript similarity to 0.90 and trips
                # the shared helper's threshold -- the same reason the tests
                # above avoid the coined word "Vevo2" in spoken text.
                "input": "The morning train arrives at the station on time.",
                "stream": False,
                "response_format": "wav",
                "ref_audio": ref_audio_data_url,
                "ref_text": REF_AUDIO_TRANSCRIPT,
                "transcript_escalation_model": "large-v3",
                "seed": seed,
            }
        )
        audio = _audio_bytes(responses)
        _assert_reasonable_duration(audio)
        return audio

    first = synthesize(1234)
    second = synthesize(1234)
    assert first == second, (
        f"Same seed returned different audio ({len(first)} vs {len(second)} bytes); "
        f"the request seed is not reaching the model"
    )

    different = synthesize(4321)
    assert different != first, "A different seed must change the output; seed appears to be ignored"
