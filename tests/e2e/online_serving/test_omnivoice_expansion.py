# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
E2E Online tests for OmniVoice TTS model via /v1/audio/speech endpoint.

Tests verify that the OmniVoice model generates valid audio when
accessed through the standard OpenAI-compatible speech API.
"""

import os

os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

import pytest

from tests.helpers.mark import hardware_test
from tests.helpers.media import get_asset_path, load_test_audio_data_url
from tests.helpers.runtime import OmniServerParams
from tests.helpers.stage_config import get_deploy_config_path
from vllm_omni.entrypoints.openai.serving_speech import _DEFAULT_VOICE_NAME

try:
    from transformers import HiggsAudioV2TokenizerModel  # noqa: F401

    _HAS_VOICE_CLONE = True
except ImportError:
    _HAS_VOICE_CLONE = False

pytestmark = [pytest.mark.slow, pytest.mark.tts]

MODEL = "k2-fsa/OmniVoice"

STAGE_CONFIG = get_deploy_config_path("omnivoice.yaml")
EXTRA_ARGS = [
    "--trust-remote-code",
    "--disable-log-stats",
]
TEST_PARAMS = [
    OmniServerParams(
        model=MODEL,
        stage_config_path=STAGE_CONFIG,
        server_args=EXTRA_ARGS,
    )
]

# Lower this in ``request_config`` via ``min_audio_bytes`` if a run produces legitimately short WAVs.
_DEFAULT_MIN_AUDIO_BYTES = 5000


REF_AUDIO_URL = load_test_audio_data_url("qwen3_tts/clone_2.wav")
REF_TEXT = "Okay. Yeah. I resent you. I love you. I respect you. But you know what? You blew it! And thanks to you."


def get_prompt(prompt_type="text"):
    prompts = {
        "text": "The weather is nice today, perfect for a walk in the park.",
    }
    return prompts.get(prompt_type, prompts["text"])


@pytest.mark.parametrize("omni_server", TEST_PARAMS, indirect=True)
class TestOmniVoiceTTS:
    """E2E tests for OmniVoice TTS model."""

    @hardware_test(res={"cuda": "L4"}, num_cards=1)
    def test_speech_auto_voice(self, omni_server, openai_client) -> None:
        """Test auto voice TTS generation (text only, no reference audio)."""
        request_config = {
            "model": omni_server.model,
            "input": get_prompt("text"),
            "response_format": "wav",
            "timeout": 180.0,
            "min_audio_bytes": _DEFAULT_MIN_AUDIO_BYTES,
        }
        openai_client.send_audio_speech_request(request_config)


@pytest.mark.parametrize("omni_server", TEST_PARAMS, indirect=True)
class TestOmniVoiceSeed:
    """E2E tests for OmniVoice Seed Params."""

    @hardware_test(res={"cuda": "L4"}, num_cards=1)
    def test_speech_auto_voice_seed_deterministic(self, omni_server, openai_client) -> None:
        cfg = {
            "model": omni_server.model,
            "input": get_prompt("text"),
            "response_format": "wav",
            "seed": 42,
            "min_audio_bytes": _DEFAULT_MIN_AUDIO_BYTES,
        }

        r1 = openai_client.send_audio_speech_request(cfg)[0]
        r2 = openai_client.send_audio_speech_request(cfg)[0]
        assert r1.audio_bytes == r2.audio_bytes

    @hardware_test(res={"cuda": "L4"}, num_cards=1)
    def test_speech_auto_voice_seed_non_deterministic(self, omni_server, openai_client) -> None:
        cfg = {
            "model": omni_server.model,
            "input": get_prompt("text"),
            "response_format": "wav",
            "min_audio_bytes": _DEFAULT_MIN_AUDIO_BYTES,
        }

        cfg1 = {**cfg, "seed": 42}
        cfg2 = {**cfg, "seed": 43}

        r1 = openai_client.send_audio_speech_request(cfg1)[0]
        r2 = openai_client.send_audio_speech_request(cfg2)[0]
        assert r1.audio_bytes != r2.audio_bytes


@pytest.mark.skipif(not _HAS_VOICE_CLONE, reason="Voice cloning requires transformers>=5.3.0")
@pytest.mark.parametrize("omni_server", TEST_PARAMS, indirect=True)
class TestOmniVoiceVoiceCloning:
    """E2E tests for OmniVoice voice cloning functionality."""

    @hardware_test(res={"cuda": "L4"}, num_cards=1)
    def test_voice_clone_ref_audio_only(self, omni_server, openai_client) -> None:
        """Test voice cloning with ref_audio only (x_vector mode)."""
        request_config = {
            "model": omni_server.model,
            "input": get_prompt("text"),
            "ref_audio": REF_AUDIO_URL,
            "response_format": "wav",
            "timeout": 180.0,
            "min_audio_bytes": _DEFAULT_MIN_AUDIO_BYTES,
        }
        openai_client.send_audio_speech_request(request_config)

    @hardware_test(res={"cuda": "L4"}, num_cards=1)
    def test_voice_clone_ref_audio_and_text(self, omni_server, openai_client) -> None:
        """Test voice cloning with ref_audio and ref_text (in-context mode)."""
        request_config = {
            "model": omni_server.model,
            "input": get_prompt("text"),
            "ref_audio": REF_AUDIO_URL,
            "ref_text": REF_TEXT,
            "response_format": "wav",
            "timeout": 180.0,
            "min_audio_bytes": _DEFAULT_MIN_AUDIO_BYTES,
        }
        openai_client.send_audio_speech_request(request_config)


@hardware_test(res={"cuda": "L4"}, num_cards=1)
@pytest.mark.parametrize("omni_server", TEST_PARAMS, indirect=True)
@pytest.mark.parametrize("voice", [_DEFAULT_VOICE_NAME, {"id": _DEFAULT_VOICE_NAME}])
def test_speech_with_voice_param_accepted(omni_server, openai_client, voice) -> None:
    """The default voice should be accepted as both a plain string and a
    VoiceID dict, even when the model has no uploaded speakers."""
    request_config = {
        "model": omni_server.model,
        "input": get_prompt("text"),
        "voice": voice,
        "response_format": "wav",
        "timeout": 180.0,
        "min_audio_bytes": _DEFAULT_MIN_AUDIO_BYTES,
    }
    openai_client.send_audio_speech_request(request_config)


@hardware_test(res={"cuda": "L4"}, num_cards=1)
@pytest.mark.parametrize("omni_server", TEST_PARAMS, indirect=True)
def test_unknown_voice_rejected(omni_server, openai_client) -> None:
    """An unrecognized voice name should be rejected with a 400."""
    request_config = {
        "model": omni_server.model,
        "input": get_prompt("text"),
        "voice": "nonexistent_voice",
        "status_code": 400,
        "err_message": "Invalid voice",
    }
    openai_client.send_audio_speech_request(request_config)


@hardware_test(res={"cuda": "L4"}, num_cards=1)
@pytest.mark.parametrize("omni_server", TEST_PARAMS, indirect=True)
def test_voice_request_before_and_after_clone_registration(omni_server, openai_client) -> None:
    """The default voice should always be listed and usable, even after
    a different cloned voice is registered."""
    speech_config = {
        "model": omni_server.model,
        "input": get_prompt("text"),
        "voice": _DEFAULT_VOICE_NAME,
    }

    # 1) "default" should appear in the voices list before any uploads
    voices = openai_client.send_audio_voices_list_http_request()[0]
    assert _DEFAULT_VOICE_NAME in voices.json_body["voices"]

    # 2) Speech with the default voice should succeed
    openai_client.send_audio_speech_request(speech_config)

    # 3) Register a cloned voice under a different name
    register_config = {
        "data": {"name": "bar", "consent": "test-consent"},
        "files": {
            "audio_sample": (
                "clone_2.wav",
                get_asset_path("qwen3_tts/clone_2.wav").read_bytes(),
                "audio/wav",
            ),
        },
    }
    openai_client.send_audio_voices_create_http_request(register_config)

    # 4) "default" should still be listed alongside the new voice
    voices = openai_client.send_audio_voices_list_http_request()[0]
    assert _DEFAULT_VOICE_NAME in voices.json_body["voices"]
    assert "bar" in voices.json_body["voices"]

    # 5) Speech with the default voice should still succeed
    openai_client.send_audio_speech_request(speech_config)


@hardware_test(res={"cuda": "L4"}, num_cards=1)
@pytest.mark.parametrize("omni_server", TEST_PARAMS, indirect=True)
def test_registered_default_voice_overrides_placeholder(omni_server, openai_client) -> None:
    """When a real voice is uploaded under the default name, it should be
    used as a cloned voice; after deletion, the placeholder behavior resumes."""
    speech_config = {
        "model": omni_server.model,
        "input": get_prompt("text"),
        "voice": _DEFAULT_VOICE_NAME,
    }

    # 1) Placeholder default works before any registration
    openai_client.send_audio_speech_request(speech_config)

    # 2) Register a real voice under the default name
    register_config = {
        "data": {"name": _DEFAULT_VOICE_NAME, "consent": "test-consent"},
        "files": {
            "audio_sample": (
                "clone_2.wav",
                get_asset_path("qwen3_tts/clone_2.wav").read_bytes(),
                "audio/wav",
            ),
        },
    }
    openai_client.send_audio_voices_create_http_request(register_config)

    # 3) After creating the voice with the default name, it's visible as an uploaded_voice
    voices = openai_client.send_audio_voices_list_http_request()[0]
    assert _DEFAULT_VOICE_NAME in voices.json_body["voices"]
    uploaded_names = [v["name"] for v in voices.json_body["uploaded_voices"]]
    assert _DEFAULT_VOICE_NAME in uploaded_names
    openai_client.send_audio_speech_request(speech_config)

    # 4) Delete the uploaded default voice
    openai_client.send_audio_voices_delete_http_request(
        {"name": _DEFAULT_VOICE_NAME},
    )

    # 5) Ensure the placeholder behavior is consistent, but it's not an uploaded voice now
    voices = openai_client.send_audio_voices_list_http_request()[0]
    assert _DEFAULT_VOICE_NAME in voices.json_body["voices"]
    uploaded_names = [v["name"] for v in voices.json_body["uploaded_voices"]]
    assert _DEFAULT_VOICE_NAME not in uploaded_names
    openai_client.send_audio_speech_request(speech_config)
