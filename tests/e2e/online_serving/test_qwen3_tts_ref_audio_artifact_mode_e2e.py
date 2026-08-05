# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""E2E regression for Qwen3-TTS ref_audio artifact mode collision (#5049).

Reproduces the issue's exact sequence on a live server: an
``x_vector_only_mode=true`` request followed by an ICL
(``x_vector_only_mode=false`` + ``ref_text``) request using the **same**
``ref_audio``.

Before the fix, the ref_audio artifact readiness was mode-agnostic: the
x-vector-only request (which caches a speaker embedding but no ``ref_code``)
marked the artifact "ready", so the ICL request took the artifact-only serving
path (``ref_audio`` stripped before dispatch), found no ``ref_code`` and no
audio to recompute from, and raised in ``build_prompt_embeds`` ->
``EngineDeadError`` — taking the whole server down rather than failing the one
request.

After the fix (readiness keyed by ``(artifact_key, x_vector_only)``), the ICL
request no longer matches the x-vector artifact, so it sends ``ref_audio`` and
the worker computes ``ref_code``. Both requests succeed and the server stays up.
"""

import os

os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

import pytest

from tests.helpers.mark import hardware_test
from tests.helpers.media import load_test_audio_data_url
from tests.helpers.runtime import OmniServerParams
from tests.helpers.stage_config import get_deploy_config_path

MODEL = "Qwen/Qwen3-TTS-12Hz-0.6B-Base"
DEFAULT_AUDIO_SPEECH_TIMEOUT_S = 180.0

# Vendored under tests/assets/qwen3_tts/clone_2.wav (see test_qwen3_tts_base.py).
REF_AUDIO_URL = load_test_audio_data_url("qwen3_tts/clone_2.wav")
REF_TEXT = "Okay. Yeah. I resent you. I love you. I respect you. But you know what? You blew it! And thanks to you."
INPUT_TEXT = "The weather is nice today, perfect for a walk in the park."

tts_server_params = [
    pytest.param(
        OmniServerParams(
            model=MODEL,
            stage_config_path=get_deploy_config_path("qwen3_tts.yaml"),
            server_args=["--trust-remote-code"],
        ),
        id="async_chunk",
    )
]


@pytest.mark.advanced_model
@pytest.mark.core_model
@pytest.mark.tts
@hardware_test(res={"cuda": "L4"}, num_cards=1)
@pytest.mark.parametrize("omni_server", tts_server_params, indirect=True)
def test_xvector_then_icl_same_ref_audio_keeps_engine_alive(omni_server, openai_client) -> None:
    """Regression for #5049: x-vector-only then ICL on the same ref_audio.

    Both requests must return audio and the engine must stay alive (pre-fix the
    second request killed EngineCore).
    """
    base_request = {
        "model": omni_server.model,
        "input": INPUT_TEXT,
        "stream": False,
        "timeout": DEFAULT_AUDIO_SPEECH_TIMEOUT_S,
        "response_format": "wav",
        "task_type": "Base",
        "ref_audio": REF_AUDIO_URL,
        # Assert non-empty audio came back (not just HTTP 200).
        "min_audio_bytes": 1,
    }

    # 1) x-vector-only request: caches a speaker embedding (no ref_code) and, on
    #    completion, marks the ref_audio artifact "ready".
    openai_client.send_audio_speech_request({**base_request, "x_vector_only_mode": True})

    # 2) ICL request with the SAME ref_audio. Pre-fix this reused the x-vector
    #    artifact via the artifact-only path, hit the missing ref_code, and
    #    killed EngineCore (#5049). It must now succeed with audio.
    openai_client.send_audio_speech_request({**base_request, "x_vector_only_mode": False, "ref_text": REF_TEXT})

    # 3) The server must still be serving after the ICL request (pre-fix this
    #    would fail with EngineDeadError / connection refused).
    openai_client.send_audio_speech_request({**base_request, "x_vector_only_mode": True})
