# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""``POST /v1/audio/translations`` error-case validation.

Verifies that a non-ASR model returns the appropriate HTTP error code
when used with the translation endpoint.

From ``tests/``::

    pytest -s -v dfx/reliability/invalid_param_test/test_invalid_audio_translation.py
"""

from __future__ import annotations

import io
import os

os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

import openai
import pytest

from tests.helpers.mark import hardware_marks
from tests.helpers.runtime import OmniServer, OmniServerParams, OnlineOmniClient

pytestmark = [pytest.mark.slow, pytest.mark.omni]

_ASR_PARAMS = [
    pytest.param(
        OmniServerParams(
            model="openai/whisper-small",
            server_args=["--enforce-eager"],
            use_omni=False,
        ),
        id="whisper_small",
        marks=hardware_marks(res={"cuda": "H100"}),
    ),
]

_NON_ASR_PARAMS = [
    pytest.param(
        OmniServerParams(
            model="JackFram/llama-68m",
            server_args=["--enforce-eager"],
            use_omni=False,
        ),
        id="llama_68m",
        marks=hardware_marks(res={"cuda": "H100"}),
    ),
]


def _get_audio_path(name: str = "mary_had_lamb") -> str:
    from vllm.assets.audio import AudioAsset

    return str(AudioAsset(name).get_local_path())


@pytest.mark.parametrize("omni_server", _ASR_PARAMS, indirect=True)
def test_translation_invalid_audio(
    omni_server: OmniServer,
    openai_client: OnlineOmniClient,
) -> None:
    """Corrupted audio data must be rejected with an error."""
    invalid_audio = io.BytesIO(b"not a valid audio file")
    invalid_audio.name = "invalid.wav"
    with pytest.raises((openai.BadRequestError, openai.APIStatusError)):
        openai_client.client.audio.translations.create(
            model=omni_server.model,
            file=invalid_audio,
            temperature=0.0,
        )


@pytest.mark.parametrize("omni_server_function", _NON_ASR_PARAMS, indirect=True)
def test_translation_non_asr_model(
    omni_server_function: OmniServer,
    openai_client_function: OnlineOmniClient,
) -> None:
    """Translation on a non-ASR model must return NotFoundError."""
    with open(_get_audio_path(), "rb") as f:
        with pytest.raises(openai.NotFoundError):
            openai_client_function.client.audio.translations.create(
                model=omni_server_function.model,
                file=f,
                temperature=0.0,
            )
