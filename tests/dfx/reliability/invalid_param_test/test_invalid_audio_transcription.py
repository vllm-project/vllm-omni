# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""``POST /v1/audio/transcriptions`` error-case validation.

Verifies that invalid inputs (bad language, corrupted audio, wrong model)
return appropriate HTTP error codes.

From ``tests/``::

    pytest -s -v dfx/reliability/invalid_param_test/test_invalid_audio_transcription.py
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
def test_transcription_invalid_language(
    omni_server: OmniServer,
    openai_client: OnlineOmniClient,
) -> None:
    """Invalid language code ``hh`` must be rejected with BadRequestError."""
    with open(_get_audio_path(), "rb") as f:
        with pytest.raises(openai.BadRequestError):
            openai_client.client.audio.transcriptions.create(
                model=omni_server.model,
                file=f,
                language="hh",
                response_format="text",
                temperature=0.0,
            )


@pytest.mark.parametrize("omni_server", _ASR_PARAMS, indirect=True)
def test_transcription_invalid_audio(
    omni_server: OmniServer,
    openai_client: OnlineOmniClient,
) -> None:
    """Corrupted audio data must be rejected with an error."""
    invalid_audio = io.BytesIO(b"not a valid audio file")
    invalid_audio.name = "invalid.wav"
    with pytest.raises((openai.BadRequestError, openai.APIStatusError)):
        openai_client.client.audio.transcriptions.create(
            model=omni_server.model,
            file=invalid_audio,
            language="en",
            response_format="text",
            temperature=0.0,
        )


@pytest.mark.parametrize("omni_server_function", _NON_ASR_PARAMS, indirect=True)
def test_transcription_model_not_found(
    omni_server_function: OmniServer,
    openai_client_function: OnlineOmniClient,
) -> None:
    """Transcription on a non-ASR model must return NotFoundError."""
    with open(_get_audio_path(), "rb") as f:
        with pytest.raises(openai.NotFoundError):
            openai_client_function.client.audio.transcriptions.create(
                model=omni_server_function.model,
                file=f,
                language="en",
                response_format="text",
                temperature=0.0,
            )
