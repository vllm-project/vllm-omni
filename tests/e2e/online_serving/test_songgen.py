# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
E2E online tests for SongGen via the /v1/audio/speech endpoint.

SongGen maps the OpenAI speech contract onto text-to-song generation:
  - ``input``        -> song lyrics (required)
  - ``instructions`` -> music style / genre description (optional)
  - ``ref_audio``    -> reference voice for timbre conditioning (optional)

The server resolves ``ref_audio`` to a waveform and forwards it to the model as
``ref_voice_array`` (the key the model reads); a minimal non-streaming WAV case
is enough to exercise the full serving path end to end. These tests are gated
behind ``full_model`` / ``tts`` and run only in the model CI lane.
"""

import os

os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

import pytest

from tests.helpers.mark import hardware_test
from tests.helpers.runtime import OmniServerParams
from tests.helpers.stage_config import get_deploy_config_path

pytestmark = [pytest.mark.full_model, pytest.mark.tts]

MODEL = "LiuZH-19/SongGen_mixed_pro"
LYRICS = "Under the moonlight, we dance through the night, stars above shining bright."
DESCRIPTION = "dreamy pop ballad with piano and strings"

# A 16 kHz song clip is far larger than this floor; the check only guards
# against an empty or truncated response, not audio quality.
_MIN_AUDIO_BYTES = 20_000

songgen_server_params = [
    pytest.param(
        OmniServerParams(
            model=MODEL,
            stage_config_path=get_deploy_config_path("songgen.yaml"),
            server_args=["--disable-log-stats"],
        ),
        id="songgen",
    )
]


@hardware_test(res={"cuda": "H100"}, num_cards=1)
@pytest.mark.parametrize("omni_server", songgen_server_params, indirect=True)
def test_text_to_song_001(omni_server, openai_client) -> None:
    """
    Text-to-song via /v1/audio/speech (lyrics + style description).
    Deploy Setting: default yaml
    Input Modal: text (lyrics) + instructions (style description)
    Output Modal: audio (16 kHz, WAV)
    Input Setting: stream=False
    Datasets: single request
    """
    request_config = {
        "model": omni_server.model,
        "input": LYRICS,
        "instructions": DESCRIPTION,
        "stream": False,
        "response_format": "wav",
        "min_audio_bytes": _MIN_AUDIO_BYTES,
    }

    openai_client.send_audio_speech_request(request_config)
