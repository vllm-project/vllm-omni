# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""E2E online tests for MiniMax Music 3 text-to-music.

This is a music model on the speech endpoint, so the request shape differs
from every other TTS model here: ``input`` carries the lyrics, ``instructions``
carries the caption that decides genre and arrangement, and both are required.

Weekly rather than per-PR: the checkpoint is 27 GB across five components and a
single request occupies two decode rows, because classifier-free guidance is
mandatory for this model. Rejection cases live in
``tests/dfx/reliability/invalid_param_test/test_invalid_audio_speech.py``.
"""

import os

os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

import pytest

from tests.helpers.mark import hardware_test
from tests.helpers.runtime import OmniServerParams
from tests.helpers.stage_config import get_deploy_config_path

MODEL = "MiniMaxAI/MiniMax-Music3"
DEFAULT_AUDIO_SPEECH_TIMEOUT_S = 600.0

# 25 frames per second, so 250 frames is ten seconds and 750 is thirty. The
# byte floors sit below the true payload (32 kHz stereo, 16-bit) to leave room
# for the model ending a song before the cap.
_FRAMES_10S = 250
_FRAMES_30S = 750
_MIN_BYTES_10S = 500_000
_MIN_BYTES_30S = 1_500_000

LYRICS = (
    "[Verse]\nWalking down the empty street at midnight\n"
    "Streetlights flicker like a broken dream\n"
    "[Chorus]\nAnd I keep on walking\nTill the morning finds me"
)

CAPTION = (
    "A melancholic lo-fi hip-hop track at 85 BPM in F minor: mellow Rhodes piano riff, "
    "soft vinyl crackle, dusty boom-bap drums with a laid-back swing, warm upright bass. "
    "Intimate bedroom production, gentle tape saturation, no bright cymbals."
)


tts_server_params = [
    pytest.param(
        OmniServerParams(
            model=MODEL,
            stage_config_path=get_deploy_config_path("minimax_music3.yaml"),
            server_args=["--trust-remote-code", "--disable-log-stats"],
        ),
        id="minimax_music3",
    )
]


@pytest.mark.slow
@pytest.mark.tts
@hardware_test(res={"cuda": "H100"}, num_cards={"cuda": 1})
@pytest.mark.parametrize("omni_server", tts_server_params, indirect=True)
def test_text_to_music_001(omni_server, openai_client) -> None:
    """Baseline: lyrics plus caption in, one ten-second song out.

    Deploy Setting: minimax_music3.yaml
    Input Modal: text (lyrics) + text (caption)
    Output Modal: audio, 32 kHz stereo
    Input Setting: stream=False, seed pinned, 250 frames
    """
    request_config = {
        "model": omni_server.model,
        "input": LYRICS,
        "instructions": CAPTION,
        "seed": 1,
        "max_new_tokens": _FRAMES_10S,
        "stream": False,
        "response_format": "wav",
        "timeout": DEFAULT_AUDIO_SPEECH_TIMEOUT_S,
        "min_audio_bytes": _MIN_BYTES_10S,
    }
    openai_client.send_audio_speech_request(request_config)


@pytest.mark.slow
@pytest.mark.tts
@hardware_test(res={"cuda": "H100"}, num_cards={"cuda": 1})
@pytest.mark.parametrize("omni_server", tts_server_params, indirect=True)
def test_text_to_music_multi_window_002(omni_server, openai_client) -> None:
    """A song long enough to span several acoustic windows.

    The acoustic stage decodes overlapping 200-frame windows on a 100-frame
    hop and crops the overlap back off, so a request that fits in one window
    never exercises the stitching. 750 frames covers seven of them.

    Deploy Setting: minimax_music3.yaml
    Input Setting: stream=False, seed pinned, 750 frames
    """
    request_config = {
        "model": omni_server.model,
        "input": LYRICS,
        "instructions": CAPTION,
        "seed": 1,
        "max_new_tokens": _FRAMES_30S,
        "stream": False,
        "response_format": "wav",
        "timeout": DEFAULT_AUDIO_SPEECH_TIMEOUT_S,
        "min_audio_bytes": _MIN_BYTES_30S,
    }
    openai_client.send_audio_speech_request(request_config)
