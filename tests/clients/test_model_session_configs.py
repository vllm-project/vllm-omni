# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Model duplex session presets under ``vllm_omni.clients.<model>``."""

import subprocess
import sys

import pytest

from vllm_omni.clients.duplex import AudioFormat
from vllm_omni.clients.minicpmo_4_5 import (
    create_duplex_session_config as create_minicpmo45_session_config,
)
from vllm_omni.clients.personaplex import (
    create_duplex_session_config as create_personaplex_session_config,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def test_minicpmo45_session_config_matches_native_duplex_deployment():
    config = create_minicpmo45_session_config(ref_audio="data:audio/wav;base64,AAA=", temperature=0.0)
    payload = config.to_session_payload(model="openbmb/MiniCPM-o-4_5")
    assert payload["ref_audio"] == "data:audio/wav;base64,AAA="
    assert payload["overlap_policy"] == "listen_only"
    assert payload["playback_commit_policy"] == "ack_only"
    assert payload["extra_body"]["native_duplex"] is True
    assert payload["extra_body"]["auto_response"] is True
    assert payload["temperature"] == 0.0


def test_personaplex_session_config_matches_deployment():
    config = create_personaplex_session_config(voice="NATF2.pt", persona="You are calm.")
    assert config.input_audio == AudioFormat("pcm_f32le", 24_000)
    payload = config.to_session_payload(model="nvidia/personaplex-7b-v1")
    assert payload["input_audio_format"] == "pcm_f32le"
    assert payload["voice"] == "NATF2.pt"
    assert payload["instructions"] == "You are calm."


@pytest.mark.parametrize(
    "duplex_package",
    [
        "vllm_omni.model_executor.models.minicpmo_4_5.duplex",
        "vllm_omni.model_executor.models.personaplex.duplex",
    ],
)
def test_model_duplex_packages_stay_clear_of_client_library(duplex_package: str):
    """Model duplex plugin packages are server-side: no vllm_omni.clients."""
    script = f"""
import sys

import {duplex_package}

loaded = [
    name
    for name in sys.modules
    if name == "vllm_omni.clients" or name.startswith("vllm_omni.clients.")
]
if loaded:
    raise SystemExit("model duplex package loaded the client library: " + ", ".join(loaded))
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert result.returncode == 0, result.stderr
