# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from __future__ import annotations

import asyncio
import os
from pathlib import Path

import pytest
from huggingface_hub import snapshot_download

from tests.e2e.online_serving.nemotron_voicechat_realtime_duplex import parse_args, run
from tests.helpers.mark import hardware_test
from tests.helpers.runtime import OmniServerParams, get_model_prefix
from tests.helpers.stage_config import get_deploy_config_path

MODEL = os.environ.get("VLLM_TEST_NEMOTRON_VOICECHAT_MODEL", "nvidia/NVIDIA-NemotronLabs-VoiceChat-11B")
DEPLOY_CONFIG = get_deploy_config_path("nemotron_labs_voicechat_duplex.yaml")
TOKENIZER = os.environ.get("VLLM_TEST_NEMOTRON_VOICECHAT_LLM_PATH")
pytestmark = [pytest.mark.full_model, pytest.mark.omni]


@hardware_test(res={"cuda": "H100"}, num_cards=1)
@pytest.mark.parametrize(
    "omni_server",
    [
        pytest.param(
            OmniServerParams(
                model=MODEL,
                stage_config_path=DEPLOY_CONFIG,
                env_dict={"NEMOTRON_VOICECHAT_LLM_PATH": TOKENIZER} if TOKENIZER else None,
            ),
            id="native-duplex",
        )
    ],
    indirect=True,
)
def test_native_duplex_turn_taking_streams_model_audio(omni_server, tmp_path: Path) -> None:
    model_prefix = get_model_prefix()
    root = Path(model_prefix) / MODEL if model_prefix else Path(MODEL)
    if not root.is_dir():
        root = Path(snapshot_download(MODEL, local_files_only=True))
    args = parse_args(
        (
            f"--url ws://{omni_server.host}:{omni_server.port}/v1/realtime --model {omni_server.model} "
            f"--input-wav {root / 'turn_taking.wav'} --input-channel 0 --max-frames 190 "
            f"--minimum-audio-chunks 48 --minimum-audio-rms 0.001 --no-realtime --timeout-s 300 "
            f"--output-dir {tmp_path / 'native_duplex'}"
        ).split()
    )
    result = asyncio.run(run(args))
    assert result["ok"] is True
    assert result["input_frames"] == 190
    assert result["event_counts"]["response.speak"] > 0
    assert result["audio_bytes"] >= result["input_frames"] * 1764 * 2 // 4
