# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""GPU coverage for PersonaPlex on the unified full-duplex framework.

Boots the PersonaPlex deploy (``vllm_omni/deploy/personaplex.yaml``) and runs
the strict Realtime driver (``personaplex_realtime_duplex.py``): two paced
24 kHz sessions, admission overflow, per-session slot recycling, audible
whole-frame output. The checkpoint is gated, so the test runs only when
``PERSONAPLEX_MODEL_PATH`` points at a local copy of ``nvidia/personaplex-7b-v1``.
"""

from __future__ import annotations

import asyncio
import os
from pathlib import Path

import pytest

from tests.e2e.online_serving import personaplex_realtime_duplex as driver
from tests.helpers.mark import hardware_test
from tests.helpers.runtime import OmniServerParams
from tests.helpers.stage_config import get_deploy_config_path

pytestmark = pytest.mark.omni

MODEL_PATH = os.environ.get("PERSONAPLEX_MODEL_PATH", "")
DEPLOY_CONFIG = get_deploy_config_path("personaplex.yaml")

SERVER_PARAMS = [
    pytest.param(
        OmniServerParams(
            model=MODEL_PATH or "nvidia/personaplex-7b-v1",
            stage_config_path=DEPLOY_CONFIG,
            use_stage_cli=False,
            server_args=["--trust-remote-code"],
        ),
        id="two-stage-single-gpu",
    )
]

requires_checkpoint = pytest.mark.skipif(
    not MODEL_PATH or not Path(MODEL_PATH).is_dir(),
    reason="set PERSONAPLEX_MODEL_PATH to a local nvidia/personaplex-7b-v1 checkout",
)


def _speech_wav() -> Path:
    """Real speech (the MiniCPM-o asset, 16 kHz; the driver resamples to 24 kHz).

    PersonaPlex answers speech, not tones: a synthetic signal yields a silent
    reply and the driver's audibility floor rightly fails it.
    """
    return Path(__file__).resolve().parents[2] / "assets" / "minicpmo_4_5" / "response_required_16k.wav"


@requires_checkpoint
@pytest.mark.advanced_model
@hardware_test(res={"cuda": "H100"}, num_cards=1)
@pytest.mark.parametrize("omni_server", SERVER_PARAMS, indirect=True)
def test_personaplex_realtime_duplex_sessions(omni_server, tmp_path: Path) -> None:
    input_wav = _speech_wav()
    args = driver.parse_args(
        [
            "--url",
            f"ws://{omni_server.host}:{omni_server.port}/v1/realtime?duplex=1",
            "--model",
            MODEL_PATH,
            "--input-wav",
            str(input_wav),
            "--output-dir",
            str(tmp_path / "out"),
        ]
    )

    result = asyncio.run(driver.run(args))

    assert result["primary"]["session_id"]
    assert result["secondary"]["session_id"] != result["primary"]["session_id"]
    assert result["replacement"]["session_id"] not in {
        result["primary"]["session_id"],
        result["secondary"]["session_id"],
    }
