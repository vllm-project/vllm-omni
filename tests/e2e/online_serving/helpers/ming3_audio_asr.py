# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Shared fixtures and argument builders for Ming-3.0-audio ASR E2E tests."""

from __future__ import annotations

import pytest

from tests.helpers.runtime import OmniServerParams
from tests.helpers.stage_config import get_deploy_config_path

MODEL = "inclusionAI/Ming-3.0-audio-tiny"
DEPLOY_CONFIG = get_deploy_config_path("ming3_audio_tiny_asr.yaml")

SERVER_PARAMS = [
    pytest.param(
        OmniServerParams(
            model=MODEL,
            stage_config_path=DEPLOY_CONFIG,
            use_stage_cli=False,
            server_args=[],
        ),
        id="ming3-audio-tiny-asr",
    )
]
