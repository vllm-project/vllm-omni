# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Materialize production JoyAI profiles for the shared server fixture."""

import atexit
import tempfile
from pathlib import Path

import yaml

from tests.helpers.runtime import OmniServerParams
from tests.helpers.stage_config import get_deploy_config_path
from vllm_omni.config.stage_config import resolve_deploy_yaml


def joyai_server_params(*, asr=False):
    profile = "joyai_vl_interaction" + ("_asr" if asr else "") + "_duplex.yaml"
    config = resolve_deploy_yaml(get_deploy_config_path(profile))
    # The fixture patches load_format on explicit stages; resolve inheritance
    # first so the test exercises the production overlay, including its limits.
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as handle:
        yaml.safe_dump(config, handle)
    atexit.register(Path(handle.name).unlink, missing_ok=True)
    return OmniServerParams(
        model="jdopensource/JoyAI-VL-Interaction-Preview",
        stage_config_path=handle.name,
    )
