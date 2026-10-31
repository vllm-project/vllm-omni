# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Shared default/V1 server configurations for VoxCPM2 online tests."""

import os

import pytest
import yaml

from tests.helpers.fixtures.runtime import omni_fixture_lock
from tests.helpers.runtime import OmniServerParams, iter_omni_server
from tests.helpers.stage_config import get_deploy_config_path

MODEL = os.environ.get("VOXCPM2_MODEL", "openbmb/VoxCPM2")


@pytest.fixture(scope="module")
def omni_server(request, run_level, tmp_path_factory):
    deploy_path = get_deploy_config_path("voxcpm2.yaml")
    if request.param == "v1":
        from vllm_omni.platforms import current_omni_platform

        if not current_omni_platform.is_cuda():
            pytest.skip("VoxCPM2 V1 opt-out regression runs on CUDA")
        with open(deploy_path) as stream:
            config = yaml.safe_load(stream)
        config["model_runner"] = "v1"
        # The CUDA override selects V2; removing it restores the base V1 audio settings.
        config.get("platforms", {}).pop("cuda", None)
        path = tmp_path_factory.mktemp("voxcpm2-v1") / "deploy.yaml"
        path.write_text(yaml.safe_dump(config))
        deploy_path = str(path)
    request.param = OmniServerParams(
        model=MODEL,
        stage_config_path=deploy_path,
        server_args=["--trust-remote-code", "--disable-log-stats"],
    )
    yield from iter_omni_server(request, run_level, omni_fixture_lock)


tts_server_params = [
    pytest.param("default", id="voxcpm2-default"),
    pytest.param("v1", id="voxcpm2-v1"),
]
