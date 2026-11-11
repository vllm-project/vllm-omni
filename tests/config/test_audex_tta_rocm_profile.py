# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Pin the ROCm metadata shim for Audex TTA's attention-free codec."""

import pytest

from tests.helpers.stage_config import get_deploy_config_path
from vllm_omni.config.stage_config import (
    _apply_platform_overrides,
    load_deploy_config,
    merge_pipeline_deploy,
)
from vllm_omni.model_executor.models.audex.pipeline import AUDEX_TTA_PIPELINE

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


@pytest.mark.parametrize("profile", ["audex_tta.yaml", "audex_tta_30b.yaml"])
def test_rocm_supplies_supported_xcodec_head_metadata(profile: str) -> None:
    deploy = _apply_platform_overrides(
        load_deploy_config(get_deploy_config_path(profile)),
        platform="rocm",
    )
    stages = merge_pipeline_deploy(AUDEX_TTA_PIPELINE, deploy)

    # XCodec has no vLLM attention layers. This value only keeps ROCm backend
    # validation from interpreting the root config's absent head metadata as 0.
    assert stages[1].yaml_engine_args["hf_overrides"]["head_dim"] == 64


@pytest.mark.parametrize("profile", ["audex_tta.yaml", "audex_tta_30b.yaml"])
def test_xcodec_head_metadata_override_is_rocm_only(profile: str) -> None:
    deploy = _apply_platform_overrides(
        load_deploy_config(get_deploy_config_path(profile)),
        platform="cuda",
    )
    stages = merge_pipeline_deploy(AUDEX_TTA_PIPELINE, deploy)

    assert "hf_overrides" not in stages[1].yaml_engine_args
