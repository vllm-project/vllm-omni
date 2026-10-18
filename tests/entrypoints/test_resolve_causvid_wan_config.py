# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import pytest

from vllm_omni.config.pipeline_registry import OMNI_PIPELINES, resolve_pipeline_config
from vllm_omni.diffusion.models.causvid_wan.pipeline_causvid_wan import CausVidWanPipeline
from vllm_omni.diffusion.registry import DiffusionModelRegistry

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def test_causvid_wan_registered_in_pipeline_registry():
    assert "causvid_wan" in OMNI_PIPELINES
    pipeline = resolve_pipeline_config("causvid_wan")
    assert pipeline is not None
    assert pipeline.model_arch == "CausVidWanPipeline"
    assert pipeline.default_deploy_config_name == "causvid_wan.yaml"
    cls = DiffusionModelRegistry._try_load_model_cls("CausVidWanPipeline")
    assert cls is CausVidWanPipeline
