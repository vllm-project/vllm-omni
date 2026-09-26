# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Deploy-config guard: embed-early MammothModa2 AR stages keep async scheduling off.

MammothModa2 is embed-early: the AR runner embeds ``input_ids`` during input
preparation (``gpu_model_runner._preprocess`` → ``embed_input_ids``), which
runs *before* the previous step's sampled token id is written back. Under
``async_scheduling=True`` a ``-1`` pending-token sentinel can therefore reach
``get_input_embeddings`` — reproduced as an out-of-range gather at batch >= 8
(#7319). All MammothModa2 deploy configs pin ``async_scheduling: false``
(the FP8-KV overlay inherits it from ``mammoth_moda2.yaml`` via
``base_config``); this test fails if an AR stage is flipped back to the async
scheduler (the ``LLM_AR`` default).
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from tests.helpers.stage_config import get_deploy_config_path
from vllm_omni.config.pipeline_registry import OMNI_PIPELINES
from vllm_omni.config.stage_config import load_deploy_config, merge_pipeline_deploy

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

# (deploy yaml, pipeline name) for every config that runs the MammothModa2
# AR backbone. The FP8-KV overlay only sets kv_cache_dtype keys; its AR stage
# must still resolve with the base config's async pin after base_config
# inheritance.
_MM2_AR_DEPLOYS = [
    ("mammoth_moda2.yaml", "mammoth_moda2"),
    ("mammoth_moda2_ar.yaml", "mammoth_moda2_ar"),
    ("mammoth_moda2_fp8_kv.yaml", "mammoth_moda2"),
]


def _ar_stages(pipeline_name: str, deploy_name: str) -> list:
    deploy = load_deploy_config(get_deploy_config_path(deploy_name))
    with patch("vllm_omni.platforms.current_omni_platform") as platform:
        platform.device_name = "cuda"
        stages = merge_pipeline_deploy(OMNI_PIPELINES[pipeline_name], deploy)

    ar_stages = [
        s
        for s in stages
        if s.scheduler_cls and s.scheduler_cls.split(".")[-1] in ("OmniARScheduler", "OmniARAsyncScheduler")
    ]
    assert ar_stages, f"{deploy_name}: expected at least one LLM_AR stage"
    return ar_stages


@pytest.mark.parametrize(("deploy_name", "pipeline_name"), _MM2_AR_DEPLOYS)
def test_mammoth_moda2_ar_stages_pin_async_scheduling_off(deploy_name: str, pipeline_name: str) -> None:
    for stage in _ar_stages(pipeline_name, deploy_name):
        # `merge_pipeline_deploy` rewrites the key from the resolved scheduler
        # class, so this is the effective value the engine will see — and it
        # must be an explicit False, not the LLM_AR default (True).
        assert stage.yaml_engine_args.get("async_scheduling") is False, (
            f"{deploy_name} stage {stage.stage_id}: embed-early AR stage must pin `async_scheduling: false`"
        )
        assert stage.scheduler_cls.endswith(".OmniARScheduler"), (
            f"{deploy_name} stage {stage.stage_id}: embed-early AR stage must "
            f"resolve to the sync OmniARScheduler, got {stage.scheduler_cls}"
        )
