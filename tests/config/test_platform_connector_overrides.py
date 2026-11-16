# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Platform stage connector refs have to reach the transfer-config parser."""

from pathlib import Path

import pytest

from vllm_omni.engine.stage_init_utils import load_omni_transfer_config_for_model

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

_HUNYUAN_DEPLOY = Path(__file__).resolve().parents[2] / "vllm_omni" / "deploy" / "hunyuan_image_3_moe.yaml"


def _edge_name(config, from_stage: str, to_stage: str) -> str:
    return config.connectors[(from_stage, to_stage)].name


@pytest.mark.parametrize(
    ("platform", "connector_name"),
    [("xpu", "SharedMemoryConnector"), ("cuda", "MooncakeTransferEngineConnector")],
)
def test_hunyuan_xpu_connector_override_reaches_the_transfer_config(platform, connector_name):
    config = load_omni_transfer_config_for_model(
        "tencent/HunyuanImage-3.0",
        str(_HUNYUAN_DEPLOY),
        platform=platform,
    )

    assert config is not None
    assert _edge_name(config, "0", "1") == connector_name
