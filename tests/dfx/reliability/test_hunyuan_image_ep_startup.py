# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""HunyuanImage expert-parallel startup coverage on ROCm.

Process-kill coverage uses expert parallel disabled because an EP image request
does not currently finish within the reliability timeout on ROCm. This test
keeps the canonical TP=4 expert-parallel deploy covered through startup and its
live health endpoint without installing a permanently failing inference job.
"""

from __future__ import annotations

from typing import Any

import pytest

from tests.dfx.reliability.helpers import FaultScenario, scenario_params
from tests.dfx.reliability.test_hunyuan_image import HunyuanImage
from tests.helpers.mark import hardware_marks


class HunyuanImageEpStartup(HunyuanImage):
    res = {"rocm": "MI325"}
    scenarios = [
        {
            "test_name": "hunyuan_image_dit_ep_startup",
            "server_params": {
                "model": "tencent/HunyuanImage-3.0-Instruct",
                "stage_config_name": "hunyuan_image3_dit.yaml",
            },
        }
    ]


class _EpStartup(FaultScenario):
    def run(self, model: Any, server: Any, client: Any, signal_name: str | None = None) -> None:
        response = client.send_health_http_request({"timeout": 30})[0]
        assert response.status_code == 200


MODEL = HunyuanImageEpStartup()

pytestmark = [
    pytest.mark.full_model,
    pytest.mark.slow,
    pytest.mark.diffusion,
    *hardware_marks(res=MODEL.res, num_cards=MODEL.num_cards),
]


@scenario_params(MODEL, _EpStartup)
def test_hunyuan_image_rocm_ep_startup(omni_server_function, online_client_function) -> None:
    """Start the TP=4 expert-parallel server and verify its health endpoint."""
    _EpStartup().run(MODEL, omni_server_function, online_client_function)
