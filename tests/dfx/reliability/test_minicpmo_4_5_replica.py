# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""MiniCPM-o 4.5 stage-replica reliability.

Uses ``minicpmo_4_5_3gpu_stage1_replicas.yaml``: one thinker, and two stage-1
replicas on GPU 1 and GPU 2. A test kills one stage-1 replica.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tests.dfx.reliability.helpers import launch_scenario_params, scenario_params
from tests.dfx.reliability.scenarios import (
    ReplicaCrashAtExit,
    ReplicaCrashAtLaunch,
    ReplicaCrashIdle,
    ReplicaCrashWithLoad,
    ReplicaGracefulShutdown,
)
from tests.dfx.reliability.test_minicpmo_4_5 import MiniCPMO45
from tests.helpers.mark import hardware_marks

_DEPLOY_DIR = Path(__file__).resolve().parents[3] / "vllm_omni" / "deploy"
_NATIVE_CHAT_TEMPLATE = (
    Path(__file__).resolve().parents[3] / "vllm_omni/transformers_utils/chat_templates/minicpmo45_native.jinja"
)


class MiniCPMO45Replica(MiniCPMO45):
    res = {"cuda": "H100", "rocm": "MI325"}
    num_cards = 3
    deploy_dir = _DEPLOY_DIR
    scenarios = [
        {
            "test_name": "minicpmo_4_5_reliability_replica",
            "server_params": {
                "model": "openbmb/MiniCPM-o-4_5",
                "stage_config_name": "minicpmo_4_5_3gpu_stage1_replicas.yaml",
                "server_args": [
                    "--trust-remote-code",
                    "--async-chunk",
                    "--chat-template",
                    str(_NATIVE_CHAT_TEMPLATE),
                    "--chat-template-content-format",
                    "openai",
                ],
            },
        }
    ]
    replica_scenarios = scenarios
    # devices "1,2": replica 1 is GPU 2.
    bad_replica_pattern = "StageEngineCoreProc_stage1_replica1"
    bad_replica_devices = "2"


MODEL = MiniCPMO45Replica()

pytestmark = [
    pytest.mark.slow,
    *MODEL.extra_marks,
    *hardware_marks(res=MODEL.res, num_cards=MODEL.num_cards),
]


@scenario_params(MODEL, ReplicaCrashIdle)
def test_replica_crash_at_idle(omni_server_function, online_client_function) -> None:
    """Kill one idle stage-1 replica. Six later multimodal requests must all succeed."""
    ReplicaCrashIdle().run(MODEL, omni_server_function, online_client_function)


@scenario_params(MODEL, ReplicaCrashWithLoad)
def test_replica_crash_at_load(omni_server_function, online_client_function) -> None:
    """Kill one stage-1 replica under multimodal load; all post-convergence requests must succeed."""
    ReplicaCrashWithLoad().run(MODEL, omni_server_function, online_client_function)


@scenario_params(MODEL, ReplicaCrashAtExit)
@pytest.mark.skip(
    reason="Stage1 wakeup FIFO remains after shutdown: https://github.com/vllm-project/vllm-omni/issues/10391"
)
def test_replica_crash_at_exit(omni_server_function, online_client_function) -> None:
    """Begin normal shutdown, kill one stage-1 replica, and require timely cleanup of the whole instance."""
    ReplicaCrashAtExit().run(MODEL, omni_server_function, online_client_function)


@launch_scenario_params(
    MODEL,
    ReplicaCrashAtLaunch,
    skip_reason="Replica crash aborts startup: https://github.com/vllm-project/vllm-omni/issues/10378",
)
def test_replica_crash_at_launch(launch_params) -> None:
    """Kill a stage-1 replica before readiness; require the original instance to serve."""
    ReplicaCrashAtLaunch().run(MODEL, launch_params)


@scenario_params(MODEL, ReplicaGracefulShutdown)
@pytest.mark.skip(
    reason="Stage1 wakeup FIFOs remain after normal shutdown: https://github.com/vllm-project/vllm-omni/issues/10391"
)
def test_graceful_shutdown(omni_server_function, online_client_function) -> None:
    """SIGTERM the healthy root; all children and resources must clear within 60s."""
    ReplicaGracefulShutdown().run(MODEL, omni_server_function, online_client_function)
