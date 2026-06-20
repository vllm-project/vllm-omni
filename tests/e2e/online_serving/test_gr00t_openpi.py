# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""End-to-end online serving test for GR00T N1.7 through the OpenPI robot endpoint."""

import os

import numpy as np
import pytest

from tests.gr00t import openpi_client_helper as openpi_client
from tests.helpers.mark import hardware_test
from tests.helpers.runtime import OmniServerParams
from tests.helpers.stage_config import get_deploy_config_path

os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

MODEL = "nvidia/GR00T-N1.7-3B"

pytest.importorskip("websockets")
pytest.importorskip("openpi_client.msgpack_numpy")

test_params = [
    pytest.param(
        OmniServerParams(
            model=MODEL,
            stage_config_path=get_deploy_config_path("Gr00tN1d7.yaml"),
            server_args=["--disable-log-stats"],
            env_dict={"VLLM_DISABLE_COMPILE_CACHE": "1", "GR00T_NOISE_SEED": "42"},
            init_timeout=1200,
            stage_init_timeout=900,
        ),
        id="gr00t-n1d7-openpi",
    )
]

# Reference values captured from the Isaac-GR00T ZMQ server (nvidia/GR00T-N1.7-3B,
# OXE_DROID_RELATIVE_EEF_RELATIVE_JOINT embodiment) using build_droid_observation()
# inputs: zero images (256×256), identity eef_9d, zero gripper/joint, "pick up the object".
# Outputs are bit-reproducible across resets and across runs (max_diff=0.0).
_REF_EEF_9D = np.array(
    [
        [
            0.014888550154864788,
            -0.00039258378092199564,
            -0.013574761338531971,
            0.9999837875366211,
            0.005678884219378233,
            -0.00034708858584053814,
            -0.005677700508385897,
            0.9999783635139465,
            0.0033210734836757183,
        ],
        [
            0.020188070833683014,
            -0.0003105341165792197,
            -0.02232760563492775,
            0.9997346997261047,
            0.02301345206797123,
            0.000983047066256404,
            -0.02302766777575016,
            0.9995648860931396,
            0.018431292846798897,
        ],
        [
            -0.007266733795404434,
            -0.05537768080830574,
            0.03667901083827019,
            0.992686927318573,
            0.1206541359424591,
            0.003906540106981993,
            -0.12024091929197311,
            0.9853788614273071,
            0.12070894986391068,
        ],
    ],
    dtype=np.float32,
)  # rows = step 0, step 4, step 39

_REF_GRIPPER = np.array([[0.0], [0.0078125], [0.939453125]], dtype=np.float32)  # steps 0, 4, 39

_REF_JOINT = np.array(
    [
        [
            -0.0010484338272362947,
            0.0014262489276006818,
            -0.003565810853615403,
            -3.846167237497866e-05,
            -0.0002604846959002316,
            0.008521700277924538,
            -0.006872728932648897,
        ],
        [
            -0.009435676969587803,
            0.0021475711837410927,
            -0.0031688229646533728,
            -1.8328893929719925e-05,
            0.0005945992306806147,
            0.019159257411956787,
            0.0009468861389905214,
        ],
        [
            -0.02944088727235794,
            -0.08419207483530045,
            -0.0251418836414814,
            0.00540524534881115,
            0.04752273112535477,
            -0.012884500436484814,
            0.024298785254359245,
        ],
    ],
    dtype=np.float32,
)  # rows = step 0, step 4, step 39


@pytest.mark.full_model
@pytest.mark.diffusion
@hardware_test(res={"cuda": "L4"}, num_cards=1)
@pytest.mark.parametrize("omni_server", test_params, indirect=True)
def test_gr00t_n1d7_openpi_online(omni_server) -> None:
    result = openpi_client.run_policy_session(
        host=omni_server.host,
        port=omni_server.port,
        session_id="gr00t-online-e2e",
    )
    openpi_client.validate_session_result(result)


@pytest.mark.full_model
@pytest.mark.diffusion
@hardware_test(res={"cuda": "L4"}, num_cards=1)
@pytest.mark.parametrize("omni_server", test_params, indirect=True)
def test_gr00t_n1d7_openpi_precision(omni_server) -> None:
    """Assert actions match Isaac-GR00T reference (GR00T_NOISE_SEED=42, zero inputs)."""
    client = openpi_client.OpenPIWebsocketClient(host=omni_server.host, port=omni_server.port)
    try:
        obs = openpi_client.build_droid_observation(session_id="gr00t-precision-e2e")
        client.reset({})
        actions = client.infer(obs)
    finally:
        client.close()

    steps = [0, 4, 39]
    np.testing.assert_allclose(
        actions["eef_9d"][0, steps, :],
        _REF_EEF_9D,
        atol=1e-2,
        rtol=0.0,
        err_msg="eef_9d action mismatch vs Isaac-GR00T reference",
    )
    np.testing.assert_allclose(
        actions["gripper_position"][0, steps, :],
        _REF_GRIPPER,
        atol=1e-2,
        rtol=0.0,
        err_msg="gripper_position action mismatch vs Isaac-GR00T reference",
    )
    np.testing.assert_allclose(
        actions["joint_position"][0, steps, :],
        _REF_JOINT,
        atol=1e-2,
        rtol=0.0,
        err_msg="joint_position action mismatch vs Isaac-GR00T reference",
    )
