# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Real-checkpoint OpenPI serving tests for LingBot-VLA 2.0; requires one CUDA GPU with 40 GiB free.

Set LINGBOT_MODEL to a local snapshot from the recipe to run offline.
These exercise serving, not robot task success.
"""

import json
import os
from pathlib import Path

import numpy as np
import pytest

from tests.helpers.mark import hardware_test
from tests.helpers.runtime import OmniServerParams

pytest.importorskip("websockets")
pytest.importorskip("openpi_client.msgpack_numpy")

REFERENCE = json.loads((Path(__file__).parent / "fixtures/lingbot_vla_v2_reference.json").read_text())
CAMERA_KEYS = tuple(REFERENCE["camera_keys"])
ACTION_SHAPE = (50, 14)
SMOKE_STEPS = 2
PARAMS = OmniServerParams(
    model=os.environ.get("LINGBOT_MODEL", "lerobot/lingbot_vla_v2_robotwin"),
    port=8093,
    server_args=[
        "--deploy-config",
        "vllm_omni/deploy/lingbot_vla_v2.yaml",
        "--enforce-eager",
        "--disable-log-stats",
    ],
    env_dict={"VLLM_WORKER_MULTIPROC_METHOD": "spawn"},
)


def build_observation(
    camera_keys: tuple[str, ...], num_inference_steps: int
) -> dict[str, np.ndarray | str | int | dict[str, int]]:
    """Reference state and prompt with deterministic 256x256 frames for the chosen cameras."""
    y, x = np.mgrid[:256, :256]
    observation: dict[str, np.ndarray | str | int | dict[str, int]] = {
        key: np.stack([(x + index * 71) % 256, (y + index * 31) % 256, ((x // 16 + y // 16) % 2) * 255], -1).astype(
            np.uint8
        )
        for index, key in enumerate(CAMERA_KEYS)
        if key in camera_keys
    }
    observation.update(
        state=np.asarray(REFERENCE["state"], dtype=np.float32),
        prompt=REFERENCE["prompt"],
        seed=123,
        sampling_params={"num_inference_steps": num_inference_steps},
    )
    return observation


@pytest.mark.slow
@pytest.mark.diffusion
@hardware_test(res={"cuda": "H100"})
@pytest.mark.parametrize("omni_server", [PARAMS], indirect=True)
def test_lingbot_openpi_returns_a_finite_chunk_for_each_camera_subset(omni_server, online_client) -> None:
    subsets = (CAMERA_KEYS[:1], CAMERA_KEYS[::2], CAMERA_KEYS)
    operations = [{"endpoint": "infer", "payload": build_observation(keys, SMOKE_STEPS)} for keys in subsets]

    (response,) = online_client.send_robot_openpi_ws_request({"operations": operations})

    metadata = response.server_metadata
    assert (metadata["action_horizon"], metadata["action_dim"]) == ACTION_SHAPE, "handshake declares a 50x14 chunk"
    assert metadata["needs_session_id"] is False, "the policy is stateless"
    for actions in response.operation_responses:
        assert actions.shape == ACTION_SHAPE, "one 50x14 chunk per camera subset"
        assert actions.dtype == np.float32, "actions are float32 on the wire"
        assert np.isfinite(actions).all(), "actions are finite"


@pytest.mark.slow
@pytest.mark.diffusion
@hardware_test(res={"cuda": "H100"})
@pytest.mark.parametrize("omni_server", [PARAMS], indirect=True)
def test_lingbot_openpi_matches_the_reference_actions(omni_server, online_client) -> None:
    observation = build_observation(CAMERA_KEYS, REFERENCE["num_inference_steps"])

    (response,) = online_client.send_robot_openpi_ws_request(
        {"operations": [{"endpoint": "infer", "payload": observation}]}
    )

    (actions,) = response.action_tensors
    np.testing.assert_allclose(
        actions, REFERENCE["actions"], atol=1e-4, rtol=1e-5, err_msg="numerically equivalent to the reference policy"
    )


@pytest.mark.slow
@pytest.mark.diffusion
@hardware_test(res={"cuda": "H100"})
@pytest.mark.parametrize("omni_server", [PARAMS], indirect=True)
def test_lingbot_openpi_repeats_actions_across_interleaving_errors_reset_and_reconnect(
    omni_server, online_client
) -> None:
    observation = build_observation(CAMERA_KEYS, REFERENCE["num_inference_steps"])
    other = observation | {"seed": 124, "prompt": "put the blue object in the tray"}
    invalid = observation | {"state": np.zeros(13, dtype=np.float32)}
    operations = [
        {"endpoint": "infer", "payload": observation},
        {"endpoint": "infer", "payload": other},
        {"endpoint": "infer", "payload": invalid, "expect_error": True},
        {"endpoint": "infer", "payload": observation},
        {"endpoint": "reset", "payload": {}},
    ]

    (session,) = online_client.send_robot_openpi_ws_request({"operations": operations})
    (reconnected,) = online_client.send_robot_openpi_ws_request(
        {"operations": [{"endpoint": "infer", "payload": observation}]}
    )

    baseline, interleaved, recovered = session.action_tensors
    assert interleaved.shape == ACTION_SHAPE, "a different observation in between is served"
    assert session.operation_responses[2]["type"] == "error", "a 13-D state is rejected"
    assert session.operation_responses[-1]["status"] == "reset successful", "reset is acknowledged"
    np.testing.assert_allclose(recovered, baseline, atol=1e-4, rtol=1e-5, err_msg="same actions after an error")
    np.testing.assert_allclose(
        reconnected.action_tensors[0], baseline, atol=1e-4, rtol=1e-5, err_msg="same actions on a new connection"
    )
