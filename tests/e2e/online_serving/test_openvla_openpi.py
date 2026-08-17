# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""End-to-end online serving test for OpenVLA-7B through the OpenPI robot endpoint.

Boots ``vllm serve --omni --deploy-config openvla.yaml`` and drives the real
msgpack websocket at ``/v1/realtime/robot/openpi`` — the wire path a robot uses.
Unlike the other robot policies here, OpenVLA is a single autoregressive stage
whose action *is* seven generated tokens, so this also covers the token → action
decode that ``vllm_omni/model_executor/models/openvla/action_decode.py`` performs
on the serving side.

Deliberately no golden action vector. OpenVLA emits bin indices, and on a
synthetic frame the model's own top-1/top-2 margin gets as low as 0.125 nat —
inside bf16 rounding resolution — so a pinned expected action would be a
flakiness generator rather than a correctness check. What is asserted instead
holds whichever bin the policy picks: the returned values must land exactly on
the action grid the checkpoint's own ``norm_stats`` define, the un-normalisation
mask must be honoured, and the per-request ``unnorm_key`` must reach the decoder.
The comparison against the reference implementation belongs on in-distribution
inputs and is reported in the PR that added this model.

Run it with real weights::

    pytest -s -v tests/e2e/online_serving/test_openvla_openpi.py --run-level full_model

The run level is not cosmetic here. ``stage_config_path_for_run_level`` adds
``load_format: dummy`` to the deploy config for every level below
``advanced_model``, so the default ``core_model`` serves random weights — under
which OpenVLA emits ids outside the action range, every one of them clips onto
the same extreme bin, and a weaker test would pass while proving nothing. The
module refuses to run there rather than going green.

Like ``test_gr00t_openpi_expansion.py`` and ``test_pi0_expansion.py``, this file
is not wired into a Buildkite job: it needs a GPU and the 14 GiB checkpoint.
"""

from __future__ import annotations

import os

import numpy as np
import pytest

from tests.helpers.mark import hardware_test
from tests.helpers.runtime import OmniServerParams
from tests.helpers.stage_config import get_deploy_config_path

os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

MODEL = "openvla/openvla-7b"
IMAGE_SIZE = 224
ACTION_DIM = 7
INSTRUCTION = "put the spoon on the towel"

# openvla.yaml pins bridge_orig; fractal20220817_data is a second embodiment in the
# same checkpoint whose q01/q99 are roughly 8x wider, so the two un-normalise the
# same bin onto visibly different numbers.
UNNORM_KEY = "bridge_orig"
ALT_UNNORM_KEY = "fractal20220817_data"

# The 255 values a normalised action dimension can take: midpoints of 256 uniform
# bins over [-1, 1]. This is checkpoint-independent — it follows from n_action_bins.
_BIN_EDGES = np.linspace(-1.0, 1.0, 256)
BIN_CENTERS = (_BIN_EDGES[:-1] + _BIN_EDGES[1:]) / 2.0

pytest.importorskip("websockets")
pytest.importorskip("openpi_client.msgpack_numpy")

test_params = [
    pytest.param(
        OmniServerParams(
            model=MODEL,
            stage_config_path=get_deploy_config_path("openvla.yaml"),
            server_args=["--disable-log-stats"],
            env_dict={"VLLM_DISABLE_COMPILE_CACHE": "1"},
            init_timeout=1200,
        ),
        id="openvla-7b-openpi",
    )
]


@pytest.fixture(scope="module", autouse=True)
def _require_real_weights(run_level: str) -> None:
    """Refuse the dummy-weight run levels instead of passing on random weights."""
    if run_level not in ("advanced_model", "full_model"):
        pytest.skip(f"OpenVLA e2e needs real weights; pass --run-level full_model (got {run_level!r})")


def _gradient_frame(size: int = IMAGE_SIZE) -> np.ndarray:
    """A deterministic RGB frame — no RNG, so it is identical across runs."""
    ys, xs = np.meshgrid(np.arange(size), np.arange(size), indexing="ij")
    return np.stack(
        [xs * 255 // (size - 1), ys * 255 // (size - 1), (xs + ys) % 256],
        axis=-1,
    ).astype(np.uint8)


def _bars_frame(size: int = IMAGE_SIZE) -> np.ndarray:
    ys, xs = np.meshgrid(np.arange(size), np.arange(size), indexing="ij")
    return np.stack(
        [((xs // 16) % 2) * 220 + 15, ((ys // 32) % 2) * 160 + 40, ((xs + ys) // 24 % 2) * 200 + 25],
        axis=-1,
    ).astype(np.uint8)


def _observation(frame: np.ndarray, **extra) -> dict:
    return {"image": frame, "prompt": INSTRUCTION, **extra}


def _norm_stats(model_prefix: str) -> dict:
    """The checkpoint's per-embodiment action statistics, read the way the server reads them."""
    from vllm.transformers_utils.config import get_config

    return get_config(f"{model_prefix}{MODEL}", trust_remote_code=False).norm_stats


def _assert_on_action_grid(action: np.ndarray, stats: dict) -> None:
    """Every dimension must be a bin centre, un-normalised iff its mask bit is set.

    This is the assertion that survives a bin flip. It fails if ``q01``/``q99``
    are taken from the wrong embodiment, if the affine un-normalisation is wrong,
    or if the per-dimension mask is dropped — for openvla-7b the gripper's mask
    bit is ``False`` in all 25 embodiments while its ``q01``/``q99`` are 0.0/1.0,
    so ignoring the mask would halve that dimension and move it off the grid.
    """
    q01 = np.asarray(stats["q01"], dtype=np.float64)
    q99 = np.asarray(stats["q99"], dtype=np.float64)
    mask = np.asarray(stats["mask"], dtype=bool)

    values = np.asarray(action, dtype=np.float64).ravel()
    normalized = np.where(mask, 2.0 * (values - q01) / (q99 - q01) - 1.0, values)
    distance = np.abs(normalized[:, None] - BIN_CENTERS[None, :]).min(axis=1)
    assert (distance < 1e-4).all(), f"action components off the {len(BIN_CENTERS)}-bin grid: {distance.tolist()}"


@pytest.mark.full_model
@pytest.mark.omni
@hardware_test(res={"cuda": "L4"}, num_cards=1)
@pytest.mark.parametrize("omni_server", test_params, indirect=True)
def test_openvla_openpi_online(omni_server, openai_client) -> None:
    """Handshake, action shape, determinism, and that the action tracks the image."""
    response = openai_client.send_robot_openpi_ws_request(
        {
            "operations": [
                {"endpoint": "infer", "payload": _observation(_gradient_frame())},
                {"endpoint": "reset", "payload": {}},
                {"endpoint": "infer", "payload": _observation(_gradient_frame())},
                {"endpoint": "infer", "payload": _observation(_bars_frame())},
            ],
        }
    )[0]

    # The handshake is derived from the checkpoint's norm_stats, not configured in
    # the deploy yaml, so a fine-tuned OpenVLA advertises itself correctly.
    metadata = response.server_metadata
    for key in ("image_resolution", "needs_session_id", "action_horizon", "action_dim", "unnorm_key"):
        assert key in metadata, f"Missing OpenVLA metadata key: {key}"
    assert tuple(metadata["image_resolution"]) == (IMAGE_SIZE, IMAGE_SIZE)
    assert metadata["needs_session_id"] is False
    assert int(metadata["action_horizon"]) == 1
    assert int(metadata["action_dim"]) == ACTION_DIM
    assert metadata["unnorm_key"] == UNNORM_KEY
    assert UNNORM_KEY in metadata["supported_embodiments"]
    assert response.operation_responses[1]["status"] == "reset successful"

    actions = response.action_tensors
    assert actions is not None and len(actions) == 3
    for index, action in enumerate(actions):
        assert action.shape == (1, ACTION_DIM), f"action {index} shape {action.shape}"
        assert np.isfinite(action).all(), f"action {index} is not finite"

    # Greedy decoding of a stateless policy: the same frame gives the same action,
    # across a reset as well.
    np.testing.assert_array_equal(actions[0], actions[1])
    # A different frame must give a different action. This is what catches the
    # observation never reaching the model: without a usable image OpenVLA emits
    # ids outside the action range, they all clip onto the same extreme bin, and
    # the policy returns one constant vector for every frame.
    assert not np.array_equal(actions[0], actions[2]), "action did not change with the observation"


@pytest.mark.full_model
@pytest.mark.omni
@hardware_test(res={"cuda": "L4"}, num_cards=1)
@pytest.mark.parametrize("omni_server", test_params, indirect=True)
def test_openvla_openpi_unnorm_key_override(omni_server, openai_client, model_prefix) -> None:
    """A request may pick the embodiment, and both answers land on that embodiment's grid."""
    frame = _gradient_frame()
    response = openai_client.send_robot_openpi_ws_request(
        {
            "operations": [
                {"endpoint": "infer", "payload": _observation(frame)},
                {"endpoint": "infer", "payload": _observation(frame, unnorm_key=ALT_UNNORM_KEY)},
            ],
        }
    )[0]

    actions = response.action_tensors
    assert actions is not None and len(actions) == 2

    norm_stats = _norm_stats(model_prefix)
    _assert_on_action_grid(actions[0], norm_stats[UNNORM_KEY]["action"])
    _assert_on_action_grid(actions[1], norm_stats[ALT_UNNORM_KEY]["action"])

    # Same image, same tokens, different statistics — so the un-normalised values
    # must differ. Equality here would mean the request's unnorm_key was ignored.
    assert not np.allclose(actions[0], actions[1]), "per-request unnorm_key did not reach the decoder"
