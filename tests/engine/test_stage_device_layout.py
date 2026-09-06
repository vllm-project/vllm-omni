# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Regression tests for issue #5003: a per-stage world size that its assigned
``devices`` cannot satisfy must fail early in ``build_vllm_config`` with a clear
message, rather than surfacing as an opaque worker-side ``local rank ... out of
bounds`` assertion.

Root cause: a top-level ``--tensor-parallel-size`` is broadcast to every stage,
but each stage's ``devices`` is not adjusted, so a stage can end up with e.g.
tensor_parallel_size=4 while still holding a single-GPU deploy default. Without
``--strategy-config`` the strategy-path device check never runs.
"""

import json
import re
import types
from unittest import mock

import pytest

from vllm_omni.engine import stage_init_utils
from vllm_omni.engine.stage_init_utils import (
    _check_stage_device_layout,
    build_vllm_config,
    compute_replica_layout,
    get_stage_devices_per_replica,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _stage(stage_id, devices, num_replicas=1, engine_args=None):
    return types.SimpleNamespace(
        stage_id=stage_id,
        stage_type="llm",
        engine_args=engine_args or {},
        runtime=types.SimpleNamespace(devices=devices, num_replicas=num_replicas),
    )


def test_tp_broadcast_without_devices_fails_early():
    """stage0 gets tensor_parallel_size=4 (broadcast) but only 1 device -> clear error."""
    stage = _stage(0, devices="0")
    engine_args = {
        "tensor_parallel_size": 4,
        "data_parallel_size": 1,
        "pipeline_parallel_size": 1,
    }
    with pytest.raises(ValueError) as excinfo:
        _check_stage_device_layout(stage, engine_args)
    msg = str(excinfo.value)
    # Message names the stage, the mismatch, and the actionable workaround.
    assert "Stage 0" in msg
    assert "1 device" in msg and "4" in msg
    assert "--stage-overrides" in msg


def test_workaround_example_sets_tp_on_every_stage():
    """The JSON example in the error message must not reproduce the bug: every
    stage entry it suggests has to set ``tensor_parallel_size`` (a single-GPU
    stage that only overrides ``devices`` would still inherit the broadcast
    tp and crash again — the original #5003 failure mode)."""
    stage = _stage(0, devices="0")
    with pytest.raises(ValueError) as excinfo:
        _check_stage_device_layout(stage, {"tensor_parallel_size": 4})
    msg = str(excinfo.value)

    match = re.search(r"\{.*\}", msg, re.DOTALL)
    assert match, f"no JSON example found in error message: {msg}"
    example = json.loads(match.group(0))

    assert example, "example must contain at least one stage"
    for stage_id, overrides in example.items():
        assert "tensor_parallel_size" in overrides, (
            f"stage {stage_id} in the suggested workaround omits "
            f"tensor_parallel_size, which reproduces #5003: {overrides}"
        )


def test_consistent_tp_and_devices_pass():
    """tensor_parallel_size=4 with 4 assigned devices is valid."""
    stage = _stage(0, devices="0,1,2,3")
    _check_stage_device_layout(
        stage,
        {"tensor_parallel_size": 4, "data_parallel_size": 1, "pipeline_parallel_size": 1},
    )


def test_single_gpu_stage_passes():
    """A TP=1 stage on a single device (talker/code2wav default) is valid."""
    stage = _stage(1, devices="1")
    _check_stage_device_layout(stage, {"tensor_parallel_size": 1})


def test_missing_devices_is_skipped():
    """No explicit devices -> vLLM assigns them; nothing to validate here."""
    stage = _stage(0, devices=None)
    _check_stage_device_layout(stage, {"tensor_parallel_size": 4})


def test_replica_pool_layout_passes():
    """Pool mode: num_replicas=2 x tp=2 => 4 devices is a valid pool shape."""
    stage = _stage(0, devices="0,1,2,3", num_replicas=2)
    _check_stage_device_layout(stage, {"tensor_parallel_size": 2})


def test_multinode_dp_uses_local_width_for_device_validation():
    """One local DP engine needs one device even when global DP spans four nodes."""
    stage = _stage(0, devices="0")
    _check_stage_device_layout(
        stage,
        {
            "tensor_parallel_size": 1,
            "data_parallel_size": 4,
            "data_parallel_size_local": 1,
            "pipeline_parallel_size": 1,
        },
    )


def test_zero_local_dp_skips_local_device_validation():
    """A head process with no local engines does not own a local device layout."""
    stage = _stage(0, devices="0")
    _check_stage_device_layout(
        stage,
        {
            "tensor_parallel_size": 4,
            "data_parallel_size": 4,
            "data_parallel_size_local": 0,
            "pipeline_parallel_size": 1,
        },
    )


@pytest.mark.parametrize(
    ("devices", "expected"),
    [
        ("0,1", ["0,1", "2,3"]),
        ("0,1,2,3", ["0,1", "2,3"]),
    ],
)
def test_replica_guard_and_splitter_share_local_world_size(devices, expected):
    """PP and local DP contribute to the same per-replica size in both paths."""
    engine_args = {
        "tensor_parallel_size": 1,
        "data_parallel_size": 4,
        "data_parallel_size_local": 1,
        "pipeline_parallel_size": 2,
    }
    stage = _stage(0, devices=devices, num_replicas=2, engine_args=engine_args)

    _check_stage_device_layout(stage, engine_args)
    assert get_stage_devices_per_replica(stage) == 2
    _, replica_devices_map = compute_replica_layout([stage])
    assert replica_devices_map == {0: expected}


def test_non_tp_layout_error_uses_generic_guidance():
    """A PP mismatch must not be reported as a top-level TP broadcast."""
    stage = _stage(0, devices="0")
    with pytest.raises(ValueError) as excinfo:
        _check_stage_device_layout(
            stage,
            {
                "tensor_parallel_size": 1,
                "data_parallel_size": 1,
                "pipeline_parallel_size": 2,
            },
        )

    msg = str(excinfo.value)
    assert "local world size" in msg
    assert "top-level --tensor-parallel-size" not in msg


def test_build_vllm_config_fails_before_engine_config_on_mismatch():
    """Pin the production wiring: ``build_vllm_config`` must run the device-layout
    guard *before* ``create_engine_config``. If the guard call is dropped or moved
    after engine-config creation, #5003 regresses while the unit tests above stay
    green — so assert here that a mismatched stage raises and that neither
    ``create_engine_config`` nor ``Executor.get_class`` is reached."""
    stage = _stage(0, devices="0")
    with (
        mock.patch.object(stage_init_utils.OmniEngineArgs, "create_engine_config") as create_engine_config,
        mock.patch.object(stage_init_utils.Executor, "get_class") as get_class,
    ):
        with pytest.raises(ValueError) as excinfo:
            build_vllm_config(
                stage,
                model="dummy-model",
                engine_args_dict={
                    "tensor_parallel_size": 4,
                    "data_parallel_size": 1,
                    "pipeline_parallel_size": 1,
                },
            )
    msg = str(excinfo.value)
    assert "Stage 0" in msg
    assert "--stage-overrides" in msg
    # Guard fired early: worker/config construction was never reached.
    create_engine_config.assert_not_called()
    get_class.assert_not_called()


def test_build_vllm_config_proceeds_on_consistent_layout():
    """The guard must not false-positive: a consistent single-GPU stage flows past
    it and reaches ``create_engine_config`` / ``Executor.get_class`` as usual."""
    stage = _stage(1, devices="1")
    fake_config = types.SimpleNamespace(
        quant_config=None,
        model_config=types.SimpleNamespace(hf_config=types.SimpleNamespace()),
    )
    sentinel_executor = object()
    with (
        mock.patch.object(
            stage_init_utils.OmniEngineArgs, "create_engine_config", return_value=fake_config
        ) as create_engine_config,
        mock.patch.object(stage_init_utils.Executor, "get_class", return_value=sentinel_executor),
        mock.patch.object(stage_init_utils.OmniINCConfig, "maybe_upgrade", side_effect=lambda quant: quant),
    ):
        vllm_config, executor_class = build_vllm_config(
            stage,
            model="dummy-model",
            engine_args_dict={"tensor_parallel_size": 1},
        )
    create_engine_config.assert_called_once()
    assert vllm_config is fake_config
    assert executor_class is sentinel_executor
