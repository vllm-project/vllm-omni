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

import pytest

from vllm_omni.engine.stage_init_utils import _check_stage_device_layout

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _stage(stage_id, devices, num_replicas=1):
    return types.SimpleNamespace(
        stage_id=stage_id,
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
