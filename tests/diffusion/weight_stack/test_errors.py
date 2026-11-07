# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Tests for the weight-stack error hierarchy and message format (RFC 9536 01-1)."""

from __future__ import annotations

import pickle

import pytest

from vllm_omni.diffusion.weight_stack.errors import (
    MAX_ERROR_MESSAGE_CHARS,
    STEP_ERROR_TYPES,
    AllocationError,
    CommitError,
    ConfigError,
    ConfirmationTimeoutError,
    ConstructionError,
    DeviceCapacityError,
    ErrorSummary,
    FinalizeError,
    GroupConfirmationError,
    HostCapacityError,
    LifecycleError,
    PeerFailureError,
    PlanningError,
    PreflightError,
    PrepareError,
    RuntimeFault,
    StackUnavailableError,
    WeightStackError,
    WriteError,
    wrap_step_error,
)

pytestmark = [pytest.mark.diffusion, pytest.mark.cpu, pytest.mark.core_model]

EXPECTED_PHASES = {
    ConfigError: "config",
    ConstructionError: "build_structure",
    PlanningError: "plan",
    PreflightError: "preflight",
    GroupConfirmationError: "group_confirmation",
    HostCapacityError: "admit",
    PrepareError: "prepare",
    DeviceCapacityError: "prepare",
    StackUnavailableError: "construct_modules",
    AllocationError: "allocate",
    WriteError: "write",
    FinalizeError: "finalize",
    CommitError: "commit",
    PeerFailureError: "group_confirmation",
    ConfirmationTimeoutError: "group_confirmation",
    LifecycleError: "lifecycle",
    RuntimeFault: "service",
}


def test_message_format_includes_all_context():
    error = ConfigError(
        "switch value 'broken' is not one of legacy, unified",
        where="vllm_omni/diffusion/data.py::OmniDiffusionConfig",
        fix="use --diffusion-weight-stack legacy",
    )
    error.stage_id = 0
    error.rank = 1
    error.world_size = 2
    assert str(error) == (
        "[weight-stack][stage 0][rank 1/2][config] switch value 'broken' is not one of legacy, unified; "
        "at vllm_omni/diffusion/data.py::OmniDiffusionConfig; fix: use --diffusion-weight-stack legacy"
    )


def test_message_omits_absent_context():
    assert str(ConfigError("bad value")) == "[weight-stack][config] bad value"

    error = ConfigError("bad value", where="config.yaml")
    assert str(error) == "[weight-stack][config] bad value; at config.yaml"


def test_rank_without_world_size_is_rendered():
    error = WriteError("write failed")
    error.rank = 3
    assert str(error) == "[weight-stack][rank 3][write] write failed"


def test_error_context_attributes_start_unset():
    error = StackUnavailableError("missing", component="loader")
    assert error.stage_id is None
    assert error.rank is None
    assert error.world_size is None
    assert error.component == "loader"


def test_config_error_is_both_runtime_and_value_error():
    error = ConfigError("bad")
    assert isinstance(error, WeightStackError)
    assert isinstance(error, RuntimeError)
    assert isinstance(error, ValueError)


@pytest.mark.parametrize(("error_type", "phase"), list(EXPECTED_PHASES.items()), ids=lambda value: str(value))
def test_phase_defaults(error_type, phase):
    assert error_type.phase == phase
    assert str(error_type("boom")).startswith(f"[weight-stack][{phase}] boom")


def test_point_scoped_errors_take_an_instance_phase():
    error = GroupConfirmationError("ranks disagree", phase="point2_ready")
    assert error.phase == "point2_ready"
    assert str(error).startswith("[weight-stack][point2_ready]")

    timeout = ConfirmationTimeoutError("rank 1 is missing", phase="point0_topology")
    assert timeout.phase == "point0_topology"


def test_error_summary_is_picklable_and_truncated():
    error = WriteError("x" * (MAX_ERROR_MESSAGE_CHARS * 2))
    error.stage_id = 1
    error.rank = 2
    error.world_size = 4
    summary = error.summary()

    assert isinstance(summary, ErrorSummary)
    assert summary.error_type == "WriteError"
    assert summary.phase == "write"
    assert summary.rank == 2
    assert len(summary.message) == MAX_ERROR_MESSAGE_CHARS
    assert summary.message.startswith("[weight-stack][stage 1][rank 2/4][write] xxx")

    restored = pickle.loads(pickle.dumps(summary))
    assert restored == summary


def test_error_summary_uses_unknown_rank_sentinel():
    summary = ConfigError("bad value").summary()
    assert summary.rank == -1


def test_step_error_mapping_matches_rfc_table():
    # RFC 01 section 3: every startup step wraps non-WeightStackError exceptions
    # into exactly this type.  The table is pinned here verbatim so a wrong
    # mapping cannot hide behind the implementation's own constant.
    expected = {
        "construct_modules": StackUnavailableError,
        "build_structure": ConstructionError,
        "plan": PlanningError,
        "preflight": PreflightError,
        "point0_topology": GroupConfirmationError,
        "point1_fingerprints": GroupConfirmationError,
        "confirm": GroupConfirmationError,
        "point1_elections": GroupConfirmationError,
        "admit": GroupConfirmationError,
        "point2_ready": GroupConfirmationError,
        "point3_commit": GroupConfirmationError,
        "allocate": AllocationError,
        "connect_writer": AllocationError,
        "prepare": PrepareError,
        "write": WriteError,
        "finalize": FinalizeError,
        "commit": CommitError,
    }
    assert dict(STEP_ERROR_TYPES) == expected


def test_wrap_step_error_preserves_cause_and_phase():
    original = ValueError("checkpoint shard is missing")
    wrapped = wrap_step_error("write", original)

    assert type(wrapped) is WriteError
    assert wrapped.phase == "write"
    assert wrapped.__cause__ is original
    assert "ValueError: checkpoint shard is missing" in str(wrapped)
    assert str(wrapped).startswith("[weight-stack][write]")


def test_wrap_step_error_returns_weight_stack_error_unchanged():
    original = PlanningError("no repeated blocks")
    assert wrap_step_error("plan", original) is original


def test_wrap_step_error_rejects_unknown_step():
    original = RuntimeError("boom")
    with pytest.raises(ValueError, match="unknown weight-stack startup step"):
        wrap_step_error("not-a-step", original)
