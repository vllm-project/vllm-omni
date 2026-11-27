# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Turn-taking policy (framework-independent core): defaults, validation and per-session overrides."""

import ast
import pathlib

import pytest

from vllm_omni.model_executor.models.sophon_realtime.core import policy
from vllm_omni.model_executor.models.sophon_realtime.core.policy import DEFAULT_POLICY, TurnPolicy, validate_session

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def test_policy_module_imports_no_framework():
    tree = ast.parse(pathlib.Path(policy.__file__).read_text())
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(a.name.split(".")[0] for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.add(node.module.split(".")[0])
    assert not names & {"vllm", "vllm_omni", "torch"}


def test_checkpoints_and_thresholds_must_pair_up_and_increase():
    with pytest.raises(ValueError):
        TurnPolicy(checks=(0.2, 0.5), thresholds=(0.9,))
    with pytest.raises(ValueError):
        TurnPolicy(checks=(0.5, 0.2), thresholds=(0.9, 0.5))
    with pytest.raises(ValueError):
        TurnPolicy(thresholds=(1.5, 0.85, 0.7, 0.5, 0.35, 0.25))


def test_negative_or_non_finite_durations_are_refused():
    with pytest.raises(ValueError):
        TurnPolicy(fallback_s=-1.0)
    with pytest.raises(ValueError):
        TurnPolicy(merge_window_s=float("nan"))


def test_session_overrides_change_only_their_fields():
    p = DEFAULT_POLICY.with_session({"turn_fallback_s": 1.2, "min_release_s": 0.3, "unrelated": 5})
    assert (p.fallback_s, p.min_release_s) == (1.2, 0.3)
    assert p.min_release_after_s == DEFAULT_POLICY.min_release_after_s
    assert p.checks == DEFAULT_POLICY.checks
    assert DEFAULT_POLICY.with_session(None) is DEFAULT_POLICY
    assert DEFAULT_POLICY.with_session({}) is DEFAULT_POLICY


@pytest.mark.parametrize(
    "extra_body",
    [
        {"turn_fallback_s": 0.0},
        {"turn_fallback_s": 99},
        {"min_release_s": float("inf")},
        {"min_release_after_s": "1"},
        {"turn_fallback_s": True},
    ],
)
def test_out_of_range_session_overrides_are_reported(extra_body):
    assert validate_session(extra_body) is not None


def test_valid_or_absent_session_overrides_pass():
    assert validate_session(None) is None
    assert validate_session({"language": "en"}) is None
    assert validate_session({"turn_fallback_s": 1.5, "min_release_s": 0, "min_release_after_s": 2}) is None
