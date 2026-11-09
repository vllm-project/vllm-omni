# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Request-level seed propagation into DreamZero's initial noise generators.

The shared robot_policy example forwards ``--seed`` onto each request's
sampling params, and DreamZeroPipeline resolves it with priority
``extra_args["seed"]`` → ``sampling_params.seed`` → the model-config default.
"""

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from vllm_omni.diffusion.models.dreamzero.pipeline_dreamzero import _resolve_request_seed
from vllm_omni.inputs.data import OmniDiffusionSamplingParams

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def test_resolve_request_seed_priority() -> None:
    assert _resolve_request_seed({"seed": 7}, SimpleNamespace(seed=123), 42) == 7
    assert _resolve_request_seed({}, SimpleNamespace(seed=123), 42) == 123
    assert _resolve_request_seed({}, SimpleNamespace(seed=None), 42) == 42


def test_resolve_request_seed_keeps_explicit_zero() -> None:
    assert _resolve_request_seed({"seed": 0}, SimpleNamespace(seed=9), 42) == 0
    assert _resolve_request_seed({}, SimpleNamespace(seed=0), 42) == 0


def _load_robot_policy_module():
    """Load the shared example by file path (``examples/`` is not a package)."""
    repo_root = Path(__file__).resolve().parents[4]
    path = repo_root / "examples/offline_inference/robot_policy/robot_policy.py"
    spec = importlib.util.spec_from_file_location("_test_robot_policy", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_robot_policy_run_inference_threads_cli_seed(monkeypatch: pytest.MonkeyPatch) -> None:
    robot_policy = _load_robot_policy_module()
    monkeypatch.setattr(robot_policy, "process_robot_actions", lambda *args, **kwargs: {})

    captured: list[OmniDiffusionSamplingParams] = []

    class StubOmni:
        def generate(self, prompt, sampling_params_list):
            captured.extend(sampling_params_list)
            return [object()]

    observations = [{"prompt": "task", "session_id": "s", "reset": True, "robot_obs": {}}]
    generator = torch.Generator(device="cpu").manual_seed(0)

    robot_policy.run_inference(StubOmni(), generator, 42, "DreamZeroPipeline", observations, {}, None)
    robot_policy.run_inference(StubOmni(), generator, 123, "DreamZeroPipeline", observations, {}, None)

    assert [sp.seed for sp in captured] == [42, 123]
