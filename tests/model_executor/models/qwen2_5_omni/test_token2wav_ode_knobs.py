# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Tests for the Qwen2.5-Omni code2wav ODE solver knobs.

Covers three layers:

1. knob resolution — absent keys keep the released recipe, the shipped deploy
   config resolves to the reduced-step default, and invalid values are
   rejected at startup;
2. integrator math — euler/midpoint/rk4 solve a known ODE at their documented
   convergence order;
3. plumbing invariants — the ``rk4`` knob resolves to the untouched released
   solver class and the eval-count arithmetic the config comments rely on.
"""

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import yaml

from vllm_omni.model_executor.models.qwen2_5_omni.qwen2_5_omni_token2wav import (
    _DEFAULT_DIT_NUM_STEPS,
    _ODE_SOLVERS,
    EulerODESolver,
    MidpointODESolver,
    RungeKutta4ODESolver,
    _build_ode_solver,
    _resolve_dit_solver_knobs,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

RELEASED_RECIPE = {
    "num_steps": 10,
    "guidance_scale": 0.5,
    "sway_coefficient": -1.0,
    "integrator": "rk4",
}


def _config(**overrides) -> SimpleNamespace:
    """A token2wav config carrying exactly the given knob overrides."""
    cfg = SimpleNamespace()
    for key, value in overrides.items():
        setattr(cfg, key, value)
    return cfg


# ---------------------------------------------------------------------------
# 1. knob resolution
# ---------------------------------------------------------------------------


def test_absent_knobs_keep_released_recipe():
    assert _resolve_dit_solver_knobs(_config()) == RELEASED_RECIPE


def test_override_values_are_resolved_and_normalized():
    knobs = _resolve_dit_solver_knobs(_config(dit_num_steps=6, dit_guidance_scale=1, dit_sway_coefficient=0))
    assert knobs == {"num_steps": 6, "guidance_scale": 1.0, "sway_coefficient": 0.0, "integrator": "rk4"}


def test_sway_null_disables_sway_sampling():
    knobs = _resolve_dit_solver_knobs(_config(dit_sway_coefficient=None))
    assert knobs["sway_coefficient"] is None


@pytest.mark.parametrize(
    "overrides",
    [
        {"dit_num_steps": 1},
        {"dit_num_steps": True},
        {"dit_num_steps": "6"},
        {"dit_guidance_scale": -0.5},
        {"dit_guidance_scale": True},
        {"dit_sway_coefficient": "strong"},
        {"dit_integrator": "dopri5"},
        {"dit_integrator": 4},
    ],
)
def test_invalid_knobs_are_rejected(overrides):
    with pytest.raises(ValueError, match="dit_"):
        _resolve_dit_solver_knobs(_config(**overrides))


def test_deploy_config_resolves_to_reduced_step_default():
    """The shipped qwen2_5_omni.yaml must land dit_num_steps on the code2wav
    stage's nested token2wav config, where the wrapper reads it."""
    deploy_path = Path(__file__).parents[4] / "vllm_omni" / "deploy" / "qwen2_5_omni.yaml"
    deploy = yaml.safe_load(deploy_path.read_text(encoding="utf-8"))
    stage2 = next(s for s in deploy["stages"] if s["stage_id"] == 2)
    token2wav_overrides = stage2["hf_overrides"]["token2wav_config"]
    knobs = _resolve_dit_solver_knobs(_config(**token2wav_overrides))
    assert knobs["num_steps"] == 6
    # Everything the YAML leaves commented stays on the released recipe.
    assert knobs["guidance_scale"] == RELEASED_RECIPE["guidance_scale"]
    assert knobs["sway_coefficient"] == RELEASED_RECIPE["sway_coefficient"]
    assert knobs["integrator"] == RELEASED_RECIPE["integrator"]


# ---------------------------------------------------------------------------
# 2. integrator math (dy/dt = y on [0, 1]; solution is exp(t))
# ---------------------------------------------------------------------------


def _exp_solver(cls):
    return cls(function=lambda t, y: y, initial_value=torch.ones(3, dtype=torch.float64))


def _error_at(num_points: int) -> dict[str, float]:
    errors = {}
    for name, cls in (("euler", EulerODESolver), ("midpoint", MidpointODESolver), ("rk4", RungeKutta4ODESolver)):
        solution = _exp_solver(cls).integrate(torch.linspace(0.0, 1.0, num_points, dtype=torch.float64))
        errors[name] = abs(float(solution[-1].mean()) - torch.e)
    return errors


@pytest.mark.parametrize(
    ("integrator", "expected_order"),
    [("euler", 1), ("midpoint", 2), ("rk4", 4)],
)
def test_solver_convergence_order(integrator, expected_order):
    """Halving the step must shrink the error by ~2^order."""
    coarse = _error_at(17)[integrator]
    fine = _error_at(33)[integrator]
    ratio = coarse / fine
    assert ratio == pytest.approx(2.0**expected_order, rel=0.05)


def test_solvers_record_grid_points_only():
    solution = _exp_solver(MidpointODESolver).integrate(torch.linspace(0.0, 1.0, 5))
    assert solution.shape == (5, 3)
    assert torch.equal(solution[0], torch.ones(3))


# ---------------------------------------------------------------------------
# 3. plumbing invariants
# ---------------------------------------------------------------------------


def test_rk4_knob_is_the_untouched_released_solver():
    """The default integrator must stay the original solver class so the
    no-override path remains bit-exact with the released recipe."""
    assert _ODE_SOLVERS["rk4"] is RungeKutta4ODESolver
    solver = _build_ode_solver("rk4", function=lambda t, y: y, initial_value=torch.zeros(1))
    assert type(solver) is RungeKutta4ODESolver


def test_unknown_integrator_is_rejected():
    with pytest.raises(ValueError, match="Unknown dit_integrator"):
        _build_ode_solver("leapfrog", function=lambda t, y: y, initial_value=torch.zeros(1))


@pytest.mark.parametrize(
    ("integrator", "evals_per_interval"),
    [("rk4", 4), ("midpoint", 2), ("euler", 1)],
)
def test_evals_per_interval_matches_config_comment(integrator, evals_per_interval):
    """The deploy-config comment computes evals/solve as
    (dit_num_steps - 1) * EVALS_PER_INTERVAL; keep that arithmetic true."""
    assert _ODE_SOLVERS[integrator].EVALS_PER_INTERVAL == evals_per_interval
    num_steps = 6
    evals = (num_steps - 1) * _ODE_SOLVERS[integrator].EVALS_PER_INTERVAL
    assert evals == 5 * evals_per_interval


def test_released_default_is_ten_grid_points():
    """The code-level default must stay the released recipe; the reduced-step
    default ships only through the deploy config."""
    assert _DEFAULT_DIT_NUM_STEPS == 10
