# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from types import SimpleNamespace

import pytest
import torch

from vllm_omni.diffusion.models.minimax_h3 import race

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]


def test_prediction_uses_two_true_endpoints_and_corrects_before_observing_refresh():
    trajectory = race.RACETrajectory()

    def state(value):
        return torch.tensor([[value]], dtype=torch.float32)

    trajectory.step(state(0), state(2), 0.0, 0.2)
    value, correction = trajectory.step(state(1), None, 0.2, 0.4)
    torch.testing.assert_close(value, state(2))
    assert correction is None

    _, correction = trajectory.step(state(2), state(6), 0.4, 0.5)
    torch.testing.assert_close(correction, state(0.4))
    value, correction = trajectory.step(state(3), None, 0.5, 0.8)
    torch.testing.assert_close(value, state(8))
    assert correction is None

    _, correction = trajectory.step(state(4), state(12), 0.8, 1.0)
    torch.testing.assert_close(correction, state(0.15))


def test_zero_state_secant_uses_latest_velocity():
    trajectory = race.RACETrajectory()
    state = torch.zeros(2, 3)
    latest = torch.full_like(state, 3)
    trajectory.step(state, torch.ones_like(state), 0.0, 0.2)
    trajectory.step(state, latest, 0.2, 0.4)
    value, correction = trajectory.step(state + 1, None, 0.4, 0.6)
    torch.testing.assert_close(value, latest)
    assert correction is None


def test_gate_accumulates_relative_change_and_forces_both_endpoints(monkeypatch):
    tp_group = SimpleNamespace(rank_in_group=0, world_size=1)
    monkeypatch.setattr(race, "get_tp_group", lambda: tp_group)
    filtered = []

    def identity_filter(value, sigma, *, dimensions):
        filtered.append(dimensions)
        return value

    monkeypatch.setattr(race, "apply_sea_filter", identity_filter)
    branch = SimpleNamespace(
        device="cpu",
        static_kwargs={"video_token_layout": SimpleNamespace(video_spans=(), latent_grid=(1, 1, 1))},
        img_pos_dev=torch.tensor([0]),
        update_mask_dev=torch.tensor([True]),
    )
    gate = race.MiniMaxH3RACE(model=None, branch=branch, num_steps=6)
    probed = []

    def probe(values):
        probed.append(values["probe"])
        return torch.tensor([values["probe"]], dtype=torch.bfloat16)

    monkeypatch.setattr(gate, "probe", probe)
    refreshes = [gate.should_refresh(i, 0.8, {"probe": value}) for i, value in enumerate([1, 1.1, 1.2, 1.5, 1.5, 1.5])]
    assert refreshes == [True, False, False, True, False, True]
    assert gate.refresh_steps == [0, 3, 5]
    assert probed == [1, 1.1, 1.2, 1.5, 1.5, 1.5]
    assert filtered == [(2, 1, 0)] * 4
