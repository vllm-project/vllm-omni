# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from types import SimpleNamespace

import pytest
import torch

from vllm_omni.diffusion.models.minimax_h3 import race

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]


@pytest.mark.parametrize(
    "config, expected",
    [
        (None, 0.3),
        ({}, 0.3),
        ({"minimax_h3_race_threshold": 0}, 0.0),
        (SimpleNamespace(minimax_h3_race_threshold=0.4), 0.4),
    ],
)
def test_race_threshold(config, expected):
    assert race.minimax_h3_race_threshold(config) == expected


@pytest.mark.parametrize("value", [True, -0.01, float("inf"), float("nan"), "0.4"])
def test_invalid_race_threshold(value):
    with pytest.raises(ValueError, match="minimax_h3_race_threshold"):
        race.minimax_h3_race_threshold({"minimax_h3_race_threshold": value})


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


@pytest.mark.parametrize("rank", [0, 1])
def test_sp_gate_uses_global_change_over_channel_shards(monkeypatch, rank):
    # The local distances straddle the threshold. Both ranks must use the
    # sum over all channels before deciding whether to enter the native DiT.
    tp_group = SimpleNamespace(rank_in_group=0, world_size=1)
    sp_group = SimpleNamespace(rank_in_group=rank, world_size=2)
    monkeypatch.setattr(race, "get_tp_group", lambda: tp_group)
    monkeypatch.setattr(race, "get_sp_group", lambda: sp_group)
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    branch = SimpleNamespace(
        device="cpu",
        static_kwargs={"video_token_layout": SimpleNamespace(video_spans=(), latent_grid=(1, 1, 1))},
        img_pos_dev=torch.tensor([0]),
        update_mask_dev=torch.tensor([True]),
    )
    gate = race.MiniMaxH3RACE(model=SimpleNamespace(hidden_size=4), branch=branch, num_steps=4)
    monkeypatch.setattr(gate, "probe", lambda inputs: inputs["values"])
    start = torch.ones(1, 1, 1, 4, dtype=torch.bfloat16)
    changed = torch.tensor([1, 1, 2, 2], dtype=torch.bfloat16).view_as(start)

    def reduce_stats(local):
        assert local.tolist() == [2.0 * rank, 2.0]
        return local.new_tensor([2, 4])

    sp_group.all_reduce = reduce_stats
    assert gate.should_refresh(0, 1.0, {"values": start})
    assert gate.should_refresh(1, 0.9, {"values": changed})
    assert gate.previous.shape == (1, 1, 1, 2)
    assert gate.previous.dtype == torch.float32
    assert gate.should_refresh(3, 0.1, {"values": changed})
