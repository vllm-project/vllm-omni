# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""CPU lock for the Qwen-Image-2.1 CFG tail slice.

2.1 prepends condition latents, so the target noise is the tail. The shared
mixin slices the head. These tests stub ``predict_noise`` and check the
returned rows are the tail, including rank-0 / rank-1 gather order.
"""

from typing import Any

import pytest
import torch

from vllm_omni.diffusion.models.qwen_image_21 import cfg_parallel as cfg_mod
from vllm_omni.diffusion.models.qwen_image_21.cfg_parallel import QwenImage21CFGParallelMixin

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]

PREFIX = 3
TARGET = 2
SCALE = 4.0


class _Stub(QwenImage21CFGParallelMixin):
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def predict_noise(self, **kwargs: Any) -> torch.Tensor:
        self.calls.append(kwargs)
        return kwargs["joint"]


class _Gather:
    def __init__(self, rank_tails: list[torch.Tensor]) -> None:
        self.rank_tails = rank_tails
        self.seen: list[torch.Tensor] = []

    def all_gather(self, input_: torch.Tensor, dim: int = 0, separate_tensors: bool = False) -> list[torch.Tensor]:
        del dim
        assert separate_tensors
        self.seen.append(input_.clone())
        return [tensor.clone() for tensor in self.rank_tails]


def _joint(fill: float) -> torch.Tensor:
    prefix = torch.full((1, PREFIX, 1), fill)
    target = torch.full((1, TARGET, 1), fill + 100.0)
    return torch.cat([prefix, target], dim=1)


def _cfg(positive: torch.Tensor, negative: torch.Tensor) -> torch.Tensor:
    return negative + SCALE * (positive - negative)


def test_single_rank_cfg_uses_target_tail_not_condition_head() -> None:
    pipe = _Stub()
    positive = _joint(1.0)
    negative = _joint(2.0)
    actual = pipe.predict_noise_maybe_with_cfg(
        do_true_cfg=True,
        true_cfg_scale=SCALE,
        positive_kwargs={"joint": positive, "branch": "pos"},
        negative_kwargs={"joint": negative, "branch": "neg"},
        output_slice=TARGET,
    )
    tail = _cfg(positive[:, -TARGET:], negative[:, -TARGET:])
    head = _cfg(positive[:, :TARGET], negative[:, :TARGET])
    torch.testing.assert_close(actual, tail)
    assert not torch.allclose(actual, head)
    assert [call["branch"] for call in pipe.calls] == ["pos", "neg"]


@pytest.mark.parametrize("rank", [0, 1])
def test_cfg_parallel_gather_order_uses_target_tails(monkeypatch: pytest.MonkeyPatch, rank: int) -> None:
    pipe = _Stub()
    positive = _joint(1.0)
    negative = _joint(2.0)
    pos_tail = positive[:, -TARGET:]
    neg_tail = negative[:, -TARGET:]
    group = _Gather([pos_tail, neg_tail])
    monkeypatch.setattr(cfg_mod, "_get_cfg_world_size_or_one", lambda: 2)
    monkeypatch.setattr(cfg_mod, "get_classifier_free_guidance_rank", lambda: rank)
    monkeypatch.setattr(cfg_mod, "get_cfg_group", lambda: group)

    actual = pipe.predict_noise_maybe_with_cfg(
        do_true_cfg=True,
        true_cfg_scale=SCALE,
        positive_kwargs={"joint": positive, "branch": "pos"},
        negative_kwargs={"joint": negative, "branch": "neg"},
        output_slice=TARGET,
    )

    expected_branch = "pos" if rank == 0 else "neg"
    expected_local = pos_tail if rank == 0 else neg_tail
    assert [call["branch"] for call in pipe.calls] == [expected_branch]
    torch.testing.assert_close(group.seen[0], expected_local)
    torch.testing.assert_close(actual, _cfg(pos_tail, neg_tail))
    head = _cfg(positive[:, :TARGET], negative[:, :TARGET])
    assert not torch.allclose(actual, head)
