# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm_omni.diffusion.models.sensenova_u1.pipeline_sensenova_u1 import SenseNovaU1Pipeline


def _pipeline_without_init() -> SenseNovaU1Pipeline:
    return object.__new__(SenseNovaU1Pipeline)


@pytest.mark.parametrize("kwargs", [None, {"step_i": 0}])
def test_combine_cfg_noise_requires_is_it2i(kwargs):
    pipe = _pipeline_without_init()
    out_cond = (torch.ones(1, 2, 3),)
    out_uncond = (torch.zeros(1, 2, 3),)

    with pytest.raises(ValueError, match="is_it2i"):
        pipe.combine_cfg_noise(
            out_cond,
            out_uncond,
            cfg_scale=4.0,
            cfg_norm="cfg_zero_star",
            kwargs=kwargs,
        )


@pytest.mark.parametrize(
    "kwargs",
    [{"is_it2i": False}, {"is_it2i": False, "step_i": None}],
)
def test_cfg_zero_star_requires_step_i(kwargs):
    pipe = _pipeline_without_init()
    out_cond = (torch.ones(1, 2, 3),)
    out_uncond = (torch.zeros(1, 2, 3),)
    with pytest.raises(ValueError, match="step_i"):
        pipe.combine_cfg_noise(
            out_cond,
            out_uncond,
            cfg_scale=4.0,
            cfg_norm="cfg_zero_star",
            kwargs=kwargs,
        )


def test_cfg_zero_star_accepts_step_i():
    pipe = _pipeline_without_init()
    out_cond = (torch.ones(1, 2, 3),)
    out_uncond = (torch.zeros(1, 2, 3),)
    result = pipe.combine_cfg_noise(
        out_cond,
        out_uncond,
        cfg_scale=4.0,
        cfg_norm="cfg_zero_star",
        kwargs={"is_it2i": False, "step_i": 0},
    )

    assert result.shape == out_cond[0].shape
    assert torch.equal(result, torch.zeros_like(out_cond[0]))
    assert torch.isfinite(result).all()
