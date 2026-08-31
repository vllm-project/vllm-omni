# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Pure-function tests for PidInferenceModel (t schedule + velocity->x0 math).

``PidInferenceModel.__init__`` constructs a real ``PidNet`` + downloads the
Gemma encoder, so these tests bypass ``__init__`` via ``object.__new__``.
"""

import pytest
import torch

from vllm_omni.diffusion.pid.config import PID_SAMPLING_CONFIG
from vllm_omni.diffusion.pid.pid_model import PidInferenceModel

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]


def _bare_model():
    """Bypass __init__ (no PidNet / Gemma), expose only the sampling config."""
    model = object.__new__(PidInferenceModel)
    model._cfg = type("Cfg", (), dict(PID_SAMPLING_CONFIG))()
    return model


def test_t_list_default_matches_config():
    model = _bare_model()
    t_list = model._get_t_list(torch.device("cpu"), num_steps=4)
    assert t_list.shape == (5,)
    assert torch.isclose(t_list[0], torch.tensor(0.999))


def test_t_list_resampled_for_other_steps():
    model = _bare_model()
    t_list = model._get_t_list(torch.device("cpu"), num_steps=2)
    assert t_list.shape == (3,)


def test_velocity_to_x0_math():
    model = _bare_model()
    x_t = torch.tensor([1.0, 2.0, 3.0])
    v = torch.tensor([0.5, 0.5, 0.5])
    t = torch.tensor([0.5, 0.5, 0.5])
    x0 = model._velocity_to_x0(x_t, v, t)  # x0 = x_t - t * v
    torch.testing.assert_close(x0, torch.tensor([0.75, 1.75, 2.75]))
