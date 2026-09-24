# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

from vllm_omni.diffusion.pid import get_pid_net_config

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]


@pytest.mark.parametrize(
    "backbone, lq_channels, down_factor",
    [
        ("qwenimage", 16, 8),
        ("flux", 16, 8),
        ("sd3", 16, 8),
        ("sdxl", 4, 8),
        ("flux2", 128, 16),
    ],
)
def test_get_pid_net_config_known_backbones(backbone, lq_channels, down_factor):
    cfg = get_pid_net_config(backbone)
    assert cfg["lq_latent_channels"] == lq_channels
    assert cfg["latent_spatial_down_factor"] == down_factor
    assert get_pid_net_config(backbone) is not cfg


def test_get_pid_net_config_unknown_backbone_raises():
    with pytest.raises(ValueError, match="Unknown backbone"):
        get_pid_net_config("not_a_backbone")
