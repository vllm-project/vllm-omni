# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from types import SimpleNamespace

import pytest
import torch

from examples.offline_inference.cosmos3_nano_sim_auto.cosmos3_nano_sim_auto import _save_latents

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]


@pytest.mark.parametrize("joint", [False, True])
def test_offline_saves_reference_latent_format(tmp_path, joint: bool) -> None:
    latents = {"vision_latent": torch.arange(16, dtype=torch.bfloat16).reshape(1, 2, 2, 2, 2)}
    if joint:
        latents["lidar_latent"] = latents["vision_latent"] + 2
    output = [SimpleNamespace(multimodal_output={"latents": latents})]
    path = _save_latents(output, tmp_path, joint=joint)
    saved = torch.load(path, weights_only=True)
    assert len(saved) == 1 and saved[0].keys() == latents.keys()
    for name, expected in latents.items():
        torch.testing.assert_close(saved[0][name], expected, rtol=0, atol=0)
    with pytest.raises(ValueError, match="requested latents"):
        _save_latents({"payload": {"latents": {}}}, tmp_path, joint=joint)
