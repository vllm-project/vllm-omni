# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from typing import cast

import pytest
import torch

from vllm_omni.diffusion.models.glm_image.pipeline_glm_image import GlmImagePipeline

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]


@pytest.mark.parametrize(
    ("do_cfg", "expected_forwards_per_step"),
    [pytest.param(True, 2, id="sequential-cfg"), pytest.param(False, 1, id="no-cfg")],
)
def test_diffuse_sets_cache_dit_cfg_mode_for_request(
    monkeypatch: pytest.MonkeyPatch,
    do_cfg: bool,
    expected_forwards_per_step: int,
) -> None:
    events = []

    class TinyTransformer:
        dtype = torch.float32

        def __call__(self, *, hidden_states, **kwargs):
            events.append("forward")
            return (torch.zeros_like(hidden_states),)

    class TinyScheduler:
        @staticmethod
        def step(noise, timestep, latents, return_dict=False):
            return (latents,)

    pipeline = cast(GlmImagePipeline, SimpleNamespace(transformer=TinyTransformer(), scheduler=TinyScheduler()))
    monkeypatch.setattr(
        "vllm_omni.diffusion.models.glm_image.pipeline_glm_image.set_separate_cfg",
        lambda transformer, do_true_cfg: events.append(("cfg-mode", transformer, do_true_cfg)),
    )
    monkeypatch.setattr(
        "vllm_omni.diffusion.models.glm_image.pipeline_glm_image.get_classifier_free_guidance_world_size",
        lambda: 1,
    )

    GlmImagePipeline.diffuse(
        pipeline,
        latents=torch.zeros(1, 1, 2, 2),
        prior_token_id=torch.zeros(1, 1, dtype=torch.long),
        prompt_embeds=torch.zeros(1, 1, 2),
        negative_prompt_embeds=torch.zeros(1, 1, 2) if do_cfg else None,
        timesteps=torch.tensor([1.0, 0.5]),
        target_size=torch.tensor([[16.0, 16.0]]),
        crop_coords=torch.zeros(1, 2),
        guidance_scale=4.0 if do_cfg else 1.0,
        do_classifier_free_guidance=do_cfg,
    )

    assert events[0] == ("cfg-mode", pipeline.transformer, do_cfg)
    assert events[1:] == ["forward"] * (2 * expected_forwards_per_step)
