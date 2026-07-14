# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import torch

from vllm_omni.diffusion.models.ltx2.ltx2_components import (
    LTX2_COMPONENT_PROFILE,
    LTX23_COMPONENT_PROFILE,
    create_transformer_from_config,
)
from vllm_omni.diffusion.models.ltx2.ltx2_pipeline_base import LTXPipelineBase
from vllm_omni.diffusion.models.ltx2.ltx2_recipes import LTX2_ONE_STAGE_RECIPE, LTX23_ONE_STAGE_RECIPE
from vllm_omni.diffusion.models.ltx2.ltx2_stage import denoise_stage
from vllm_omni.diffusion.models.ltx2.pipeline_ltx2 import LTX2Pipeline
from vllm_omni.diffusion.models.ltx2.pipeline_ltx2_3 import LTX23Pipeline


def test_ltx_versions_share_runtime_without_cross_version_inheritance():
    from vllm_omni.diffusion.models.ltx2 import pipeline_ltx2, pipeline_ltx2_3

    assert issubclass(LTX2Pipeline, LTXPipelineBase)
    assert issubclass(LTX23Pipeline, LTXPipelineBase)
    assert not issubclass(LTX23Pipeline, LTX2Pipeline)
    assert LTX2Pipeline._pack_latents is LTX23Pipeline._pack_latents
    assert LTX2Pipeline.component_profile is LTX2_COMPONENT_PROFILE
    assert LTX23Pipeline.component_profile is LTX23_COMPONENT_PROFILE
    assert LTX2Pipeline.one_stage_recipe is LTX2_ONE_STAGE_RECIPE
    assert LTX23Pipeline.one_stage_recipe is LTX23_ONE_STAGE_RECIPE
    assert pipeline_ltx2.create_transformer_from_config is create_transformer_from_config
    assert pipeline_ltx2_3.create_transformer_from_config is create_transformer_from_config


def test_denoise_stage_owns_progress_interrupt_and_current_timestep():
    updates = []

    class Progress:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, traceback):
            return False

        def update(self):
            updates.append(True)

    pipeline = SimpleNamespace(
        _current_timestep=None,
        interrupt=False,
        progress_bar=lambda total: Progress(),
    )
    seen = []
    timesteps = torch.tensor([3.0, 2.0, 1.0])

    with denoise_stage(pipeline, timesteps) as (steps, progress):
        for index, timestep in steps:
            seen.append((index, timestep.item(), pipeline._current_timestep.item()))
            progress.update()
            if index == 0:
                pipeline.interrupt = True

    assert seen == [(0, 3.0, 3.0)]
    assert updates == [True]
