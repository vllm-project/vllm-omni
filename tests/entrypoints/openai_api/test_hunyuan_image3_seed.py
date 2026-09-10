# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""HunyuanImage3 request seed overrides through real pipeline metadata."""

from types import SimpleNamespace

import pytest
from vllm.sampling_params import SamplingParams

from vllm_omni.inputs.data import OmniDiffusionSamplingParams

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


@pytest.fixture
def serving_chat():
    from vllm_omni.entrypoints.openai.serving_chat import OmniOpenAIServingChat

    return object.__new__(OmniOpenAIServingChat)


@pytest.mark.parametrize(
    "pipeline_name", ["HUNYUAN_IMAGE3_PIPELINE", "HUNYUAN_IMAGE3_AR_PIPELINE", "HUNYUAN_IMAGE3_PD_PIPELINE"]
)
@pytest.mark.parametrize("request_seed", [43, 0, None])
def test_hunyuan_request_seed_uses_real_pipeline_ownership(serving_chat, pipeline_name, request_seed):
    from vllm_omni.config.omni_config import BaseVllmOmniStageConfig
    from vllm_omni.entrypoints.pd_utils import PDDisaggregationMixin
    from vllm_omni.model_executor.models.hunyuan_image3 import pipeline as hy3_pipeline

    serving_chat._diffusion_extra_body_params = frozenset()
    topology = getattr(hy3_pipeline, pipeline_name)
    stages = []
    defaults = []
    for stage in topology.stages:
        # Exercise the real structured-config properties and real topology.
        # No model/runtime is needed to project stage_type/is_comprehension.
        config = object.__new__(BaseVllmOmniStageConfig)
        config.stage_pipeline_config = stage
        stages.append(config)
        defaults.append(
            SamplingParams(temperature=1, seed=42, max_tokens=32)
            if config.stage_type == "llm"
            else OmniDiffusionSamplingParams(seed=42)
        )
    engine = SimpleNamespace(
        stage_configs=stages,
        default_sampling_params_list=defaults,
        _pd_separation_pair=PDDisaggregationMixin.detect_pd_separation_from_stage_configs(stages),
    )
    _, resolved = serving_chat._build_multistage_generation_inputs(
        engine=engine,
        prompt="generate a sleeping cat",
        extra_body={},
        reference_images=[],
        gen_params=OmniDiffusionSamplingParams(seed=request_seed),
    )
    expected_seed = 42 if request_seed is None else request_seed
    for stage, params in zip(stages, resolved):
        if stage.stage_type == "llm":
            assert params.seed == expected_seed
            assert params.temperature == 1 and params.max_tokens == 32
    assert all(params.seed == 42 for params in defaults)
