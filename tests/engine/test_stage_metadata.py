import copy
import operator

import pytest
from vllm.sampling_params import SamplingParams

from vllm_omni.config.omni_config import VllmOmniConfig
from vllm_omni.config.stage_config import (
    DeployConfig,
    PipelineConfig,
    StageDeployConfig,
    StageExecutionType,
    StagePipelineConfig,
)
from vllm_omni.engine.stage_init_utils import (
    extract_stage_metadata,
)
from vllm_omni.inputs.data import OmniDiffusionSamplingParams

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _metadata_inputs() -> tuple[PipelineConfig, DeployConfig]:
    pipeline = PipelineConfig(
        model_type="metadata-test",
        stages=(
            StagePipelineConfig(
                stage_id=0,
                model_stage="thinker",
                execution_type=StageExecutionType.LLM_AR,
                owns_tokenizer=True,
                requires_multimodal_data=True,
                engine_output_type="token_ids",
                custom_process_input_func="operator.add",
                prompt_expand_func="operator.mul",
            ),
            StagePipelineConfig(
                stage_id=1,
                model_stage="talker",
                execution_type=StageExecutionType.LLM_GENERATION,
                input_sources=(0,),
                engine_output_type="audio_tokens",
                custom_process_input_func="operator.sub",
                sync_process_input_func="operator.floordiv",
            ),
            StagePipelineConfig(
                stage_id=2,
                model_stage="dit",
                execution_type=StageExecutionType.DIFFUSION,
                input_sources=(1,),
                final_output=True,
                final_output_type="image",
                custom_process_input_func="operator.neg",
                cfg_kv_collect_func="operator.concat",
            ),
        ),
    )
    deploy = DeployConfig(
        async_chunk=False,
        stages=[
            StageDeployConfig(
                stage_id=0,
                devices="0",
                env={"STAGE": "thinker"},
                default_sampling_params={"temperature": 0.25},
            ),
            StageDeployConfig(
                stage_id=1,
                devices="1,2",
                num_replicas=2,
                env={"STAGE": "talker"},
                default_sampling_params={"temperature": 0.75},
            ),
            StageDeployConfig(
                stage_id=2,
                devices="3",
                env={"STAGE": "dit"},
                default_sampling_params={"seed": 7},
            ),
        ],
    )
    return pipeline, deploy


def test_extract_stage_metadata_projects_runtime_fields():
    pipeline, deploy = _metadata_inputs()
    omni_config = VllmOmniConfig.from_pipeline_config(
        pipeline,
        user_deploy_config=copy.deepcopy(deploy),
    )
    for stage in omni_config.stage_configs:
        metadata = extract_stage_metadata(stage)
        assert metadata.stage_id == stage.stage_id
        assert metadata.engine_input_source == stage.input_sources
        assert metadata.runtime_cfg is stage.runtime_config
        assert metadata.final_output == stage.final_output
    assert extract_stage_metadata(omni_config.stage_by_id(0)).engine_input_source == []

    thinker = extract_stage_metadata(omni_config.stage_by_id(0))
    talker = extract_stage_metadata(omni_config.stage_by_id(1))
    diffusion = extract_stage_metadata(omni_config.stage_by_id(2))

    assert isinstance(thinker.default_sampling_params, SamplingParams)
    assert thinker.default_sampling_params.temperature == 0.25
    assert thinker.custom_process_input_func is operator.add
    assert thinker.prompt_expand_func is operator.mul

    assert isinstance(talker.default_sampling_params, SamplingParams)
    assert talker.default_sampling_params.temperature == 0.75
    assert talker.custom_process_input_func is operator.floordiv

    assert isinstance(diffusion.default_sampling_params, OmniDiffusionSamplingParams)
    assert diffusion.default_sampling_params.seed == 7
    assert diffusion.custom_process_input_func is operator.neg
    assert diffusion.cfg_kv_collect_func is operator.concat
