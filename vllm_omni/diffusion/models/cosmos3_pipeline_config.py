# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Cosmos3 tower-disaggregated topology (T2I only).

    stage 0 (reasoner, UND tower) --per-layer text K/V--> stage 1 (generator, GEN tower + VAE)

Lives outside the ``cosmos3`` package so the orchestrator can import it without
importing the diffusion pipeline. See ``cosmos3/pipeline_cosmos3_disagg.py``.
"""

from vllm_omni.config.stage_config import (
    PipelineConfig,
    StageExecutionType,
    StagePipelineConfig,
)

#: Key under which the reasoner's K/V travels to the generator.
COSMOS3_UND_KV_KEY = "cosmos3_und_kv"

COSMOS3_DISAGG_PIPELINE = PipelineConfig(
    model_type="cosmos3_omni_disagg",
    default_deploy_config_name="cosmos3_super_t2i_disagg.yaml",
    # No hf_architectures: the checkpoint is shared with the co-located pipeline,
    # so this is selected only via ``pipeline: cosmos3_omni_disagg`` in the deploy YAML.
    model_arch="Cosmos3ForConditionalGeneration",
    stages=(
        StagePipelineConfig(
            stage_id=0,
            model_stage="reasoner",
            execution_type=StageExecutionType.DIFFUSION,
            owns_tokenizer=True,
            final_output=False,
            model_arch="Cosmos3ReasonerPipeline",
        ),
        StagePipelineConfig(
            stage_id=1,
            model_stage="generator",
            execution_type=StageExecutionType.DIFFUSION,
            input_sources=(0,),
            final_output=True,
            final_output_type="image",
            model_arch="Cosmos3GeneratorPipeline",
            custom_process_input_func="vllm_omni.model_executor.stage_input_processors.cosmos3.reasoner2generator",
            # The K/V travels in the stage payload, not the AR KV-transfer path.
            omni_kv_config={"need_recv_cache": False},
        ),
    ),
)
