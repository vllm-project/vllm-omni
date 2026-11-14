# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Frame-v5 duplex topology on official session/plugin and stage engines."""

from dataclasses import replace

from vllm_omni.config.stage_config import PipelineConfig, StageExecutionType, StagePipelineConfig

_PROCESSOR = "vllm_omni.model_executor.stage_input_processors.duo"
DUO_PIPELINE = PipelineConfig(
    model_type="duo",
    model_arch="DuoIntentForConditionalGeneration",
    default_deploy_config_name="duo_dmd.yaml",
    duplex_plugin="vllm_omni.model_executor.models.duo.duplex.plugin.DuoDuplexPlugin",
    stages=(
        StagePipelineConfig(
            stage_id=0,
            model_stage="intent",
            execution_type=StageExecutionType.LLM_AR,
            owns_tokenizer=True,
            requires_multimodal_data=True,
            model_arch="DuoIntentForConditionalGeneration",
            engine_output_type="text",
        ),
        StagePipelineConfig(
            stage_id=1,
            model_stage="realization",
            execution_type=StageExecutionType.DIFFUSION,
            input_sources=(0,),
            model_arch="DuoPipeline",
            custom_process_input_func=f"{_PROCESSOR}.intent_to_realization",
            inline_diffusion=True,
        ),
        StagePipelineConfig(
            stage_id=2,
            model_stage="codec",
            execution_type=StageExecutionType.DIFFUSION,
            input_sources=(1,),
            model_arch="DuoCodecPipeline",
            final_output=True,
            final_output_type="video",
            custom_process_input_func=f"{_PROCESSOR}.realization_to_codec",
            inline_diffusion=True,
        ),
    ),
)


DUO_ASR_PIPELINE = PipelineConfig(
    model_type="duo_asr",
    model_arch="Qwen3ASRForConditionalGeneration",
    default_deploy_config_name="duo_tp8_shared.yaml",
    duplex_plugin=DUO_PIPELINE.duplex_plugin,
    stages=(
        StagePipelineConfig(
            stage_id=0,
            model_stage="asr",
            execution_type=StageExecutionType.LLM_AR,
            input_sources=(),
            requires_multimodal_data=True,
            engine_output_type="text",
            model_arch="Qwen3ASRForConditionalGeneration",
            sampling_constraints={"detokenize": True},
            bypass_without_modalities=("audio",),
        ),
        *(
            replace(
                s,
                stage_id=s.stage_id + 1,
                input_sources=tuple(i + 1 for i in s.input_sources) if s.input_sources else (0,),
            )
            for s in DUO_PIPELINE.stages
        ),
    ),
)
