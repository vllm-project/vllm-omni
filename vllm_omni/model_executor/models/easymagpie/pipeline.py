# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""EasyMagpie: Nemotron-H and local transformer followed by a causal codec."""

from vllm_omni.config.stage_config import PipelineConfig, StageExecutionType, StagePipelineConfig

_PROC = "vllm_omni.model_executor.stage_input_processors.easymagpie"

EASYMAGPIE_PIPELINE = PipelineConfig(
    model_type="easymagpie",
    model_arch="EasyMagpieTTSForConditionalGeneration",
    hf_architectures=("EasyMagpieTTSForConditionalGeneration", "EasyMagpieTTS"),
    default_deploy_config_name="easymagpie.yaml",
    stages=(
        StagePipelineConfig(
            stage_id=0,
            model_stage="easymagpie",
            execution_type=StageExecutionType.LLM_AR,
            input_sources=(),
            owns_tokenizer=True,
            engine_output_type="latent",
            scheduler_cls="vllm_omni.model_executor.models.easymagpie.scheduler.EasyMagpieARAsyncScheduler",
            async_chunk_process_next_stage_input_func=f"{_PROC}.talker2code2wav_async_chunk",
            custom_process_next_stage_input_func=f"{_PROC}.talker2code2wav_full_payload",
            sampling_constraints={"detokenize": False, "stop_token_ids": [1]},
            recompute_preemption="fail",
        ),
        StagePipelineConfig(
            stage_id=1,
            model_stage="easymagpie_codec",
            execution_type=StageExecutionType.LLM_GENERATION,
            input_sources=(0,),
            retains_state_across_chunks=True,
            final_output=True,
            final_output_type="audio",
            engine_output_type="audio",
            model_arch="EasyMagpieCodecForConditionalGeneration",
            model_subdir="codec_native",
            scheduler_cls="vllm_omni.model_executor.models.easymagpie.scheduler.EasyMagpieCodecScheduler",
            sync_process_input_func=f"{_PROC}.talker2code2wav_token_only",
            sampling_constraints={"detokenize": True},
        ),
    ),
)
