# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Native Lychee-FD MRV2 AR and response-owned synthesis pipeline."""

from vllm_omni.config.stage_config import PipelineConfig, StageExecutionType, StagePipelineConfig

LYCHEE_FD_PIPELINE = PipelineConfig(
    model_type="step_audio_2_full_duplex",
    default_deploy_config_name="lychee_fd_single_gpu.yaml",
    model_arch="LycheeFullDuplexForConditionalGeneration",
    duplex_plugin="vllm_omni.model_executor.models.lychee_fd.duplex.plugin.LycheeDuplexPlugin",
    hf_architectures=(
        "StepAudio2FullDuplex",
        "LycheeFullDuplex",
        "LycheeFullDuplexForConditionalGeneration",
    ),
    stages=(
        StagePipelineConfig(
            stage_id=0,
            model_stage="lychee_fd",
            execution_type=StageExecutionType.LLM_AR,
            supports_native_preemption=False,
            supports_running_prefix_cache_reset=False,
            input_sources=(),
            final_output=True,
            final_output_type="text",
            owns_tokenizer=True,
            requires_multimodal_data=False,
            engine_output_type="latent",
            model_arch="LycheeFullDuplexForConditionalGeneration",
            sampling_constraints={"detokenize": False},
        ),
        StagePipelineConfig(
            stage_id=1,
            model_stage="lychee_token2wav",
            execution_type=StageExecutionType.LLM_GENERATION,
            input_sources=(0,),
            final_output=True,
            final_output_type="audio",
            engine_output_type="audio",
            model_arch="LycheeToken2WavForConditionalGeneration",
            sync_process_input_func="vllm_omni.model_executor.stage_input_processors.lychee_fd.lychee2token2wav",
            sampling_constraints={"detokenize": False},
        ),
    ),
)


__all__ = ["LYCHEE_FD_PIPELINE"]
