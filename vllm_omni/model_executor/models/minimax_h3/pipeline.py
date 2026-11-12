# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Two-stage MiniMax H3 topology."""

from dataclasses import replace

from vllm_omni.config.stage_config import (
    PipelineConfig,
    StageExecutionType,
    StagePipelineConfig,
)

_PROCESSOR = "vllm_omni.model_executor.stage_input_processors.minimax_h3"
_CHECKPOINT = "vllm_omni.model_executor.models.minimax_h3.checkpoint"
_DIFFUSION_PIPELINE = "vllm_omni.diffusion.models.minimax_h3.pipeline_minimax_h3"
_DECODER_PIPELINE = "vllm_omni.diffusion.models.minimax_h3.pipeline_minimax_h3_decoder"

MINIMAX_H3_PIPELINE = PipelineConfig(
    model_type="minimax_h3_disaggregated",
    default_deploy_config_name="minimax_h3_disaggregated.yaml",
    stage_cli_aliases={"text_encoder_tp_size": (0, "tensor_parallel_size")},
    model_arch="MiniMaxH3Encoder",
    stages=(
        StagePipelineConfig(
            stage_id=0,
            model_stage="encoder",
            execution_type=StageExecutionType.LLM_AR,
            input_sources=(),
            owns_tokenizer=True,
            requires_multimodal_data=True,
            model_arch="MiniMaxH3Encoder",
            engine_output_type="latent",
            prompt_transform_func=f"{_PROCESSOR}.prepare_encoder_prompt",
            custom_process_next_stage_input_func=f"{_PROCESSOR}.encoder2diffusion_full_payload",
            sampling_constraints={
                "max_tokens": 1,
                "temperature": 0.0,
                "detokenize": False,
            },
            model_path_resolver=f"{_CHECKPOINT}.resolve_minimax_h3_model_root",
        ),
        StagePipelineConfig(
            stage_id=1,
            model_stage="dit",
            execution_type=StageExecutionType.DIFFUSION,
            input_sources=(0,),
            final_output=True,
            final_output_type="video",
            requires_multimodal_data=False,
            model_arch="MiniMaxH3Pipeline",
            custom_process_input_func=f"{_PROCESSOR}.encoder2diffusion",
            stage_input_payload_keys=("encoder_output",),
            omni_kv_config={"need_recv_cache": False},
            model_path_resolver=f"{_DIFFUSION_PIPELINE}.resolve_minimax_h3_diffusion_model_path",
            inline_diffusion=True,
        ),
    ),
)


MINIMAX_H3_DECODE_PIPELINE = PipelineConfig(
    model_type="minimax_h3_disaggregated_decode",
    default_deploy_config_name="minimax_h3_disaggregated_decode.yaml",
    stage_cli_aliases={"text_encoder_tp_size": (0, "tensor_parallel_size")},
    model_arch="MiniMaxH3Encoder",
    stages=(
        replace(
            MINIMAX_H3_PIPELINE.stages[0],
            prompt_transform_func=f"{_PROCESSOR}.prepare_encoder_prompt_with_decoder",
        ),
        replace(MINIMAX_H3_PIPELINE.stages[1], final_output=False, final_output_type=None),
        StagePipelineConfig(
            stage_id=2,
            model_stage="vae_decode",
            execution_type=StageExecutionType.DIFFUSION,
            input_sources=(1,),
            final_output=True,
            final_output_type="video",
            model_arch="MiniMaxH3DecoderPipeline",
            custom_process_input_func=f"{_PROCESSOR}.diffusion2decoder",
            model_path_resolver=f"{_DECODER_PIPELINE}.resolve_minimax_h3_decoder_model_path",
            inline_diffusion=True,
        ),
    ),
)


# Explicit opt-in topology.  The legacy decode pipeline above remains the
# default so existing deployments keep the linear Encoder -> DiT -> Decoder
# behavior.
MINIMAX_H3_PARALLEL_DECODE_PIPELINE = PipelineConfig(
    model_type="minimax_h3_disaggregated_parallel_decode",
    default_deploy_config_name="minimax_h3_disaggregated_parallel_decode.yaml",
    stage_cli_aliases={"text_encoder_tp_size": (0, "tensor_parallel_size")},
    model_arch="MiniMaxH3Encoder",
    stages=(
        replace(
            MINIMAX_H3_PIPELINE.stages[0],
            prompt_transform_func=f"{_PROCESSOR}.prepare_encoder_prompt_with_decoder",
        ),
        replace(
            MINIMAX_H3_PIPELINE.stages[1],
            final_output=False,
            final_output_type=None,
            fanout_targets=(2, 3),
        ),
        StagePipelineConfig(
            stage_id=2,
            model_stage="video_vae",
            execution_type=StageExecutionType.DIFFUSION,
            input_sources=(1,),
            final_output=True,
            final_output_type="video",
            model_arch="MiniMaxH3VideoDecoderPipeline",
            custom_process_input_func=f"{_PROCESSOR}.diffusion2video_decoder",
            input_payload_selector="video",
            result_aggregator="minimax_h3_av",
            model_path_resolver=f"{_DECODER_PIPELINE}.resolve_minimax_h3_decoder_model_path",
            inline_diffusion=False,
        ),
        StagePipelineConfig(
            stage_id=3,
            model_stage="audio_vae",
            execution_type=StageExecutionType.DIFFUSION,
            input_sources=(1,),
            final_output=True,
            final_output_type="audio",
            model_arch="MiniMaxH3AudioDecoderPipeline",
            custom_process_input_func=f"{_PROCESSOR}.diffusion2audio_decoder",
            input_payload_selector="audio",
            result_aggregator="minimax_h3_av",
            model_path_resolver=f"{_DECODER_PIPELINE}.resolve_minimax_h3_decoder_model_path",
            inline_diffusion=False,
        ),
    ),
)
