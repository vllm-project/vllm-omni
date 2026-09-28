# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Spark-TTS: Qwen2 token generator followed by full-utterance BiCodec."""

from vllm_omni.config.stage_config import PipelineConfig, StageExecutionType, StagePipelineConfig

_PROC = "vllm_omni.model_executor.stage_input_processors.spark_tts"

SPARK_TTS_PIPELINE = PipelineConfig(
    model_type="spark_tts",
    default_deploy_config_name="spark_tts.yaml",
    stages=(
        StagePipelineConfig(
            stage_id=0,
            model_stage="spark_tts_llm",
            execution_type=StageExecutionType.LLM_AR,
            model_arch="SparkQwen2ForCausalLM",
            model_subdir="LLM",
            tokenizer_subdir="LLM",
            input_sources=(),
            owns_tokenizer=True,
            engine_output_type="latent",
            custom_process_next_stage_input_func=f"{_PROC}.llm_to_bicodec_full_payload",
            sampling_constraints={"detokenize": False, "stop_token_ids": [151645]},
        ),
        StagePipelineConfig(
            stage_id=1,
            model_stage="spark_tts_bicodec",
            execution_type=StageExecutionType.LLM_GENERATION,
            model_arch="SparkBiCodecDecoder",
            model_subdir="BiCodec",
            tokenizer_subdir="LLM",
            input_sources=(0,),
            final_output=True,
            final_output_type="audio",
            engine_output_type="audio",
            sync_process_input_func=f"{_PROC}.bicodec_token_only",
            requires_full_payload_input=True,
            sampling_constraints={"detokenize": True},
        ),
    ),
)
