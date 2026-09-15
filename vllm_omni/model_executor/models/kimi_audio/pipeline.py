# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Kimi-Audio: prepared input -> dual-stream AR -> acoustic decoding."""

from vllm_omni.config.stage_config import PipelineConfig, StageExecutionType, StagePipelineConfig

_PROC = "vllm_omni.model_executor.stage_input_processors.kimi_audio"


def register_kimi_audio_renderer():
    """Register the Omni renderer while retaining vLLM's existing tokenizer."""
    from vllm.renderers.registry import RENDERER_REGISTRY
    from vllm.tokenizers.registry import TokenizerRegistry

    TokenizerRegistry.register("kimi_audio_omni", "vllm.tokenizers.kimi_audio", "KimiAudioTokenizer")
    RENDERER_REGISTRY.register(
        "kimi_audio_omni", "vllm_omni.model_executor.models.kimi_audio.renderer", "KimiAudioRenderer"
    )


KIMI_AUDIO_PIPELINE = PipelineConfig(
    model_type="kimi_audio",
    model_arch="KimiAudioForConditionalGeneration",
    hf_architectures=("MoonshotKimiaForCausalLM",),
    default_deploy_config_name="kimi_audio.yaml",
    stages=(
        StagePipelineConfig(
            stage_id=0,
            model_stage="kimi_audio_ar",
            execution_type=StageExecutionType.LLM_AR,
            final_output=True,
            final_output_type="text",
            owns_tokenizer=True,
            engine_output_type="latent",
            prompt_transform_func=f"{_PROC}.prepare_kimi_audio_request",
            async_chunk_process_next_stage_input_func=f"{_PROC}.kimi_audio_to_decoder_async_chunk",
            sampling_constraints={
                "detokenize": True,
                # ids.output already excludes EOS and control tokens. Native
                # detokenization must retain the final visible text token.
                "include_stop_str_in_output": True,
                # Checked against the prepared tokenizer vocabulary on admission.
                "stop_token_ids": [151645],
            },
        ),
        StagePipelineConfig(
            stage_id=1,
            model_stage="kimi_audio_decoder",
            execution_type=StageExecutionType.LLM_GENERATION,
            input_sources=(0,),
            final_output=True,
            final_output_type="audio",
            engine_output_type="audio",
            retains_state_across_chunks=True,
            sync_process_input_func=f"{_PROC}.kimi_audio_to_decoder",
            sampling_constraints={"detokenize": False, "max_tokens": 1},
        ),
    ),
)
