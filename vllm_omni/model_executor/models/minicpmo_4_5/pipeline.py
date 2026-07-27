# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""MiniCPM-o 4.5 pipeline topology (frozen).

Stage 0: Thinker — multimodal understanding + text generation.
Stage 1: Talker  — MiniCPMTTS, emits codec tokens.
Stage 2: Code2Wav — codec tokens to the final audio waveform.

The thinker -> talker bridge uses ``llm2tts``. The talker -> Code2Wav bridge
streams request-routed codec chunks.
"""

from vllm_omni.config.stage_config import (
    PipelineConfig,
    StageExecutionType,
    StagePipelineConfig,
)

_PROC = "vllm_omni.model_executor.stage_input_processors.minicpmo_4_5_omni"


MINICPMO_4_5_PIPELINE = PipelineConfig(
    model_type="minicpmo_4_5",
    default_deploy_config_name="minicpmo_4_5.yaml",
    model_arch="MiniCPMO45OmniForConditionalGeneration",
    duplex_runtime_extension=("vllm_omni.experimental.fullduplex.minicpmo45.runtime.MiniCPMO45DuplexRuntimeExtension"),
    duplex_serving_adapter=(
        "vllm_omni.experimental.fullduplex.minicpmo45.serving_adapter.MiniCPMO45ServingRuntimeAdapter"
    ),
    duplex_control_enabled=True,
    # MiniCPM-o 4.5's HF config.json reports `model_type="minicpmo"` and
    # `architectures=["MiniCPMO"]` — both shared verbatim with older MiniCPM-o
    # 1.0 / 2.6 checkpoints. The only field distinguishing the generations is
    # the top-level ``version`` string, so we register both the shared
    # ``MiniCPMO`` arch (for auto-detection) and the 4.5-specific arch (for
    # repos that opt into the explicit name later), then pin the routing to
    # 4.5 via ``hf_config_predicate``. Without the predicate, loading a 2.6
    # checkpoint would also intersect ``["MiniCPMO"]`` here and get routed
    # into the 4.5 pipeline, which would then fail at load time.
    hf_architectures=("MiniCPMO", "MiniCPMO45OmniForConditionalGeneration"),
    hf_config_predicate=lambda c: str(getattr(c, "version", "")) == "4.5",
    stages=(
        StagePipelineConfig(
            stage_id=0,
            model_stage="llm",
            execution_type=StageExecutionType.LLM_AR,
            input_sources=(),
            final_output=True,
            final_output_type="text",
            owns_tokenizer=True,
            requires_multimodal_data=True,
            engine_output_type="latent",
            sampling_constraints={"detokenize": True},
        ),
        StagePipelineConfig(
            stage_id=1,
            model_stage="tts",
            execution_type=StageExecutionType.LLM_AR,
            input_sources=(0,),
            hf_config_name="tts_config",
            engine_output_type="latent",
            custom_process_input_func=f"{_PROC}.llm2tts",
            async_chunk_process_next_stage_input_func=f"{_PROC}.tts2code2wav_async_chunk",
            sampling_constraints={"detokenize": False},
        ),
        StagePipelineConfig(
            stage_id=2,
            model_stage="code2wav",
            execution_type=StageExecutionType.LLM_GENERATION,
            input_sources=(1,),
            final_output=True,
            final_output_type="audio",
            engine_output_type="audio",
            model_arch="MiniCPMO45Code2Wav",
            sampling_constraints={"detokenize": True},
        ),
    ),
)
