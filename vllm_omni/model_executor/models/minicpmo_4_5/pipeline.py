# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
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
MINICPMO45_REFERENCE_AUDIO_KEY = "_minicpmo45_reference_audio"


def _talker_stop_token_ids() -> list[int]:
    """Stage 1's default stop id: the one the multi-frame head can emit.

    Stage 1's deploy config carries the ``speculative_config`` that arms the
    multi-frame decode, and ``compute_logits`` then collapses the Talker's
    vLLM-level head to the two-wide continue/stop row. The only sampleable ids
    are therefore 0 and 1, and the codec EOS can never appear at this level (the
    model samples codec ids inside the loop and forwards them to stage 2
    itself).

    Pinning this to the codec EOS regardless is what made every multi-frame
    request run to ``max_tokens``: the model did emit the stop marker,
    ``check_stop`` compared it against 6561, and the request kept its slot until
    the context ran out. A deployment that drops the ``speculative_config`` from
    stage 1 falls back to one frame per step, where the head is the full codec
    vocabulary again and the codec EOS is the stop id; it overrides this per
    stage through the usual ``stop_token_ids`` sampling parameter.
    """
    from vllm_omni.platforms.npu.worker.talker_multiframe import STOP_TOKEN_ID

    return [STOP_TOKEN_ID]


MINICPMO_4_5_PIPELINE = PipelineConfig(
    model_type="minicpmo_4_5",
    default_deploy_config_name="minicpmo_4_5.yaml",
    model_arch="MiniCPMO45OmniForConditionalGeneration",
    duplex_plugin="vllm_omni.model_executor.models.minicpmo_4_5.duplex.plugin.MiniCPMO45DuplexPlugin",
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
            sampling_constraints={
                "detokenize": True,
                # The llm2tts bridge discards this boundary and every row
                # after it, so stop Stage 0 as soon as either valid
                # MiniCPM-o 4.5 turn terminator is sampled.
                "stop_token_ids": [151704, 151645],
            },
        ),
        StagePipelineConfig(
            stage_id=1,
            model_stage="tts",
            execution_type=StageExecutionType.LLM_AR,
            input_sources=(0,),
            hf_config_name="tts_config",
            engine_output_type="latent",
            custom_process_input_func=f"{_PROC}.llm2tts",
            custom_process_next_stage_input_func=f"{_PROC}.tts2code2wav_full_payload",
            async_chunk_process_next_stage_input_func=f"{_PROC}.tts2code2wav_async_chunk",
            sampling_constraints={
                "detokenize": False,
                # The stop id has to be one the vLLM-level head can actually
                # emit, and that head changes shape when the K-frame loop is
                # armed -- see _talker_stop_token_ids.
                "stop_token_ids": _talker_stop_token_ids(),
            },
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
            sync_process_input_func=f"{_PROC}.tts2code2wav_token_only",
            sampling_constraints={"detokenize": True},
            requires_full_payload_input=True,
        ),
    ),
)
