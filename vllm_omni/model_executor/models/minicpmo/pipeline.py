"""MiniCPM-o-4_5 S2S pipeline: LM (text) → Token2Speech (speech → WAV).

Stage 0: standard vllm MiniCPMO4_5 LM — generates text token IDs.
Stage 1: MiniCPMOToken2Speech — MiniCPMTTS AR + stepaudio2.Token2wav → WAV.

Text-only conditioning: Stage 1 uses tts.emb_text(token_ids) without LM
hidden states. Hidden states improve prosody but are not required for the
smoke test (≥0.5s, non-silent WAV).
"""

from vllm_omni.config.stage_config import (
    PipelineConfig,
    StageExecutionType,
    StagePipelineConfig,
    register_pipeline,
)

_PROC = "vllm_omni.model_executor.stage_input_processors.minicpmo"

MINICPMO_S2S_PIPELINE = PipelineConfig(
    model_type="minicpmo",
    model_arch="MiniCPMO",
    stages=(
        StagePipelineConfig(
            stage_id=0,
            model_stage="minicpmo",
            execution_type=StageExecutionType.LLM_AR,
            input_sources=(),
            owns_tokenizer=True,
            engine_output_type="latent",
            sampling_constraints={"detokenize": False},
        ),
        StagePipelineConfig(
            stage_id=1,
            model_stage="minicpmo_token2speech",
            execution_type=StageExecutionType.LLM_GENERATION,
            input_sources=(0,),
            final_output=True,
            final_output_type="audio",
            engine_output_type="audio",
            model_arch="MiniCPMOToken2Speech",
            sync_process_input_func=f"{_PROC}.text2speech",
            sampling_constraints={"detokenize": True},
        ),
    ),
)

register_pipeline(MINICPMO_S2S_PIPELINE)
