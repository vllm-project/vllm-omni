# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""PersonaPlex pipeline: Talker (AR decode -> Mimi codebooks) -> Code2Wav (codebooks -> 24 kHz PCM).

PersonaPlex is a Moshi finetune (full-duplex speech-to-speech). For offline/batch
runs it is served as a 2-stage vllm-omni audio->audio pipeline, reusing the
Qwen3-TTS staged topology:

* Stage 0 (``personaplex``) is the AR talker: the Helium temporal transformer
  plus the depformer (both built by the lead). It emits the per-frame audio
  codebooks as latents.
* Stage 1 (``code2wav``) wraps the external Mimi codec and decodes the active
  codebooks (``cb 0..7``) into PCM.

The inter-stage input processors named below live in
``vllm_omni.model_executor.stage_input_processors.personaplex`` and are part of
the talker<->code2wav seam; they are built by the lead alongside the talker.
"""

from vllm_omni.config.stage_config import (
    PipelineConfig,
    StageExecutionType,
    StagePipelineConfig,
)

# Talker<->Code2Wav input processors (built by lead; coupled to talker output).
_PROC = "vllm_omni.model_executor.stage_input_processors.personaplex"

PERSONAPLEX_PIPELINE = PipelineConfig(
    model_type="personaplex",
    # Pipeline-level default; the code2wav stage overrides per-stage below.
    model_arch="PersonaPlexTalkerForConditionalGeneration",
    default_deploy_config_name="personaplex.yaml",
    duplex_runtime_extension=(
        "vllm_omni.experimental.fullduplex.personaplex.runtime_extension.PersonaPlexDuplexRuntimeExtension"
    ),
    duplex_serving_adapter=(
        "vllm_omni.experimental.fullduplex.personaplex.serving_adapter.PersonaPlexServingRuntimeAdapter"
    ),
    duplex_control_enabled=True,
    stages=(
        StagePipelineConfig(
            stage_id=0,
            model_stage="personaplex",
            execution_type=StageExecutionType.LLM_AR,
            input_sources=(),
            owns_tokenizer=True,
            engine_output_type="latent",
            # built by lead: talker emits a 17-row Mimi token stack per frame;
            # these processors keep the PCM-bearing rows (cb 0..7) and flatten
            # them codebook-major for the Code2Wav stage.
            async_chunk_process_next_stage_input_func=(f"{_PROC}.talker2code2wav_async_chunk"),
            custom_process_next_stage_input_func=f"{_PROC}.talker2code2wav_full_payload",
            sampling_constraints={
                "detokenize": False,
            },
        ),
        StagePipelineConfig(
            stage_id=1,
            model_stage="code2wav",
            execution_type=StageExecutionType.LLM_GENERATION,
            input_sources=(0,),
            final_output=True,
            final_output_type="audio",
            engine_output_type="audio",
            model_arch="PersonaPlexCode2Wav",
            # In sync (non-async-chunk) mode the only pre-stage input proc is a
            # length-only placeholder; the bulk codec payload ships via the
            # worker connector from stage 0's full-payload producer.
            # built by lead.
            sync_process_input_func=f"{_PROC}.talker2code2wav_token_only",
            sampling_constraints={"detokenize": True},
        ),
    ),
)
