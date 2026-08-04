# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Qwen3-ASR pipeline topologies.

``qwen3_asr`` is a single comprehension stage: audio in, transcript text out.
``Qwen3ASRForConditionalGeneration`` already implements vLLM's
``SupportsTranscription``, so this topology is what ``AsyncOmniEngine``'s
capability probe looks for -- a transcription-capable comprehension stage that
is also the text terminal -- and it is what makes ``/v1/audio/transcriptions``
reachable.

Contrast ``aura_omni``, which runs the same model at stage 0 but as an
intermediate feeding a Qwen3-VL stage. A request there resolves to the *last*
text stage, so the ASR stage cannot be addressed on its own.

``qwen3_asr_align`` adds a forced-aligner stage that emits word timestamps.
Two declared topologies rather than one topology with a runtime switch, which
is the existing convention for optional stages here (``audex_tts`` /
``audex_thinker_only``, ``qwen2_5_omni`` / ``qwen2_5_omni_thinker_only``,
``step_audio_2`` / ``step_audio_2_asr``).
"""

from vllm_omni.config.stage_config import (
    PipelineConfig,
    StageExecutionType,
    StagePipelineConfig,
)

_PROC = "vllm_omni.model_executor.stage_input_processors.qwen3_asr_align"

_ASR_STAGE = StagePipelineConfig(
    stage_id=0,
    model_stage="asr",
    execution_type=StageExecutionType.LLM_AR,
    input_sources=(),
    final_output=True,
    final_output_type="text",
    owns_tokenizer=True,
    requires_multimodal_data=True,
    engine_output_type="text",
    model_arch="Qwen3ASRForConditionalGeneration",
    sampling_constraints={"detokenize": True},
)


QWEN3_ASR_PIPELINE = PipelineConfig(
    model_type="qwen3_asr",
    model_arch="Qwen3ASRForConditionalGeneration",
    hf_architectures=("Qwen3ASRForConditionalGeneration",),
    stages=(_ASR_STAGE,),
)


QWEN3_ASR_ALIGN_PIPELINE = PipelineConfig(
    model_type="qwen3_asr_align",
    model_arch="Qwen3ASRForConditionalGeneration",
    default_deploy_config_name="qwen3_asr_align.yaml",
    stages=(
        _ASR_STAGE,
        StagePipelineConfig(
            stage_id=1,
            model_stage="forced_aligner",
            # A pooling stage is still an LLM stage; what makes it pooling is
            # runner="pooling" in the deploy config, not the execution type.
            execution_type=StageExecutionType.LLM_AR,
            input_sources=(0,),
            # Terminal side-output: word timestamps go back to the client
            # rather than feeding a downstream stage. Declaring it a second
            # final output works because a request may already resolve to
            # several final stages (qwen2_5_omni returns text and audio the
            # same way).
            final_output=True,
            final_output_type="latent",
            # Not a comprehension stage: owns_tokenizer drives is_comprehension,
            # which is what /v1/audio/transcriptions probes for a
            # transcription-capable model. The aligner classifies, it does not
            # transcribe, and its prompt is built upstream in the input
            # processor, so it needs neither the flag nor its own tokenizer.
            # False also routes the stage through vLLM's native pooling output
            # path, which is where a PoolingRequestOutput is produced.
            owns_tokenizer=False,
            requires_multimodal_data=True,
            engine_output_type="latent",
            model_arch="Qwen3ASRForcedAlignerForTokenClassification",
            # Pairs stage 0's transcript with the audio the request arrived
            # with; the aligner needs both and the audio is no stage's output.
            custom_process_input_func=f"{_PROC}.asr2aligner",
        ),
    ),
)
