# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""MiniCPM-o 4.5 pipeline topology (frozen).

Stage 0: Thinker — multimodal understanding + text generation.
Stage 1: Talker  — MiniCPMTTS + Token2Wav, emits the final audio waveform.

The thinker -> talker bridge passes the hidden states + token ids extracted
from the thinker output through ``minicpmo_4_5_omni.llm2tts``; the talker
runs MiniCPMTTS and the on-device Token2wav vocoder in the same process and
returns the waveform directly as the pipeline's final audio output.
"""

from vllm_omni.config.stage_config import (
    PipelineConfig,
    StageExecutionType,
    StagePipelineConfig,
)

_PROC = "vllm_omni.model_executor.stage_input_processors.minicpmo_4_5_omni"


MINICPMO_4_5_PIPELINE = PipelineConfig(
    model_type="minicpmo_4_5",
    model_arch="MiniCPMO45OmniForConditionalGeneration",
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
            # Stage 1 shares the top-level wrapper class
            # (``MiniCPMO45OmniForConditionalGeneration``) inherited from
            # ``model_arch`` above. The wrapper dispatches on ``model_stage``
            # and, for ``"tts"``, instantiates the standalone TTS submodule
            # (``MiniCPMO45OmniTTSForConditionalGeneration``) internally.
            # Routing through the wrapper is required so that the runner-side
            # ``runtime_additional_information`` payload reaches the talker
            # (the standalone TTS class only reads ``additional_information``,
            # so wiring stage 1 directly to it would always trigger the dummy
            # path) and so the resulting waveform is packaged as
            # ``OmniOutput.multimodal_outputs["model_outputs"]`` instead of
            # being returned as a bare tuple that the AR runner would mistake
            # for hidden states. ``hf_config_name="tts_config"`` keeps KV
            # cache / mrope sizing scoped to the talker sub-config.
            execution_type=StageExecutionType.LLM_AR,
            input_sources=(0,),
            final_output=True,
            final_output_type="audio",
            hf_config_name="tts_config",
            engine_output_type="audio",
            custom_process_input_func=f"{_PROC}.llm2tts",
            sampling_constraints={"detokenize": False},
        ),
    ),
)
