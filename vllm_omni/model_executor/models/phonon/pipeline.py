# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from dataclasses import replace

from vllm_omni.config.stage_config import PipelineConfig, StageExecutionType, StagePipelineConfig

PHONON_PIPELINE = PipelineConfig(
    model_type="gradium_phonon_tts",
    default_deploy_config_name="phonon.yaml",
    model_arch="PhononTalkerForConditionalGeneration",
    stages=(
        StagePipelineConfig(
            stage_id=0,
            model_stage="phonon",
            execution_type=StageExecutionType.LLM_AR,
            input_sources=(),
            final_output=True,
            final_output_type="audio",
            owns_tokenizer=True,
            engine_output_type="audio",
            recompute_preemption="fail",
            sampling_constraints={"detokenize": False},
        ),
    ),
)


def resolve_phonon_pipeline(hf_config=None):
    from vllm_omni.transformers_utils.configs.phonon import PhononConfig

    eos = getattr(hf_config, "eos_token_id", None)
    if eos is None:
        eos = PhononConfig().eos_token_id
    if isinstance(eos, list):
        if len(eos) != 1:
            raise ValueError("Phonon requires one EOS protocol token")
        eos = eos[0]
    stage = replace(
        PHONON_PIPELINE.stages[0],
        sampling_constraints={"detokenize": False, "stop_token_ids": [eos]},
    )
    return replace(PHONON_PIPELINE, stages=(stage,))
