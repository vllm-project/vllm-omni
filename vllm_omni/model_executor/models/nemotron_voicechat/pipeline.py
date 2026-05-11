# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""``nemotron_voicechat`` Omni pipeline (NemotronDuplexH → EarTTS).

Two AR streaming stages for real-time speech-to-speech via
:class:`AsyncOmni` and per-step :class:`StreamingInput`:

* **Stage 0 — NemotronDuplexHForCausalLM.** Consumes the per-step
  acoustic encoder embedding plus a pre-computed prefill combined
  embedding, and samples a text token plus an ASR token at every step.

* **Stage 1 — EarTTSForCausalLM (chunk-driven).** Pre-armed at request
  start via :func:`eartts_prewarm_input` (placeholder prompt of length
  ``Tref = speaker_latent.shape[0]``, independent of Nemotron's prefill
  length), then driven step-by-step over the shared-memory connector by
  :func:`nemotron2eartts_async_chunk`: emits one acoustic frame per
  Nemotron text token.

The pipeline is registered against ``model_type = "nemotron_voicechat"``,
which neither component checkpoint reports natively, so the user
assembles a wrapper directory that :class:`AsyncOmni` loads with a
single ``model=`` argument::

    <wrapper>/
        config.json               # {"model_type": "nemotron_voicechat"}
        nemotron/                 # directory or symlink → Nemotron ckpt
        eartts/                   # directory or symlink → EarTTS ckpt

The deploy YAML at ``vllm_omni/deploy/nemotron_voicechat.yaml`` points
each stage at its component via per-stage ``engine_extras.model_subdir``
/ ``tokenizer_subdir`` (see :data:`NEMOTRON_SUBDIR`, :data:`EARTTS_SUBDIR`).
"""

from __future__ import annotations

from vllm_omni.config.stage_config import (
    PipelineConfig,
    StageExecutionType,
    StagePipelineConfig,
)

_PROC = "vllm_omni.model_executor.stage_input_processors.nemotron_voicechat"


# Subdirectory names inside the user-managed wrapper directory. The
# deploy YAML references these names via per-stage
# ``engine_extras.model_subdir`` (and matching ``tokenizer_subdir``).
NEMOTRON_SUBDIR = "nemotron"
EARTTS_SUBDIR = "eartts"


NEMOTRON_VOICECHAT_PIPELINE = PipelineConfig(
    model_type="nemotron_voicechat",
    model_arch="NemotronDuplexHForCausalLM",
    stages=(
        StagePipelineConfig(
            stage_id=0,
            model_stage="nemotron",
            execution_type=StageExecutionType.LLM_AR,
            input_sources=(),
            final_output=True,
            final_output_type="text",
            owns_tokenizer=True,
            model_arch="NemotronDuplexHForCausalLM",
            engine_output_type="latent",
            custom_process_next_stage_input_func=(
                f"{_PROC}.nemotron2eartts_async_chunk"
            ),
            sampling_constraints={"detokenize": False},
        ),
        StagePipelineConfig(
            stage_id=1,
            model_stage="eartts",
            execution_type=StageExecutionType.LLM_AR,
            input_sources=(0,),
            final_output=True,
            final_output_type="audio_codes",
            model_arch="EarTTSForCausalLM",
            engine_output_type="audio_codes",
            sampling_constraints={"detokenize": False},
            prewarm_input_func=f"{_PROC}.eartts_prewarm_input",
        ),
    ),
)


__all__ = [
    "EARTTS_SUBDIR",
    "NEMOTRON_SUBDIR",
    "NEMOTRON_VOICECHAT_PIPELINE",
]
