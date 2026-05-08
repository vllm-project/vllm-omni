# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""``nemotron_voicechat`` Omni pipeline (NemotronDuplexH → EarTTS).

Two AR streaming stages, mirroring the qwen3-omni thinker → talker
topology, packaged for real-time speech-to-speech via
:class:`AsyncOmni` and per-step :class:`StreamingInput`.

* **Stage 0 — NemotronDuplexHForCausalLM (AR).** Consumes the per-step
  acoustic encoder embedding (one row per ``StreamingInput`` chunk)
  and a pre-computed prefill combined embedding (one-shot per
  request). Samples a text token (vLLM's standard sampler) and an ASR
  token (the model's own ``asr_head``) at every step; after each step
  the chunk transfer adapter calls
  :func:`...stage_input_processors.nemotron_voicechat.nemotron2eartts_async_chunk`,
  which forwards the cumulative list of sampled text tokens to stage
  1. ``speaker_latent`` is forwarded once on chunk 0 so EarTTS can
  build its prefill tensors.

* **Stage 1 — EarTTSForCausalLM (AR), chunk-driven streaming mode.**
  Receives one chunk per Nemotron step. Step 0 is EarTTS' own prefill
  (uses the forwarded ``speaker_latent``); step ``k ≥ 1`` consumes
  the ``k``-th Nemotron text token from ``input_text_tokens``
  (indexed via the model's ``ear_decode_offset``) and emits one
  acoustic frame.

Independent per-stage prefill lengths
-------------------------------------
EarTTS registers a ``prewarm_input_func`` (``eartts_prewarm_input``)
that the orchestrator calls inside
:meth:`vllm_omni.engine.orchestrator.Orchestrator._prewarm_async_chunk_stages`.
The hook returns a placeholder prompt of length
``Tref = speaker_latent.shape[0]`` (instead of Nemotron's
``T_PREFILL``), with
``additional_information = {"speaker_latent": ...}`` to seed the
runner's per-request intermediate buffer. ``Tref`` and ``T_PREFILL``
are therefore allowed to differ; the user does not need to pre-pad /
truncate the reference latent.

If the user does not supply ``speaker_latent`` on the prompt the
hook returns ``None`` and the orchestrator falls back to the default
(share Nemotron's prompt length). EarTTS' preprocess will then
raise — same behavior as the legacy path.

Model directory layout
----------------------

The pipeline is registered against ``model_type = "nemotron_voicechat"``;
since neither component checkpoint reports that ``model_type`` natively,
the user manually constructs a tiny *wrapper* directory that
:class:`AsyncOmni` can load with a single ``model=`` argument::

    <wrapper>/
        config.json               # {"model_type": "nemotron_voicechat"}
        nemotron/                 # directory or symlink → Nemotron ckpt
        eartts/                   # directory or symlink → EarTTS ckpt

The deploy YAML at ``vllm_omni/deploy/nemotron_voicechat.yaml`` then
points each stage at its component via per-stage
``engine_extras.model_subdir`` / ``tokenizer_subdir`` (see
:data:`NEMOTRON_SUBDIR` and :data:`EARTTS_SUBDIR`). The two checkpoints
stay separate inside the wrapper while still being addressable by a
single ``AsyncOmni(model=<wrapper>)`` argument.
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
