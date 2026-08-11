# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Falcon Perception pipeline topology.

Two stages:

  0. **thinker** (``LLM_AR``) — image + text query -> token stream, plus a
     ``(x, y, h, w)`` box per detected instance and one hidden state per
     ``<seg>`` token.
  1. **segmentation** (``LLM_GENERATION``) — the ``<seg>`` features and the
     image patch features are upsampled by AnyUp into per-instance masks.

The mask decoder (i.e. segmentation mask generator) is not autoregressive and runs once per request,
so it is kept out of the AR loop as its own generation stage
(the same shape as the TTS talker -> code2wav split).

Detection-only (single-stage) serving for falcon perception is deliberately not offered:
the geometry feedback loop needs ``postprocess`` output (i.e. the last hidden state),
which the AR runner only collects when a downstream stage exists
(``gpu_ar_model_runner.py``: ``needs_payload = final_stage_id is None or final_stage_id > 0``).
With stage 0 final, the filter set is empty and the model doesn't get correct
feedback from the geometry feedback loop, hence producing wrong results.
Enabling it means generalizing that shared gate, which belongs in its own PR.
"""

from vllm_omni.config.stage_config import (
    PipelineConfig,
    StageExecutionType,
    StagePipelineConfig,
)

_PROC = "vllm_omni.model_executor.stage_input_processors.falcon_perception"

# Greedy decoding, and the reference's second stop token (``<|end_of_query|>``)
# alongside EOS.
_THINKER_SAMPLING = {
    "detokenize": True,
    "stop_token_ids": [11, 263],  # [eos, end_of_query]
}

FALCON_PERCEPTION_PIPELINE = PipelineConfig(
    model_type="falcon_perception",
    default_deploy_config_name="falcon_perception.yaml",
    model_arch="FalconPerceptionThinker",
    hf_architectures=("FalconPerceptionForSegmentation",),
    stages=(
        StagePipelineConfig(
            stage_id=0,
            model_stage="thinker",
            execution_type=StageExecutionType.LLM_AR,
            input_sources=(),
            final_output=True,  # this stage is allowed to surface results to the caller (not only an internal bridge).
            # final_output_type tells orchestration and APIs how to interpret and route each stage’s OmniRequestOutput
            final_output_type="text",  # those surfaced results are treated as text
            owns_tokenizer=True,
            requires_multimodal_data=True,
            # "latent": the stage also ships its hidden-state trajectory to
            # stage1 , exactly like the Qwen2.5-Omni thinker. The user-visible
            # output is still text (``final_output_type`` above).
            engine_output_type="latent",  # latent never reaches the user (client)
            # these latents are used by the segmentation stage (next stage in the pipeline) to generate the segm masks
            # Stage0's ``RequestOutput`` carries the last-layer hidden-states (``multimodal_output["latent"]``).
            # These hidden-states (and related mm fields) are also used for the geometry feedback path in the AR loop.
            model_arch="FalconPerceptionThinker",
            sampling_constraints=_THINKER_SAMPLING,
        ),
        StagePipelineConfig(
            stage_id=1,
            model_stage="segmentation",
            execution_type=StageExecutionType.LLM_GENERATION,
            input_sources=(0,),
            final_output=True,
            final_output_type="image",
            # AnyUp guides the upsampling with the original pixels, so the
            # request's image is re-supplied to this stage.
            requires_multimodal_data=True,
            engine_output_type="image",
            model_arch="FalconPerceptionSegmentation",
            sync_process_input_func=f"{_PROC}.thinker2segmentation_token_only",
            sampling_constraints={"detokenize": False},
        ),
    ),
)
