# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""VoxCPM2 pipeline topology (frozen).

Single-stage AR TTS: text → speech waveform in one pass.
Uses the native MiniCPM4 base_lm with a per-request StaticKVCache that the
talker restores into the paged attention layer at step boundaries.
"""

from vllm_omni.config.stage_config import (
    PipelineConfig,
    StageExecutionType,
    StagePipelineConfig,
)

VOXCPM2_PIPELINE = PipelineConfig(
    model_type="voxcpm2",
    default_deploy_config_name="voxcpm2.yaml",
    model_arch="VoxCPM2TalkerForConditionalGeneration",
    stages=(
        StagePipelineConfig(
            stage_id=0,
            model_stage="latent_generator",
            execution_type=StageExecutionType.LLM_AR,
            input_sources=(),
            final_output=True,
            final_output_type="audio",
            owns_tokenizer=True,
            engine_output_type="audio",
            scheduler_cls="vllm_omni.model_executor.models.voxcpm2.scheduler.VoxCPM2OmniARAsyncScheduler",
            sampling_constraints={
                "detokenize": False,
                "stop_token_ids": [1],
            },
            # ``curr_embed_for_next`` / ``prev_feat_embed`` / CFM-VAE progress
            # are per-request side state; replaying the prompt embeds audio ids
            # through the text embedding table and faults (#6179).
            # Decode feeds PagedAttention embeddings this stage cannot rebuild
            # from ``output_token_ids``; a recompute preemption would splice or
            # corrupt the audio (vllm-project/vllm-omni#6179).
            recompute_preemption="fail",
        ),
    ),
)
