# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DeepSeek Janus omni pipeline topology.

Single-stage (image generation only):
  Stage 0: Diffusion worker runs :class:`JanusPipeline`
  (AR loop + VQ decode in one pipeline with torch.compile,
   CUDA Graph via CUDAGraphWrapper, and flash_attn).

Two-stage (AR image token generation → VQ decode):
  Stage 0: vLLM AR engine runs :class:`JanusForImageGeneration`
  (full PagedAttention, CUDAGraphWrapper, chunked prefill,
   prefix caching, FP8 KV cache, FlashInfer, tensor parallelism,
   continuous batching — all via vLLM's GPU model runner).
  Stage 1: Diffusion worker runs :class:`JanusVQDecodePipeline`
  (VQ decoder only, <1ms per request).

For the two-stage deployment, CFG is handled by the AR model's
``compute_logits()`` which merges conditional and unconditional
logits before sampling.  The VQ stage simply decodes the resulting
image token grid.
"""

from vllm_omni.config.stage_config import (
    PipelineConfig,
    StageExecutionType,
    StagePipelineConfig,
)

_PROC = "vllm_omni.model_executor.models.deepseek_janus.stage_input_processors"

DEEPSEEK_JANUS_SINGLE_STAGE_PIPELINE = PipelineConfig(
    model_type="deepseek_janus_single_stage",
    model_arch="MultiModalityCausalLM",
    hf_architectures=("MultiModalityCausalLM",),
    stages=(
        StagePipelineConfig(
            stage_id=0,
            model_stage="dit",
            execution_type=StageExecutionType.DIFFUSION,
            input_sources=(),
            final_output=True,
            final_output_type="image",
        ),
    ),
)

DEEPSEEK_JANUS_TWO_STAGE_PIPELINE = PipelineConfig(
    model_type="deepseek_janus_two_stage",
    model_arch="JanusForImageGeneration",
    hf_architectures=("MultiModalityCausalLM",),
    stages=(
        StagePipelineConfig(
            stage_id=0,
            model_stage="AR",
            execution_type=StageExecutionType.LLM_AR,
            input_sources=(),
            final_output=False,
            owns_tokenizer=True,
            requires_multimodal_data=False,
            model_arch="JanusForImageGeneration",
            engine_output_type="image_tokens",
            sampling_constraints={"detokenize": False},
        ),
        StagePipelineConfig(
            stage_id=1,
            model_stage="vae",
            execution_type=StageExecutionType.DIFFUSION,
            input_sources=(0,),
            final_output=True,
            final_output_type="image",
            model_arch="JanusVQDecodePipeline",
            custom_process_input_func=f"{_PROC}.ar_tokens_to_vq",
            omni_kv_config={"need_recv_cache": False},
        ),
    ),
)
