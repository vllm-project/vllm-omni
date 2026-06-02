# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Lance (ByteDance) — vLLM-Omni model_executor package.

Single-stage topology: a self-contained diffusion stage (``LancePipeline``
defined in ``vllm_omni.diffusion.models.lance``) that owns the Qwen2-MoT LLM,
the Qwen2.5-VL ViT, the Wan2.2 VAE and the tokenizer — directly analogous to
``bagel_single_stage``. The ``PipelineConfig`` lives in :mod:`.pipeline`.
"""
