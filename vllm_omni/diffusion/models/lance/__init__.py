# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Lance (ByteDance) diffusion model components.

Lance is a unified autoregressive + diffusion multimodal model on a
Qwen2.5-VL-3B backbone.  Architecturally it is the BAGEL family (ByteDance
Mixture-of-Transformers): the released ``Lance_3B`` checkpoint uses the exact
same ``*_moe_gen`` MoT weight layout as BAGEL, plus ``vae2llm`` / ``llm2vae`` /
``time_embedder`` / ``latent_pos_embed`` connectors.  The deltas vs BAGEL are:

  * backbone is Qwen2.5-VL (mRoPE) instead of Qwen2,
  * understanding ViT is Qwen2.5-VL vision (not SigLIP), loaded from the base
    ``Qwen/Qwen2.5-VL-3B-Instruct`` rather than from the Lance checkpoint,
  * VAE is Wan2.2 (reused from the vLLM-Omni WAN path) instead of the BAGEL AE,
  * video path adds 3D latent position embeddings (follow-up; this module
    implements the image path first).

Because Lance is BAGEL-lineage, the transformer core is reused verbatim from
``vllm_omni.diffusion.models.bagel.bagel_transformer`` and only the pipeline
wiring (ViT / VAE / checkpoint layout) is specialized here.
"""

from vllm_omni.diffusion.models.lance.pipeline_lance import (
    LancePipeline,
    get_lance_post_process_func,
)

__all__ = [
    "LancePipeline",
    "get_lance_post_process_func",
]
