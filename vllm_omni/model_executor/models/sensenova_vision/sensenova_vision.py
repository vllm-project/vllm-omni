# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""SenseNova-Vision-7B-MoT omni model.

SenseNova-Vision is a fork of Bagel with identical parameter-bearing modules.
This class reuses the MoT/ViT/VAE embedding logic from the BAGEL integration
and only overrides the SenseNovaVision checkpoint defaults plus additive features
(e.g. ``return_raw_latent``).
"""

from __future__ import annotations

from vllm.config import VllmConfig
from vllm.multimodal import MULTIMODAL_REGISTRY

from vllm_omni.model_executor.models.bagel.bagel import (
    OmniBagelDummyInputsBuilder,
    OmniBagelForConditionalGeneration,
    OmniBagelMultiModalProcessor,
    OmniBagelProcessingInfo,
)

# SenseNova-Vision-7B-MoT defaults.  The checkpoint ships metadata-only
# ``config.json`` (no ``architectures``), so these constants mirror what the
# official ``SenseNovaVisionModel._build_model`` applies at load time
# (inference/sensenova_vision.py).
SENSENOVA_VISION_DEFAULT_LAYER_MODULE = "Qwen2MoTDecoderLayer"
SENSENOVA_VISION_DEFAULT_QK_NORM = True
SENSENOVA_VISION_DEFAULT_TIE_WORD_EMBEDDINGS = False
SENSENOVA_VISION_DEFAULT_VISUAL_GEN = True
SENSENOVA_VISION_DEFAULT_VISUAL_UND = True
SENSENOVA_VISION_DEFAULT_MAX_LATENT_SIZE = 64
SENSENOVA_VISION_DEFAULT_VIT_MAX_NUM_PATCH_PER_SIDE = 70


@MULTIMODAL_REGISTRY.register_processor(
    OmniBagelMultiModalProcessor,
    info=OmniBagelProcessingInfo,
    dummy_inputs=OmniBagelDummyInputsBuilder,
)
class OmniSenseNovaVisionForConditionalGeneration(OmniBagelForConditionalGeneration):
    """SenseNova-Vision-7B-MoT omni model (subclass of the BAGEL integration).

    Inherits the entire MoT/ViT/VAE embedding and KV-transfer logic from
    :class:`OmniBagelForConditionalGeneration`.  Only the checkpoint-specific
    defaults differ:

    - ``layer_module="Qwen2MoTDecoderLayer"``
    - ``qk_norm=True``
    - ``tie_word_embeddings=False``
    - ``visual_gen=True`` / ``visual_und=True``
    - ``max_latent_size=64`` (BAGEL ships 32)
    - ``vit_max_num_patch_per_side=70``

    Additive SenseNovaVision features (``return_raw_latent`` for the diffusion
    pipeline) are exposed here without forking the base implementation.
    """

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        config = vllm_config.model_config.hf_config
        self._apply_sensenova_vision_config_defaults(config)
        super().__init__(vllm_config=vllm_config, prefix=prefix)

    @staticmethod
    def _apply_sensenova_vision_config_defaults(config) -> None:
        """Force SenseNovaVision checkpoint defaults on the HF config in place."""
        config.visual_gen = SENSENOVA_VISION_DEFAULT_VISUAL_GEN
        config.visual_und = SENSENOVA_VISION_DEFAULT_VISUAL_UND
        config.max_latent_size = SENSENOVA_VISION_DEFAULT_MAX_LATENT_SIZE
        config.vit_max_num_patch_per_side = SENSENOVA_VISION_DEFAULT_VIT_MAX_NUM_PATCH_PER_SIDE

        llm_config = config.llm_config
        llm_config.layer_module = SENSENOVA_VISION_DEFAULT_LAYER_MODULE
        llm_config.qk_norm = SENSENOVA_VISION_DEFAULT_QK_NORM
        llm_config.tie_word_embeddings = SENSENOVA_VISION_DEFAULT_TIE_WORD_EMBEDDINGS

    def get_raw_latent(self) -> None:
        """SenseNovaVision additive feature: raw-latent flag extension point.

        The AR stage produces KV caches, never latents, so this returns
        ``None``.  The DiT stage (``SenseNovaVisionPipeline``) honors
        ``return_raw_latent`` when decoding images.
        """
        return None
