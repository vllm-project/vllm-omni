# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Falcon Perception configuration.

The published checkpoints (``tiiuae/Falcon-Perception``,
``tiiuae/Falcon-Perception-300M``) ship a flat ``config.json`` that uses the
reference implementation's own field names (``dim``, ``n_layers``, ``ffn_dim``,
...) rather than the usual Transformers ones. This class accepts those names
and mirrors them onto the standard aliases so vLLM's generic config plumbing
(head size, KV-cache sizing, max model len) works unchanged.
"""

from __future__ import annotations

from transformers.configuration_utils import PretrainedConfig


class FalconPerceptionConfig(PretrainedConfig):
    """Config for Falcon Perception / Falcon OCR style checkpoints."""

    model_type = "falcon_perception"

    # Image tokens attend bidirectionally within their own image (the reference
    # mask is ``causal OR same-image``). vLLM reads this attribute directly in
    # ``ModelArchConfigConvertor.is_mm_prefix_lm`` before consulting its own
    # hardcoded model list, so declaring it here is enough — no vLLM patching
    # (unlike for HunyuanImage3 where we need to patch the model_type in patch.py)
    is_mm_prefix_lm: bool = True

    def __init__(
        self,
        # --- backbone ---
        dim: int = 1024,
        n_layers: int = 28,
        n_heads: int = 16,
        head_dim: int = 128,
        n_kv_heads: int = 8,
        vocab_size: int = 65536,
        ffn_dim: int = 3072,
        norm_eps: float = 1e-5,
        max_seq_len: int = 8192,
        rope_theta: float = 10000.0,
        # --- image patchification ---
        channel_size: int = 3,
        spatial_patch_size: int = 16,
        temporal_patch_size: int = 1,
        # --- perception heads ---
        perception_heads: bool = True,
        do_segmentation: bool = True,
        coord_enc_dim: int = 512,
        coord_dec_dim: int = 8192,
        coord_out_dim: int = 2048,
        size_enc_dim: int = 512,
        size_dec_dim: int = 8192,
        size_out_dim: int = 2048,
        segm_out_dim: int = 256,
        num_segm_layers: int = 3,
        # --- special token ids ---
        eos_id: int = 11,
        img_id: int = 227,
        img_end_id: int = 230,
        image_cls_token_id: int = 244,
        image_reg_1_token_id: int = 245,
        image_reg_2_token_id: int = 246,
        image_reg_3_token_id: int = 247,
        image_reg_4_token_id: int = 248,
        coord_token_id: int = 240,
        size_token_id: int = 241,
        seg_token_id: int = 262,
        end_of_query_token_id: int = 263,
        # --- image preprocessing (mirrors the reference engine defaults) ---
        min_image_size: int = 256,
        max_image_size: int = 1024,
        min_pixels: int = 56 * 56,
        max_pixels: int = 28 * 28 * 1280,
        # --- vLLM-Omni stage-1 deployment knobs ---
        hr_cache_mb: int | None = None,
        compile_anyup: bool | None = None,
        **kwargs,
    ) -> None:
        self.dim = dim
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.head_dim = head_dim or (dim // n_heads)
        self.n_kv_heads = n_kv_heads or n_heads
        self.vocab_size = vocab_size
        self.ffn_dim = ffn_dim
        self.norm_eps = norm_eps
        self.max_seq_len = max_seq_len
        self.rope_theta = float(rope_theta)

        self.channel_size = channel_size
        self.spatial_patch_size = spatial_patch_size
        self.temporal_patch_size = temporal_patch_size

        self.perception_heads = perception_heads
        self.do_segmentation = do_segmentation
        self.coord_enc_dim = coord_enc_dim
        self.coord_dec_dim = coord_dec_dim
        self.coord_out_dim = coord_out_dim
        self.size_enc_dim = size_enc_dim
        self.size_dec_dim = size_dec_dim
        self.size_out_dim = size_out_dim
        self.segm_out_dim = segm_out_dim
        self.num_segm_layers = num_segm_layers

        self.eos_id = eos_id
        self.img_id = img_id
        self.img_end_id = img_end_id
        self.image_cls_token_id = image_cls_token_id
        self.image_reg_1_token_id = image_reg_1_token_id
        self.image_reg_2_token_id = image_reg_2_token_id
        self.image_reg_3_token_id = image_reg_3_token_id
        self.image_reg_4_token_id = image_reg_4_token_id
        self.coord_token_id = coord_token_id
        self.size_token_id = size_token_id
        self.seg_token_id = seg_token_id
        self.end_of_query_token_id = end_of_query_token_id

        self.min_image_size = min_image_size
        self.max_image_size = max_image_size
        self.min_pixels = min_pixels
        self.max_pixels = max_pixels
        self.hr_cache_mb = hr_cache_mb
        self.compile_anyup = compile_anyup

        # Standard Transformers aliases so vLLM's generic accessors work.
        self.hidden_size = self.dim
        self.num_hidden_layers = self.n_layers
        self.num_attention_heads = self.n_heads
        self.num_key_value_heads = self.n_kv_heads
        self.intermediate_size = self.ffn_dim
        self.max_position_embeddings = self.max_seq_len
        self.rms_norm_eps = self.norm_eps

        # The model applies its own split RoPE (1-D temporal on the first half
        # of head_dim, learned 2-D "golden" on the second half). Declaring an
        # ``mrope_section`` is what makes ``ModelConfig.uses_mrope`` true so the
        # runner allocates the [3, N] positions buffer that
        # ``get_mrope_input_positions`` fills; the H/W sections are 0 because
        # the spatial half is not driven by integer positions.
        kwargs.setdefault(
            "rope_parameters",
            {"rope_type": "default", "rope_theta": self.rope_theta, "mrope_section": [self.head_dim // 2, 0, 0]},
        )

        kwargs.setdefault("tie_word_embeddings", False)
        super().__init__(**kwargs)

    @property
    def image_structural_token_ids(self) -> list[int]:
        """The 5 tokens that precede an image's patch tokens, in order."""
        return [
            self.image_cls_token_id,
            self.image_reg_1_token_id,
            self.image_reg_2_token_id,
            self.image_reg_3_token_id,
            self.image_reg_4_token_id,
        ]
