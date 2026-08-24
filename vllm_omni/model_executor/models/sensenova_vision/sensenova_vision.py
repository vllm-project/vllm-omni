# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""SenseNova-Vision-7B-MoT omni model.

SenseNova-Vision is a fork of Bagel with identical parameter-bearing modules.
This class reuses the MoT/ViT/VAE embedding logic from the BAGEL integration
and only overrides the SenseNovaVision checkpoint defaults plus additive features
(e.g. ``return_raw_latent``).
"""

from __future__ import annotations

from collections.abc import Mapping

import torch
from vllm.config import VllmConfig
from vllm.multimodal import MULTIMODAL_REGISTRY

from vllm_omni.model_executor.models.bagel.bagel import (
    OmniBagelDummyInputsBuilder,
    OmniBagelForConditionalGeneration,
    OmniBagelMultiModalProcessor,
    OmniBagelProcessingInfo,
)

# Per-task image target side for the generation output grid.  Recon3D decodes
# ``num_views`` square views at this VAE side; the AR stage caches the same
# value in ``kv_metadata["image_shape"]`` so the DiT stage (SenseNovaVisionPipeline)
# and the AR prefill agree on the latent grid.
RECON3D_VAE_SIDE = 512

# ``num_output_vae`` in upstream ``gen_image`` (inferencer.py) when no explicit
# per-request ``num_views`` is supplied.
RECON3D_DEFAULT_NUM_VIEWS = 4

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


class OmniSenseNovaVisionProcessingInfo(OmniBagelProcessingInfo):
    """Multi-modal limits for SenseNova-Vision.

    The shared BAGEL base caps ``image`` / ``img2img`` at 1 item per request;
    SenseNova-Vision raises both to 10, matching the upstream recon3d
    ``max_images=10``.  A finite cap (never ``None``) keeps mm memory
    profiling bounded.
    """

    def get_supported_mm_limits(self) -> Mapping[str, int | None]:
        return {"image": 10, "img2img": 10}


class OmniSenseNovaVisionMultiModalProcessor(OmniBagelMultiModalProcessor):
    """SenseNovaVision multimodal processor with recon3d view plumbing.

    Subclasses :class:`OmniBagelMultiModalProcessor` additively: it passes
    ``target_h``/``target_w`` through for the generation ``image`` modality so a
    recon3 request can request the VAE target shape per view, and records
    ``num_views`` for downstream sampling knobs.  Add-only — the base processor
    is left registered for SenseNovaVision because the AR stage feeds it the
    same BAGEL-compatible fields.
    """

    def _mm_kwargs_for_bagel_img2img_hf(self, mm_kwargs):
        # SenseNovaVision recon3d views are decoded additively from the AR KV
        # cache; ``target_h``/``target_w`` select the VAE output side and are
        # preserved through to the ``image_shape`` used by the DiT stage.
        return dict(mm_kwargs)


@MULTIMODAL_REGISTRY.register_processor(
    OmniBagelMultiModalProcessor,
    info=OmniSenseNovaVisionProcessingInfo,
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
        # Make AutoTokenizer.from_pretrained resolve VLLMSenseNovaVisionTokenizer
        # (id-preserving) in this process BEFORE the BAGEL core builds its own
        # tokenizer at OmniBagelForConditionalGeneration.__init__ (bagel.py) and
        # derives the img2img marker ids. Without registration the checkpoint's
        # declared tokenizer_class cannot be resolved from remote code (the
        # tokenization_sensenova_vision.py source file is never written into the
        # checkpoint dir), so AutoTokenizer would renumber the added tokens past
        # the 152064 embedding rows and trip the embed gather device-side assert.
        from vllm_omni.diffusion.models.sensenova_vision.tokenization_sensenova_vision import (
            register_vllm_sensenova_vision_tokenizer,
        )

        register_vllm_sensenova_vision_tokenizer()
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

    def _adjust_positions_for_img2img(
        self,
        positions: torch.Tensor,
        input_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Rewrite position IDs for img2img.

        Supports an optional ``pre_text_len`` prefix (thinking-mode) detected
        via the ``<|fim_middle|>`` token in *input_ids*:

            pre_text -> 0 .. M-1
            VAE      -> M       (all share)
            separator-> M
            ViT      -> M+1     (all share)
            post_text-> M+2, M+3, ...

        When M=0 (standard img2img) this reduces to VAE->0, ViT->1, text->2..

        A single request may carry SEVERAL img2img blocks (multi-image
        requests): every block collapses to two logical positions anchored at
        its own M, with interleaved text resuming at M+2, so a request with
        k blocks occupies ``#text_tokens + 2*k`` logical positions.
        """
        info_list = self._pending_img2img_info
        self._pending_img2img_info = []

        if not info_list:
            self._vae_token_mask = None
            self._has_vae_tokens = False
            self._has_non_vae_tokens = True
            return positions

        boundaries = [0]
        # Copy positions to the host once: indexing the CUDA tensor element by
        # element in the loop below would sync the device on every iteration.
        pos_list = positions.tolist()
        for i in range(1, len(pos_list)):
            if pos_list[i] < pos_list[i - 1]:
                boundaries.append(i)
        boundaries.append(len(pos_list))

        num_requests = len(boundaries) - 1
        new_positions = positions.clone()
        vae_mask = torch.zeros(len(positions), dtype=torch.bool, device=positions.device)

        img2img_idx = 0
        # Host copy of input_ids for placeholder matching (same rationale as
        # the positions copy above: per-element device indexing would sync on
        # every token).
        ids_list = input_ids.tolist() if input_ids is not None else None
        for req_idx in range(num_requests):
            start = boundaries[req_idx]
            end = boundaries[req_idx + 1]
            req_len = end - start

            # Match this request's <|fim_middle|> blocks against the pending
            # infos. A block is a contiguous placeholder stretch of exactly
            # ``num_vae + 1 + num_vit`` tokens for its image, so adjacent
            # blocks (no text between them) concatenate into longer runs that
            # this scan still splits correctly by advancing block_len tokens
            # per matched info. Infos left over stay queued for the following
            # requests in the batch.
            spans = []
            if ids_list is not None and img2img_idx < len(info_list):
                req_ids = ids_list[start:end]
                tok = self._img2img_token_id
                scan = 0
                info_i = img2img_idx
                while info_i < len(info_list):
                    num_vae, num_vit = info_list[info_i][0], info_list[info_i][1]
                    block_len = num_vae + 1 + num_vit
                    while scan < req_len and req_ids[scan] != tok:
                        scan += 1
                    if scan >= req_len or req_len - scan < block_len:
                        break
                    if any(req_ids[scan + j] != tok for j in range(block_len)):
                        break
                    spans.append((scan, *info_list[info_i]))
                    scan += block_len
                    info_i += 1

            if spans:
                # Logical positions are rebased per request: leading text keeps
                # 0..M1-1, every block collapses to TWO shared logical positions
                # (VAE+separator -> M, ViT -> M+1) regardless of token count,
                # and text after a block resumes at M+2. NOTE: a block's logical
                # anchor M is NOT its token offset once earlier blocks have
                # compressed their tokens, hence the threaded cursor.
                first_off = spans[0][0]
                if first_off > 0:
                    new_positions[start : start + first_off] = torch.arange(
                        0, first_off, device=positions.device, dtype=positions.dtype
                    )
                logical_m = first_off
                for k, (off, num_vae, num_vit, img_H, img_W) in enumerate(spans):
                    img_start = start + off
                    vit_start = img_start + num_vae + 1
                    new_positions[img_start:vit_start] = logical_m  # VAE section + separator
                    new_positions[vit_start : vit_start + num_vit] = logical_m + 1  # ViT section
                    vae_lo = img_start + 1
                    vae_hi = img_start + num_vae - 1
                    if vae_hi > vae_lo:
                        vae_mask[vae_lo:vae_hi] = True
                    block_end = off + num_vae + 1 + num_vit
                    next_off = spans[k + 1][0] if k + 1 < len(spans) else req_len
                    gap_len = next_off - block_end
                    if gap_len > 0:
                        new_positions[start + block_end : start + block_end + gap_len] = torch.arange(
                            logical_m + 2,
                            logical_m + 2 + gap_len,
                            device=positions.device,
                            dtype=positions.dtype,
                        )
                    logical_m += 2 + gap_len
                self._ropes_pending.append(
                    {
                        "ropes": [logical_m],
                        "image_shape": [spans[-1][3], spans[-1][4]],
                        "prefill_position_count": req_len,
                    }
                )
                img2img_idx += len(spans)
                continue

            if img2img_idx < len(info_list):
                cur_info = info_list[img2img_idx]
            elif self._last_img2img_info is not None:
                cur_info = self._last_img2img_info
            else:
                cur_info = None

            if cur_info is not None:
                num_vae, num_vit, img_H, img_W = cur_info
                num_img2img = num_vae + 1 + num_vit  # +1 separator

                if req_len >= num_img2img:
                    pre_text_len = 0
                    if input_ids is not None:
                        req_ids_slice = input_ids[start:end]
                        indices = (req_ids_slice == self._img2img_token_id).nonzero(as_tuple=True)[0]
                        if indices.numel() > 0:
                            pre_text_len = int(indices[0].item())

                    M = pre_text_len
                    img_start = start + M
                    post_text_start = img_start + num_img2img

                    if M > 0:
                        new_positions[start:img_start] = torch.arange(
                            0, M, device=positions.device, dtype=positions.dtype
                        )

                    new_positions[img_start : img_start + num_vae] = M
                    new_positions[img_start + num_vae] = M  # separator
                    vit_start = img_start + num_vae + 1
                    new_positions[vit_start : vit_start + num_vit] = M + 1

                    num_post_text = end - post_text_start
                    if num_post_text > 0:
                        new_positions[post_text_start:end] = torch.arange(
                            M + 2,
                            M + 2 + num_post_text,
                            device=positions.device,
                            dtype=positions.dtype,
                        )

                    vae_patches_start = img_start + 1
                    vae_patches_end = img_start + num_vae - 1
                    if vae_patches_end > vae_patches_start:
                        vae_mask[vae_patches_start:vae_patches_end] = True

                    rope = M + 2 + num_post_text
                    self._ropes_pending.append(
                        {
                            "ropes": [rope],
                            "image_shape": [img_H, img_W],
                            "prefill_position_count": req_len,
                        }
                    )
                    img2img_idx += 1
                    continue

            rope = int(new_positions[end - 1].item()) + 1
            self._ropes_pending.append({"ropes": [rope]})

        # Resolve mask occupancy once here (the only .any() syncs on this path)
        # and cache it; the per-layer routing reads these flags instead of
        # re-checking the mask on every decoder layer.
        has_vae = bool(vae_mask.any())
        self._vae_token_mask = vae_mask if has_vae else None
        self._has_vae_tokens = has_vae
        self._has_non_vae_tokens = bool((~vae_mask).any()) if has_vae else True
        return new_positions
