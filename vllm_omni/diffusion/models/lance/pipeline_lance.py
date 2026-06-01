# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""LancePipeline — Lance (ByteDance) packaged for the vLLM-Omni diffusion engine.

Lance is BAGEL-lineage (Qwen2-MoT unified AR+diffusion), so the transformer
core and the entire generation/forward machinery are inherited unchanged from
:class:`BagelPipeline`.  Only model *construction* differs, and only in three
well-localized places:

  1. Checkpoint layout — the HF repo ``bytedance-research/Lance`` bundles
     ``Lance_3B/`` (image) and ``Lance_3B_Video/`` (video) LLM checkpoints,
     ``Qwen2.5-VL-ViT/`` (understanding ViT) and ``Wan2.2_VAE.pth`` (VAE) in a
     single repo.  There is no BAGEL-style top-level ``config.json`` carrying
     ``vae_config`` / ``vit_config`` / ``latent_patch_size`` — those are Lance
     constants taken from upstream ``config/config_factory.py`` and
     ``inference_lance.sh`` and hardcoded in :data:`LANCE_DEFAULTS`.
  2. Understanding ViT — Qwen2.5-VL vision tower (bundled
     ``Qwen2.5-VL-ViT/vit.safetensors``) instead of SigLIP.
  3. VAE — Wan2.2 (``Wan2.2_VAE.pth``) instead of the BAGEL autoencoder.

Scope: this lands the **image path** (``t2i`` / ``image_edit`` / ``x2t_image``)
which is the direct BAGEL analogue.  The ``Lance_3B_Video`` path needs the 3-D
latent position embedding (:class:`LancePositionEmbedding3D`) and temporal VAE
handling and is an explicit follow-up — see the PR description.

Bring-up status (verified on a B300 against ``bytedance-research/Lance``):
  * t2i: end-to-end working (1024x1024 image in ~6 s, 0 missing keys).
  * x2t (image understanding): plumbing wired (Qwen2.5-VL ViT + Qwen2-VL
    image processor + no-op connector/vit_pos_embed; VAE prefill skipped
    for Lance via :class:`LanceBagel`).  The pipeline runs to completion
    without crashes but currently emits an immediate EOS because the
    Qwen2.5-VL backbone needs mRoPE position ids on the image+text
    sequence and we presently force scalar positions
    (``rope_scaling = None``); enabling mRoPE end-to-end is the next
    follow-up.
  * image_edit (img2img): blocked on the same VAE-prefill issue
    (Wan2.2 latents do not map onto BAGEL's ``latent_pos_embed`` grid);
    needs a Lance-specific ``prepare_vae_images``.
  * video (``Lance_3B_Video``): needs the 3-D latent position embedding
    wired into ``Bagel`` plus a multi-frame VAE decode path in the
    pipeline.  :class:`LanceWanVAE.decode_video` and
    :class:`LancePositionEmbedding3D` are already in place.
  * Two-stage (AR thinker + DiT): needs ``LanceConfig`` / ``LanceProcessor``
    registered in the ``vllm`` package; tracked in a separate PR.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass

import torch
from PIL import Image
from vllm.logger import init_logger

from vllm_omni.diffusion.data import OmniDiffusionConfig
from vllm_omni.diffusion.models.bagel.pipeline_bagel import (
    BagelPipeline,
    add_special_tokens,
)
from vllm_omni.model_executor.model_loader.weight_utils import (
    download_weights_from_hf_specific,
)

from .lance_transformer import (
    LanceBagel,
    LanceIdentityConnector,
    LancePositionEmbedding3D,
    LanceQwen2_5_VLNaViTWrapper,
    LanceZeroVitPosEmbed,
    Qwen2MoTConfig,
    Qwen2MoTForCausalLM,
)
from .prompts import SYSTEM_PROMPTS
from .wan_vae import LanceWanVAE

logger = init_logger(__name__)


def _extract_user_instruction(rendered: str) -> str:
    """Pull just the user instruction out of a chat-template-rendered prompt.

    ``render_lance_prompt`` wraps the instruction in:

        <|im_start|>system\\n{sys}<|im_end|>\\n
        <|im_start|>user\\n{vision_block}{user_text}<|im_end|>\\n
        <|im_start|>assistant\\n

    The ``image_edit`` pipeline rebuilds the chat template segment-by-segment
    (so the rope positions and segment boundaries match upstream Lance), and
    therefore needs only ``user_text`` — not the wrapped string.  This helper
    extracts it; if the input doesn't look templated it's returned as-is.
    """
    if "<|im_start|>user" not in rendered:
        return rendered.strip()
    # User block sits between the user header and the next ``<|im_end|>``.
    after_user = rendered.split("<|im_start|>user", 1)[1]
    after_user = after_user.lstrip("\n")
    user_block = after_user.split("<|im_end|>", 1)[0]
    # Strip any vision_start..vision_end blocks.
    out = []
    i = 0
    while i < len(user_block):
        if user_block.startswith("<|vision_start|>", i):
            j = user_block.find("<|vision_end|>", i)
            if j == -1:
                break
            i = j + len("<|vision_end|>")
            continue
        out.append(user_block[i])
        i += 1
    return "".join(out).strip()


@dataclass(frozen=True)
class LanceDefaults:
    """Lance constants that upstream keeps in ``config/config_factory.py`` /
    ``inference_lance.sh`` rather than in any shipped JSON.

    Verified against ``bytedance-research/Lance``'s released ``Lance_3B/model.safetensors``:
    ``vae2llm.weight = (2048, 48)`` ⇒ ``patch_latent_dim = latent_patch_size**2 * z_channels = 1 * 48``,
    i.e. Lance does *not* unfold the Wan latent into a 2×2 patch the way BAGEL does (Wan2.2 already
    patchifies internally), and ``latent_pos_embed.pos_embed = (4096, 2048)`` ⇒ ``max_latent_size = 64``.
    """

    # latent / patch geometry (pt, ph, pw); image path only uses the spatial 1.
    latent_patch_size_spatial: int = 1
    latent_patch_size_temporal: int = 1
    max_latent_size: int = 64
    # Lance_3B_Video ships ``latent_pos_embed.pos_embed`` of shape
    # ``(31 * 64 * 64, 2048) = (126976, 2048)`` ⇒ max_num_frames in latent
    # space is 31, equivalent to ``(num_frames - 1) // downsample_temporal + 1``
    # for up to 121 RGB frames.
    max_num_video_latent_frames: int = 31
    vit_max_num_patch_per_side: int = 70
    connector_act: str = "gelu_pytorch_tanh"
    timestep_shift: float = 3.5
    num_timesteps: int = 30
    cfg_text_scale: float = 4.0
    # Wan2.2 VAE
    vae_z_channels: int = 48
    vae_downsample_spatial: int = 16
    vae_downsample_temporal: int = 4


LANCE_DEFAULTS = LanceDefaults()

# Subdirectories inside the HF repo (see config.json::checkpoint_directories).
_IMAGE_CKPT_DIR = "Lance_3B"
_VIDEO_CKPT_DIR = "Lance_3B_Video"
_VIT_DIR = "Qwen2.5-VL-ViT"
_VAE_FILE = "Wan2.2_VAE.pth"


def get_lance_post_process_func(od_config: OmniDiffusionConfig):
    """Lance returns PIL.Image.Image directly, same as BAGEL."""

    def post_process_func(x):
        return x

    return post_process_func


def get_lance_pre_process_func(od_config: OmniDiffusionConfig):
    def pre_process_func(x):
        return x

    return pre_process_func


@dataclass
class _LanceVaeCfg:
    z_channels: int = LANCE_DEFAULTS.vae_z_channels
    # BAGEL core computes ``latent_downsample = vae.downsample * latent_patch_size``.
    # For Lance image latents that is ``16 * 2 = 32``.
    downsample: int = LANCE_DEFAULTS.vae_downsample_spatial


@dataclass
class _LanceVitCfg:
    patch_size: int = 14
    hidden_size: int = 1280  # Qwen2.5-VL vision hidden (out_hidden_size=2048)


class LancePipeline(BagelPipeline):
    """Lance pipeline.  Inherits BAGEL's forward/generation; overrides only
    construction (checkpoint layout, Qwen2.5-VL ViT, Wan2.2 VAE)."""

    def __init__(self, *, od_config: OmniDiffusionConfig, prefix: str = ""):
        # Intentionally do NOT call BagelPipeline.__init__ — its assumptions
        # about config.json / vit_config.json / SigLIP / the BAGEL AE do not
        # hold for Lance.  We replicate the BAGEL construction sequence with
        # Lance-specific component builders, then reuse every inherited method.
        import torch.nn as nn
        from vllm.transformers_utils.configs.bagel import BagelConfig

        from vllm_omni.diffusion.distributed.utils import get_local_device
        from vllm_omni.diffusion.model_loader.diffusers_loader import (
            DiffusersPipelineLoader,
        )

        nn.Module.__init__(self)
        self.od_config = od_config
        self.device = get_local_device()
        self.scheduler = None
        self.scheduler_kwargs = {}

        model = od_config.model
        if os.path.exists(model):
            repo_root = model
        else:
            repo_root = download_weights_from_hf_specific(model, od_config.revision, ["*"])

        # If --model points directly at ``Lance_3B`` / ``Lance_3B_Video`` (or
        # ``Qwen2.5-VL-ViT``), walk up so ``repo_root`` is the bundled top-level
        # dir that owns the sibling components (VAE, ViT).
        base = os.path.basename(repo_root.rstrip("/"))
        if base in {_IMAGE_CKPT_DIR, _VIDEO_CKPT_DIR, _VIT_DIR}:
            parent = os.path.dirname(repo_root.rstrip("/"))
            if os.path.isdir(parent) and os.path.isfile(os.path.join(parent, _VAE_FILE)):
                repo_root = parent

        is_video = self._select_video_variant(od_config) or base == _VIDEO_CKPT_DIR
        ckpt_dir = _VIDEO_CKPT_DIR if is_video else _IMAGE_CKPT_DIR
        ckpt_path = os.path.join(repo_root, ckpt_dir)
        if not os.path.isdir(ckpt_path):
            # Some users point --model directly at a single checkpoint dir.
            ckpt_path = repo_root
        self._repo_root = repo_root
        self._ckpt_path = ckpt_path
        self._is_video = is_video
        if is_video:
            logger.info(
                "Lance video checkpoint selected (%s) — wired with "
                "LancePositionEmbedding3D and Wan2.2 multi-frame VAE decode "
                "(text-to-video). Image-edit / x2t_video remain follow-ups.",
                ckpt_dir,
            )

        # ---- LLM (Qwen2-MoT; identical weight layout to BAGEL) ----
        llm_cfg_path = os.path.join(ckpt_path, "llm_config.json")
        llm_config = Qwen2MoTConfig.from_json_file(llm_cfg_path)
        llm_config.qk_norm = True
        # The released Lance checkpoint ships a separate ``language_model.lm_head.weight``
        # tensor even though ``llm_config.json`` says ``tie_word_embeddings=True``; keep
        # the head untied so the checkpoint loads with zero missing/unexpected keys.
        llm_config.tie_word_embeddings = False
        llm_config.layer_module = od_config.override_transformer_cls_name or "Qwen2MoTDecoderLayer"
        # Lance is Qwen2.5-VL-MoT and ships ``rope_scaling = {"type": "mrope",
        # "mrope_section": [16, 24, 24]}``.  Keep the mRoPE configuration:
        # :class:`BagelRotaryEmbedding` now auto-dispatches on ``rope_type`` and
        # consumes 3-D ``(t, h, w)`` position ids, while :class:`LanceBagel`
        # broadcasts scalar positions to ``(3, S)`` for text-only blocks and
        # emits true per-token ``(t, h, w)`` for video latent blocks.  Required
        # for the Qwen2.5-VL backbone to produce coherent x2t / t2v output.
        if not isinstance(getattr(llm_config, "rope_scaling", None), dict):
            llm_config.rope_scaling = {"rope_type": "mrope", "mrope_section": [16, 24, 24]}
        else:
            llm_config.rope_scaling.setdefault("rope_type", llm_config.rope_scaling.get("type", "mrope"))
            llm_config.rope_scaling.setdefault("mrope_section", [16, 24, 24])

        self.tokenizer = self._load_tokenizer(ckpt_path)
        self.tokenizer, self.new_token_ids, _ = add_special_tokens(self.tokenizer)
        # Image / video preprocessor for the understanding paths
        # (x2t_image / x2t_video / image_edit / video_edit).  Lance's HF repo
        # does not bundle a ``preprocessor_config.json``; we use transformers'
        # ``Qwen2VLImageProcessor`` / ``Qwen2VLVideoProcessor`` with their
        # default CLIP-style normalization, ``patch_size=14``, ``merge_size=2``,
        # which matches the bundled ``Qwen2.5-VL-ViT/config.json`` exactly.
        self.image_processor = self._build_image_processor()
        self.video_processor = self._build_video_processor()
        tok_len = len(self.tokenizer)
        required_max_id = max(int(v) for v in self.new_token_ids.values())
        llm_config.vocab_size = max(
            int(getattr(llm_config, "vocab_size", tok_len)),
            int(tok_len),
            int(required_max_id + 1),
        )

        parallel_config = od_config.parallel_config if od_config else None
        quant_config = od_config.quantization_config

        self.language_model = Qwen2MoTForCausalLM(
            llm_config,
            parallel_config=parallel_config,
            quant_config=quant_config,
            prefix="bagel.language_model",
        )
        self.transformer = self.language_model.model

        # ---- Understanding ViT: Qwen2.5-VL vision (bundled) ----
        self.vit_model = self._build_qwen2_5_vl_vit(repo_root)

        # ---- VAE: Wan2.2 (bundled .pth) ----
        self.vae = self._build_wan22_vae(repo_root)

        vae_cfg = _LanceVaeCfg()
        vit_cfg = _LanceVitCfg(
            patch_size=14,
            hidden_size=getattr(getattr(self.vit_model, "config", None), "hidden_size", 1280),
        )

        # Lance uses Qwen2.5-VL's vision tower whose ``merger`` already projects
        # to the LLM hidden size and which carries its own positional encoding;
        # there is no separate BAGEL-style ``connector`` / ``vit_pos_embed`` in
        # the released checkpoint (Lance_3B safetensors only contain
        # ``vit_model.*``, never ``connector.*`` / ``vit_pos_embed.*``).  We keep
        # BAGEL's ``visual_und=True`` so ``vit_model`` is registered as a child
        # of ``self.bagel`` and BAGEL's ``forward_cache_update_vit`` works
        # unchanged, then immediately replace ``connector`` with an identity and
        # ``vit_pos_embed`` with a zero op so the strict load check does not
        # demand those phantom weights and the addition in
        # ``forward_cache_update_vit`` is numerically a no-op.
        und_enabled = self.vit_model is not None
        self.bagel = LanceBagel(
            language_model=self.language_model,
            vit_model=self.vit_model,
            parallel_config=parallel_config,
            quant_config=quant_config,
            prefix="bagel",
            config=BagelConfig(
                llm_config=llm_config,
                vae_config=vae_cfg,
                vit_config=vit_cfg,
                vit_max_num_patch_per_side=LANCE_DEFAULTS.vit_max_num_patch_per_side,
                connector_act=LANCE_DEFAULTS.connector_act,
                interpolate_pos=False,
                latent_patch_size=LANCE_DEFAULTS.latent_patch_size_spatial,
                max_latent_size=LANCE_DEFAULTS.max_latent_size,
                timestep_shift=LANCE_DEFAULTS.timestep_shift,
                visual_gen=True,
                visual_und=und_enabled,
            ),
        )
        if und_enabled:
            self.bagel.connector = LanceIdentityConnector()
            self.bagel.vit_pos_embed = LanceZeroVitPosEmbed()
            # Hand the Qwen2-VL processors to LanceBagel.prepare_vit_images /
            # prepare_vit_videos so they can re-call them and recover
            # ``image_grid_thw`` / ``video_grid_thw`` (BAGEL's lambda drops the
            # grid).
            self.bagel._lance_image_processor = self.image_processor
            self.bagel._lance_video_processor = self.video_processor

        # Lance_3B_Video ships a 3-D latent positional table
        # (``latent_pos_embed.pos_embed`` shape ``(max_num_frames * max_latent_size**2, hidden_size)``)
        # in place of BAGEL's 2-D ``PositionEmbedding(max_latent_size, hidden)``;
        # swap in :class:`LancePositionEmbedding3D` so the video checkpoint loads
        # cleanly.  ``max_num_frames`` is derived from the table size.
        if is_video:
            self.bagel.latent_pos_embed = LancePositionEmbedding3D(
                max_num_frames=LANCE_DEFAULTS.max_num_video_latent_frames,
                max_num_patch_per_side=LANCE_DEFAULTS.max_latent_size,
                hidden_size=llm_config.hidden_size,
            )

        # Weight sources.  Verified against bytedance-research/Lance:
        #   * Lance_3B/model.safetensors carries the LLM + connectors with
        #     keys ``language_model.* / vae2llm.* / llm2vae.* / time_embedder.* /
        #     latent_pos_embed.*`` — load under the ``bagel.`` namespace.
        #   * Lance_3B_Video/model.safetensors additionally includes
        #     ``vit_model.*`` (390 tensors).
        #   * Qwen2.5-VL-ViT/vit.safetensors (image checkpoint only) carries
        #     the bare ``blocks.* / merger.* / patch_embed.*`` and must land
        #     under ``bagel.vit_model.vision_model.*`` to match the
        #     :class:`LanceQwen2_5_VLNaViTWrapper` hierarchy.
        # Always use the resolved repo_root for weight sources so subfolder paths
        # resolve correctly even when the user passed a per-checkpoint subdir
        # (we already walked ``repo_root`` up to the bundled root above).
        weights_model = repo_root if os.path.isdir(repo_root) else od_config.model
        self.weights_sources = [
            DiffusersPipelineLoader.ComponentSource(
                model_or_path=weights_model,
                subfolder=ckpt_dir if ckpt_path != repo_root else None,
                revision=od_config.revision,
                prefix="bagel.",
                fall_back_to_pt=False,
            ),
        ]
        # Always overlay ``Qwen2.5-VL-ViT/vit.safetensors`` when the bundle
        # contains it.  Upstream Lance's ``inference_lance.py`` loads ViT
        # weights from ``vit.safetensors`` unconditionally (line 455-458),
        # using the released understanding-ViT checkpoint regardless of t2i
        # vs t2v / image_edit vs video_edit.  Lance_3B_Video bundles its own
        # ``vit_model.*`` inside ``model.safetensors``, but loading those
        # produced byte-different ViT outputs from upstream (rel_l2 ≈ 9
        # at the merger output) — the bundled video-checkpoint ViT diverges
        # from the standalone vit.safetensors.  Mirroring upstream's load
        # order (vit.safetensors LAST so it wins) restores byte-identical
        # ViT outputs for both image_edit and video_edit.
        if und_enabled and os.path.isdir(os.path.join(repo_root, _VIT_DIR)):
            self.weights_sources.append(
                DiffusersPipelineLoader.ComponentSource(
                    model_or_path=weights_model,
                    subfolder=_VIT_DIR,
                    revision=od_config.revision,
                    prefix="bagel.vit_model.vision_model.",
                    fall_back_to_pt=False,
                )
            )

        if quant_config is None and not (od_config.enable_layerwise_offload or od_config.parallel_config.use_hsdp):
            self.to(self.device)
        self.setup_diffusion_pipeline_profiler(
            enable_diffusion_pipeline_profiler=self.od_config.enable_diffusion_pipeline_profiler
        )

    # ------------------------------------------------------------------ #
    # Lance-specific component builders
    # ------------------------------------------------------------------ #
    @staticmethod
    def _select_video_variant(od_config: OmniDiffusionConfig) -> bool:
        """Pick Lance_3B vs Lance_3B_Video.  Defaults to image; a user can
        force video via the model path ending in the video dir or an
        od_config extra flag."""
        model = str(od_config.model)
        if model.rstrip("/").endswith(_VIDEO_CKPT_DIR):
            return True
        extra = getattr(od_config, "extra", None) or {}
        return bool(extra.get("lance_video", False))

    @staticmethod
    def _load_tokenizer(ckpt_path: str):
        from transformers import AutoTokenizer

        return AutoTokenizer.from_pretrained(ckpt_path, local_files_only=True, trust_remote_code=True)

    @staticmethod
    def _build_image_processor():
        """Construct a Qwen2.5-VL-compatible image preprocessor.

        Defaults mirror ``Qwen/Qwen2.5-VL-3B-Instruct`` (CLIP normalization,
        14-pixel patches, 2x spatial merge).
        """
        from transformers import Qwen2VLImageProcessor

        return Qwen2VLImageProcessor()

    @staticmethod
    def _build_video_processor():
        """Construct a Qwen2.5-VL-compatible video preprocessor.

        Used by the ``x2t_video`` understanding path.  Returns
        ``pixel_values_videos`` + 3-D ``video_grid_thw = [T_lat, H, W]``.
        """
        from transformers import Qwen2VLVideoProcessor

        return Qwen2VLVideoProcessor()

    def _build_qwen2_5_vl_vit(self, repo_root: str):
        """Build the bundled Qwen2.5-VL vision tower and wrap it NaViT-style.

        Lance's bundled ``Qwen2.5-VL-ViT/config.json`` sets
        ``_attn_implementation = "flash_attention_2"``; we force ``sdpa``
        instead so the tower constructs on hardware (e.g. Blackwell) where
        ``flash-attn`` is not present, falling back to PyTorch's native
        scaled-dot-product attention.  Numerically equivalent for inference.
        """
        vit_dir = os.path.join(repo_root, _VIT_DIR)
        cfg_path = os.path.join(vit_dir, "config.json")
        with open(cfg_path, encoding="utf-8") as f:
            vit_cfg_dict = json.load(f)
        vit_cfg_dict = dict(vit_cfg_dict)
        vit_cfg_dict["_attn_implementation"] = "sdpa"
        try:
            from transformers.models.qwen2_5_vl.configuration_qwen2_5_vl import (
                Qwen2_5_VLVisionConfig,
            )
            from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import (
                Qwen2_5_VisionTransformerPretrainedModel,
            )

            vit_conf = Qwen2_5_VLVisionConfig(**vit_cfg_dict)
            vision = Qwen2_5_VisionTransformerPretrainedModel(vit_conf)
        except Exception as e:
            logger.warning(
                "Could not instantiate Qwen2.5-VL vision tower (%s). "
                "Understanding (x2t) path will be unavailable; generation (t2i) "
                "does not use the ViT.",
                e,
            )
            return None
        return LanceQwen2_5_VLNaViTWrapper(vision, spatial_merge_size=int(vit_cfg_dict.get("spatial_merge_size", 2)))

    # ------------------------------------------------------------------ #
    # Shared prefill helpers used by image_edit / i2v / video_edit.
    # Each mutates ``ctx`` in place: ``ctx["past_key_values"]`` /
    # ``ctx["kv_lens"]`` / ``ctx["ropes"]``.
    # ------------------------------------------------------------------ #
    def _autocast_kwargs(self) -> dict:
        return dict(
            device_type=self.device.type,
            enabled=self.device.type != "cpu",
            dtype=self.od_config.dtype,
        )

    def _new_gen_context(self) -> dict:
        from .lance_transformer import NaiveCache

        return {
            "kv_lens": [0],
            "ropes": [0],
            "past_key_values": NaiveCache(self.bagel.config.llm_config.num_hidden_layers),
        }

    @staticmethod
    def _segment_strings(task: str, modality: str) -> tuple[str, str]:
        sys_prompt = SYSTEM_PROMPTS[(task, modality)]
        return (
            f"<|im_start|>system\n{sys_prompt}<|im_end|>\n<|im_start|>user\n",
            "<|im_end|>\n<|im_start|>assistant\n",
        )

    def _raw_text_prefill(self, ctx: dict, text_str: str) -> None:
        """Tokenize ``text_str`` (no bos/eos) and append to ``ctx`` KV cache."""
        text_ids = self.tokenizer.encode(text_str, add_special_tokens=False)
        if not text_ids:
            return
        curr_kvlen = ctx["kv_lens"][0]
        curr_rope = ctx["ropes"][0]
        seq_len = len(text_ids)
        inp = {
            "text_token_lens": torch.tensor([seq_len], dtype=torch.int, device=self.device),
            "packed_text_ids": torch.tensor(text_ids, dtype=torch.long, device=self.device),
            "packed_text_position_ids": torch.arange(
                curr_rope, curr_rope + seq_len, dtype=torch.long, device=self.device
            ),
        }
        with torch.autocast(**self._autocast_kwargs()):
            ctx["past_key_values"] = self.bagel.forward_cache_update_text(ctx["past_key_values"], **inp)
        ctx["kv_lens"] = [curr_kvlen + seq_len]
        ctx["ropes"] = [curr_rope + seq_len]

    # ``prepare_vit_*`` returns ``packed_indexes`` / ``packed_key_value_indexes``
    # / ``key_values_lens`` for historical reasons; ``forward_cache_update_vit``
    # post-main-merge no longer accepts them.
    _STALE_PREFILL_KEYS = ("packed_indexes", "packed_key_value_indexes", "key_values_lens")

    def _vit_image_prefill(self, ctx: dict, images: list, transforms) -> None:
        """ViT prefill from PIL images via ``prepare_vit_images``."""
        inp, new_kvlens, new_rope = self.bagel.prepare_vit_images(
            curr_kvlens=ctx["kv_lens"],
            curr_rope=ctx["ropes"],
            images=images,
            transforms=transforms,
            new_token_ids=self.new_token_ids,
        )
        for k, v in inp.items():
            if torch.is_tensor(v):
                inp[k] = v.to(self.device)
        for k in self._STALE_PREFILL_KEYS:
            inp.pop(k, None)
        with torch.autocast(**self._autocast_kwargs()):
            ctx["past_key_values"] = self.bagel.forward_cache_update_vit(ctx["past_key_values"], **inp)
        ctx["kv_lens"] = new_kvlens
        ctx["ropes"] = new_rope

    def _vit_video_prefill(
        self,
        ctx: dict,
        videos: list,
        precomputed_vit: list | None = None,
    ) -> None:
        """ViT prefill from videos via ``prepare_vit_videos``.

        Pass ``precomputed_vit=[(pixels, grid_thw)]`` to bypass the
        Qwen2VLImageProcessor smart-resize and feed already-bucketed
        ViT patches directly (used by i2v / video_edit to keep grid
        dimensions aligned with the ref VAE block).  Omit it to let
        ``prepare_vit_videos`` run the processor itself (x2t_video).
        """
        kwargs = dict(
            curr_kvlens=ctx["kv_lens"],
            curr_rope=ctx["ropes"],
            videos=videos,
            new_token_ids=self.new_token_ids,
        )
        if precomputed_vit is not None:
            kwargs["precomputed_vit"] = precomputed_vit
        inp, new_kvlens, new_rope = self.bagel.prepare_vit_videos(**kwargs)
        for k, v in inp.items():
            if torch.is_tensor(v):
                inp[k] = v.to(self.device)
        for k in self._STALE_PREFILL_KEYS:
            inp.pop(k, None)
        with torch.autocast(**self._autocast_kwargs()):
            ctx["past_key_values"] = self.bagel.forward_cache_update_vit(ctx["past_key_values"], **inp)
        ctx["kv_lens"] = new_kvlens
        ctx["ropes"] = new_rope

    def _vae_ref_prefill(
        self,
        ctx: dict,
        images: list,
        transforms,
        *,
        is_video: bool,
        latent_cache: list,
    ) -> None:
        """VAE prefill of a reference image / video at timestep 0.

        ``latent_cache`` is a single-slot list used to share the encoded
        posterior between the gen branch and the cfg_text branch — Wan2.2's
        ``mu + std * randn_like(std)`` samples a fresh latent every call,
        which otherwise drifts the two branches' KV caches.
        """
        inp, new_kvlens, new_rope = self.bagel._lance_native_prepare_vae_images(
            curr_kvlens=ctx["kv_lens"],
            curr_rope=ctx["ropes"],
            images=images,
            transforms=transforms,
            new_token_ids=self.new_token_ids,
            is_video=is_video,
        )
        for k, v in inp.items():
            if torch.is_tensor(v):
                inp[k] = v.to(self.device)
        if not latent_cache:
            latent_cache.append(self.vae.encode(inp["padded_images"].to(self.device)))
        inp["precomputed_latent"] = latent_cache[0]
        with torch.autocast(**self._autocast_kwargs()):
            ctx["past_key_values"] = self.bagel.forward_cache_update_vae(self.vae, ctx["past_key_values"], **inp)
        ctx["kv_lens"] = new_kvlens
        ctx["ropes"] = new_rope

    # ------------------------------------------------------------------ #
    # Lance text-to-video forward path
    # ------------------------------------------------------------------ #
    def forward(self, req):  # type: ignore[override]
        """Dispatch on prompt modality.

        - ``modalities == ["video"]`` (text-to-video) → :meth:`_forward_t2v`
          (3-D latents + ``LanceWanVAE.decode_video``).
        - ``modalities == ["text"]`` + ``multi_modal_data.video`` (x2t_video) →
          :meth:`_forward_x2t_video` (multi-frame Qwen2.5-VL ViT prefill).
        - ``modalities == ["image"]`` + ``multi_modal_data.img2img`` (image_edit)
          → :meth:`_forward_image_edit` (Lance-native VAE+ViT prefill + image gen).
        - ``modalities == ["video"]`` + ``multi_modal_data.video`` (video_edit)
          → :meth:`_forward_video_edit` (Lance-native multi-frame VAE+ViT prefill
          + video gen).
        - Everything else falls through to :meth:`BagelPipeline.forward`
          (t2i, x2t_image).
        """
        first_prompt = req.prompts[0] if req.prompts else None
        modalities: list[str] = []
        mm_data: dict = {}
        if isinstance(first_prompt, dict):
            modalities = first_prompt.get("modalities") or []
            mm_data = first_prompt.get("multi_modal_data") or {}
        if "video" in modalities and mm_data.get("first_frame") is not None:
            return self._forward_i2v(req)
        if "video" in modalities and mm_data.get("video") is not None:
            return self._forward_video_edit(req)
        if "video" in modalities:
            return self._forward_t2v(req)
        if "text" in modalities and mm_data.get("video") is not None:
            return self._forward_x2t_video(req)
        if "text" in modalities and mm_data.get("image") is not None:
            return self._forward_x2t_image(req)
        if "image" in modalities and (mm_data.get("img2img") is not None or mm_data.get("image") is not None):
            return self._forward_image_edit(req)
        # t2i falls through to BAGEL parent.  Inject Lance defaults if the
        # caller hasn't set them: upstream uses ``cfg_text_scale=4.0`` and
        # ``cfg_vit_scale=1.0`` (= cfg_img_scale=1.0 in BAGEL terms).
        sp = req.sampling_params
        if not hasattr(sp, "extra_args") or sp.extra_args is None:
            sp.extra_args = {}
        sp.extra_args.setdefault("cfg_img_scale", 1.0)
        return super().forward(req)

    @torch.inference_mode()
    def _forward_t2v(self, req):
        """Minimal text-to-video forward: text prefill + 3-D latent denoising +
        Wan2.2 multi-frame decode.  Mirrors :meth:`BagelPipeline.forward`'s t2i
        branch but with 3-D latents (no image input, no x2t)."""
        from copy import deepcopy

        from vllm_omni.diffusion.data import DiffusionOutput

        from .lance_transformer import NaiveCache

        first_prompt = req.prompts[0]
        if isinstance(first_prompt, dict):
            prompt = first_prompt.get("prompt") or ""
            extra_args = first_prompt.get("extra_args") or {}
        else:
            prompt = str(first_prompt)
            extra_args = {}
        # Sampling-side extras override prompt-side.
        sp_extra = getattr(req.sampling_params, "extra_args", {}) or {}
        extra_args = {**extra_args, **sp_extra}

        # Video shape.  T = number of RGB frames (1..121), H/W in pixels.
        T = int(extra_args.get("num_frames", 25))
        H = int(req.sampling_params.height or extra_args.get("video_height", 480))
        W = int(req.sampling_params.width or extra_args.get("video_width", 768))
        max_lat = self.bagel.max_latent_size
        max_hw = max_lat * self.bagel.latent_downsample
        if H > max_hw or W > max_hw:
            raise ValueError(f"Requested video resolution {H}x{W} exceeds Lance limit {max_hw}x{max_hw}")
        downsample_t = int(getattr(self.bagel.config.vae_config, "downsample_temporal", 4))
        max_T_lat = LANCE_DEFAULTS.max_num_video_latent_frames
        if (T - 1) // downsample_t + 1 > max_T_lat:
            raise ValueError(
                f"Requested num_frames={T} exceeds Lance video temporal limit "
                f"{(max_T_lat - 1) * downsample_t + 1} (max_num_video_latent_frames={max_T_lat})"
            )
        video_shape = (T, H, W)
        logger.info("Lance t2v: video_shape=%s", video_shape)

        cfg_text_scale = float(extra_args.get("cfg_text_scale", 4.0))
        # Upstream Lance defaults cfg_interval=[0.4, 1.0] (turn off CFG below
        # t=0.4); without this vllm-omni keeps CFG on for late iters and the
        # last few denoise steps diverge from upstream.  cfg_renorm_type /
        # cfg_renorm_min default to 'global' / 0 on both sides.
        cfg_interval = tuple(extra_args.get("cfg_interval", (0.4, 1.0)))
        cfg_renorm_type = str(extra_args.get("cfg_renorm_type", "global"))
        cfg_renorm_min = float(extra_args.get("cfg_renorm_min", 0.0))
        timestep_shift = float(extra_args.get("timestep_shift", LANCE_DEFAULTS.timestep_shift))
        num_timesteps = int(req.sampling_params.num_inference_steps or LANCE_DEFAULTS.num_timesteps)

        if req.sampling_params.seed is not None:
            torch.manual_seed(req.sampling_params.seed)
            if self.device.type == "cuda":
                torch.cuda.manual_seed(req.sampling_params.seed)

        # ---- Build positive-prompt KV cache ----
        gen_context = {
            "kv_lens": [0],
            "ropes": [0],
            "past_key_values": NaiveCache(self.bagel.config.llm_config.num_hidden_layers),
        }
        cfg_text_context = deepcopy(gen_context)

        gen_input_text, newlens, new_rope = self.bagel.prepare_prompts(
            curr_kvlens=gen_context["kv_lens"],
            curr_rope=gen_context["ropes"],
            prompts=[prompt],
            tokenizer=self.tokenizer,
            new_token_ids=self.new_token_ids,
        )
        for k, v in gen_input_text.items():
            if torch.is_tensor(v):
                gen_input_text[k] = v.to(self.device)
        with torch.autocast(
            device_type=self.device.type,
            enabled=self.device.type != "cpu",
            dtype=self.od_config.dtype,
        ):
            gen_context["past_key_values"] = self.bagel.forward_cache_update_text(
                gen_context["past_key_values"], **gen_input_text
            )
        gen_context["kv_lens"] = newlens
        gen_context["ropes"] = new_rope

        # ---- Build CFG text-unconditional KV cache (empty prompt) ----
        if cfg_text_scale > 1.0:
            neg_prompt = str(extra_args.get("negative_prompt") or "")
            neg_input, neg_newlens, neg_rope = self.bagel.prepare_prompts(
                curr_kvlens=cfg_text_context["kv_lens"],
                curr_rope=cfg_text_context["ropes"],
                prompts=[neg_prompt],
                tokenizer=self.tokenizer,
                new_token_ids=self.new_token_ids,
            )
            for k, v in neg_input.items():
                if torch.is_tensor(v):
                    neg_input[k] = v.to(self.device)
            with torch.autocast(
                device_type=self.device.type,
                enabled=self.device.type != "cpu",
                dtype=self.od_config.dtype,
            ):
                cfg_text_context["past_key_values"] = self.bagel.forward_cache_update_text(
                    cfg_text_context["past_key_values"], **neg_input
                )
            cfg_text_context["kv_lens"] = neg_newlens
            cfg_text_context["ropes"] = neg_rope

        # ---- 3-D latent init + CFG side metadata ----
        gen_input_lat = self.bagel.prepare_video_latent(
            curr_kvlens=gen_context["kv_lens"],
            curr_rope=gen_context["ropes"],
            video_shapes=[video_shape],
            new_token_ids=self.new_token_ids,
        )
        for k, v in gen_input_lat.items():
            if torch.is_tensor(v):
                gen_input_lat[k] = v.to(self.device)
        cfg_text_lat = self.bagel.prepare_video_latent_cfg(
            curr_kvlens=cfg_text_context["kv_lens"],
            curr_rope=cfg_text_context["ropes"],
            video_shapes=[video_shape],
        )
        cfg_img_lat = self.bagel.prepare_video_latent_cfg(
            curr_kvlens=gen_context["kv_lens"],
            curr_rope=gen_context["ropes"],
            video_shapes=[video_shape],
        )
        for k, v in cfg_text_lat.items():
            if torch.is_tensor(v):
                cfg_text_lat[k] = v.to(self.device)
        for k, v in cfg_img_lat.items():
            if torch.is_tensor(v):
                cfg_img_lat[k] = v.to(self.device)

        self._regen_init_noise_on_device(gen_input_lat, req.sampling_params.seed)

        # ---- Denoising loop (Bagel.generate_image is rank-agnostic over packed tokens) ----
        with torch.autocast(
            device_type=self.device.type,
            enabled=self.device.type != "cpu",
            dtype=self.od_config.dtype,
        ):
            # ``prepare_video_latent`` / ``prepare_vae_latent`` still return
            # ``key_values_lens`` / ``packed_indexes`` / ``packed_key_value_indexes``
            # but post-main-merge ``generate_image`` doesn't accept them — drop
            # before unpacking via ``**gen_input_lat``.
            for _drop in ("key_values_lens", "packed_indexes", "packed_key_value_indexes"):
                gen_input_lat.pop(_drop, None)
            latents, *_ = self.bagel.generate_image(
                past_key_values=gen_context["past_key_values"],
                cfg_text_past_key_values=cfg_text_context["past_key_values"],
                cfg_img_past_key_values=gen_context["past_key_values"],  # no img CFG branch
                num_timesteps=num_timesteps,
                timestep_shift=timestep_shift,
                cfg_text_scale=cfg_text_scale,
                cfg_img_scale=1.0,
                cfg_interval=cfg_interval,
                cfg_renorm_type=cfg_renorm_type,
                cfg_renorm_min=cfg_renorm_min,
                **gen_input_lat,
                cfg_text_packed_position_ids=cfg_text_lat["cfg_packed_position_ids"],
                # ``cfg_text_packed_query_indexes`` / ``cfg_text_key_values_lens``
                # / ``cfg_text_packed_key_value_indexes`` removed post-main-merge
                # (derived from ``cfg_text_past_key_values``).
                cfg_img_packed_position_ids=cfg_img_lat["cfg_packed_position_ids"],
                # ``cfg_img_*`` index/lens kwargs removed — same as above.
            )

        frames_np = self._decode_video_from_latent(self.bagel, self.vae, latents[0], video_shape)
        # Convert numpy frames to PIL.Image list for downstream serialization.
        frames = [Image.fromarray(f) for f in frames_np]
        logger.info("Lance t2v: decoded %d frames at %dx%d", len(frames), frames[0].width, frames[0].height)
        return DiffusionOutput(
            output=frames[0],
            custom_output={"video_frames": frames, "video_shape": video_shape},
            stage_durations=self.stage_durations if hasattr(self, "stage_durations") else None,
        )

    @torch.inference_mode()
    def _forward_image_edit(self, req):
        """Image edit (img2img): reference image + text prompt → modified image.

        Matches upstream Lance's image_edit prefill structure (see
        ``Lance.validation_gen_KVcache`` and ``validation_dataset.py``).  The
        sequence is laid out as 5 prefill segments + 1 noise QUERY:

          seg1 (causal, modality=system_prompt):
              ``<|im_start|>system\\n{sys}<|im_end|>\\n<|im_start|>user\\n``
          seg2 (full,   modality=ref_vit):
              ViT(ref) — ref image through Qwen2.5-VL ViT
          seg3 (full_noise, modality=ref_source):
              VAE(ref) — ref image through Wan2.2 VAE, time_embed(t=0)
          seg4 (causal, modality=text):  the user instruction
          seg5 (causal, modality=system_prompt):
              ``<|im_end|>\\n<|im_start|>assistant\\n``
          seg6 (noise QUERY): the gen latent to denoise

        cfg_text_context skips ONLY seg4 (the user text instruction) — every
        other segment, including the system header, ViT and VAE prefill, the
        separator and the noise query, is shared between gen_context and
        cfg_text_context.  The rope counter still advances by seg4's length
        for cfg_text_context so that seg5's rope position matches between
        the two branches.

        cfg_img is OFF by default (``cfg_img_scale=1.0`` matches upstream's
        ``cfg_vit_scale=1.0``).
        """

        import numpy as np

        from vllm_omni.diffusion.data import DiffusionOutput

        from .lance_transformer import NaiveCache

        first_prompt = req.prompts[0]
        assert isinstance(first_prompt, dict), "image_edit requires dict-style prompt"
        # The caller may pass either a raw user instruction (e.g. "Remove the
        # hat from the painting.") OR an already-rendered template string
        # (with system + chat template).  Extract just the user instruction
        # so we can rebuild the segments cleanly.
        user_text = first_prompt.get("user_text")
        rendered = first_prompt.get("prompt") or ""
        if not user_text:
            user_text = _extract_user_instruction(rendered)
        mm_data = first_prompt.get("multi_modal_data") or {}
        image_input = mm_data.get("img2img") or mm_data.get("image")
        if image_input is None:
            raise ValueError("image_edit requires multi_modal_data.img2img (or .image).")
        if not isinstance(image_input, list):
            image_input = [image_input]
        image_input = [Image.open(im) if isinstance(im, str) else im for im in image_input]

        extra_args = first_prompt.get("extra_args") or {}
        sp_extra = getattr(req.sampling_params, "extra_args", {}) or {}
        extra_args = {**extra_args, **sp_extra}
        cfg_text_scale = float(extra_args.get("cfg_text_scale", 4.0))
        cfg_img_scale = float(extra_args.get("cfg_img_scale", 1.0))
        cfg_interval = tuple(extra_args.get("cfg_interval", (0.4, 1.0)))
        cfg_renorm_type = str(extra_args.get("cfg_renorm_type", "global"))
        cfg_renorm_min = float(extra_args.get("cfg_renorm_min", 0.0))
        timestep_shift = float(extra_args.get("timestep_shift", LANCE_DEFAULTS.timestep_shift))
        num_timesteps = int(req.sampling_params.num_inference_steps or LANCE_DEFAULTS.num_timesteps)

        # Resize the reference image to a multiple of latent_downsample and
        # within the max latent grid.
        stride = self.bagel.latent_downsample
        max_hw = int(self.bagel.max_latent_size * stride)

        def _resize_to_stride(img):
            if img.mode != "RGB":
                img = img.convert("RGB")
            w, h = img.size
            scale = min(max_hw / max(w, h), 1.0)
            scale = max(scale, min(256, max_hw) / min(w, h))
            new_w = max(stride, int(round(w * scale / stride) * stride))
            new_h = max(stride, int(round(h * scale / stride) * stride))
            new_w = min(new_w, max_hw)
            new_h = min(new_h, max_hw)
            if new_w != w or new_h != h:
                img = img.resize((new_w, new_h), Image.BICUBIC)
            return img

        image_input = [_resize_to_stride(im) for im in image_input]
        resized_w, resized_h = image_input[0].size
        image_shape = (resized_h, resized_w)
        logger.info("Lance image_edit: ref image %dx%d, user_text=%r", resized_w, resized_h, user_text)

        def vae_transforms(img):
            arr = torch.from_numpy(np.array(img)).float() / 127.5 - 1.0
            return arr.permute(2, 0, 1)  # (C, H, W)

        def vit_transforms(img):
            return self.image_processor(images=img, return_tensors="pt").pixel_values[0]

        if req.sampling_params.seed is not None:
            torch.manual_seed(req.sampling_params.seed)
            if self.device.type == "cuda":
                torch.cuda.manual_seed(req.sampling_params.seed)

        gen_context = {
            "kv_lens": [0],
            "ropes": [0],
            "past_key_values": NaiveCache(self.bagel.config.llm_config.num_hidden_layers),
        }
        cfg_text_context = {
            "kv_lens": [0],
            "ropes": [0],
            "past_key_values": NaiveCache(self.bagel.config.llm_config.num_hidden_layers),
        }

        # Single-slot cache shared between gen + cfg branches so the Wan2.2
        # VAE's ``mu + std * randn_like(std)`` sample is computed once.
        ref_latent_cache: list = []

        # ----- prefill segments in upstream order -----
        seg1_str, seg5_str = self._segment_strings("image_edit", "image")

        # seg1: system + user header  (shared)
        self._raw_text_prefill(gen_context, seg1_str)
        if cfg_text_scale > 1.0:
            self._raw_text_prefill(cfg_text_context, seg1_str)

        # seg2: ViT(ref)  (shared)
        self._vit_image_prefill(gen_context, image_input, vit_transforms)
        if cfg_text_scale > 1.0:
            self._vit_image_prefill(cfg_text_context, image_input, vit_transforms)

        # seg3: VAE(ref) with timestep=0  (shared)
        #
        # IMPORTANT: upstream Lance places the gen latent (the noise QUERY)
        # at the SAME mRoPE positions as the ref VAE — they share the same
        # spatial grid (T_lat × H_lat × W_lat), and the model treats them
        # as two timesteps of the same latent location.  Concretely, upstream
        # ``get_rope_index`` emits ``[base_t, base_h+hi, base_w+wi]`` for
        # both the VAE cond block AND the noise block, anchored at the SAME
        # base.  This is the inductive bias that lets the model "edit" the
        # ref image — without this overlap the model has no way to map noise
        # tokens to ref tokens.
        #
        # We therefore snapshot ``ropes`` BEFORE the VAE prefill and reuse
        # that as the gen latent's ``curr_rope`` below.
        rope_before_vae = gen_context["ropes"][0]
        cfg_rope_before_vae = cfg_text_context["ropes"][0] if cfg_text_scale > 1.0 else rope_before_vae
        self._vae_ref_prefill(gen_context, image_input, vae_transforms, is_video=False, latent_cache=ref_latent_cache)
        if cfg_text_scale > 1.0:
            self._vae_ref_prefill(
                cfg_text_context, image_input, vae_transforms, is_video=False, latent_cache=ref_latent_cache
            )

        # seg4: user instruction  (gen only).  Upstream Lance's cfg_text
        # branch is a SHORTER sequence with the user_text segment REMOVED
        # (see ``uncond_split_pro_kvcache`` + ``get_rope_index`` on the
        # filtered text_ids), so cfg_text's rope counter MUST NOT advance
        # past the user_text region.  seg5 (separator) therefore lands at
        # different absolute rope positions in gen vs cfg branches — that's
        # the intended behavior.
        self._raw_text_prefill(gen_context, user_text)

        # seg5: separator + assistant header  (gen + cfg).  Each branch
        # places it at its OWN current rope.
        self._raw_text_prefill(gen_context, seg5_str)
        if cfg_text_scale > 1.0:
            self._raw_text_prefill(cfg_text_context, seg5_str)

        # -- (4) Gen latent at the SAME rope position as the ref VAE block.
        # See the comment on `rope_before_vae` above.  Note the KV cache
        # still contains all prefilled segments (system, ViT, ref VAE, user
        # text, separator); only the gen latent's QUERY rope coordinates are
        # rewound to overlap with the ref VAE.
        gen_input_lat = self.bagel.prepare_vae_latent(
            curr_kvlens=gen_context["kv_lens"],
            curr_rope=[rope_before_vae],
            image_sizes=[image_shape],
            new_token_ids=self.new_token_ids,
        )
        for k, v in gen_input_lat.items():
            if torch.is_tensor(v):
                gen_input_lat[k] = v.to(self.device)
        cfg_text_lat = self.bagel.prepare_vae_latent_cfg(
            curr_kvlens=cfg_text_context["kv_lens"],
            curr_rope=[cfg_rope_before_vae],
            image_sizes=[image_shape],
        )
        for k, v in cfg_text_lat.items():
            if torch.is_tensor(v):
                cfg_text_lat[k] = v.to(self.device)

        # cfg_img branch: keep off (cfg_img_scale=1.0 matches upstream
        # ``cfg_vit_scale=1.0``).  generate_image needs the cfg_img_* args
        # but skips the branch when scale<=1.0; just point them at the
        # cond's lat metadata so shapes/types are fine.
        cfg_img_lat = cfg_text_lat

        self._regen_init_noise_on_device(gen_input_lat, req.sampling_params.seed)

        with torch.autocast(**self._autocast_kwargs()):
            # ``prepare_video_latent`` / ``prepare_vae_latent`` still return
            # ``key_values_lens`` / ``packed_indexes`` / ``packed_key_value_indexes``
            # but post-main-merge ``generate_image`` doesn't accept them — drop
            # before unpacking via ``**gen_input_lat``.
            for _drop in ("key_values_lens", "packed_indexes", "packed_key_value_indexes"):
                gen_input_lat.pop(_drop, None)
            latents, *_ = self.bagel.generate_image(
                past_key_values=gen_context["past_key_values"],
                cfg_text_past_key_values=cfg_text_context["past_key_values"]
                if cfg_text_scale > 1.0
                else gen_context["past_key_values"],
                cfg_img_past_key_values=gen_context["past_key_values"],
                num_timesteps=num_timesteps,
                timestep_shift=timestep_shift,
                cfg_text_scale=cfg_text_scale,
                cfg_img_scale=cfg_img_scale,
                cfg_interval=cfg_interval,
                cfg_renorm_type=cfg_renorm_type,
                cfg_renorm_min=cfg_renorm_min,
                **gen_input_lat,
                cfg_text_packed_position_ids=cfg_text_lat["cfg_packed_position_ids"],
                # ``cfg_text_packed_query_indexes`` / ``cfg_text_key_values_lens``
                # / ``cfg_text_packed_key_value_indexes`` removed post-main-merge
                # (derived from ``cfg_text_past_key_values``).
                cfg_img_packed_position_ids=cfg_img_lat["cfg_packed_position_ids"],
                # ``cfg_img_*`` index/lens kwargs removed — same as above.
            )

        img = self._decode_image_from_latent(self.bagel, self.vae, latents[0], image_shape)
        # Put the PIL image into ``custom_output`` too — the orchestrator
        # serializes ``custom_output`` across the IPC boundary but strips
        # the bare ``output`` field, so consumers that only see
        # ``outputs[0].output`` (e.g. gradio_demo) get ``None`` otherwise.
        return DiffusionOutput(
            output=img,
            custom_output={"image": img, "image_shape": image_shape},
            stage_durations=self.stage_durations if hasattr(self, "stage_durations") else None,
        )

    @torch.inference_mode()
    @torch.inference_mode()
    def _forward_i2v(self, req):
        """Image-to-Video (upstream-faithful first-frame conditioning).

        Mirrors upstream Lance PR #33's ``ff2v_sample`` + ``validation_gen_KVcache``
        math byte-for-byte at the algorithmic level:

          1. Treat the input image as 4 replicated pixel frames
             ``(3, 4, H, W)``.  This matches upstream
             ``vae_tensor[:, :4, :, :] = vae_image_tensor[:, 0:1, :, :].repeat(1, 4, 1, 1)``
             — required because Wan2.2 VAE has temporal stride 4, so 4
             pixel frames collapse to 1 latent frame.

          2. VAE-encode → first latent frame at the image content.

          3. Inject the encoded latent into ``packed_init_noises`` at
             the first ``h_lat × w_lat`` token slice (frame 0).

          4. Pass ``frame_condition_token_indexes = arange(0, h_lat*w_lat)``
             to ``Bagel.generate_image`` so that:
               * ``timestep[cond] = 0`` every iter (clean signal),
               * ``x_t[cond]`` is restored verbatim after each
                 velocity update (never denoised).

        Plus a ViT+VAE-ref prefill of the same image (so the LLM also
        attends to it through KV cache, matching upstream's
        gen-pre-condition pass).
        """

        import numpy as _np
        from PIL import Image as _PILImage

        from vllm_omni.diffusion.data import DiffusionOutput

        from .lance_transformer import NaiveCache

        first_prompt = req.prompts[0]
        assert isinstance(first_prompt, dict), "i2v requires dict-style prompt"
        user_text = first_prompt.get("user_text")
        rendered = first_prompt.get("prompt") or ""
        if not user_text:
            user_text = _extract_user_instruction(rendered)
        mm_data = first_prompt.get("multi_modal_data") or {}
        image_input = mm_data.get("first_frame")
        if image_input is None:
            raise ValueError("i2v requires multi_modal_data.first_frame.")

        # Resolve image to (H, W, 3) uint8 RGB, then wrap as 1-frame video.
        if isinstance(image_input, _PILImage.Image):
            image_raw = _np.array(image_input.convert("RGB"))
        elif isinstance(image_input, str):
            image_raw = _np.array(_PILImage.open(image_input).convert("RGB"))
        elif isinstance(image_input, _np.ndarray):
            image_raw = image_input if image_input.ndim == 3 else image_input.squeeze()
        else:
            raise ValueError(f"Unsupported first_frame type {type(image_input)}")
        video_raw = image_raw[None]  # (1, H, W, 3)

        extra_args = first_prompt.get("extra_args") or {}
        sp_extra = getattr(req.sampling_params, "extra_args", {}) or {}
        extra_args = {**extra_args, **sp_extra}

        # Output video shape — independent of input image dims.
        num_frames_out = int(extra_args.get("num_frames", 61))
        out_H = int(extra_args.get("video_height", 480))
        out_W = int(extra_args.get("video_width", 848))
        origin_fps = float(extra_args.get("origin_fps", 24.0))

        # Upstream Lance i2v default is cfg_text_scale=4.0.  Note: vllm-omni's
        # bf16 attention trajectory (vllm-native kernels vs upstream's
        # HF-native ones) compounds per-layer drift that may DAMPEN prompt-
        # directed motion for subtle prompts (e.g. micro facial expression).
        # For such cases, boost cfg_text_scale to 10-15 via
        # ``extra_args["cfg_text_scale"]`` to amplify prompt influence.
        # WARNING: too-high cfg (e.g. 15) can cause anatomical distortion
        # for natural-motion prompts (e.g. animals walking).  Tune per case.
        cfg_text_scale = float(extra_args.get("cfg_text_scale", 4.0))
        cfg_img_scale = float(extra_args.get("cfg_img_scale", 1.0))
        # Upstream Lance i2v default (config/config_factory.py:cfg_interval):
        # [0.4, 1.0] — CFG active in high-noise phase only.  Always-on CFG
        # ``(0, 1)`` over-constrains the late-step velocity toward the
        # conditioned first frame, suppressing subtle motion (e.g. micro
        # facial expression changes in ex5).
        cfg_interval = tuple(extra_args.get("cfg_interval", (0.4, 1.0)))
        cfg_renorm_type = str(extra_args.get("cfg_renorm_type", "global"))
        cfg_renorm_min = float(extra_args.get("cfg_renorm_min", 0.0))
        timestep_shift = float(extra_args.get("timestep_shift", LANCE_DEFAULTS.timestep_shift))
        num_timesteps = int(req.sampling_params.num_inference_steps or LANCE_DEFAULTS.num_timesteps)

        # Bucket-resize + VAE preprocess the 1-frame "video" (image).  i2v
        # does not run a ViT prefill on the ref (the first-frame latent pin
        # provides all the image conditioning), so the ViT outputs are
        # discarded — see the comment further down on upstream PR #33.
        from .lance_transformer import LanceBagel as _LanceBagel

        vae_video, _, _, _ = _LanceBagel._lance_video_preprocess(video_raw, origin_fps)
        T_ref = int(vae_video.shape[1])
        H_ref = int(vae_video.shape[2])
        W_ref = int(vae_video.shape[3])

        # Target output shape (independent of ref) — pass through the same
        # VAE-aligned bucket so the gen latent dims are valid.  We pick H/W
        # from extra_args and snap to the VAE downsample grid (×8 spatial,
        # ×4 temporal for Wan2.2 VAE).  num_frames_out controls T.
        # Wan2.2 latent dims: t_lat = (T - 1) // 4 + 1, h_lat = H // 8, w_lat = W // 8.
        out_shape = (num_frames_out, out_H, out_W)

        logger.info(
            "Lance i2v: ref image %dx%d → 1-frame VAE %dx%dx%d, target output %dx%dx%d (T,H,W), user_text=%r",
            image_raw.shape[0],
            image_raw.shape[1],
            T_ref,
            H_ref,
            W_ref,
            num_frames_out,
            out_H,
            out_W,
            user_text,
        )

        if req.sampling_params.seed is not None:
            torch.manual_seed(req.sampling_params.seed)
            if self.device.type == "cuda":
                torch.cuda.manual_seed(req.sampling_params.seed)

        gen_context = {
            "kv_lens": [0],
            "ropes": [0],
            "past_key_values": NaiveCache(self.bagel.config.llm_config.num_hidden_layers),
        }
        cfg_text_context = {
            "kv_lens": [0],
            "ropes": [0],
            "past_key_values": NaiveCache(self.bagel.config.llm_config.num_hidden_layers),
        }

        # i2v system prompt (matches t2v wording per upstream).
        seg1_str, seg5_str = self._segment_strings("i2v", "video")

        # Upstream Lance PR #33 i2v does NOT prefill the input image via
        # ViT/VAE as a "reference".  The image conditioning happens entirely
        # through the first-frame latent pin (cond positions in x_t set to
        # VAE-encoded image, never updated through denoise loop).  Prefill
        # is text-only: system prompt → user instruction → assistant header.
        # Adding ViT/VAE prefill (which video_edit does) over-constrains the
        # generation and prevents the prompt-driven motion from developing.
        self._raw_text_prefill(gen_context, seg1_str)
        if cfg_text_scale > 1.0:
            self._raw_text_prefill(cfg_text_context, seg1_str)

        # Snapshot rope BEFORE adding user text; gen latent will use this
        # as its starting rope (matches upstream's modality==1≡2 trick).
        rope_before_vae = gen_context["ropes"][0]
        cfg_rope_before_vae = cfg_text_context["ropes"][0] if cfg_text_scale > 1.0 else rope_before_vae

        self._raw_text_prefill(gen_context, user_text)
        self._raw_text_prefill(gen_context, seg5_str)
        if cfg_text_scale > 1.0:
            self._raw_text_prefill(cfg_text_context, seg5_str)

        # GEN LATENT — at OUTPUT shape (not ref shape).  Place at rope
        # starting from rope_before_vae (the modality==1≡2 trick).  The KV
        # cache still holds the ref VAE at its original (smaller) rope
        # range; the noise query's own rope just starts at the same
        # ``rope_before_vae`` and extends through num_vid_tokens.
        gen_input_lat = self.bagel.prepare_video_latent(
            curr_kvlens=gen_context["kv_lens"],
            curr_rope=[rope_before_vae],
            video_shapes=[out_shape],
            new_token_ids=self.new_token_ids,
        )
        for k, v in gen_input_lat.items():
            if torch.is_tensor(v):
                gen_input_lat[k] = v.to(self.device)
        cfg_text_lat = self.bagel.prepare_video_latent_cfg(
            curr_kvlens=cfg_text_context["kv_lens"],
            curr_rope=[cfg_rope_before_vae],
            video_shapes=[out_shape],
        )
        for k, v in cfg_text_lat.items():
            if torch.is_tensor(v):
                cfg_text_lat[k] = v.to(self.device)
        cfg_img_lat = cfg_text_lat

        self._regen_init_noise_on_device(gen_input_lat, req.sampling_params.seed)

        # ── First-frame conditioning: VAE-encode the FULL pixel-space
        # target tensor (matches upstream Lance PR #33's ff2v_sample):
        #
        #   vae_tensor = torch.randn([3, num_frames, H, W])
        #   vae_tensor[:, :4, :, :] = image.repeat(1, 4, 1, 1)
        #
        # Then VAE-encode the whole thing and take latent frame 0.  This is
        # important — Wan2.2 VAE has temporal convolutions, so the first
        # latent frame's value depends on neighbouring pixel frames (not
        # just frames 0-3).  Encoding only the 4 image-replicated frames
        # (which is what we did before) gives a *different* latent than
        # upstream's full-sequence encode, and that delta produces an
        # over-anchoring effect at i2v inference time (ex5 portrait case
        # was visibly frozen because of this).
        h_lat = int(out_H // self.bagel.latent_downsample)
        w_lat = int(out_W // self.bagel.latent_downsample)
        first_slice_n = h_lat * w_lat

        from torchvision.transforms.functional import resize as _tv_resize

        img_chw = torch.from_numpy(image_raw).permute(2, 0, 1).float() / 127.5 - 1.0  # (3, H, W) in [-1, 1]
        img_chw = _tv_resize(img_chw, [out_H, out_W], antialias=True)

        # Build the full pixel-space target tensor: random noise everywhere,
        # except the first 4 pixel frames are the input image (replicated).
        # Wan2.2 VAE collapses every 4 pixel frames → 1 latent frame, so the
        # first latent slice will represent the conditioning image.
        full_pixel_tensor = torch.randn((3, num_frames_out, out_H, out_W), dtype=torch.float32, device=self.device)
        full_pixel_tensor[:, :4, :, :] = img_chw.to(self.device).unsqueeze(1).repeat(1, 4, 1, 1)
        full_pixel_5d = full_pixel_tensor.unsqueeze(0)  # (1, 3, num_frames_out, H, W)

        with torch.autocast(**self._autocast_kwargs()):
            image_latent_5d = self.vae.encode(full_pixel_5d)  # (1, 48, T_lat, h_lat, w_lat)

        # Extract first latent frame in (h_lat, w_lat, 48) layout, then
        # flatten to (h_lat*w_lat, 48) to match packed_init_noises[:first_slice_n].
        if image_latent_5d.dim() == 5:
            image_latent = image_latent_5d[0, :, 0, :, :]  # (48, h_lat, w_lat)
        elif image_latent_5d.dim() == 4:
            # encode may squeeze T=1 if the caller passed 4-D input — we
            # passed 5-D so this branch shouldn't fire, but keep defensive.
            image_latent = image_latent_5d[0]
        else:
            raise RuntimeError(f"Unexpected VAE-encoded shape {tuple(image_latent_5d.shape)}")
        image_latent_packed = image_latent.permute(1, 2, 0).reshape(first_slice_n, -1)

        init_noises = gen_input_lat["packed_init_noises"]
        image_latent_packed = image_latent_packed.to(device=init_noises.device, dtype=init_noises.dtype)
        init_noises[:first_slice_n] = image_latent_packed
        gen_input_lat["packed_init_noises"] = init_noises

        frame_condition_token_indexes = torch.arange(first_slice_n, dtype=torch.long, device=self.device)

        logger.info(
            "Lance i2v cond: latent grid h_lat=%d w_lat=%d -> pinning first %d tokens (frame 0) at VAE-encoded image",
            h_lat,
            w_lat,
            first_slice_n,
        )

        with torch.autocast(**self._autocast_kwargs()):
            # ``prepare_video_latent`` / ``prepare_vae_latent`` still return
            # ``key_values_lens`` / ``packed_indexes`` / ``packed_key_value_indexes``
            # but post-main-merge ``generate_image`` doesn't accept them — drop
            # before unpacking via ``**gen_input_lat``.
            for _drop in ("key_values_lens", "packed_indexes", "packed_key_value_indexes"):
                gen_input_lat.pop(_drop, None)
            latents, *_ = self.bagel.generate_image(
                past_key_values=gen_context["past_key_values"],
                cfg_text_past_key_values=cfg_text_context["past_key_values"]
                if cfg_text_scale > 1.0
                else gen_context["past_key_values"],
                cfg_img_past_key_values=gen_context["past_key_values"],
                num_timesteps=num_timesteps,
                timestep_shift=timestep_shift,
                cfg_text_scale=cfg_text_scale,
                cfg_img_scale=cfg_img_scale,
                cfg_interval=cfg_interval,
                cfg_renorm_type=cfg_renorm_type,
                cfg_renorm_min=cfg_renorm_min,
                **gen_input_lat,
                cfg_text_packed_position_ids=cfg_text_lat["cfg_packed_position_ids"],
                # ``cfg_text_packed_query_indexes`` / ``cfg_text_key_values_lens``
                # / ``cfg_text_packed_key_value_indexes`` removed post-main-merge
                # (derived from ``cfg_text_past_key_values``).
                cfg_img_packed_position_ids=cfg_img_lat["cfg_packed_position_ids"],
                # ``cfg_img_*`` index/lens kwargs removed — same as above.
                frame_condition_token_indexes=frame_condition_token_indexes,
            )

        frames_np = self._decode_video_from_latent(self.bagel, self.vae, latents[0], out_shape)
        frames = [Image.fromarray(f) for f in frames_np]
        return DiffusionOutput(
            output=frames[0],
            custom_output={"video_frames": frames, "video_shape": out_shape},
            stage_durations=self.stage_durations if hasattr(self, "stage_durations") else None,
        )

    def _forward_video_edit(self, req):
        """Video edit: reference video + text prompt → modified video.

        Mirrors :meth:`_forward_image_edit`'s segmented prefill layout with
        a temporal axis threaded through both prefill and the gen latent.
        Verified against upstream Lance's ``Lance.validation_gen_KVcache`` for
        ``--task video_edit --resolution video_480p``:

          seg1 (causal, modality=system_prompt):
              ``<|im_start|>system\\n{sys}<|im_end|>\\n<|im_start|>user\\n``
          seg2 (full,   modality=ref_vit):
              ViT(video) — ref video through Qwen2.5-VL ViT, t-axis +1000
          seg3 (full_noise, modality=ref_source):
              VAE(video) — ref video through Wan2.2 VAE, time_embed(t=0).
              Rope shares range with the gen noise latent (modality 1==2
              trick) so the model can align ref↔output per ``(t,h,w)``.
          seg4 (causal, modality=text):  the user instruction
          seg5 (causal, modality=system_prompt):
              ``<|im_end|>\\n<|im_start|>assistant\\n``
          seg6 (noise QUERY): 3-D gen latent at the SAME rope as seg3.

        cfg_text_context skips ONLY seg4 (user text), matching upstream's
        ``uncond_split_pro_kvcache``.  cfg_img defaults to off
        (``cfg_img_scale=1.0`` → upstream's ``cfg_vit_scale=1.0``).

        Replaces the previous shifted-rope layout (``_EDIT_SHIFT=10000``)
        which kept ViT/text at a far-away rope range — that caused 15.5%
        rel_l2 divergence at layer 0 output (cond) vs upstream because the
        noise query attended to a cached K/V at rope (0..47) while upstream
        attends at rope (111..159).
        """

        import numpy as _np

        from vllm_omni.diffusion.data import DiffusionOutput

        from .lance_transformer import NaiveCache

        first_prompt = req.prompts[0]
        assert isinstance(first_prompt, dict), "video_edit requires dict-style prompt"
        user_text = first_prompt.get("user_text")
        rendered = first_prompt.get("prompt") or ""
        if not user_text:
            user_text = _extract_user_instruction(rendered)
        mm_data = first_prompt.get("multi_modal_data") or {}
        video_input = mm_data.get("video")
        if video_input is None:
            raise ValueError("video_edit requires multi_modal_data.video.")
        # Resolve to raw (T, H, W, 3) uint8 ndarray + origin_fps so the
        # upstream-style bucket resize + frame sampler can replicate the
        # exact preprocessing path.  If the caller passed an already-decoded
        # ndarray with no FPS, default to 24fps (matches the official
        # video_edit examples).
        extra_args = first_prompt.get("extra_args") or {}
        sp_extra = getattr(req.sampling_params, "extra_args", {}) or {}
        extra_args = {**extra_args, **sp_extra}
        origin_fps_default = float(extra_args.get("origin_fps", 24.0))
        # Path-only fast path: use decord (matches upstream Lance's
        # ``data/datasets_custom/validation_dataset.py`` which decodes via
        # ``decord.VideoReader``).  decord vs cv2 produce different YUV→RGB
        # conversions (~0.1 RGB units of max diff), and that propagates into
        # both the VAE and ViT pixel inputs.  Falling back to cv2 only when
        # decord is unavailable keeps single-machine parity with upstream
        # while not requiring decord in production.
        if isinstance(video_input, str):
            try:
                import decord as _decord
                from decord import VideoReader as _VideoReader  # type: ignore

                vr = _VideoReader(video_input, ctx=_decord.cpu(0))
                origin_fps = float(vr.get_avg_fps()) or origin_fps_default
                idx_all = list(range(len(vr)))
                video_raw = vr.get_batch(idx_all).asnumpy()  # (T, H, W, 3) uint8 RGB
            except ImportError:
                import cv2 as _cv2

                cap = _cv2.VideoCapture(video_input)
                origin_fps = float(cap.get(_cv2.CAP_PROP_FPS) or origin_fps_default)
                frames_bgr = []
                while True:
                    ok, f = cap.read()
                    if not ok:
                        break
                    frames_bgr.append(_cv2.cvtColor(f, _cv2.COLOR_BGR2RGB))
                cap.release()
                video_raw = _np.stack(frames_bgr, axis=0)
        elif isinstance(video_input, _np.ndarray):
            video_raw = video_input
            origin_fps = origin_fps_default
        elif torch.is_tensor(video_input):
            arr = video_input.detach().cpu().numpy()
            if arr.max() <= 1.5:
                arr = (arr * 255.0).clip(0, 255).astype(_np.uint8)
            video_raw = arr
            origin_fps = origin_fps_default
        else:
            raise ValueError(f"Unsupported video_input type {type(video_input)}")

        cfg_text_scale = float(extra_args.get("cfg_text_scale", 4.0))
        cfg_img_scale = float(extra_args.get("cfg_img_scale", 1.0))
        # Upstream Lance default is ``cfg_interval=[0, 1]`` (CFG active on every
        # denoise step).  vllm-omni's previous (0.4, 1.0) skipped CFG for the
        # last ~10 iterations after timestep_shift=3.5, which contributed
        # most of the late-trajectory drift from upstream.
        cfg_interval = tuple(extra_args.get("cfg_interval", (0.0, 1.0)))
        cfg_renorm_type = str(extra_args.get("cfg_renorm_type", "global"))
        cfg_renorm_min = float(extra_args.get("cfg_renorm_min", 0.0))
        timestep_shift = float(extra_args.get("timestep_shift", LANCE_DEFAULTS.timestep_shift))
        num_timesteps = int(req.sampling_params.num_inference_steps or LANCE_DEFAULTS.num_timesteps)

        # Run upstream's BucketResize + frame sampler so VAE / ViT see
        # byte-identical pixels to the reference implementation.  Without this,
        # vllm-omni's ViT grid was (T=23, H=40, W=54) for the car video versus
        # upstream's (T=23, H=36, W=50) — different K/V at ViT positions
        # accounted for the residual ~9% rel_l2 at layer 0 output.
        from .lance_transformer import LanceBagel as _LanceBagel

        vae_video, vit_pixels, vit_grid_thw, sampled_T = _LanceBagel._lance_video_preprocess(video_raw, origin_fps)
        video_chw = vae_video  # (C, T_sampled, H_vae, W_vae) — model-space [-1,1]
        T_sampled = int(video_chw.shape[1])
        H_vae = int(video_chw.shape[2])
        W_vae = int(video_chw.shape[3])
        video_shape = (T_sampled, H_vae, W_vae)
        logger.info(
            "Lance video_edit: input %dx%dx%d → sampled %dx%dx%d (T,H,W) @ %.2ffps, ViT grid_thw=%s, user_text=%r",
            video_raw.shape[0],
            video_raw.shape[1],
            video_raw.shape[2],
            T_sampled,
            H_vae,
            W_vae,
            origin_fps,
            tuple(vit_grid_thw[0].tolist()),
            user_text,
        )

        if req.sampling_params.seed is not None:
            torch.manual_seed(req.sampling_params.seed)
            if self.device.type == "cuda":
                torch.cuda.manual_seed(req.sampling_params.seed)

        gen_context = {
            "kv_lens": [0],
            "ropes": [0],
            "past_key_values": NaiveCache(self.bagel.config.llm_config.num_hidden_layers),
        }
        cfg_text_context = {
            "kv_lens": [0],
            "ropes": [0],
            "past_key_values": NaiveCache(self.bagel.config.llm_config.num_hidden_layers),
        }

        # Shared latent cache so gen + cfg branches see the same VAE sample.
        ref_latent_cache: list = []

        def _vae_transforms(_):
            return video_chw

        # ----- prefill segments in upstream order -----
        seg1_str, seg5_str = self._segment_strings("video_edit", "video")

        # seg1: system + user header  (shared)
        self._raw_text_prefill(gen_context, seg1_str)
        if cfg_text_scale > 1.0:
            self._raw_text_prefill(cfg_text_context, seg1_str)

        # seg2: ViT(ref)  (shared) — feed pre-bucketed patches directly so
        # the Qwen2VLImageProcessor smart-resize is bypassed.
        precomputed_vit = [(vit_pixels, vit_grid_thw)]
        self._vit_video_prefill(gen_context, [video_raw], precomputed_vit=precomputed_vit)
        if cfg_text_scale > 1.0:
            self._vit_video_prefill(cfg_text_context, [video_raw], precomputed_vit=precomputed_vit)

        # seg3: VAE(ref) — snapshot rope BEFORE the prefill so the gen noise
        # latent can be placed at the SAME rope range (matches upstream's
        # modality==1≡2 trick).
        rope_before_vae = gen_context["ropes"][0]
        cfg_rope_before_vae = cfg_text_context["ropes"][0] if cfg_text_scale > 1.0 else rope_before_vae
        self._vae_ref_prefill(gen_context, [video_chw], _vae_transforms, is_video=True, latent_cache=ref_latent_cache)
        if cfg_text_scale > 1.0:
            self._vae_ref_prefill(
                cfg_text_context, [video_chw], _vae_transforms, is_video=True, latent_cache=ref_latent_cache
            )

        # seg4: user instruction  (gen only — cfg branch skips per upstream).
        self._raw_text_prefill(gen_context, user_text)

        # seg5: separator + assistant header  (both gen and cfg, each at its
        # own current rope; cfg's lands earlier than gen's because cfg
        # skipped seg4).
        self._raw_text_prefill(gen_context, seg5_str)
        if cfg_text_scale > 1.0:
            self._raw_text_prefill(cfg_text_context, seg5_str)

        # seg6 (noise QUERY): gen latent at the SAME rope as the ref VAE
        # block.  KV cache still contains sys+ViT+VAE_ref+user_text+sep at
        # their post-prefill rope positions; only the noise query's Q rope
        # is rewound to overlap with the VAE_ref block.
        gen_input_lat = self.bagel.prepare_video_latent(
            curr_kvlens=gen_context["kv_lens"],
            curr_rope=[rope_before_vae],
            video_shapes=[video_shape],
            new_token_ids=self.new_token_ids,
        )
        for k, v in gen_input_lat.items():
            if torch.is_tensor(v):
                gen_input_lat[k] = v.to(self.device)
        cfg_text_lat = self.bagel.prepare_video_latent_cfg(
            curr_kvlens=cfg_text_context["kv_lens"],
            curr_rope=[cfg_rope_before_vae],
            video_shapes=[video_shape],
        )
        for k, v in cfg_text_lat.items():
            if torch.is_tensor(v):
                cfg_text_lat[k] = v.to(self.device)

        # cfg_img branch is off (cfg_img_scale=1.0); generate_image needs the
        # arg surface but skips the branch when scale<=1.0.
        cfg_img_lat = cfg_text_lat

        self._regen_init_noise_on_device(gen_input_lat, req.sampling_params.seed)

        with torch.autocast(**self._autocast_kwargs()):
            # ``prepare_video_latent`` / ``prepare_vae_latent`` still return
            # ``key_values_lens`` / ``packed_indexes`` / ``packed_key_value_indexes``
            # but post-main-merge ``generate_image`` doesn't accept them — drop
            # before unpacking via ``**gen_input_lat``.
            for _drop in ("key_values_lens", "packed_indexes", "packed_key_value_indexes"):
                gen_input_lat.pop(_drop, None)
            latents, *_ = self.bagel.generate_image(
                past_key_values=gen_context["past_key_values"],
                cfg_text_past_key_values=cfg_text_context["past_key_values"]
                if cfg_text_scale > 1.0
                else gen_context["past_key_values"],
                cfg_img_past_key_values=gen_context["past_key_values"],
                num_timesteps=num_timesteps,
                timestep_shift=timestep_shift,
                cfg_text_scale=cfg_text_scale,
                cfg_img_scale=cfg_img_scale,
                cfg_interval=cfg_interval,
                cfg_renorm_type=cfg_renorm_type,
                cfg_renorm_min=cfg_renorm_min,
                **gen_input_lat,
                cfg_text_packed_position_ids=cfg_text_lat["cfg_packed_position_ids"],
                # ``cfg_text_packed_query_indexes`` / ``cfg_text_key_values_lens``
                # / ``cfg_text_packed_key_value_indexes`` removed post-main-merge
                # (derived from ``cfg_text_past_key_values``).
                cfg_img_packed_position_ids=cfg_img_lat["cfg_packed_position_ids"],
                # ``cfg_img_*`` index/lens kwargs removed — same as above.
            )

        frames_np = self._decode_video_from_latent(self.bagel, self.vae, latents[0], video_shape)
        frames = [Image.fromarray(f) for f in frames_np]
        return DiffusionOutput(
            output=frames[0],
            custom_output={"video_frames": frames, "video_shape": video_shape},
            stage_durations=self.stage_durations if hasattr(self, "stage_durations") else None,
        )

    @torch.inference_mode()
    def _forward_x2t_video(self, req):
        """Video understanding (x2t_video): video → text caption / VQA answer.

        Builds the prefill in three segments matching upstream's chat
        template — see :meth:`_forward_x2t_image` for the rationale on
        why naïve concatenation (text prompt → ViT) lands the EOS / vit
        markers in the wrong place and degrades the answer format.
        """

        from vllm_omni.diffusion.data import DiffusionOutput

        from .lance_transformer import NaiveCache

        first_prompt = req.prompts[0]
        assert isinstance(first_prompt, dict), "x2t_video requires dict-style prompt"
        video_input = (first_prompt.get("multi_modal_data") or {}).get("video")
        if video_input is None:
            raise ValueError("x2t_video requires multi_modal_data.video to be a video tensor/array/path.")
        # Accept a path → load via imageio; else assume tensor / numpy array (T, H, W, 3).
        if isinstance(video_input, str):
            import imageio.v3 as iio

            video_input = iio.imread(video_input)
        # Wrap single video as a list for the prep function.
        videos = [video_input]

        extra_args = first_prompt.get("extra_args") or {}
        sp_extra = getattr(req.sampling_params, "extra_args", {}) or {}
        extra_args = {**extra_args, **sp_extra}
        max_text_tokens = int(extra_args.get("max_think_tokens", 200))
        do_sample = bool(extra_args.get("do_sample", False))
        text_temperature = float(extra_args.get("text_temperature", 0.3))

        prefix_text, suffix_text = self._split_x2t_prompt(
            first_prompt.get("prompt") or "",
            user_text_override=extra_args.get("user_text"),
            system_prompt_override=extra_args.get("system_prompt"),
            default_system_prompt="Watch the video carefully and answer the question.",
        )

        if req.sampling_params.seed is not None:
            torch.manual_seed(req.sampling_params.seed)
            if self.device.type == "cuda":
                torch.cuda.manual_seed(req.sampling_params.seed)

        gen_context = {
            "kv_lens": [0],
            "ropes": [0],
            "past_key_values": NaiveCache(self.bagel.config.llm_config.num_hidden_layers),
        }

        self._raw_text_prefill(gen_context, prefix_text)
        self._vit_video_prefill(gen_context, videos)
        self._raw_text_prefill(gen_context, suffix_text)

        start_input = self._x2t_prepare_assistant_start(gen_context)
        for k, v in start_input.items():
            if torch.is_tensor(v):
                start_input[k] = v.to(self.device)
        with torch.autocast(**self._autocast_kwargs()):
            token_ids = self.bagel.generate_text(
                past_key_values=gen_context["past_key_values"],
                max_length=max_text_tokens,
                do_sample=do_sample,
                temperature=text_temperature,
                end_token_id=self.new_token_ids["eos_token_id"],
                **start_input,
            )
        decoded = self.tokenizer.decode(token_ids[:, 0].tolist())
        text_output = decoded.split("<|im_end|>")[0]
        if "<|im_start|>" in text_output:
            text_output = text_output.split("<|im_start|>")[-1]
        # Strip the leading newline that was fed as the start token (see
        # ``_x2t_prepare_assistant_start``); see x2t_image for rationale.
        text_output = text_output.lstrip("\n")
        logger.info("Lance x2t_video: generated %d tokens", token_ids.shape[0])
        return DiffusionOutput(
            output=text_output,
            custom_output={"text_output": text_output},
            stage_durations=self.stage_durations if hasattr(self, "stage_durations") else None,
        )

    @torch.inference_mode()
    def _forward_x2t_image(self, req):
        """Image understanding (x2t_image): image + question → text answer.

        Builds the prefill in three text/vit segments matching upstream's
        chat-template layout::

            <|im_start|>system\\n{system}<|im_end|>\\n<|im_start|>user\\n
            <|vision_start|>{N×ViT}<|vision_end|>
            {question}<|im_end|>\\n<|im_start|>assistant\\n

        ``prepare_prompts`` wraps the *whole* string with ``[bos]…[eos]``
        and ``prepare_vit_images`` adds ``<|vision_start|>``/``<|vision_end|>``
        of its own — concatenating them naïvely (text first, then vit)
        produced ``…assistant\\n[eos]<|vision_start|>{ViT}<|vision_end|>``,
        so the model saw an EOS right after the assistant tag and a
        bare ``<|video_pad|>`` token inside the user message.  Segmenting
        the text into prefix/suffix and sliding the vit prefill between
        them puts the markers exactly where Lance was trained on.
        """

        from PIL import Image as _PIL_Image

        from vllm_omni.diffusion.data import DiffusionOutput

        from .lance_transformer import NaiveCache

        first_prompt = req.prompts[0]
        assert isinstance(first_prompt, dict), "x2t_image requires dict-style prompt"
        image_input = (first_prompt.get("multi_modal_data") or {}).get("image")
        if image_input is None:
            raise ValueError("x2t_image requires multi_modal_data.image to be a PIL image / array / path.")
        if isinstance(image_input, str):
            image_input = _PIL_Image.open(image_input).convert("RGB")
        images = [image_input]

        extra_args = first_prompt.get("extra_args") or {}
        sp_extra = getattr(req.sampling_params, "extra_args", {}) or {}
        extra_args = {**extra_args, **sp_extra}
        max_text_tokens = int(extra_args.get("max_think_tokens", 256))
        do_sample = bool(extra_args.get("do_sample", False))
        text_temperature = float(extra_args.get("text_temperature", 0.3))

        prefix_text, suffix_text = self._split_x2t_prompt(
            first_prompt.get("prompt") or "",
            user_text_override=extra_args.get("user_text"),
            system_prompt_override=extra_args.get("system_prompt"),
            default_system_prompt="Look at the image carefully and answer the question.",
        )

        if req.sampling_params.seed is not None:
            torch.manual_seed(req.sampling_params.seed)
            if self.device.type == "cuda":
                torch.cuda.manual_seed(req.sampling_params.seed)

        def vit_transforms(img):
            return self.image_processor(images=img, return_tensors="pt").pixel_values[0]

        gen_context = {
            "kv_lens": [0],
            "ropes": [0],
            "past_key_values": NaiveCache(self.bagel.config.llm_config.num_hidden_layers),
        }

        self._raw_text_prefill(gen_context, prefix_text)
        self._vit_image_prefill(gen_context, images, vit_transforms)
        self._raw_text_prefill(gen_context, suffix_text)

        start_input = self._x2t_prepare_assistant_start(gen_context)
        for k, v in start_input.items():
            if torch.is_tensor(v):
                start_input[k] = v.to(self.device)
        with torch.autocast(**self._autocast_kwargs()):
            token_ids = self.bagel.generate_text(
                past_key_values=gen_context["past_key_values"],
                max_length=max_text_tokens,
                do_sample=do_sample,
                temperature=text_temperature,
                end_token_id=self.new_token_ids["eos_token_id"],
                **start_input,
            )
        decoded = self.tokenizer.decode(token_ids[:, 0].tolist())
        text_output = decoded.split("<|im_end|>")[0]
        if "<|im_start|>" in text_output:
            text_output = text_output.split("<|im_start|>")[-1]
        # Strip the leading newline that was fed as the start token (see
        # ``_x2t_prepare_assistant_start``); it always sits at index 0 of
        # the generated sequence and is just plumbing, not model output.
        text_output = text_output.lstrip("\n")
        logger.info("Lance x2t_image: generated %d tokens", token_ids.shape[0])
        return DiffusionOutput(
            output=text_output,
            custom_output={"text_output": text_output},
            stage_durations=self.stage_durations if hasattr(self, "stage_durations") else None,
        )

    @staticmethod
    def _split_x2t_prompt(
        rendered_prompt: str,
        *,
        user_text_override: str | None,
        system_prompt_override: str | None,
        default_system_prompt: str,
    ) -> tuple[str, str]:
        """Split a rendered Lance x2t chat template at the vision block.

        Accepts either a pre-rendered template containing
        ``<|vision_start|>…<|vision_end|>`` (the historical
        ``render_lance_prompt`` output) or recovers prefix/suffix from
        ``user_text_override`` + ``system_prompt_override`` when the
        template was not pre-rendered.  Returns ``(prefix, suffix)`` such
        that the segmented prefill is ``prefix + ViT + suffix``.

        ``suffix`` deliberately ends at ``assistant`` (NOT ``assistant\\n``):
        :meth:`Bagel.prepare_start_tokens` injects ``<|im_start|>`` (Qwen
        bos == 151644) as the first generation step, and ``\\n<|im_start|>``
        right after ``assistant\\n`` tells the model "begin a new
        message" — it answers with an immediate ``<|im_end|>`` and
        produces an empty caption.  Holding the trailing ``\\n`` out of
        the prefill (it gets re-applied via ``_x2t_prepare_assistant_start``)
        lets the next forward step start *inside* the assistant turn
        instead of opening a fresh one.
        """
        sys_prompt = system_prompt_override or default_system_prompt

        if user_text_override is not None:
            user_text = user_text_override
        elif "<|vision_start|>" in rendered_prompt and "<|vision_end|>" in rendered_prompt:
            _, _, after_vis_end = rendered_prompt.partition("<|vision_end|>")
            user_text, _, _ = after_vis_end.partition("<|im_end|>")
        else:
            # No vision markers and no override — best-effort fallback:
            # treat the entire prompt as the user question and rebuild
            # the chat scaffolding.
            user_text = rendered_prompt

        prefix = f"<|im_start|>system\n{sys_prompt}<|im_end|>\n<|im_start|>user\n"
        suffix = f"{user_text}<|im_end|>\n<|im_start|>assistant"
        return prefix, suffix

    def _x2t_prepare_assistant_start(self, gen_context):
        """Build a ``generate_text`` start input where the first fed token
        is the newline that closes ``<|im_start|>assistant\\n``.

        :meth:`Bagel.prepare_start_tokens` would instead feed
        ``<|im_start|>``, opening a fresh message right after the
        assistant tag and triggering an immediate EOS on Lance_3B (the
        image-only checkpoint is stricter about chat-turn framing than
        the video model, which happens to recover).  Feeding ``\\n``
        keeps the model *inside* the assistant turn so the next argmax
        is the first answer token.
        """
        newline_ids = self.tokenizer.encode("\n", add_special_tokens=False)
        assert len(newline_ids) == 1, f"expected single token for '\\n', got {newline_ids!r}"
        nl_id = newline_ids[0]

        curr_kvlens = gen_context["kv_lens"]
        curr_ropes = gen_context["ropes"]
        packed_key_value_indexes: list[int] = []
        packed_query_position_ids: list[int] = []
        packed_start_tokens: list[int] = []
        curr = 0
        for curr_kvlen, curr_position_id in zip(curr_kvlens, curr_ropes):
            packed_key_value_indexes.extend(range(curr, curr + curr_kvlen))
            packed_start_tokens.append(nl_id)
            packed_query_position_ids.append(curr_position_id)
            curr += curr_kvlen
        # ``Bagel.generate_text`` post-main-merge no longer accepts
        # ``key_values_lens`` / ``packed_key_value_indexes`` — they're
        # derived from ``past_key_values``.
        return {
            "packed_start_tokens": torch.tensor(packed_start_tokens, dtype=torch.long),
            "packed_query_position_ids": torch.tensor(packed_query_position_ids, dtype=torch.long),
        }

    @staticmethod
    def _decode_video_from_latent(
        bagel,
        vae: LanceWanVAE,
        latent: torch.Tensor,
        video_shape: tuple[int, int, int],
    ):
        """Pack a flat Lance latent into ``(B, 48, t, h, w)`` and decode to a
        ``(T_lat, H, W, 3)`` numpy frame list via :meth:`LanceWanVAE.decode_video`.
        """

        T, H, W = video_shape
        downsample_t = int(getattr(bagel.config.vae_config, "downsample_temporal", 4))
        t_lat = (T - 1) // downsample_t + 1
        h_lat = H // bagel.latent_downsample
        w_lat = W // bagel.latent_downsample
        p = bagel.latent_patch_size
        c = bagel.latent_channel
        # ``latent`` is flat ``(t_lat*h_lat*w_lat, c * p * p)`` (Lance image
        # path uses ``p=1`` so this is ``(N, c)``).  Reshape into the 5-D layout
        # the Wan2.2 VAE decoder consumes.
        latent = latent.reshape(t_lat, h_lat, w_lat, p, p, c)
        latent = torch.einsum("thwpqc->cthpwq", latent)
        latent = latent.reshape(1, c, t_lat, h_lat * p, w_lat * p)

        vae_dtype = next(vae.parameters()).dtype
        latent = latent.to(vae_dtype)
        video = vae.decode_video(latent)  # (1, 3, T_out, H_out, W_out)
        # Scale [-1,1] -> [0,255] uint8 frames; the upstream VAE already clamps.
        video = (video[0] * 0.5 + 0.5).clamp(0, 1).permute(1, 2, 3, 0) * 255
        return video.to(torch.uint8).cpu().numpy()  # (T_out, H_out, W_out, 3)

    def _build_wan22_vae(self, repo_root: str) -> LanceWanVAE:
        """Wrap the bundled ``Wan2.2_VAE.pth`` with a BAGEL-compatible
        ``.encode(images)`` / ``.decode(latent)`` surface.

        Uses the upstream Wan2.2 VAE module (ported in :mod:`wan_vae`) since
        the released ``.pth`` is keyed for that module — *not* for the diffusers
        ``AutoencoderKLWan`` state dict that ``OmniAutoencoderKLWan`` expects.
        Single images are treated as 1-frame clips; video latents pass through
        the same module via the 5-D ``encode_video``/``decode_video`` path.
        Wan2.2 VAE constants: z=48ch, /16 spatial, /4 temporal.
        """
        vae_path = os.path.join(repo_root, _VAE_FILE)
        return LanceWanVAE(vae_path=vae_path, dtype=torch.bfloat16, device=self.device)
