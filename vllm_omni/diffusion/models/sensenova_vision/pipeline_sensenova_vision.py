# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""SenseNova-Vision-7B-MoT diffusion pipeline.

SenseNova-Vision is a Bagel fork; the denoising model, VAE, and ViT are
weight-compatible with the BAGEL integration.  This pipeline subclasses
:class:`BagelPipeline` and overrides only the SenseNovaVision checkpoint defaults:

- ``max_latent_size=64`` (BAGEL ships 32)
- ``vit_max_num_patch_per_side=70``
- VAE image transform ``ImageTransform(1024, 512, 16)``
- ViT image transform ``ImageTransform(980, 224, 14)``
- per-mode ``BASE_PARAMS`` from
  ``SenseNova-Vision/inference/sensenova_vision.py``
"""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Any

import numpy as np
import PIL.Image
import torch
from vllm.logger import init_logger

from vllm_omni.diffusion.data import DiffusionOutput, OmniDiffusionConfig
from vllm_omni.diffusion.models.bagel.bagel_transformer import NaiveCache
from vllm_omni.diffusion.models.bagel.pipeline_bagel import BagelGenParams, BagelPipeline
from vllm_omni.diffusion.models.sensenova_vision.single_stage import SenseNovaVisionSingleStageMixin
from vllm_omni.diffusion.models.sensenova_vision.tokenization_sensenova_vision import (
    VLLMSenseNovaVisionTokenizer,
)
from vllm_omni.diffusion.models.sensenova_vision.transforms_sensenova_vision import (
    PER_TASK_VAE_SIDE,
    recon3d_packing,
)
from vllm_omni.diffusion.worker.request_batch import DiffusionRequestBatch
from vllm_omni.model_executor.model_loader.weight_utils import download_weights_from_hf_specific

# BAGEL-compatible image-processor recipe lives in the model-executor
# SenseNova-Vision layer as the single canonical copy shared by the AR stage
# (vllm_omni/engine/arg_utils.py) and this DiT stage so the two stages stay
# in lockstep.
from vllm_omni.model_executor.models.sensenova_vision.cfg_expand import IMG2IMG_PLACEHOLDER
from vllm_omni.model_executor.models.sensenova_vision.configuration_sensenova_vision import (
    SENSENOVA_VISION_PREPROCESSOR_CONFIG as _SENSENOVA_VISION_PREPROCESSOR_CONFIG,
)

if TYPE_CHECKING:
    from vllm_omni.diffusion.models.bagel.autoencoder import AutoEncoder
    from vllm_omni.diffusion.models.bagel.bagel_transformer import Bagel
    from vllm_omni.inputs.data import OmniDiffusionSamplingParams

logger = init_logger(__name__)


@dataclass
class SenseNovaVisionGenParams:
    """SenseNovaVision per-mode generation parameters (BASE_PARAMS equivalents)."""

    num_timesteps: int = 50
    timestep_shift: float = 3.0
    cfg_text_scale: float = 4.0
    cfg_img_scale: float = 1.0
    cfg_interval: tuple = (0.4, 1.0)
    cfg_renorm_min: float = 1.0
    cfg_renorm_type: str = "global"
    # SenseNovaVision-specific additive flags (not consumed by the BAGEL core).
    think: bool = False
    understanding_output: bool = False
    max_think_token_n: int = 1000
    do_sample: bool = False
    text_temperature: float = 0.3
    extra: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_base_params(cls, mode: str) -> SenseNovaVisionGenParams:
        """Build defaults for a SenseNovaVision inference mode.

        Mirrors ``BASE_PARAMS`` in
        ``SenseNova-Vision/inference/sensenova_vision.py``.
        """
        base = dict(_BASE_PARAMS.get(mode, {}))
        cfg_interval = base.pop("cfg_interval", (0.4, 1.0))
        if not isinstance(cfg_interval, (tuple, list)) or len(cfg_interval) != 2:
            cfg_interval = (0.4, 1.0)
        cfg_interval = (float(cfg_interval[0]), float(cfg_interval[1]))
        return cls(
            num_timesteps=int(base.pop("num_timesteps", 50)),
            timestep_shift=float(base.pop("timestep_shift", 3.0)),
            cfg_text_scale=float(base.pop("cfg_text_scale", 4.0)),
            cfg_img_scale=float(base.pop("cfg_img_scale", 1.0)),
            cfg_interval=cfg_interval,
            cfg_renorm_min=float(base.pop("cfg_renorm_min", 1.0)),
            cfg_renorm_type=str(base.pop("cfg_renorm_type", "global")),
            think=bool(base.pop("think", False)),
            understanding_output=bool(base.pop("understanding_output", False)),
            max_think_token_n=int(base.pop("max_think_token_n", 1000)),
            do_sample=bool(base.pop("do_sample", False)),
            text_temperature=float(base.pop("text_temperature", 0.3)),
            extra=base,
        )


# Per-mode defaults, transcribed verbatim from
# SenseNova-Vision/inference/sensenova_vision.py (BASE_PARAMS).
_BASE_PARAMS: dict[str, dict[str, Any]] = {
    "generate": {
        "cfg_text_scale": 4.0,
        "cfg_img_scale": 1.0,
        "cfg_interval": [0.4, 1.0],
        "timestep_shift": 3.0,
        "num_timesteps": 50,
        "cfg_renorm_min": 1.0,
        "cfg_renorm_type": "global",
    },
    "think_generate": {
        "max_think_token_n": 1000,
        "do_sample": False,
        "cfg_text_scale": 4.0,
        "cfg_img_scale": 1.0,
        "cfg_interval": [0.4, 1.0],
        "timestep_shift": 3.0,
        "num_timesteps": 50,
        "cfg_renorm_min": 1.0,
        "cfg_renorm_type": "global",
        "think": True,
    },
    "caption_generate": {
        "max_think_token_n": 8192,
        "do_sample": False,
        "cfg_text_scale": 4.0,
        "cfg_img_scale": 1.0,
        "cfg_interval": [0.0, 1.0],
        "timestep_shift": 4.0,
        "num_timesteps": 50,
        "cfg_renorm_min": 1.0,
        "cfg_renorm_type": "global",
        "think": True,  # `caption` serve same purpose as `think` flag in og repo.
        # "caption": True,
    },
    "dense_perception": {
        "cfg_text_scale": 4.0,
        "cfg_img_scale": 1.0,
        "cfg_interval": [0.0, 1.0],
        "timestep_shift": 4.0,
        "num_timesteps": 50,
        "cfg_renorm_min": 1.0,
        "cfg_renorm_type": "text_channel",
    },
    "edit": {
        "cfg_text_scale": 4.0,
        "cfg_img_scale": 2.0,
        "cfg_interval": [0.0, 1.0],
        "timestep_shift": 4.0,
        "num_timesteps": 50,
        "cfg_renorm_min": 1.0,
        "cfg_renorm_type": "text_channel",
    },
    "think_edit": {
        "max_think_token_n": 1000,
        "do_sample": False,
        "cfg_text_scale": 4.0,
        "cfg_img_scale": 2.0,
        "cfg_interval": [0.4, 1.0],
        "timestep_shift": 3.0,
        "num_timesteps": 50,
        "cfg_renorm_min": 0.0,
        "cfg_renorm_type": "text_channel",
        "think": True,
    },
    "understanding": {
        "max_think_token_n": 8192,
        "do_sample": False,
        "understanding_output": True,
    },
    "think_understanding": {
        "max_think_token_n": 8192,
        "do_sample": False,
        "understanding_output": True,
        "think": True,
    },
    "dense_detection": {
        "max_think_token_n": 8192,
        "do_sample": False,
        "understanding_output": True,
    },
    "dense_OCR": {
        "max_think_token_n": 20000,
        "do_sample": False,
        "understanding_output": True,
    },
    "recon3d": {
        "cfg_text_scale": 1.0,
        "cfg_img_scale": 1.0,
        "cfg_interval": [0.0, 1.0],
        "timestep_shift": 4.0,
        "num_timesteps": 50,
        "cfg_renorm_min": 1.0,
        "cfg_renorm_type": "text_channel",
    },
}


def get_sensenova_vision_post_process_func(od_config: OmniDiffusionConfig):
    """SenseNovaVision post-processing: pipelines return PIL images directly."""
    del od_config  # unused

    def post_process_func(x):
        return x

    return post_process_func


def build_sensenova_vision_diffusion_output(
    *,
    text: str | None = None,
    image: Any = None,
    think_text: str | None = None,
    stage_durations: dict[str, float] | None = None,
) -> DiffusionOutput:
    """Build a canonical SenseNovaVision ``DiffusionOutput`` envelope.

    Mirrors the envelope contract used by :class:`BagelPipeline`:

    * text-only   -> ``{"payload": {"text": ...}, "metadata": {"text": {"text_output": ...}}}``
    * image-only  -> ``{"payload": {"image": ...}, "metadata": {...}}``
    * mixed       -> ``{"payload": {"text": ..., "image": ...}, "metadata": {...}}``

    ``text``/``image`` are raw producer values (a decoded ``str`` caption and a
    ``PIL.Image`` respectively). ``think_text`` is any reasoning/caption string
    produced alongside the image (e.g. ``think_generate``) and is recorded under
    the shared ``text`` metadata group. This is deliberately generic: it carries
    only the existing ``TEXT``/``IMAGE`` output-modality contract and never
    introduces a SenseNovaVision-specific modality key.
    """
    payload: dict[str, Any] = {}
    metadata: dict[str, Any] = {}
    text_meta: dict[str, str] = {}

    if text is not None:
        payload["text"] = text
        text_meta["text_output"] = text
    if think_text is not None:
        text_meta["think_text"] = think_text
    if text_meta:
        metadata["text"] = text_meta
    if image is not None:
        payload["image"] = image

    return DiffusionOutput(
        output={
            "payload": payload,
            "metadata": metadata,
        },
        stage_durations=stage_durations,
    )


class SenseNovaVisionPipeline(SenseNovaVisionSingleStageMixin, BagelPipeline):
    """SenseNova-Vision-7B-MoT diffusion pipeline.

    Subclasses :class:`BagelPipeline` and reuses the entire BAGEL weight
    loading / denoising machinery.  Only the SenseNovaVision checkpoint defaults
    differ; these are applied in :meth:`__init__` and per-request mode
    defaults are applied in :meth:`forward` via ``extra_args``.

    The checkpoint does not ship a ``preprocessor_config.json`` (BAGEL does),
    which the BAGEL core needs for ``SiglipImageProcessor``.  When it is
    missing, :meth:`__init__` patches a temp directory with a generated
    BAGEL-compatible ``preprocessor_config.json`` plus symlinks to every
    checkpoint file and points a copy of ``od_config`` at it.

     :meth:`__init__` constructs
    :class:`~vllm_omni.diffusion.models.sensenova_vision.tokenization_sensenova_vision.VLLMSenseNovaVisionTokenizer`
    in-process from the checkpoint path and passes it to the BAGEL core via
    the ``tokenizer`` kwarg.  This is required because the BAGEL core
    otherwise loads the tokenizer with
    ``AutoTokenizer.from_pretrained(..., trust_remote_code=True)``, and the
    SenseNova-Vision checkpoint's stock tokenizer would renumber its 2033
    added tokens (including the four control tokens BAGEL uses) past the
    LLM's 152064 embedding rows — see :mod:`tokenization_sensenova_vision`
    for the full explanation.  Loading the class directly keeps those ids
    verbatim with **zero writes** to the checkpoint directory (no
    ``tokenizer_config.json`` rewrite, no copied source file, no temp dir
    for the tokenizer).
    """

    # SenseNovaVision checkpoint overrides applied on top of BAGEL.
    _sensenova_vision_max_latent_size = 64
    _sensenova_vision_vit_max_num_patch_per_side = 70
    # Official SenseNovaVision image transforms (vae, vit).
    _sensenova_vision_vae_transform = (1024, 512, 16)
    _sensenova_vision_vit_transform = (980, 224, 14)

    def __init__(self, *, od_config: OmniDiffusionConfig, prefix: str = ""):
        # Resolve the checkpoint root exactly like the BAGEL core (local dir,
        # otherwise HF snapshot).
        model = od_config.model
        if os.path.exists(model):
            model_path = model
        else:
            model_path = download_weights_from_hf_specific(model, od_config.revision, ["*"])

        # Fallback strictly for checkpoints that do not ship a
        # preprocessor_config.json (SenseNova-Vision-7B-MoT). BAGEL checkpoints
        # ship the file and never reach this path.
        if not os.path.isfile(os.path.join(model_path, "preprocessor_config.json")):
            patched_dir = tempfile.mkdtemp(prefix="sensenova_vision_preprocessor_")
            for entry in os.listdir(model_path):
                src = os.path.join(model_path, entry)
                dst = os.path.join(patched_dir, entry)
                if not os.path.lexists(dst):
                    os.symlink(src, dst)
            with open(os.path.join(patched_dir, "preprocessor_config.json"), "w", encoding="utf-8") as f:
                json.dump(_SENSENOVA_VISION_PREPROCESSOR_CONFIG, f)
            # Keep the patched dir alive for the lifetime of the pipeline
            # (it is referenced by self.od_config.model / weights sources).
            self._sensenova_vision_patched_dir = patched_dir
            logger.info(
                "SenseNova-Vision: checkpoint %s lacks preprocessor_config.json; "
                "patched %s and pointed the diffusion stage at it",
                model_path,
                patched_dir,
            )
            od_config = replace(od_config, model=patched_dir)

        # Construct the custom tokenizer in-process and inject it into the BAGEL
        # core.  ``VLLMSenseNovaVisionTokenizer`` loads the base vocab and the
        # checkpoint's ``added_tokens_decoder`` ids verbatim, so no
        # ``tokenizer_config.json`` rewrite and no source file are needed — the
        # BAGEL core receives a ready-made tokenizer via the ``tokenizer`` kwarg
        # instead of calling ``AutoTokenizer.from_pretrained`` itself.
        tokenizer = VLLMSenseNovaVisionTokenizer.from_pretrained(
            model_path,
            local_files_only=True,
            trust_remote_code=True,
        )

        super().__init__(od_config=od_config, prefix=prefix, tokenizer=tokenizer)
        self._apply_sensenova_vision_defaults()

    def _apply_sensenova_vision_defaults(self) -> None:
        """Force SenseNovaVision defaults on the loaded Bagel core after init."""
        bagel = self.bagel
        bagel.max_latent_size = self._sensenova_vision_max_latent_size
        if hasattr(bagel.config, "max_latent_size"):
            bagel.config.max_latent_size = self._sensenova_vision_max_latent_size
        if hasattr(bagel.config, "vit_max_num_patch_per_side"):
            bagel.config.vit_max_num_patch_per_side = self._sensenova_vision_vit_max_num_patch_per_side
        if hasattr(bagel.latent_pos_embed, "max_num_patch_per_side"):
            bagel.latent_pos_embed.max_num_patch_per_side = self._sensenova_vision_max_latent_size

    def forward(self, req) -> DiffusionOutput:
        """Run SenseNovaVision image/text generation with per-mode defaults.

        The BAGEL core already models ``num_output_vae`` view packing for a
        multi-view image decode directly (``image_sizes`` length-N,
        ``packed_seqlens`` length-N, ``generate_image`` returns N unpacked
        latents).  SenseNovaVision only needs to branch the AR-supplied KV
        context across those ``N`` view branches and emit one image per
        view. ``recon3d`` is the first task to request this. For one-stage
        image requests, local context prefill remains in this pipeline so its
        VAE/ViT resize chain stays aligned with the model executor.

        The BAGEL core already decodes latents through the instance method
        :meth:`_decode_image_from_latent`: when the request opts into
        ``output_type="raw_tensor"`` (see :meth:`_should_return_raw_tensor`)
        the override returns raw HxWx3 float32 VAE tensors (upstream
        ``output_raw_tensor=True``) instead of 8-bit PIL images, for every
        image-producing mode including ``recon3d``.
        """
        injected_kv = req.sampling_params.past_key_values
        if injected_kv is not None:
            logger.info("[SenseNova-Vision] diffusion stage - get injected_kv")
        self._apply_mode_defaults(req)
        if self._is_recon3d(req):
            return self._forward_recon3d(req)
        output = super().forward(req)
        return self._merge_mixed_task_text(req, output)

    @staticmethod
    def _should_return_raw_tensor(params: OmniDiffusionSamplingParams) -> bool:
        """True when the request opts into raw float32 tensor image outputs.

        Reads the request-scoped ``output_type`` knob (request-scoped so the
        OpenAI-compatible server, which sets no such flag, always keeps the
        PIL decode): ``params.output_type`` is the canonical field; the
        ``extra_args["output_type"]`` fallback supports callers that only
        speak ``extra_args``.  ``"raw_tensor"`` is the only raw mode; any
        other value (including ``None``/Wan's ``"latent"``/``"np"``) keeps
        the default PIL behavior.
        """
        output_type = getattr(params, "output_type", None)
        if output_type is None:
            extra_args = getattr(params, "extra_args", None) or {}
            output_type = extra_args.get("output_type")
        return output_type == "raw_tensor"

    @staticmethod
    def _is_recon3d(req: DiffusionRequestBatch) -> bool:
        """True when the request selects the multi-view ``recon3d`` mode."""
        params = getattr(req, "sampling_params", None)
        extra_args = getattr(params, "extra_args", None) or {}
        return bool(extra_args.get("sensenova_vision_mode") == "recon3d")

    @staticmethod
    def _count_conditioned_views(req: DiffusionRequestBatch) -> int:
        """Number of VAE+ViT conditioning blocks in the recon3d prompt.

        Each input view contributes exactly one ``<|fim_middle|>`` marker
        (``_format_recon3d_prompts`` builds the marker block per view, and the
        orchestrator hands the original prompt dict to this stage unchanged),
        so the marker count equals the number of conditioned views.  Counting
        markers also stays correct when ``multi_modal_data`` is dropped on the
        AR->DiT stage boundary.
        """
        prompts = getattr(req, "prompts", None) or []
        prompt = prompts[0] if prompts else None
        if not isinstance(prompt, dict):
            return 0
        return str(prompt.get("prompt", "")).count(IMG2IMG_PLACEHOLDER)

    @staticmethod
    def _count_single_stage_images(req: DiffusionRequestBatch) -> int:
        """Return the locally supplied context-image count for a one-stage request."""
        prompts = getattr(req, "prompts", None) or []
        prompt = prompts[0] if prompts else None
        if not isinstance(prompt, dict):
            return 0
        data = prompt.get("multi_modal_data") or {}
        images = data.get("img2img") or data.get("image")
        if images is None:
            return 0
        return len(images) if isinstance(images, list) else 1

    def _forward_recon3d(self, req: DiffusionRequestBatch) -> DiffusionOutput:
        """Multi-view ``recon3d`` decode: one AR context, N output views.

        Mirrors the upstream ``gen_image`` packing (``gen_image`` in
        ``SenseNova-Vision/inference/inferencer.py``): ``prepare_vae_latent``
        is fed one entry per view (``curr_kvlens = [kv_len] + [0]*(N-1)``,
        ``curr_rope = base + 0..N-1``) so the packed indexes / RoPE / position
        ids are built per view, and the per-view ``packed_seqlens`` are then
        **collapsed into a single packed query sequence** (upstream
        ``inferencer.py:164-166``).  All N views therefore co-attend, within
        one non-causal sequence per CFG branch, to the full AR KV cache *and*
        to each other.  The denoised latents are finally re-split by the saved
        per-view ``packed_seqlens - 2`` into ``num_views`` chunks, which this
        override decodes individually.

        Fidelity note: unless the request opts into
        ``output_type="raw_tensor"`` the views are 8-bit PIL images
        (matching upstream ``decode_image(output_raw_tensor=False)``); with
        the flag set, :meth:`_decode_latent_raw` returns the float32 VAE
        tensors exactly like upstream ``decode_image(...,
        output_raw_tensor=True)``, so point-map evaluators keep full float
        precision without an 8-bit round-trip.
        """
        params = req.sampling_params
        extra_args = getattr(params, "extra_args", None) or {}

        # ``num_output_vae`` in upstream ``gen_image``.  Upstream
        # ``reconstruct_3d`` runs with ``output_multiple_vae=True`` so
        # ``interleave_inference`` derives it from the input view count
        # (``max(input_image_count, 1)``) and every input view gets one point
        # map; mirror that here by defaulting to the number of conditioned
        # views (the ``<|fim_middle|>`` VAE+ViT blocks in the prompt) instead
        # of a hardcoded 4.  ``num_views`` stays the explicit per-request
        # override (e.g. end2end.py --num-views) and must cover every
        # conditioned view.
        num_conditioned_views = self._count_conditioned_views(req)
        # A staged request carries one ``<|fim_middle|>`` per view.  The
        # single-stage BAGEL-compatible prompt instead contains image-pad
        # placeholders, so derive its count from the local multimodal payload.
        if getattr(params, "past_key_values", None) is None:
            num_conditioned_views = max(num_conditioned_views, self._count_single_stage_images(req))
        num_views = int(extra_args.get("num_views", max(num_conditioned_views, 1)))
        if num_views < 1:
            raise ValueError(f"recon3d requires num_views >= 1, got {num_views}.")
        if num_views < num_conditioned_views:
            raise ValueError(
                f"recon3d conditioned on {num_conditioned_views} view(s) but num_views "
                f"is {num_views}; the output view count must cover every conditioned view."
            )

        injected_kv = getattr(params, "past_key_values", None)
        if injected_kv is None:
            # Build the three local CFG contexts with SenseNova's VAE->ViT
            # transforms, then feed them through the same collapsed multi-view
            # denoising path as staged recon3d.
            gen_cfg_context, cfg_text_context, cfg_img_context, image_shape = self._prepare_single_stage_contexts(
                req.prompts[0], params
            )
            gen_cache = gen_cfg_context["past_key_values"]
            kv_len = int(gen_cfg_context["kv_lens"][0])
            base_rope = int(gen_cfg_context["ropes"][0])
            for context in (gen_cfg_context, cfg_text_context, cfg_img_context):
                context["past_key_values"].key_values_lens = [int(context["kv_lens"][0])]
        else:
            gen_cache = NaiveCache.from_object(injected_kv)
            kv_len = gen_cache.key_cache[0].shape[0]

            kv_metadata = getattr(params, "kv_metadata", None) or {}
            if "image_shape" in kv_metadata:
                image_shape = tuple(kv_metadata["image_shape"])
            else:
                # Per-task VAE target side (512 for recon3d), clamped to the
                # checkpoint latent budget so the DiT grid stays in-bounds.
                side = PER_TASK_VAE_SIDE.get("recon3d")
                if side is None or side // self.bagel.latent_downsample > self.bagel.max_latent_size:
                    side = int(self.bagel.max_latent_size * self.bagel.latent_downsample)
                image_shape = (side, side)

            ropes = kv_metadata.get("ropes") or [kv_len]
            base_rope = int(ropes[0])
            gen_cfg_context = {"kv_lens": [kv_len], "ropes": [base_rope], "past_key_values": gen_cache}
        kv_lens, view_ropes = recon3d_packing(num_views, kv_len, base_rope)
        # Every view attends to the whole AR context, so the gen cache is ONE
        # packed sequence (a single ``key_values_lens`` entry) rather than one
        # slice per view: ``PackedAttentionMoT._forward_gen`` then splits the
        # merged cache once per CFG branch, and each branch's query sequence —
        # all N views — sees the full AR KV.  Upstream collapses
        # ``key_values_lens`` the same way (``inferencer.py:164-166``).
        gen_cache.key_values_lens = [kv_len]

        # The per-view entries build the packed indexes / RoPE / position ids.
        generation_input = self.bagel.prepare_vae_latent(
            curr_kvlens=kv_lens,
            curr_rope=view_ropes,
            image_sizes=[image_shape] * num_views,
            new_token_ids=self.new_token_ids,
        )
        for k, v in generation_input.items():
            if torch.is_tensor(v):
                generation_input[k] = v.to(self.device)

        # Collapse the N per-view sequences into ONE packed query sequence
        # (upstream ``inferencer.py:164-166``): the views co-attend inside a
        # single non-causal sequence per CFG branch, instead of N independent
        # branches where only view 0 owns the AR cache and the rest denoise
        # from pure noise.  The pre-collapse ``packed_seqlens`` is kept as the
        # unpack lengths, since the denoised ``x_t`` still holds one
        # contiguous chunk of ``h*w`` latent tokens per view (upstream keeps
        # the same variable to split ``x_0`` at ``inferencer.py:225``).
        unpack_seqlens = generation_input["packed_seqlens"]
        generation_input["packed_seqlens"] = torch.sum(unpack_seqlens, dim=0, keepdim=True, dtype=unpack_seqlens.dtype)

        gen_params = BagelGenParams(
            num_timesteps=int(params.num_inference_steps or 50),
            timestep_shift=float(extra_args.get("timestep_shift", 3.0)),
            cfg_text_scale=float(extra_args.get("cfg_text_scale", 1.0)),
            cfg_img_scale=float(extra_args.get("cfg_img_scale", 1.0)),
            cfg_interval=extra_args.get("cfg_interval", (0.0, 1.0)),
            cfg_renorm_type=extra_args.get("cfg_renorm_type", "global"),
            cfg_renorm_min=float(extra_args.get("cfg_renorm_min", 0.0)),
        )

        # Build the CFG branch inputs exactly like the standard BAGEL path
        # (``pipeline_bagel._forward_single``: prepare_vae_latent_cfg for both
        # branches, then pass cfg_*_packed_position_ids and cfg_*_past_key_values
        # into ``generate_image``) and upstream ``InterleaveInferencer.gen_image``
        # (``inferencer.py:168-222``).  The AR stage prefills all companion
        # requests (gen / cfg_text / cfg_img); recon3d must feed matching branch
        # position-ids and KV caches back into the BAGEL core, otherwise the
        # sequential-CFG/SP path dereferences a None branch pid.
        #
        # Each CFG branch shares the same multi-view layout as the gen branch:
        # ``prepare_vae_latent_cfg`` still receives per-view inputs
        # (``kv_lens_cfg + [0]*(num_views-1)``, ``rope + 0..num_views-1``) while
        # the branch cache itself is the single collapsed AR sequence.
        use_cfg_text = gen_params.cfg_text_scale > 1.0
        use_cfg_img = use_cfg_text and gen_params.cfg_img_scale > 1.0

        def _branch_context(
            kv_attr: str,
            metadata_attr: str,
            default_context: dict[str, Any],
        ) -> dict[str, Any]:
            """Resolve one CFG branch context from the transferred companion KV.

            Falls back to the gen branch (full AR context) when the companion
            was not transferred, mirroring ``pipeline_bagel`` where the text-
            unconditional / no-image branches reuse the gen KV.
            """
            kv = getattr(params, kv_attr, None)
            if kv is None:
                return default_context
            cache = NaiveCache.from_object(kv)
            seq_len = cache.key_cache[0].shape[0]
            metadata = getattr(params, metadata_attr, None) or {}
            rope = int((metadata.get("ropes") or [seq_len])[0])
            # Every CFG branch carries one packed sequence holding all views,
            # so its cache is a single entry — the same collapsed shape as the
            # gen cache.  ``_forward_gen`` splits the merged cache by these
            # lengths (``batched_seqlens`` = the collapsed ``packed_seqlens``
            # repeated once per CFG branch), so the merged list must hold
            # exactly one entry per branch.
            cache.key_values_lens = [seq_len]
            return {"kv_lens": [seq_len], "ropes": [rope], "past_key_values": cache}

        # cfg_text branch: unconditional text.  Upstream clones gen_context
        # right BEFORE the last prompt string is added (so it holds all views
        # but no prompt text); in the staged AR->DiT flow this is the
        # ``__cfg_text`` companion KV.
        if injected_kv is not None:
            cfg_text_context = _branch_context("cfg_text_past_key_values", "cfg_text_kv_metadata", gen_cfg_context)

        # cfg_img branch: same text as gen but WITHOUT image conditioning.  For
        # text2img the upstream / BAGEL paths reuse the gen KV here
        # (``pipeline_bagel.py:639-649``); recon3d has no separate no-image
        # preconditioned context either, so reuse the gen branch KV when the
        # ``__cfg_img`` companion was not transferred.
        if injected_kv is not None:
            cfg_img_context = _branch_context("cfg_img_past_key_values", "cfg_img_kv_metadata", gen_cfg_context)

        def _cfg_pids(context: dict[str, Any]) -> torch.Tensor:
            """Multi-view CFG branch position-ids (mirrors prepare_vae_latent_cfg)."""
            cfg_kv_lens, cfg_ropes = context["kv_lens"], context["ropes"]
            pids = self.bagel.prepare_vae_latent_cfg(
                curr_kvlens=cfg_kv_lens + [0] * (num_views - 1),
                curr_rope=list(cfg_ropes[0] + x for x in range(num_views)),
                image_sizes=[image_shape] * num_views,
            )
            return pids["cfg_packed_position_ids"].to(self.device)

        cfg_branch_inputs: dict[str, Any] = {}
        cfg_branch_caches: dict[str, Any] = {}
        if use_cfg_text:
            cfg_branch_inputs["cfg_text_packed_position_ids"] = _cfg_pids(cfg_text_context)
            cfg_branch_caches["cfg_text_past_key_values"] = cfg_text_context["past_key_values"]
        if use_cfg_img:
            cfg_branch_inputs["cfg_img_packed_position_ids"] = _cfg_pids(cfg_img_context)
            cfg_branch_caches["cfg_img_past_key_values"] = cfg_img_context["past_key_values"]

        if params.seed is not None:
            torch.Generator(device=self.device.type).manual_seed(params.seed)

        with torch.autocast(
            device_type=self.device.type,
            enabled=self.device.type != "cpu",
            dtype=self.od_config.dtype,
        ):
            latents, _, _, _ = self.bagel.generate_image(
                past_key_values=gen_cache,
                num_timesteps=gen_params.num_timesteps,
                timestep_shift=gen_params.timestep_shift,
                cfg_text_scale=gen_params.cfg_text_scale,
                cfg_img_scale=gen_params.cfg_img_scale,
                cfg_interval=gen_params.cfg_interval,
                cfg_renorm_min=gen_params.cfg_renorm_min,
                cfg_renorm_type=gen_params.cfg_renorm_type,
                return_trajectory_latents=False,
                scheduler=self.scheduler,
                scheduler_kwargs=self.scheduler_kwargs,
                unpack_seqlens=unpack_seqlens,
                **generation_input,
                **cfg_branch_inputs,
                **cfg_branch_caches,
            )

        # Per-view decode: ``generate_image`` unpacked the denoised ``x_t`` by
        # ``unpack_seqlens - 2`` (= ``h*w`` per view, upstream
        # ``inferencer.py:225``), so each entry is one view's
        # ``(h*w, latent_channel*patch^2)`` latent reshaped to the shared
        # ``image_shape`` and passed to the VAE decoder.
        # ``_decode_image_from_latent`` returns raw float32 tensors when the
        # request opts into ``output_type="raw_tensor"``, else the standard
        # 8-bit PIL image (see method docstring).
        images = [
            self._decode_image_from_latent(self.bagel, self.vae, lat, image_shape, params)
            for lat in latents
            if lat is not None
        ]
        return build_sensenova_vision_diffusion_output(
            image=images,
            stage_durations=getattr(self, "stage_durations", None),
        )

    def _decode_latent_raw(
        self,
        bagel: Bagel,
        vae: AutoEncoder,
        latent: torch.Tensor,
        image_shape: tuple[int, int],
    ) -> np.ndarray:
        """Decode a latent to a raw HxWx3 float32 array (upstream ``output_raw_tensor=True``).

        Mirrors :meth:`BagelPipeline._decode_image_from_latent` but returns
        the VAE output itself instead of an 8-bit PIL image: the einsum
        unpack + VAE dtype cast are identical, only the float-normalized
        clamp and ``uint8`` conversion are skipped, matching upstream
        ``decode_image(latent, image_shape, output_raw_tensor=True)`` which
        returns ``image[0].permute(1, 2, 0).float().cpu().numpy()``.
        """
        H, W = image_shape
        h, w = H // bagel.latent_downsample, W // bagel.latent_downsample
        p = bagel.latent_patch_size
        c = bagel.latent_channel
        latent = latent.reshape(1, h, w, p, p, c)
        latent = torch.einsum("nhwpqc->nchpwq", latent)
        latent = latent.reshape(1, c, h * p, w * p)

        # Cast to VAE dtype (e.g. bfloat16) as latents might remain float32 from generation loop
        vae_dtype = next(vae.parameters()).dtype
        latent = latent.to(vae_dtype)

        image = vae.decode(latent)
        return np.asarray(image[0].permute(1, 2, 0).float().cpu())

    def _decode_image_from_latent(
        self,
        bagel: Bagel,
        vae: AutoEncoder,
        latent: torch.Tensor,
        image_shape: tuple[int, int],
        params: OmniDiffusionSamplingParams | None = None,
    ) -> PIL.Image.Image | np.ndarray:
        """Decode a latent to an image, dispatching on the request raw-tensor flag.

        The BAGEL core decodes every image-producing mode (text2img, img2img,
        dense, edit, mixed, think-*) through this method; ``_forward_recon3d``
        calls it once per view.  Default behavior is unchanged (8-bit PIL via
        ``super()``); only a request that opted into
        ``output_type="raw_tensor"`` receives the raw HxWx3 float32 array.
        """
        if self._should_return_raw_tensor(params):
            # logger.info(f"[SenseNova-Vision] return raw tensor: {params=}")
            return self._decode_latent_raw(bagel, vae, latent, image_shape)
        return super()._decode_image_from_latent(bagel, vae, latent, image_shape)

    def _merge_mixed_task_text(self, req: DiffusionRequestBatch, output: DiffusionOutput) -> DiffusionOutput:
        """Lift an available caption/think string into the mixed text+image payload.

        The BAGEL core returns image generations as ``payload["image"]`` and, for
        thinking modes, records the generated text only under
        ``metadata["text"]["think_text"]``. SenseNovaVision mixed tasks (e.g.
        ``caption_generate``/``think_generate``) must represent both under the
        existing ``TEXT | IMAGE`` output-modality contract, so when an image
        payload carries an available text it is also exposed as
        ``payload["text"]``. Additive only: outputs that already carry a text
        payload, or that have no available text, are returned unchanged.
        """
        raw_output = output.output
        if not isinstance(raw_output, dict):
            return output
        payload = raw_output.get("payload")
        if not isinstance(payload, dict) or "image" not in payload or "text" in payload:
            return output

        text = None
        text_meta = raw_output.get("metadata") or {}
        if isinstance(text_meta, dict):
            text_group = text_meta.get("text")
            if isinstance(text_group, dict):
                text = text_group.get("text_output") or text_group.get("think_text")
        if text is None:
            extra_args = getattr(req.sampling_params, "extra_args", None) or {}
            # Only lift an AR-supplied caption for thinking modes.  ``think``
            # is injected by ``_apply_mode_defaults`` from the mode's
            # ``_BASE_PARAMS``; dense/edit/generate modes are False, so a stray
            # 1-token AR decode (max_tokens=1) must never surface as text.
            if extra_args.get("think"):
                candidate = extra_args.get("text_output")
                if isinstance(candidate, str) and candidate:
                    text = candidate
        if text is None:
            return output

        payload["text"] = text
        text_meta = raw_output.setdefault("metadata", {})
        text_group = text_meta.get("text")
        if not isinstance(text_group, dict):
            text_meta["text"] = text_group = {}
        text_group.setdefault("text_output", text)
        return output

    def _apply_mode_defaults(self, req: DiffusionRequestBatch) -> None:
        """Inject SenseNovaVision ``BASE_PARAMS`` defaults into sampling params.

        The BAGEL core reads CFG/timestep knobs from ``extra_args``; SenseNovaVision
        defines those defaults per mode.  User-supplied values always win.
        """
        if not req.requests:
            return
        params = req.requests[0].sampling_params
        if params is None:
            return

        mode = None
        prompt = req.prompts[0] if req.prompts else None
        if isinstance(prompt, dict):
            mode = prompt.get("mode") or prompt.get("sensenova_vision_mode")

        if not mode:
            # No explicit mode: text-only output uses understanding-style
            # defaults, otherwise use "generate" image defaults.
            modalities = prompt.get("modalities", []) if isinstance(prompt, dict) else []
            if "text" in modalities and "image" not in modalities:
                mode = "understanding"
            else:
                mode = "generate"

        defaults = SenseNovaVisionGenParams.from_base_params(mode)
        extra_args = dict(getattr(params, "extra_args", None) or {})

        # Only fill unset knobs; explicit user args win.
        extra_args.setdefault("cfg_text_scale", defaults.cfg_text_scale)
        extra_args.setdefault("cfg_img_scale", defaults.cfg_img_scale)
        extra_args.setdefault("cfg_interval", defaults.cfg_interval)
        extra_args.setdefault("timestep_shift", defaults.timestep_shift)
        extra_args.setdefault("cfg_renorm_min", defaults.cfg_renorm_min)
        extra_args.setdefault("cfg_renorm_type", defaults.cfg_renorm_type)
        extra_args.setdefault("think", defaults.think)
        extra_args.setdefault("max_think_tokens", defaults.max_think_token_n)
        extra_args.setdefault("do_sample", defaults.do_sample)
        extra_args.setdefault("text_temperature", defaults.text_temperature)
        extra_args.setdefault("sensenova_vision_mode", mode)
        if defaults.extra:
            for k, v in defaults.extra.items():
                extra_args.setdefault(k, v)

        params.extra_args = extra_args
        if params.num_inference_steps is None:
            params.num_inference_steps = defaults.num_timesteps
