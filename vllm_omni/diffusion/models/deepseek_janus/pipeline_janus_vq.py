# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Janus VQ-decode-only component.

This pipeline only loads ``gen_vision_model`` and performs VQ decode
(``decode_code``) to convert a predicted image-token grid to an RGB image.
The supported end-to-end Janus deployment remains :class:`JanusPipeline`,
which keeps prompt formatting, AR token generation, CFG, and VQ decode in one
validated stage.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any, ClassVar

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from transformers import AutoConfig
from vllm.logger import init_logger

from vllm_omni.diffusion.data import DiffusionOutput, OmniDiffusionConfig
from vllm_omni.diffusion.distributed.utils import get_local_device
from vllm_omni.diffusion.model_loader.diffusers_loader import DiffusersPipelineLoader
from vllm_omni.diffusion.models.interface import SupportsComponentDiscovery
from vllm_omni.diffusion.profiler.diffusion_pipeline_profiler import DiffusionPipelineProfilerMixin
from vllm_omni.diffusion.worker.request_batch import DiffusionRequestBatch

logger = init_logger(__name__)

_JANUS_IMAGE_TOKEN_NUM = 576
_JANUS_IMAGE_SIZE = 384
_JANUS_PATCH_SIZE = 16
_JANUS_TOKEN_GRID_SIZE = 24


def _model_name_to_cls(cls_name: str):
    if "VQ" in cls_name:
        from vllm_omni.diffusion.models.deepseek_janus._janus_hf_vendor.vq_model import VQ_models

        return VQ_models[cls_name]
    raise ValueError(f"Unknown gen_vision class: {cls_name}")


def _resolve_prompt_extra(req: DiffusionRequestBatch) -> dict[str, Any]:
    if not req.prompts:
        return {}
    first_prompt = req.prompts[0]
    if not isinstance(first_prompt, dict):
        return {}
    resolved: dict[str, Any] = {}
    extra = first_prompt.get("extra")
    if isinstance(extra, dict):
        resolved.update(extra)
    for key in ("img_size", "patch_size", "image_tokens"):
        if key in first_prompt and key not in resolved:
            resolved[key] = first_prompt[key]
    return resolved


def _resolve_janus_vq_geometry(extra: dict[str, Any], prompt_extra: dict[str, Any]) -> tuple[int, int]:
    img_size = int(extra.get("img_size", prompt_extra.get("img_size", _JANUS_IMAGE_SIZE)))
    patch_size = int(extra.get("patch_size", prompt_extra.get("patch_size", _JANUS_PATCH_SIZE)))
    grid = img_size // patch_size if patch_size > 0 else 0
    if img_size != _JANUS_IMAGE_SIZE or patch_size != _JANUS_PATCH_SIZE or grid != _JANUS_TOKEN_GRID_SIZE:
        raise ValueError(
            "DeepSeek Janus VQ decode uses a fixed 8x24x24 grid "
            "(384x384 output with patch_size=16). "
            f"Got img_size={img_size}, patch_size={patch_size}."
        )
    return img_size, patch_size


class JanusVQDecodePipeline(nn.Module, SupportsComponentDiscovery, DiffusionPipelineProfilerMixin):
    """VQ-decode-only pipeline for Janus two-stage deployment.

    This pipeline loads the VQ decoder and converts token IDs to image pixels.

    Optimisation note:
      There are no AR steps in this pipeline; it performs a single VQ decode
      call per request. The VQ model is a convolutional encoder/decoder.
    """

    _dit_modules: ClassVar[list[str]] = []
    _encoder_modules: ClassVar[list[str]] = []
    _vae_modules: ClassVar[list[str]] = ["_vq_model"]
    _resident_modules: ClassVar[list[str]] = []

    def __init__(self, od_config: OmniDiffusionConfig) -> None:
        super().__init__()
        self.od_config = od_config
        self.device = get_local_device()

        dtype = getattr(od_config, "dtype", None) or torch.bfloat16
        from vllm_omni.diffusion.models.deepseek_janus import _register_janus_hf_classes

        _register_janus_hf_classes()

        cfg_kw: dict[str, Any] = {}
        if getattr(od_config, "revision", None):
            cfg_kw["revision"] = od_config.revision
        cfg = AutoConfig.from_pretrained(od_config.model, **cfg_kw)

        gen_vision_cfg = cfg.gen_vision_config
        vq_cls = _model_name_to_cls(gen_vision_cfg.cls)
        self._vq_model = vq_cls()

        self._vq_model.to(dtype=dtype, device=self.device)
        self._vq_model.eval()

        self.weights_sources = [
            DiffusersPipelineLoader.ComponentSource(
                model_or_path=od_config.model,
                subfolder=None,
                revision=getattr(od_config, "revision", None),
                prefix="mm_model.gen_vision_model.",
                fall_back_to_pt=True,
            )
        ]

        self.setup_diffusion_pipeline_profiler(
            enable_diffusion_pipeline_profiler=od_config.enable_diffusion_pipeline_profiler,
        )

    @torch.inference_mode()
    def forward(self, req: DiffusionRequestBatch) -> DiffusionOutput:
        """VQ decode: image token grid → RGB image.

        Expects ``req.extra_kwargs["image_tokens"]`` to contain the
        576 token IDs generated by the AR stage, shaped as
        ``[parallel_size, 576]`` with dtype long.
        """
        extra = req.sampling_params.extra_step_kwargs or {}
        prompt_extra = _resolve_prompt_extra(req)
        sp = req.sampling_params
        parallel_size = max(1, int(sp.num_outputs_per_prompt))
        img_size, patch_size = _resolve_janus_vq_geometry(extra, prompt_extra)

        images_out: list[Image.Image] = []

        # Support both explicit image_tokens and AR-stage-generated tokens
        image_tokens = extra.get("image_tokens")
        if image_tokens is None:
            image_tokens = prompt_extra.get("image_tokens")
        if image_tokens is not None:
            if isinstance(image_tokens, torch.Tensor):
                image_tokens = image_tokens.to(device=self.device, dtype=torch.int)
            else:
                image_tokens = torch.tensor(image_tokens, dtype=torch.int, device=self.device)
            if image_tokens.dim() == 1:
                image_tokens = image_tokens.unsqueeze(0)
            if image_tokens.shape[-1] != _JANUS_IMAGE_TOKEN_NUM:
                raise ValueError(
                    "DeepSeek Janus VQ decode requires exactly 576 image tokens per image, "
                    f"got {image_tokens.shape[-1]}."
                )
        else:
            # Fallback: run the full AR pipeline (for testing)
            return DiffusionOutput(
                error="image_tokens not provided; VQ pipeline requires AR stage output.",
                aborted=True,
            )

        for bi in range(min(parallel_size, image_tokens.shape[0])):
            tokens = image_tokens[bi : bi + 1].to(dtype=torch.int, device=self.device)
            dec = self._vq_model.decode_code(
                tokens,
                shape=[1, 8, img_size // patch_size, img_size // patch_size],
            )
            dec_np = dec.to(torch.float32).cpu().numpy().transpose(0, 2, 3, 1)
            dec_np = np.clip((dec_np + 1.0) / 2.0 * 255.0, 0, 255).astype(np.uint8)
            images_out.append(Image.fromarray(dec_np[0]))

        return DiffusionOutput(
            output={"payload": {"image": images_out}},
            trajectory_decoded=None,
            stage_durations=self.stage_durations if hasattr(self, "stage_durations") else None,
        )

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        """Load VQ model weights with prefix stripping."""
        from vllm_omni.diffusion.models.deepseek_janus import _load_param

        loaded: set[str] = set()
        for name, tensor in weights:
            stripped = name.replace("mm_model.gen_vision_model.", "").replace("gen_vision_model.", "")
            try:
                _load_param(self._vq_model, stripped, tensor)
                loaded.add(name)
            except Exception as e:
                logger.warning("Failed to load VQ weight %s → %s: %s", name, stripped, e)
        return loaded
