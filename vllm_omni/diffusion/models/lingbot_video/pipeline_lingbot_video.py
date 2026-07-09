# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from __future__ import annotations

import inspect
import os
from contextlib import contextmanager
from typing import Any, ClassVar

import torch
from diffusers.utils import BaseOutput
from torch import nn

from vllm_omni.diffusion.data import DiffusionOutput, OmniDiffusionConfig
from vllm_omni.diffusion.distributed.utils import get_local_device
from vllm_omni.diffusion.models.interface import SupportsComponentDiscovery
from vllm_omni.diffusion.request import OmniDiffusionRequest


def _dtype_from_name(value: Any, default: torch.dtype) -> torch.dtype:
    if isinstance(value, torch.dtype):
        return value
    if value is None:
        return default
    normalized = str(value).lower()
    if normalized in {"bf16", "bfloat16", "torch.bfloat16"}:
        return torch.bfloat16
    if normalized in {"fp16", "float16", "torch.float16"}:
        return torch.float16
    if normalized in {"fp32", "float32", "torch.float32"}:
        return torch.float32
    raise ValueError(f"Unsupported LingBot dtype: {value!r}.")


def _extract_prompt(req: OmniDiffusionRequest) -> tuple[str, str | None]:
    if len(req.prompts) != 1:
        raise ValueError("LingBotVideoPipeline currently supports exactly one prompt per request.")
    prompt_obj = req.prompts[0]
    if isinstance(prompt_obj, str):
        return prompt_obj, None
    prompt = prompt_obj.get("prompt", "")
    negative_prompt = prompt_obj.get("negative_prompt")
    return prompt, negative_prompt


@contextmanager
def _patch_qwen3vl_from_pretrained():
    try:
        from transformers import Qwen3VLForConditionalGeneration
    except Exception:
        yield
        return

    original_from_pretrained = Qwen3VLForConditionalGeneration.from_pretrained
    attn_implementation = os.environ.get("LINGBOT_QWEN_ATTN_IMPLEMENTATION", "flash_attention_3")

    @classmethod
    def patched_from_pretrained(cls, pretrained_model_name_or_path, *args, **kwargs):
        if attn_implementation:
            kwargs.setdefault("attn_implementation", attn_implementation)
        if "torch_dtype" in kwargs and "dtype" not in kwargs:
            kwargs["dtype"] = kwargs.pop("torch_dtype")
        return original_from_pretrained(pretrained_model_name_or_path, *args, **kwargs)

    Qwen3VLForConditionalGeneration.from_pretrained = patched_from_pretrained
    try:
        yield
    finally:
        Qwen3VLForConditionalGeneration.from_pretrained = original_from_pretrained


def get_lingbot_video_post_process_func(od_config: OmniDiffusionConfig):
    del od_config

    def post_process_func(frames, output_type: str = "np"):
        del output_type
        if isinstance(frames, list) and len(frames) == 1:
            return frames[0]
        return frames

    return post_process_func


class LingBotVideoPipeline(nn.Module, SupportsComponentDiscovery):
    """Native vLLM-Omni entry for the dense LingBot-Video Diffusers checkpoint.

    This first integration intentionally delegates component construction and
    dense DiT math to the official ``lingbot_video`` package.  It gives
    vLLM-Omni a registered native model class and request surface for the dense
    checkpoint while keeping MoE/fused-expert kernels out of the initial scope.
    """

    supports_step_execution: ClassVar[bool] = False
    _dit_modules: ClassVar[list[str]] = []
    _encoder_modules: ClassVar[list[str]] = []
    _vae_modules: ClassVar[list[str]] = []

    def __init__(self, *, od_config: OmniDiffusionConfig, prefix: str = ""):
        super().__init__()
        del prefix
        self.od_config = od_config
        self.device = get_local_device()
        self.weights_sources = ()

        try:
            from lingbot_video.pipeline_lingbot_video import (
                DEFAULT_NEGATIVE_PROMPT,
            )
            from lingbot_video.pipeline_lingbot_video import (
                LingBotVideoPipeline as OfficialLingBotVideoPipeline,
            )
            from lingbot_video.transformer_lingbot_video import LingBotVideoTransformer3DModel
        except ImportError as exc:
            raise ImportError(
                "LingBotVideoPipeline requires the official `lingbot_video` package. "
                "Install it with `pip install -e <path-to-Robbyant/lingbot-video>` "
                "before using robbyant/lingbot-video-dense-1.3b in vLLM-Omni."
            ) from exc

        dtype = getattr(od_config, "dtype", torch.bfloat16)
        model_config = getattr(od_config, "model_config", None) or {}
        transformer_dtype = _dtype_from_name(model_config.get("transformer_dtype"), dtype)
        text_encoder_dtype = _dtype_from_name(model_config.get("text_encoder_dtype"), dtype)
        vae_dtype = _dtype_from_name(model_config.get("vae_dtype"), torch.float32)
        dtype_map = {
            "default": dtype,
            "transformer": transformer_dtype,
            "text_encoder": text_encoder_dtype,
            "vae": vae_dtype,
        }

        transformer_subfolder = str(model_config.get("transformer_subfolder", "transformer"))
        transformer = LingBotVideoTransformer3DModel.from_pretrained(
            od_config.model,
            subfolder=transformer_subfolder,
            torch_dtype=transformer_dtype,
        )
        with _patch_qwen3vl_from_pretrained():
            self._pipeline = OfficialLingBotVideoPipeline.from_pretrained(
                od_config.model,
                transformer=transformer,
                trust_remote_code=True,
                torch_dtype=dtype_map,
            )
        self._pipeline.to(self.device)
        self._pipeline.set_progress_bar_config(disable=bool(model_config.get("quiet_progress", True)))
        self._call_kwargs = set(inspect.signature(self._pipeline.__call__).parameters)
        self.default_negative_prompt = DEFAULT_NEGATIVE_PROMPT

    def to(self, *args, **kwargs):
        device, _, _, _ = torch._C._nn._parse_to(*args, **kwargs)
        super().to(*args, **kwargs)
        if device is not None and hasattr(self, "_pipeline"):
            self._pipeline.to(device)
            self.device = torch.device(device)
        return self

    def load_weights(self, weights) -> set[str]:
        del weights
        return set()

    @torch.inference_mode()
    def forward(self, req: OmniDiffusionRequest) -> DiffusionOutput:
        prompt, prompt_negative = _extract_prompt(req)
        sampling = req.sampling_params
        extra_args = dict(sampling.extra_args or {})

        generator = sampling.generator
        if isinstance(generator, list):
            generator = generator[0] if generator else None
        if generator is None:
            seed = sampling.seed
            if seed is not None:
                generator = torch.Generator(device=self.device).manual_seed(int(seed))

        height = sampling.height if sampling.height is not None else extra_args.pop("height", 480)
        width = sampling.width if sampling.width is not None else extra_args.pop("width", 480)
        num_frames = sampling.num_frames or extra_args.pop("num_frames", 81)
        num_inference_steps = (
            sampling.num_inference_steps
            if sampling.num_inference_steps is not None
            else extra_args.pop("num_inference_steps", 40)
        )
        guidance_scale = (
            sampling.guidance_scale
            if sampling.guidance_scale_provided or sampling.guidance_scale > 0
            else extra_args.pop("guidance_scale", 6.0)
        )
        shift = extra_args.pop("shift", getattr(self.od_config, "flow_shift", None) or 3.0)
        negative_prompt = extra_args.pop("negative_prompt", prompt_negative or self.default_negative_prompt)
        output_type = sampling.output_type or getattr(self.od_config, "output_type", None) or extra_args.pop(
            "output_type", "np"
        )
        if output_type not in {"np", "latent"}:
            output_type = "np"

        extra_args = {key: value for key, value in extra_args.items() if key in self._call_kwargs}

        output = self._pipeline(
            prompt=prompt,
            negative_prompt=negative_prompt,
            height=height,
            width=width,
            num_frames=num_frames,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            shift=shift,
            generator=generator,
            latents=sampling.latents,
            output_type=output_type,
            batch_cfg=bool(extra_args.pop("batch_cfg", False)),
            null_cond_clone_zero=bool(extra_args.pop("null_cond_clone_zero", False)),
            offload_vae_during_denoise=bool(extra_args.pop("offload_vae_during_denoise", False)),
            return_dict=True,
            **extra_args,
        )
        frames = output.frames if isinstance(output, BaseOutput) else output
        return DiffusionOutput(output=frames)
