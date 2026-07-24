# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DreamX-World-5B-Cam pipeline: Wan2.2 TI2V-5B I2V + PRoPE camera/action control.

Subclasses ``Wan22Pipeline`` (the TI2V-5B ``expand_timesteps`` I2V path) and adds
camera control with minimal overrides: a two-root load (camera transformer from the
transformer-only DreamX repo, VAE / text_encoder / tokenizer from the base
``Wan-AI/Wan2.2-TI2V-5B-Diffusers``), and a PRoPE camera condition built in
pre-process from ``action_seq`` / ``action_speed_list`` and injected into the
transformer as a CFG-invariant ``cam_emb``.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any

import torch
from torch import nn
from transformers import AutoTokenizer, UMT5EncoderModel

from vllm_omni.diffusion.data import OmniDiffusionConfig
from vllm_omni.diffusion.distributed.autoencoders.autoencoder_kl_wan import DistributedAutoencoderKLWan
from vllm_omni.diffusion.distributed.utils import get_local_device
from vllm_omni.diffusion.model_loader.diffusers_loader import DiffusersPipelineLoader
from vllm_omni.diffusion.model_loader.hub_prefetch import from_pretrained_with_prefetch, prefetch_subfolders
from vllm_omni.diffusion.request import OmniDiffusionRequest
from vllm_omni.diffusion.worker.request_batch import DiffusionRequestBatch

from .camera_pose_utils import build_camera_condition
from .pipeline_wan2_2 import (
    Wan22Pipeline,
    build_wan_scheduler,
    get_wan22_post_process_func,
    get_wan22_pre_process_func,
)
from .wan2_2_camera_transformer import WanCameraTransformer3DModel, create_camera_transformer_from_config

logger = logging.getLogger(__name__)

# Default base repo providing VAE / text_encoder / tokenizer / scheduler (DreamX's
# HF metadata lists Wan2.2-T2V-5B; inference uses the unified TI2V-5B diffusers repo).
DEFAULT_BASE_MODEL = "Wan-AI/Wan2.2-TI2V-5B-Diffusers"

# Default negative prompt, vendored verbatim from DreamX upstream
# (inference_dreamx5b.py). Used when the request leaves negative_prompt empty.
DREAMX_NEGATIVE_PROMPT = (
    "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，"
    "低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，"
    "毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走"
)

# Wan2.2 VAE temporal compression (latent frames satisfy the 1+4k pattern).
_VAE_TEMPORAL = 4


def _load_root_transformer_config(model: str, local_files_only: bool) -> dict:
    """Read ``config.json`` at the repo ROOT (the DreamX transformer-only layout).

    The base ``load_transformer_config`` assumes a ``transformer/`` subfolder and,
    for remote repos, would build the broken filename ``"/config.json"`` from an
    empty subfolder; this reads the root config directly for both local and Hub.
    """
    if local_files_only:
        config_path = os.path.join(model, "config.json")
        if os.path.exists(config_path):
            with open(config_path) as f:
                return json.load(f)
        return {}
    from huggingface_hub import hf_hub_download

    config_path = hf_hub_download(repo_id=model, filename="config.json")
    with open(config_path) as f:
        return json.load(f)


def get_wan22_camera_post_process_func(od_config: OmniDiffusionConfig):
    """Reuse the standard Wan2.2 video post-process (tensor -> frames)."""
    return get_wan22_post_process_func(od_config)


def get_wan22_camera_pre_process_func(od_config: OmniDiffusionConfig):
    """Resize the start image (base behaviour) + build the PRoPE camera condition.

    Reads ``action_seq`` / ``action_speed_list`` from
    ``request.sampling_params.extra_args`` (routed there from ``extra_body`` via the
    ``model_extras`` declaration), snaps ``num_frames`` to the 1+4k VAE pattern (so
    the camera frame count matches the latent frame count), builds
    ``{"viewmats", "K"}`` on CPU, and stashes it in ``additional_information``.
    """
    base_pre = get_wan22_pre_process_func(od_config)

    def pre_process_func(request: OmniDiffusionRequest) -> OmniDiffusionRequest:
        request = base_pre(request)  # image resize + height/width

        sp = request.sampling_params
        extra = getattr(sp, "extra_args", {}) or {}
        action_seq = extra.get("action_seq")
        action_speed_list = extra.get("action_speed_list")
        if action_seq is None or action_speed_list is None:
            if request.is_dummy_run():
                # Engine warmup does not route extra_body; use a minimal trajectory.
                action_seq = ["w"]
                action_speed_list = [1]
            else:
                # Camera control is the whole point — fail loud rather than
                # silently degrade to plain I2V.
                raise ValueError(
                    "WanCameraPipeline requires camera control: pass both 'action_seq' and "
                    "'action_speed_list' via extra_body. For plain image-to-video without "
                    "camera control, use the base Wan2.2-TI2V-5B (WanPipeline)."
                )

        num_frames = sp.num_frames
        if num_frames is None:
            raise ValueError("num_frames must be set when using camera control (action_seq/action_speed_list)")
        # Snap to 1+4k BEFORE building the trajectory so the camera frame count
        # equals the latent frame count (the transformer asserts cameras ==
        # post_patch_num_frames). For canonical 1+4k inputs (121/81) this matches
        # upstream exactly.
        if num_frames % _VAE_TEMPORAL != 1:
            snapped = num_frames // _VAE_TEMPORAL * _VAE_TEMPORAL + 1
            logger.warning(
                "num_frames=%d does not satisfy the Wan2.2 VAE 1+4k temporal pattern; snapping to %d.",
                num_frames,
                snapped,
            )
            num_frames = snapped
        num_frames = max(num_frames, 1)
        sp.num_frames = num_frames

        camera_condition = build_camera_condition(
            action_seq,
            action_speed_list,
            sp.height or 0,  # unused by the PRoPE path
            sp.width or 0,
            num_frames,
        )

        # base_pre already normalised request.prompt to an OmniTextPrompt with
        # an "additional_information" dict (bare strings included).
        prompt = request.prompt
        prompt["additional_information"]["camera_condition"] = camera_condition
        # Apply the upstream DreamX default negative prompt when the caller
        # left it empty (the base encoder otherwise uses ""), for parity.
        if not prompt.get("negative_prompt"):
            prompt["negative_prompt"] = DREAMX_NEGATIVE_PROMPT
        return request

    return pre_process_func


def _extract_camera_condition(req: DiffusionRequestBatch) -> dict[str, torch.Tensor]:
    """Read the pre-process camera condition and add a batch dim ([T,4,4]->[1,T,4,4]).

    Raises instead of falling back to plain I2V: a missing condition means the
    camera pre-process was bypassed or the controls were never attached, and
    silently continuing would hide such integration regressions. The dummy
    warmup always passes through the pre-process, which attaches a minimal
    trajectory, so this never fires during engine startup.

    Batched requests are rejected up front: this helper only reads ``prompts[0]``,
    and camera trajectories are per-request.
    """
    prompts = req.prompts
    if len(prompts) > 1:
        raise ValueError(
            "WanCameraPipeline supports batch size 1: camera trajectories are "
            "per-request and cannot be shared across a batched request."
        )
    first = prompts[0] if prompts else None
    cc = None
    if first is not None and not isinstance(first, str):
        cc = first.get("additional_information", {}).get("camera_condition")
    if cc is None:
        raise ValueError(
            "WanCameraPipeline.forward received no camera_condition: the camera "
            "pre-process was bypassed or 'action_seq'/'action_speed_list' were not "
            "attached to the request. For plain image-to-video without camera "
            "control, use the base Wan2.2-TI2V-5B (WanPipeline)."
        )
    return {k: (v.unsqueeze(0) if v.dim() == 3 else v) for k, v in cc.items()}


class Wan22CameraPipeline(Wan22Pipeline):
    """Wan2.2 TI2V-5B + PRoPE camera control (DreamX-World-5B-Cam)."""

    def __init__(self, *, od_config: OmniDiffusionConfig, prefix: str = ""):
        # Initialise the mixin/nn.Module chain WITHOUT Wan22Pipeline.__init__,
        # which assumes a single diffusers root (DreamX ships a transformer-only,
        # native-named repo + needs base components from a second root).
        super(Wan22Pipeline, self).__init__()
        self.od_config = od_config
        self.device = get_local_device()
        dtype = getattr(od_config, "dtype", torch.bfloat16)

        model = od_config.model  # DreamX transformer repo (root: config.json + safetensors)
        local_files_only = os.path.exists(model)

        model_config = getattr(od_config, "model_config", None) or {}
        base_model = model_config.get("base_model_path") or DEFAULT_BASE_MODEL
        base_local = os.path.exists(base_model)

        # TI2V-5B: single transformer, expand_timesteps I2V; no MoE boundary.
        self.expand_timesteps = True
        self.has_transformer_2 = False
        self.boundary_ratio = od_config.boundary_ratio

        # Transformer weights come from the DreamX repo ROOT. Use subfolder=None
        # (the loader's root convention, as in bagel/sensenova/cosmos3); an empty
        # string would build a broken "/..." path for remote repos.
        self.weights_sources = [
            DiffusersPipelineLoader.ComponentSource(
                model_or_path=model,
                subfolder=None,
                revision=None,
                prefix="transformer.",
                fall_back_to_pt=True,
            )
        ]

        # VAE / text_encoder / tokenizer from the base diffusers repo.
        component_subfolders = ["tokenizer", "text_encoder", "vae"]
        prefetch_subfolders(base_model, component_subfolders, local_files_only=base_local)
        self.tokenizer = from_pretrained_with_prefetch(
            AutoTokenizer.from_pretrained,
            base_model,
            subfolder="tokenizer",
            prefetch_list=component_subfolders,
            local_files_only=base_local,
        )
        self.text_encoder = from_pretrained_with_prefetch(
            UMT5EncoderModel.from_pretrained,
            base_model,
            subfolder="text_encoder",
            prefetch_list=component_subfolders,
            local_files_only=base_local,
            torch_dtype=dtype,
        ).to(self.device)
        self.vae = from_pretrained_with_prefetch(
            DistributedAutoencoderKLWan.from_pretrained,
            base_model,
            subfolder="vae",
            prefetch_list=component_subfolders,
            local_files_only=base_local,
            torch_dtype=dtype,
        ).to(self.device)

        # Transformer structure from the DreamX native config.json (repo root).
        transformer_config = _load_root_transformer_config(model, local_files_only=local_files_only)
        if not transformer_config:
            raise RuntimeError(f"Could not load DreamX transformer config.json from {model!r}")
        self.transformer = self._create_transformer(transformer_config)
        self.transformer_2 = None
        self.transformer_config = self.transformer.config

        self._sample_solver = "unipc"
        # DreamX default flow_shift is 3.0 (vs the base Wan default of 5.0). The base
        # forward re-resolves flow_shift every request via resolve_wan_flow_shift,
        # which falls back to 5.0 when od_config.flow_shift is None — so persist the
        # 3.0 default onto od_config (unless the caller set one) to keep parity.
        if od_config.flow_shift is None:
            od_config.flow_shift = 3.0
        self._flow_shift = od_config.flow_shift
        self.scheduler = build_wan_scheduler(self._sample_solver, self._flow_shift)

        self.vae_scale_factor_temporal = self.vae.config.scale_factor_temporal if getattr(self, "vae", None) else 4
        self.vae_scale_factor_spatial = self.vae.config.scale_factor_spatial if getattr(self, "vae", None) else 16

        self._guidance_scale = None
        self._guidance_scale_2 = None
        self._num_timesteps = None
        self._current_timestep = None
        self._active_cam_emb: dict | None = None

        self.setup_diffusion_pipeline_profiler(
            enable_diffusion_pipeline_profiler=od_config.enable_diffusion_pipeline_profiler
        )

    def _create_transformer(self, config: dict) -> WanCameraTransformer3DModel:
        quant_config = getattr(self.od_config, "quantization_config", None)
        return create_camera_transformer_from_config(config, quant_config=quant_config)

    def predict_noise(self, current_model: nn.Module | None = None, **kwargs: Any):
        # Inject the (CFG-invariant) camera condition into every transformer call.
        if self._active_cam_emb is not None:
            kwargs["cam_emb"] = self._active_cam_emb
        return super().predict_noise(current_model=current_model, **kwargs)

    def forward(self, req: DiffusionRequestBatch, *args, **kwargs):
        # Hold the camera condition for predict_noise. Device/dtype move and
        # frame-count validation happen inside the transformer forward.
        self._active_cam_emb = _extract_camera_condition(req)
        try:
            return super().forward(req, *args, **kwargs)
        finally:
            self._active_cam_emb = None
