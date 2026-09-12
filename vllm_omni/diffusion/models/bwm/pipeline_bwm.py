# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Boundless-World-Model (BWM) pipeline.

BWM (https://github.com/boundless-large-model/boundless-world-model,
weights: https://huggingface.co/BLM-Lab/Boundless-World-Model) is an
action-conditioned video world model for robotic manipulation built on
Wan2.2-TI2V-5B: given history frames and a normalized robot action
trajectory, it generates the resulting manipulation video. Concrete model
integration under the world-model track (RFC #1987), request-level like
Cosmos3 ``forward_dynamics``: one chunk per request, autoregressive rollouts
loop client-side (see ``examples/offline_inference/bwm/``).

Differences from the stock Wan2.2 TI2V path:

* no text/CLIP encoders: the cross-attention context is the action
  encoder's per-frame tokens (see ``BWMConditionEmbedder``);
* multi-frame history conditioning: the first ``num_history_frames`` pixel
  frames are VAE-encoded and pinned at timestep 0 via the expand-timesteps
  mask (the stock I2V pipeline pins only the first frame);
* per-latent-frame action modulation added to the time embedding (adaLN).

Request contract (all under ``multi_modal_data``):

* ``video``: history frames as a ``(T, H, W, C)`` uint8 array, list of PIL
  images, or a single PIL image (T=1);
* ``action``: ``(frames, action_dim)`` float array, normalized with the
  dataset statistics client-side, ``frames >= 1 + 4 * (T_latent - 1)``.

Weights layout (assembled by ``examples/offline_inference/bwm/download_bwm.py``):
``transformer/`` (fine-tuned DiT converted to diffusers naming), ``vae/``
(stock Wan2.2 VAE), ``action_encoder/`` (BWM action encoder), and a
``model_index.json`` with ``_class_name: BoundlessWorldModelPipeline``.
"""

from __future__ import annotations

import logging
import os
from typing import Any, ClassVar

import numpy as np
import PIL.Image
import torch
from diffusers.utils.torch_utils import randn_tensor
from torch import nn

from vllm_omni.diffusion.data import DiffusionOutput, OmniDiffusionConfig
from vllm_omni.diffusion.distributed.autoencoders.autoencoder_kl_wan import DistributedAutoencoderKLWan
from vllm_omni.diffusion.distributed.utils import get_local_device
from vllm_omni.diffusion.forward_context import set_forward_context_denoise_step_idx
from vllm_omni.diffusion.model_loader.diffusers_loader import DiffusersPipelineLoader
from vllm_omni.diffusion.model_loader.hub_prefetch import from_pretrained_with_prefetch, prefetch_subfolders
from vllm_omni.diffusion.models.bwm.bwm_action_encoder import BWMActionEncoder
from vllm_omni.diffusion.models.bwm.bwm_condition_embedder import BWMConditionEmbedder
from vllm_omni.diffusion.models.interface import SupportImageInput, SupportsComponentDiscovery
from vllm_omni.diffusion.models.progress_bar import ProgressBarMixin
from vllm_omni.diffusion.models.wan2_2.pipeline_wan2_2 import (
    build_wan_scheduler,
    create_transformer_from_config,
    load_transformer_config,
    retrieve_latents,
)
from vllm_omni.diffusion.worker.request_batch import DiffusionRequestBatch

logger = logging.getLogger(__name__)

# BWM release defaults (configs/infer/infer.yaml in the reference repo).
# Release defaults are 57-frame chunks with 9 history frames; the chunk
# length is derived from the action trajectory (see resolve_num_frames).
DEFAULT_NUM_INFERENCE_STEPS = 50
DEFAULT_FLOW_SHIFT = 5.0
DEFAULT_ACTION_DIM = 14


def resolve_num_frames(action_frames: int, requested: int | None) -> int:
    """Resolve the chunk length in pixel frames.

    ``OmniDiffusionSamplingParams.num_frames`` defaults to 1 (an image-model
    default), so a value of ``None`` or ``<= 1`` means the caller did not ask
    for a specific length; in that case derive it from the action trajectory
    (one action per pixel frame, ``1 + 4 * (T_latent - 1)`` alignment).
    Explicit requests are snapped down to the 4n+1 frame grid.
    """
    if requested is None or requested <= 1:
        if action_frames < 1:
            raise ValueError("Action trajectory must contain at least one frame")
        return 1 + 4 * ((action_frames - 1) // 4)
    return 1 + 4 * ((requested - 1) // 4)


def _reject_parallel_config(od_config: OmniDiffusionConfig) -> None:
    """BWM runs single-device only.

    CFG parallel is inapplicable (the model runs without classifier-free
    guidance), and pipeline/sequence/tensor parallel are unsupported because
    the action modulation reaches the condition embedder through module
    state rather than a forward argument, which does not propagate across
    stages the way real inputs do.
    """
    parallel = getattr(od_config, "parallel_config", None)
    if parallel is None:
        return
    for field in ("tensor_parallel_size", "sequence_parallel_size", "pipeline_parallel_size", "cfg_parallel_size"):
        size = int(getattr(parallel, field, 1) or 1)
        if size > 1:
            raise ValueError(f"BoundlessWorldModelPipeline is single-device only; got {field}={size}")


def get_bwm_post_process_func(od_config: OmniDiffusionConfig):
    from diffusers.video_processor import VideoProcessor

    video_processor = VideoProcessor(vae_scale_factor=16)

    def post_process_func(video: torch.Tensor, output_type: str = "np", sampling_params=None):
        if output_type == "latent":
            return video
        return {"video": video_processor.postprocess_video(video, output_type=output_type)}

    return post_process_func


class BoundlessWorldModelPipeline(
    nn.Module,
    SupportImageInput,
    ProgressBarMixin,
    SupportsComponentDiscovery,
):
    """Action-conditioned world-model pipeline for BWM (Wan2.2-TI2V-5B)."""

    _dit_modules: ClassVar[list[str]] = ["transformer"]
    _encoder_modules: ClassVar[list[str]] = []
    _vae_modules: ClassVar[list[str]] = ["vae"]
    _resident_modules: ClassVar[list[str]] = ["action_encoder"]

    # Skip the generic text-prompt warmup: BWM requires history frames and an
    # action trajectory, which the dummy request cannot provide.
    dummy_run_num_frames: ClassVar[int] = 0

    def __init__(self, *, od_config: OmniDiffusionConfig, prefix: str = ""):
        super().__init__()
        self.od_config = od_config
        self.device = get_local_device()
        dtype = getattr(od_config, "dtype", torch.bfloat16)
        _reject_parallel_config(od_config)

        model = od_config.model
        local_files_only = os.path.exists(model)

        self.weights_sources = [
            DiffusersPipelineLoader.ComponentSource(
                model_or_path=model,
                subfolder="transformer",
                revision=None,
                prefix="transformer.",
                fall_back_to_pt=True,
            ),
            DiffusersPipelineLoader.ComponentSource(
                model_or_path=model,
                subfolder="action_encoder",
                revision=None,
                prefix="action_encoder.",
                fall_back_to_pt=True,
            ),
        ]

        subfolders = ["vae"]
        prefetch_subfolders(model, subfolders, local_files_only=local_files_only)
        self.vae = from_pretrained_with_prefetch(
            DistributedAutoencoderKLWan.from_pretrained,
            model,
            subfolder="vae",
            prefetch_list=subfolders,
            local_files_only=local_files_only,
            torch_dtype=dtype,
        ).to(self.device)

        transformer_config = load_transformer_config(model, "transformer", local_files_only)
        self.transformer = create_transformer_from_config(
            transformer_config,
            quant_config=od_config.quantization_config,
        )
        # Swap the condition embedder's class in place: identical parameters
        # (checkpoint loading unchanged), BWM forward semantics.
        self.transformer.condition_embedder.__class__ = BWMConditionEmbedder
        self.transformer.condition_embedder.action_mod_emb = None

        model_config = getattr(od_config, "model_config", None) or {}
        action_dim = int(model_config.get("action_dim", DEFAULT_ACTION_DIM))
        # Explicit dtype cast (mirrors the VAE): the loader constructs the
        # pipeline under set_default_torch_dtype(od_config.dtype), but the
        # encoder must match the transformer even without that context.
        self.action_encoder = BWMActionEncoder(
            action_dim=action_dim,
            dim=int(
                transformer_config.get("num_attention_heads", 24) * transformer_config.get("attention_head_dim", 128)
            ),
        ).to(dtype)

        self._flow_shift = od_config.flow_shift if od_config.flow_shift is not None else DEFAULT_FLOW_SHIFT
        self.scheduler = build_wan_scheduler("euler", self._flow_shift)

        self.vae_scale_factor_temporal = self.vae.config.scale_factor_temporal if hasattr(self.vae, "config") else 4
        self.vae_scale_factor_spatial = self.vae.config.scale_factor_spatial if hasattr(self.vae, "config") else 16

        self._num_timesteps = None
        self._current_timestep = None

    # ------------------------------------------------------------ inputs

    @staticmethod
    def _history_frames_to_tensor(raw: Any, height: int | None = None, width: int | None = None) -> torch.Tensor:
        """Return ``(1, C, T, H, W)`` float tensor in [-1, 1].

        Value contract by input dtype: integer arrays are treated as
        [0, 255]; float arrays with values in [0, 1] are rescaled to
        [-1, 1]; other float arrays are assumed to already be in [-1, 1].
        When ``height``/``width`` differ from the frame size, frames are
        resized (bilinear, antialiased), mirroring the reference repo's
        resize-to-render-resolution step.
        """
        if isinstance(raw, PIL.Image.Image):
            raw = [raw]
        if isinstance(raw, (list, tuple)):
            frames = [np.asarray(f.convert("RGB") if isinstance(f, PIL.Image.Image) else f) for f in raw]
            raw = np.stack(frames, axis=0)
        if isinstance(raw, np.ndarray):
            raw = torch.from_numpy(raw)
        if not isinstance(raw, torch.Tensor):
            raise TypeError(f"Unsupported history video type {type(raw)}")
        if raw.ndim != 4:
            raise ValueError(f"History video must be (T, H, W, C), got shape {tuple(raw.shape)}")

        is_integer = not raw.is_floating_point()
        video = raw.permute(3, 0, 1, 2).unsqueeze(0).float()  # (1, C, T, H, W)
        if is_integer:
            video = video / 127.5 - 1.0
        elif video.min() >= 0.0 and video.max() <= 1.0:
            video = video * 2.0 - 1.0

        if height is not None and width is not None and video.shape[-2:] != (height, width):
            frames_2d = video.squeeze(0).permute(1, 0, 2, 3)  # (T, C, H, W)
            frames_2d = torch.nn.functional.interpolate(
                frames_2d, size=(height, width), mode="bilinear", align_corners=False, antialias=True
            )
            video = frames_2d.permute(1, 0, 2, 3).unsqueeze(0)
        return video

    # ------------------------------------------------------------ latents

    def prepare_latents(
        self,
        history_video: torch.Tensor,
        num_frames: int,
        height: int,
        width: int,
        dtype: torch.dtype,
        device: torch.device,
        generator: torch.Generator | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Noise latents, zero-padded history condition, and denoise mask.

        The mask is 0 on the history latent frames (pinned at timestep 0,
        expand-timesteps style) and 1 on frames to denoise.
        """
        num_latent_frames = (num_frames - 1) // self.vae_scale_factor_temporal + 1
        latent_height = height // self.vae_scale_factor_spatial
        latent_width = width // self.vae_scale_factor_spatial
        num_channels_latents = self.transformer.config.out_channels

        shape = (1, num_channels_latents, num_latent_frames, latent_height, latent_width)
        latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)

        history_video = history_video.to(device=device, dtype=self.vae.dtype)
        latent_condition = retrieve_latents(self.vae.encode(history_video), sample_mode="argmax")

        latents_mean = (
            torch.tensor(self.vae.config.latents_mean)
            .view(1, self.vae.config.z_dim, 1, 1, 1)
            .to(latent_condition.device, latent_condition.dtype)
        )
        latents_std_inv = 1.0 / torch.tensor(self.vae.config.latents_std).view(1, self.vae.config.z_dim, 1, 1, 1).to(
            latent_condition.device, latent_condition.dtype
        )
        latent_condition = ((latent_condition - latents_mean) * latents_std_inv).to(dtype)

        history_t = min(int(latent_condition.shape[2]), num_latent_frames)
        condition = torch.zeros_like(latents)
        condition[:, :, :history_t] = latent_condition[:, :, :history_t]

        mask = torch.ones(1, 1, num_latent_frames, latent_height, latent_width, dtype=dtype, device=device)
        mask[:, :, :history_t] = 0
        return latents, condition, mask

    # ------------------------------------------------------------ forward

    def forward(self, req: DiffusionRequestBatch) -> DiffusionOutput:
        if len(req.prompts) != 1:
            raise ValueError("BWM supports a single request at a time.")
        prompt = req.prompts[0]
        multi_modal_data = prompt.get("multi_modal_data", {}) if not isinstance(prompt, str) else {}

        raw_video = multi_modal_data.get("video", multi_modal_data.get("image"))
        if raw_video is None:
            raise ValueError(
                "BWM requires history frames: "
                '`"multi_modal_data": {"video": <(T,H,W,C) array or list of PIL images>, ...}`'
            )
        raw_action = multi_modal_data.get("action")
        if raw_action is None:
            extra_args = getattr(req.sampling_params, "extra_args", None) or {}
            raw_action = extra_args.get("action")
        if raw_action is None:
            raise ValueError(
                "BWM requires a normalized action trajectory: "
                '`"multi_modal_data": {"action": <(frames, action_dim) array>, ...}`'
            )

        action = torch.as_tensor(np.asarray(raw_action), dtype=torch.float32)
        if action.ndim == 2:
            action = action.unsqueeze(0)  # (1, frames, action_dim)
        if action.ndim != 3 or action.shape[-1] != self.action_encoder.action_dim:
            raise ValueError(
                f"Action must be (frames, {self.action_encoder.action_dim}), got shape {tuple(action.shape)}"
            )

        history_video = self._history_frames_to_tensor(raw_video)
        height = req.sampling_params.height or history_video.shape[-2]
        width = req.sampling_params.width or history_video.shape[-1]
        num_steps = req.sampling_params.num_inference_steps or DEFAULT_NUM_INFERENCE_STEPS
        output_type = req.sampling_params.output_type or "np"

        mod_value = self.vae_scale_factor_spatial * 2  # spatial compression x patch size
        if height % mod_value != 0 or width % mod_value != 0:
            raise ValueError(f"height/width must be divisible by {mod_value}, got {height}x{width}")
        if history_video.shape[-2:] != (height, width):
            history_video = self._history_frames_to_tensor(raw_video, height=height, width=width)

        num_frames = resolve_num_frames(int(action.shape[1]), req.sampling_params.num_frames)
        num_latent_frames = (num_frames - 1) // self.vae_scale_factor_temporal + 1

        # One action per pixel frame, aligned to the latent grouping.
        target_action_frames = 1 + 4 * (num_latent_frames - 1)
        if action.shape[1] > target_action_frames:
            action = action[:, :target_action_frames]
        elif action.shape[1] < target_action_frames:
            raise ValueError(
                f"Action sequence too short: got {action.shape[1]} frames, need {target_action_frames} "
                f"for {num_frames} video frames"
            )

        device = self.device
        dtype = self.transformer.dtype

        generator = req.sampling_params.generator
        if generator is None and req.sampling_params.seed is not None:
            generator = torch.Generator(device=device).manual_seed(req.sampling_params.seed)

        action = action.to(device=device, dtype=dtype)
        action_context_emb, action_mod_emb = self.action_encoder(action)

        self.scheduler.set_timesteps(num_steps, device=device)
        timesteps = self.scheduler.timesteps
        self._num_timesteps = len(timesteps)

        latents, condition, mask = self.prepare_latents(
            history_video=history_video,
            num_frames=num_frames,
            height=height,
            width=width,
            dtype=torch.float32,
            device=device,
            generator=generator,
        )

        self.transformer.condition_embedder.action_mod_emb = action_mod_emb
        try:
            with self.progress_bar(total=len(timesteps)) as pbar:
                for step_idx, t in enumerate(timesteps):
                    self._current_timestep = t
                    set_forward_context_denoise_step_idx(step_idx)

                    latent_model_input = ((1 - mask) * condition + mask * latents).to(dtype)
                    # Per-token timesteps: 0 on history tokens (2x2 spatial patch).
                    temp_ts = (mask[0][0][:, ::2, ::2] * t).flatten()
                    timestep = temp_ts.unsqueeze(0).expand(latents.shape[0], -1)

                    noise_pred = self.transformer(
                        hidden_states=latent_model_input,
                        timestep=timestep,
                        encoder_hidden_states=action_context_emb,
                        return_dict=False,
                    )[0]

                    latents = self.scheduler.step(noise_pred, t, latents, return_dict=False)[0]
                    pbar.update()
        finally:
            self.transformer.condition_embedder.action_mod_emb = None
        self._current_timestep = None

        latents = (1 - mask) * condition + mask * latents

        if output_type == "latent":
            return DiffusionOutput(output=latents)

        latents = latents.to(self.vae.dtype)
        latents_mean = (
            torch.tensor(self.vae.config.latents_mean)
            .view(1, self.vae.config.z_dim, 1, 1, 1)
            .to(latents.device, latents.dtype)
        )
        latents_std_inv = 1.0 / torch.tensor(self.vae.config.latents_std).view(1, self.vae.config.z_dim, 1, 1, 1).to(
            latents.device, latents.dtype
        )
        latents = latents / latents_std_inv + latents_mean
        output = self.vae.decode(latents, return_dict=False)[0]
        return DiffusionOutput(output=output)

    def load_weights(self, weights):
        from vllm.model_executor.models.utils import AutoWeightsLoader

        return AutoWeightsLoader(self).load_weights(weights)
