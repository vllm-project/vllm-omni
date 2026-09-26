# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Native SeedVR2 3B whole-clip restoration (one Euler step, CFG=1)."""

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import ClassVar

import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.nn import functional as F

from vllm_omni.diffusion.data import DiffusionOutput, OmniDiffusionConfig
from vllm_omni.diffusion.distributed.utils import get_local_device
from vllm_omni.diffusion.model_loader.diffusers_loader import DiffusersPipelineLoader
from vllm_omni.diffusion.models.seedvr2.color_fix import (
    COLOR_CORRECTION_METHODS,
    DEFAULT_COLOR_CORRECTION_METHOD,
    correct_video_color,
)
from vllm_omni.diffusion.models.seedvr2.config import validate_seedvr2_config
from vllm_omni.diffusion.models.seedvr2.nadit import SEEDVR2_3B_CONFIG, SeedVR2NaDiT
from vllm_omni.diffusion.models.seedvr2.vae import SeedVR2VAE
from vllm_omni.diffusion.models.seedvr2.video import (
    MAX_CLIP_PIXELS,
    MAX_FRAME_PIXELS,
    SourceVideo,
    read_video,
    sharded_budget,
    validate_clip_size,
)
from vllm_omni.diffusion.request import OmniDiffusionRequest
from vllm_omni.diffusion.worker.request_batch import DiffusionRequestBatch
from vllm_omni.errors import OmniClientError
from vllm_omni.model_executor.model_loader.weight_utils import download_weights_from_hf_specific


def _check_frames(frames: torch.Tensor, height: int, width: int) -> None:
    """Validate host frames: uint8 ``[T,H,W,3]`` or float ``[T,3,H,W]`` in ``[0,1]``."""
    rgb_axis = 3 if frames.dtype == torch.uint8 else 1
    if frames.ndim != 4 or frames.shape[rgb_axis] != 3 or min(frames.shape) < 1:
        raise OmniClientError("SeedVR2 requires nonempty RGB frames with shape [T,3,H,W]")
    if min(height, width) < 16 or height % 16 or width % 16:
        raise OmniClientError("SeedVR2 output dimensions must be positive multiples of 16")
    if frames.dtype == torch.uint8:
        return
    if not frames.is_floating_point() or not torch.isfinite(frames).all():
        raise OmniClientError("SeedVR2 frames must be finite floating-point RGB values")
    if torch.any(frames < 0) or torch.any(frames > 1):
        raise OmniClientError("SeedVR2 RGB values must be in [0,1]")


def prepare_video(frames: torch.Tensor, height: int, width: int, device: torch.device) -> torch.Tensor:
    """Resize RGB frames to the model grid on ``device`` and repeat the final frame.

    ``frames`` is uint8 ``[T,H,W,3]`` or float ``[T,3,H,W]`` in ``[0,1]``. Frames
    are converted one at a time, so the device never holds a float copy of the
    whole clip, and returns fp16 ``[1,3,T',H,W]`` in ``[-1,1]`` with ``T' = 4n+1``.
    """
    count = frames.shape[0]
    sample = torch.empty((1, 3, count + (1 - count) % 4, height, width), device=device, dtype=torch.float16)
    for index in range(count):
        frame = frames[index].to(device, non_blocking=True)
        frame = frame.permute(2, 0, 1).float() / 255 if frame.dtype == torch.uint8 else frame.float()
        if frame.shape[-2:] != (height, width):
            frame = F.interpolate(
                frame.unsqueeze(0), size=(height, width), mode="bicubic", align_corners=False, antialias=True
            ).squeeze(0)
        sample[0, :, index] = frame.clamp(0, 1) * 2 - 1
    sample[0, :, count:] = sample[0, :, count - 1 : count]
    return sample


@dataclass(frozen=True)
class SeedVR2Input:
    # uint8 [T,H,W,3] for decoded video and PIL frames, float [T,3,H,W] otherwise;
    # resizing and normalization run on the device in ``prepare_video``.
    frames: torch.Tensor
    frame_count: int
    source: SourceVideo | None = None


def _admission_budget(config: OmniDiffusionConfig) -> tuple[int, int]:
    """Pick the budget for the serving profile; more ranks never admit less."""
    parallel = config.parallel_config
    sharded = parallel.ulysses_degree == parallel.vae_patch_parallel_size
    if config.vae_use_tiling and sharded and parallel.ulysses_degree >= 4:
        return sharded_budget()
    return MAX_FRAME_PIXELS, MAX_CLIP_PIXELS


def prepare_request(
    request: OmniDiffusionRequest,
    *,
    frame_pixels: int = MAX_FRAME_PIXELS,
    clip_pixels: int = MAX_CLIP_PIXELS,
) -> OmniDiffusionRequest:
    params = request.sampling_params
    if params.num_inference_steps not in (None, 1) or params.guidance_scale != 1.0:
        raise OmniClientError("SeedVR2 whole-clip restoration requires one Euler step and guidance_scale=1")
    method = params.extra_args.get("color_correction_method")
    if method is not None and method not in COLOR_CORRECTION_METHODS:
        raise OmniClientError(f"SeedVR2 color_correction_method must be one of {list(COLOR_CORRECTION_METHODS)}")
    prompt = request.prompt
    if not isinstance(prompt, dict) or "multi_modal_data" not in prompt:
        raise OmniClientError("SeedVR2 requires multi_modal_data.video")
    if params.height is None or params.width is None:
        raise OmniClientError("SeedVR2 requires explicit output height and width")
    if min(params.height, params.width) < 16 or params.height % 16 or params.width % 16:
        raise OmniClientError("SeedVR2 output dimensions must be positive multiples of 16")
    validate_clip_size(1, params.height, params.width, frame_pixels, clip_pixels)
    frames = prompt["multi_modal_data"].get("video")
    source = None
    if isinstance(frames, list) and len(frames) == 1 and isinstance(frames[0], (str, Path)):
        frames = frames[0]
    if isinstance(frames, (str, Path)):
        frames, source = read_video(frames, frame_pixels, clip_pixels)
        if params.fps is not None and abs(params.fps - source.fps) > 1e-6:
            raise OmniClientError("SeedVR2 preserves source FPS; omit fps or match the input frame rate")
    if isinstance(frames, list) and frames and all(isinstance(frame, Image.Image) for frame in frames):
        for frame in frames:
            validate_clip_size(len(frames), frame.height, frame.width, frame_pixels, clip_pixels)
        frames = torch.stack([torch.from_numpy(np.array(frame.convert("RGB"))) for frame in frames])
    if not isinstance(frames, torch.Tensor):
        raise OmniClientError("SeedVR2 video must be a TCHW RGB tensor or a list of PIL frames")
    if frames.ndim == 4:
        frame_height, frame_width = frames.shape[1:3] if frames.dtype == torch.uint8 else frames.shape[2:]
        validate_clip_size(frames.shape[0], frame_height, frame_width, frame_pixels, clip_pixels)
        validate_clip_size(frames.shape[0], params.height, params.width, frame_pixels, clip_pixels)
    _check_frames(frames, params.height, params.width)
    if (prompt.get("prompt") or "").strip():
        raise OmniClientError("SeedVR2 uses fixed checkpoint conditioning; prompt text is unsupported")
    if params.num_outputs_per_prompt != 1:
        raise OmniClientError("SeedVR2 produces one restored video per request")
    request.prepared_layout = SeedVR2Input(frames, frames.shape[0], source)
    return request


def get_seedvr2_pre_process_func(
    od_config: OmniDiffusionConfig,
) -> Callable[[OmniDiffusionRequest], OmniDiffusionRequest]:
    frame_pixels, clip_pixels = _admission_budget(od_config)
    return partial(prepare_request, frame_pixels=frame_pixels, clip_pixels=clip_pixels)


def _seedvr2_post_process(output: dict[str, object]) -> dict[str, object]:
    payload = output["payload"]
    assert isinstance(payload, dict)
    video = payload["video"]
    assert isinstance(video, torch.Tensor) and video.dtype == torch.uint8
    # The device already produced uint8 [B,T,H,W,3], the layout encoders take.
    return {"payload": {**payload, "video": video.cpu().numpy()}, "metadata": output["metadata"]}


def get_seedvr2_post_process_func(_od_config: OmniDiffusionConfig) -> Callable[[dict[str, object]], dict[str, object]]:
    return _seedvr2_post_process


def finish_video(decoded: torch.Tensor, sample: torch.Tensor, frame_count: int, method: str) -> torch.Tensor:
    """Colour-correct and quantize decoded ``[B,3,T,H,W]`` frames to uint8 ``[B,T,H,W,3]``.

    One frame at a time: whole-clip float copies would dominate device memory on
    long clips, and a per-frame slice of one would need 64-bit indexing once the
    clip passes 2**31 pixels.
    """
    batch_size, _, _, height, width = decoded.shape
    video = torch.empty((batch_size, frame_count, height, width, 3), device=decoded.device, dtype=torch.uint8)
    for index in range(frame_count):
        frame = slice(index, index + 1)
        # Restoration shifts global colour, so the resized input carries the
        # reference colour back onto the restored detail.
        corrected = correct_video_color(
            ((decoded[:, :, frame].float() + 1) / 2).clamp(0, 1),
            ((sample[:, :, frame].float() + 1) / 2).clamp(0, 1),
            method=method,
        )
        # Quantize on the device, matching the encoders' rint(clip(x) * 255), so
        # the host copy and the IPC payload are a quarter of the float size.
        video[:, index] = (corrected[:, :, 0].clamp(0, 1) * 255).round_().permute(0, 2, 3, 1)
    return video


def sample_noise(condition: torch.Tensor, generator: torch.Generator) -> torch.Tensor:
    # Reference randn_like preserves latent strides; RNG values follow storage order.
    return torch.empty_like(condition).normal_(generator=generator)


class SeedVR2Pipeline(nn.Module):
    supports_request_batch = True
    _dit_modules: ClassVar[list[str]] = ["transformer"]
    _vae_modules: ClassVar[list[str]] = ["vae"]
    _encoder_modules: ClassVar[list[str]] = []

    def __init__(self, *, od_config: OmniDiffusionConfig, prefix: str = "") -> None:
        super().__init__()
        validate_seedvr2_config(od_config)
        self.device = get_local_device()
        self.od_config = od_config
        self.frame_pixels, self.clip_pixels = _admission_budget(od_config)
        self.transformer = SeedVR2NaDiT(**SEEDVR2_3B_CONFIG, use_varlen_kernel=False)
        self.vae = SeedVR2VAE()
        self.weights_sources = [
            DiffusersPipelineLoader.ComponentSource(
                model_or_path=od_config.model,
                subfolder=None,
                revision=od_config.revision,
                prefix=component + ".",
                fall_back_to_pt=False,
                allow_patterns_overrides=[filename],
            )
            for component, filename in (
                ("transformer", "seedvr2_ema_3b_fp16.safetensors"),
                ("vae", "ema_vae_fp16.safetensors"),
            )
        ]
        model_path = Path(od_config.model)
        if not model_path.is_dir():
            model_path = Path(
                download_weights_from_hf_specific(od_config.model, None, ["pos_emb.pt"], revision=od_config.revision)
            )
        text = torch.load(model_path / "pos_emb.pt", map_location="cpu", weights_only=True)
        if not isinstance(text, torch.Tensor) or text.ndim != 2 or text.shape[1] != 5120 or text.shape[0] == 0:
            raise ValueError("SeedVR2 pos_emb.pt must contain a tensor of shape [L,5120]")
        self.register_buffer("text", text.to(device=self.device, dtype=od_config.dtype), persistent=False)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        targets = self.state_dict()
        loaded: set[str] = set()
        for name, weight in weights:
            if name.endswith(".rope.rope.freqs"):
                name = name.removesuffix(".rope.rope.freqs") + ".rope.freqs"
            if name not in targets or name in loaded:
                raise ValueError(f"Unexpected or duplicate SeedVR2 checkpoint key: {name}")
            if targets[name].shape != weight.shape:
                raise ValueError(f"SeedVR2 checkpoint shape mismatch for {name}")
            targets[name].copy_(weight)
            loaded.add(name)
        missing = targets.keys() - loaded
        if missing:
            raise ValueError(f"Missing SeedVR2 checkpoint keys: {sorted(missing)}")
        return loaded

    @torch.inference_mode()
    def forward(self, batch: DiffusionRequestBatch) -> list[DiffusionOutput]:
        outputs = []
        for request in batch.requests:
            params = request.sampling_params
            if not isinstance(request.prepared_layout, SeedVR2Input):
                prepare_request(request, frame_pixels=self.frame_pixels, clip_pixels=self.clip_pixels)
            prepared = request.prepared_layout
            sample = prepare_video(prepared.frames, params.height, params.width, self.device)
            generator = params.generator
            if generator is None:
                generator = torch.Generator(device=self.device).manual_seed(params.seed)
            if not isinstance(generator, torch.Generator):
                raise ValueError("SeedVR2 accepts one generator per request")
            latent = self.vae.encode(sample).sample(generator=generator)
            condition = latent.permute(0, 2, 3, 4, 1).squeeze(0) * 0.9152
            noise = sample_noise(condition, generator)
            video = torch.cat((noise, condition, torch.ones_like(condition[..., :1])), dim=-1).reshape(-1, 33)
            shape = torch.tensor([condition.shape[:3]], device=self.device, dtype=torch.long)
            text_shape = torch.tensor([[self.text.shape[0]]], device=self.device, dtype=torch.long)
            timestep = torch.tensor([1000.0], device=self.device, dtype=torch.float16)
            runtime = self.transformer.build_runtime(
                self.transformer.token_grid_for(shape),
                text_len=self.text.shape[0],
                parallel_config=self.od_config.parallel_config,
            )
            velocity = self.transformer(video, self.text, shape, text_shape, timestep, runtime).vid_sample
            # Reference Euler returns fp32, then VAE casts to fp16 before scaling.
            restored = noise - velocity.reshape_as(noise)
            decoded = self.vae.decode((restored / 0.9152).permute(3, 0, 1, 2).unsqueeze(0))
            method = params.extra_args.get("color_correction_method") or DEFAULT_COLOR_CORRECTION_METHOD
            video = finish_video(decoded, sample, prepared.frame_count, method)
            payload: dict[str, object] = {"video": video}
            fps = prepared.source.fps if prepared.source is not None else params.fps
            metadata: dict[str, object] = {"video": {"fps": fps}} if fps is not None else {}
            if prepared.source is not None and prepared.source.audio is not None:
                payload["audio"] = prepared.source.audio
                metadata["audio"] = {"sample_rate": prepared.source.audio_sample_rate}
            outputs.append(DiffusionOutput(output={"payload": payload, "metadata": metadata}))
        return outputs
