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
from diffusers.video_processor import VideoProcessor
from PIL import Image
from torch import nn
from torch.nn import functional as F

from vllm_omni.diffusion.data import DiffusionOutput, OmniDiffusionConfig
from vllm_omni.diffusion.distributed.utils import get_local_device
from vllm_omni.diffusion.model_loader.diffusers_loader import DiffusersPipelineLoader
from vllm_omni.diffusion.models.seedvr2.config import validate_seedvr2_config
from vllm_omni.diffusion.models.seedvr2.nadit import SEEDVR2_3B_CONFIG, SeedVR2NaDiT
from vllm_omni.diffusion.models.seedvr2.vae import SeedVR2VAE
from vllm_omni.diffusion.models.seedvr2.video import (
    MAX_CLIP_PIXELS,
    MAX_FRAME_PIXELS,
    MAX_SP4_CLIP_PIXELS,
    MAX_SP4_FRAME_PIXELS,
    SourceVideo,
    read_video,
    validate_clip_size,
)
from vllm_omni.diffusion.request import OmniDiffusionRequest
from vllm_omni.diffusion.worker.request_batch import DiffusionRequestBatch
from vllm_omni.errors import OmniClientError
from vllm_omni.model_executor.model_loader.weight_utils import download_weights_from_hf_specific


def prepare_video(frames: torch.Tensor, height: int, width: int) -> torch.Tensor:
    """Resize RGB TCHW to the model grid and repeat the final temporal frame."""
    if frames.ndim != 4 or frames.shape[1] != 3 or min(frames.shape) < 1:
        raise OmniClientError("SeedVR2 requires nonempty RGB frames with shape [T,3,H,W]")
    if min(height, width) < 16 or height % 16 or width % 16:
        raise OmniClientError("SeedVR2 output dimensions must be positive multiples of 16")
    if not frames.is_floating_point() or not torch.isfinite(frames).all():
        raise OmniClientError("SeedVR2 frames must be finite floating-point RGB values")
    if torch.any(frames < 0) or torch.any(frames > 1):
        raise OmniClientError("SeedVR2 RGB values must be in [0,1]")
    frames = F.interpolate(frames.float(), size=(height, width), mode="bicubic", align_corners=False, antialias=True)
    frames = frames.clamp(0, 1)
    padding = (1 - frames.shape[0]) % 4
    if padding:
        frames = torch.cat((frames, frames[-1:].expand(padding, -1, -1, -1)))
    return (frames * 2 - 1).permute(1, 0, 2, 3).unsqueeze(0).contiguous()


@dataclass(frozen=True)
class SeedVR2Input:
    sample: torch.Tensor
    frame_count: int
    source: SourceVideo | None = None


def _admission_budget(config: OmniDiffusionConfig) -> tuple[int, int]:
    parallel = config.parallel_config
    if config.vae_use_tiling and parallel.ulysses_degree == parallel.vae_patch_parallel_size == 4:
        return MAX_SP4_FRAME_PIXELS, MAX_SP4_CLIP_PIXELS
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
        frames = frames.permute(0, 3, 1, 2).float() / 255
    if not isinstance(frames, torch.Tensor):
        raise OmniClientError("SeedVR2 video must be a TCHW RGB tensor or a list of PIL frames")
    if frames.ndim == 4:
        validate_clip_size(frames.shape[0], frames.shape[2], frames.shape[3], frame_pixels, clip_pixels)
        validate_clip_size(frames.shape[0], params.height, params.width, frame_pixels, clip_pixels)
    if (prompt.get("prompt") or "").strip():
        raise OmniClientError("SeedVR2 uses fixed checkpoint conditioning; prompt text is unsupported")
    if params.num_outputs_per_prompt != 1:
        raise OmniClientError("SeedVR2 produces one restored video per request")
    request.prepared_layout = SeedVR2Input(prepare_video(frames, params.height, params.width), frames.shape[0], source)
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
    assert isinstance(video, torch.Tensor)
    frames = VideoProcessor(vae_scale_factor=8).postprocess_video(
        video, output_type="np", do_denormalize=[False] * video.shape[2]
    )
    return {"payload": {**payload, "video": frames}, "metadata": output["metadata"]}


def get_seedvr2_post_process_func(_od_config: OmniDiffusionConfig) -> Callable[[dict[str, object]], dict[str, object]]:
    return _seedvr2_post_process


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
            sample = prepared.sample.to(self.device, dtype=torch.float16)
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
                ulysses=self.od_config.parallel_config.ulysses_degree > 1,
            )
            velocity = self.transformer(video, self.text, shape, text_shape, timestep, runtime).vid_sample
            # Reference Euler returns fp32, then VAE casts to fp16 before scaling.
            restored = noise - velocity.reshape_as(noise)
            decoded = self.vae.decode((restored / 0.9152).permute(3, 0, 1, 2).unsqueeze(0))
            decoded = ((decoded[:, :, : prepared.frame_count].float() + 1) / 2).clamp(0, 1)
            payload: dict[str, object] = {"video": decoded}
            fps = prepared.source.fps if prepared.source is not None else params.fps
            metadata: dict[str, object] = {"video": {"fps": fps}} if fps is not None else {}
            if prepared.source is not None and prepared.source.audio is not None:
                payload["audio"] = prepared.source.audio
                metadata["audio"] = {"sample_rate": prepared.source.audio_sample_rate}
            outputs.append(DiffusionOutput(output={"payload": payload, "metadata": metadata}))
        return outputs
