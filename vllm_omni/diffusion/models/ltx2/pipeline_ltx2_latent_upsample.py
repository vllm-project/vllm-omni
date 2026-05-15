import os

import torch
import torch.nn as nn
from diffusers import AutoencoderKLLTX2Video
from diffusers.image_processor import PipelineImageInput
from diffusers.pipelines.ltx2.latent_upsampler import LTX2LatentUpsamplerModel
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion_img2img import retrieve_latents
from diffusers.utils.torch_utils import randn_tensor
from diffusers.video_processor import VideoProcessor
from vllm.logger import init_logger

from vllm_omni.diffusion.data import DiffusionOutput, OmniDiffusionConfig
from vllm_omni.diffusion.distributed.utils import get_local_device

logger = init_logger(__name__)


class LTX2LatentUpsamplePipeline(nn.Module):
    def __init__(
        self,
        od_config: OmniDiffusionConfig,
        vae: AutoencoderKLLTX2Video,
        latent_upsampler: LTX2LatentUpsamplerModel = None,
    ) -> None:
        super().__init__()

        if vae is None:
            raise ValueError("vae must be provided")
        self.vae = vae

        self.device = get_local_device()
        model = od_config.model
        local_files_only = os.path.exists(model)

        if latent_upsampler is None:
            # Use cpu context to create latent upsampler. The code k[:, None] @ k[None, :] in
            # diffuser's BlurDownsample is not supported on GPU as k is type of torch.Int64
            with torch.device("cpu"):
                latent_upsampler = LTX2LatentUpsamplerModel.from_pretrained(
                    model,
                    subfolder="latent_upsampler",
                    torch_dtype=torch.bfloat16,
                    local_files_only=local_files_only,
                ).to(self.device)
        self.latent_upsampler = latent_upsampler

        self.vae_spatial_compression_ratio = (
            self.vae.spatial_compression_ratio if getattr(self, "vae", None) is not None else 32
        )
        self.vae_temporal_compression_ratio = (
            self.vae.temporal_compression_ratio if getattr(self, "vae", None) is not None else 8
        )
        self.video_processor = VideoProcessor(vae_scale_factor=self.vae_spatial_compression_ratio)

    def prepare_latents(
        self,
        video: torch.Tensor | None = None,
        batch_size: int = 1,
        num_frames: int = 121,
        height: int = 512,
        width: int = 768,
        spatial_patch_size: int = 1,
        temporal_patch_size: int = 1,
        dtype: torch.dtype | None = None,
        device: torch.device | None = None,
        generator: torch.Generator | None = None,
        latents: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if latents is not None:
            if latents.ndim == 3:
                # Convert token seq [B, S, D] to latent video [B, C, F, H, W]
                latent_num_frames = (num_frames - 1) // self.vae_temporal_compression_ratio + 1
                latent_height = height // self.vae_spatial_compression_ratio
                latent_width = width // self.vae_spatial_compression_ratio
                latents = self._unpack_latents(
                    latents, latent_num_frames, latent_height, latent_width, spatial_patch_size, temporal_patch_size
                )
            return latents.to(device=device, dtype=dtype)

        video = video.to(device=device, dtype=self.vae.dtype)
        if isinstance(generator, list):
            if len(generator) != batch_size:
                raise ValueError(
                    f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                    f" size of {batch_size}. Make sure the batch size matches the length of the generators."
                )

            init_latents = [
                retrieve_latents(self.vae.encode(video[i].unsqueeze(0)), generator[i]) for i in range(batch_size)
            ]
        else:
            init_latents = [retrieve_latents(self.vae.encode(vid.unsqueeze(0)), generator) for vid in video]

        init_latents = torch.cat(init_latents, dim=0).to(dtype)
        # NOTE: latent upsampler operates on the unnormalized latents, so don't normalize here
        # init_latents = self._normalize_latents(init_latents, self.vae.latents_mean, self.vae.latents_std)
        return init_latents

    def adain_filter_latent(self, latents: torch.Tensor, reference_latents: torch.Tensor, factor: float = 1.0):
        result = latents.clone()

        for i in range(latents.size(0)):
            for c in range(latents.size(1)):
                r_sd, r_mean = torch.std_mean(reference_latents[i, c], dim=None)  # index by original dim order
                i_sd, i_mean = torch.std_mean(result[i, c], dim=None)

                result[i, c] = ((result[i, c] - i_mean) / i_sd) * r_sd + r_mean

        result = torch.lerp(latents, result, factor)
        return result

    def tone_map_latents(self, latents: torch.Tensor, compression: float) -> torch.Tensor:
        # Remap [0-1] to [0-0.75] and apply sigmoid compression in one shot
        scale_factor = compression * 0.75
        abs_latents = torch.abs(latents)

        # Sigmoid compression: sigmoid shifts large values toward 0.2, small values stay ~1.0
        # When scale_factor=0, sigmoid term vanishes, when scale_factor=0.75, full effect
        sigmoid_term = torch.sigmoid(4.0 * scale_factor * (abs_latents - 1.0))
        scales = 1.0 - 0.8 * scale_factor * sigmoid_term

        filtered = latents * scales
        return filtered

    @staticmethod
    # Copied from diffusers.pipelines.ltx2.pipeline_ltx2.LTX2Pipeline._denormalize_latents
    def _denormalize_latents(
        latents: torch.Tensor, latents_mean: torch.Tensor, latents_std: torch.Tensor, scaling_factor: float = 1.0
    ) -> torch.Tensor:
        # Denormalize latents across the channel dimension [B, C, F, H, W]
        latents_mean = latents_mean.view(1, -1, 1, 1, 1).to(latents.device, latents.dtype)
        latents_std = latents_std.view(1, -1, 1, 1, 1).to(latents.device, latents.dtype)
        latents = latents * latents_std / scaling_factor + latents_mean
        return latents

    @staticmethod
    # Copied from diffusers.pipelines.ltx2.pipeline_ltx2.LTX2Pipeline._unpack_latents
    def _unpack_latents(
        latents: torch.Tensor, num_frames: int, height: int, width: int, patch_size: int = 1, patch_size_t: int = 1
    ) -> torch.Tensor:
        # Packed latents of shape [B, S, D] (S is the effective video sequence length, D is the effective feature dimensions) # noqa
        # are unpacked and reshaped into a video tensor of shape [B, C, F, H, W]. This is the inverse operation of
        # what happens in the `_pack_latents` method.
        batch_size = latents.size(0)
        latents = latents.reshape(batch_size, num_frames, height, width, -1, patch_size_t, patch_size, patch_size)
        latents = latents.permute(0, 4, 1, 5, 2, 6, 3, 7).flatten(6, 7).flatten(4, 5).flatten(2, 3)
        return latents

    def check_inputs(self, video, height, width, latents, tone_map_compression_ratio):
        if height % self.vae_spatial_compression_ratio != 0 or width % self.vae_spatial_compression_ratio != 0:
            raise ValueError(f"`height` and `width` have to be divisible by 32 but are {height} and {width}.")

        if video is not None and latents is not None:
            raise ValueError("Only one of `video` or `latents` can be provided.")
        if video is None and latents is None:
            raise ValueError("One of `video` or `latents` has to be provided.")

        if not (0 <= tone_map_compression_ratio <= 1):
            raise ValueError("`tone_map_compression_ratio` must be in the range [0, 1]")

    def forward(
        self,
        video: list[PipelineImageInput] | None = None,
        height: int = 512,
        width: int = 768,
        num_frames: int = 121,
        spatial_patch_size: int = 1,
        temporal_patch_size: int = 1,
        latents: torch.Tensor | None = None,
        latents_normalized: bool = False,
        decode_timestep: float | list[float] = 0.0,
        decode_noise_scale: float | list[float] | None = None,
        adain_factor: float = 0.0,
        tone_map_compression_ratio: float = 0.0,
        generator: torch.Generator | list[torch.Generator] | None = None,
        output_type: str | None = "pil",
        return_dict: bool = True,
    ):
        self.check_inputs(
            video=video,
            height=height,
            width=width,
            latents=latents,
            tone_map_compression_ratio=tone_map_compression_ratio,
        )

        if video is not None:
            # Batched video input is not yet tested/supported. TODO: take a look later
            batch_size = 1
        else:
            batch_size = latents.shape[0]
        device = self.device

        if video is not None:
            num_frames = len(video)
            if num_frames % self.vae_temporal_compression_ratio != 1:
                num_frames = num_frames // self.vae_temporal_compression_ratio * self.vae_temporal_compression_ratio + 1
                video = video[:num_frames]
                logger.warning(
                    f"Video length expected to be of the form `k * {self.vae_temporal_compression_ratio} + 1` but is {len(video)}. Truncating to {num_frames} frames."  # noqa
                )
            video = self.video_processor.preprocess_video(video, height=height, width=width)
            video = video.to(device=device, dtype=torch.float32)

        latents_supplied = latents is not None
        latents = self.prepare_latents(
            video=video,
            batch_size=batch_size,
            num_frames=num_frames,
            height=height,
            width=width,
            spatial_patch_size=spatial_patch_size,
            temporal_patch_size=temporal_patch_size,
            dtype=torch.float32,
            device=device,
            generator=generator,
            latents=latents,
        )

        if latents_supplied and latents_normalized:
            latents = self._denormalize_latents(
                latents, self.vae.latents_mean, self.vae.latents_std, self.vae.config.scaling_factor
            )
        latents = latents.to(self.latent_upsampler.dtype)
        latents_upsampled = self.latent_upsampler(latents)

        if adain_factor > 0.0:
            latents = self.adain_filter_latent(latents_upsampled, latents, adain_factor)
        else:
            latents = latents_upsampled

        if tone_map_compression_ratio > 0.0:
            latents = self.tone_map_latents(latents, tone_map_compression_ratio)

        if output_type == "latent":
            video = latents
        else:
            if not self.vae.config.timestep_conditioning:
                timestep = None
            else:
                noise = randn_tensor(latents.shape, generator=generator, device=device, dtype=latents.dtype)
                if not isinstance(decode_timestep, list):
                    decode_timestep = [decode_timestep] * batch_size
                if decode_noise_scale is None:
                    decode_noise_scale = decode_timestep
                elif not isinstance(decode_noise_scale, list):
                    decode_noise_scale = [decode_noise_scale] * batch_size

                timestep = torch.tensor(decode_timestep, device=device, dtype=latents.dtype)
                decode_noise_scale = torch.tensor(decode_noise_scale, device=device, dtype=latents.dtype)[
                    :, None, None, None, None
                ]
                latents = (1 - decode_noise_scale) * latents + decode_noise_scale * noise

            video = self.vae.decode(latents, timestep, return_dict=False)[0]
            video = self.video_processor.postprocess_video(video, output_type=output_type)

        if not return_dict:
            return (video,)

        return DiffusionOutput(output=(video,))
