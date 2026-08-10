# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import logging
import math
import os
from collections.abc import Iterable
from typing import Any, ClassVar

import numpy as np
import PIL.Image
import torch
from diffusers.utils.torch_utils import randn_tensor
from einops import rearrange
from torch import nn
from transformers import Wav2Vec2FeatureExtractor
from vllm.model_executor.models.utils import AutoWeightsLoader
from vllm.multimodal.media.audio import load_audio
from vllm.sequence import IntermediateTensors

from vllm_omni.diffusion.data import DiffusionOutput, OmniDiffusionConfig
from vllm_omni.diffusion.distributed.cfg_parallel import CFGParallelMixin
from vllm_omni.diffusion.distributed.utils import get_local_device
from vllm_omni.diffusion.forward_context import DenoiseProgressMixin
from vllm_omni.diffusion.model_loader.diffusers_loader import DiffusersPipelineLoader
from vllm_omni.diffusion.models.interface import (
    SupportAudioInput,
    SupportImageInput,
    SupportsComponentDiscovery,
)
from vllm_omni.diffusion.models.progress_bar import ProgressBarMixin, _is_rank_zero
from vllm_omni.diffusion.models.skyreels_v3.a2v_avatar_utils import (
    ASPECT_RATIO_627,
    ASPECT_RATIO_960,
    match_and_blend_colors,
)
from vllm_omni.diffusion.models.skyreels_v3.a2v_clip import CLIPModel
from vllm_omni.diffusion.models.skyreels_v3.a2v_t5 import T5EncoderModel
from vllm_omni.diffusion.models.skyreels_v3.a2v_transformer import WanModel
from vllm_omni.diffusion.models.skyreels_v3.a2v_vae import WanVAE
from vllm_omni.diffusion.models.skyreels_v3.a2v_wav2vec2 import Wav2Vec2Model
from vllm_omni.diffusion.profiler.diffusion_pipeline_profiler import (
    DiffusionPipelineProfilerMixin,
)
from vllm_omni.diffusion.request import OmniDiffusionRequest
from vllm_omni.diffusion.worker.request_batch import DiffusionRequestBatch
from vllm_omni.inputs.data import OmniTextPrompt
from vllm_omni.platforms import current_omni_platform

logger = logging.getLogger(__name__)

DEFAULT_SKYREELS_A2V_FPS = 25
DEFAULT_SKYREELS_A2V_FRAMES = 81
DEFAULT_SKYREELS_A2V_STEPS = 40
DEFAULT_SKYREELS_A2V_SHIFT = 11.0
DEFAULT_SKYREELS_A2V_TEXT_GUIDANCE = 1.0
DEFAULT_SKYREELS_A2V_AUDIO_GUIDANCE = 1.0
DEFAULT_SKYREELS_A2V_MAX_FRAMES = 5000
DEFAULT_SKYREELS_A2V_MOTION_FRAME = 5
DEFAULT_SKYREELS_A2V_DROP_FRAME = 12
DEFAULT_SKYREELS_A2V_NEG_PROMPT = (
    "bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, "
    "images, static, overall gray, worst quality, low quality, JPEG compression residue, ugly, "
    "incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, "
    "misshapen limbs, fused fingers, still picture, messy background, three legs, many people in "
    "the background, walking backwards"
)


def _resolve_a2v_model_path(model: str) -> str:
    if os.path.isdir(model):
        return model
    from vllm_omni.model_executor.model_loader.weight_utils import download_weights_from_hf_specific

    allow_patterns = [
        "*.safetensors",
        "*.safetensors.index.json",
        "*.json",
        "*.pth",
        "google/**",
        "xlm-roberta-large/**",
        "chinese-wav2vec2-base/**",
    ]
    return download_weights_from_hf_specific(model, None, allow_patterns)


def _load_image(image: str | PIL.Image.Image) -> PIL.Image.Image:
    if isinstance(image, str):
        return PIL.Image.open(image).convert("RGB")
    if isinstance(image, PIL.Image.Image):
        return image.convert("RGB")
    raise TypeError(f"Unsupported image format {image.__class__}.")


def _normalize_avatar_image(raw_image: Any) -> PIL.Image.Image:
    if raw_image is None:
        raise ValueError(
            "SkyReels V3 A2V requires a portrait image. Set `multi_modal_data` with key `image` or `cond_image`."
        )
    if isinstance(raw_image, list):
        if len(raw_image) != 1:
            raise ValueError(f"SkyReels V3 A2V supports exactly one portrait image, got {len(raw_image)}.")
        raw_image = raw_image[0]
    return _load_image(raw_image)


def _load_audio_array(raw_audio: Any) -> tuple[np.ndarray, int]:
    if raw_audio is None:
        raise ValueError(
            "SkyReels V3 A2V requires driving audio. Set `multi_modal_data` with key `audio` or `cond_audio`."
        )
    if isinstance(raw_audio, tuple) and len(raw_audio) == 2:
        audio, sample_rate = raw_audio
        return np.asarray(audio, dtype=np.float32), int(sample_rate)
    if isinstance(raw_audio, np.ndarray):
        return raw_audio.astype(np.float32), 16000
    if isinstance(raw_audio, str):
        audio, sample_rate = load_audio(raw_audio, sr=16000, mono=True)
        return audio.astype(np.float32), int(sample_rate)
    raise TypeError(f"Unsupported audio format {raw_audio.__class__}.")


def _select_bucket_size(image: PIL.Image.Image, size_bucket: str) -> tuple[int, int]:
    bucket_name = size_bucket.upper()
    if bucket_name == "480P":
        bucket_config = ASPECT_RATIO_627
    elif bucket_name == "720P":
        bucket_config = ASPECT_RATIO_960
    else:
        raise ValueError(f"SkyReels V3 A2V supports `resolution` 480P or 720P, got {size_bucket!r}.")

    ratio = image.height / image.width
    closest_bucket = min(bucket_config, key=lambda key: abs(float(key) - ratio))
    target_h, target_w = bucket_config[closest_bucket][0]
    return int(target_h), int(target_w)


def _resize_and_centercrop_image(image: PIL.Image.Image, height: int, width: int) -> PIL.Image.Image:
    scale = max(height / image.height, width / image.width)
    final_h = math.ceil(scale * image.height)
    final_w = math.ceil(scale * image.width)
    resized = image.resize((final_w, final_h), resample=PIL.Image.Resampling.BILINEAR)
    left = max(0, (final_w - width) // 2)
    top = max(0, (final_h - height) // 2)
    return resized.crop((left, top, left + width, top + height)).convert("RGB")


def _image_to_a2v_tensor(image: PIL.Image.Image, device: torch.device | str) -> torch.Tensor:
    array = np.asarray(image, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(array).permute(2, 0, 1).contiguous()
    tensor = tensor.unsqueeze(0).unsqueeze(2)
    return (tensor - 0.5).mul_(2.0).to(device)


def _validate_audio_duration(audio: np.ndarray, sample_rate: int, max_frames_num: int) -> None:
    duration = len(audio) / float(sample_rate)
    if duration < 0.4:
        raise ValueError(f"Audio duration is too short: {duration:.2f}s. Minimum allowed: 0.4s.")
    max_duration = max_frames_num / DEFAULT_SKYREELS_A2V_FPS
    if duration > max_duration:
        raise ValueError(f"Audio duration is too long: {duration:.2f}s. Maximum allowed: {max_duration:.2f}s.")


def _resolve_guidance_scales(sampling_params: Any, extra_args: dict[str, Any]) -> tuple[float, float]:
    text_scale = (
        sampling_params.guidance_scale
        if sampling_params.guidance_scale_provided
        else DEFAULT_SKYREELS_A2V_TEXT_GUIDANCE
    )
    text_scale = extra_args.get("text_guide_scale", extra_args.get("cfg_text_scale", text_scale))
    audio_scale = extra_args.get(
        "audio_guide_scale",
        extra_args.get("cfg_audio_scale", DEFAULT_SKYREELS_A2V_AUDIO_GUIDANCE),
    )
    return float(text_scale), float(audio_scale)


def _timestep_transform(t: torch.Tensor, shift: float = DEFAULT_SKYREELS_A2V_SHIFT, num_timesteps: int = 1000):
    t = t / num_timesteps
    new_t = shift * t / (1 + (shift - 1) * t)
    return new_t * num_timesteps


def get_skyreels_v3_a2v_post_process_func(
    od_config: OmniDiffusionConfig,
):
    from diffusers.video_processor import VideoProcessor

    video_processor = VideoProcessor(vae_scale_factor=8)

    def post_process_func(
        output: torch.Tensor | tuple[Any, ...],
        output_type: str = "np",
        sampling_params=None,
    ):
        if sampling_params is not None and getattr(sampling_params, "output_type", None):
            output_type = sampling_params.output_type
        if isinstance(output, tuple) and len(output) == 3:
            video, audio_waveform, audio_sample_rate = output
        else:
            video = output
            audio_waveform = None
            audio_sample_rate = None

        if output_type == "latent":
            return video

        video_metadata = {"fps": DEFAULT_SKYREELS_A2V_FPS}
        if sampling_params is not None and getattr(sampling_params, "enable_frame_interpolation", False):
            from vllm_omni.diffusion.postprocess import interpolate_video_tensor

            video, multiplier = interpolate_video_tensor(
                video,
                exp=sampling_params.frame_interpolation_exp,
                scale=sampling_params.frame_interpolation_scale,
                model_path=sampling_params.frame_interpolation_model_path,
            )
            video_metadata["video_fps_multiplier"] = multiplier

        payload: dict[str, Any] = {
            "video": video_processor.postprocess_video(video, output_type=output_type),
        }
        metadata: dict[str, Any] = {"video": video_metadata}
        if audio_waveform is not None and audio_sample_rate is not None:
            payload["audio"] = audio_waveform
            metadata["audio"] = {"sample_rate": audio_sample_rate}
        return {"payload": payload, "metadata": metadata}

    return post_process_func


def get_skyreels_v3_a2v_pre_process_func(
    od_config: OmniDiffusionConfig,
):
    def pre_process_func(request: OmniDiffusionRequest) -> OmniDiffusionRequest:
        prompt = request.prompt
        if isinstance(prompt, str):
            prompt = OmniTextPrompt(prompt=prompt)
        multi_modal_data = prompt.get("multi_modal_data") or {}

        raw_image = multi_modal_data.get("image", multi_modal_data.get("cond_image"))
        image = _normalize_avatar_image(raw_image)

        raw_audio = multi_modal_data.get("audio", multi_modal_data.get("cond_audio"))
        if isinstance(raw_audio, dict):
            raw_audio = raw_audio.get("person1")
        audio, sample_rate = _load_audio_array(raw_audio)

        extra_args = request.sampling_params.extra_args or {}
        max_frames_num = int(extra_args.get("max_frames_num", DEFAULT_SKYREELS_A2V_MAX_FRAMES))
        _validate_audio_duration(audio, sample_rate, max_frames_num)

        if request.sampling_params.height is None or request.sampling_params.width is None:
            bucket = str(extra_args.get("resolution", extra_args.get("size_bucket", "720P")))
            height, width = _select_bucket_size(image, bucket)
            if request.sampling_params.height is None:
                request.sampling_params.height = height
            if request.sampling_params.width is None:
                request.sampling_params.width = width

        height = int(request.sampling_params.height)
        width = int(request.sampling_params.width)
        if height % 16 != 0 or width % 16 != 0:
            raise ValueError(f"`height` and `width` must be divisible by 16, got {height} and {width}.")

        prompt["multi_modal_data"] = dict(multi_modal_data)
        prompt["multi_modal_data"]["image"] = _resize_and_centercrop_image(image, height, width)
        prompt["multi_modal_data"]["audio"] = audio
        prompt.setdefault("additional_information", {})["audio_sample_rate"] = sample_rate
        request.prompt = prompt
        return request

    return pre_process_func


class SkyReelsV3A2VPipeline(
    nn.Module,
    SupportImageInput,
    SupportAudioInput,
    CFGParallelMixin,
    ProgressBarMixin,
    DiffusionPipelineProfilerMixin,
    DenoiseProgressMixin,
    SupportsComponentDiscovery,
):
    _dit_modules: ClassVar[list[str]] = ["transformer"]
    _encoder_modules: ClassVar[list[str]] = ["text_encoder", "clip.model", "audio_encoder"]
    _vae_modules: ClassVar[list[str]] = ["vae.vae"]
    dummy_run_num_frames: ClassVar[int] = 5

    def __init__(
        self,
        *,
        od_config: OmniDiffusionConfig,
        prefix: str = "",
    ):
        super().__init__()
        self.od_config = od_config
        self.device = get_local_device()
        self.param_dtype = getattr(od_config, "dtype", torch.bfloat16)

        self.model_path = _resolve_a2v_model_path(od_config.model)
        self.weights_sources = [
            DiffusersPipelineLoader.ComponentSource(
                model_or_path=self.model_path,
                subfolder=None,
                revision=None,
                prefix="transformer.",
                fall_back_to_pt=False,
            )
        ]

        self.text_encoder = T5EncoderModel(
            text_len=512,
            checkpoint_path=os.path.join(self.model_path, "models_t5_umt5-xxl-enc-bf16.pth"),
            tokenizer_path=os.path.join(self.model_path, "google", "umt5-xxl"),
            shard_fn=None,
        ).to(self.param_dtype)
        self.clip = CLIPModel(
            dtype=torch.float16,
            device=self.device,
            checkpoint_path=os.path.join(
                self.model_path,
                "models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth",
            ),
            tokenizer_path=os.path.join(self.model_path, "xlm-roberta-large"),
        )
        self.vae_stride = (4, 8, 8)
        self.patch_size = (1, 2, 2)
        self.vae = WanVAE(vae_pth=os.path.join(self.model_path, "Wan2.1_VAE.pth"))

        wav2vec_path = os.path.join(self.model_path, "chinese-wav2vec2-base")
        self.audio_feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(
            wav2vec_path,
            local_files_only=os.path.isdir(wav2vec_path),
        )
        self.audio_encoder = Wav2Vec2Model.from_pretrained(
            wav2vec_path,
            local_files_only=os.path.isdir(wav2vec_path),
        ).to(self.device)
        self.audio_encoder.feature_extractor._freeze_parameters()
        self.audio_encoder.eval().requires_grad_(False)

        self.transformer = WanModel.from_config(os.path.join(self.model_path, "config.json")).to(self.param_dtype)

        self.text_encoder.to(self.device)
        self.vae.to(self.device)

        self.num_train_timesteps = 1000
        self.sp_size = 1
        self._guidance_scale = None
        self._audio_guidance_scale = None
        self._num_timesteps = None
        self._current_timestep = None
        self.setup_diffusion_pipeline_profiler(
            enable_diffusion_pipeline_profiler=self.od_config.enable_diffusion_pipeline_profiler
        )

    @property
    def guidance_scale(self):
        return self._guidance_scale

    @property
    def audio_guidance_scale(self):
        return self._audio_guidance_scale

    @property
    def do_classifier_free_guidance(self):
        return (
            self._guidance_scale is not None
            and self._audio_guidance_scale is not None
            and (self._guidance_scale > 1.0 or self._audio_guidance_scale > 1.0)
        )

    @property
    def num_timesteps(self):
        return self._num_timesteps

    @property
    def current_timestep(self):
        return self._current_timestep

    def to(self, *args: Any, **kwargs: Any):
        super().to(*args, **kwargs)
        if hasattr(self, "clip"):
            self.clip.model.to(*args, **kwargs)
        if hasattr(self, "vae"):
            self.vae.to(*args, **kwargs)
        return self

    def combine_multi_branch_cfg_noise(
        self,
        predictions: list[torch.Tensor | tuple[torch.Tensor, ...]],
        true_cfg_scale: float | dict[str, Any],
        cfg_normalize: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, ...]:
        if not isinstance(true_cfg_scale, dict) or true_cfg_scale.get("mode") != "skyreels_v3_a2v":
            return super().combine_multi_branch_cfg_noise(predictions, true_cfg_scale, cfg_normalize)
        if any(isinstance(pred, tuple) for pred in predictions):
            raise ValueError("SkyReels V3 A2V CFG expects tensor predictions.")

        cond = predictions[0]
        text_scale = float(true_cfg_scale["text_guide_scale"])
        audio_scale = float(true_cfg_scale["audio_guide_scale"])
        branch = true_cfg_scale.get("branch")
        if branch == "text_audio":
            if len(predictions) != 3:
                raise ValueError(f"SkyReels V3 A2V text+audio CFG expects 3 branches, got {len(predictions)}.")
            drop_text, uncond = predictions[1], predictions[2]
            combined = uncond + text_scale * (cond - drop_text) + audio_scale * (drop_text - uncond)
        elif branch == "text":
            if len(predictions) != 2:
                raise ValueError(f"SkyReels V3 A2V text CFG expects 2 branches, got {len(predictions)}.")
            drop_text = predictions[1]
            combined = drop_text + text_scale * (cond - drop_text)
        elif branch == "audio":
            if len(predictions) != 2:
                raise ValueError(f"SkyReels V3 A2V audio CFG expects 2 branches, got {len(predictions)}.")
            drop_audio = predictions[1]
            combined = drop_audio + audio_scale * (cond - drop_audio)
        else:
            raise ValueError(f"Unsupported SkyReels V3 A2V CFG branch {branch!r}.")
        if cfg_normalize:
            combined = self.cfg_normalize_function(cond, combined)
        return combined

    def encode_audio(
        self,
        audio: np.ndarray,
        sample_rate: int,
        frame_num: int,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> torch.Tensor:
        device = device or self.device
        dtype = dtype or self.param_dtype
        audio_duration = len(audio) / float(sample_rate)
        video_length = max(1, int(np.ceil(audio_duration * DEFAULT_SKYREELS_A2V_FPS)))

        audio_feature = np.squeeze(self.audio_feature_extractor(audio, sampling_rate=sample_rate).input_values)
        audio_feature_tensor = torch.from_numpy(audio_feature).float().to(device=self.device).unsqueeze(0)
        with torch.no_grad():
            embeddings = self.audio_encoder(
                audio_feature_tensor,
                seq_len=video_length,
                output_hidden_states=True,
            )
        if embeddings.hidden_states is None:
            raise ValueError("Failed to extract SkyReels V3 A2V audio embeddings.")

        audio_emb = torch.stack(embeddings.hidden_states[1:], dim=1).squeeze(0)
        full_audio_emb = rearrange(audio_emb, "b s d -> s b d").detach().cpu()

        indices = torch.arange(2 * 2 + 1) - 2
        center_indices = torch.arange(frame_num).unsqueeze(1) + indices.unsqueeze(0)
        center_indices = torch.clamp(center_indices, min=0, max=full_audio_emb.shape[0] - 1).cpu()
        audio_context = full_audio_emb[center_indices][None, ...].to(device=device, dtype=dtype)
        return audio_context

    def encode_text(
        self,
        prompt: str,
        negative_prompt: str,
        connection_prompt: str,
        device: torch.device | str,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        context = self.text_encoder.encode([prompt, negative_prompt, connection_prompt]).to(device=device)
        return context[0], context[1], context[2]

    def prepare_condition(
        self,
        image: PIL.Image.Image,
        frame_num: int,
        height: int,
        width: int,
        generator: torch.Generator | list[torch.Generator] | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if isinstance(generator, list):
            if len(generator) != 1:
                raise ValueError("SkyReels V3 A2V only supports a single generator for single-request inference.")
            generator = generator[0]

        cond_image = _image_to_a2v_tensor(image, self.device)
        original_color_reference = cond_image.clone()
        lat_h = height // self.vae_stride[1]
        lat_w = width // self.vae_stride[2]
        latent_frames = (frame_num - 1) // self.vae_stride[0] + 1

        noise = randn_tensor(
            (16, latent_frames, lat_h, lat_w),
            generator=generator,
            device=self.device,
            dtype=torch.float32,
        )

        mask = torch.ones(1, frame_num, lat_h, lat_w, device=self.device)
        mask[:, 1:] = 0
        mask = torch.concat([torch.repeat_interleave(mask[:, 0:1], repeats=4, dim=1), mask[:, 1:]], dim=1)
        mask = mask.view(1, mask.shape[1] // 4, 4, lat_h, lat_w)
        mask = mask.transpose(1, 2).to(self.param_dtype)

        with torch.no_grad():
            clip_context = self.clip.visual(cond_image[:, :, :1, :, :]).to(self.param_dtype)
            video_frames = torch.zeros(
                1,
                cond_image.shape[1],
                frame_num - cond_image.shape[2],
                height,
                width,
                device=self.device,
            )
            padding_frames_pixels_values = torch.concat([cond_image, video_frames], dim=2)
            y = self.vae.encode(padding_frames_pixels_values).to(self.param_dtype)
            y = torch.concat([mask, y], dim=1)

        ref_target_masks = torch.ones(3, height, width, device=self.device, dtype=torch.float32)
        ref_target_masks = nn.functional.interpolate(
            ref_target_masks.unsqueeze(0), size=(lat_h, lat_w), mode="nearest"
        ).squeeze(0)
        ref_target_masks = (ref_target_masks > 0).float().to(self.device)
        return noise, y, clip_context, ref_target_masks, original_color_reference

    def predict_noise(
        self,
        latent: torch.Tensor,
        timestep: torch.Tensor,
        context: torch.Tensor,
        clip_context: torch.Tensor,
        seq_len: int,
        y: torch.Tensor,
        audio: torch.Tensor,
        ref_target_masks: torch.Tensor,
        block_offload: bool = False,
        **kwargs: Any,
    ) -> torch.Tensor | IntermediateTensors:
        result = self.transformer(
            [latent.to(self.device)],
            t=timestep,
            context=[context],
            clip_fea=clip_context,
            seq_len=seq_len,
            y=y,
            audio=audio,
            ref_target_masks=ref_target_masks,
            block_offload=block_offload,
        )
        if isinstance(result, IntermediateTensors):
            return result
        return result[0]

    def diffuse(
        self,
        latent: torch.Tensor,
        context: torch.Tensor,
        context_null: torch.Tensor,
        clip_context: torch.Tensor,
        seq_len: int,
        y: torch.Tensor,
        audio: torch.Tensor,
        ref_target_masks: torch.Tensor,
        num_steps: int,
        shift: float,
        text_guide_scale: float,
        audio_guide_scale: float,
    ) -> torch.Tensor:
        timesteps = list(np.linspace(self.num_train_timesteps, 1, num_steps, dtype=np.float32))
        timesteps.append(0.0)
        timesteps = [torch.tensor([t], device=self.device) for t in timesteps]
        timesteps = [_timestep_transform(t, shift=shift, num_timesteps=self.num_train_timesteps) for t in timesteps]
        self._num_timesteps = len(timesteps) - 1

        zero_audio = torch.zeros_like(audio)[-1:]
        with self.progress_bar(total=len(timesteps) - 1) as pbar:
            for step_idx in range(len(timesteps) - 1):
                timestep = timesteps[step_idx]
                self._current_timestep = timestep
                self.record_denoise_step(step_idx, timestep)

                positive_kwargs = {
                    "latent": latent,
                    "timestep": timestep,
                    "context": context,
                    "clip_context": clip_context,
                    "seq_len": seq_len,
                    "y": y,
                    "audio": audio,
                    "ref_target_masks": ref_target_masks,
                }
                branches_kwargs = [positive_kwargs]
                cfg_scale = {
                    "mode": "skyreels_v3_a2v",
                    "text_guide_scale": text_guide_scale,
                    "audio_guide_scale": audio_guide_scale,
                    "branch": None,
                }
                if text_guide_scale > 1.0 and audio_guide_scale > 1.0:
                    branches_kwargs.append({**positive_kwargs, "context": context_null})
                    branches_kwargs.append({**positive_kwargs, "context": context_null, "audio": zero_audio})
                    cfg_scale["branch"] = "text_audio"
                elif text_guide_scale > 1.0:
                    branches_kwargs.append({**positive_kwargs, "context": context_null})
                    cfg_scale["branch"] = "text"
                elif audio_guide_scale > 1.0:
                    branches_kwargs.append({**positive_kwargs, "audio": zero_audio})
                    cfg_scale["branch"] = "audio"

                noise_pred = self.predict_noise_with_multi_branch_cfg(
                    do_true_cfg=len(branches_kwargs) > 1,
                    true_cfg_scale=cfg_scale,
                    branches_kwargs=branches_kwargs,
                    cfg_normalize=False,
                )
                if isinstance(noise_pred, tuple):
                    raise ValueError("SkyReels V3 A2V transformer must return a single tensor prediction.")
                noise_pred = -noise_pred
                dt = (timesteps[step_idx] - timesteps[step_idx + 1]) / self.num_train_timesteps
                latent = latent + noise_pred * dt[:, None, None, None]
                pbar.update()
        return latent

    def forward(self, req: DiffusionRequestBatch) -> DiffusionOutput:
        if len(req.prompts) != 1:
            raise ValueError("SkyReels V3 A2V only supports a single prompt per request.")

        first_prompt = req.prompts[0]
        prompt = first_prompt if isinstance(first_prompt, str) else (first_prompt.get("prompt") or "")
        if not prompt:
            raise ValueError("Prompt is required for SkyReels V3 A2V generation.")
        negative_prompt = None if isinstance(first_prompt, str) else first_prompt.get("negative_prompt")

        multi_modal_data = first_prompt.get("multi_modal_data", {}) if not isinstance(first_prompt, str) else {}
        image = _normalize_avatar_image(multi_modal_data.get("image", multi_modal_data.get("cond_image")))
        raw_audio = multi_modal_data.get("audio", multi_modal_data.get("cond_audio"))
        if isinstance(raw_audio, dict):
            raw_audio = raw_audio.get("person1")
        audio, sample_rate = _load_audio_array(raw_audio)

        extra_args = req.sampling_params.extra_args or {}
        if req.sampling_params.height is None or req.sampling_params.width is None:
            bucket = str(extra_args.get("resolution", extra_args.get("size_bucket", "720P")))
            bucket_h, bucket_w = _select_bucket_size(image, bucket)
            height = int(req.sampling_params.height or bucket_h)
            width = int(req.sampling_params.width or bucket_w)
        else:
            height = int(req.sampling_params.height)
            width = int(req.sampling_params.width)
        if height % 16 != 0 or width % 16 != 0:
            raise ValueError(f"`height` and `width` must be divisible by 16, got {height} and {width}.")
        if image.size != (width, height):
            image = _resize_and_centercrop_image(image, height, width)

        frame_num = int(extra_args.get("frame_num", req.sampling_params.num_frames or DEFAULT_SKYREELS_A2V_FRAMES))
        if frame_num <= 1:
            frame_num = DEFAULT_SKYREELS_A2V_FRAMES
        if (frame_num - 1) % self.vae_stride[0] != 0:
            raise ValueError(f"SkyReels V3 A2V `frame_num` must satisfy (frame_num - 1) % 4 == 0, got {frame_num}.")

        num_steps = int(
            extra_args.get(
                "sampling_steps",
                req.sampling_params.num_inference_steps or DEFAULT_SKYREELS_A2V_STEPS,
            )
        )
        shift = float(extra_args.get("shift", self.od_config.flow_shift or DEFAULT_SKYREELS_A2V_SHIFT))
        text_guide_scale, audio_guide_scale = _resolve_guidance_scales(req.sampling_params, extra_args)
        self._guidance_scale = text_guide_scale
        self._audio_guidance_scale = audio_guide_scale
        max_frames_num = int(extra_args.get("max_frames_num", DEFAULT_SKYREELS_A2V_MAX_FRAMES))
        _validate_audio_duration(audio, sample_rate, max_frames_num)
        if _is_rank_zero() and len(audio) / float(sample_rate) > frame_num / DEFAULT_SKYREELS_A2V_FPS:
            logger.warning(
                "SkyReels V3 A2V currently generates one %d-frame clip; audio beyond %.2fs is truncated.",
                frame_num,
                frame_num / DEFAULT_SKYREELS_A2V_FPS,
            )

        n_prompt = str(extra_args.get("n_prompt", negative_prompt or DEFAULT_SKYREELS_A2V_NEG_PROMPT))
        connection_prompt = str(extra_args.get("connection_prompt", "a person is talking"))
        generator = req.sampling_params.generator
        if generator is None and req.sampling_params.seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(req.sampling_params.seed)

        dtype = self.param_dtype
        if _is_rank_zero():
            logger.debug(
                "SkyReels V3 A2V request: size=%sx%s frames=%s steps=%s text_cfg=%s audio_cfg=%s",
                width,
                height,
                frame_num,
                num_steps,
                text_guide_scale,
                audio_guide_scale,
            )

        context, context_null, _connection_embedding = self.encode_text(
            prompt,
            n_prompt,
            connection_prompt,
            self.device,
        )
        context = context.to(dtype=dtype)
        context_null = context_null.to(dtype=dtype)
        audio_context = self.encode_audio(audio, sample_rate, frame_num, self.device, dtype)
        noise, y, clip_context, ref_target_masks, original_color_reference = self.prepare_condition(
            image=image,
            frame_num=frame_num,
            height=height,
            width=width,
            generator=generator,
        )

        lat_h = height // self.vae_stride[1]
        lat_w = width // self.vae_stride[2]
        max_seq_len = (
            ((frame_num - 1) // self.vae_stride[0] + 1) * lat_h * lat_w // (self.patch_size[1] * self.patch_size[2])
        )
        max_seq_len = int(math.ceil(max_seq_len / self.sp_size)) * self.sp_size

        latent = self.diffuse(
            latent=noise,
            context=context,
            context_null=context_null,
            clip_context=clip_context,
            seq_len=max_seq_len,
            y=y,
            audio=audio_context,
            ref_target_masks=ref_target_masks,
            num_steps=num_steps,
            shift=shift,
            text_guide_scale=text_guide_scale,
            audio_guide_scale=audio_guide_scale,
        )
        self._current_timestep = None

        output_type = req.sampling_params.output_type or "np"
        if output_type == "latent":
            output: torch.Tensor | tuple[Any, ...] = latent
        else:
            with torch.no_grad():
                video = self.vae.decode(latent.to(torch.float32).unsqueeze(0))
                video = match_and_blend_colors(video, original_color_reference, 1.0)
            audio_samples = min(len(audio), int(round(frame_num / DEFAULT_SKYREELS_A2V_FPS * sample_rate)))
            output = (video, audio[:audio_samples], sample_rate)

        if current_omni_platform.is_available():
            current_omni_platform.empty_cache()

        return DiffusionOutput(
            output=output,
            stage_durations=self.stage_durations if hasattr(self, "stage_durations") else None,
        )

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        loader = AutoWeightsLoader(self)
        return loader.load_weights(weights)
