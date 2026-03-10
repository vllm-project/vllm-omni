  # SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import logging
import tempfile
import wave
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
from PIL import Image

from vllm_omni.diffusion.data import DiffusionOutput, OmniDiffusionConfig
from vllm_omni.diffusion.distributed.utils import get_local_device
from vllm_omni.diffusion.request import OmniDiffusionRequest
from vllm_omni.diffusion.model_loader.diffusers_loader import DiffusersPipelineLoader
from vllm.model_executor.models.utils import AutoWeightsLoader

logger = logging.getLogger(__name__)


def _resolve_default_config_path() -> Path:
    base_dir = Path(__file__).resolve().parent / "dreamid_omni"
    return base_dir / "configs" / "inference" / "inference_r2av.yaml"


def get_dreamid_omni_post_process_func(od_config: OmniDiffusionConfig):
    def post_process_func(output: tuple[torch.Tensor, torch.Tensor] | torch.Tensor | tuple[np.ndarray, np.ndarray]):
        if isinstance(output, tuple) and len(output) == 2:
            video, audio = output
            if isinstance(audio, torch.Tensor):
                audio = audio.detach().cpu()
            return {"video": video, "audio": audio}
        return output

    return post_process_func


def _normalize_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return list(value)
    return [value]


def _get_prompt_field(prompt: Any, key: str) -> Any:
    if isinstance(prompt, dict):
        if key in prompt:
            return prompt.get(key)
        additional = prompt.get("additional_information")
        if isinstance(additional, dict) and key in additional:
            return additional.get(key)
    return None


def _collect_media_inputs(prompt: Any, keys: tuple[str, ...]) -> list[Any]:
    for key in keys:
        value = _get_prompt_field(prompt, key)
        if value is not None:
            return _normalize_list(value)
    return []


def _snap_hw_to_multiple(value: int, multiple: int) -> int:
    if multiple <= 0:
        return value
    return max(multiple, (int(value) // multiple) * multiple)


def _write_silence_wav(path: Path, sample_rate: int, duration_sec: float = 1.0) -> None:
    num_frames = max(1, int(sample_rate * duration_sec))
    with wave.open(str(path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)  # 16-bit PCM
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(b"\x00\x00" * num_frames)


class DreamIDOmniPipeline(torch.nn.Module):
    support_image_input: bool = True
    color_format: str = "RGB"

    def __init__(self, *, od_config: OmniDiffusionConfig, prefix: str = ""):
        super().__init__()
        self.od_config = od_config
        self.device = get_local_device()
        if self.device.type != "cuda":
            raise RuntimeError("DreamID-Omni requires CUDA; CPU execution is not supported.")

        from omegaconf import OmegaConf
        from vllm_omni.diffusion.models.dreamid_omni.dreamid_omni.dreamid_omni_engine import (
            DreamIDOmniEngine,
        )

        custom_args = getattr(od_config, "custom_pipeline_args", None) or {}

        config_path = custom_args.get("config_path")
        if config_path is None:
            config_path = _resolve_default_config_path()
        config_path = Path(config_path)
        if not config_path.exists():
            raise FileNotFoundError(f"DreamID-Omni config not found at {config_path}")

        config = OmegaConf.load(str(config_path))

        ckpt_dir = custom_args.get("ckpt_dir") or od_config.model or config.get("ckpt_dir")
        if ckpt_dir is not None:
            config.ckpt_dir = ckpt_dir
        model_name = custom_args.get("model_name")
        if model_name is not None:
            config.model_name = model_name
        cpu_offload = custom_args.get("cpu_offload")
        if cpu_offload is not None:
            config.cpu_offload = bool(cpu_offload)

        self.default_video_hw = list(config.get("video_frame_height_width", [720, 720]))
        self.default_sample_steps = int(config.get("sample_steps", 50))
        self.default_solver_name = str(config.get("solver_name", "unipc"))
        self.default_shift = float(config.get("shift", 5.0))
        self.default_video_guidance_scale = float(config.get("video_guidance_scale", 4.0))
        self.default_audio_guidance_scale = float(config.get("audio_guidance_scale", 3.0))
        self.default_video_negative_prompt = str(config.get("video_negative_prompt", ""))
        self.default_audio_negative_prompt = str(config.get("audio_negative_prompt", ""))
        self.default_seed = config.get("seed", None)
        self.default_slg_layer = int(config.get("slg_layer", 9))
        self.audio_sample_rate = int(config.get("audio_sample_rate", 16000))

        self.weights_sources = []

        self.engine = DreamIDOmniEngine(
            config=config,
            device=self.device,
            target_dtype=od_config.dtype,
        )


        class DummyVAE:
            use_slicing = False
            use_tiling = False

        self.vae = DummyVAE()

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        """
        Load weights for the pipeline.
        
        
        Args:
            weights: Iterable of (param_name, param_tensor) tuples
            
        Returns:
            Set of loaded parameter names
        """

        loader = AutoWeightsLoader(self)
        return loader.load_weights(weights)

    def forward(self, req: OmniDiffusionRequest) -> DiffusionOutput:
        if not req.prompts:
            raise ValueError("DreamID-Omni received an empty prompt list.")

        if len(req.prompts) > 1:
            logger.warning("DreamID-Omni only supports a single prompt per request; using the first prompt.")

        prompt = req.prompts[0]
        text_prompt = prompt if isinstance(prompt, str) else (prompt.get("prompt") or prompt.get("text") or "")

        extra_args = getattr(req.sampling_params, "extra_args", {}) or {}

        image_inputs = _collect_media_inputs(prompt, ("image_paths", "images", "image_path", "image"))
        audio_inputs = _collect_media_inputs(prompt, ("audio_paths", "audios", "audio_path", "audio"))

        if isinstance(prompt, dict):
            multimodal = prompt.get("multi_modal_data")
            if isinstance(multimodal, dict):
                image_inputs.extend(_normalize_list(multimodal.get("image")))
                audio_inputs.extend(_normalize_list(multimodal.get("audio")))

        image_paths: list[str] = []
        audio_paths: list[str] = []
        pil_images: list[Image.Image] = []

        for item in image_inputs:
            if isinstance(item, (str, Path)):
                image_paths.append(str(item))
            elif isinstance(item, Image.Image):
                pil_images.append(item.convert("RGB"))
            else:
                logger.warning("Unsupported image input type: %s", type(item))

        for item in audio_inputs:
            if isinstance(item, (str, Path)):
                audio_paths.append(str(item))
            else:
                logger.warning("Unsupported audio input type: %s", type(item))

        temp_dir: tempfile.TemporaryDirectory[str] | None = None
        if pil_images:
            temp_dir = tempfile.TemporaryDirectory(prefix="dreamid_omni_")
            for idx, img in enumerate(pil_images):
                path = Path(temp_dir.name) / f"ref_{idx}.png"
                img.save(path)
                image_paths.append(str(path))

        if not image_paths and not audio_paths:
            raise ValueError("DreamID-Omni requires at least one reference image or audio input.")

        if len(image_paths) > 2:
            logger.warning("DreamID-Omni accepts at most two reference images; extra inputs will be ignored.")
        if len(audio_paths) > 2:
            logger.warning("DreamID-Omni accepts at most two reference audios; extra inputs will be ignored.")

        image0_path = image_paths[0] if len(image_paths) > 0 else None
        image1_path = image_paths[1] if len(image_paths) > 1 else None
        audio0_path = audio_paths[0] if len(audio_paths) > 0 else None
        audio1_path = audio_paths[1] if len(audio_paths) > 1 else None

        video_hw = _get_prompt_field(prompt, "video_frame_height_width")
        if video_hw is None:
            video_hw = extra_args.get("video_frame_height_width")
        if video_hw is None:
            height = getattr(req.sampling_params, "height", None)
            width = getattr(req.sampling_params, "width", None)
            if height is not None and width is not None:
                video_hw = [int(height), int(width)]
        if video_hw is None:
            video_hw = self.default_video_hw
        if not (isinstance(video_hw, (list, tuple)) and len(video_hw) == 2):
            raise ValueError(f"Invalid video_frame_height_width: {video_hw}")
        video_hw = [int(video_hw[0]), int(video_hw[1])]

        patch_size = None
        try:
            patch_size = self.engine.model.video_model.patch_size
        except Exception:
            patch_size = None
        if patch_size is not None and len(patch_size) >= 3:
            multiple_h = 16 * int(patch_size[1])
            multiple_w = 16 * int(patch_size[2])
        else:
            multiple_h = 32
            multiple_w = 32

        snapped_hw = [
            _snap_hw_to_multiple(video_hw[0], multiple_h),
            _snap_hw_to_multiple(video_hw[1], multiple_w),
        ]
        if snapped_hw != video_hw:
            logger.warning(
                "Adjusted video_frame_height_width from %s to %s for model compatibility.",
                video_hw,
                snapped_hw,
            )
            video_hw = snapped_hw

        is_dummy_warmup = (text_prompt == "dummy run")
        if is_dummy_warmup and not audio_paths:
            if temp_dir is None:
                temp_dir = tempfile.TemporaryDirectory(prefix="dreamid_omni_")
            dummy_audio_path = Path(temp_dir.name) / "dummy_audio.wav"
            _write_silence_wav(dummy_audio_path, self.audio_sample_rate, duration_sec=1.0)
            audio_paths.append(str(dummy_audio_path))
            audio0_path = audio_paths[0]
            audio1_path = audio_paths[1] if len(audio_paths) > 1 else None

        video_negative_prompt = extra_args.get("video_negative_prompt")
        audio_negative_prompt = extra_args.get("audio_negative_prompt")
        shared_negative = _get_prompt_field(prompt, "negative_prompt")
        if video_negative_prompt is None:
            video_negative_prompt = shared_negative or self.default_video_negative_prompt
        if audio_negative_prompt is None:
            audio_negative_prompt = shared_negative or self.default_audio_negative_prompt

        if req.sampling_params.guidance_scale_provided:
            video_guidance_scale = float(req.sampling_params.guidance_scale)
        else:
            video_guidance_scale = float(extra_args.get("video_guidance_scale", self.default_video_guidance_scale))

        guidance_scale_2 = getattr(req.sampling_params, "guidance_scale_2", None)
        if guidance_scale_2 is not None:
            audio_guidance_scale = float(guidance_scale_2)
        else:
            audio_guidance_scale = float(extra_args.get("audio_guidance_scale", self.default_audio_guidance_scale))

        sample_steps = int(req.sampling_params.num_inference_steps or self.default_sample_steps)
        solver_name = str(extra_args.get("solver_name", self.default_solver_name))
        shift = float(extra_args.get("shift", self.default_shift))
        slg_layer = int(extra_args.get("slg_layer", self.default_slg_layer))

        seed = req.sampling_params.seed
        if seed is None:
            seed = extra_args.get("seed", self.default_seed)
        seed = int(seed) if seed is not None else 0

        try:
            video, audio, _ = self.engine.generate(
                text_prompt=text_prompt,
                image0_path=image0_path,
                image1_path=image1_path,
                audio0_path=audio0_path,
                audio1_path=audio1_path,
                video_frame_height_width=video_hw,
                seed=seed,
                solver_name=solver_name,
                sample_steps=sample_steps,
                shift=shift,
                video_guidance_scale=video_guidance_scale,
                audio_guidance_scale=audio_guidance_scale,
                slg_layer=slg_layer,
                video_negative_prompt=video_negative_prompt,
                audio_negative_prompt=audio_negative_prompt,
            )
        finally:
            if temp_dir is not None:
                temp_dir.cleanup()

        if video is None or audio is None:
            raise RuntimeError("DreamID-Omni generation failed.")

        if isinstance(video, np.ndarray):
            video = torch.from_numpy(video)
        if isinstance(audio, np.ndarray):
            audio = torch.from_numpy(audio)

        return DiffusionOutput(
            output=(video, audio),
            custom_output={
                "audio_sample_rate": self.audio_sample_rate,
                "video_frame_height_width": video_hw,
            },
        )
