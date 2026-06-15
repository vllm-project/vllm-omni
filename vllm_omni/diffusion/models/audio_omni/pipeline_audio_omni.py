# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import json
import os
from collections.abc import Iterable
from typing import Any, ClassVar

import torch
from diffusers import AutoencoderOobleck
from torch import nn
from vllm.logger import init_logger

from vllm_omni.diffusion.data import DiffusionOutput, OmniDiffusionConfig
from vllm_omni.diffusion.distributed.utils import get_local_device
from vllm_omni.diffusion.models.audio_omni.audio_omni_transformer import AudioOmniDiT
from vllm_omni.diffusion.models.audio_omni.conditioners import (
    AudioMelConditioner,
    OmniConditioner,
    SynchformerConditioner,
    TTSConditioner,
)
from vllm_omni.diffusion.models.interface import SupportAudioOutput
from vllm_omni.diffusion.profiler.diffusion_pipeline_profiler import DiffusionPipelineProfilerMixin
from vllm_omni.diffusion.request import OmniDiffusionRequest
from vllm_omni.transformers_utils.processors import audio_omni as _audio_omni_processing

logger = init_logger(__name__)

_SYNC_FRAME_COUNT = 240
_SYNC_FEATURE_DIM = 768

_AUDIO_OMNI_OOBLECK_CONFIG = {
    "audio_channels": 2,
    "channel_multiples": [1, 2, 4, 8, 16],
    "decoder_channels": 128,
    "decoder_input_channels": 64,
    "downsampling_ratios": [2, 4, 4, 8, 8],
    "encoder_hidden_size": 128,
    "sampling_rate": 44100,
}


def get_audio_omni_post_process_func(od_config: OmniDiffusionConfig):
    """Convert the pipeline's float audio tensor to a CPU numpy array for serving."""

    def post_process_func(audio: torch.Tensor) -> Any:
        if isinstance(audio, torch.Tensor):
            return audio.detach().cpu().float().numpy()
        return audio

    return post_process_func


def _remap_oobleck_state_dict(sd: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """stable-audio-tools Oobleck layout -> diffusers AutoencoderOobleck layout.

    Structural rules (per side, encoder/decoder have mirrored block ordering):
      layers.0 -> conv1 ; layers.6 -> snake1 ; layers.7 -> conv2
      decoder block (layers.1..5 -> block.0..4): snake1, conv_t1, res_unit1..3
      encoder block: res_unit1..3, snake1, conv1
      residual unit inner: snake1, conv1, snake2, conv2
    Snake alpha/beta reshape [C] -> [1, C, 1].
    """
    _RES_INNER = {"0": "snake1", "1": "conv1", "2": "snake2", "3": "conv2"}

    def _map_one(key: str) -> str:
        side, rest = key.split(".", 1)  # encoder|decoder
        parts = rest.split(".")
        assert parts[0] == "layers", key
        idx = int(parts[1])
        if idx == 0:
            return f"{side}.conv1.{'.'.join(parts[2:])}"
        if idx == 6:
            return f"{side}.snake1.{'.'.join(parts[2:])}"
        if idx == 7:
            return f"{side}.conv2.{'.'.join(parts[2:])}"
        block = f"{side}.block.{idx - 1}"
        assert parts[2] == "layers", key
        sub = parts[3]
        tail = parts[4:]
        if side == "decoder":
            if sub == "0":
                return f"{block}.snake1.{'.'.join(tail)}"
            if sub == "1":
                return f"{block}.conv_t1.{'.'.join(tail)}"
            res_idx = int(sub) - 1  # layers.2..4 -> res_unit1..3
        else:  # encoder: res, res, res, snake, conv
            if sub == "3":
                return f"{block}.snake1.{'.'.join(tail)}"
            if sub == "4":
                return f"{block}.conv1.{'.'.join(tail)}"
            res_idx = int(sub) + 1  # layers.0..2 -> res_unit1..3
        assert tail[0] == "layers", key
        return f"{block}.res_unit{res_idx}.{_RES_INNER[tail[1]]}.{'.'.join(tail[2:])}"

    out = {}
    for key, tensor in sd.items():
        new_key = _map_one(key)
        if new_key.endswith(".alpha") or new_key.endswith(".beta"):
            tensor = tensor.reshape(1, -1, 1)
        out[new_key] = tensor
    return out


class AudioOmniPipeline(nn.Module, SupportAudioOutput, DiffusionPipelineProfilerMixin):
    support_audio_output: ClassVar[bool] = True
    audio_sample_rate: ClassVar[int] = 44100
    audio_channels: ClassVar[int] = 2
    _PROFILER_TARGETS: ClassVar[list[str]] = ["diffuse"]

    def __init__(self, *, od_config: OmniDiffusionConfig, prefix: str = ""):
        super().__init__()
        self.od_config = od_config
        self.device = get_local_device()
        if od_config.model is None:
            raise ValueError(
                "AudioOmniPipeline requires od_config.model "
                "(directory or HF id with Audio-Omni.json + model.ckpt, e.g. HKUSTAudio/Audio-Omni)."
            )

        if os.path.exists(od_config.model):
            self._model_root = os.path.abspath(od_config.model)
        else:
            from vllm_omni.model_executor.model_loader.weight_utils import download_weights_from_hf_specific

            self._model_root = download_weights_from_hf_specific(od_config.model, None, ["*"])

        # Upstream ships the bundle config as ``Audio-Omni.json``; the engine's model
        # discovery (and the AudioX precedent) keys off a top-level ``config.json``
        # carrying ``model_type``. Accept either, preferring config.json when present.
        config_path = os.path.join(self._model_root, "config.json")
        if not os.path.exists(config_path):
            config_path = os.path.join(self._model_root, "Audio-Omni.json")
        with open(config_path, encoding="utf-8") as f:
            self._model_config = json.load(f)

        prev_default_dtype = torch.get_default_dtype()
        torch.set_default_dtype(torch.float32)
        try:
            self._build_modules()
        finally:
            torch.set_default_dtype(prev_default_dtype)

        self._load_checkpoint(os.path.join(self._model_root, "model.ckpt"))

        self.setup_diffusion_pipeline_profiler(
            profiler_targets=list(self._PROFILER_TARGETS),
            enable_diffusion_pipeline_profiler=od_config.enable_diffusion_pipeline_profiler,
        )

    def _build_modules(self) -> None:
        model_config = self._model_config["model"]
        self._sample_rate = int(self._model_config["sample_rate"])
        self._sample_size = int(self._model_config["sample_size"])
        self.io_channels = int(model_config["io_channels"])

        diffusion_config = dict(model_config["diffusion"]["config"])
        self.model = AudioOmniDiT(**diffusion_config)

        cond_configs = {c["id"]: dict(c["config"]) for c in model_config["conditioning"]["configs"]}
        self.high_level_cond_ids = list(model_config["diffusion"]["high_level_cond_ids"])
        self.low_level_cond_ids = list(model_config["diffusion"]["low_level_cond_ids"])

        self.omni_conditioner = OmniConditioner(**cond_configs["omni_prompt"])
        self.speech_conditioner = TTSConditioner(
            seq_len=int(cond_configs["speech_prompt"]["seq_len"]),
            proj_seq_len=int(cond_configs["speech_prompt"]["proj_seq_len"]),
        )
        self.mel_conditioner = AudioMelConditioner(**cond_configs["audio_input_prompt"])
        self.sync_conditioner = SynchformerConditioner(**cond_configs["sync_feature"])

        self.pretransform = AutoencoderOobleck(**_AUDIO_OMNI_OOBLECK_CONFIG)
        self._latent_len = self._sample_size // int(self.pretransform.hop_length)

    # ------------------------------------------------------------------ weights
    def _load_checkpoint(self, ckpt_path: str) -> None:
        logger.info("Loading Audio-Omni checkpoint from %s (mmap)", ckpt_path)
        sd = torch.load(ckpt_path, map_location="cpu", mmap=True, weights_only=True)["state_dict"]

        def _sub(prefix: str) -> dict[str, torch.Tensor]:
            return {k[len(prefix) :]: v for k, v in sd.items() if k.startswith(prefix)}

        loaded = self.model.load_weights(iter(_sub("model.model.").items()))
        logger.info("DiT: loaded %d params", len(loaded))

        cond_modules = {
            "omni_prompt": self.omni_conditioner,
            "speech_prompt": self.speech_conditioner,
            "audio_input_prompt": self.mel_conditioner,
            "sync_feature": self.sync_conditioner,
        }
        for cond_id, module in cond_modules.items():
            cond_sd = _sub(f"conditioner.conditioners.{cond_id}.")
            missing, unexpected = module.load_state_dict(cond_sd, strict=False)
            if unexpected:
                raise RuntimeError(f"conditioner {cond_id}: unexpected checkpoint keys {unexpected[:5]}")
            real_missing = [m for m in missing if not m.startswith("_mel.")]
            if real_missing:
                raise RuntimeError(f"conditioner {cond_id}: missing checkpoint keys {real_missing[:5]}")

        vae_sd = _remap_oobleck_state_dict(_sub("pretransform.model."))
        self.pretransform.load_state_dict(vae_sd, strict=True)

        self.eval().requires_grad_(False)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> None:
        self.to(torch.float32)
        self.omni_conditioner.model.thinker.to(torch.bfloat16)
        self.eval().requires_grad_(False)
        return None

    # ------------------------------------------------------------- conditioning
    def _encode_conditioning_tensors(self, batch_metadata: list[dict[str, Any]]) -> dict[str, Any]:
        device = self.device
        return {
            "omni_prompt": self.omni_conditioner([m["omni_prompt"] for m in batch_metadata], device),
            "speech_prompt": self.speech_conditioner([m["speech_prompt"] for m in batch_metadata], device),
            "audio_input_prompt": self.mel_conditioner([m["audio_input_wav"] for m in batch_metadata], device),
            "sync_feature": self.sync_conditioner([m["sync_feature"] for m in batch_metadata], device),
        }

    def get_conditioning_inputs(self, conditioning_tensors: dict[str, Any]) -> dict[str, Any]:
        """High-level conditioners concat into cross-attention tokens (seq dim), low-level
        ones into the global token sequence (upstream ConditionedDiffusionModelWrapper)."""
        cross = [conditioning_tensors[k] for k in self.high_level_cond_ids]
        cross_attn_cond = torch.cat([t for t, _ in cross], dim=1)
        cross_attn_mask = torch.cat([m.bool() for _, m in cross], dim=1)
        global_cond = torch.cat([conditioning_tensors[k][0] for k in self.low_level_cond_ids], dim=-2)
        return {
            "cross_attn_cond": cross_attn_cond,
            "cross_attn_mask": cross_attn_mask,
            "global_cond": global_cond,
        }

    # ------------------------------------------------------------------ diffuse
    def diffuse(
        self,
        *,
        steps: int,
        cfg_scale: float,
        conditioning_inputs: dict[str, Any],
        batch_size: int,
        generator: torch.Generator,
        inject_noise: torch.Tensor | None = None,
    ) -> torch.Tensor:
        device = self.device
        model_dtype = next(self.model.parameters()).dtype

        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
        torch.backends.cudnn.benchmark = False

        # inject_noise pins the initial latent for parity tests that call diffuse()
        # directly; normal generation draws fresh noise from the generator.
        if inject_noise is not None:
            noise = inject_noise.to(device=device, dtype=model_dtype)
            if noise.shape != (batch_size, self.io_channels, self._latent_len):
                raise ValueError(f"injected noise must have shape {(batch_size, self.io_channels, self._latent_len)}")
        else:
            noise = torch.randn(
                [batch_size, self.io_channels, self._latent_len],
                device=device,
                generator=generator,
                dtype=model_dtype,
            )

        cond = {
            k: (v.type(model_dtype) if isinstance(v, torch.Tensor) and v.is_floating_point() else v)
            for k, v in conditioning_inputs.items()
        }

        cross_attn_mask = cond["cross_attn_mask"]
        if cross_attn_mask is not None and bool(cross_attn_mask.all()):
            cross_attn_mask = None

        # Rectified-flow Euler, fp16 autocast.
        x = noise
        t_schedule = torch.linspace(1, 0, steps + 1)
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            for t_curr, t_prev in zip(t_schedule[:-1], t_schedule[1:]):
                t_in = t_curr * torch.ones((x.shape[0],), dtype=x.dtype, device=device)
                dt = t_prev - t_curr  # negative: solving backwards
                x = x + dt * self.model(
                    x,
                    t_in,
                    cross_attn_cond=cond["cross_attn_cond"],
                    cross_attn_mask=cross_attn_mask,
                    global_cond=cond["global_cond"],
                    cfg_scale=cfg_scale,
                )

        # VAE decode forced float32.
        vae = self.pretransform.to(device=device, dtype=torch.float32).eval()
        with torch.autocast(device_type="cuda", enabled=False):
            return vae.decode(x.to(torch.float32), return_dict=True).sample

    # ------------------------------------------------------------------ forward
    def forward(self, req: OmniDiffusionRequest) -> DiffusionOutput:
        if req.prompts is None or len(req.prompts) == 0:
            raise ValueError("AudioOmniPipeline requires at least one prompt (the transcript for TTS).")
        prompts = [p["prompt"] if isinstance(p, dict) else str(p) for p in req.prompts]

        sampling_params = req.sampling_params
        if sampling_params.num_inference_steps is None:
            raise ValueError("AudioOmniPipeline requires sampling_params.num_inference_steps.")
        steps = int(sampling_params.num_inference_steps)
        cfg_scale = float(sampling_params.guidance_scale)
        generator = sampling_params.generator
        if generator is None:
            raise ValueError("AudioOmniPipeline requires sampling_params.generator.")
        if sampling_params.num_outputs_per_prompt not in (None, 1) and int(sampling_params.num_outputs_per_prompt) != 1:
            raise ValueError("AudioOmniPipeline currently supports num_outputs_per_prompt=1.")

        extra_args = sampling_params.extra_args or {}
        task = str(extra_args.get("task", "tts")).strip().lower()
        if task != "tts":
            raise ValueError(f"AudioOmniPipeline milestone 1 supports task='tts' only, got {task!r}.")
        voice_prompt_path = extra_args.get("voice_prompt_path")
        voice_ref_text = extra_args.get("voice_ref_text")
        seconds_total = int(extra_args.get("seconds_total", 10))

        device = self.device
        voice_ref_duration = 0.0
        if voice_prompt_path:
            voice_wav, voice_ref_duration = _audio_omni_processing.load_voice_prompt(str(voice_prompt_path))
        else:
            voice_wav = torch.zeros(self._sample_rate * seconds_total, dtype=torch.float32)

        batch_metadata = [
            {
                "omni_prompt": _audio_omni_processing.build_tts_prompt(),
                "speech_prompt": _audio_omni_processing.build_speech_prompt(prompt, voice_ref_text),
                "audio_input_wav": voice_wav,
                "sync_feature": torch.zeros(1, _SYNC_FRAME_COUNT, _SYNC_FEATURE_DIM, device=device),
            }
            for prompt in prompts
        ]

        conditioning_tensors = self._encode_conditioning_tensors(batch_metadata)
        conditioning_inputs = self.get_conditioning_inputs(conditioning_tensors)

        audio = self.diffuse(
            steps=steps,
            cfg_scale=cfg_scale,
            conditioning_inputs=conditioning_inputs,
            batch_size=len(prompts),
            generator=generator,
        )

        return DiffusionOutput(
            output=audio,
            custom_output={"task": task, "voice_ref_duration": voice_ref_duration},
            stage_durations=self.stage_durations
            if getattr(self, "enable_diffusion_pipeline_profiler", False)
            else None,
        )
