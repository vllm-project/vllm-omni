# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import json
import os
from typing import Any

import torch
import torchaudio
from safetensors import safe_open
from safetensors.torch import load_file
from torch import nn
from vllm.logger import init_logger

from vllm_omni.diffusion.models.nava.config import NAVAConfig
from vllm_omni.diffusion.models.nava.vocoder import LTX2Vocoder, LTX2VocoderWithBWE

logger = init_logger(__name__)


def _read_safetensors_config(path: str) -> dict[str, Any]:
    with safe_open(path, framework="pt") as handle:
        metadata = handle.metadata() or {}
    raw_config = metadata.get("config")
    if not raw_config:
        return {}
    parsed = json.loads(raw_config)
    return parsed if isinstance(parsed, dict) else {}


def _audio_vae_kwargs(config: dict[str, Any]) -> dict[str, Any]:
    audio_cfg = config.get("audio_vae", {})
    params = audio_cfg.get("model", {}).get("params", {})
    ddconfig = params.get("ddconfig", {})
    preprocessing = audio_cfg.get("preprocessing", {})
    stft_cfg = preprocessing.get("stft", {})
    mel_cfg = preprocessing.get("mel", {})
    variables = audio_cfg.get("variables", {})
    mel_bins = ddconfig.get("mel_bins") or mel_cfg.get("n_mel_channels") or variables.get("mel_bins")
    kwargs = {
        "base_channels": ddconfig.get("ch", 128),
        "output_channels": ddconfig.get("out_ch", 2),
        "ch_mult": tuple(ddconfig.get("ch_mult", (1, 2, 4))),
        "num_res_blocks": ddconfig.get("num_res_blocks", 2),
        "attn_resolutions": tuple(ddconfig.get("attn_resolutions", (8, 16, 32))),
        "in_channels": ddconfig.get("in_channels", 2),
        "resolution": ddconfig.get("resolution", 256),
        "latent_channels": ddconfig.get("z_channels", 8),
        "norm_type": ddconfig.get("norm_type", "pixel"),
        "causality_axis": ddconfig.get("causality_axis", "height"),
        "dropout": ddconfig.get("dropout", 0.0),
        "mid_block_add_attention": ddconfig.get("mid_block_add_attention", True),
        "sample_rate": params.get("sampling_rate", 16000),
        "mel_hop_length": stft_cfg.get("hop_length", 160),
        "is_causal": stft_cfg.get("causal", True),
        "mel_bins": mel_bins,
        "double_z": ddconfig.get("double_z", True),
    }
    return {key: value for key, value in kwargs.items() if value is not None}


def _vocoder_kwargs(config: dict[str, Any]) -> tuple[bool, dict[str, Any]]:
    vocoder_cfg = config.get("vocoder", {})
    if "bwe" not in vocoder_cfg:
        return False, _single_vocoder_kwargs(vocoder_cfg)

    base = _single_vocoder_kwargs(vocoder_cfg.get("vocoder", {}))
    bwe = _single_vocoder_kwargs(vocoder_cfg["bwe"])
    return True, {
        "in_channels": base["in_channels"],
        "hidden_channels": base["hidden_channels"],
        "out_channels": base["out_channels"],
        "upsample_kernel_sizes": base["upsample_kernel_sizes"],
        "upsample_factors": base["upsample_factors"],
        "resnet_kernel_sizes": base["resnet_kernel_sizes"],
        "resnet_dilations": base["resnet_dilations"],
        "act_fn": base["act_fn"],
        "final_act_fn": base["final_act_fn"],
        "final_bias": base["final_bias"],
        "bwe_in_channels": bwe["in_channels"],
        "bwe_hidden_channels": bwe["hidden_channels"],
        "bwe_out_channels": bwe["out_channels"],
        "bwe_upsample_kernel_sizes": bwe["upsample_kernel_sizes"],
        "bwe_upsample_factors": bwe["upsample_factors"],
        "bwe_resnet_kernel_sizes": bwe["resnet_kernel_sizes"],
        "bwe_resnet_dilations": bwe["resnet_dilations"],
        "bwe_act_fn": bwe["act_fn"],
        "bwe_final_act_fn": bwe["final_act_fn"],
        "bwe_final_bias": bwe["final_bias"],
        "filter_length": vocoder_cfg["bwe"].get("n_fft", 512),
        "hop_length": vocoder_cfg["bwe"].get("hop_length", 80),
        "window_length": vocoder_cfg["bwe"].get("n_fft", 512),
        "num_mel_channels": vocoder_cfg["bwe"].get("num_mels", 64),
        "input_sampling_rate": vocoder_cfg["bwe"].get("input_sampling_rate", 16000),
        "output_sampling_rate": vocoder_cfg["bwe"].get("output_sampling_rate", 48000),
    }


def _single_vocoder_kwargs(config: dict[str, Any]) -> dict[str, Any]:
    return {
        "in_channels": 128,
        "hidden_channels": config.get("upsample_initial_channel", 1024),
        "out_channels": 2,
        "upsample_kernel_sizes": config.get("upsample_kernel_sizes", [16, 15, 8, 4, 4]),
        "upsample_factors": config.get("upsample_rates", [6, 5, 2, 2, 2]),
        "resnet_kernel_sizes": config.get("resblock_kernel_sizes", [3, 7, 11]),
        "resnet_dilations": config.get("resblock_dilation_sizes", [[1, 3, 5], [1, 3, 5], [1, 3, 5]]),
        "act_fn": config.get("activation", "snake" if config.get("resblock", "1") != "AMP1" else "snakebeta"),
        "final_act_fn": None if not config.get("use_tanh_at_final", True) else "tanh",
        "final_bias": config.get("use_bias_at_final", True),
        "output_sampling_rate": config.get("output_sampling_rate", 24000),
    }


def _map_audio_vae_key(key: str) -> str | None:
    if key.startswith("audio_vae.encoder."):
        return "encoder." + key.removeprefix("audio_vae.encoder.")
    if key.startswith("audio_vae.decoder."):
        return "decoder." + key.removeprefix("audio_vae.decoder.")
    if key in {"audio_vae.per_channel_statistics.mean", "audio_vae.per_channel_statistics.mean-of-means"}:
        return "latents_mean"
    if key in {"audio_vae.per_channel_statistics.std", "audio_vae.per_channel_statistics.std-of-means"}:
        return "latents_std"
    if key.startswith(("encoder.", "decoder.")) or key in {"latents_mean", "latents_std"}:
        return key
    return None


def _map_vocoder_key(key: str) -> str | None:
    if not key.startswith("vocoder."):
        return None
    key = key.removeprefix("vocoder.")
    return _map_single_vocoder_key(key)


def _map_single_vocoder_key(key: str) -> str:
    replacements = (
        ("conv_pre.", "conv_in."),
        ("conv_post.", "conv_out."),
        ("act_post.", "act_out."),
        ("ups.", "upsamplers."),
        ("resblocks.", "resnets."),
        ("downsample.lowpass.", "downsample."),
    )
    for old, new in replacements:
        key = key.replace(old, new)
    return key


def _load_mapped_state(module: nn.Module, checkpoint: dict[str, torch.Tensor], mapper) -> None:
    expected = module.state_dict()
    mapped = {}
    for key, value in checkpoint.items():
        mapped_key = mapper(key)
        if mapped_key in expected and expected[mapped_key].shape == value.shape:
            mapped[mapped_key] = value
    if not mapped:
        raise ValueError(f"No compatible NAVA audio checkpoint tensors found for {module.__class__.__name__}.")
    load_result = module.load_state_dict(mapped, strict=False)
    if load_result.missing_keys:
        missing = sorted(load_result.missing_keys)
        preview = ", ".join(missing[:20])
        if len(missing) > 20:
            preview += ", ..."
        logger.warning(
            "NAVA audio checkpoint loaded %d/%d tensors for %s; missing %d tensor(s): %s",
            len(mapped),
            len(expected),
            module.__class__.__name__,
            len(missing),
            preview,
        )


class NAVAAudioVAE(nn.Module):
    def __init__(self, model_root: str, config: NAVAConfig) -> None:
        super().__init__()
        self.sample_rate = config.audio_sample_rate
        self.ckpt_path = os.path.join(
            model_root, config.audio_vae_ckpt_dir, "LTX2", "ltx-2.3-22b-dev_audio_vae.safetensors"
        )
        if not os.path.exists(self.ckpt_path):
            raise FileNotFoundError(f"NAVA audio VAE checkpoint not found: {self.ckpt_path}")
        self.audio_vae, self.vocoder = self._load_components(self.ckpt_path)
        self.audio_vae.requires_grad_(False).eval()
        self.vocoder.requires_grad_(False).eval()

    def decode(self, audio_latents: torch.Tensor) -> torch.Tensor:
        latent_channels = int(self.audio_vae.config.latent_channels)
        if audio_latents.ndim == 3:
            batch, frames, channels = audio_latents.shape
            if channels % latent_channels != 0:
                raise ValueError(f"NAVA audio latent channels must be divisible by {latent_channels}, got {channels}.")
            audio_latents = self._denormalize_latents(audio_latents)
            mel_bins = channels // latent_channels
            audio_latents = audio_latents.reshape(batch, frames, latent_channels, mel_bins).permute(0, 2, 1, 3)
        elif audio_latents.ndim == 4:
            batch, channels, frames, mel_bins = audio_latents.shape
            audio_latents = audio_latents.permute(0, 2, 1, 3).reshape(batch, frames, channels * mel_bins)
            audio_latents = self._denormalize_latents(audio_latents)
            audio_latents = audio_latents.reshape(batch, frames, channels, mel_bins).permute(0, 2, 1, 3)
        else:
            raise ValueError(f"NAVA audio latents must be [B, T, C] or [B, C, T, M], got {audio_latents.shape}.")

        dtype = next(self.audio_vae.parameters()).dtype
        device = next(self.audio_vae.parameters()).device
        mel = self.audio_vae.decode(audio_latents.to(device=device, dtype=dtype), return_dict=False)[0]
        waveform = self.vocoder(mel).float()
        source_rate = int(getattr(self.vocoder, "output_sampling_rate", self.sample_rate))
        if source_rate != self.sample_rate:
            waveform = torchaudio.functional.resample(waveform, orig_freq=source_rate, new_freq=self.sample_rate)
        return waveform.clamp(-0.99, 0.99)

    def _denormalize_latents(self, audio_latents: torch.Tensor) -> torch.Tensor:
        mean = self.audio_vae.latents_mean.to(device=audio_latents.device, dtype=audio_latents.dtype)
        std = self.audio_vae.latents_std.to(device=audio_latents.device, dtype=audio_latents.dtype)
        if audio_latents.shape[-1] != mean.numel():
            raise ValueError(
                "NAVA audio latent channel count must match checkpoint statistics: "
                f"got {audio_latents.shape[-1]}, expected {mean.numel()}."
            )
        return audio_latents * std.view(1, 1, -1) + mean.view(1, 1, -1)

    @staticmethod
    def _load_components(checkpoint_path: str):
        from diffusers import AutoencoderKLLTX2Audio

        metadata = _read_safetensors_config(checkpoint_path)
        checkpoint = load_file(checkpoint_path, device="cpu")
        audio_vae = AutoencoderKLLTX2Audio(**_audio_vae_kwargs(metadata))
        with_bwe, vocoder_kwargs = _vocoder_kwargs(metadata)
        vocoder_cls = LTX2VocoderWithBWE if with_bwe else LTX2Vocoder
        vocoder = vocoder_cls(**vocoder_kwargs)
        _load_mapped_state(audio_vae, checkpoint, _map_audio_vae_key)
        _load_mapped_state(vocoder, checkpoint, _map_vocoder_key)
        return audio_vae, vocoder
