# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from __future__ import annotations

import math
from io import BytesIO
from typing import Any

import torch

from .config_ming_tts import (
    AUDIO_FRAME_HOP,
    LATENT_DIM,
    PATCH_SIZE,
    SAMPLE_RATE,
    VAE_PATCH_SIZE,
    VISION_START_TOKEN_ID,
    MingTTSConfig,
)


def pad_prompt_waveform(
    waveform: Any,
    *,
    patch_size: int = PATCH_SIZE,
    sample_rate: int = SAMPLE_RATE,
) -> torch.Tensor:
    tensor = coerce_prompt_waveform(waveform)
    pad_align = int((float(sample_rate) / 12.5) * int(patch_size))
    new_len = ((int(tensor.shape[-1]) + pad_align - 1) // pad_align) * pad_align
    if new_len == int(tensor.shape[-1]):
        return tensor
    padded = torch.zeros((1, new_len), dtype=tensor.dtype, device=tensor.device)
    padded[:, : tensor.shape[-1]] = tensor
    return padded


def coerce_prompt_waveform(value: Any) -> torch.Tensor:
    if value is None:
        raise ValueError("prompt waveform cannot be None")
    if isinstance(value, torch.Tensor):
        tensor = value.detach()
        if tensor.ndim == 1:
            return tensor.unsqueeze(0).to(torch.float32)
        if tensor.ndim == 2:
            if tensor.shape[0] != 1:
                return tensor.reshape(1, -1).to(torch.float32)
            return tensor.to(torch.float32)
        raise ValueError(f"Unsupported Ming prompt waveform rank: {tuple(tensor.shape)}")
    if isinstance(value, (list, tuple)):
        parts = [coerce_prompt_waveform(item) for item in value if item is not None]
        if not parts:
            raise ValueError("prompt waveform list was empty")
        return torch.cat(parts, dim=-1)
    return coerce_prompt_waveform(torch.as_tensor(value))


def coerce_speaker_embeddings(value: Any, *, use_zero_spk_emb: bool = False) -> list[torch.Tensor] | None:
    if value is None:
        return [torch.zeros((192,), dtype=torch.float32)] if use_zero_spk_emb else None
    if isinstance(value, torch.Tensor):
        tensor = value.detach()
        if tensor.ndim == 1:
            tensor = tensor.unsqueeze(0)
        if tensor.ndim != 2:
            raise ValueError(f"Unsupported Ming speaker embedding shape: {tuple(tensor.shape)}")
        items = [row.reshape(-1).to(torch.float32).cpu() for row in tensor]
    elif isinstance(value, (list, tuple)):
        if value and all(not isinstance(item, (list, tuple, torch.Tensor)) for item in value):
            items = [torch.as_tensor(value).detach().reshape(-1).to(torch.float32).cpu()]
        else:
            items = []
            for item in value:
                if item is None:
                    continue
                if not isinstance(item, torch.Tensor):
                    item = torch.as_tensor(item)
                items.append(item.detach().reshape(-1).to(torch.float32).cpu())
    else:
        return coerce_speaker_embeddings(torch.as_tensor(value), use_zero_spk_emb=use_zero_spk_emb)
    if not items:
        return [torch.zeros((192,), dtype=torch.float32)] if use_zero_spk_emb else None
    for item in items:
        if int(item.numel()) != 192:
            raise ValueError(f"Ming speaker embedding must have 192 dims, got {int(item.numel())}")
    return items


def count_prompt_latent_patches(
    value: Any,
    *,
    patch_size: int = PATCH_SIZE,
    latent_dim: int = LATENT_DIM,
) -> int:
    if value is None:
        return 0
    if not isinstance(value, torch.Tensor):
        value = torch.as_tensor(value)
    latents = value.detach()
    if latents.ndim == 3 and latents.shape[0] == 1:
        latents = latents.squeeze(0)
    if latents.ndim == 3 and latents.shape[-2:] == (patch_size, latent_dim):
        return int(latents.shape[0])
    if latents.ndim != 2 or latents.shape[-1] != latent_dim:
        raise ValueError(f"Unsupported Ming prompt_latents shape: {tuple(latents.shape)}")
    if latents.shape[0] % patch_size != 0:
        raise ValueError(
            f"Ming prompt_latents frame count must be divisible by patch_size={patch_size}, "
            f"got frames={int(latents.shape[0])}"
        )
    return int(latents.shape[0] // patch_size)


def count_prompt_waveform_patches(
    value: Any,
    *,
    patch_size: int = PATCH_SIZE,
    frame_hop: int = AUDIO_FRAME_HOP,
    vae_patch_size: int = VAE_PATCH_SIZE,
) -> int:
    if value is None:
        return 0
    waveform = pad_prompt_waveform(value, patch_size=patch_size)
    frame_count = int(math.ceil(float(waveform.shape[-1]) / float(frame_hop)))
    latent_frames = int(math.ceil(float(frame_count) / float(vae_patch_size)))
    if latent_frames % int(patch_size) != 0:
        raise ValueError(
            f"Ming prompt waveform produced latent frame count not divisible by patch_size={patch_size}: "
            f"frames={latent_frames}"
        )
    return int(latent_frames // int(patch_size))


def _normalize_prompt_waveform(value: Any, *, target_sr: int) -> torch.Tensor:
    if isinstance(value, bytes):
        import torchaudio

        waveform, sr = torchaudio.load(BytesIO(value))
        waveform = waveform[:1].to(torch.float32)
        if int(sr) != int(target_sr):
            from torchaudio.functional import resample as resample_audio

            waveform = resample_audio(waveform, int(sr), int(target_sr))
        return waveform

    if isinstance(value, tuple) and len(value) == 2 and isinstance(value[1], int):
        waveform = coerce_prompt_waveform(value[0])
        if int(value[1]) != int(target_sr):
            from torchaudio.functional import resample as resample_audio

            waveform = resample_audio(waveform, int(value[1]), int(target_sr))
        return waveform

    if isinstance(value, dict):
        samples = value.get("samples", value.get("array", value.get("waveform")))
        sr = value.get("sample_rate", value.get("sr", target_sr))
        return _normalize_prompt_waveform((samples, int(sr)), target_sr=target_sr)

    return coerce_prompt_waveform(value)


def _coerce_prompt_latents(
    value: Any,
    *,
    patch_size: int,
    latent_dim: int,
) -> dict[str, torch.Tensor] | None:
    if value is None:
        return None
    if not isinstance(value, torch.Tensor):
        value = torch.as_tensor(value)

    latents = value.detach()
    if latents.ndim == 3 and latents.shape[0] == 1:
        latents = latents.squeeze(0)

    if latents.ndim == 3 and latents.shape[-2:] == (patch_size, latent_dim):
        patches = latents
        # [Patch, Time, Dimension] -> [Frame, Dimension] for history seeding.
        frames = patches.reshape(-1, latent_dim)
        return {"patches": patches, "frames": frames}

    if latents.ndim != 2 or latents.shape[-1] != latent_dim:
        raise ValueError(f"Unsupported prompt latent shape: {tuple(latents.shape)}")
    if latents.shape[0] % patch_size != 0:
        raise ValueError(
            f"Prompt latent frame count must be divisible by patch_size={patch_size}, "
            f"got frames={int(latents.shape[0])}"
        )
    # [Frame, Dimension] -> [Patch, Time, Dimension] for Aggregator prompt slots.
    patches = latents.reshape(-1, patch_size, latent_dim) if latents.shape[0] > 0 else None
    return {"patches": patches, "frames": latents}


def _initial_history(
    frames: torch.Tensor | None,
    *,
    history_size: int,
    latent_dim: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    history = torch.zeros((history_size, latent_dim), device=device, dtype=dtype)
    if frames is None or frames.numel() == 0:
        return history
    frames = frames.to(device=device, dtype=dtype)
    take = min(history_size, int(frames.shape[0]))
    history[-take:] = frames[-take:]
    return history


def _take_scalar(value: Any, idx: int) -> float | None:
    if not isinstance(value, torch.Tensor) or value.numel() == 0:
        return None
    return float(value.reshape(-1)[idx].item())


def _find_audio_placeholder_positions(input_ids: torch.Tensor, cfg: MingTTSConfig) -> torch.Tensor:
    dummy_pos = (input_ids == cfg.audio_dummy_token_id).nonzero(as_tuple=True)[0]
    if dummy_pos.numel() == 0:
        return dummy_pos

    audio_start_pos = (input_ids == cfg.audio_start_token_id).nonzero(as_tuple=True)[0]
    audio_end_pos = (input_ids == cfg.audio_end_token_id).nonzero(as_tuple=True)[0]
    if audio_start_pos.numel() == 0:
        return dummy_pos

    start = int(audio_start_pos[0].item())
    end = int(audio_end_pos[0].item()) if audio_end_pos.numel() > 0 else int(input_ids.shape[0])
    keep = (dummy_pos > start) & (dummy_pos < end)
    filtered = dummy_pos[keep]
    return filtered if filtered.numel() > 0 else dummy_pos


def _find_speaker_placeholder_positions(input_ids: torch.Tensor, hf_config: Any) -> list[int]:
    vision_start_token_id = getattr(hf_config, "vision_start_token_id", VISION_START_TOKEN_ID)
    vision_start_pos = (input_ids == int(vision_start_token_id)).nonzero(as_tuple=True)[0]
    if vision_start_pos.numel() == 0:
        return []

    slots = []
    for pos in vision_start_pos:
        slot = int(pos.item()) + 1
        if slot < int(input_ids.shape[0]):
            slots.append(slot)
    return slots


__all__ = [
    "coerce_prompt_waveform",
    "coerce_speaker_embeddings",
    "count_prompt_latent_patches",
    "count_prompt_waveform_patches",
    "pad_prompt_waveform",
    "_coerce_prompt_latents",
    "_find_audio_placeholder_positions",
    "_find_speaker_placeholder_positions",
    "_initial_history",
    "_normalize_prompt_waveform",
    "_take_scalar",
]
