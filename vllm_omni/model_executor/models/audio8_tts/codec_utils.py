# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Loading / caching helpers for the Audio8 TTS ``codec.pth`` checkpoint.

Both stages need the codec but for opposite halves: Stage 0 encodes reference
audio for zero-shot voice cloning, Stage 1 decodes generated codes to waveform.
The unused half is pruned before the module is moved to the device so a stage
never pays GPU memory for weights it cannot reach.
"""

from __future__ import annotations

import os
from functools import lru_cache

import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils.parametrize import remove_parametrizations
from vllm.logger import init_logger

from vllm_omni.model_executor.models.audio8_tts.codec import ArkttsCodec, build_arktts_codec
from vllm_omni.model_executor.models.audio8_tts.configuration_audio8_tts import (
    ARKTTS_CODEC_FRAME_SIZE,
    ARKTTS_CODEC_SAMPLE_RATE,
)

logger = init_logger(__name__)

CODEC_FILENAME = "codec.pth"

_codec_cache: dict[tuple[str, str, str, str], ArkttsCodec] = {}


def resolve_codec_path(model_path: str, filename: str = CODEC_FILENAME) -> str:
    """Return a local path to ``filename``, downloading from the Hub if needed."""
    local = os.path.join(model_path, filename)
    if os.path.exists(local):
        return local

    from transformers.utils.hub import cached_file

    cached = cached_file(model_path, filename)
    if cached is None or not os.path.exists(cached):
        raise FileNotFoundError(
            f"{filename} not found for {model_path}; the Audio8 TTS checkpoint must ship its neural audio codec."
        )
    return cached


def bake_weight_norm(module: nn.Module) -> int:
    """Fold weight-norm parametrisations into plain weights (inference only)."""
    baked = 0
    for submodule in module.modules():
        parametrizations = getattr(submodule, "parametrizations", None)
        if not parametrizations:
            continue
        for name in list(parametrizations.keys()):
            remove_parametrizations(submodule, name, leave_parametrized=True)
            baked += 1
    return baked


def load_arktts_codec(
    model_path: str,
    *,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float32,
    role: str = "decode",
    post_n_layer: int = 8,
    post_n_head: int = 16,
    post_n_local_heads: int = 8,
    post_intermediate_size: int = 1216,
) -> ArkttsCodec:
    """Load ``codec.pth`` for ``role`` in {"encode", "decode", "both"}.

    Instances are cached per (path, device, dtype, role) because both the Stage-0
    voice-clone path and the Stage-1 decoder may ask for the same codec many
    times per request.
    """
    if role not in {"encode", "decode", "both"}:
        raise ValueError(f"role must be encode/decode/both, got {role!r}")
    device = torch.device(device)
    cache_key = (model_path, str(device), str(dtype), role)
    cached_codec = _codec_cache.get(cache_key)
    if cached_codec is not None:
        return cached_codec

    codec_path = resolve_codec_path(model_path)
    codec = build_arktts_codec(
        post_n_layer=post_n_layer,
        post_n_head=post_n_head,
        post_n_local_heads=post_n_local_heads,
        post_intermediate_size=post_intermediate_size,
    )

    state = torch.load(codec_path, map_location="cpu", weights_only=True)
    if "state_dict" in state:
        state = state["state_dict"]
    if any("generator." in key for key in state):
        state = {key.replace("generator.", ""): value for key, value in state.items() if "generator." in key}
    state = {key: value for key, value in state.items() if not key.endswith(("freqs_cis", "causal_mask"))}
    codec.load_state_dict(state, strict=True)

    baked = bake_weight_norm(codec)
    if role == "encode":
        codec.decoder = None
    elif role == "decode":
        codec.encoder = None
        codec.quantizer.pre_module = None
        codec.quantizer.downsample = None

    codec = codec.to(device=device, dtype=dtype)
    codec.eval()
    _codec_cache[cache_key] = codec
    logger.info(
        "Loaded Audio8 TTS codec from %s (role=%s, device=%s, dtype=%s, baked_weight_norms=%d)",
        codec_path,
        role,
        device,
        dtype,
        baked,
    )
    return codec


@lru_cache(maxsize=8)
def _resample_transform(source_sr: int, target_sr: int, device: torch.device, dtype: torch.dtype):
    import torchaudio

    return torchaudio.transforms.Resample(source_sr, target_sr).to(device=device, dtype=dtype)


def prepare_reference_waveform(
    wav_samples: list[float] | np.ndarray | torch.Tensor,
    sample_rate: int,
    *,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Down-mix to mono and resample reference audio to the codec's rate.

    Returns a 1-D waveform tensor.
    """
    if isinstance(wav_samples, torch.Tensor):
        wav = wav_samples.detach()
    else:
        wav = torch.as_tensor(np.asarray(wav_samples))
    wav = wav.to(device=device, dtype=dtype)

    if wav.ndim == 2:
        # Accept [channels, samples] and [samples, channels].
        if wav.shape[0] <= 8 and wav.shape[1] > wav.shape[0]:
            wav = wav.mean(dim=0)
        elif wav.shape[-1] <= 8 and wav.shape[0] > wav.shape[-1]:
            wav = wav.mean(dim=-1)
        else:
            wav = wav.mean(dim=0)
    elif wav.ndim > 2:
        wav = wav.reshape(-1, wav.shape[-1]).mean(dim=0)
    wav = wav.reshape(-1)

    if wav.numel() == 0:
        raise ValueError("Reference audio must not be empty")
    if int(sample_rate) <= 0:
        raise ValueError(f"Reference audio sample rate must be positive, got {sample_rate}")
    if int(sample_rate) != ARKTTS_CODEC_SAMPLE_RATE:
        transform = _resample_transform(int(sample_rate), ARKTTS_CODEC_SAMPLE_RATE, wav.device, wav.dtype)
        wav = transform(wav.unsqueeze(0)).squeeze(0)
    return wav.contiguous()


@torch.no_grad()
def encode_reference_audio_codes(
    model_path: str,
    wav_samples: list[float] | np.ndarray | torch.Tensor,
    sample_rate: int,
    *,
    device: torch.device | str,
    **codec_kwargs: int,
) -> torch.Tensor:
    """Encode reference audio into codec codes.

    Returns:
        ``[frames, num_codebooks]`` int64 codes on ``device``.
    """
    device = torch.device(device)
    codec = load_arktts_codec(model_path, device=device, dtype=torch.float32, role="encode", **codec_kwargs)
    wav = prepare_reference_waveform(wav_samples, sample_rate, device=device)
    lengths = torch.tensor([wav.numel()], device=device, dtype=torch.long)
    codes, code_lengths = codec.encode(wav.reshape(1, 1, -1), lengths)
    frames = int(code_lengths[0].item())
    codes_fq = codes[0, :, :frames].transpose(0, 1).to(dtype=torch.long).contiguous()
    logger.info(
        "Encoded Audio8 TTS reference audio: %d samples @ %d Hz -> frames=%d codebooks=%d",
        int(wav.numel()),
        int(sample_rate),
        int(codes_fq.shape[0]),
        int(codes_fq.shape[1]),
    )
    return codes_fq


def estimate_reference_code_frames(num_samples: int, sample_rate: int) -> int:
    """Frames the codec will emit for ``num_samples`` at ``sample_rate``.

    Used by the serving layer to size the prompt placeholder without paying for
    a codec encode on the API thread.
    """
    if num_samples <= 0 or sample_rate <= 0:
        raise ValueError("Reference audio must have a positive length and sample rate")
    resampled = max(1, -(-num_samples * ARKTTS_CODEC_SAMPLE_RATE // sample_rate))
    return max(1, -(-resampled // ARKTTS_CODEC_FRAME_SIZE))


__all__ = [
    "CODEC_FILENAME",
    "bake_weight_norm",
    "encode_reference_audio_codes",
    "estimate_reference_code_frames",
    "load_arktts_codec",
    "prepare_reference_waveform",
    "resolve_codec_path",
]
