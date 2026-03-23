"""DAC codec encoder for Fish Speech S2 Pro voice cloning.

Encodes reference audio into VQ codes for use as prompt conditioning.
Runs on CPU in the API server process -- loaded lazily on first use.
"""

from __future__ import annotations

import os

import numpy as np
import torch
import torch.nn as nn
from vllm.logger import init_logger

from vllm_omni.model_executor.models.fish_speech.dac_utils import (
    DAC_SAMPLE_RATE,
    build_dac_codec,
)

logger = init_logger(__name__)

_codec_cache: dict[str, nn.Module] = {}


def _load_dac_codec(model_path: str) -> nn.Module:
    """Load the DAC codec model from codec.pth (cached per model_path)."""
    if model_path in _codec_cache:
        return _codec_cache[model_path]

    codec_path = os.path.join(model_path, "codec.pth")
    if not os.path.exists(codec_path):
        from transformers.utils.hub import cached_file

        cached = cached_file(model_path, "codec.pth")
        if cached is not None:
            codec_path = cached

    if not os.path.exists(codec_path):
        raise FileNotFoundError(
            f"codec.pth not found for {model_path}. Required for voice cloning with Fish Speech S2 Pro."
        )

    codec = build_dac_codec()

    state_dict = torch.load(codec_path, map_location="cpu", weights_only=True)
    if "generator" in state_dict:
        state_dict = state_dict["generator"]
    codec.load_state_dict(state_dict, strict=False)
    codec.eval()

    _codec_cache[model_path] = codec
    logger.info("Loaded DAC codec encoder from %s (CPU)", codec_path)
    return codec


def _resample(wav: np.ndarray, sr: int, target_sr: int) -> np.ndarray:
    """Resample audio using torchaudio's polyphase resampling."""
    if sr == target_sr:
        return wav
    import torchaudio

    wav_t = torch.from_numpy(wav).unsqueeze(0).float()
    wav_t = torchaudio.functional.resample(wav_t, sr, target_sr)
    return wav_t.squeeze(0).numpy()


@torch.no_grad()
def encode_reference_audio(
    model_path: str,
    wav_samples: list[float] | np.ndarray,
    sample_rate: int,
) -> list[int]:
    """Encode reference audio into semantic token IDs for prompt conditioning.

    Args:
        model_path: HuggingFace model path (for locating codec.pth).
        wav_samples: Audio waveform samples (mono, float).
        sample_rate: Sample rate of the input audio.

    Returns:
        List of semantic token IDs (151678 + code_value for each frame).
    """
    codec = _load_dac_codec(model_path)

    wav = np.asarray(wav_samples, dtype=np.float32)
    if wav.ndim > 1:
        wav = np.mean(wav, axis=-1)

    # Resample to DAC sample rate (44100).
    wav = _resample(wav, sample_rate, DAC_SAMPLE_RATE)

    # Encode: [1, 1, T] -> codes [1, num_codebooks, num_frames]
    wav_tensor = torch.from_numpy(wav).unsqueeze(0).unsqueeze(0).float()
    feature_lengths = torch.tensor([wav_tensor.shape[-1]])
    codes, feature_lengths_out = codec.encode(wav_tensor, feature_lengths)

    # Extract semantic codebook (index 0) - shape [num_frames].
    semantic_codes = codes[0, 0, :].tolist()

    # Convert to semantic token IDs: <|semantic:{i}|> = 151678 + i
    SEMANTIC_TOKEN_OFFSET = 151678
    semantic_token_ids = [SEMANTIC_TOKEN_OFFSET + int(c) for c in semantic_codes]

    logger.info(
        "Encoded reference audio: %d samples @ %dHz -> %d semantic tokens",
        len(wav_samples),
        sample_rate,
        len(semantic_token_ids),
    )
    return semantic_token_ids
