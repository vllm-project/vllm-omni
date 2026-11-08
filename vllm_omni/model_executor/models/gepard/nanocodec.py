# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""NeMo NanoCodec wrapper for Gepard — per-dimension FSQ decode.

Gepard's 32 FSQ heads emit per-dimension codes (8 groups x 4 dims), while stock
NeMo decode wants composed per-group mixed-radix indices. ``decode_from_codes``
dequantizes per-dim instead and hands the result to stock ``decode_audio`` — a
few lines on public NeMo API, no fork.

NeMo is a heavy optional dependency, imported lazily inside ``load``. Decode
only; ``encode_to_dequantized`` belongs to the voice-cloning follow-up.
"""

from __future__ import annotations

import os

import torch
import torch.nn as nn
from vllm.logger import init_logger

logger = init_logger(__name__)

# Left-context frames prepended to every streaming decode window. NanoCodec is
# convolutional, so decoding a bare chunk boundary yields a discontinuity; the
# extra frames are decoded for receptive-field context and trimmed afterwards.
# Mirrors the reference server's TTS_LOOKBACK_FRAMES.
LOOKBACK_FRAMES = int(os.environ.get("VLLM_GEPARD_LOOKBACK_FRAMES", "8"))

# End-of-clip shaping. The codec leaves the waveform on a non-zero sample and
# the hard step to silence is an audible click; a linear fade ramps it out.
# Both default to 0.0 (off), matching the reference server's defaults.
END_OF_SPEECH_FADE_MS = float(os.environ.get("VLLM_GEPARD_END_FADE_MS", "0.0"))
END_OF_SPEECH_SILENCE_MS = float(os.environ.get("VLLM_GEPARD_END_SILENCE_MS", "0.0"))


def apply_end_of_speech_tail(
    audio: torch.Tensor,
    sample_rate: int,
    fade_ms: float = END_OF_SPEECH_FADE_MS,
    silence_ms: float = END_OF_SPEECH_SILENCE_MS,
) -> torch.Tensor:
    """Fade the final ``fade_ms`` to zero, then append ``silence_ms`` of pad.

    The fade is what removes the end-of-clip click; padding alone does not.
    """
    if audio.numel() == 0 or (fade_ms <= 0 and silence_ms <= 0):
        return audio
    out = audio
    if fade_ms > 0:
        n = min(out.shape[0], int(sample_rate * fade_ms / 1000.0))
        if n > 0:
            # Copy: the caller's tensor is still referenced by the audio queue.
            out = out.clone()
            out[-n:] *= torch.linspace(1.0, 0.0, n, dtype=out.dtype, device=out.device)
    if silence_ms > 0:
        pad = torch.zeros(int(sample_rate * silence_ms / 1000.0), dtype=out.dtype, device=out.device)
        out = torch.cat([out, pad], dim=0)
    return out


def _make_unfolded_codec_cls():
    """Build the ``UnfoldedCodecModel`` subclass. Imports NeMo (optional dep).

    Defined lazily so importing this module never requires NeMo.  The subclass
    is OUR code layered on stock ``AudioCodecModel`` — not a NeMo fork.
    """
    try:
        from nemo.collections.tts.models import AudioCodecModel  # noqa: PLC0415
        from omegaconf import open_dict  # noqa: PLC0415
    except ImportError as e:  # pragma: no cover - env-dependent
        raise ImportError(
            "Gepard TTS needs NeMo for the NanoCodec decoder. Install it with "
            "`pip install nemo_toolkit[tts]` (heavy; NVIDIA Open Model License, "
            "commercial use OK), or gate the model behind that extra."
        ) from e

    class UnfoldedCodecModel(AudioCodecModel):
        """AudioCodecModel + direct decode from per-dimension FSQ codes."""

        def __init__(self, cfg, trainer=None):
            # SLMDiscriminator pulls microsoft/wavlm-base-plus (~360 MB) and is
            # training-only — strip it from the config before init.
            with open_dict(cfg):
                disc = cfg.get("discriminator", None)
                if disc is not None and "discriminators" in disc:
                    disc.discriminators = [d for d in disc.discriminators if "SLM" not in d._target_]
            super().__init__(cfg, trainer)

        def decode_from_codes(self, codes: torch.Tensor, codes_len: torch.Tensor):
            """Decode per-dimension discrete FSQ codes to a waveform.

            codes: (B, D, T) per-dim discrete values, D = num_groups * dims.
            codes_len: (B,) valid frame count per batch element.
            Returns (audio (B, T_audio), audio_len (B,)).
            """
            num_levels = self.vector_quantizer.fsqs[0].num_levels.squeeze()
            scale = (num_levels // 2).float().to(codes.device)  # e.g. [4,3,3,3]
            groups = codes.chunk(self.vector_quantizer.num_groups, dim=1)
            dequantized = torch.cat(
                [(g - scale[None, :, None]) / scale[None, :, None] for g in groups],
                dim=1,
            )
            return self.decode_audio(inputs=dequantized, input_len=codes_len)

    return UnfoldedCodecModel


class NanoCodec(nn.Module):
    """Thin wrapper owning the (optional) NeMo NanoCodec decoder.

    Constructing this is cheap and NeMo-free; ``load`` does the heavy import +
    weight load so ``load_format=dummy`` and NeMo-less environments don't pay
    for it at model construction.
    """

    def __init__(self, codec_id: str, sample_rate: int) -> None:
        super().__init__()
        self.codec_id = codec_id
        self.sample_rate = sample_rate
        self._codec: nn.Module | None = None

    @property
    def is_loaded(self) -> bool:
        return self._codec is not None

    def load(self, device: torch.device) -> None:
        """Import NeMo, build the UnfoldedCodecModel, move to device, eval()."""
        cls = _make_unfolded_codec_cls()
        codec = cls.from_pretrained(self.codec_id)
        codec.to(device=device)
        codec.eval()
        self._codec = codec
        logger.info("Gepard NanoCodec loaded from %s on %s", self.codec_id, device)

    @torch.inference_mode()
    def decode_from_codes(self, codes: torch.Tensor, codes_len: torch.Tensor) -> torch.Tensor:
        """(B, 32, T) per-dim codes -> waveform (B, T_audio). Returns audio only."""
        if self._codec is None:
            raise RuntimeError("NanoCodec.decode_from_codes called before load()")
        audio, _audio_len = self._codec.decode_from_codes(codes, codes_len)
        return audio

    @torch.inference_mode()
    def decode_stream(
        self,
        frames: torch.Tensor,
        *,
        start_idx: int,
        lookback: int = LOOKBACK_FRAMES,
        is_final: bool = False,
    ) -> torch.Tensor | None:
        """Decode the tail of a ``(T, 32)`` code history past ``start_idx``.

        The window starts ``lookback`` frames earlier for receptive-field
        context and those samples are trimmed after, so successive calls
        concatenate to a single whole-history decode. None if nothing is new.
        """
        count = int(frames.shape[0])
        if count <= start_idx:
            return None
        decode_start = max(0, start_idx - lookback)
        window = frames[decode_start:count]  # (T_win, 32)
        codes = window.t().unsqueeze(0).contiguous().long()  # (1, 32, T_win)
        codes_len = torch.full((1,), codes.shape[-1], dtype=torch.long, device=codes.device)
        audio = self.decode_from_codes(codes, codes_len).reshape(-1)
        # Derive samples_per_frame from this decode rather than assuming it.
        samples_per_frame = audio.shape[0] // (count - decode_start)
        audio = audio[(start_idx - decode_start) * samples_per_frame :]
        if is_final:
            audio = apply_end_of_speech_tail(audio, self.sample_rate)
        return audio
