# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Audio serving utility mixin.

PUT HERE:
  - AudioMixin helpers that convert audio tensors/bytes/formats for speech
    and audio serving classes (shared by multiple serving paths).

DO NOT PUT HERE:
  - FastAPI route / form / job orchestration peeled from ``api_server.py``.
    Put those under an audio package ``helpers.py`` (or audio serving modules)
    when extracted.

LONGEVITY:
  - This root mixin is a **temporary shared home**.
  - TODO(#5227, P1.1): tidy up / move with the audio/speech family split
    in the Phase 1 audio PR; do not treat this file as the long-term owner.
  - Endpoint-family helpers under an audio package are the longer home for
    route-adjacent logic.

See ``openai/README.md`` (utils vs helpers, no overlap).
"""

from io import BytesIO

import numpy as np
import torch
import torchaudio
from vllm.logger import init_logger

from vllm_omni.entrypoints.openai.protocol.audio import (
    DEFAULT_AUDIO_FORMAT,
    AudioChunkMetadata,
    AudioResponse,
    CreateAudio,
)

# Re-exported: serving_speech and the audio tests import it from here.
from vllm_omni.utils.audio_resample import StreamingAudioResampler  # noqa: F401

try:
    import soundfile
except ImportError:
    soundfile = None

logger = init_logger(__name__)


class AudioMixin:
    """Mixin class to add audio-related utilities."""

    def _normalize_audio_tensor_for_output(self, audio_tensor) -> np.ndarray:
        """Normalize mono/stereo audio into a shape soundfile can write reliably."""
        audio_tensor = np.asarray(audio_tensor)

        if audio_tensor.ndim == 0:
            return audio_tensor.reshape(1)
        if audio_tensor.ndim == 1:
            return audio_tensor
        if audio_tensor.ndim == 2:
            if 1 in audio_tensor.shape:
                return audio_tensor.reshape(-1)
            return audio_tensor

        # Higher-rank tensors (e.g. [1, 1, T] from diffusion outputs): squeeze
        # trailing/leading singleton dims first — the historical behavior here
        # was a plain .squeeze() — and only fail if the result is still not
        # mono/stereo.
        squeezed = audio_tensor.squeeze()
        if squeezed.ndim <= 2:
            return self._normalize_audio_tensor_for_output(squeezed)
        raise ValueError(
            f"Unsupported audio tensor dimension: {audio_tensor.ndim}. Only mono (1D) and stereo (2D) are supported."
        )

    def create_audio(self, audio_obj: CreateAudio) -> AudioResponse:
        """Convert audio tensor to bytes in the specified format."""

        audio_tensor = self._normalize_audio_tensor_for_output(audio_obj.audio_tensor)
        sample_rate = audio_obj.sample_rate
        response_format = audio_obj.response_format.lower()
        base64_encode = audio_obj.base64_encode
        speed = audio_obj.speed

        if soundfile is None:
            raise ImportError(
                "soundfile is required for audio generation. Please install it with: pip install soundfile"
            )

        if audio_tensor.ndim > 2:
            raise ValueError(
                f"Unsupported audio tensor dimension: {audio_tensor.ndim}. "
                "Only mono (1D) and stereo (2D) are supported."
            )

        if audio_tensor.ndim == 2 and audio_tensor.shape[0] == 2:
            # Convert from [channels, samples] to [samples, channels]
            audio_tensor = audio_tensor.T

        audio_tensor, sample_rate = self._apply_speed_adjustment(audio_tensor, speed, sample_rate)

        if audio_obj.output_sample_rate is not None and audio_obj.output_sample_rate != sample_rate:
            audio_tensor = self._resample_audio(audio_tensor, sample_rate, audio_obj.output_sample_rate)
            sample_rate = audio_obj.output_sample_rate

        supported_formats = {
            "wav": ("WAV", "audio/wav", {}),
            "pcm": ("RAW", "audio/pcm", {"subtype": "PCM_16"}),
            "flac": ("FLAC", "audio/flac", {}),
            "mp3": ("MP3", "audio/mpeg", {}),
            "opus": ("OGG", "audio/ogg", {"subtype": "OPUS"}),
        }

        if response_format not in supported_formats:
            logger.warning(f"Unsupported response format '{response_format}', defaulting to '{DEFAULT_AUDIO_FORMAT}'.")
            response_format = DEFAULT_AUDIO_FORMAT

        soundfile_format, media_type, kwargs = supported_formats[response_format]

        with BytesIO() as buffer:
            soundfile.write(buffer, audio_tensor, sample_rate, format=soundfile_format, **kwargs)
            audio_bytes = buffer.getvalue()

        output_data: bytes | str = audio_bytes
        if base64_encode:
            import base64

            output_data = base64.b64encode(audio_bytes).decode("utf-8")

        return AudioResponse(
            audio_data=output_data,
            media_type=media_type,
            audio_metadata=AudioChunkMetadata(
                format=response_format,
                sample_rate_hz=int(sample_rate),
                frame_count=int(audio_tensor.shape[0]),
                channels=1 if audio_tensor.ndim == 1 else int(audio_tensor.shape[1]),
            ),
        )

    @staticmethod
    def _resample_audio(audio_tensor: np.ndarray, source_rate: int, target_rate: int) -> np.ndarray:
        """Resample complete audio while preserving soundfile's channels-last layout."""
        if source_rate == target_rate:
            return audio_tensor

        audio_array = np.asarray(audio_tensor)
        if not np.issubdtype(audio_array.dtype, np.floating):
            audio_array = audio_array.astype(np.float32)
        waveform = torch.from_numpy(audio_array.T.copy() if audio_array.ndim == 2 else audio_array.copy())
        resampled = torchaudio.functional.resample(waveform, source_rate, target_rate).cpu().numpy()
        return resampled.T if audio_array.ndim == 2 else resampled

    def _apply_speed_adjustment(self, audio_tensor: np.ndarray, speed: float, sample_rate: int):
        """Apply speed adjustment to the audio tensor while preserving pitch.

        Uses torchaudio's phase vocoder (Spectrogram → TimeStretch →
        InverseSpectrogram) to stretch/compress audio in time without
        changing pitch.
        """
        if speed == 1.0:
            return audio_tensor, sample_rate

        try:
            if not np.issubdtype(audio_tensor.dtype, np.floating):
                audio_tensor = audio_tensor.astype(np.float32)

            # Stereo numpy arrays use channels-last (T, C);
            # torch expects channels-first (C, T).
            channels_last = audio_tensor.ndim == 2
            if channels_last:
                waveform = torch.from_numpy(audio_tensor.T)
            else:
                waveform = torch.from_numpy(audio_tensor).unsqueeze(0)

            # Use a speech-sized analysis window. The previous 2048-sample
            # window is tuned for music and can smear short consonants after
            # aggressive compression, which makes ASR transcript checks flaky.
            n_fft = 768
            hop_length = n_fft // 4
            window = torch.hann_window(n_fft, device=waveform.device, dtype=waveform.dtype)
            to_spec = torchaudio.transforms.Spectrogram(
                n_fft=n_fft,
                hop_length=hop_length,
                window_fn=lambda *_args, **_kwargs: window,
                power=None,
            )
            stretch = torchaudio.transforms.TimeStretch(
                n_freq=n_fft // 2 + 1,
                hop_length=hop_length,
            )
            to_wave = torchaudio.transforms.InverseSpectrogram(
                n_fft=n_fft,
                hop_length=hop_length,
                window_fn=lambda *_args, **_kwargs: window,
            )

            spec = to_spec(waveform)
            stretched = stretch(spec, speed)
            expected_length = int(audio_tensor.shape[0] / speed)
            result = to_wave(stretched, length=expected_length)

            result = result.squeeze(0).numpy()
            if channels_last:
                result = result.T
            return result, sample_rate
        except Exception as e:
            logger.error(f"An error occurred during speed adjustment: {e}")
            raise ValueError("Failed to apply speed adjustment.") from e
