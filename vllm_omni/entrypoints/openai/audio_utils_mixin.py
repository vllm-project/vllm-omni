from io import BytesIO

import numpy as np
import torch
import torchaudio
from vllm.logger import init_logger

from vllm_omni.entrypoints.openai.protocol.audio import AudioResponse, CreateAudio

try:
    import soundfile
except ImportError:
    soundfile = None

logger = init_logger(__name__)


class AudioMixin:
    """Mixin class to add audio-related utilities."""

    def create_audio(self, audio_obj: CreateAudio) -> AudioResponse:
        """Convert audio tensor to bytes in the specified format."""

        audio_tensor = audio_obj.audio_tensor
        sample_rate = audio_obj.sample_rate
        response_format = audio_obj.response_format.lower()
        stream_format = audio_obj.stream_format
        base64_encode = audio_obj.base64_encode
        speed = audio_obj.speed

        if stream_format != "audio":
            raise ValueError(f"Unsupported stream format: {stream_format}")

        if soundfile is None:
            raise ImportError(
                "soundfile is required for audio generation. Please install it with: pip install soundfile"
            )

        if audio_tensor.ndim > 2:
            raise ValueError(
                f"Unsupported audio tensor dimension: {audio_tensor.ndim}. "
                "Only mono (1D) and stereo (2D) are supported."
            )

        audio_tensor, sample_rate = self._apply_speed_adjustment(audio_tensor, speed, sample_rate)

        supported_formats = {
            "wav": ("WAV", "audio/wav", {}),
            "pcm": ("RAW", "audio/pcm", {"subtype": "PCM_16"}),
            "flac": ("FLAC", "audio/flac", {}),
            "mp3": ("MP3", "audio/mpeg", {}),
            "aac": ("AAC", "audio/aac", {}),
            "opus": ("OGG", "audio/ogg", {"subtype": "OPUS"}),
        }

        if response_format not in supported_formats:
            logger.warning(f"Unsupported response format '{response_format}', defaulting to 'wav'.")
            response_format = "wav"

        soundfile_format, media_type, kwargs = supported_formats[response_format]

        with BytesIO() as buffer:
            soundfile.write(buffer, audio_tensor, sample_rate, format=soundfile_format, **kwargs)
            audio_data = buffer.getvalue()

        if base64_encode:
            import base64

            audio_data = base64.b64encode(audio_data).decode("utf-8")

        return AudioResponse(audio_data=audio_data, media_type=media_type)

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

            # Match librosa.stft defaults: n_fft=2048, hop_length=n_fft//4
            n_fft = 2048
            hop_length = n_fft // 4
            to_spec = torchaudio.transforms.Spectrogram(
                n_fft=n_fft,
                hop_length=hop_length,
                power=None,
            )
            stretch = torchaudio.transforms.TimeStretch(
                n_freq=n_fft // 2 + 1,
                hop_length=hop_length,
            )
            to_wave = torchaudio.transforms.InverseSpectrogram(
                n_fft=n_fft,
                hop_length=hop_length,
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
