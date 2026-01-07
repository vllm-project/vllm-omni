from io import BytesIO

import numpy as np
from vllm.logger import init_logger

from vllm_omni.entrypoints.openai.protocol.audio import AudioResponse, CreateAudio

try:
    import soundfile
except ImportError:
    soundfile = None

try:
    import librosa
except ImportError:
    librosa = None

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
        """Apply speed adjustment to the audio tensor while preserving pitch."""
        if speed == 1.0:
            return audio_tensor, sample_rate

        if librosa is None:
            raise ImportError("librosa is required for speed adjustment. Please install it with: pip install librosa")

        try:
            # librosa.effects.time_stretch requires a float audio tensor.
            if not np.issubdtype(audio_tensor.dtype, np.floating):
                audio_tensor = audio_tensor.astype(np.float32)

            stretched_audio = librosa.effects.time_stretch(y=audio_tensor, rate=speed)
            return stretched_audio, sample_rate
        except Exception as e:
            logger.error(f"An error occurred during speed adjustment: {e}")
            raise ValueError("Failed to apply speed adjustment.") from e
