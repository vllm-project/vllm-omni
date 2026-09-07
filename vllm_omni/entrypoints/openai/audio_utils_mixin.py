# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from io import BytesIO

import numpy as np
import torch
import torchaudio
from vllm.logger import init_logger

from vllm_omni.entrypoints.openai.protocol.audio import DEFAULT_AUDIO_FORMAT, AudioResponse, CreateAudio

try:
    import soundfile
except ImportError:
    soundfile = None

logger = init_logger(__name__)


class StreamingAudioResampler:
    """Stateful resampler for streaming mono audio.

    Only integer downsampling ratios are supported so every input chunk can
    produce output without buffering the complete stream.
    """

    def __init__(self, source_rate: int, target_rate: int):
        if source_rate <= 0 or target_rate <= 0:
            raise ValueError("Audio sample rates must be positive")
        self.source_rate = source_rate
        self.target_rate = target_rate
        if source_rate < target_rate or source_rate % target_rate != 0:
            raise ValueError(
                "Streaming audio resampling requires an integer downsampling ratio, "
                f"got {source_rate} Hz to {target_rate} Hz"
            )

        self._ratio = source_rate // target_rate
        self._buffer = np.empty((0,), dtype=np.float32)
        self._buffer_start = 0
        self._total_samples = 0
        self._next_center = 0

        if self._ratio == 1:
            self._taps = np.ones((1,), dtype=np.float32)
        else:
            # Use a Kaiser-windowed sinc and leave a small transition band
            # below the target Nyquist frequency.
            num_taps = 20 * self._ratio + 1
            half = num_taps // 2
            offsets = np.arange(num_taps, dtype=np.float64) - half
            cutoff = 0.95 / self._ratio
            taps = cutoff * np.sinc(cutoff * offsets) * np.kaiser(num_taps, 5.0)
            self._taps = (taps / taps.sum()).astype(np.float32)

    def process(self, audio: np.ndarray, *, final: bool = False) -> np.ndarray:
        chunk = np.asarray(audio, dtype=np.float32)
        if chunk.ndim != 1:
            raise ValueError(f"Streaming audio resampling only supports mono audio, got shape {chunk.shape}")

        if self._ratio == 1:
            return chunk

        if chunk.size:
            self._buffer = np.concatenate((self._buffer, chunk))
            self._total_samples += int(chunk.size)

        half = self._taps.size // 2
        max_center = self._total_samples - 1 if final else self._total_samples - 1 - half
        if self._next_center > max_center:
            return np.empty((0,), dtype=np.float32)

        centers = np.arange(self._next_center, max_center + 1, self._ratio, dtype=np.int64)
        first_center = int(centers[0])
        last_center = int(centers[-1])
        window_start = first_center - half
        window_end = last_center + half
        segment = np.zeros((window_end - window_start + 1,), dtype=np.float32)

        copy_start = max(window_start, self._buffer_start, 0)
        copy_end = min(window_end + 1, self._buffer_start + self._buffer.size)
        if copy_end > copy_start:
            dst_start = copy_start - window_start
            src_start = copy_start - self._buffer_start
            segment[dst_start : dst_start + copy_end - copy_start] = self._buffer[
                src_start : src_start + copy_end - copy_start
            ]

        windows = np.lib.stride_tricks.sliding_window_view(segment, self._taps.size)[:: self._ratio]
        output = windows @ self._taps[::-1]
        self._next_center = last_center + self._ratio

        keep_from = max(0, self._next_center - half)
        drop = min(max(keep_from - self._buffer_start, 0), self._buffer.size)
        if drop:
            self._buffer = self._buffer[drop:]
            self._buffer_start += int(drop)

        return np.asarray(output, dtype=np.float32)


class AudioMixin:
    """Mixin class to add audio-related utilities."""

    def create_audio(self, audio_obj: CreateAudio) -> AudioResponse:
        """Convert audio tensor to bytes in the specified format."""

        audio_tensor = audio_obj.audio_tensor
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
            audio_data = buffer.getvalue()

        if base64_encode:
            import base64

            audio_data = base64.b64encode(audio_data).decode("utf-8")

        return AudioResponse(audio_data=audio_data, media_type=media_type)

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
