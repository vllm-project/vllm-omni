# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import math
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
    """Stateful polyphase resampler for streaming mono float audio.

    Retains filter state so output is invariant to input chunk boundaries.
    """

    _half_filter_width = 10
    _kaiser_beta = 5.0
    _max_polyphase_factor = 2_048

    def __init__(self, source_rate: int, target_rate: int):
        if source_rate <= 0 or target_rate <= 0:
            raise ValueError("Audio sample rates must be positive")
        self.source_rate = source_rate
        self.target_rate = target_rate
        rate_gcd = math.gcd(source_rate, target_rate)
        self._up = target_rate // rate_gcd
        self._down = source_rate // rate_gcd
        if max(self._up, self._down) > self._max_polyphase_factor:
            raise ValueError("source and target sample rates have an unsupported resampling ratio")
        self._half_len, self._phase_kernels = self._design_polyphase_filter()
        self._history_samples = self._phase_kernels.shape[1] - 1
        self.reset()

    def _design_polyphase_filter(self) -> tuple[int, np.ndarray]:
        if self._up == self._down:
            return 0, np.ones((1, 1), dtype=np.float32)

        max_rate = max(self._up, self._down)
        half_len = self._half_filter_width * max_rate
        offsets = np.arange(-half_len, half_len + 1, dtype=np.float64)
        cutoff = 1.0 / max_rate
        taps = cutoff * np.sinc(cutoff * offsets)
        taps *= np.kaiser(taps.size, self._kaiser_beta)
        taps /= np.sum(taps)
        taps *= self._up
        taps = np.ascontiguousarray(taps, dtype=np.float32)

        phases = [taps[phase :: self._up] for phase in range(self._up)]
        phase_width = max(phase.size for phase in phases)
        kernels = np.zeros((self._up, phase_width), dtype=np.float32)
        for phase_index, phase in enumerate(phases):
            kernels[phase_index, -phase.size :] = phase[::-1]
        return half_len, kernels

    @property
    def scratch_bytes(self) -> int:
        pending_samples = self._ceil_div(self._input_samples * self._up, self._down) - self._output_samples
        return max(0, pending_samples) * np.dtype(np.float32).itemsize

    @staticmethod
    def _ceil_div(numerator: int, denominator: int) -> int:
        return -(-numerator // denominator)

    def _render_outputs(
        self,
        combined: np.ndarray,
        *,
        first_output: int,
        output_count: int,
        input_start: int,
    ) -> np.ndarray:
        if output_count <= 0:
            return np.empty(0, dtype=np.float32)

        output_indexes = np.arange(first_output, first_output + output_count, dtype=np.int64)
        filter_indexes = output_indexes * self._down + self._half_len
        source_indexes = filter_indexes // self._up
        phases = filter_indexes % self._up
        windows = np.lib.stride_tricks.sliding_window_view(
            combined,
            self._phase_kernels.shape[1],
        )
        selected_windows = windows[source_indexes - input_start]
        return np.sum(
            selected_windows * self._phase_kernels[phases],
            axis=1,
            dtype=np.float32,
        )

    def _push(self, chunk: np.ndarray) -> np.ndarray:
        if chunk.size == 0:
            return np.empty(0, dtype=np.float32)
        if self._flushed:
            raise RuntimeError("cannot process audio after the resampler has been finalized; call reset first")

        combined = np.concatenate((self._history, chunk))
        next_input_samples = self._input_samples + chunk.size
        stable_output_samples = max(
            0,
            self._ceil_div(next_input_samples * self._up - self._half_len, self._down),
        )
        output = self._render_outputs(
            combined,
            first_output=self._output_samples,
            output_count=stable_output_samples - self._output_samples,
            input_start=self._input_samples,
        )
        if self._history_samples:
            self._history = np.ascontiguousarray(combined[-self._history_samples :], dtype=np.float32)
        self._input_samples = next_input_samples
        self._output_samples = stable_output_samples
        return output

    def _flush(self) -> np.ndarray:
        if self._flushed:
            return np.empty(0, dtype=np.float32)

        total_output_samples = self._ceil_div(self._input_samples * self._up, self._down)
        output_count = total_output_samples - self._output_samples
        if output_count:
            last_output = total_output_samples - 1
            last_source_index = (last_output * self._down + self._half_len) // self._up
            right_padding = max(0, last_source_index - self._input_samples + 1)
            combined = np.concatenate((self._history, np.zeros(right_padding, dtype=np.float32)))
            output = self._render_outputs(
                combined,
                first_output=self._output_samples,
                output_count=output_count,
                input_start=self._input_samples,
            )
        else:
            output = np.empty(0, dtype=np.float32)
        self._output_samples = total_output_samples
        self._flushed = True
        return output

    def process(self, audio: np.ndarray, *, final: bool = False) -> np.ndarray:
        chunk = np.asarray(audio, dtype=np.float32)
        if chunk.ndim != 1:
            raise ValueError(f"Streaming audio resampling only supports mono audio, got shape {chunk.shape}")
        chunk = np.ascontiguousarray(chunk, dtype=np.float32)
        output = self._push(chunk)
        if not final:
            return output
        tail = self._flush()
        if not output.size:
            return tail
        if not tail.size:
            return output
        return np.concatenate((output, tail))

    def reset(self) -> None:
        self._history = np.zeros(self._history_samples, dtype=np.float32)
        self._input_samples = 0
        self._output_samples = 0
        self._flushed = False


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
