# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass
from threading import Lock
from typing import ClassVar, Generic, TypeVar

import torch
from vllm.logger import init_logger

from vllm_omni.watermarking.types import AudioTensor

logger = init_logger(__name__)

MediaT = TypeVar("MediaT")
RequestStateT = TypeVar("RequestStateT")
AudioImplStateT = TypeVar("AudioImplStateT")


class WatermarkFailureError(Exception):
    """Raised when a watermark backend fails."""


class Watermarker(ABC, Generic[MediaT, RequestStateT]):
    """Manage shared watermarking resources and per-request state."""

    supported_types: ClassVar[tuple[type[object], ...]] = ()

    def __init__(self) -> None:
        """Initialize request state storage."""
        self._request_states: dict[str, RequestStateT] = {}
        self._lock = Lock()
        self._closed = False

    def watermark(self, request_id: str, data: MediaT) -> MediaT:
        """Watermark the next media chunk for a request."""
        self._validate_type(data)
        # Implementations may swap request state on one model; async callers must offload this entire call.
        with self._lock, torch.inference_mode():
            self._ensure_open()
            if request_id not in self._request_states:
                self._request_states[request_id] = self._new_state(data)
                logger.debug("Created %s state for request %s", type(self).__name__, request_id)
            completed = False
            try:
                result = self._watermark(data, self._request_states[request_id])
                self._validate_output(data, result)
                completed = True
                return result
            finally:
                if not completed:
                    self._close_state(self._request_states.pop(request_id))

    def watermark_output(
        self,
        request_id: str,
        data: torch.Tensor,
        metadata: Mapping[str, object],
    ) -> torch.Tensor:
        """Watermark a generated tensor using its output metadata."""
        # NOTE: We currently have a layer of canonicalization in the watermarker
        # to ensure formats are standardized. This is arguable the wrong place to do
        # this in the long term (i.e., the output format of models should be consistent).
        std_tensor = self._from_output(data, metadata)
        wm_tensor = self.watermark(request_id, std_tensor)
        return self._to_output(wm_tensor)

    def is_watermarked(self, data: MediaT) -> bool:
        """Return whether every batch item contains a watermark."""
        self._validate_type(data)
        with self._lock, torch.inference_mode():
            self._ensure_open()
            return self._is_watermarked(data)

    def discard_request_state(self, request_id: str) -> None:
        """Release request state when present."""
        with self._lock:
            if request_id in self._request_states:
                self._close_state(self._request_states.pop(request_id))
                logger.debug("Discarded %s state for request %s", type(self).__name__, request_id)

    def close(self) -> None:
        """Release all request and shared state."""
        with self._lock:
            if self._closed:
                return
            self._closed = True
            states = tuple(self._request_states.values())
            self._request_states.clear()
            for state in states:
                self._close_state(state)
            self._close()
            logger.debug("Closed %s; discarded %d request states", type(self).__name__, len(states))

    def _ensure_open(self) -> None:
        """Reject operations after closure."""
        if self._closed:
            raise RuntimeError(f"{type(self).__name__} is closed")

    def _validate_type(self, data: object) -> None:
        """Validate that this implementation supports the media type."""
        if not isinstance(data, self.supported_types):
            raise TypeError(f"{type(self).__name__} does not support {type(data).__name__}")

    def _validate_output(self, source: MediaT, watermarked: MediaT) -> None:
        """Validate the watermarked result."""
        self._validate_type(watermarked)

    def _from_output(self, data: torch.Tensor, metadata: Mapping[str, object]) -> MediaT:
        """Convert a generated tensor to this watermarker's media type."""
        raise NotImplementedError

    def _to_output(self, data: MediaT) -> torch.Tensor:
        """Convert watermarked media back to its generated tensor."""
        raise NotImplementedError

    @abstractmethod
    def _new_state(self, data: MediaT) -> RequestStateT:
        """Create state for a new request."""

    @abstractmethod
    def _watermark(self, data: MediaT, state: RequestStateT) -> MediaT:
        """Watermark data using its request state."""

    @abstractmethod
    def _is_watermarked(self, data: MediaT) -> bool:
        """Check data for this implementation's watermark."""

    def _close_state(self, state: RequestStateT) -> None:
        """Release resources held by one request state."""

    def _close(self) -> None:
        """Release resources shared by all requests."""


@dataclass
class _AudioRequestState(Generic[AudioImplStateT]):
    """Hold channel metadata and implementation state for one request."""

    channels: int
    audio_state: AudioImplStateT


class AudioWatermarkerBase(Watermarker[AudioTensor, _AudioRequestState[AudioImplStateT]], Generic[AudioImplStateT]):
    """Provide common audio validation, stereo handling, and request state."""

    # Audio watermarkers operate on float waveforms & media boundaries handle PCM quantization
    supported_types = (AudioTensor,)
    supports_stereo: ClassVar[bool]

    def _from_output(self, data: torch.Tensor, metadata: Mapping[str, object]) -> AudioTensor:
        """Attach the sampling rate to the generated audio."""
        sample_rate = metadata.get("sr")
        if isinstance(sample_rate, torch.Tensor):
            if sample_rate.numel() != 1:
                raise ValueError("audio sample rate must be scalar")
            sample_rate = sample_rate.item()
        if isinstance(sample_rate, bool) or not isinstance(sample_rate, int):
            raise TypeError("audio output requires integer 'sr' metadata")
        return AudioTensor(data, sample_rate)

    def _to_output(self, data: AudioTensor) -> torch.Tensor:
        """Return the watermarked audio samples."""
        return data.samples

    def _new_state(self, data: AudioTensor) -> _AudioRequestState[AudioImplStateT]:
        """Prepare audio and create implementation state for a request."""
        data = self._canonicalize_audio(data)
        self._validate_audio(data)
        if data.samples.shape[1] == 2 and not self.supports_stereo:
            logger.warning("%s does not support stereo; watermarking the mono mid signal", type(self).__name__)
        prepared = data if data.samples.shape[1] == 1 or self.supports_stereo else self._to_mono(data)
        return _AudioRequestState(data.samples.shape[1], self._new_audio_state(prepared))

    def _watermark(self, data: AudioTensor, state: _AudioRequestState[AudioImplStateT]) -> AudioTensor:
        """Prepare, watermark, and restore one audio chunk."""
        source = data
        data = self._canonicalize_audio(data)
        self._validate_audio(data)
        if data.samples.shape[1] != state.channels:
            raise ValueError("audio channel count must remain stable within a stream")
        prepared = data if state.channels == 1 or self.supports_stereo else self._to_mono(data)
        watermarked = self._watermark_audio(prepared, state.audio_state)
        self._validate_output(prepared, watermarked)
        if prepared is not data:
            residual = watermarked.samples - prepared.samples
            watermarked = AudioTensor(self._add_residual_with_headroom(data.samples, residual), data.sample_rate)
        return AudioTensor(watermarked.samples.reshape(source.samples.shape), watermarked.sample_rate)

    def _is_watermarked(self, data: AudioTensor) -> bool:
        """Prepare audio and check it for a watermark."""
        data = self._canonicalize_audio(data)
        self._validate_audio(data)
        prepared = data if data.samples.shape[1] == 1 or self.supports_stereo else self._to_mono(data)
        return self._is_audio_watermarked(prepared)

    def _close_state(self, state: _AudioRequestState[AudioImplStateT]) -> None:
        """Release implementation state for one audio request."""
        self._close_audio_state(state.audio_state)

    @staticmethod
    def _to_mono(data: AudioTensor) -> AudioTensor:
        """Downmix audio channels to mono."""
        return AudioTensor(data.samples.mean(dim=1, keepdim=True), data.sample_rate)

    @staticmethod
    def _canonicalize_audio(data: AudioTensor) -> AudioTensor:
        """Convert mono or channel-first audio to batched channel-first layout."""
        if data.samples.ndim == 1:
            return AudioTensor(data.samples[None, None], data.sample_rate)
        if data.samples.ndim == 2:
            return AudioTensor(data.samples[None], data.sample_rate)
        return data

    @staticmethod
    def _validate_audio(data: AudioTensor) -> None:
        """Validate decoded audio layout and metadata."""
        samples = data.samples
        if samples.ndim != 3:
            raise ValueError(f"audio samples must have shape [batch, channels, samples], got {tuple(samples.shape)}")
        if not samples.is_floating_point():
            raise TypeError("audio samples must use a floating-point dtype")
        if samples.shape[1] not in (1, 2):
            raise ValueError(
                "audio channel dimension must be axis 1 in [batch, channels, samples] layout, "
                f"got {tuple(samples.shape)}"
            )
        if data.sample_rate <= 0:
            raise ValueError("audio sample rate must be positive")

    def _validate_output(self, source: AudioTensor, watermarked: AudioTensor) -> None:
        """Ensure watermarking preserved audio shape and sample rate."""
        super()._validate_output(source, watermarked)
        if watermarked.samples.shape != source.samples.shape:
            raise ValueError("audio watermarker changed the audio shape")
        if watermarked.sample_rate != source.sample_rate:
            raise ValueError("audio watermarker changed the sample rate")

    @staticmethod
    def _add_residual_with_headroom(samples: torch.Tensor, residual: torch.Tensor) -> torch.Tensor:
        """Add a mono residual without clipping either channel."""
        scale = torch.ones_like(residual)
        scale = torch.where(
            residual > 0,
            (1 - samples.amax(dim=1, keepdim=True)) / residual,
            scale,
        )
        scale = torch.where(
            residual < 0,
            (samples.amin(dim=1, keepdim=True) + 1) / -residual,
            scale,
        )
        return samples + residual * scale.clamp_(0, 1)

    @abstractmethod
    def _new_audio_state(self, data: AudioTensor) -> AudioImplStateT:
        """Create implementation state for a new audio request."""

    @abstractmethod
    def _watermark_audio(self, data: AudioTensor, state: AudioImplStateT) -> AudioTensor:
        """Watermark normalized audio using its request state."""

    @abstractmethod
    def _is_audio_watermarked(self, data: AudioTensor) -> bool:
        """Check normalized audio for this implementation's watermark."""

    def _close_audio_state(self, state: AudioImplStateT) -> None:
        """Release implementation resources for one audio request."""
