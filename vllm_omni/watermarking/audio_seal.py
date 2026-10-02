# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

import torch
from vllm.logger import init_logger
from vllm.utils.import_utils import PlaceholderModule

from vllm_omni.watermarking.base import AudioWatermarkerBase
from vllm_omni.watermarking.types import AudioTensor

logger = init_logger(__name__)

if TYPE_CHECKING:
    from audioseal.models import AudioSealDetector, AudioSealWM

try:
    import audioseal.loader as loader
except ImportError:
    loader = PlaceholderModule("audioseal")  # type: ignore[assignment]


@dataclass
class _AudioSealState:
    """Hold state retained between chunks of one AudioSeal stream."""

    batch_size: int
    sample_rate: int
    message: torch.Tensor
    streaming_state: dict[str, object] | None = None


class AudioSealWatermarker(AudioWatermarkerBase[_AudioSealState]):
    """Watermark audio with Meta AudioSeal."""

    supports_stereo = False

    def __init__(self) -> None:
        """Load the streaming AudioSeal generator."""
        super().__init__()
        if isinstance(loader, PlaceholderModule):
            raise ImportError("Audio watermarking requires `pip install 'vllm-omni[watermarking]'`")
        logger.info("Loading AudioSeal watermark generator on CPU")
        self._model = self._load_generator()
        # Detector isn't needed at inference time, but is convenient for testing etc, so we defer loading
        self._detector: AudioSealDetector | None = None

    def _watermark_audio(self, data: AudioTensor, state: _AudioSealState) -> AudioTensor:
        """Watermark one chunk using retained AudioSeal state."""
        self._validate_audioseal_input(data)
        if data.samples.shape[0] != state.batch_size or data.sample_rate != state.sample_rate:
            raise ValueError("audio batch size and sample rate must remain stable within a stream")
        if data.samples.shape[-1] == 0:
            return data

        original_device = data.samples.device
        original_dtype = data.samples.dtype
        source = data.samples.to(device="cpu", dtype=torch.float32)

        # NOTE: AudioSeal >=0.2 accepts native-rate audio; we avoid resampling here to
        # avoid potential boundary artifacts across streamed chunks.
        # Furthermore, streaming handles whole / chunked output with comparable measured throughput
        with self._model.streaming(state.batch_size):
            if state.streaming_state is not None:
                self._model.encoder.set_streaming_state(state.streaming_state)  # type: ignore[attr-defined]
            watermarked = self._model(
                source,
                message=state.message,
            )
            state.streaming_state = self._model.encoder.get_streaming_state()  # type: ignore[attr-defined]

        watermarked = self._add_residual_with_headroom(source, watermarked - source)

        return AudioTensor(
            samples=watermarked.to(device=original_device, dtype=original_dtype),
            sample_rate=data.sample_rate,
        )

    def _new_audio_state(self, data: AudioTensor) -> _AudioSealState:
        """Create a deterministic random message for an AudioSeal stream."""
        self._validate_audioseal_input(data)
        batch_size = data.samples.shape[0]
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(0)
            message = self._model.random_message(batch_size)
        return _AudioSealState(batch_size, data.sample_rate, message)

    @staticmethod
    def _load_generator() -> AudioSealWM:
        """Load the AudioSeal generator without changing global RNG state."""
        with torch.random.fork_rng(devices=[]):
            generator = cast(
                "AudioSealWM",
                loader.AudioSeal.load_generator(
                    "audioseal_wm_streaming",
                    device=torch.device("cpu"),
                ),
            )
        generator.eval()
        return generator

    def _validate_audioseal_input(self, data: AudioTensor) -> None:
        """Validate AudioSeal channel requirements."""
        if data.samples.shape[1] != 1:
            raise ValueError("AudioSeal implementation requires mono audio")

    def _close_audio_state(self, state: _AudioSealState) -> None:
        """Discard retained AudioSeal streaming state."""
        state.streaming_state = None

    ### Related to detection / verifying existing watermarks
    def _is_audio_watermarked(self, data: AudioTensor) -> bool:
        """Detect an AudioSeal watermark in audio."""
        self._validate_audioseal_input(data)
        if data.samples.shape[-1] == 0:
            return False
        detector = self._detector
        if detector is None:
            logger.info("Loading AudioSeal watermark detector on CPU")
            detector = self._load_detector()
            self._detector = detector
        samples = data.samples.to(device="cpu", dtype=torch.float32)
        probability, _ = detector.detect_watermark(samples)
        return torch.as_tensor(probability).min().item() > 0.5

    @staticmethod
    def _load_detector() -> AudioSealDetector:
        """Load the AudioSeal detector without changing global RNG state."""
        with torch.random.fork_rng(devices=[]):
            detector = cast(
                "AudioSealDetector",
                loader.AudioSeal.load_detector(
                    "audioseal_detector_streaming",
                    device=torch.device("cpu"),
                ),
            )
        detector.eval()
        return detector
