# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from pathlib import Path

import numpy as np
import pytest
import soundfile
import torch
from vllm.utils.import_utils import PlaceholderModule

from vllm_omni.outputs import OmniRequestOutput
from vllm_omni.outputs.output_modality import OutputModalityNames
from vllm_omni.watermarking import AudioSealWatermarker, AudioTensor, AudioWatermarkerBase, audio_seal
from vllm_omni.watermarking.utils import watermark_outputs, watermark_payload

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]
TEST_SAMPLE_RATE = 16_000


class _RecordingAudioWatermarker(AudioWatermarkerBase[object]):
    """Record audio layouts received by an implementation."""

    supports_stereo = True

    def __init__(self) -> None:
        super().__init__()
        self.to_wm_shapes: list[torch.Size] = []
        self.verify_wm_shapes: list[torch.Size] = []
        self.watermarked: set[int] = set()

    def _new_audio_state(self, data: AudioTensor) -> object:
        return object()

    def _watermark_audio(self, data: AudioTensor, state: object) -> AudioTensor:
        self.to_wm_shapes.append(data.samples.shape)
        watermarked = data.samples.clone()
        self.watermarked.add(self._tensor_hash(watermarked))
        return AudioTensor(watermarked, data.sample_rate)

    def _is_audio_watermarked(self, data: AudioTensor) -> bool:
        self.verify_wm_shapes.append(data.samples.shape)
        return self._tensor_hash(data.samples) in self.watermarked

    @staticmethod
    def _tensor_hash(tensor: torch.Tensor) -> int:
        """Hash tensor storage so views retain identity."""
        return hash(tensor.untyped_storage())


@pytest.mark.parametrize(
    ("input_shape", "implementation_shape"),
    [
        ((100,), (1, 1, 100)),
        ((1, 100), (1, 1, 100)),
        ((2, 100), (1, 2, 100)),
        ((1, 2, 100), (1, 2, 100)),
    ],
)
def test_audio_base_round_trips_common_layouts(
    input_shape: tuple[int, ...],
    implementation_shape: tuple[int, ...],
) -> None:
    """Ensure watermarking round-trips common layouts through canonical audio."""
    watermarker = _RecordingAudioWatermarker()
    samples = torch.randn(input_shape)
    audio = AudioTensor(samples, TEST_SAMPLE_RATE)

    wm_audio = watermarker.watermark_output("request", audio.samples, {"sr": audio.sample_rate})
    assert wm_audio.shape == input_shape
    assert not watermarker.is_watermarked(audio)
    assert watermarker.is_watermarked(AudioTensor(wm_audio, audio.sample_rate))

    assert watermarker.to_wm_shapes == [implementation_shape]
    assert watermarker.verify_wm_shapes == [implementation_shape, implementation_shape]
    assert audio.samples.shape == input_shape
    watermarker.close()


@pytest.mark.parametrize("bad_sampling_rate", [None, 24_000.0])
def test_audio_output_requires_integer_sample_rate(bad_sampling_rate: object) -> None:
    """Ensure audio output rejects missing or non-integer sample rates."""
    watermarker = _RecordingAudioWatermarker()

    with pytest.raises(TypeError, match="integer 'sr'"):
        watermarker.watermark_output("request", torch.zeros(100), {"sr": bad_sampling_rate})


def test_watermark_payload_preserves_numpy_audio_output() -> None:
    watermarker = _RecordingAudioWatermarker()
    samples = np.zeros(100, dtype=np.float32)
    payload: dict[str, object] = {"audio": samples, "audio_sample_rate": TEST_SAMPLE_RATE}

    try:
        watermark_payload("request", OutputModalityNames.AUDIO, watermarker, payload)
    finally:
        watermarker.close()

    output = payload["audio"]
    assert isinstance(output, np.ndarray)
    assert output.shape == samples.shape


def test_watermark_outputs_handles_diffusion_audio() -> None:
    watermarker = _RecordingAudioWatermarker()
    samples = np.zeros(100, dtype=np.float32)
    output = OmniRequestOutput.from_diffusion(
        request_id="request",
        images=[],
        multimodal_output={"audio": samples, "audio_sample_rate": TEST_SAMPLE_RATE},
        final_output_type="audio",
    )

    watermark_outputs([output], {"audio": watermarker})

    assert isinstance(output.multimodal_output["audio"], np.ndarray)
    assert watermarker.to_wm_shapes == [torch.Size((1, 1, 100))]


def test_watermark_outputs_reports_failed_requests() -> None:
    """Ensure failed watermarking on requests are returned by ID."""
    watermarker = _RecordingAudioWatermarker()
    # Create two requests, where one is valid, but the other isn't since it has no sampling rate
    outputs = [
        OmniRequestOutput.from_diffusion(
            request_id=request_id,
            images=[],
            multimodal_output={"audio": np.zeros(100, dtype=np.float32), "audio_sample_rate": sample_rate},
            final_output_type="audio",
        )
        for request_id, sample_rate in (("bad", None), ("good", TEST_SAMPLE_RATE))
    ]

    assert watermark_outputs(outputs, {"audio": watermarker}) == {"bad"}
    assert watermarker.to_wm_shapes == [torch.Size((1, 1, 100))]


def test_missing_audioseal_names_install_extra(monkeypatch: pytest.MonkeyPatch) -> None:
    """Ensure missing AudioSeal reports the required install extra."""
    monkeypatch.setattr(audio_seal, "loader", PlaceholderModule("audioseal"))

    with pytest.raises(ImportError, match=r"vllm-omni\[watermarking\]"):
        AudioSealWatermarker()


@pytest.mark.local_model
@pytest.mark.slow
@pytest.mark.tts
def test_audioseal_rejects_unsupported_channel_layout() -> None:
    """Ensure AudioSeal rejects unsupported channel layouts."""
    pytest.importorskip("audioseal")
    watermarker = AudioSealWatermarker()

    with pytest.raises(ValueError, match="channel dimension must be axis 1"):
        watermarker.watermark(
            "request",
            AudioTensor(torch.zeros((1, 100, 2)), TEST_SAMPLE_RATE),
        )

    watermarker.close()


@pytest.mark.local_model
@pytest.mark.slow
@pytest.mark.tts
@pytest.mark.parametrize(
    ("channels", "chunk_samples", "noise_std"),
    [
        pytest.param(1, None, 0.05, id="mono-whole"),
        pytest.param(1, 8_000, 0.05, id="mono-streaming"),
        pytest.param(1, 8_000, 0.95, id="mono-streaming-loud"),
        pytest.param(2, 8_000, 0.05, id="stereo-streaming"),
        pytest.param(2, 8_000, 0.95, id="stereo-streaming-loud"),
    ],
)
def test_audioseal_survives_pcm16_wav_encoding(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
    channels: int,
    chunk_samples: int | None,
    noise_std: float,
) -> None:
    """Ensure mono and stereo watermarks survive PCM16 quantization."""
    pytest.importorskip("audioseal")
    sample_rate = TEST_SAMPLE_RATE
    generator = torch.Generator().manual_seed(0)
    source = (torch.randn((1, channels, sample_rate * 2), generator=generator) * noise_std).clamp_(-1, 1)
    if noise_std == 0.95:
        assert (source.abs() == 1).any()
    watermarker = AudioSealWatermarker()
    rng_state = torch.random.get_rng_state()
    chunk_samples = chunk_samples or source.shape[-1]

    chunks = [
        watermarker.watermark("request", AudioTensor(chunk, sample_rate)).samples
        for chunk in source.split(chunk_samples, dim=-1)
    ]
    assert torch.equal(torch.random.get_rng_state(), rng_state)
    assert ("does not support stereo" in caplog.text) == (channels == 2)

    path = tmp_path / f"watermarked-{sample_rate}hz-{channels}ch.wav"
    watermarked = torch.cat(chunks, dim=-1).squeeze(0)
    assert watermarked.shape == source.squeeze(0).shape
    assert watermarked.abs().amax() <= 1
    if channels == 2:
        torch.testing.assert_close(
            watermarked[0] - watermarked[1],
            source[0, 0] - source[0, 1],
        )
    soundfile.write(path, watermarked.T.cpu().numpy(), sample_rate, subtype="PCM_16")
    assert soundfile.info(path).subtype == "PCM_16"
    encoded, encoded_rate = soundfile.read(path, dtype="float32", always_2d=True)

    assert encoded_rate == sample_rate
    reloaded = torch.from_numpy(encoded.T.copy()).unsqueeze(0)
    assert watermarker.is_watermarked(AudioTensor(reloaded, encoded_rate))
    watermarker.discard_request_state("request")
    watermarker.close()


@pytest.mark.local_model
@pytest.mark.slow
@pytest.mark.tts
@pytest.mark.parametrize(
    ("sample_rate", "chunk_samples"),
    [
        pytest.param(8_000, 8_000, id="8khz-streaming"),
        pytest.param(22_050, 11_025, id="22.05khz-streaming"),
        pytest.param(24_000, None, id="24khz-whole"),
        pytest.param(48_000, 24_000, id="48khz-streaming"),
    ],
)
def test_audioseal_watermarks_native_sample_rates(
    sample_rate: int,
    chunk_samples: int | None,
) -> None:
    """Ensure native-rate whole and streamed audio remains detectable."""
    pytest.importorskip("audioseal")
    source = (
        torch.randn(
            (1, 1, sample_rate * 2),
            generator=torch.Generator().manual_seed(0),
        )
        * 0.05
    )
    watermarker = AudioSealWatermarker()

    try:
        chunk_samples = chunk_samples or source.shape[-1]
        chunks = [
            watermarker.watermark("request", AudioTensor(chunk, sample_rate))
            for chunk in source.split(chunk_samples, dim=-1)
        ]
        watermarked = AudioTensor(
            torch.cat([chunk.samples for chunk in chunks], dim=-1),
            sample_rate,
        )

        assert watermarked.samples.shape == source.shape
        assert all(chunk.sample_rate == sample_rate for chunk in chunks)
        assert watermarker.is_watermarked(watermarked)
    finally:
        watermarker.close()


@pytest.mark.local_model
@pytest.mark.slow
@pytest.mark.tts
def test_audioseal_is_deterministic_across_interleaved_requests() -> None:
    """Ensure interleaved requests remain isolated and deterministic."""
    pytest.importorskip("audioseal")
    sample_rate = TEST_SAMPLE_RATE
    sources = {
        request_id: torch.randn((1, 1, sample_rate), generator=torch.Generator().manual_seed(seed)) * 0.05
        for request_id, seed in (("request-1", 1), ("request-2", 2))
    }

    rng_state = torch.random.get_rng_state()
    watermarker = AudioSealWatermarker()
    chunks = {request_id: source.split(sample_rate // 2, dim=-1) for request_id, source in sources.items()}
    expected = {}
    for request_id, request_chunks in chunks.items():
        expected[request_id] = torch.cat(
            [
                watermarker.watermark(f"expected-{request_id}", AudioTensor(chunk, sample_rate)).samples
                for chunk in request_chunks
            ],
            dim=-1,
        )
        watermarker.discard_request_state(f"expected-{request_id}")

    interleaved: dict[str, list[torch.Tensor]] = {"request-1": [], "request-2": []}
    for chunk_index in range(len(next(iter(chunks.values())))):
        for request_id in interleaved:
            data = AudioTensor(chunks[request_id][chunk_index], sample_rate)
            interleaved[request_id].append(watermarker.watermark(request_id, data).samples)

    assert all(torch.equal(torch.cat(interleaved[request_id], dim=-1), expected[request_id]) for request_id in sources)
    assert torch.equal(torch.random.get_rng_state(), rng_state)
    watermarker.close()
