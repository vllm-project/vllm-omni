# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""CPU tests for the streaming uint8 preparation in the MiniMax-H3 video VAE.

``_stream_prepare_video_tensor`` must be numerically identical to the
checkpoint's whole-video numpy path (``convert_numpy_to_tensor`` →
``transform_tensor`` → ``transpose``) while padding on the host, so the
device-side copies never materialize.
"""

import numpy as np
import pytest
import torch

from vllm_omni.diffusion.models.minimax_h3.vae import MiniMaxH3VideoVAE

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
CLIP_LENGTH = 4


class _FakeNormalize:
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std


class _FakeProcessor:
    # Read by the stream prep via getattr(..., False); annotation only so
    # tests can attach it per case.
    isolated_last_frame: bool

    def __init__(self):
        self.transform = _FakeNormalize(IMAGENET_MEAN, IMAGENET_STD)


class _FakeCheckpointModel:
    def __init__(self, **overrides):
        self.clip_length = CLIP_LENGTH
        self.isolated_first_frame = False
        self.frame_pre_padding = 0
        self.processor = _FakeProcessor()
        for key, value in overrides.items():
            setattr(self, key, value)


def _vae(model=None):
    vae = object.__new__(MiniMaxH3VideoVAE)
    vae.model = model if model is not None else _FakeCheckpointModel()
    return vae


def _legacy_reference(frames: np.ndarray) -> torch.Tensor:
    """The checkpoint's numpy path, op for op, on the CPU.

    convert_numpy_to_tensor: astype(fp32) → permute → ÷255
    transform_tensor: Normalize = (x - mean) / std
    encode_videos: transpose(0, 1)
    """
    tensor = torch.from_numpy(frames.astype(np.float32)).permute(0, 3, 1, 2)
    tensor = tensor / 255.0
    mean = torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(1, 3, 1, 1)
    tensor = (tensor - mean) / std
    return tensor.transpose(0, 1).contiguous()


def _video(num_frames, height=6, width=8, seed=0):
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, size=(num_frames, height, width, 3), dtype=np.uint8)


def _pad_frames(frames: np.ndarray, pad: int) -> np.ndarray:
    if not pad:
        return frames
    return np.concatenate([frames, np.repeat(frames[-1:], pad, axis=0)])


def test_stream_prep_matches_legacy_path_bitwise():
    frames = _video(10)
    out = _vae()._stream_prepare_video_tensor(frames, torch.device("cpu"))
    assert out is not None
    assert out.shape == (3, 12, 6, 8)  # pad = (0 - 10) % 4 == 2
    assert out.dtype == torch.float32
    assert torch.equal(out, _legacy_reference(_pad_frames(frames, 2)))


def test_stream_prep_isolated_first_frame_without_last_frame_skips_pad():
    # Asymmetric checkpoint (isolated first frame, no isolated last frame):
    # trim-stable lengths are k*clip + 0, which never satisfy the
    # encode_temporal offset of 1, so the remote re-pads through its own
    # whole-video cat regardless. Padding here would only upload frames the
    # remote trim discards, so the legacy frame count is kept.
    model = _FakeCheckpointModel(isolated_first_frame=True)
    frames = _video(10)
    out = _vae(model)._stream_prepare_video_tensor(frames, torch.device("cpu"))
    assert out is not None
    assert out.shape[1] == 10
    assert torch.equal(out, _legacy_reference(frames))


def test_stream_prep_symmetric_isolated_frames_pad_to_trim_stable():
    # isolated first + last frames: offset == tail % clip == 1, so padding
    # to the next trim-stable length makes the remote trim a no-op and keeps
    # encode_temporal from re-padding: T=10, clip=4, tail=1 → 3*4+1 == 13.
    model = _FakeCheckpointModel(isolated_first_frame=True)
    model.processor.isolated_last_frame = True
    frames = _video(10)
    out = _vae(model)._stream_prepare_video_tensor(frames, torch.device("cpu"))
    assert out is not None
    assert out.shape[1] == 13
    assert out.shape[1] % 4 == 1
    assert torch.equal(out, _legacy_reference(_pad_frames(frames, 3)))


def test_stream_prep_uses_processor_align_when_available():
    calls = []

    class _AlignedProcessor(_FakeProcessor):
        def align_video_length(self, video_length, mode="pad", granularity="chunk"):
            calls.append((video_length, mode, granularity))
            return 5

    model = _FakeCheckpointModel()
    model.processor = _AlignedProcessor()
    frames = _video(10)
    out = _vae(model)._stream_prepare_video_tensor(frames, torch.device("cpu"))
    assert out is not None
    assert calls == [(10, "pad", "chunk")]
    assert out.shape[1] == 15
    assert torch.equal(out, _legacy_reference(_pad_frames(frames, 5)))


def test_stream_prep_pads_with_last_frame():
    frames = _video(2)  # pad = (0 - 2) % 4 == 2
    out = _vae()._stream_prepare_video_tensor(frames, torch.device("cpu"))
    padded = np.concatenate([frames, np.repeat(frames[-1:], 2, axis=0)])
    assert torch.equal(out, _legacy_reference(padded))


def test_stream_prep_aligned_input_needs_no_pad():
    frames = _video(8)  # 8 % 4 == 0
    out = _vae()._stream_prepare_video_tensor(frames, torch.device("cpu"))
    assert out.shape[1] == 8
    assert torch.equal(out, _legacy_reference(frames))


@pytest.mark.parametrize(
    "frames",
    [
        _video(5).astype(np.float32),  # not uint8
        _video(5)[:, :, :, :1],  # single channel
        np.zeros((5, 6, 8, 4), dtype=np.uint8),  # four channels
        np.zeros((5, 6, 8), dtype=np.uint8),  # missing channel axis
        np.zeros((0, 6, 8, 3), dtype=np.uint8),  # empty
    ],
)
def test_stream_prep_rejects_unsupported_inputs(frames):
    assert _vae()._stream_prepare_video_tensor(frames, torch.device("cpu")) is None


@pytest.mark.parametrize(
    "model",
    [
        _FakeCheckpointModel(clip_length=None),
        _FakeCheckpointModel(clip_length=0),
        type("NoProcessor", (), {"clip_length": CLIP_LENGTH})(),
        type(
            "NoTransform",
            (),
            {"clip_length": CLIP_LENGTH, "processor": type("P", (), {})()},
        )(),
    ],
)
def test_stream_prep_rejects_missing_checkpoint_contract(model):
    frames = _video(5)
    assert _vae(model)._stream_prepare_video_tensor(frames, torch.device("cpu")) is None


def test_stream_prep_rejects_bad_norm_arity():
    model = _FakeCheckpointModel()
    model.processor.transform = _FakeNormalize((0.5, 0.5), IMAGENET_STD)
    assert _vae(model)._stream_prepare_video_tensor(_video(5), torch.device("cpu")) is None


class _RecordingModel(_FakeCheckpointModel):
    """Feeds encode_video a well-shaped latent and records its input."""

    def __init__(self):
        super().__init__()
        self.received = None
        self.latent = torch.randn(1, 4, 3, 4, 4)  # (B, C, T', H', W'), even H/W

    def encode_videos(self, videos, use_fp16_latent=False, **kwargs):
        self.received = videos
        assert use_fp16_latent is True
        return [self.latent]


def _vae_for_encode_video():
    vae = object.__new__(MiniMaxH3VideoVAE)
    vae.model = _RecordingModel()
    vae.device_module = torch
    vae.config_dict = {
        "latent_channels": 4,
        "latents_mean": [0.0] * 4,
        "latents_std": [1.0] * 4,
    }
    # A plain tensor (not nn.Parameter): nn.Module.__setattr__ refuses
    # Parameters on instances built via object.__new__.
    vae._dummy_parameter = torch.zeros(1, dtype=torch.float32)
    vae.parameters = lambda: iter([vae._dummy_parameter])
    return vae


def test_encode_video_streams_uint8_numpy(monkeypatch):
    monkeypatch.delenv("VLLM_OMNI_VAE_ENCODE_LEGACY_PREP", raising=False)
    vae = _vae_for_encode_video()
    rows, shape = vae.encode_video(_video(10))
    assert isinstance(vae.model.received, list)
    assert isinstance(vae.model.received[0], torch.Tensor)
    assert vae.model.received[0].shape == (3, 12, 6, 8)
    assert rows.shape[0] == shape[0] * shape[1] * shape[2] // 4  # patchify rows


def test_encode_video_legacy_env_escapes_to_checkpoint_path(monkeypatch):
    monkeypatch.setenv("VLLM_OMNI_VAE_ENCODE_LEGACY_PREP", "1")
    vae = _vae_for_encode_video()
    frames = _video(10)
    vae.encode_video(frames)
    assert vae.model.received is frames  # untouched numpy, checkpoint converts


def test_encode_video_falls_back_when_preparation_unsupported(monkeypatch):
    monkeypatch.delenv("VLLM_OMNI_VAE_ENCODE_LEGACY_PREP", raising=False)
    vae = _vae_for_encode_video()
    vae.model.clip_length = None  # contract not discoverable
    frames = _video(10)
    vae.encode_video(frames)
    assert vae.model.received is frames
