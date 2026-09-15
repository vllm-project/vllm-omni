# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Rolling-window mel slicing must match full-history slicing bit-for-bit.

The duplex thinker keeps a bounded trailing audio window instead of the full
session history (the mel featurizer is local: hop-aligned STFT columns, no
cross-stream normalization).  These tests replay a stream both ways through
the vendored preprocessor and require identical chunk tensors.
"""

from types import SimpleNamespace

import pytest
import torch

from vllm_omni.model_executor.models.nemotron_voicechat.nemotron_voicechat_thinker import (
    _DUPLEX_MEL_MARGIN_COLS,
    perception_chunk_geometry,
    slice_perception_streaming_mel,
)

FRAME_SAMPLES = 1280  # 80 ms @ 16 kHz
HOP = 160  # 10 ms window stride

pytestmark = pytest.mark.core_model


def _make_preprocessor():
    from vllm_omni.model_executor.models.nemotron_voicechat.nemo_vendored.asr.audio_preprocessing import (
        AudioToMelSpectrogramPreprocessor,
    )

    # Mirrors model.stt.model.perception.preprocessor in the checkpoint with
    # the thinker's streaming overrides (dither=0, pad_to=0).
    return (
        AudioToMelSpectrogramPreprocessor(
            sample_rate=16000,
            window_size=0.025,
            window_stride=0.01,
            n_fft=512,
            features=128,
            dither=0.0,
            pad_to=0,
            normalize="NA",
            log=True,
            frame_splicing=1,
        )
        .to(torch.float32)
        .eval()
    )


def _mel(preprocessor, audio: torch.Tensor) -> torch.Tensor:
    with torch.inference_mode():
        processed, _ = preprocessor(
            input_signal=audio.unsqueeze(0),
            length=torch.tensor([audio.numel()], dtype=torch.long),
        )
    return processed


@pytest.mark.parametrize(
    "streaming_cfg",
    [
        SimpleNamespace(
            chunk_size=[9, 8],
            shift_size=[9, 8],
            pre_encode_cache_size=[0, 9],
            drop_extra_pre_encoded=3,
        ),
        SimpleNamespace(
            chunk_size=[17, 16],
            shift_size=[9, 8],
            pre_encode_cache_size=[0, 17],
            drop_extra_pre_encoded=5,
        ),
    ],
)
def test_windowed_mel_chunks_match_full_history(streaming_cfg):
    torch.manual_seed(0)
    preprocessor = _make_preprocessor()
    num_frames = 80  # 6.4 s: far past the point where trimming kicks in
    stream = torch.randn(num_frames * FRAME_SAMPLES, dtype=torch.float32) * 0.1

    window = None
    window_col0 = 0
    trimmed = False
    for frame_idx in range(num_frames):
        frame = stream[frame_idx * FRAME_SAMPLES : (frame_idx + 1) * FRAME_SAMPLES]
        full_audio = stream[: (frame_idx + 1) * FRAME_SAMPLES]
        window = frame if window is None else torch.cat([window, frame])

        if frame_idx > 0:
            total_samples = window_col0 * HOP + int(window.numel())
            stream_total_cols = total_samples // HOP + 1
            assert stream_total_cols == full_audio.numel() // HOP + 1
            cache_start, _, _ = perception_chunk_geometry(frame_idx, stream_total_cols, streaming_cfg)
            desired_col0 = max(0, cache_start - _DUPLEX_MEL_MARGIN_COLS)
            if desired_col0 > window_col0:
                window = window[(desired_col0 - window_col0) * HOP :]
                window_col0 = desired_col0
                trimmed = True

        expected, expected_drop = slice_perception_streaming_mel(
            _mel(preprocessor, full_audio), frame_idx, streaming_cfg
        )
        actual, actual_drop = slice_perception_streaming_mel(
            _mel(preprocessor, window), frame_idx, streaming_cfg, window_col0=window_col0
        )
        assert actual_drop == expected_drop
        assert actual.shape == expected.shape, f"frame {frame_idx}"
        assert torch.equal(actual, expected), f"frame {frame_idx} mel chunk diverged (window_col0={window_col0})"

    assert trimmed, "test never exercised the trimmed-window path"
    # The rolling window must stay bounded instead of growing with the stream.
    assert window.numel() < 4 * 16000, f"window grew to {window.numel()} samples"


def test_stream_start_requires_untrimmed_window():
    streaming_cfg = SimpleNamespace(
        chunk_size=[9, 8], shift_size=[9, 8], pre_encode_cache_size=[0, 9], drop_extra_pre_encoded=3
    )
    mel = torch.zeros(1, 128, 9)
    with pytest.raises(ValueError, match="untrimmed"):
        slice_perception_streaming_mel(mel, 0, streaming_cfg, window_col0=4)


def test_window_missing_needed_columns_rejected():
    streaming_cfg = SimpleNamespace(
        chunk_size=[9, 8], shift_size=[9, 8], pre_encode_cache_size=[0, 9], drop_extra_pre_encoded=3
    )
    mel = torch.zeros(1, 128, 24)
    with pytest.raises(ValueError, match="needs column"):
        slice_perception_streaming_mel(mel, 40, streaming_cfg, window_col0=340)
