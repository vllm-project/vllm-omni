# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for realtime streaming helpers (PR #2581 /v1/realtime path)."""

from __future__ import annotations

import base64

import numpy as np
import pytest
import torch
from vllm.sampling_params import RequestOutputKind, SamplingParams

from vllm_omni.entrypoints.async_omni import AsyncOmni
from vllm_omni.entrypoints.openai.realtime_connection import RealtimeConnection

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


@pytest.fixture
def realtime_conn() -> RealtimeConnection:
    return RealtimeConnection.__new__(RealtimeConnection)


class TestRealtimeConnectionTensorAndPcm:
    def test_tensor_to_numpy_none(self) -> None:
        assert RealtimeConnection._tensor_to_numpy(None) is None

    def test_tensor_to_numpy_1d_numpy(self) -> None:
        arr = np.array([1.0, 2.0], dtype=np.float64)
        out = RealtimeConnection._tensor_to_numpy(arr)
        assert out is not None
        assert out.dtype == np.float32
        assert out.shape == (2,)

    def test_tensor_to_numpy_2d_numpy_flattened(self) -> None:
        arr = np.array([[0.5], [-0.5]], dtype=np.float32)
        out = RealtimeConnection._tensor_to_numpy(arr)
        assert out is not None
        assert out.shape == (2,)

    def test_tensor_to_numpy_torch(self) -> None:
        t = torch.tensor([[0.25, -0.25]], dtype=torch.float32)
        out = RealtimeConnection._tensor_to_numpy(t)
        assert out is not None
        assert out.shape == (2,)
        np.testing.assert_allclose(out, [0.25, -0.25], rtol=1e-5)

    def test_pcm16_b64_roundtrip(self) -> None:
        audio = np.array([0.0, 1.0, -1.0], dtype=np.float32)
        b64 = RealtimeConnection._pcm16_b64(audio)
        raw = base64.b64decode(b64)
        assert len(raw) == 6
        pcm = np.frombuffer(raw, dtype=np.int16)
        assert pcm[0] == 0
        assert pcm[1] == 32767
        assert pcm[2] == -32767


class TestAsyncOmniStreamingParamsValidation:
    def test_accepts_streaming_friendly_params(self) -> None:
        p = SamplingParams(
            n=1,
            stop=[],
            output_kind=RequestOutputKind.DELTA,
        )
        AsyncOmni._validate_streaming_input_sampling_params(p)

    def test_rejects_non_sampling_params(self) -> None:
        with pytest.raises(ValueError, match="Input streaming"):
            AsyncOmni._validate_streaming_input_sampling_params(object())  # type: ignore[arg-type]

    def test_rejects_n_greater_than_one(self) -> None:
        p = SamplingParams(n=2, stop=[], output_kind=RequestOutputKind.DELTA)
        with pytest.raises(ValueError, match="Input streaming"):
            AsyncOmni._validate_streaming_input_sampling_params(p)

    def test_rejects_final_only(self) -> None:
        p = SamplingParams(n=1, stop=[], output_kind=RequestOutputKind.FINAL_ONLY)
        with pytest.raises(ValueError, match="Input streaming"):
            AsyncOmni._validate_streaming_input_sampling_params(p)

    def test_rejects_stop_strings(self) -> None:
        p = SamplingParams(n=1, stop=["\n"], output_kind=RequestOutputKind.DELTA)
        with pytest.raises(ValueError, match="Input streaming"):
            AsyncOmni._validate_streaming_input_sampling_params(p)
