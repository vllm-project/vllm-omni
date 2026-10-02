# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""HiFT vocoder shapes outside the capture buckets: zero-width mels.

Mock-driven: no HiFT runs, so these hold on CPU.
"""

from __future__ import annotations

from unittest.mock import Mock

import pytest
import torch
import torch.nn as nn

from vllm_omni.model_executor.models.minicpmo_4_5.batched_token2wav import BatchedToken2Wav
from vllm_omni.model_executor.models.minicpmo_4_5.cuda_graph_wrapper import (
    HiFTGraphWrapper,
    empty_hift_outputs,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _no_device(*_args, **_kwargs):
    raise AssertionError("a zero-width mel must not reach the device")


def _bare_backend(hift: object, wrapper: object | None) -> BatchedToken2Wav:
    backend = object.__new__(BatchedToken2Wav)
    nn.Module.__init__(backend)
    backend.hift = hift
    backend.hift_graph_wrapper = wrapper
    return backend


def _mock_wrapper(monkeypatch: pytest.MonkeyPatch) -> HiFTGraphWrapper:
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    wrapper = object.__new__(HiFTGraphWrapper)
    wrapper.capture_batch_sizes = [1]
    wrapper._legit_shapes = {(50, 0)}
    wrapper.graph = {}
    wrapper.lazy_graph_count = 0
    wrapper.max_lazy_graphs = 8
    wrapper.max_serial_batch = 4
    wrapper.decode_fn = Mock(side_effect=_no_device)
    wrapper._capture = Mock(side_effect=_no_device)
    return wrapper


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_empty_hift_outputs_match_hift_output_layout(dtype: torch.dtype) -> None:
    speech, source = empty_hift_outputs(torch.zeros(3, 80, 0, dtype=dtype))

    assert speech.shape == (3, 0)
    assert source.shape == (3, 1, 0)
    assert speech.dtype == dtype and source.dtype == dtype


@pytest.mark.parametrize("cache_len", [0, 3840])
def test_zero_width_mel_never_runs_hift_or_captures(monkeypatch: pytest.MonkeyPatch, cache_len: int) -> None:
    wrapper = _mock_wrapper(monkeypatch)
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", _no_device)

    speech, source = wrapper.replay(torch.zeros(2, 80, 0), torch.zeros(2, 1, cache_len))

    assert speech.shape == (2, 0)
    assert source.shape == (2, 1, 0)
    wrapper.decode_fn.assert_not_called()
    wrapper._capture.assert_not_called()
    assert wrapper.lazy_graph_count == 0


@pytest.mark.parametrize("with_wrapper", [False, True])
def test_backend_zero_width_mel_skips_the_vocoder(with_wrapper: bool) -> None:
    hift = Mock()
    hift.inference = Mock(side_effect=_no_device)
    wrapper = Mock()
    wrapper.replay = Mock(side_effect=_no_device)
    backend = _bare_backend(hift, wrapper if with_wrapper else None)

    speech, source = backend._hift_inference(torch.zeros(1, 80, 0), torch.zeros(1, 1, 0))

    assert speech.shape == (1, 0)
    assert source.shape == (1, 1, 0)
    hift.inference.assert_not_called()
    wrapper.replay.assert_not_called()


def test_backend_nonempty_mel_still_reaches_the_vocoder() -> None:
    expected = (torch.zeros(1, 480 * 4), torch.zeros(1, 1, 480 * 4))
    hift = Mock()
    hift.inference = Mock(return_value=expected)
    backend = _bare_backend(hift, None)
    mel = torch.zeros(1, 80, 4)
    source = torch.zeros(1, 1, 0)

    assert backend._hift_inference(mel, source) is expected
    hift.inference.assert_called_once_with(mel, source)
