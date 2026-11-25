# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from dataclasses import dataclass

import pytest
import torch

from vllm_omni.watermarking import Watermarker

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


@dataclass
class _RequestState:
    closed: bool = False


class NoopWatermarker(Watermarker[bool, _RequestState]):
    """Use boolean data to indicate whether watermarking should succeed."""

    supported_types = (bool,)

    def __init__(self) -> None:
        super().__init__()
        self.created: list[_RequestState] = []

    def _new_state(self, data: bool) -> _RequestState:
        state = _RequestState()
        self.created.append(state)
        return state

    def _watermark(self, data: bool, state: _RequestState) -> bool:
        assert torch.is_inference_mode_enabled()
        if not data:
            raise ValueError("cannot watermark invalid data")
        return data

    def _is_watermarked(self, data: bool) -> bool:
        return data

    def _close_state(self, state: _RequestState) -> None:
        state.closed = True


def test_watermarker_keeps_request_state_isolated() -> None:
    """Ensure each request uses its own persistent state."""
    watermarker = NoopWatermarker()
    watermarker.watermark("request-1", True)
    assert len(watermarker.created) == 1

    watermarker.watermark("request-1", True)
    watermarker.watermark("request-2", True)

    assert len(watermarker.created) == 2
    watermarker.close()


def test_watermarker_closes_finished_requests() -> None:
    """Ensure finished and active request states are released."""
    watermarker = NoopWatermarker()
    watermarker.watermark("request-1", True)
    watermarker.watermark("request-2", True)

    watermarker.discard_request_state("request-1")
    watermarker.watermark("request-1", True)
    watermarker.close()

    assert len(watermarker.created) == 3
    assert all(state.closed for state in watermarker.created)
    with pytest.raises(RuntimeError, match="closed"):
        watermarker.watermark("request-3", True)


def test_watermarker_discards_failed_request_state() -> None:
    """Ensure failing to watermark does not persist request state."""
    watermarker = NoopWatermarker()

    with pytest.raises(ValueError, match="invalid"):
        watermarker.watermark("request", False)
    watermarker.watermark("request", True)

    assert len(watermarker.created) == 2
    assert watermarker.created[0].closed
    watermarker.close()
    assert all(state.closed for state in watermarker.created)
