# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""What is allowed to keep the CFM wrapper from capturing a graph.

These run without a GPU: ``_capture`` is mocked, so nothing here touches a
CUDA context. The point is the decision made *before* capture, not the capture.
"""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from vllm_omni.model_executor.models.minicpmo_4_5.cuda_graph_wrapper import (
    WholeEulerCFMGraphWrapper,
    _format_memory_delta,
    _memory_snapshot,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _wrapper(monkeypatch: pytest.MonkeyPatch, *, max_graphs: int = 4) -> WholeEulerCFMGraphWrapper:
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    wrapper = object.__new__(WholeEulerCFMGraphWrapper)
    wrapper.max_graphs = max_graphs
    wrapper.max_graph_batch = 16
    wrapper.micro_batch_size = 4
    wrapper.pad_max_rows = 1
    wrapper.query_bucket_frames = 0
    wrapper.query_widths = ()
    wrapper.offset_bucket_frames = 0
    wrapper.ragged_body = None
    wrapper.att_slots = 0
    wrapper.slot_pool = None
    wrapper.row_offsets = False
    wrapper.npu_mixed_history = False
    wrapper.n_timesteps = 10
    wrapper.enabled = True
    wrapper.device = torch.device("cuda:0")
    wrapper.dtype = torch.float32
    wrapper.att_cache_dtype = torch.float32
    wrapper._att_capacity = 0
    wrapper._cache = {}
    wrapper._slot_graphs = {}
    wrapper._capture_key_masks = {
        (10, 0, 0, 10, 10, torch.device("cuda:0")): torch.ones(1, 1, 10, dtype=torch.bool),
    }
    wrapper.arena = SimpleNamespace(
        n_timesteps=10,
        depth=2,
        cnn_channels=8,
        cnn_width=2,
        heads=2,
        att_width=8,
        att_cache_dtype=torch.float32,
    )
    wrapper._unsupported = set()
    wrapper._stats = {"calls": 0, "hits": 0, "captures": 0, "flushes": 0, "eager": 0, "slot_replays": 0}
    wrapper._capture = Mock(return_value=None)
    return wrapper


def test_capture_is_not_gated_on_free_device_memory(monkeypatch: pytest.MonkeyPatch) -> None:
    """Whether to capture must not be decided from free device memory.

    A capture is served by the caching allocator out of its private graph
    pool, and the allocator falls back on memory it has already reserved. Free
    device memory is therefore neither an upper nor a lower bound on what a
    capture can draw, and a threshold on it decides the wrong question -- one
    whose answer also silently varies with card size and with whatever else
    shares the device. Anything consulting it here would fail this test.
    """
    import vllm_omni.platforms as platforms_module

    def _saturated(device: torch.device | None = None) -> int:
        raise AssertionError("capture must not depend on device-level free memory")

    # Setting the attribute on the module shadows the lazy ``__getattr__``, so
    # this neither builds nor requires a real platform singleton.
    monkeypatch.setattr(
        platforms_module,
        "current_omni_platform",
        Mock(get_free_memory=_saturated),
        raising=False,
    )

    wrapper = _wrapper(monkeypatch)
    real_zeros = torch.zeros
    monkeypatch.setattr(torch, "empty", lambda shape, **kw: real_zeros(shape, dtype=kw.get("dtype", torch.float32)))

    def _cuda_tensor(shape: tuple[int, ...]) -> SimpleNamespace:
        return SimpleNamespace(
            shape=shape,
            dtype=torch.float32,
            device=torch.device("cuda:0"),
            new_empty=lambda out_shape: real_zeros(out_shape),
        )

    wrapper.replay(
        x=_cuda_tensor((2, 16, 10)),
        mu_cfg=_cuda_tensor((4, 16, 10)),
        speakers_cfg=_cuda_tensor((4, 8)),
        cond_cfg=_cuda_tensor((4, 16, 10)),
        cnn_cache=None,
        att_cache=None,
    )

    wrapper._capture.assert_called_once()


def test_memory_reporting_is_inert_off_cuda() -> None:
    """Capture logging must not break the platforms that have no such counters."""
    assert _memory_snapshot(torch.device("cpu")) is None
    assert _format_memory_delta(None, (1, 2)) == ""
    assert _format_memory_delta((1, 2), None) == ""
    assert "allocated" in _format_memory_delta((0, 0), (1 << 20, 2 << 20))
