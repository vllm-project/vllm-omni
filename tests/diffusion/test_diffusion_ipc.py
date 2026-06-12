# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Unit tests for the streaming IPC layer in ``vllm_omni.diffusion.ipc``.

Covers both bare-tensor and container (dict / list / tuple) round-trips
through the chunked SHM protocol via a lightweight in-process mock queue.
"""

from __future__ import annotations

import threading
import time

import pytest
import torch

from vllm_omni.diffusion.data import DiffusionOutput
from vllm_omni.diffusion.ipc import (
    _SHM_PLACEHOLDER_KEY,
    _SHM_TENSOR_THRESHOLD,
    _extract_tensors,
    _restore_placeholders,
    pack_diffusion_output_shm,
    unpack_diffusion_output_shm,
)
from vllm_omni.diffusion.worker.utils import RunnerOutput

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]


# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------


def _large_numel(dtype: torch.dtype) -> int:
    """Smallest number of elements that exceeds the SHM threshold."""
    return (_SHM_TENSOR_THRESHOLD // torch.empty((), dtype=dtype).element_size()) + 1


class _MockMessageQueue:
    """Simple in-process mock for ``MessageQueue`` used by the IPC layer.

    Supports concurrent enqueue / dequeue across threads so that
    ``pack_diffusion_output_shm`` and ``unpack_diffusion_output_shm`` can
    run in separate threads while sharing the same mock queues.
    """

    def __init__(self):
        self._items: list = []
        self._lock = threading.Lock()
        self._not_empty = threading.Condition(self._lock)

    def enqueue(self, msg: object) -> None:
        with self._not_empty:
            self._items.append(msg)
            self._not_empty.notify()

    def dequeue(self, timeout: float | None = None) -> object:
        deadline = None if timeout is None else time.monotonic() + timeout
        with self._not_empty:
            while not self._items:
                if timeout is not None:
                    remaining = (deadline - time.monotonic()) if deadline else -1
                    if remaining <= 0:
                        raise TimeoutError("Mock queue empty")
                    self._not_empty.wait(remaining)
                else:
                    self._not_empty.wait()
            return self._items.pop(0)


def _round_trip(output: object) -> object:
    """Pack *output* through a mock queue, then unpack and return the result.

    The producer and consumer run in separate threads because
    ``pack_diffusion_output_shm`` blocks on ACKs that are only produced
    by ``unpack_diffusion_output_shm``.
    """
    result_mq = _MockMessageQueue()
    ack_mq = _MockMessageQueue()
    unpack_result: list = []

    def _consume() -> None:
        unpack_result.append(unpack_diffusion_output_shm(result_mq, ack_mq))

    t = threading.Thread(target=_consume, daemon=True)
    t.start()
    pack_diffusion_output_shm(output, result_mq, ack_mq)
    t.join(timeout=5)
    if t.is_alive():
        raise TimeoutError("unpack_diffusion_output_shm did not finish")
    return unpack_result[0]


# ---------------------------------------------------------------------------
# _extract_tensors / _restore_placeholders unit tests
# ---------------------------------------------------------------------------


def test_extract_tensors_strips_large_tensors_from_flat_dict():
    large = torch.arange(_large_numel(torch.float32), dtype=torch.float32)
    small = torch.arange(8, dtype=torch.float32)
    payload = {"image": large, "video": small, "meta": "text"}

    fields: list = []
    stripped = _extract_tensors(payload, "output", fields)

    # Large tensor → placeholder, small tensor stays inline, scalars pass through.
    assert isinstance(stripped["image"], dict) and stripped["image"][_SHM_PLACEHOLDER_KEY] is True
    assert stripped["video"] is small
    assert stripped["meta"] == "text"

    # tensor_fields: (field_name, tensor, use_shm)
    assert len(fields) == 2
    assert fields[0] == ("output", large, True)
    assert fields[1] == ("output", small, False)


def test_extract_tensors_strips_from_nested_containers():
    large = torch.arange(_large_numel(torch.float32), dtype=torch.float32)
    payload = {"media": [large, "keep"]}

    fields: list = []
    stripped = _extract_tensors(payload, "output", fields)

    assert isinstance(stripped["media"], list)
    assert isinstance(stripped["media"][0], dict)
    assert stripped["media"][0][_SHM_PLACEHOLDER_KEY] is True
    assert stripped["media"][1] == "keep"
    assert fields == [("output", large, True)]


def test_restore_placeholders_reconstructs_flat_dict_ordering():
    a = torch.arange(4)
    b = torch.arange(4) + 10
    stripped = {"a": {_SHM_PLACEHOLDER_KEY: True}, "b": {_SHM_PLACEHOLDER_KEY: True}}

    restored = _restore_placeholders(stripped, [a, b], [0])
    torch.testing.assert_close(restored["a"], a)
    torch.testing.assert_close(restored["b"], b)


def test_restore_placeholders_reconstructs_nested_containers():
    t = torch.arange(4)
    stripped = {"nested": [{"deep": {_SHM_PLACEHOLDER_KEY: True}}]}

    restored = _restore_placeholders(stripped, [t], [0])
    torch.testing.assert_close(restored["nested"][0]["deep"], t)


def test_restore_placeholders_preserves_inline_values():
    stripped = {"a": 1, "b": "hello", "c": None}
    restored = _restore_placeholders(stripped, [], [0])
    assert restored == {"a": 1, "b": "hello", "c": None}


# ---------------------------------------------------------------------------
# Full round-trip tests (pack → mock queue → unpack)
# ---------------------------------------------------------------------------


def test_bare_tensor_round_trip():
    tensor = torch.arange(_large_numel(torch.float32), dtype=torch.float32)
    output = DiffusionOutput(output=tensor)
    result = _round_trip(output)
    torch.testing.assert_close(result.output, tensor)


def test_bare_tensor_below_threshold_kept_inline():
    tensor = torch.arange(8, dtype=torch.float32)
    output = DiffusionOutput(output=tensor)
    result = _round_trip(output)
    torch.testing.assert_close(result.output, tensor)


def test_dict_container_round_trip():
    image = torch.arange(_large_numel(torch.float32), dtype=torch.float32)
    video = torch.arange(_large_numel(torch.float32), dtype=torch.float32) * 2
    output = DiffusionOutput(output={"image": image, "video": video, "metadata": {"prompt": "a cat"}})
    result = _round_trip(output)
    assert isinstance(result.output, dict)
    torch.testing.assert_close(result.output["image"], image)
    torch.testing.assert_close(result.output["video"], video)
    assert result.output["metadata"] == {"prompt": "a cat"}


def test_tuple_container_round_trip():
    a = torch.arange(_large_numel(torch.float32), dtype=torch.float32)
    b = torch.arange(_large_numel(torch.float32), dtype=torch.float32) * 3
    output = DiffusionOutput(output=(a, b))
    result = _round_trip(output)
    assert isinstance(result.output, tuple)
    torch.testing.assert_close(result.output[0], a)
    torch.testing.assert_close(result.output[1], b)


def test_list_container_round_trip():
    frames = [
        torch.arange(_large_numel(torch.float32), dtype=torch.float32),
        torch.arange(_large_numel(torch.float32), dtype=torch.float32) + 1,
    ]
    output = DiffusionOutput(output=frames)
    result = _round_trip(output)
    assert isinstance(result.output, list)
    torch.testing.assert_close(result.output[0], frames[0])
    torch.testing.assert_close(result.output[1], frames[1])


def test_mixed_container_some_large_some_small():
    large = torch.arange(_large_numel(torch.float32), dtype=torch.float32)
    small = torch.arange(8, dtype=torch.float32)
    output = DiffusionOutput(output={"lg": large, "sm": small})
    result = _round_trip(output)
    torch.testing.assert_close(result.output["lg"], large)
    torch.testing.assert_close(result.output["sm"], small)


def test_container_with_no_large_tensors_all_inline():
    a = torch.arange(8, dtype=torch.float32)
    b = torch.arange(16, dtype=torch.float32)
    payload = {"a": a, "b": b, "text": "inline"}
    output = DiffusionOutput(output=payload)
    result = _round_trip(output)
    assert result.output == payload
    torch.testing.assert_close(result.output["a"], a)
    torch.testing.assert_close(result.output["b"], b)


def test_bfloat16_round_trip():
    tensor = torch.arange(_large_numel(torch.bfloat16), dtype=torch.float32).to(torch.bfloat16).reshape(1, -1)
    output = DiffusionOutput(output=tensor)
    result = _round_trip(output)
    assert result.output.dtype == torch.bfloat16
    torch.testing.assert_close(result.output, tensor)


def test_non_contiguous_tensor_round_trip():
    tensor = torch.arange(_large_numel(torch.float32) * 2, dtype=torch.float32).reshape(-1, 2)[:, 0]
    assert not tensor.is_contiguous()
    output = DiffusionOutput(output=tensor)
    result = _round_trip(output)
    torch.testing.assert_close(result.output, tensor)


def test_trajectory_fields_round_trip():
    latents = torch.arange(_large_numel(torch.float32), dtype=torch.float32)
    timesteps = torch.arange(10, dtype=torch.float32)  # small, stays inline
    log_probs = torch.arange(_large_numel(torch.float32), dtype=torch.float32) * 2
    output = DiffusionOutput(
        output=torch.arange(8, dtype=torch.float32),  # small inline
        trajectory_latents=latents,
        trajectory_timesteps=timesteps,
        trajectory_log_probs=log_probs,
    )
    result = _round_trip(output)
    torch.testing.assert_close(result.output, output.output)
    torch.testing.assert_close(result.trajectory_latents, latents)
    torch.testing.assert_close(result.trajectory_timesteps, timesteps)
    torch.testing.assert_close(result.trajectory_log_probs, log_probs)


def test_non_diffusion_output_passthrough():
    # Non-DiffusionOutput objects should pass through unchanged.
    payload = {"status": "ok", "value": 42}
    result = _round_trip(payload)
    assert result == payload


def test_extract_tensors_keeps_tensor_at_threshold_inline():
    # Tensor exactly at the SHM threshold should stay inline (≤, not <).
    numel = _SHM_TENSOR_THRESHOLD // torch.empty((), dtype=torch.float32).element_size()
    tensor = torch.arange(numel, dtype=torch.float32)

    fields: list = []
    stripped = _extract_tensors(tensor, "output", fields)

    # Must NOT be a placeholder — tensor is at threshold, stays inline.
    assert stripped is tensor
    assert fields == [("output", tensor, False)]


def test_extract_tensors_does_not_mutate_original_payload():
    large = torch.arange(_large_numel(torch.float32), dtype=torch.float32)
    small = torch.arange(8, dtype=torch.float32)
    original = {
        "media": {"lg": large, "sm": small},
        "items": [large, "keep"],
        "meta": {"key": "val"},
    }

    fields: list = []
    _extract_tensors(original, "output", fields)

    # Original payload must be untouched.
    assert original["media"]["lg"] is large
    assert original["media"]["sm"] is small
    assert original["items"][0] is large
    assert original["items"][1] == "keep"
    assert original["meta"] == {"key": "val"}


def test_runner_output_wrapper_round_trip():
    """DiffusionOutput wrapped in RunnerOutput (the typical worker return shape)."""
    tensor = torch.arange(_large_numel(torch.float32), dtype=torch.float32)
    output = RunnerOutput(request_id="req-1", finished=True, result=DiffusionOutput(output=tensor))
    result = _round_trip(output)
    assert result.request_id == "req-1"
    assert result.finished is True
    torch.testing.assert_close(result.result.output, tensor)


def test_runner_output_small_tensor_inline():
    """Small tensor through RunnerOutput — stays inline, no SHM overhead."""
    tensor = torch.arange(100, dtype=torch.float32)
    output = RunnerOutput(request_id="req-1", finished=True, result=DiffusionOutput(output=tensor))
    result = _round_trip(output)
    assert torch.equal(result.result.output, tensor)
