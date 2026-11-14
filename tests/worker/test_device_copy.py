# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
import numpy as np
import pytest
import torch

from tests.helpers.device import DEVICE_MODULE, DEVICE_TYPE
from tests.helpers.mark import hardware_test
from vllm_omni.utils.device_copy import to_device_nonblocking

pytestmark = pytest.mark.core_model


@pytest.mark.cpu
def test_tensor_copy_cpu_preserves_view_and_dtype():
    source = torch.arange(12, dtype=torch.float64).reshape(3, 4).T
    result = to_device_nonblocking(source, "cpu")
    assert result is source
    assert result.dtype == torch.float64


@hardware_test(res={"cuda": "L4", "rocm": "MI325", "xpu": "B60"}, num_cards=1)
def test_tensor_copy_cuda_handles_pageable_and_pinned_sources():
    if not DEVICE_MODULE.is_available():
        pytest.skip(f"no {DEVICE_TYPE} device")
    source = torch.arange(12, dtype=torch.float32).reshape(3, 4).T
    for host in (source, source.pin_memory()):
        result = to_device_nonblocking(host, DEVICE_TYPE)
        torch.testing.assert_close(result.cpu(), host)
        assert to_device_nonblocking(result, result.device) is result


@pytest.mark.cpu
def test_device_stager_cpu_falls_back_to_a_copy():
    from vllm_omni.utils.device_copy import DeviceStager

    stager = DeviceStager()
    result = stager([[1, 2], [3, 4]], "cpu")
    assert result.tolist() == [[1, 2], [3, 4]] and result.dtype == torch.int64


@hardware_test(res={"cuda": "L4", "rocm": "MI325", "xpu": "B60"}, num_cards=1)
def test_device_stager_views_survive_until_their_readers_finish():
    if not DEVICE_MODULE.is_available():
        pytest.skip(f"no {DEVICE_TYPE} device")
    from vllm_omni.utils.device_copy import DeviceStager

    slots = 4
    stager = DeviceStager(slots=slots, group=2, capacity=4)
    src = torch.arange(100, device=DEVICE_TYPE)
    side = DEVICE_MODULE.Stream()
    outs = []
    for step in range(4 * slots):
        values = [step, step + 1, step + 2] if step % 2 else list(range(step, step + 9))  # also grows past 4
        rows = stager(values, DEVICE_TYPE)
        assert rows.device.type == DEVICE_TYPE
        # a slow reader queued behind a long kernel must still see this step's values
        DEVICE_MODULE._sleep(2_000_000)
        outs.append(src.index_select(0, rows))
        if step % 3 == 0:
            # a reader on another stream, kept alive through retire()
            side.wait_stream(DEVICE_MODULE.current_stream())
            with DEVICE_MODULE.stream(side):
                DEVICE_MODULE._sleep(2_000_000)
                outs.append(src.index_select(0, rows) + 0)
                done = DEVICE_MODULE.Event()
                done.record(side)
            stager.retire(done)
    torch.accelerator.synchronize()
    expected = []
    for step in range(4 * slots):
        values = [step, step + 1, step + 2] if step % 2 else list(range(step, step + 9))
        expected.append(values)
        if step % 3 == 0:
            expected.append(values)
    assert [o.tolist() for o in outs] == expected
    shaped = stager(np.arange(6).reshape(2, 3), DEVICE_TYPE)
    assert shaped.shape == (2, 3) and shaped.cpu().tolist() == [[0, 1, 2], [3, 4, 5]]


@hardware_test(res={"cuda": "L4", "rocm": "MI325", "xpu": "B60"}, num_cards=1)
def test_device_stager_empty_input_does_not_consume_a_ring_slot():
    from vllm_omni.utils.device_copy import DeviceStager

    stager = DeviceStager()
    for shape in ((0,), (2, 0, 3)):
        empty = stager(np.empty(shape, dtype=np.int64), DEVICE_TYPE)
        assert empty.shape == shape and empty.dtype == torch.int64 and empty.device.type == DEVICE_TYPE
        assert stager._prev == -1
    assert stager([2, 7], DEVICE_TYPE).cpu().tolist() == [2, 7]
