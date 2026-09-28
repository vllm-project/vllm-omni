# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import pytest
import torch

from vllm_omni.worker_v2.shared_pcm_copy import copy_shared_pcm_views

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


@pytest.mark.parametrize("trim", [0, 1, 4])
@pytest.mark.parametrize("reverse", [False, True])
def test_stereo_views_order_trim_and_owned_output(trim, reverse):
    batch = torch.arange(3 * 2 * 15, dtype=torch.float32).reshape(3, 2, 15)
    values = [batch[i, :, trim:] for i in range(3)]
    if reverse:
        values.reverse()
    expected = [v.clone() for v in values]
    copied = []

    def copy(value):
        copied.append(value.numel())
        return value.clone()

    outputs = copy_shared_pcm_views(values, copy)
    assert len(copied) == 1
    batch.fill_(-1)
    for output, target in zip(outputs, expected):
        torch.testing.assert_close(output, target, rtol=0, atol=0)
    outputs[0].zero_()
    torch.testing.assert_close(outputs[1], expected[1], rtol=0, atol=0)


def test_mixed_allocations_dtypes_empty_and_overlap():
    batch = torch.arange(48).reshape(4, 12)
    other = torch.arange(12, dtype=torch.float32)
    values = [batch[2], other, batch[0], torch.empty(0), batch[3]]
    outputs = copy_shared_pcm_views(values, lambda value: value.clone())
    for output, value in zip(outputs, values):
        torch.testing.assert_close(output, value, rtol=0, atol=0)
    overlap = copy_shared_pcm_views([batch[0], batch[0]], lambda value: value.clone())
    overlap[0].zero_()
    torch.testing.assert_close(overlap[1], batch[0])


def test_sparse_views_do_not_copy_unbounded_gaps():
    batch = torch.arange(10000)
    sizes = []

    def copy(value):
        sizes.append(value.numel())
        return value.clone()

    copy_shared_pcm_views([batch[:10], batch[-10:]], copy)
    assert sizes == [10, 10]
