# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Tests for worker-v2 asynchronous host copies."""

import pytest
import torch

import vllm_omni.worker_v2.omni_ar_model_runner as omni_ar_model_runner

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


class _FakeStream:
    pass


@pytest.mark.parametrize("pin_memory", [False, True])
def test_cpu_copy_records_source_lifetime_only_while_async(mocker, pin_memory):
    source = mocker.Mock(spec=torch.Tensor)
    source.detach.return_value = source
    source.device = torch.device("cuda")
    destination = mocker.Mock(spec=torch.Tensor)
    mocker.patch.object(torch, "empty_like", return_value=destination)
    stream = _FakeStream()

    result = omni_ar_model_runner._async_copy_tensor(source, copy_stream=stream, pin_memory=pin_memory)

    assert result is destination
    destination.copy_.assert_called_once_with(source, non_blocking=pin_memory)
    if pin_memory:
        source.record_stream.assert_called_once_with(stream)
    else:
        source.record_stream.assert_not_called()
