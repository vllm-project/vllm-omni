# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Unit tests for the shared runner helpers in vllm_omni.worker.mixins."""

from types import SimpleNamespace

import pytest
import torch

from vllm_omni.worker.mixins import maybe_unpad_input_ids

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


@pytest.mark.parametrize(
    "flag, padded_len, unpadded, expected",
    [
        (True, 4, 3, 3),
        (False, 4, 3, 4),
        (True, 3, 3, 3),
    ],
)
def test_maybe_unpad_input_ids(flag, padded_len, unpadded, expected):
    """#6712: opted-in models get input_ids at the real token count; everything
    else keeps the graph-bucket padded buffer."""
    model = SimpleNamespace(requires_exact_input_shape=flag)
    ids = torch.zeros(padded_len, dtype=torch.int32)
    assert maybe_unpad_input_ids(model, ids, unpadded).numel() == expected


def test_maybe_unpad_input_ids_without_capability_attribute():
    ids = torch.zeros(4, dtype=torch.int32)
    assert maybe_unpad_input_ids(SimpleNamespace(), ids, 3).numel() == 4


def test_maybe_unpad_input_ids_handles_none():
    model = SimpleNamespace(requires_exact_input_shape=True)
    assert maybe_unpad_input_ids(model, None, 3) is None
