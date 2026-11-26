# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""CUDA-only coverage for the pinned staging in ``HyNomasPipeline._to_host``.

Split out of ``test_hy_nomas.py``: that module carries a module-level
``pytest.mark.cpu`` and ``hardware_test(..., num_cards=1)`` adds no ``skipif``
guard for a missing GPU, so the pinned branch could only ever run in a CUDA job.
Keeping it in its own, non-``cpu`` module means it is collected only by a job
that selects ``cuda`` for this path.
"""

import numpy as np
import pytest
import torch

from tests.helpers.mark import hardware_test
from vllm_omni.diffusion.models.hy_nomas.pipeline_hy_nomas import HyNomasPipeline

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion]


@hardware_test(res={"cuda": ["L4", "H100"]}, num_cards=1)
def test_to_host_stages_cuda_chunk_through_pinned_memory(mocker):
    """A CUDA chunk must be copied through a pinned buffer and match a pageable copy."""
    frames = torch.randint(0, 256, (2, 4, 8, 8), dtype=torch.uint8, device="cuda")
    empty = mocker.spy(torch, "empty")

    staged = HyNomasPipeline._to_host(frames)

    assert empty.call_args.kwargs["pin_memory"] is True, "staging buffer must be pinned for an event-tracked copy"
    assert staged.dtype == np.uint8
    assert staged.flags["C_CONTIGUOUS"]
    np.testing.assert_array_equal(staged, frames.cpu().numpy())


@hardware_test(res={"cuda": ["L4", "H100"]}, num_cards=1)
def test_to_host_does_not_alias_the_source_tensor():
    """The returned array must own its buffer, so a later chunk cannot overwrite it."""
    frames = torch.zeros((1, 2, 2, 2), dtype=torch.uint8, device="cuda")

    staged = HyNomasPipeline._to_host(frames)
    staged[:] = 255

    assert int(frames.max()) == 0
