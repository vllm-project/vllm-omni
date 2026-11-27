# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import pytest
import torch

from vllm_omni.diffusion.models.helios.pipeline_helios import resolve_history_sizes

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]


def test_keep_first_frame_true_keeps_sizes_and_total():
    history_sizes = [16, 2, 1]
    total = resolve_history_sizes(history_sizes, keep_first_frame=True)
    assert history_sizes == [16, 2, 1]
    assert total == sum(history_sizes) == 19


def test_keep_first_frame_false_grows_short_history_and_total():
    history_sizes = [16, 2, 1]
    total = resolve_history_sizes(history_sizes, keep_first_frame=False)
    assert history_sizes == [16, 2, 2]
    # Regression: the buffer sized by the returned total must be splittable
    # with the resolved sizes (it was one frame short before the fix, which
    # crashed chunk 0 of every keep_first_frame=False request).
    buffer = torch.zeros(1, 16, total, 8, 8)
    windows = buffer[:, :, -total:].split(history_sizes, dim=2)
    assert [w.shape[2] for w in windows] == [16, 2, 2]
