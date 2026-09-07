# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

from vllm_omni.diffusion.models.sana_video import transformer_sana_video


@pytest.fixture(autouse=True)
def sp_world_1(monkeypatch):
    """L1 runs single-process without a sequence parallel group; keep the
    transformer on its SP1 path unless a test mocks a larger SP world."""
    monkeypatch.setattr(transformer_sana_video, "get_sequence_parallel_world_size", lambda: 1)
