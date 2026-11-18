# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Non-CUDA runtimes must not receive Triton's ``launch_pdl`` keyword."""

import pytest

import vllm_omni.model_executor.models.moss_tts.triton_launch as triton_launch

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def test_non_cuda_omits_unrecognised_launch_pdl_keyword(monkeypatch):
    monkeypatch.setattr(triton_launch.current_platform, "is_cuda", lambda: False)

    assert triton_launch.cuda_pdl_launch_kwargs(False) == {}
    assert triton_launch.cuda_pdl_launch_kwargs(True) == {}


def test_cuda_keeps_launch_pdl_keyword(monkeypatch):
    monkeypatch.setattr(triton_launch.current_platform, "is_cuda", lambda: True)

    assert triton_launch.cuda_pdl_launch_kwargs(False) == {"launch_pdl": False}
    assert triton_launch.cuda_pdl_launch_kwargs(True) == {"launch_pdl": True}
