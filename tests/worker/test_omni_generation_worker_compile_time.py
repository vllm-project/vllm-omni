# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

from vllm_omni.worker.gpu_generation_worker import _make_compilation_times, _supports_generation_device_type

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def test_make_compilation_times_matches_current_vllm_shape():
    result = _make_compilation_times(0.0)

    assert result.language_model == 0.0
    assert result.encoder == 0.0


if __name__ == "__main__":
    test_make_compilation_times_matches_current_vllm_shape()


def test_generation_worker_keeps_musa_support():
    assert _supports_generation_device_type("cuda")
    assert _supports_generation_device_type("musa")
    assert not _supports_generation_device_type("cpu")
