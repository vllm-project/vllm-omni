# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import inspect

import vllm_omni.worker.gpu_generation_worker as worker
from vllm_omni.worker.gpu_generation_worker import _make_compilation_times


def test_make_compilation_times_matches_current_vllm_shape():
    result = _make_compilation_times(0.0)

    if isinstance(result, float):
        assert result == 0.0
    else:
        assert result.language_model == 0.0
        assert result.encoder == 0.0


if __name__ == "__main__":
    test_make_compilation_times_matches_current_vllm_shape()


def test_make_compilation_times_filters_to_current_fields():
    result = _make_compilation_times(1.5, speculative_model=2.0)

    if isinstance(result, float):
        assert result == 1.5
    else:
        assert result.language_model == 1.5
        assert result.encoder == 0.0


def test_compilation_times_import_is_lazy():
    source_before_helper = inspect.getsource(worker).split("def _make_compilation_times", 1)[0]
    assert "CompilationTimes" not in source_before_helper
