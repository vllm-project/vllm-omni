# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""A pooling stage's request output must not break the stage-pool metrics.

vLLM's ``OutputProcessor._new_request_output`` unwraps the one-element list
before building ``PoolingRequestOutput``, so a pooling stage's
``request_output.outputs`` is a single ``PoolingOutput`` rather than a list of
completions. The stage-pool metrics used to iterate it as a sequence and
raised ``TypeError: 'PoolingOutput' object is not iterable``.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from vllm_omni.engine.stage_pool import StagePool

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _pool() -> StagePool:
    client = SimpleNamespace(stage_type="llm", final_output_type="audio")
    return StagePool(0, [client])  # type: ignore[arg-type]


def _pooling_request_output(data: object) -> SimpleNamespace:
    return SimpleNamespace(outputs=SimpleNamespace(data=data))


def test_has_non_empty_output_accepts_single_pooling_output() -> None:
    pool = _pool()

    assert pool.has_non_empty_output(_pooling_request_output(SimpleNamespace(shape=(2, 2)))) is True


def test_has_non_empty_output_rejects_empty_pooling_output() -> None:
    pool = _pool()

    assert pool.has_non_empty_output(_pooling_request_output(SimpleNamespace(shape=(0, 2)))) is False
