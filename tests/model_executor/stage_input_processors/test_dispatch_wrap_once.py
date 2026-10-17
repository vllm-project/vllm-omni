# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Orchestrator dispatch wrap-once semantics (table-driven).

``invoke_orchestrator_processor`` runs per forward, so
``wrap_orchestrator_processor`` must not re-wrap — nor re-emit the legacy
``DeprecationWarning`` — on every forward: the result is memoized per callable.

Pure-logic, CPU-only (no model loading, no vllm runtime).
"""

from __future__ import annotations

from typing import Any

import pytest

from vllm_omni.model_executor.stage_input_processors._dispatch import (
    OrchestratorInputContext,
    invoke_orchestrator_processor,
    wrap_orchestrator_processor,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _legacy_c2(source_outputs, prompt=None, requires_multimodal_data=False, streaming_context=None):
    """Legacy C2 positional shell (like qwen3_omni.thinker2talker_token_only)."""
    return [source_outputs, prompt, requires_multimodal_data, streaming_context]


def _legacy_c0(source_outputs, prompt=None, requires_multimodal_data=False):
    """Legacy C0 3-arg shell."""
    return [source_outputs, prompt, requires_multimodal_data]


def _c1(source_outputs: list[Any], ctx: OrchestratorInputContext):
    """Already speaks the C1 ``(source_outputs, ctx)`` contract."""
    return [source_outputs, ctx]


def _legacy_c3_keyword_only(source_outputs, prompt=None, requires_multimodal_data=False, *, sampling_params=None):
    """C3 shell whose ``sampling_params`` is keyword-only."""
    return [source_outputs, prompt, requires_multimodal_data, sampling_params]


def _legacy_c3_required_keyword_only(source_outputs, prompt=None, requires_multimodal_data=False, *, sampling_params):
    """C3 shell with a REQUIRED keyword-only ``sampling_params``."""
    return [source_outputs, prompt, requires_multimodal_data, sampling_params]


def _legacy_c2_other_name(source_outputs, prompt=None, requires_multimodal_data=False, streaming_ctx=None):
    """Four-positional shell with a non-standard streaming param name."""
    return [source_outputs, prompt, requires_multimodal_data, streaming_ctx]


def _legacy_c2_keyword_only(source_outputs, prompt=None, requires_multimodal_data=False, *, streaming_context=None):
    """C2 shell whose ``streaming_context`` is keyword-only."""
    return [source_outputs, prompt, requires_multimodal_data, streaming_context]


def _count_deprecation_warnings(trigger) -> int:
    """Run *trigger* under ``simplefilter("always")``; count DeprecationWarnings."""
    import warnings

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        trigger()
    return sum(1 for w in caught if issubclass(w.category, DeprecationWarning))


_INVOKE_CASES = [
    pytest.param(
        _legacy_c2,
        {"prompt": "p", "requires_multimodal_data": True, "streaming_context": "s"},
        [["out"], "p", True, "s"],
        id="c2-streaming-context",
    ),
    pytest.param(
        _legacy_c3_keyword_only,
        {"prompt": "p", "requires_multimodal_data": True, "sampling_params": "sp"},
        [["out"], "p", True, "sp"],
        id="c3-keyword-only-sampling-params",
    ),
    pytest.param(
        _legacy_c3_required_keyword_only,
        {"prompt": "p", "requires_multimodal_data": True, "sampling_params": "sp"},
        [["out"], "p", True, "sp"],
        id="c3-required-keyword-only-sampling-params",
    ),
    pytest.param(
        _legacy_c2_other_name,
        {"prompt": "p", "requires_multimodal_data": True, "streaming_context": "s"},
        [["out"], "p", True, "s"],
        id="c2-four-positional-other-name",
    ),
    pytest.param(
        _legacy_c2_keyword_only,
        {"prompt": "p", "requires_multimodal_data": True, "streaming_context": "s"},
        [["out"], "p", True, "s"],
        id="c2-keyword-only-streaming-context",
    ),
]


def test_wrap_legacy_shape_warns_once_then_memoized():
    assert _count_deprecation_warnings(lambda: wrap_orchestrator_processor(_legacy_c2)) == 1
    # Subsequent wraps reuse the memoized adapter: no further warning.
    assert _count_deprecation_warnings(lambda: wrap_orchestrator_processor(_legacy_c2)) == 0
    assert _count_deprecation_warnings(lambda: wrap_orchestrator_processor(_legacy_c2)) == 0


@pytest.mark.parametrize("fn,ctx_kwargs,expected", _INVOKE_CASES)
def test_invoke_adapts_legacy_shape_and_forwards_context(fn, ctx_kwargs, expected):
    ctx = OrchestratorInputContext(**ctx_kwargs)
    assert invoke_orchestrator_processor(fn, ["out"], ctx) == expected


def test_invoke_does_not_rewarn_on_every_forward():
    ctx = OrchestratorInputContext(prompt="p", requires_multimodal_data=True, streaming_context="s")

    def invoke_many():
        for _ in range(5):
            invoke_orchestrator_processor(_legacy_c0, ["out"], ctx)

    # Exactly one DeprecationWarning total across 5 forwards (wrap happens once).
    assert _count_deprecation_warnings(invoke_many) == 1


def test_c1_processor_passes_through_without_warning():
    wrapped = wrap_orchestrator_processor(_c1)
    assert wrapped is _c1  # already speaks ctx: returned unchanged
    ctx = OrchestratorInputContext(prompt="p")
    assert invoke_orchestrator_processor(_c1, ["out"], ctx) == [["out"], ctx]
    assert _count_deprecation_warnings(lambda: wrap_orchestrator_processor(_c1)) == 0
