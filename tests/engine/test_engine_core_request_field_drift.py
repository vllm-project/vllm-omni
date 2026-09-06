# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Contract test: OmniEngineCoreRequest.from_request must forward every upstream field."""

from __future__ import annotations

import ast
import inspect
import textwrap

import pytest

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _extract_cls_call_kwargs(method) -> set[str]:
    """Parse the source of *method* and return keyword names from the ``cls(...)`` call."""
    source = textwrap.dedent(inspect.getsource(method))
    tree = ast.parse(source)
    kwargs: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if isinstance(func, ast.Name) and func.id == "cls":
            for kw in node.keywords:
                if kw.arg is not None:
                    kwargs.add(kw.arg)
    return kwargs


def test_from_request_forwards_all_upstream_fields():
    """Catch silent field drift between EngineCoreRequest and OmniEngineCoreRequest.from_request."""
    pytest.importorskip("vllm.v1.engine")

    from vllm.v1.engine import EngineCoreRequest

    from vllm_omni.engine import OmniEngineCoreRequest

    upstream_fields = set(EngineCoreRequest.__struct_fields__)
    forwarded_kwargs = _extract_cls_call_kwargs(OmniEngineCoreRequest.from_request)

    missing = upstream_fields - forwarded_kwargs
    assert not missing, (
        f"Upstream EngineCoreRequest added field(s) {missing} — "
        f"update OmniEngineCoreRequest.from_request to handle them"
    )


def test_from_request_no_stale_fields():
    """Detect fields that from_request copies but upstream has removed."""
    pytest.importorskip("vllm.v1.engine")

    from vllm.v1.engine import EngineCoreRequest

    from vllm_omni.engine import OmniEngineCoreRequest

    upstream_fields = set(EngineCoreRequest.__struct_fields__)
    omni_own_fields = set(OmniEngineCoreRequest.__struct_fields__) - upstream_fields
    forwarded_kwargs = _extract_cls_call_kwargs(OmniEngineCoreRequest.from_request)

    stale = forwarded_kwargs - upstream_fields - omni_own_fields
    assert not stale, (
        f"from_request copies field(s) {stale} that no longer exist on "
        f"EngineCoreRequest or OmniEngineCoreRequest — remove them"
    )
