# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Isolate scheduler dataclass inheritance from optional backend imports."""

import ast
from dataclasses import dataclass, field, make_dataclass
from pathlib import Path

import pytest

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


@pytest.mark.parametrize("upstream_default", [False, True])
def test_cached_request_data_accepts_upstream_default_fields(upstream_default):
    base_fields = [("req_ids", list[str])]
    if upstream_default:
        # vLLM-Ascend's bounded-replay patch adds this defaulted base field.
        base_fields.append(("replay_start", dict[str, int], field(default_factory=dict)))
    base = make_dataclass("CachedRequestData", base_fields)
    path = Path(__file__).parents[3] / "vllm_omni/core/sched/output.py"
    source = ast.parse(path.read_text())
    definition = next(
        node for node in source.body if isinstance(node, ast.ClassDef) and node.name == "OmniCachedRequestData"
    )
    namespace = {"dataclass": dataclass, "CachedRequestData": base}
    exec(compile(ast.Module(body=[definition], type_ignores=[]), str(path), "exec"), namespace)
    cached_type = namespace["OmniCachedRequestData"]
    kwargs = dict(req_ids=["req"], prompt_token_ids={"req": [1]}, additional_information={"req": None})
    cached = cached_type(**kwargs)
    assert cached.req_ids == ["req"]
    assert cached.prompt_token_ids == {"req": [1]}
    assert cached.additional_information == {"req": None}
    with pytest.raises(TypeError, match="prompt_token_ids"):
        cached_type(req_ids=["req"], additional_information={})
    if upstream_default:
        assert cached.replay_start == {}
        cached.replay_start["req"] = 1
        assert cached_type(**kwargs).replay_start == {}
