# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Import the real scheduler output module against upstream dataclass variants."""

import importlib.util
import inspect
import sys
from dataclasses import MISSING, field, fields, make_dataclass
from pathlib import Path
from types import ModuleType

import pytest

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


@pytest.mark.parametrize(
    "parent_has_default,replay_start",
    [(False, None), (True, None), (True, [3])],
    ids=["upstream", "ascend-default", "ascend-explicit"],
)
def test_cached_request_import_and_keyword_construction(monkeypatch, parent_has_default, replay_start):
    # Model the upstream fields and the default-valued field added by Ascend.
    # Only dependencies are substituted: every Omni class is imported from the
    # actual production file, including its dataclass decorator and annotations.
    base_fields = [
        ("req_ids", list[str]),
        ("resumed_req_ids", set[str]),
        ("new_token_ids", list[list[int]]),
        ("all_token_ids", dict[str, list[int]]),
        ("new_block_ids", list[tuple[list[int], ...] | None]),
        ("num_computed_tokens", list[int]),
        ("num_output_tokens", list[int]),
    ]
    extra_fields = [("replay_start", list[int] | None, field(default=None))] if parent_has_default else []
    base = make_dataclass("CachedRequestData", [*base_fields, *extra_fields])
    upstream = ModuleType("vllm.v1.core.sched.output")
    upstream.__dict__.update(
        CachedRequestData=base,
        NewRequestData=make_dataclass("NewRequestData", []),
        SchedulerOutput=make_dataclass("SchedulerOutput", []),
    )
    request = ModuleType("vllm.v1.request")
    request.__dict__["Request"] = object
    engine = ModuleType("vllm_omni.engine")
    engine.__dict__["AdditionalInformationPayload"] = object
    for dependency in (upstream, request, engine):
        monkeypatch.setitem(sys.modules, dependency.__name__, dependency)

    path = Path(__file__).parents[3] / "vllm_omni/core/sched/output.py"
    spec = importlib.util.spec_from_file_location("_omni_output_dataclass_regression", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    cls = module.OmniCachedRequestData
    assert issubclass(cls, base)

    kwargs = dict(
        req_ids=["req"],
        resumed_req_ids={"req"},
        new_token_ids=[[4]],
        all_token_ids={"req": [1, 2, 3, 4]},
        new_block_ids=[([7, 8],)],
        num_computed_tokens=[3],
        num_output_tokens=[1],
        prompt_token_ids={"req": [1, 2, 3]},
        additional_information={"req": {"stage": 1}},
    )
    if replay_start is not None:
        kwargs["replay_start"] = replay_start
    instance = cls(**kwargs)
    for name, value in kwargs.items():
        assert getattr(instance, name) == value
    if parent_has_default:
        assert instance.replay_start == replay_start
    else:
        assert not hasattr(instance, "replay_start")

    parameters = inspect.signature(cls).parameters
    dataclass_fields = {item.name: item for item in fields(cls)}
    for name in ("prompt_token_ids", "additional_information"):
        assert parameters[name].kind is inspect.Parameter.KEYWORD_ONLY
        assert dataclass_fields[name].default is MISSING
        assert dataclass_fields[name].default_factory is MISSING
        with pytest.raises(TypeError, match=name):
            cls(**{key: value for key, value in kwargs.items() if key != name})
