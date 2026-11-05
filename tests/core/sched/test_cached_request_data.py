# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Required Omni payloads must coexist with platform-added base defaults."""

import importlib.util
import sys
from dataclasses import dataclass, field, fields
from pathlib import Path

import msgspec
import pytest
import torch
from vllm.v1.core.sched import output as upstream_output

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


@dataclass
class _ReplayCachedRequestData(upstream_output.CachedRequestData):
    # Ascend's bounded-replay adaptation adds this default before Omni imports.
    replay_start: dict[str, int] = field(default_factory=dict)


@pytest.fixture(params=[False, True], ids=["official", "ascend-replay-default"])
def cached_data(request, monkeypatch):
    base_type = _ReplayCachedRequestData if request.param else upstream_output.CachedRequestData
    # Execute the real source against each base before its dataclass decorator,
    # without reloading or replacing the canonical module used by other tests.
    source = Path(__file__).resolve().parents[3] / "vllm_omni/core/sched/output.py"
    spec = importlib.util.spec_from_file_location("_omni_cached_request_data_test", source)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    with monkeypatch.context() as import_patch:
        import_patch.setattr(upstream_output, "CachedRequestData", base_type)
        spec.loader.exec_module(module)
    arguments = {
        "req_ids": ["req"],
        "resumed_req_ids": {"req"},
        "new_token_ids": [[11]],
        "all_token_ids": {"req": [10, 11]},
        "new_block_ids": [([7, 8],)],
        "num_computed_tokens": [2],
        "num_output_tokens": [1],
    }
    if request.param:
        arguments["replay_start"] = {"req": 2}
    base = base_type(**arguments)
    return base_type, module.OmniCachedRequestData, base


@pytest.mark.parametrize("missing", ["prompt_token_ids", "additional_information"])
def test_omni_cached_fields_remain_mandatory(cached_data, missing):
    base_type, omni_type, base = cached_data
    arguments = {item.name: getattr(base, item.name) for item in fields(base_type)}
    arguments.update(prompt_token_ids={"req": [10]}, additional_information={"req": None})
    del arguments[missing]
    with pytest.raises(TypeError, match=missing):
        omni_type(**arguments)


def test_inherited_positional_constructor_contract_is_preserved(cached_data):
    base_type, omni_type, base = cached_data
    base_arguments = [getattr(base, item.name) for item in fields(base_type)]
    prompt_ids = {"req": [10]}
    information = {"req": {"audio_codes": [1, 2]}}
    result = omni_type(*base_arguments, prompt_token_ids=prompt_ids, additional_information=information)
    assert isinstance(result, base_type)
    for item in fields(base_type):
        assert getattr(result, item.name) is getattr(base, item.name)
    assert result.prompt_token_ids is prompt_ids
    assert result.additional_information is information
    if isinstance(base, _ReplayCachedRequestData):
        # Leaving the platform field out must retain its per-instance factory.
        first = omni_type(*base_arguments[:-1], prompt_token_ids=prompt_ids, additional_information=information)
        second = omni_type(*base_arguments[:-1], prompt_token_ids=prompt_ids, additional_information=information)
        assert first.replay_start == second.replay_start == {}
        assert first.replay_start is not second.replay_start
        first.replay_start["req"] = 2
        assert second.replay_start == {}


def test_from_base_preserves_nonempty_platform_state_and_payload_identity(cached_data):
    base_type, omni_type, base = cached_data
    prompt_ids = {"req": [10]}
    tensor = torch.tensor([1.0, 2.0])
    information = {"req": {"tensor": tensor}}
    result = omni_type.from_base(base, prompt_token_ids=prompt_ids, additional_information=information)
    for item in fields(base_type):
        assert getattr(result, item.name) is getattr(base, item.name)
    if isinstance(base, _ReplayCachedRequestData):
        assert result.replay_start == {"req": 2}
        assert result.replay_start is base.replay_start
    assert result.prompt_token_ids is prompt_ids
    assert result.additional_information is information
    assert result.additional_information["req"]["tensor"] is tensor


def test_typed_msgpack_roundtrip_retains_fields_and_requires_omni_payloads(cached_data):
    _, omni_type, base = cached_data
    original = omni_type.from_base(
        base, prompt_token_ids={"req": [10]}, additional_information={"req": {"audio_codes": [1, 2]}}
    )
    encoded = msgspec.msgpack.encode(original)
    decoded = msgspec.msgpack.decode(encoded, type=omni_type)
    assert decoded == original
    wire = msgspec.msgpack.decode(encoded)
    assert set(wire) == {item.name for item in fields(omni_type)}
    if isinstance(base, _ReplayCachedRequestData):
        assert wire["replay_start"] == {"req": 2}
    for missing in ("prompt_token_ids", "additional_information"):
        incomplete = {key: value for key, value in wire.items() if key != missing}
        with pytest.raises(msgspec.ValidationError, match=missing):
            msgspec.msgpack.decode(msgspec.msgpack.encode(incomplete), type=omni_type)
