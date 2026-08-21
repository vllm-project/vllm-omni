# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import math

import pytest
from vllm.lora.request import LoRARequest

from vllm_omni.diffusion.lora.types import (
    is_registered_lora_request,
    lora_batch_key_fields,
    normalize_lora_composition,
    parse_lora_adapter_specs,
    parse_lora_registration_specs,
)
from vllm_omni.entrypoints.openai.utils import parse_lora_request
from vllm_omni.lora.utils import stable_lora_int_id

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _request(adapter_id: int, path: str | None = None) -> LoRARequest:
    return LoRARequest(
        lora_name=f"adapter-{adapter_id}",
        lora_int_id=adapter_id,
        lora_path=path or f"/tmp/adapter-{adapter_id}",
    )


def test_composition_is_sorted_and_combines_duplicate_scales() -> None:
    composition = normalize_lora_composition(
        (_request(2), _request(1), _request(2)),
        (0.25, 0.5, 0.75),
    )

    assert [adapter.adapter_id for adapter in composition] == [1, 2]
    assert [adapter.scale for adapter in composition] == [0.5, 1.0]


def test_composition_rejects_invalid_scales_and_id_collisions() -> None:
    with pytest.raises(ValueError, match="same length"):
        normalize_lora_composition((_request(1), _request(2)), (1.0,))
    with pytest.raises(ValueError, match="finite"):
        normalize_lora_composition(_request(1), math.inf)
    with pytest.raises(ValueError, match="refers to both"):
        normalize_lora_composition(
            (_request(1, "/tmp/a"), _request(1, "/tmp/b")),
            (1.0, 1.0),
        )


def test_batch_key_fields_coalesce_inactive_compositions() -> None:
    assert lora_batch_key_fields(None) == (None, 1.0)
    assert lora_batch_key_fields((), ()) == (None, 1.0)
    assert lora_batch_key_fields((_request(2), _request(1)), (0.25, 0.75)) == (
        (1, 2),
        (0.75, 0.25),
    )


def test_startup_specs_support_repeated_weighted_adapters() -> None:
    composition = parse_lora_adapter_specs(
        [
            "/tmp/adapter-a=0.25",
            '{"path":"/tmp/adapter-b","name":"style","scale":0.75,"int_id":22}',
        ]
    )

    by_name = {adapter.request.lora_name: adapter for adapter in composition}
    assert by_name["style"].scale == 0.75
    assert by_name["style"].adapter_id == 22
    assert by_name["adapter-a"].scale == 0.25


def test_startup_specs_reject_registered_request_sentinel() -> None:
    with pytest.raises(ValueError, match="reserved prefix"):
        parse_lora_adapter_specs(["vllm-omni://registered-lora/7"])


def test_dynamic_registration_rejects_scale_ids_and_duplicate_names() -> None:
    with pytest.raises(ValueError, match="does not accept startup scales"):
        parse_lora_registration_specs(["/tmp/adapter-a=0.25"])
    with pytest.raises(ValueError, match="does not accept int_id"):
        parse_lora_registration_specs(['{"path":"/tmp/adapter-a","name":"style","int_id":7}'])
    with pytest.raises(ValueError, match="name 'style' is registered more than once"):
        parse_lora_registration_specs(
            [
                '{"path":"/tmp/adapter-a","name":"style"}',
                '{"path":"/tmp/adapter-b","name":"style"}',
            ]
        )


def test_lora_specs_reject_unknown_and_ambiguous_fields() -> None:
    with pytest.raises(ValueError, match="unknown field.*sclae"):
        parse_lora_adapter_specs(['{"path":"/tmp/adapter-a","sclae":0.5}'])
    with pytest.raises(ValueError, match="multiple path fields"):
        parse_lora_adapter_specs(['{"path":"/tmp/adapter-a","lora_path":"/tmp/adapter-b"}'])
    with pytest.raises(ValueError, match="unknown field.*sclae"):
        parse_lora_request({"name": "style", "sclae": 0.5})
    with pytest.raises(ValueError, match="multiple scale fields"):
        parse_lora_request({"name": "style", "scale": 1.0, "lora_scale": 0.5})


def test_dynamic_registration_uses_unique_names_and_private_ids() -> None:
    registry = parse_lora_registration_specs(
        [
            "/tmp/adapter-a",
            '{"path":"/tmp/adapter-b","name":"style"}',
        ]
    )

    assert [request.lora_name for request in registry] == ["adapter-a", "style"]
    assert [request.lora_int_id for request in registry] == [
        stable_lora_int_id("adapter-a"),
        stable_lora_int_id("style"),
    ]


def test_request_parser_accepts_weighted_list_and_explicit_empty() -> None:
    requests, scales = parse_lora_request(
        [
            {"name": "cinematic", "scale": 0.8},
            {"name": "style", "scale": 0.2},
        ]
    )

    assert isinstance(requests, tuple)
    assert {request.lora_name for request in requests} == {"cinematic", "style"}
    assert dict(zip((request.lora_name for request in requests), scales, strict=True)) == {
        "cinematic": 0.8,
        "style": 0.2,
    }
    assert parse_lora_request([]) == ((), ())


def test_request_parser_accepts_registered_names_without_paths() -> None:
    requests, scales = parse_lora_request(
        [
            {"name": "cinematic", "scale": 0.8},
            {"name": "style", "scale": 0.2},
        ]
    )

    assert isinstance(requests, tuple)
    assert {request.lora_name for request in requests} == {"cinematic", "style"}
    assert all(is_registered_lora_request(request) for request in requests)
    assert dict(zip((request.lora_name for request in requests), scales, strict=True)) == {
        "cinematic": 0.8,
        "style": 0.2,
    }


def test_request_parser_rejects_missing_name_and_internal_id() -> None:
    with pytest.raises(ValueError, match="name must be a non-empty string"):
        parse_lora_request({"scale": 1.0})
    with pytest.raises(ValueError, match="int_id is internal"):
        parse_lora_request({"int_id": 2})


def test_request_parser_rejects_paths() -> None:
    with pytest.raises(ValueError, match="paths are not accepted"):
        parse_lora_request({"name": "style", "path": "/tmp/style"})


def test_request_parser_preserves_scale_cancellation_as_explicit_empty() -> None:
    assert parse_lora_request(
        [
            {"name": "style", "scale": 1.0},
            {"name": "style", "scale": -1.0},
        ]
    ) == ((), ())
