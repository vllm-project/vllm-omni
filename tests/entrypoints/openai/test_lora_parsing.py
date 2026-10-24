# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Unit tests for request-level LoRA parsing helpers (single + stacked)."""

import pytest
from fastapi import HTTPException

from vllm_omni.entrypoints.openai.lora import _get_lora_from_json_str, _parse_lora_requests
from vllm_omni.entrypoints.openai.utils import parse_lora_request, parse_lora_requests
from vllm_omni.lora.request import LoRARequest
from vllm_omni.lora.utils import stable_lora_int_id

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


class TestParseLoraRequest:
    def test_none_returns_none(self) -> None:
        assert parse_lora_request(None) == (None, None)

    def test_single_dict(self) -> None:
        lora_request, scale = parse_lora_request({"name": "a", "path": "/tmp/a", "scale": 0.7})
        assert isinstance(lora_request, LoRARequest)
        assert lora_request.lora_name == "a"
        assert lora_request.lora_path == "/tmp/a"
        assert lora_request.lora_int_id == stable_lora_int_id("/tmp/a")
        assert scale == 0.7

    def test_rejects_non_dict(self) -> None:
        with pytest.raises(ValueError, match="expected an object"):
            parse_lora_request(["a"])

    def test_rejects_missing_fields(self) -> None:
        with pytest.raises(ValueError, match="name and path are required"):
            parse_lora_request({"name": "a"})


class TestParseLoraRequests:
    def test_none_returns_none(self) -> None:
        assert parse_lora_requests(None) == (None, None)

    def test_single_dict_backward_compatible(self) -> None:
        # A single object keeps the legacy scalar shape so model-specific
        # pipeline hooks that read lora_request/lora_scale directly keep working.
        lora_request, scale = parse_lora_requests({"name": "a", "path": "/tmp/a", "scale": 0.7})
        assert isinstance(lora_request, LoRARequest)
        assert lora_request.lora_name == "a"
        assert scale == 0.7

    def test_one_element_array_unwraps_to_scalar(self) -> None:
        lora_request, scale = parse_lora_requests([{"name": "a", "path": "/tmp/a", "scale": 0.7}])
        assert isinstance(lora_request, LoRARequest)
        assert lora_request.lora_name == "a"
        assert scale == 0.7

    def test_scale_defaults_to_one(self) -> None:
        lora_request, scale = parse_lora_requests({"name": "a", "path": "/tmp/a"})
        assert isinstance(lora_request, LoRARequest)
        assert scale == 1.0

    def test_list_of_dicts(self) -> None:
        lora_requests, scales = parse_lora_requests(
            [
                {"name": "a", "path": "/tmp/a", "scale": 0.7},
                {"name": "b", "path": "/tmp/b"},
            ]
        )
        assert [req.lora_name for req in lora_requests] == ["a", "b"]
        assert scales == [0.7, 1.0]

    def test_rejects_empty_list(self) -> None:
        with pytest.raises(ValueError, match="non-empty array"):
            parse_lora_requests([])

    def test_rejects_non_dict_non_list(self) -> None:
        with pytest.raises(ValueError, match="expected an object"):
            parse_lora_requests("not-a-dict")

    def test_rejects_non_dict_list_item(self) -> None:
        with pytest.raises(ValueError, match="expected an object"):
            parse_lora_requests([{"name": "a", "path": "/tmp/a"}, "oops"])

    def test_rejects_null_list_item(self) -> None:
        with pytest.raises(ValueError, match="array items must be objects"):
            parse_lora_requests([None])

    def test_rejects_list_item_missing_fields(self) -> None:
        with pytest.raises(ValueError, match="name and path are required"):
            parse_lora_requests([{"name": "a"}])


class TestGetLoraFromJsonStr:
    def test_none_returns_none(self) -> None:
        assert _get_lora_from_json_str(None) is None

    def test_object(self) -> None:
        assert _get_lora_from_json_str('{"name": "a", "path": "/tmp/a"}') == {"name": "a", "path": "/tmp/a"}

    def test_array(self) -> None:
        parsed = _get_lora_from_json_str('[{"name": "a", "path": "/tmp/a"}]')
        assert parsed == [{"name": "a", "path": "/tmp/a"}]

    def test_rejects_scalar(self) -> None:
        with pytest.raises(HTTPException) as exc_info:
            _get_lora_from_json_str('"just-a-string"')
        assert exc_info.value.status_code == 400

    def test_rejects_invalid_json(self) -> None:
        with pytest.raises(HTTPException) as exc_info:
            _get_lora_from_json_str("{not json")
        assert exc_info.value.status_code == 400


class TestParseLoraRequestsHttpWrapper:
    def test_value_error_becomes_http_400(self) -> None:
        with pytest.raises(HTTPException) as exc_info:
            _parse_lora_requests([{"name": "a"}])
        assert exc_info.value.status_code == 400

    def test_none_passthrough(self) -> None:
        assert _parse_lora_requests(None) == (None, None)
