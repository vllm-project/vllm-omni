# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""The image endpoints' ``lora``/``lora_strength`` form fields: a JSON object or a preloaded adapter's name."""

import pytest
from fastapi import HTTPException

from vllm_omni.entrypoints.openai.lora import _lora_form_dict, _parse_lora_request

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def test_bare_name_with_strength_names_a_preloaded_adapter():
    lora = _lora_form_dict("style", 0.8)
    assert lora == {"name": "style", "scale": 0.8}
    request, scale = _parse_lora_request(lora)
    assert request.lora_name == "style" and request.lora_path == "style" and scale == 0.8


def test_none_and_empty_mean_no_adapter():
    assert _lora_form_dict(None, 1.0) is None
    assert _lora_form_dict("none", 1.0) is None
    assert _lora_form_dict("", None) is None


def test_json_object_keeps_its_own_scale_and_path():
    lora = _lora_form_dict('{"name": "a", "path": "/x/a.safetensors", "scale": 0.5}', 0.9)
    request, scale = _parse_lora_request(lora)
    assert request.lora_path == "/x/a.safetensors" and scale == 0.5
    with pytest.raises(HTTPException):
        _parse_lora_request(_lora_form_dict('{"foo": "bar"}'))
