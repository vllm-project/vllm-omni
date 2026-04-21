# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest

from vllm_omni.model_executor.models.minicpmo4_5.minicpmo4_5 import (
    MiniCPMO4_5ForConditionalGeneration,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def test_wrapper_make_omni_output_passthrough():
    wrapper = object.__new__(MiniCPMO4_5ForConditionalGeneration)
    wrapper.model = SimpleNamespace(make_omni_output=lambda outputs, **kwargs: ("wrapped", outputs, kwargs))

    assert wrapper.make_omni_output("x", foo="bar") == ("wrapped", "x", {"foo": "bar"})
