# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Tests for the acceleration-neutral semantic model-region seam."""

from collections.abc import Callable
from typing import TypeVar

import torch.nn as nn

from vllm_omni.diffusion.forward_context import ForwardContext, override_forward_context
from vllm_omni.diffusion.model_region import (
    ModelRegion,
    execute_model_region,
)

T = TypeVar("T")


class _Handler:
    def __init__(self):
        self.calls = 0

    def execute(
        self,
        region: ModelRegion,
        owner: nn.Module,
        compute: Callable[[], T],
    ) -> T:
        self.calls += 1
        return compute()


def test_no_context_directly_computes():
    assert execute_model_region(ModelRegion.REFERENCE_HINTS, nn.Identity(), lambda: "fresh") == "fresh"


def test_context_without_handler_directly_computes():
    with override_forward_context(ForwardContext()):
        assert execute_model_region(ModelRegion.REFERENCE_HINTS, nn.Identity(), lambda: "fresh") == "fresh"


def test_context_handler_intercepts_once():
    handler = _Handler()
    context = ForwardContext(model_region_handler=handler)
    with override_forward_context(context):
        result = execute_model_region(ModelRegion.REFERENCE_HINTS, nn.Identity(), lambda: "fresh")
    assert result == "fresh"
    assert handler.calls == 1
