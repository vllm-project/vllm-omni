# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from types import ModuleType
from unittest.mock import Mock

import pytest
import torch

from vllm_omni.diffusion.layers import residual_gate_fusion as fusion

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def test_cpu_does_not_import_optional_backend(monkeypatch):
    load = Mock(side_effect=AssertionError("must not import mindiesd on CPU"))
    monkeypatch.setattr(fusion.current_omni_platform, "is_npu", lambda: False)
    monkeypatch.setattr(fusion, "import_module", load)
    assert fusion.resolve_residual_gate_add() is None
    load.assert_not_called()


def test_cpu_inputs_keep_existing_caller_fallback():
    x = torch.randn(1, 3, 65, dtype=torch.bfloat16)
    y = torch.randn_like(x)
    gate = torch.randn(1, 1, 65, dtype=x.dtype)
    op = Mock(side_effect=AssertionError("must not dispatch CPU inputs"))
    assert fusion.try_residual_gate_add(op, x, y, gate) is None
    op.assert_not_called()


def test_missing_backend_keeps_existing_caller_fallback(monkeypatch):
    monkeypatch.setattr(fusion.current_omni_platform, "is_npu", lambda: True)
    monkeypatch.setattr(fusion, "import_module", Mock(side_effect=ImportError("optional package unavailable")))
    assert fusion.resolve_residual_gate_add() is None


def test_missing_callable_keeps_existing_caller_fallback(monkeypatch):
    monkeypatch.setattr(fusion.current_omni_platform, "is_npu", lambda: True)
    monkeypatch.setattr(fusion, "import_module", Mock(return_value=ModuleType("missing_residual_backend")))
    assert fusion.resolve_residual_gate_add() is None
