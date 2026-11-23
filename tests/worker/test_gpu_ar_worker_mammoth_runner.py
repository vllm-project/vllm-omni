# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Mammoth specialty graphs fail clearly on unsupported V2 runners."""

from types import SimpleNamespace
from unittest.mock import patch

import pytest
from vllm.config import CUDAGraphMode

from vllm_omni.cudagraph.moe_routing_compat import ensure_moe_specialty_runner_compatible
from vllm_omni.model_executor.models.mammoth_moda2.mammoth_moda2 import (
    _enable_moe_routing_cudagraph_specialize,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def test_enable_moe_routing_specialize_auto_on_for_full_when_unset():
    compilation_config = SimpleNamespace(cudagraph_mode=CUDAGraphMode.FULL)
    _enable_moe_routing_cudagraph_specialize(SimpleNamespace(compilation_config=compilation_config))
    assert compilation_config.cudagraph_specialize_moe_routing is True


def test_enable_moe_routing_specialize_respects_explicit_false():
    compilation_config = SimpleNamespace(
        cudagraph_mode=CUDAGraphMode.FULL,
        cudagraph_specialize_moe_routing=False,
    )
    _enable_moe_routing_cudagraph_specialize(SimpleNamespace(compilation_config=compilation_config))
    assert compilation_config.cudagraph_specialize_moe_routing is False


def test_enable_moe_routing_specialize_noop_without_compilation_config():
    # Lightweight AR unit stubs may omit compilation_config on VllmConfig.
    _enable_moe_routing_cudagraph_specialize(SimpleNamespace())


def test_ensure_moe_specialty_compatible_rejects_v2_runner_when_flag_on():
    from vllm_omni.worker_v2.omni_model_runner import OmniGPUModelRunner as V2OmniGPUModelRunner

    runner = object.__new__(V2OmniGPUModelRunner)
    runner.compilation_config = SimpleNamespace(cudagraph_specialize_moe_routing=True)
    with pytest.raises(RuntimeError, match="unsupported by this runner") as excinfo:
        ensure_moe_specialty_runner_compatible(runner)
    assert "cudagraph_mode without FULL" in str(excinfo.value)
    assert "cudagraph_specialize_moe_routing=False" not in str(excinfo.value)


def test_v2_load_model_calls_ensure_after_publish(monkeypatch):
    from vllm_omni.worker_v2.omni_model_runner import OmniGPUModelRunner as V2OmniGPUModelRunner

    runner = object.__new__(V2OmniGPUModelRunner)
    runner._weight_share_backend = None
    runner.use_aux_hidden_state_outputs = False
    runner._model_has_structured_output = False
    runner.model = SimpleNamespace(has_preprocess=False)
    runner.supports_mm_inputs = False
    runner.compilation_config = SimpleNamespace(cudagraph_specialize_moe_routing=True)
    order: list[str] = []

    class _Activate:
        def __enter__(self):
            return None

        def __exit__(self, *args):
            return False

    runner.jit_warmup_registry = SimpleNamespace(activate=lambda: _Activate())

    def _super_load(self, *args, **kwargs):
        order.append("super")

    def _ensure(r):
        order.append("ensure")
        ensure_moe_specialty_runner_compatible(r)

    monkeypatch.setattr(V2OmniGPUModelRunner, "_configure_cudagraph_output_contract", lambda self: None)
    monkeypatch.setattr(V2OmniGPUModelRunner, "_publish_shared_backbone_if_configured", lambda self: None)

    # ``super().load_model`` resolves to MRv2PrefixCacheMixin first; stub that
    # entry so we don't need the full upstream load path.
    parent = next(cls for cls in V2OmniGPUModelRunner.__mro__[1:] if "load_model" in vars(cls))
    assert parent.__name__ == "MRv2PrefixCacheMixin", parent
    monkeypatch.setattr(parent, "load_model", _super_load)

    with patch(
        "vllm_omni.cudagraph.moe_routing_compat.ensure_moe_specialty_runner_compatible",
        _ensure,
    ):
        with pytest.raises(RuntimeError, match="unsupported by this runner"):
            V2OmniGPUModelRunner.load_model(runner)
    assert order == ["super", "ensure"]
