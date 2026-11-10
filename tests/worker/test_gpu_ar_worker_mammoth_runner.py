# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Worker selects MammothGPUARModelRunner for Mammoth checkpoints (#9911)."""

from types import SimpleNamespace
from unittest.mock import patch

import pytest
from vllm.config import CUDAGraphMode
from vllm.v1.worker.gpu_model_runner import GPUModelRunner

from vllm_omni.cudagraph.moe_routing_compat import ensure_moe_specialty_runner_compatible
from vllm_omni.model_executor.models.mammoth_moda2.gpu_ar_model_runner import MammothGPUARModelRunner
from vllm_omni.model_executor.models.mammoth_moda2.mammoth_moda2 import (
    _enable_moe_routing_cudagraph_specialize,
)
from vllm_omni.worker.gpu_ar_model_runner import GPUARModelRunner
from vllm_omni.worker.gpu_ar_worker import (
    _is_mammoth_moda2_vllm_config,
    resolve_gpu_ar_model_runner_cls,
)
from vllm_omni.worker.gpu_model_runner import OmniGPUModelRunner

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _vllm_config(*, model_type=None, architectures=(), specialize: bool = False):
    return SimpleNamespace(
        model_config=SimpleNamespace(
            hf_config=SimpleNamespace(model_type=model_type),
            architectures=list(architectures),
        ),
        compilation_config=SimpleNamespace(cudagraph_specialize_moe_routing=specialize),
    )


@pytest.mark.parametrize(
    ("model_type", "architectures", "expected"),
    [
        ("mammoth_moda2", (), True),
        ("mammoth_moda2_ar", (), True),
        ("qwen2", (), False),
        (None, ("MammothModa2ForConditionalGeneration",), True),
        (None, ("Qwen2ForCausalLM",), False),
        (None, (), False),
    ],
)
def test_is_mammoth_moda2_vllm_config(model_type, architectures, expected):
    assert _is_mammoth_moda2_vllm_config(_vllm_config(model_type=model_type, architectures=architectures)) is expected


def test_resolve_gpu_ar_model_runner_cls_selects_mammoth_for_mammoth_checkpoints():
    cfg = _vllm_config(model_type="mammoth_moda2")
    assert resolve_gpu_ar_model_runner_cls(cfg, GPUARModelRunner) is MammothGPUARModelRunner


def test_resolve_gpu_ar_model_runner_cls_keeps_default_for_non_mammoth():
    cfg = _vllm_config(model_type="qwen2")
    assert resolve_gpu_ar_model_runner_cls(cfg, GPUARModelRunner) is GPUARModelRunner


def test_resolve_gpu_ar_model_runner_cls_respects_custom_runner_cls():
    class CustomARRunner(GPUARModelRunner):
        pass

    cfg = _vllm_config(model_type="mammoth_moda2")
    assert resolve_gpu_ar_model_runner_cls(cfg, CustomARRunner) is CustomARRunner


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


def test_ensure_moe_specialty_compatible_raises_for_non_mammoth_after_flag_set():
    """Flag is normally set during model ``__init__`` inside ``load_model``."""
    runner = object.__new__(OmniGPUModelRunner)
    runner.compilation_config = SimpleNamespace(cudagraph_specialize_moe_routing=True)
    with pytest.raises(RuntimeError, match=r"cudagraph_mode without FULL|MammothGPUARModelRunner"):
        ensure_moe_specialty_runner_compatible(runner)


def test_ensure_moe_specialty_compatible_allows_mammoth_runner():
    runner = object.__new__(MammothGPUARModelRunner)
    runner.compilation_config = SimpleNamespace(cudagraph_specialize_moe_routing=True)
    ensure_moe_specialty_runner_compatible(runner)


def test_ensure_moe_specialty_compatible_rejects_v2_runner_when_flag_on():
    from vllm_omni.worker_v2.omni_model_runner import OmniGPUModelRunner as V2OmniGPUModelRunner

    runner = object.__new__(V2OmniGPUModelRunner)
    runner.compilation_config = SimpleNamespace(cudagraph_specialize_moe_routing=True)
    with pytest.raises(RuntimeError, match="MammothGPUARModelRunner"):
        ensure_moe_specialty_runner_compatible(runner)


def test_ensure_moe_specialty_compatible_noop_when_flag_off():
    runner = object.__new__(OmniGPUModelRunner)
    runner.compilation_config = SimpleNamespace(cudagraph_specialize_moe_routing=False)
    ensure_moe_specialty_runner_compatible(runner)


def test_v1_load_model_ensures_compat_after_model_sets_flag(monkeypatch):
    """``load_model`` must check after super() — when Mammoth flips the flag."""
    runner = object.__new__(OmniGPUModelRunner)
    runner.vllm_config = SimpleNamespace(weight_sharing=None)
    runner.compilation_config = SimpleNamespace(cudagraph_specialize_moe_routing=False)
    order: list[str] = []

    def _super_load(self, *args, **kwargs):
        order.append("super")
        self.compilation_config.cudagraph_specialize_moe_routing = True

    def _ensure(self):
        order.append("ensure")
        ensure_moe_specialty_runner_compatible(self)

    monkeypatch.setattr(GPUModelRunner, "load_model", _super_load)
    monkeypatch.setattr(OmniGPUModelRunner, "_ensure_moe_specialty_runner_compatible", _ensure)
    for name in (
        "_snapshot_prefix_cache_model_policy",
        "_maybe_enable_output_token_ids_for_model_sampler",
        "_init_talker_mtp",
        "_prewarm_attention_capture_workspaces",
        "_report_model_local_kv",
        "_warn_unexposed_stage_hooks",
    ):
        monkeypatch.setattr(OmniGPUModelRunner, name, lambda *a, **k: None)

    with pytest.raises(RuntimeError, match="MammothGPUARModelRunner"):
        OmniGPUModelRunner.load_model(runner)
    assert order == ["super", "ensure"]


def test_v2_load_model_calls_ensure_after_publish(monkeypatch):
    from vllm_omni.worker_v2.omni_model_runner import OmniGPUModelRunner as V2OmniGPUModelRunner

    runner = object.__new__(V2OmniGPUModelRunner)
    runner._weight_share_backend = None
    runner.use_aux_hidden_state_outputs = False
    runner._model_returns_tuple = False
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
        with pytest.raises(RuntimeError, match="MammothGPUARModelRunner"):
            V2OmniGPUModelRunner.load_model(runner)
    assert order == ["super", "ensure"]


def test_base_moe_hooks_fail_fast_when_specialize_enabled():
    runner = object.__new__(OmniGPUModelRunner)
    runner.compilation_config = SimpleNamespace(cudagraph_specialize_moe_routing=True)
    batch_desc = object()
    with pytest.raises(RuntimeError, match="MammothGPUARModelRunner"):
        OmniGPUModelRunner._annotate_moe_routing_batch_descriptor(runner, batch_desc, num_tokens=1)
    with pytest.raises(RuntimeError, match="MammothGPUARModelRunner"):
        OmniGPUModelRunner._select_moe_replay_batch_descriptor(runner, object(), batch_desc, 1)


def test_base_moe_hooks_remain_noop_when_specialize_disabled():
    runner = object.__new__(OmniGPUModelRunner)
    runner.compilation_config = SimpleNamespace(cudagraph_specialize_moe_routing=False)
    batch_desc = object()
    assert OmniGPUModelRunner._annotate_moe_routing_batch_descriptor(runner, batch_desc, num_tokens=1) is batch_desc
    assert OmniGPUModelRunner._select_moe_replay_batch_descriptor(runner, object(), batch_desc, 1) is batch_desc
