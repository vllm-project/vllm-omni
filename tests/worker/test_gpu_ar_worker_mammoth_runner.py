# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Worker selects MammothGPUARModelRunner for Mammoth checkpoints (#9911)."""

from types import SimpleNamespace
from unittest.mock import patch

import pytest

from vllm_omni.model_executor.models.mammoth_moda2.gpu_ar_model_runner import MammothGPUARModelRunner
from vllm_omni.worker.gpu_ar_model_runner import GPUARModelRunner
from vllm_omni.worker.gpu_ar_worker import (
    _is_mammoth_moda2_vllm_config,
    _warn_moe_specialty_runner_mismatch_once,
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


def test_warn_moe_specialty_runner_mismatch_once_for_non_mammoth(caplog):
    _warn_moe_specialty_runner_mismatch_once._warned = False  # type: ignore[attr-defined]
    cfg = _vllm_config(model_type="qwen2", specialize=True)
    with caplog.at_level("WARNING"):
        _warn_moe_specialty_runner_mismatch_once(cfg, GPUARModelRunner)
        _warn_moe_specialty_runner_mismatch_once(cfg, GPUARModelRunner)
    matches = [r for r in caplog.records if "cudagraph_specialize_moe_routing is enabled" in r.getMessage()]
    assert len(matches) == 1


def test_warn_moe_specialty_runner_mismatch_skipped_for_mammoth(caplog):
    _warn_moe_specialty_runner_mismatch_once._warned = False  # type: ignore[attr-defined]
    cfg = _vllm_config(model_type="mammoth_moda2", specialize=True)
    with caplog.at_level("WARNING"):
        _warn_moe_specialty_runner_mismatch_once(cfg, MammothGPUARModelRunner)
    assert not any("cudagraph_specialize_moe_routing is enabled" in r.getMessage() for r in caplog.records)


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


def test_init_device_constructs_resolved_mammoth_runner():
    """Exercise the same resolve → construct sequence ``init_device`` uses."""
    from vllm_omni.worker import gpu_ar_worker as worker_mod

    cfg = _vllm_config(model_type="mammoth_moda2")
    runner_cls = resolve_gpu_ar_model_runner_cls(cfg, GPUARModelRunner)
    assert runner_cls is MammothGPUARModelRunner

    created: list[type] = []

    def _fake_init(self, vllm_config, device):
        created.append(type(self))
        self.vllm_config = vllm_config
        self.device = device

    with patch.object(MammothGPUARModelRunner, "__init__", _fake_init):
        runner = runner_cls(cfg, "cuda:0")
        worker_mod._warn_moe_specialty_runner_mismatch_once(
            _vllm_config(model_type="mammoth_moda2", specialize=True),
            type(runner),
        )

    assert created == [MammothGPUARModelRunner]
