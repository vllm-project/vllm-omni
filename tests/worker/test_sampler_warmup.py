# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""MRv2 startup registration of the seeded-sampling Triton kernels."""

import sys
from types import ModuleType, SimpleNamespace

import pytest
from vllm.model_executor.warmup.jit_warmup import JitWarmupRegistry

from vllm_omni.worker import gpu_ar_worker, sampler_warmup
from vllm_omni.worker.base import OmniGPUWorkerBase
from vllm_omni.worker.gpu_ar_worker import GPUARWorker
from vllm_omni.worker.sampler_warmup import register_top_k_top_p_warmup

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

_SAMPLER_MODULE = "vllm.v1.sample.ops.topk_topp_sampler"


class _FakeKernel:
    pass


@pytest.fixture
def fake_kernels(monkeypatch):
    kernels = [_FakeKernel(), _FakeKernel()]
    module = ModuleType(_SAMPLER_MODULE)

    def register_top_k_top_p_warmups():
        for kernel in kernels:
            JitWarmupRegistry.register(kernel)

    module.register_top_k_top_p_warmups = register_top_k_top_p_warmups  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, _SAMPLER_MODULE, module)
    monkeypatch.setattr("vllm.triton_utils.HAS_TRITON", True)
    monkeypatch.setattr(sampler_warmup.current_platform, "is_cuda", lambda: True)
    return kernels


def _worker(*, use_v2=True, is_last_pp_rank=True, is_pooling_model=False):
    return SimpleNamespace(
        use_v2_model_runner=use_v2,
        model_runner=SimpleNamespace(
            jit_warmup_registry=JitWarmupRegistry(vllm_config=None),
            is_last_pp_rank=is_last_pp_rank,
            is_pooling_model=is_pooling_model,
        ),
    )


def test_registers_kernels_with_v2_runner_registry(fake_kernels):
    worker = _worker()

    assert register_top_k_top_p_warmup(worker) is True
    assert len(worker.model_runner.jit_warmup_registry) == len(fake_kernels)


@pytest.mark.parametrize(
    "kwargs",
    [{"use_v2": False}, {"is_last_pp_rank": False}, {"is_pooling_model": True}],
    ids=["v1-runner", "non-last-pp-rank", "pooling-model"],
)
def test_skips_runners_that_never_sample_with_mrv2(fake_kernels, kwargs):
    worker = _worker(**kwargs)

    assert register_top_k_top_p_warmup(worker) is False
    assert len(worker.model_runner.jit_warmup_registry) == 0


def test_skips_non_cuda_platforms(fake_kernels, monkeypatch):
    monkeypatch.setattr(sampler_warmup.current_platform, "is_cuda", lambda: False)
    worker = _worker()

    assert register_top_k_top_p_warmup(worker) is False
    assert len(worker.model_runner.jit_warmup_registry) == 0


def test_missing_upstream_helper_skips_warmup_instead_of_failing(fake_kernels, monkeypatch):
    monkeypatch.setitem(sys.modules, _SAMPLER_MODULE, ModuleType(_SAMPLER_MODULE))
    worker = _worker()

    assert register_top_k_top_p_warmup(worker) is False
    assert len(worker.model_runner.jit_warmup_registry) == 0


def test_changed_upstream_helper_signature_skips_warmup_instead_of_failing(fake_kernels, monkeypatch):
    def register_top_k_top_p_warmups(max_num_seqs):
        raise AssertionError("unreachable")

    module = ModuleType(_SAMPLER_MODULE)
    module.register_top_k_top_p_warmups = register_top_k_top_p_warmups  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, _SAMPLER_MODULE, module)
    worker = _worker()

    assert register_top_k_top_p_warmup(worker) is False
    assert len(worker.model_runner.jit_warmup_registry) == 0


def test_ar_worker_registers_sampler_warmup_before_upstream_kernel_warmup(monkeypatch):
    # The patch below only intercepts super() if no class between GPUARWorker
    # and OmniGPUWorkerBase in the MRO overrides compile_or_warm_up_model.
    next_impl = next(cls for cls in GPUARWorker.__mro__[1:] if "compile_or_warm_up_model" in vars(cls))
    assert next_impl is OmniGPUWorkerBase
    calls = []
    monkeypatch.setattr(gpu_ar_worker, "warm_up_moe_prefill_kernels", lambda worker: calls.append("moe_prefill"))
    monkeypatch.setattr(gpu_ar_worker, "register_top_k_top_p_warmup", lambda worker: calls.append("sampler"))
    monkeypatch.setattr(OmniGPUWorkerBase, "compile_or_warm_up_model", lambda self: calls.append("upstream"))

    GPUARWorker.compile_or_warm_up_model(object.__new__(GPUARWorker))

    assert calls == ["moe_prefill", "sampler", "upstream"]
