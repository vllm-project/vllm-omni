# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""MRv2 startup registration of the seeded-sampling Triton kernel."""

import sys
from types import ModuleType, SimpleNamespace

import pytest
from vllm.model_executor.warmup.jit_warmup import JitWarmupRegistry

from vllm_omni.worker import sampler_warmup
from vllm_omni.worker.sampler_warmup import register_top_k_top_p_warmup

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


class _FakeTopKTopP:
    def register_warmup(self):
        JitWarmupRegistry.register(self)


@pytest.fixture
def fake_kernel(monkeypatch):
    kernel = _FakeTopKTopP()
    module = ModuleType("vllm.v1.sample.ops.topk_topp_triton")
    module._topk_topp = kernel  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "vllm.v1.sample.ops.topk_topp_triton", module)
    monkeypatch.setattr("vllm.triton_utils.HAS_TRITON", True)
    monkeypatch.setattr(sampler_warmup.current_platform, "is_cuda", lambda: True)
    return kernel


def _worker(*, use_v2=True, is_last_pp_rank=True, is_pooling_model=False):
    return SimpleNamespace(
        use_v2_model_runner=use_v2,
        model_runner=SimpleNamespace(
            jit_warmup_registry=JitWarmupRegistry(vllm_config=None),
            is_last_pp_rank=is_last_pp_rank,
            is_pooling_model=is_pooling_model,
        ),
    )


def test_registers_kernel_with_v2_runner_registry(fake_kernel):
    worker = _worker()

    assert register_top_k_top_p_warmup(worker) is True
    assert len(worker.model_runner.jit_warmup_registry) == 1


@pytest.mark.parametrize(
    "kwargs",
    [{"use_v2": False}, {"is_last_pp_rank": False}, {"is_pooling_model": True}],
    ids=["v1-runner", "non-last-pp-rank", "pooling-model"],
)
def test_skips_runners_that_never_sample_with_mrv2(fake_kernel, kwargs):
    worker = _worker(**kwargs)

    assert register_top_k_top_p_warmup(worker) is False
    assert len(worker.model_runner.jit_warmup_registry) == 0


def test_skips_non_cuda_platforms(fake_kernel, monkeypatch):
    monkeypatch.setattr(sampler_warmup.current_platform, "is_cuda", lambda: False)
    worker = _worker()

    assert register_top_k_top_p_warmup(worker) is False
    assert len(worker.model_runner.jit_warmup_registry) == 0
