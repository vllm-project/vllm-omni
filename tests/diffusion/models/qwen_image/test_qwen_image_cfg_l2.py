# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from datetime import timedelta

import pytest
import torch

from tests.helpers.mark import hardware_test
from vllm_omni.diffusion.distributed.cfg_parallel import CFGParallelMixin
from vllm_omni.diffusion.models.qwen_image import pipeline_qwen_image as model
from vllm_omni.platforms import current_omni_platform

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion]


def _pipeline(cls=model.QwenImagePipeline):
    # Exercise real caller methods without downloading the checkpoint.
    obj = cls.__new__(cls)
    torch.nn.Module.__init__(obj)
    obj._use_fused_cfg_l2 = cls is model.QwenImagePipeline
    return obj


@pytest.mark.cpu
def test_single_output_routing_and_fallback(monkeypatch):
    pipeline = _pipeline()
    p, n = torch.randn(2, 3, 64), torch.randn(2, 3, 64)
    calls = []

    def fused(p, n, w):
        calls.append((p, n, w))
        return p + 10

    monkeypatch.setattr(model, "try_fused_cfg_l2", fused)
    for positive, negative in ((p, n), ((p,), (n,))):
        assert torch.equal(pipeline.combine_cfg_noise(positive, negative, 4.0, True), p + 10)
    assert len(calls) == 2
    # Disabled normalization and multi-output containers must not call fusion.
    for normalize, positive, negative in ((False, p, n), (True, (p, p), (n, n))):
        result = pipeline.combine_cfg_noise(positive, negative, 4.0, normalize, {"context": "retained"})
        expected = CFGParallelMixin.combine_cfg_noise(pipeline, positive, negative, 4.0, normalize)
        torch.testing.assert_close(result, expected)
    assert len(calls) == 2
    monkeypatch.setattr(model, "try_fused_cfg_l2", lambda *args: None)
    torch.testing.assert_close(
        pipeline.combine_cfg_noise(p, n, 4.0, True), CFGParallelMixin.combine_cfg_noise(pipeline, p, n, 4.0, True)
    )


@pytest.mark.cpu
def test_custom_normalization_and_combine_are_preserved(monkeypatch):
    class CustomNormalization(model.QwenImagePipeline):
        def cfg_normalize_function(self, p, combined):
            return combined + 42

    class CustomCombination(model.QwenImagePipeline):
        def combine_cfg_noise(self, p, n, scale, cfg_normalize=False, kwargs=None):
            return p - 42

    monkeypatch.setattr(model, "try_fused_cfg_l2", lambda *args: pytest.fail("custom subclass must retain reference"))
    p, n = torch.randn(2, 3, 64), torch.randn(2, 3, 64)
    pipeline = _pipeline(CustomNormalization)
    assert torch.equal(pipeline.combine_cfg_noise(p, n, 4.0, True), n + 4.0 * (p - n) + 42)
    assert torch.equal(_pipeline(CustomCombination).combine_cfg_noise(p, n, 4.0, True), p - 42)


def _worker(rank, init_file):
    from vllm_omni.diffusion.distributed.parallel_state import (
        destroy_distributed_env,
        init_distributed_environment,
        initialize_model_parallel,
    )

    current_omni_platform.set_device(rank)
    torch.distributed.init_process_group(
        "nccl", init_method=f"file://{init_file}", rank=rank, world_size=2, timeout=timedelta(seconds=90)
    )
    init_distributed_environment(
        world_size=2, rank=rank, local_rank=rank, distributed_init_method=f"file://{init_file}", backend="nccl"
    )
    initialize_model_parallel(cfg_parallel_size=2, sequence_parallel_size=1, backend="nccl")
    try:
        pipeline = _pipeline()
        pipeline.predict_noise = lambda prediction: prediction
        generator = torch.Generator(device=f"cuda:{rank}").manual_seed(7382)
        for dtype in (torch.float32, torch.bfloat16):
            p = torch.randn(2, 9, 64, device=f"cuda:{rank}", dtype=dtype, generator=generator)
            n = torch.randn(2, 9, 64, device=f"cuda:{rank}", dtype=dtype, generator=generator)
            # Branch data identical on both ranks, so the local reference is
            # also the sequential-CFG result. Slice uses actual mixin logic.
            actual = pipeline.predict_noise_maybe_with_cfg(
                True, 4.0, {"prediction": p}, {"prediction": n}, cfg_normalize=True, output_slice=7
            )
            expected = CFGParallelMixin.combine_cfg_noise(pipeline, p[:, :7], n[:, :7], 4.0, True)
            atol, rtol = (0.015625, 0.016) if dtype == torch.bfloat16 else (3e-6, 3e-6)
            torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol)
            assert actual.shape == (2, 7, 64)
    finally:
        destroy_distributed_env()


@hardware_test(res={"cuda": "L4"}, num_cards=2)
@pytest.mark.parallel
def test_real_cfg_exchange_then_fusion(tmp_path):
    torch.multiprocessing.spawn(_worker, args=(str(tmp_path / "cfg-init"),), nprocs=2, join=True)
