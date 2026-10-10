# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""CPU contracts for the estimator, with synthetic local block outputs."""

import numpy as np
import pytest
import torch

from vllm_omni.diffusion.cache.teacache import coefficient_estimator as estimator
from vllm_omni.diffusion.cache.teacache.extractors import EXTRACTOR_REGISTRY, CacheContext
from vllm_omni.diffusion.distributed.cfg_parallel import CFGParallelMixin

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


class _Transformer(torch.nn.Module):
    def forward(self, x):
        return (x,)


def _extract(module, x):
    return CacheContext(
        modulated_input=x,
        hidden_states=x,
        encoder_hidden_states=None,
        temb=x,
        run_transformer_blocks=lambda: (x.square(),),
        postprocess=lambda h: (h,),
    )


class _Pipeline(CFGParallelMixin):
    def __init__(self, stage, cfg):
        self.transformer = _Transformer()
        self.stage = stage
        self.cfg = cfg
        self.requests = 0
        self.expected = []

    def forward(self, batch):
        # Distinct stages, requests and branches must never become adjacent
        # timesteps merely because their transformer calls were interleaved.
        offset = 1000 * self.stage + 100 * self.requests
        positive = [torch.tensor([[offset + t + 1.0]]) for t in range(5)]
        negative = [torch.tensor([[offset + 20.0 + 2 * t]]) for t in range(5)]
        for pos, neg in zip(positive, negative):
            self.predict_noise_maybe_with_cfg(
                do_true_cfg=self.cfg,
                true_cfg_scale=5.0,
                positive_kwargs={"x": pos},
                negative_kwargs={"x": neg} if self.cfg else None,
                cfg_normalize=False,
            )
        self.expected.extend([positive, negative] if self.cfg else [positive])
        self.requests += 1


@pytest.mark.parametrize("cfg", [False, True])
def test_estimator_fits_branch_local_pairs_across_requests_and_stages(monkeypatch, cfg):
    monkeypatch.setitem(EXTRACTOR_REGISTRY, "WanTransformer3DModel", _extract)
    monkeypatch.setattr(estimator.WanAdapter, "get_transformer", lambda p: (p.transformer, "WanTransformer3DModel"))
    monkeypatch.setattr(torch.accelerator, "empty_cache", lambda: None)
    collectors = []
    for stage in (0, 1):
        pipeline = _Pipeline(stage, cfg)
        monkeypatch.setattr(estimator.WanAdapter, "load_pipeline", lambda *args: pipeline)
        collector = estimator.TeaCacheCoefficientEstimator("unused", model_type="Wan", device="cpu")
        for _ in range(2):
            collector.collect_from_prompt("synthetic", num_inference_steps=5)
        collectors.append((collector, pipeline))

    polyfit = np.polyfit
    fits = []

    def record_fit(x, y, degree):
        fits.append((x.copy(), y.copy()))
        return polyfit(x, y, degree)

    monkeypatch.setattr(estimator.np, "polyfit", record_fit)
    for collector, pipeline in collectors:
        collector.estimate(poly_order=1)
        expected_x, expected_y = [], []
        assert len(collector.collected_data) == (4 if cfg else 2)
        for actual, expected in zip(collector.collected_data, pipeline.expected):
            assert len(actual) == 5
            for (feature, output), x in zip(actual, expected):
                np.testing.assert_array_equal(feature, x.numpy())
                np.testing.assert_array_equal(output, x.square().numpy())
            for current, next_ in zip(expected, expected[1:]):
                expected_x.append(estimator.calculate_relative_l1(current.numpy(), next_.numpy()))
                expected_y.append(estimator.calculate_relative_l1(current.square().numpy(), next_.square().numpy()))
        x, y = fits[-1]
        assert len(x) == (16 if cfg else 8)  # 8 branch-local pairs per CFG request, never 9.
        np.testing.assert_allclose(x, expected_x)
        np.testing.assert_allclose(y, expected_y)
    assert not np.array_equal(fits[0][0], fits[1][0])


def test_wan_collector_rejects_unstamped_cfg(monkeypatch):
    monkeypatch.setitem(EXTRACTOR_REGISTRY, "WanTransformer3DModel", _extract)
    hook = estimator.DataCollectionHook("WanTransformer3DModel")
    module = _Transformer()
    module.do_true_cfg = True
    with pytest.raises(ValueError, match="requires an explicit cfg_branch"):
        hook.new_forward(module, torch.ones(1, 1))
    assert hook.stop_collection() == []
