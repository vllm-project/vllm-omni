# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from vllm_omni.diffusion.distributed import cfg_parallel
from vllm_omni.diffusion.models.z_image.pipeline_z_image import ZImagePipeline

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]


def _pipeline_for_contract_tests() -> ZImagePipeline:
    pipeline = object.__new__(ZImagePipeline)
    nn.Module.__init__(pipeline)
    return pipeline


def test_zimage_cfg_formula_matches_legacy_semantics():
    pipeline = _pipeline_for_contract_tests()
    positive = torch.tensor([[[[2.0, 1.0]]]])
    negative = torch.tensor([[[[0.5, 0.25]]]])

    actual = pipeline.combine_cfg_noise(positive, negative, 3.0, cfg_normalize=False)

    torch.testing.assert_close(actual, positive + 3.0 * (positive - negative))


def test_zimage_cfg_normalization_clamps_to_positive_norm():
    pipeline = _pipeline_for_contract_tests()
    positive = torch.tensor([[[[3.0, 4.0]]]])
    negative = torch.zeros_like(positive)

    actual = pipeline.combine_cfg_noise(positive, negative, 3.0, cfg_normalize=True)

    expected = positive / positive.norm() * positive.norm()
    torch.testing.assert_close(actual, expected)


class _FakeTransformer(nn.Module):
    def forward(self, x, t, cap_feats):
        del t
        outputs = []
        for sample, condition in zip(x, cap_feats):
            outputs.append(torch.full_like(sample, condition.mean()))
        return outputs, {}


class _FakeScheduler:
    def step(self, noise_pred, timestep, latents, return_dict=False):
        del timestep, return_dict
        return (latents + noise_pred,)


class _FakeCfgGroup:
    def __init__(self, gathered):
        self.gathered = gathered

    def all_gather(self, _value, separate_tensors=False):
        assert separate_tensors
        return [value.clone() for value in self.gathered]


def test_zimage_cfg_parallel_dispatches_one_branch_per_rank(monkeypatch):
    pipeline = _pipeline_for_contract_tests()
    pipeline.transformer = _FakeTransformer()
    positive_kwargs = {
        "x": [torch.zeros((1, 1, 1, 1))],
        "t": torch.ones(1),
        "cap_feats": [torch.full((1, 1), 2.0)],
    }
    negative_kwargs = {
        **positive_kwargs,
        "cap_feats": [torch.full((1, 1), 1.0)],
    }
    monkeypatch.setattr(cfg_parallel, "_get_cfg_world_size_or_one", lambda: 2)
    monkeypatch.setattr(cfg_parallel, "get_classifier_free_guidance_rank", lambda: 1)
    monkeypatch.setattr(
        cfg_parallel,
        "get_cfg_group",
        lambda: _FakeCfgGroup([torch.full((1, 1, 1, 1), 2.0), torch.full((1, 1, 1, 1), 1.0)]),
    )

    actual = pipeline.predict_noise_maybe_with_cfg(
        do_true_cfg=True,
        true_cfg_scale=2.0,
        positive_kwargs=positive_kwargs,
        negative_kwargs=negative_kwargs,
        cfg_normalize=False,
    )

    torch.testing.assert_close(actual, torch.full_like(actual, 4.0))


def test_zimage_diffuse_uses_framework_adapter_and_preserves_sign(monkeypatch):
    pipeline = _pipeline_for_contract_tests()
    pipeline.transformer = _FakeTransformer()
    pipeline.scheduler = _FakeScheduler()
    pipeline.od_config = SimpleNamespace(dtype=torch.float32)
    pipeline._interrupt = False

    latents = torch.zeros((1, 1, 1, 1), dtype=torch.float32)
    positive = [torch.full((1, 1), 2.0)]
    negative = [torch.full((1, 1), 1.0)]

    result = pipeline.diffuse(
        prompt_embeds=positive,
        negative_prompt_embeds=negative,
        latents=latents,
        timesteps=torch.tensor([500.0]),
        do_true_cfg=True,
        true_cfg_scale=2.0,
        cfg_normalize=False,
        cfg_truncation=None,
    )

    # Z-Image combines 2 + 2*(2-1) = 4 and negates before scheduler.step.
    torch.testing.assert_close(result, torch.full_like(latents, -4.0))


def test_zimage_diffuse_keeps_cfg_truncation_per_step():
    pipeline = _pipeline_for_contract_tests()
    pipeline.transformer = _FakeTransformer()
    pipeline.scheduler = _FakeScheduler()
    pipeline.od_config = SimpleNamespace(dtype=torch.float32)
    pipeline._interrupt = False

    result = pipeline.diffuse(
        prompt_embeds=[torch.full((1, 1), 2.0)],
        negative_prompt_embeds=[torch.full((1, 1), 1.0)],
        latents=torch.zeros((1, 1, 1, 1)),
        timesteps=torch.tensor([900.0, 100.0]),
        do_true_cfg=True,
        true_cfg_scale=2.0,
        cfg_normalize=False,
        cfg_truncation=0.5,
    )

    # t_norm is 0.1 then 0.9: the second step uses only the positive branch.
    torch.testing.assert_close(result, torch.full((1, 1, 1, 1), -6.0))
