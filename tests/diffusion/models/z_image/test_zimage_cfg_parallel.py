# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from vllm_omni.diffusion.distributed import cfg_parallel
from vllm_omni.diffusion.forward_context import get_forward_context, set_forward_context
from vllm_omni.diffusion.models.ming_image.pipeline import MingImageDiffusionPipeline
from vllm_omni.diffusion.models.ming_image.transformer import MingImageTransformer2DModel
from vllm_omni.diffusion.models.z_image.pipeline_z_image import ZImagePipeline
from vllm_omni.diffusion.models.z_image.z_image_transformer import ZImageTransformer2DModel

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]


def _pipeline_for_contract_tests() -> ZImagePipeline:
    pipeline = object.__new__(ZImagePipeline)
    nn.Module.__init__(pipeline)
    return pipeline


@pytest.mark.parametrize("normalize", [False, None])
def test_zimage_cfg_formula_matches_legacy_semantics(normalize):
    pipeline = _pipeline_for_contract_tests()
    positive = torch.tensor([[[[2.0, 1.0]]]])
    negative = torch.tensor([[[[0.5, 0.25]]]])

    actual = pipeline.combine_cfg_noise(positive, negative, 3.0, cfg_normalize=normalize)

    torch.testing.assert_close(actual, positive + 3.0 * (positive - negative))


def test_zimage_cfg_normalization_clamps_to_positive_norm():
    pipeline = _pipeline_for_contract_tests()
    positive = torch.tensor([[[[3.0, 4.0]]]])
    negative = torch.zeros_like(positive)

    actual = pipeline.combine_cfg_noise(positive, negative, 3.0, cfg_normalize=True)

    expected = positive / positive.norm() * positive.norm()
    torch.testing.assert_close(actual, expected)


def test_zimage_cfg_normalization_is_per_sample():
    pipeline = _pipeline_for_contract_tests()
    positive = torch.tensor([[[[1.0]]], [[[10.0]]]])
    negative = torch.zeros_like(positive)

    actual = pipeline.combine_cfg_noise(positive, negative, 3.0, cfg_normalize=1.0)

    torch.testing.assert_close(actual, positive)


class _FakeTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.calls = []

    def forward(self, x, t, cap_feats):
        self.calls.append([item.clone() for item in cap_feats])
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

    def all_gather(self, value, separate_tensors=False):
        assert separate_tensors
        # Rank one contributes the negative result; do not ignore its actual prediction.
        torch.testing.assert_close(value, self.gathered[1])
        return [value.clone() for value in self.gathered]


def test_zimage_cfg_parallel_dispatches_one_branch_per_rank(monkeypatch):
    """Isolated CFG dispatch: DiT samples are [C,F,H,W], predictions [B,C,F,H,W].

    Expected source: the Z-Image transformer list/stack boundary and CFG rank
    contract. Regression: canned gather must not hide a wrong local branch,
    shape or sequential dispatch on rank one.
    """
    pipeline = _pipeline_for_contract_tests()
    pipeline.transformer = _FakeTransformer()
    positive_kwargs = {
        "x": [torch.zeros((1, 1, 4, 6))],
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
        lambda: _FakeCfgGroup([torch.full((1, 1, 1, 4, 6), 2.0), torch.full((1, 1, 1, 4, 6), 1.0)]),
    )

    actual = pipeline.predict_noise_maybe_with_cfg(
        do_true_cfg=True,
        true_cfg_scale=2.0,
        positive_kwargs=positive_kwargs,
        negative_kwargs=negative_kwargs,
        cfg_normalize=False,
    )

    torch.testing.assert_close(actual, torch.full_like(actual, 4.0))
    assert len(pipeline.transformer.calls) == 1
    torch.testing.assert_close(pipeline.transformer.calls[0][0], negative_kwargs["cap_feats"][0])


@pytest.mark.parametrize("cfg", [False, True])
@pytest.mark.parametrize("rank", [4, 5])
def test_graph_boundary_and_frame_rank_survive_real_cfg_dispatch(monkeypatch, cfg, rank):
    """Input: Ming-Image's [B,C,F,H,W] or Z-Image's [B,C,H,W] layout.

    Expected source: each actual transformer invocation has a graph boundary;
    sequential CFG invokes twice and preserves F. Regression: loop-level-only
    marker and unconditional unsqueeze/squeeze. The mixin is not mocked.
    """
    pipe = object.__new__(MingImageDiffusionPipeline) if rank == 5 else _pipeline_for_contract_tests()
    if rank == 5:
        nn.Module.__init__(pipe)
        pipe._num_frames_per_prompt = 2
    pipe.scheduler = _FakeScheduler()
    pipe.od_config = SimpleNamespace(dtype=torch.float32)
    pipe._interrupt = False
    pipe._uses_cudagraph_trees = True
    events = []
    monkeypatch.setattr(torch.compiler, "cudagraph_mark_step_begin", lambda: events.append("mark"))

    class ShapeObserver(nn.Module):
        def forward(self, x, t, cap_feats):
            events.append(get_forward_context().cfg_branch)
            assert all(item.shape == (1, 2 if rank == 5 else 1, 4, 6) for item in x)
            return [torch.full_like(item, condition.mean()) for item, condition in zip(x, cap_feats)], {}

    pipe.transformer = ShapeObserver()
    pipe.vae_scale_factor = 2
    initial = pipe.prepare_latents(1, 1, 8, 12, torch.float32, torch.device("cpu"), torch.Generator().manual_seed(71))
    shape = (1, 1, 2, 4, 6) if rank == 5 else (1, 1, 4, 6)
    assert initial.shape == shape
    with set_forward_context(), torch.inference_mode():
        get_forward_context().cfg_branch = "outer"
        result = pipe.diffuse(
            prompt_embeds=[torch.full((2, 2), 2.0)],
            negative_prompt_embeds=[torch.ones(2, 2)],
            latents=initial,
            timesteps=torch.tensor([900.0, 500.0]),
            do_true_cfg=cfg,
            true_cfg_scale=2.0,
            cfg_truncation=None,
        )
        assert get_forward_context().cfg_branch == "outer"
    assert events == (["mark", "positive", "mark", "negative"] if cfg else ["mark", "positive"]) * 2
    assert result.shape == shape
    torch.testing.assert_close(result, initial - (8.0 if cfg else 4.0))


@pytest.mark.parametrize("branch,start", [("positive", 0), ("negative", 2)])
def test_ming_image_fused_context_selects_matching_cfg_half(monkeypatch, branch, start):
    """Producer: MingImageDiffusionPipeline fuses positive/zero direct conditions at 2B.

    Expected source: sequential CFG consumes exactly its B rows of reference/direct
    conditioning. This boundary unit test does not claim real-model validation.
    Regression: both branches consume the positive half or all 2B rows.
    """
    pipe = object.__new__(MingImageTransformer2DModel)
    nn.Module.__init__(pipe)
    reference = torch.arange(4.0).reshape(4, 1, 1, 1, 1).expand(4, 1, 1, 4, 6)
    direct = torch.arange(4.0).reshape(4, 1, 1).expand(4, 2, 3)
    seen = {}

    def parent(self, x, t, cap_feats, **kwargs):
        seen.update(kwargs)
        return x, {}

    monkeypatch.setattr(ZImageTransformer2DModel, "forward", parent)
    with set_forward_context():
        ctx = get_forward_context()
        ctx.ref_latent, ctx.direct_condition, ctx.cfg_branch = reference, direct, branch
        pipe([torch.ones(1, 1, 4, 6)] * 2, torch.ones(2), [torch.ones(2, 3)] * 2)
    torch.testing.assert_close(torch.stack(seen["ref_x"]), reference[start : start + 2])
    torch.testing.assert_close(torch.stack(seen["cap_feats_2"]), direct[start : start + 2])


def test_zimage_diffuse_uses_framework_adapter_and_preserves_sign():
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
