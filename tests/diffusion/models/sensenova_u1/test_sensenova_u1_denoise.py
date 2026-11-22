# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from dataclasses import dataclass

import pytest
import torch
from torch import nn

from tests.helpers.mark import hardware_test
from vllm_omni.diffusion.cache.teacache.extractors import extract_sensenova_u1_context
from vllm_omni.diffusion.data import OmniDiffusionConfig
from vllm_omni.diffusion.models.sensenova_u1.pipeline_sensenova_u1 import (
    NEOVisionEmbeddings,
    NEOVisionModel,
    SenseNovaU1Pipeline,
    TimestepEmbedder,
    VisionGrid,
    _patchify,
)
from vllm_omni.diffusion.models.sensenova_u1.sensenova_u1_transformer import SenseNovaU1Model
from vllm_omni.platforms import current_omni_platform
from vllm_omni.transformers_utils.configs.sensenova_u1 import SenseNovaU1Config, SenseNovaU1VisionConfig

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]


class _BranchProbe(nn.Module):
    def __init__(self):
        super().__init__()
        self.input_layernorm_mot_gen = nn.Identity()
        self.seen = []

    def forward(self, h, *, exist_und, exist_gen, **kwargs):
        self.seen.append((exist_und, exist_gen))
        return h + exist_und + 2 * exist_gen


class _RoutingModel(nn.Module):
    forward = SenseNovaU1Model.forward

    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([_BranchProbe()])
        self.norm = nn.Identity()
        self.norm_mot_gen = nn.Identity()

    @staticmethod
    def rotary_emb(h, positions):
        return h, h

    rotary_emb_hw = rotary_emb


def _forbid_item(self, *args, **kwargs):
    raise AssertionError("Unexpected host scalar read")


@pytest.mark.parametrize(
    "indicators,expected",
    [
        (None, (True, False)),
        ([False, False], (True, False)),
        ([True, True], (False, True)),
        ([False, True], (True, True)),
    ],
)
def test_token_routing_fallback(indicators, expected):
    model = _RoutingModel()
    model(
        inputs_embeds=torch.zeros(1, 2, 4),
        indexes=torch.zeros(3, 2, dtype=torch.long),
        image_gen_indicators=None if indicators is None else torch.tensor([indicators]),
    )
    assert model.layers[0].seen == [expected]


@pytest.mark.parametrize("teacache", [False, True])
def test_known_generation_type_avoids_scalar_reads(monkeypatch, teacache):
    model = _RoutingModel()
    embeds = torch.randn(1, 2, 4)
    kwargs = dict(
        inputs_embeds=embeds, indexes=torch.zeros(3, 2, dtype=torch.long), attention_mask={"full_attention": None}
    )
    expected = model(**kwargs, image_gen_indicators=torch.ones(1, 2, dtype=torch.bool)).last_hidden_state
    with monkeypatch.context() as patch:
        patch.setattr(torch.Tensor, "item", _forbid_item)
        if teacache:
            wrapper = nn.Module()
            wrapper.model = model
            ctx = extract_sensenova_u1_context(wrapper, **kwargs, exist_und=False, exist_gen=True, compute_logits=False)
            actual = ctx.postprocess(*ctx.run_transformer_blocks()).hidden_states
        else:
            actual = model(**kwargs, exist_und=False, exist_gen=True).last_hidden_state
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert model.layers[0].seen == [(False, True), (False, True)]


def _vision_config():
    return SenseNovaU1VisionConfig(
        hidden_size=16, llm_hidden_size=[8], patch_size=2, downsample_ratio=[0.5], max_position_embeddings_vision=64
    )


@pytest.mark.parametrize("sizes", [[[4, 6]], [[4, 6], [6, 4]]])
@torch.inference_mode()
def test_vision_grid_reuse_preserves_changing_image_features(monkeypatch, sizes):
    torch.manual_seed(42)
    model = NEOVisionEmbeddings(_vision_config())
    grid_hw = torch.tensor(sizes)
    grid = VisionGrid.from_tensor(grid_hw)
    count = sum(h * w for h, w in sizes)
    results = []
    for _ in range(3):
        pixels = torch.randn(count, 12)
        expected = model(pixels, grid_hw)
        with monkeypatch.context() as patch:
            patch.setattr(VisionGrid, "from_tensor", classmethod(lambda *args: pytest.fail("Grid rebuilt")))
            actual = model(pixels, grid=grid)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        results.append(actual)
    assert not torch.equal(results[0], results[1])


@dataclass
class _Params:
    image_size: tuple[int, int] = (12, 8)
    batch_size: int = 1
    num_steps: int = 4
    cfg_interval: tuple[float, float] = (0.25, 0.75)
    cfg_scale: float = 4.0
    img_cfg_scale: float = 2.0
    seed: int = 42
    timestep_shift: float = 1.0


class _LoopProbe(SenseNovaU1Pipeline):
    def __init__(self, device="cpu", noise_embedding=True):
        nn.Module.__init__(self)
        self.device = torch.device(device)
        self.patch_size = 2
        self.merge_size = 2
        self.model_cfg = SenseNovaU1Config(add_noise_scale_embedding=noise_embedding)
        self.od_config = OmniDiffusionConfig(dtype=torch.float32)
        self.fm_modules = nn.ModuleDict(
            {
                "vision_model_mot_gen": NEOVisionModel(_vision_config()),
                "timestep_embedder": TimestepEmbedder(8, 8),
                "noise_scale_embedder": TimestepEmbedder(8, 8),
            }
        ).to(device)
        self.steps = []
        self.noise_embed_calls = 0
        self.fm_modules["noise_scale_embedder"].register_forward_hook(self._count_noise)

    def _count_noise(self, module, args, output):
        self.noise_embed_calls += 1

    def _denoise(self, image_prediction, ns, t, z, image_embeds, caches, p, step_i, is_it2i, use_cfg):
        self.steps.append((use_cfg, image_embeds.detach().clone()))
        return torch.ones_like(z) * 0.1


@pytest.mark.parametrize(
    "is_it2i,interval,scale,img_scale",
    [
        (False, (0.25, 0.75), 4.0, 1.0),
        (False, (0.0, 1.0), 1.0, 1.0),
        (True, (0.25, 0.75), 4.0, 2.0),
        (True, (0.0, 0.25), 4.0, 2.0),
        (True, (0.0, 1.0), 1.0, 1.0),
    ],
)
@pytest.mark.parametrize("noise_embedding", [False, True])
@torch.inference_mode()
def test_denoise_loop_preserves_cfg_boundaries_and_request_isolation(
    is_it2i, interval, scale, img_scale, noise_embedding
):
    torch.manual_seed(42)
    pipe = _LoopProbe(noise_embedding=noise_embedding)
    for size in [(12, 8), (8, 12)]:
        p = _Params(image_size=size, cfg_interval=interval, cfg_scale=scale, img_cfg_scale=img_scale)
        ns = pipe._init_noise_and_schedule(p)
        expected = []
        for t in ns.timesteps[:-1]:
            if is_it2i:
                enabled = ((t > interval[0] and t < interval[1]) or interval[0] == 0) and (scale != 1 or img_scale != 1)
            else:
                enabled = t >= interval[0] and t <= interval[1] and scale > 1
            expected.append(bool(enabled))
        expected_embeds = []
        image = ns.image_prediction
        for step_i, t in enumerate(ns.timesteps[:-1]):
            pixels = _patchify(image, pipe.patch_size, channel_first=True)
            embeds = pipe._extract_feature(pixels.view(-1, 12), gen_model=True, grid_hw=ns.grid_hw).view(1, -1, 8)
            expanded_t = t.expand(ns.token_h * ns.token_w)
            time_emb = pipe.fm_modules["timestep_embedder"](expanded_t).view(1, -1, 8)
            if noise_embedding:
                noise = torch.full_like(expanded_t, ns.noise_scale / pipe.model_cfg.noise_scale_max_value)
                time_emb = time_emb + pipe.fm_modules["noise_scale_embedder"](noise).view(1, -1, 8)
            expected_embeds.append(embeds + time_emb)
            image = image + (ns.timesteps[step_i + 1] - t) * 0.1
        pipe.steps.clear()
        pipe.noise_embed_calls = 0
        result = pipe._run_denoising_loop(ns, {}, p, is_it2i=is_it2i)
        assert [step[0] for step in pipe.steps] == expected
        assert pipe.noise_embed_calls == int(noise_embedding)
        for (_, actual), expected_embed in zip(pipe.steps, expected_embeds):
            torch.testing.assert_close(actual, expected_embed, rtol=0, atol=0)
        assert result.output["payload"]["image"].size == size
        assert not torch.equal(pipe.steps[0][1], pipe.steps[1][1])


@pytest.mark.skipif(not current_omni_platform.is_cuda(), reason="CUDA required")
@hardware_test(res={"cuda": "L4"})
@torch.inference_mode()
def test_cached_vision_grid_has_no_device_scalar_reads():
    model = NEOVisionEmbeddings(_vision_config()).to("cuda")
    grid_hw = torch.tensor([[4, 6]], device="cuda")
    grid = VisionGrid.from_tensor(grid_hw)
    pixels = torch.randn(24, 12, device="cuda")
    expected = model(pixels, grid_hw)
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as profile:
        actual = model(pixels, grid=grid)
    assert not any(event.key == "aten::_local_scalar_dense" for event in profile.key_averages())
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
