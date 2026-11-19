# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from dataclasses import dataclass
from types import SimpleNamespace

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
from vllm_omni.diffusion.models.sensenova_u1.sensenova_u1_transformer import (
    SenseNovaU1Model,
    _build_3d_rope,
)
from vllm_omni.platforms import current_omni_platform
from vllm_omni.transformers_utils.configs.sensenova_u1 import (
    SenseNovaU1Config,
    SenseNovaU1LLMConfig,
    SenseNovaU1VisionConfig,
)

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion]


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
@pytest.mark.cpu
def test_token_routing_fallback(indicators, expected):
    model = _RoutingModel()
    model(
        inputs_embeds=torch.zeros(1, 2, 4),
        indexes=torch.zeros(3, 2, dtype=torch.long),
        image_gen_indicators=None if indicators is None else torch.tensor([indicators]),
    )
    assert model.layers[0].seen == [expected]


@pytest.mark.parametrize("teacache", [False, True])
@pytest.mark.cpu
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
@pytest.mark.cpu
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
    cfg_norm: str = "none"
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
@pytest.mark.cpu
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


def _make_step_extra(pipe, p, is_it2i=False, img_cond=True):
    ns = pipe._init_noise_and_schedule(p)
    caches = {"idx_cond": torch.zeros(3, ns.token_h * ns.token_w, dtype=torch.long)}
    if is_it2i and img_cond:
        caches["img_cond"] = {}
    return {
        "p": p,
        "ns": ns,
        "caches": caches,
        "is_it2i": is_it2i,
        "_image_prediction": ns.image_prediction,
        "_current_step_index": 0,
    }


@pytest.mark.parametrize(
    "is_it2i,interval,scale,img_scale",
    [
        (False, (0.25, 0.75), 4.0, 1.0),
        (False, (0.0, 1.0), 1.0, 1.0),
        (True, (0.25, 0.75), 4.0, 2.0),
        (True, (0.0, 1.0), 1.0, 1.0),
    ],
)
@torch.inference_mode()
@pytest.mark.cpu
def test_single_step_denoise_passes_cfg_decision(is_it2i, interval, scale, img_scale):
    """Single-request step path must derive and pass use_cfg (PR review P1)."""
    pipe = _LoopProbe(noise_embedding=False)
    p = _Params(cfg_interval=interval, cfg_scale=scale, img_cfg_scale=img_scale)
    extra = _make_step_extra(pipe, p, is_it2i=is_it2i)
    ns = extra["ns"]
    for step_i in range(p.num_steps):
        extra["_current_step_index"] = step_i
        pipe._step_denoise_single(ns.image_prediction, step_i, extra)
        t = ns.timesteps[step_i]
        if is_it2i:
            expected = ((t > interval[0] and t < interval[1]) or interval[0] == 0) and (scale != 1 or img_scale != 1)
        else:
            expected = t >= interval[0] and t <= interval[1] and scale > 1
        assert pipe.steps[-1][0] == bool(expected)


@pytest.mark.cpu
@torch.inference_mode()
def test_single_step_equal_scale_it2i_uses_it2i_interval():
    """Equal-scale IT2I omits img_cond; step path must still apply IT2I CFG."""
    pipe = _LoopProbe(noise_embedding=False)
    p = _Params(cfg_interval=(0.25, 0.75), cfg_scale=2.0, img_cfg_scale=2.0)
    extra = _make_step_extra(pipe, p, is_it2i=True, img_cond=False)
    ns = extra["ns"]
    expected = []
    for step_i in range(p.num_steps):
        extra["_current_step_index"] = step_i
        pipe._step_denoise_single(ns.image_prediction, step_i, extra)
        t_val = float(ns.timesteps[step_i])
        # IT2I: exclusive interval (or lo == 0), plus either scale != 1.
        expected.append((t_val > 0.25 and t_val < 0.75) and (p.cfg_scale != 1 or p.img_cfg_scale != 1))
    assert [step[0] for step in pipe.steps] == expected


@pytest.mark.cpu
def test_combine_cfg_equal_scale_it2i_uses_uncond():
    """Equal-scale IT2I (no img_cond) must combine cond/uncond, not fall to T2I."""
    pipe = _LoopProbe(noise_embedding=False)
    p = _Params(cfg_scale=2.0, img_cfg_scale=2.0)
    cond = torch.ones(1, 1)
    uncond = torch.full((1, 1), 3.0)
    actual = pipe._combine_cfg(cond, uncond, None, p, 0, is_it2i=True)
    expected = uncond + p.cfg_scale * (cond - uncond)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def _flash_cache():
    layer = SimpleNamespace(
        flash_prefix_len=1,
        flash_total_len=2,
        flash_k_cache=torch.zeros(1, 1, 1, 1),
        flash_v_cache=torch.zeros(1, 1, 1, 1),
    )
    return SimpleNamespace(layers=[layer]), layer


@torch.inference_mode()
@pytest.mark.cpu
def test_step_batch_drain_and_cancellation_cleanup(monkeypatch):
    """Multi-request batch -> cancel one -> drain to single -> retire all (PR review P1/P2)."""
    pipe = _LoopProbe(noise_embedding=False)
    pipe._step_states = {}
    requests = {}
    layers = {}
    for rid in ("a", "b"):
        extra = _make_step_extra(pipe, _Params())
        cache, layer = _flash_cache()
        extra["caches"]["cond"] = cache
        pipe._step_states[rid] = extra
        requests[rid] = SimpleNamespace(request_id=rid, extra=extra)
        layers[rid] = layer

    batched_calls = []
    monkeypatch.setattr(
        pipe, "_batched_denoise_step", lambda ib: batched_calls.append(list(ib.request_ids)) or torch.zeros(1)
    )
    pipe.denoise_step(SimpleNamespace(request_ids=["a", "b"]))
    assert batched_calls == [["a", "b"]]

    # Cancellation before final decode must release pipeline-owned state.
    pipe.release_step_request(requests["b"])
    assert "b" not in pipe._step_states
    assert not hasattr(layers["b"], "flash_k_cache")
    pipe.release_step_request(requests["b"])  # idempotent
    assert "a" in pipe._step_states
    assert hasattr(layers["a"], "flash_k_cache")

    # Draining to one request routes to the single-request path (with use_cfg).
    pipe.denoise_step(SimpleNamespace(request_ids=["a"]))
    assert len(pipe.steps) == 1

    pipe.release_step_request(requests["a"])
    assert pipe._step_states == {}
    assert not hasattr(layers["a"], "flash_k_cache")


class _BatchedLayerProbe(nn.Module):
    def __init__(self):
        super().__init__()
        self.seen = None

    def _forward_gen_batched(self, hidden_states, position_embeddings, prefix_kv_list, q_lens):
        self.seen = position_embeddings
        return hidden_states


class _BatchedRoutingModel(nn.Module):
    forward_gen_batched = SenseNovaU1Model.forward_gen_batched

    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([_BatchedLayerProbe()])
        self.norm_mot_gen = nn.Identity()
        self.rotary_emb, self.rotary_emb_hw = _build_3d_rope(SenseNovaU1LLMConfig(head_dim=16))


@torch.inference_mode()
@pytest.mark.cpu
def test_batched_forward_builds_rope_before_layer_calls():
    """Batched layers must receive (cos, sin) pairs, not raw indexes (PR review P1)."""
    model = _BatchedRoutingModel()
    inputs_embeds = torch.randn(2, 3, 8)
    indexes = torch.zeros(3, 2, 3, dtype=torch.long)
    indexes[1] = 2
    indexes[2] = 3

    def _empty_prefix():
        layer = SimpleNamespace(
            flash_prefix_len=0,
            flash_k_cache=torch.zeros(1, 0, 1, 1),
            flash_v_cache=torch.zeros(1, 0, 1, 1),
        )
        return SimpleNamespace(layers=[layer])

    model.forward_gen_batched(inputs_embeds, indexes, [_empty_prefix(), _empty_prefix()], [3, 3])

    seen = model.layers[0].seen
    assert len(seen) == 3
    expected = (
        model.rotary_emb(inputs_embeds, indexes[0]),
        model.rotary_emb_hw(inputs_embeds, indexes[1]),
        model.rotary_emb_hw(inputs_embeds, indexes[2]),
    )
    for (cos, sin), (expected_cos, expected_sin) in zip(seen, expected):
        torch.testing.assert_close(cos, expected_cos, rtol=0, atol=0)
        torch.testing.assert_close(sin, expected_sin, rtol=0, atol=0)
        assert cos.shape[0] == 2  # per-request batch dim preserved


class _BatchedCausalLM(nn.Module):
    """Wraps the probe model with the same return contract as the real causal LM.

    `_batched_predict_v` reads `outputs.hidden_states`; the real
    `SenseNovaU1ForCausalLM.forward_gen_batched` returns exactly that field.
    """

    def __init__(self):
        super().__init__()
        self.model = _BatchedRoutingModel()

    def forward_gen_batched(self, inputs_embeds, indexes, past_key_values_list, q_lens):
        outputs = self.model.forward_gen_batched(inputs_embeds, indexes, past_key_values_list, q_lens)
        return SimpleNamespace(hidden_states=outputs.last_hidden_state)


@torch.inference_mode()
@pytest.mark.cpu
def test_batched_denoise_step_runs_real_batch_forward_and_cleanup():
    """Two requests must share one real batched forward, then drain and clean up.

    Unlike the routing-only test above, this does not patch `_batched_denoise_step`:
    it drives `denoise_step` through `_batched_predict_v` and the real
    `SenseNovaU1Model.forward_gen_batched` (the P1 RoPE fix) with probe decoder
    layers, so the batch entry and RoPE construction are exercised for real.
    """
    pipe = _LoopProbe(noise_embedding=False)
    pipe.fm_modules["fm_head"] = nn.Linear(8, 48)
    pipe.language_model = _BatchedCausalLM()
    pipe._step_states = {}

    requests = {}
    layers = {}
    for rid in ("a", "b"):
        extra = _make_step_extra(pipe, _Params())
        cache, layer = _flash_cache()
        extra["caches"]["cond"] = cache
        pipe._step_states[rid] = extra
        requests[rid] = SimpleNamespace(request_id=rid, extra=extra)
        layers[rid] = layer

    v_pred = pipe.denoise_step(SimpleNamespace(request_ids=["a", "b"]))
    assert v_pred.shape[0] == 2  # both requests went through one batched forward

    # The probe layer observed the real (cos, sin) pairs, not raw integer indexes.
    seen = pipe.language_model.model.layers[0].seen
    assert len(seen) == 3
    for cos, sin in seen:
        assert cos.dtype.is_floating_point
        assert cos.shape[0] == 2  # per-request batch dim preserved

    # Cancellation before final decode must release pipeline-owned state.
    pipe.release_step_request(requests["b"])
    assert "b" not in pipe._step_states
    assert not hasattr(layers["b"], "flash_k_cache")
    pipe.release_step_request(requests["b"])  # idempotent
    assert "a" in pipe._step_states
    assert hasattr(layers["a"], "flash_k_cache")

    # Draining to one request routes to the single-request path (with use_cfg).
    pipe.denoise_step(SimpleNamespace(request_ids=["a"]))
    assert len(pipe.steps) == 1

    pipe.release_step_request(requests["a"])
    assert pipe._step_states == {}
    assert not hasattr(layers["a"], "flash_k_cache")


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
