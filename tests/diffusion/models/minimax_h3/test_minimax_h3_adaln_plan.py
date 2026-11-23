# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Request-level AdaLN projections must return exactly what the per-step projection returns, or nothing."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from vllm.model_executor.layers.linear import UnquantizedLinearMethod

from tests.helpers.mark import hardware_test
from vllm_omni.diffusion.models.minimax_h3 import adaln_plan
from vllm_omni.diffusion.models.minimax_h3 import minimax_h3_blocks as blocks
from vllm_omni.diffusion.models.minimax_h3 import minimax_h3_transformer as h3
from vllm_omni.diffusion.models.minimax_h3.adaln_cache import MiniMaxH3RuntimeAdalnCache
from vllm_omni.diffusion.models.minimax_h3.adaln_plan import MiniMaxH3AdalnPlan, build_adaln_plan

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion]


class _Linear(torch.nn.Linear):
    def __init__(self, in_features, out_features, *, params_dtype, bias=True, **kwargs):
        super().__init__(in_features, out_features, bias=bias, dtype=params_dtype)
        self.quant_method = UnquantizedLinearMethod()

    def forward(self, x):
        return super().forward(x), None


@pytest.fixture(autouse=True)
def tp1(monkeypatch):
    import vllm.distributed

    monkeypatch.setattr(vllm.distributed, "get_tensor_model_parallel_world_size", lambda: 1)
    monkeypatch.setattr(blocks, "ColumnParallelLinear", _Linear)


def _proj(device, *, hidden_size=8, time_embed_dim=4, seed=0):
    torch.manual_seed(seed)
    arch = h3.MiniMaxH3DiTArchConfig(hidden_size=hidden_size, time_embed_dim=time_embed_dim)
    cache = MiniMaxH3RuntimeAdalnCache(max_bytes=0, plan=True)
    proj = h3.MiniMaxH3AdalnProj(
        arch, 18 * hidden_size, None, expand_ratio=6, modality_num=3, prefix="blocks.0.adaln_proj", adaln_cache=cache
    ).to(device)
    with torch.no_grad():
        proj.linear.weight.normal_(std=0.02)
        proj.linear.bias.normal_(std=0.02)
    return proj, cache


def _time_embedder(dim, device):
    freqs = torch.linspace(0.5, 4.0, dim, device=device)
    return lambda unique: torch.sin(unique.reshape(-1, 1) * freqs)


def _steps(device):
    """(host step key, that forward's unique timesteps)."""
    sets = ([0.25, 0.999], [0.5, 0.999], [0.1, 0.6, 0.999], [0.999])  # the last has one row: never planned
    return [(("step", i), torch.tensor(values, device=device)) for i, values in enumerate(sets)]


def _run(proj, cache, plan, key, unique, embed):
    """Return (planned forward, per-step forward) for one step."""
    t_emb = embed(unique)
    cache.plan = plan
    plan.select(key)
    plan.bind(t_emb)
    planned = proj(t_emb)
    cache.plan = None
    return planned, proj(t_emb)


@pytest.mark.cpu
def test_planned_projection_matches_per_step_cpu():
    proj, cache = _proj("cpu")
    embed = _time_embedder(4, "cpu")
    steps = _steps("cpu")
    with torch.inference_mode():
        plan = MiniMaxH3AdalnPlan(embed, [proj], steps)
        assert plan.planned_steps == 3
        for key, unique in steps:
            planned, expected = _run(proj, cache, plan, key, unique, embed)
            for got, want in zip(planned, expected, strict=True):
                assert torch.equal(got, want)
    # CPU BLAS may pick different kernels for the batched call; then the plan keeps nothing and misses.
    assert plan.hits + plan.misses == len(steps)
    assert plan.misses >= 1


@hardware_test(res={"cuda": ["H100", "B200"]})
@pytest.mark.parametrize("width", ["small", "h3"])
def test_planned_projection_matches_per_step_cuda(width):
    if not torch.cuda.is_available():
        pytest.skip("Requires CUDA")
    # "h3" is MiniMax-H3's block projection: 2688 -> 18 * 5376.
    hidden_size, time_embed_dim = (8, 4) if width == "small" else (5376, 2688)
    proj, cache = _proj("cuda", hidden_size=hidden_size, time_embed_dim=time_embed_dim)
    embed = _time_embedder(time_embed_dim, "cuda")
    steps = _steps("cuda")
    with torch.inference_mode():
        plan = MiniMaxH3AdalnPlan(embed, [proj], steps)
        for key, unique in steps:
            planned, expected = _run(proj, cache, plan, key, unique, embed)
            for got, want in zip(planned, expected, strict=True):
                assert torch.equal(got, want)
    if width == "h3":
        # The real width must take the planned path for every multi-row step.
        assert (plan.hits, plan.misses) == (3, 1)


@pytest.mark.cpu
def test_single_row_and_unplanned_steps_fall_back():
    proj, cache = _proj("cpu")
    embed = _time_embedder(4, "cpu")
    with torch.inference_mode():
        plan = MiniMaxH3AdalnPlan(embed, [proj], [("one-row", torch.tensor([0.999]))])
        assert plan.planned_steps == 0
        for key, values in (("one-row", [0.999]), ("unplanned", [0.3, 0.999]), (None, [0.3, 0.999])):
            planned, expected = _run(proj, cache, plan, key, torch.tensor(values), embed)
            assert all(torch.equal(a, b) for a, b in zip(planned, expected, strict=True))
    assert (plan.hits, plan.misses) == (0, 3)


@pytest.mark.cpu
def test_bind_checks_host_metadata_and_consumes_the_key(monkeypatch):
    proj, _ = _proj("cpu")
    embed = _time_embedder(4, "cpu")
    unique = torch.tensor([0.25, 0.999])
    plan = MiniMaxH3AdalnPlan(embed, [proj], [("k", unique)], safe=lambda _: True)
    monkeypatch.setattr(plan, "_tables", {id(proj): {2: torch.zeros(1, 2, 18 * 8)}})
    for t_emb in (embed(unique).to(torch.bfloat16), embed(torch.tensor([0.1, 0.25, 0.999]))):
        # A dtype cast before AdaLN (control branch) or a different row count disables the lookup.
        plan.select("k")
        plan.bind(t_emb)
        assert plan.lookup(proj) is None
    plan.select("k")
    plan.bind(embed(unique))
    assert plan.lookup(proj) is not None
    plan.bind(embed(unique))  # the key was consumed by the previous forward
    assert plan.lookup(proj) is None


@pytest.mark.cpu
def test_batched_result_that_differs_is_dropped():
    proj, _ = _proj("cpu")
    original = proj.linear.forward

    def row_count_dependent(x):
        out, bias = original(x)
        return out + (x.shape[0] > 3), bias

    proj.linear.forward = row_count_dependent
    embed = _time_embedder(4, "cpu")
    steps = _steps("cpu")[:2]
    plan = MiniMaxH3AdalnPlan(embed, [proj], steps)
    plan.select(steps[0][0])
    plan.bind(embed(steps[0][1]))
    assert plan.lookup(proj) is None


@pytest.mark.cpu
@pytest.mark.parametrize("case", ["lora", "hook", "quantized", "offloaded", "tp2"])
def test_unsafe_projections_are_not_planned(monkeypatch, case):
    proj, _ = _proj("cpu")
    if case == "lora":
        import vllm.lora.layers

        monkeypatch.setattr(vllm.lora.layers, "BaseLayerWithLoRA", _Linear)
    elif case == "hook":
        proj.linear.register_forward_hook(lambda *args: None)
    elif case == "quantized":
        proj.linear.quant_method = object()
    elif case == "offloaded":
        proj._offload_weights = True
    else:
        import vllm.distributed

        monkeypatch.setattr(vllm.distributed, "get_tensor_model_parallel_world_size", lambda: 2)
    assert not adaln_plan.projection_plan_safe(proj)
    embed = _time_embedder(4, "cpu")
    unique = torch.tensor([0.25, 0.999])
    plan = MiniMaxH3AdalnPlan(embed, [proj], [("k", unique)])
    plan.select("k")
    plan.bind(embed(unique))
    assert plan.lookup(proj) is None


class _Branch:
    seq_len, device = 6, torch.device("cpu")

    def fill_timesteps(self, timesteps, *, t_video, t_audio, imgvid_cond_timestep, audio_ref_cond_timestep):
        timesteps.copy_(torch.tensor([t_video, t_video, t_audio, imgvid_cond_timestep, audio_ref_cond_timestep, 0.0]))


@pytest.mark.cpu
def test_build_plans_every_schedule_step_and_respects_the_switch():
    proj, cache = _proj("cpu")
    model = SimpleNamespace(
        adaln_cache=cache, blocks=[SimpleNamespace(adaln_proj=proj)], time_embedder=_time_embedder(4, "cpu")
    )
    sigmas = [1.0, 0.75, 0.5, 0.0]
    branch = _Branch()
    plan = build_adaln_plan(model, branch, sigmas, sigmas, (0.999, 1.0))
    assert plan.planned_steps == 3
    # The sampler names step 1 by the same host scalars: times 1 - sigma, condition max(t, anchor).
    plan.select(adaln_plan.adaln_step_key(branch, (0.25, 0.25), (0.999, 1.0)))
    plan.bind(model.time_embedder(torch.tensor([0.0, 0.25, 0.999, 1.0])))
    assert plan.lookup(proj) is not None or plan._tables == {}  # CPU BLAS may drop the table
    assert plan._entry == (4, 0)
    cache.plan_enabled = False
    assert build_adaln_plan(model, _Branch(), sigmas, sigmas, (0.999, 1.0)) is None


@pytest.mark.cpu
@pytest.mark.parametrize("value", [None, True, False, "true"])
def test_constructor_reads_the_plan_switch(monkeypatch, value):
    from tests.diffusion.models.minimax_h3.test_minimax_h3_quantization import (
        _FakeAttention,
        _FakeLinear,
        _small_od_config,
    )

    for name in ("ColumnParallelLinear", "RowParallelLinear"):
        monkeypatch.setattr(h3, name, _FakeLinear)
    for name in ("ColumnParallelLinear", "RowParallelLinear", "MergedColumnParallelLinear", "QKVParallelLinear"):
        monkeypatch.setattr(blocks, name, _FakeLinear)
    monkeypatch.setattr(blocks, "Attention", _FakeAttention)
    monkeypatch.setattr(h3, "get_tensor_model_parallel_world_size", lambda: 1)
    config = _small_od_config()
    config.cache_config = {} if value is None else {"minimax_h3_adaln_plan": value}
    if value == "true":
        with pytest.raises(ValueError, match="minimax_h3_adaln_plan must be a boolean"):
            h3.MiniMaxH3DiTModel(config, diffusers_weights=False)
        return
    model = h3.MiniMaxH3DiTModel(config, diffusers_weights=False)
    assert model.adaln_cache.plan_enabled is (value is not False)
    assert model.adaln_cache.plan is None
