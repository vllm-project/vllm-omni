# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Request-level AdaLN projections must return exactly what the per-step projection returns, or nothing."""

from __future__ import annotations

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


def _proj(device, *, hidden_size=8, time_embed_dim=4, offload=False, rowwise=False):
    torch.manual_seed(0)
    arch = h3.MiniMaxH3DiTArchConfig(hidden_size=hidden_size, time_embed_dim=time_embed_dim)
    cache = MiniMaxH3RuntimeAdalnCache(max_bytes=0, offload_weights=offload, plan=True)
    proj = h3.MiniMaxH3AdalnProj(
        arch, 18 * hidden_size, None, expand_ratio=6, modality_num=3, prefix="blocks.0.adaln_proj", adaln_cache=cache
    ).to(device)
    with torch.no_grad():
        proj.linear.weight.normal_(std=0.02)
        proj.linear.bias.normal_(std=0.02)
    if rowwise:
        # Row by row, so the batched table call is bit-identical to the per-step call on any CPU BLAS.
        linear = proj.linear
        proj.linear.forward = lambda x: (torch.stack([torch.nn.Linear.forward(linear, r) for r in x]), None)
    return proj, cache


def _time_embedder(dim, device):
    freqs = torch.linspace(0.5, 4.0, dim, device=device)
    return lambda unique: torch.sin(unique.reshape(-1, 1) * freqs)


def _steps(device):
    """(host step key, that forward's unique timesteps); the last has one row and is never planned."""
    sets = ([0.25, 0.999], [0.5, 0.999], [0.1, 0.6, 0.999], [0.999])
    return [(("step", i), torch.tensor(values, device=device)) for i, values in enumerate(sets)]


def _run(proj, cache, plan, key, t_emb):
    """Return (planned forward, per-step forward) for one step."""
    cache.plan = plan
    plan.select(key)
    plan.bind(t_emb)
    planned = proj(t_emb)
    cache.plan = None
    return planned, proj(t_emb)


def _check_steps(proj, cache, plan, steps, embed):
    for key, unique in steps:
        planned, expected = _run(proj, cache, plan, key, embed(unique))
        assert all(torch.equal(got, want) for got, want in zip(planned, expected, strict=True))


@pytest.mark.cpu
def test_planned_projection_matches_per_step_cpu():
    proj, cache = _proj("cpu", rowwise=True)
    embed = _time_embedder(4, "cpu")
    with torch.inference_mode():
        plan = MiniMaxH3AdalnPlan(embed, _steps("cpu"), safe=lambda _: True)
        _check_steps(proj, cache, plan, _steps("cpu"), embed)
    assert set(plan._tables[id(proj)]) == {2, 3}


@hardware_test(res={"cuda": ["H100", "B200"]})
def test_planned_h3_projection_matches_per_step_cuda():
    if not torch.cuda.is_available():
        pytest.skip("Requires CUDA")
    # MiniMax-H3's block projection: 2688 -> 18 * 5376.
    proj, cache = _proj("cuda", hidden_size=5376, time_embed_dim=2688)
    embed = _time_embedder(2688, "cuda")
    with torch.inference_mode():
        plan = MiniMaxH3AdalnPlan(embed, _steps("cuda"))
        _check_steps(proj, cache, plan, _steps("cuda"), embed)
    # The real width keeps both row counts' tables: step 1 is served from the table step 0 built.
    assert set(plan._tables[id(proj)]) == {2, 3}


@pytest.mark.cpu
def test_unplanned_forwards_fall_back():
    proj, _ = _proj("cpu")
    embed = _time_embedder(4, "cpu")
    unique = torch.tensor([0.25, 0.999])
    plan = MiniMaxH3AdalnPlan(embed, [("k", unique), ("one-row", torch.tensor([0.999]))], safe=lambda _: True)
    plan._tables = {id(proj): {2: torch.zeros(1, 2, 18 * 8)}}
    # Single row, unknown or no key, an embedding cast before AdaLN, or a different row count: per-step call.
    for key, t_emb in (
        ("one-row", embed(torch.tensor([0.999]))),
        ("unplanned", embed(unique)),
        (None, embed(unique)),
        ("k", embed(unique).to(torch.bfloat16)),
        ("k", embed(torch.tensor([0.1, 0.25, 0.999]))),
    ):
        plan.select(key)
        plan.bind(t_emb)
        assert plan.lookup(proj, t_emb) is None
    plan.select("k")
    t_emb = embed(unique)
    plan.bind(t_emb)
    assert plan.lookup(proj, t_emb) is not None
    # A forward that did not bind (another embedding object) never reuses the bound step's row.
    assert plan.lookup(proj, embed(unique)) is None
    plan.bind(t_emb)  # the key was consumed by the previous forward
    assert plan.lookup(proj, t_emb) is None


@pytest.mark.cpu
def test_time_embeddings_are_computed_inside_the_forward():
    # Under model-level CPU offload the time embedder is only resident inside the model's forward.
    resident = []
    embed = _time_embedder(4, "cpu")

    def offloaded(unique):
        assert resident, "time embedder called before the model forward"
        return embed(unique)

    plan = MiniMaxH3AdalnPlan(offloaded, _steps("cpu"))
    resident.append(True)
    plan.select(("step", 0))
    plan.bind(embed(torch.tensor([0.25, 0.999])))
    assert plan.rows().shape == (2, 2, 4)  # steps 0 and 1 have two rows each


@pytest.mark.cpu
def test_tables_over_budget_are_not_kept(monkeypatch):
    monkeypatch.setattr(adaln_plan, "MAX_TABLE_BYTES", 0)
    proj, cache = _proj("cpu", rowwise=True)
    embed = _time_embedder(4, "cpu")
    with torch.inference_mode():
        plan = MiniMaxH3AdalnPlan(embed, _steps("cpu"), safe=lambda _: True)
        _check_steps(proj, cache, plan, _steps("cpu"), embed)
    assert plan._tables == {} and plan._pending == {}


@pytest.mark.cpu
def test_batched_result_that_differs_is_dropped():
    proj, cache = _proj("cpu")
    original = proj.linear.forward
    proj.linear.forward = lambda x: (original(x)[0] + (x.shape[0] > 3), None)
    embed = _time_embedder(4, "cpu")
    steps = _steps("cpu")[:2]
    with torch.inference_mode():
        plan = MiniMaxH3AdalnPlan(embed, steps, safe=lambda _: True)
        _check_steps(proj, cache, plan, steps, embed)
    assert plan._tables == {} and plan._pending == {}


@pytest.mark.cpu
@pytest.mark.parametrize("case", ["lora", "hook", "quantized", "tp2"])
def test_unsafe_projections_are_not_planned(monkeypatch, case):
    proj, _ = _proj("cpu")
    if case == "lora":
        import vllm.lora.layers

        monkeypatch.setattr(vllm.lora.layers, "BaseLayerWithLoRA", _Linear)
    elif case == "hook":
        proj.linear.register_forward_hook(lambda *args: None)
    elif case == "quantized":
        proj.linear.quant_method = object()
    else:
        import vllm.distributed

        monkeypatch.setattr(vllm.distributed, "get_tensor_model_parallel_world_size", lambda: 2)
    assert not adaln_plan.projection_plan_safe(proj)


class _CountingStager:
    def __init__(self):
        self.loads = self.offloads = 0

    def load(self):
        self.loads += 1

    def offload(self):
        self.offloads += 1


@pytest.mark.cpu
def test_offloaded_weights_are_staged_once_per_request():
    proj, cache = _proj("cpu", offload=True, rowwise=True)
    stager = _CountingStager()
    proj._prepare_weight_stager = lambda _device: setattr(proj, "_weight_stager", stager)
    embed = _time_embedder(4, "cpu")
    steps = [(("step", i), torch.tensor([0.05 * (i + 1), 0.999])) for i in range(10)]
    planned_loads = []
    with torch.inference_mode():
        plan = MiniMaxH3AdalnPlan(embed, steps, safe=lambda _: True)
        for key, unique in steps:
            before = stager.loads
            cache.plan = plan
            t_emb = embed(unique)
            plan.select(key)
            plan.bind(t_emb)
            planned = proj(t_emb)
            planned_loads.append(stager.loads - before)
            cache.plan = None
            assert all(torch.equal(a, b) for a, b in zip(planned, proj(embed(unique)), strict=True))
    assert stager.loads == stager.offloads
    # One staging builds the table; later forwards stage nothing.
    assert planned_loads == [1] + [0] * 9


class _Branch:
    seq_len, device = 6, torch.device("cpu")

    def fill_timesteps(self, timesteps, *, t_video, t_audio, imgvid_cond_timestep, audio_ref_cond_timestep):
        timesteps.copy_(torch.tensor([t_video, t_video, t_audio, imgvid_cond_timestep, audio_ref_cond_timestep, 0.0]))


@pytest.mark.cpu
def test_build_plans_every_schedule_step_and_respects_the_switch():
    proj, cache = _proj("cpu")
    model = torch.nn.Module()
    model.proj, model.adaln_cache, model.time_embedder = proj, cache, _time_embedder(4, "cpu")
    sigmas = [1.0, 0.75, 0.5, 0.0]
    branch = _Branch()
    plan = build_adaln_plan(model, branch, sigmas, sigmas, (0.999, 1.0))
    assert cache.plan is plan and len(plan._index) == 3
    # The sampler names step 1 by the same host scalars: times 1 - sigma, condition max(t, anchor).
    plan.select(adaln_plan.adaln_step_key(branch, (0.25, 0.25), (0.999, 1.0)))
    t_emb = model.time_embedder(torch.tensor([0.0, 0.25, 0.999, 1.0]))
    plan.bind(t_emb)
    assert plan.wants(proj, t_emb) and plan.rows().shape == (2, 4, 4)
    cache.plan = None
    proj.linear.register_forward_hook(lambda *args: None)
    assert build_adaln_plan(model, _Branch(), sigmas, sigmas, (0.999, 1.0)) is None  # nothing plan-safe
    cache.plan_enabled = False
    assert build_adaln_plan(model, _Branch(), sigmas, sigmas, (0.999, 1.0)) is None


@pytest.mark.cpu
@pytest.mark.parametrize("value", [None, False])
def test_plan_is_on_by_default(monkeypatch, value):
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
    assert h3.MiniMaxH3DiTModel(config, diffusers_weights=False).adaln_cache.plan_enabled is (value is None)
