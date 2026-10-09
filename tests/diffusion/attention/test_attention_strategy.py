# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Assignment and execution tests; kernel numerical coverage lives in sparse tests."""

import json
from dataclasses import FrozenInstanceError, asdict
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from vllm_omni.diffusion.attention.contracts import StrategyModelSupport
from vllm_omni.diffusion.attention.strategy import (
    AttentionOperation,
    ForwardStrategyPlan,
    finalize_attention_strategy,
    validate_pipeline_attention_strategy,
    validate_strategy_runtime,
)
from vllm_omni.diffusion.data import (
    AttentionConfig,
    OmniDiffusionConfig,
    build_attention_config,
    parse_attention_config,
)
from vllm_omni.diffusion.forward_context import ForwardContext, override_forward_context

pytestmark = [pytest.mark.diffusion, pytest.mark.core_model, pytest.mark.cpu]


def recipe():
    return json.loads(
        (Path(__file__).resolve().parents[3] / "recipes/attention/cosmos3-subblock-strategy.json").read_text()
    )


def inventory():
    return [
        AttentionOperation(
            f"{role}.{i}", i, f"cosmos3.{role}", "self", invariant=role == "und", component="transformer"
        )
        for role in ("und", "gen", "gen_multi_control")
        for i in range(36)
    ]


def test_recipe_preserves_historical_assignments():
    config = AttentionConfig(**recipe())
    strategy = config.strategy
    operations = inventory()
    strategy.validate_inventory(operations)
    for total in (4, 10, 11, 35, 50):
        for step in range(total):
            layout = ForwardStrategyPlan.from_strategy(strategy).layout_for_step(step, total)
            for operation in operations:
                selected = strategy.assignments(operation)[layout]
                expected_sparse = (
                    step >= 10
                    and operation.role == "cosmos3.gen"
                    and operation.layer not in (1, 2, 15, 30, 31, 32, 33, 34)
                )
                assert (getattr(selected, "name", None) == "block_sparse") == expected_sparse
    # Configuration serialization excludes compiled plans and mutable request state.
    restored = AttentionConfig(**asdict(config))
    assert ForwardStrategyPlan.from_strategy(restored.strategy) == ForwardStrategyPlan.from_strategy(strategy)


def test_static_fractional_and_environment_compatibility(monkeypatch):
    raw = recipe()
    raw["schedule"] = {
        "coordinate": "step_fraction",
        "phases": [
            {"until": 0.3, "layout": "dense"},
            {"until": 0.8, "layout": "mixed"},
            {"until": 1.0, "layout": "dense"},
        ],
    }
    config = AttentionConfig(**raw)
    plan = ForwardStrategyPlan.from_strategy(config.strategy)
    assert [plan.layout_for_step(step, 30) for step in range(30)] == [0] * 9 + [1] * 15 + [0] * 6
    assert plan.layout_for_step(0, 1) == 0
    monkeypatch.setenv("DIFFUSION_ATTENTION_BACKEND", "INVALID_PROVIDER")
    assert build_attention_config(raw).default is None
    static = {"presets": raw["presets"], "layout": raw["layouts"]["mixed"]}
    assert ForwardStrategyPlan.from_strategy(AttentionConfig(**static).strategy).layout_for_step(34, 35) == 0
    with pytest.raises(ValueError, match="mutually exclusive"):
        parse_attention_config(raw, attention_backend="auto")
    monkeypatch.setenv("DIFFUSION_ATTENTION_BACKEND", "TORCH_SDPA")
    inactive = build_attention_config({"presets": raw["presets"]})
    assert inactive.strategy is None and inactive.default.backend == "TORCH_SDPA"


@pytest.mark.parametrize(
    "change",
    [
        {"default": {"backend": "TORCH_SDPA"}},
        {"per_role": {"self": {"backend": "TORCH_SDPA"}}},
        {"layout": {"default": "fa3_dense"}},
        {"schedule": None},
        {"schedule": {"coordinate": "step_index", "phases": [{"until": 10, "layout": "dense"}]}},
        {
            "schedule": {
                "coordinate": "step_index",
                "phases": [{"until": True, "layout": "dense"}, {"until": None, "layout": "mixed"}],
            }
        },
        {"schedule": {"coordinate": "step_fraction", "phases": [{"until": float("nan"), "layout": "dense"}]}},
        {"schedule": {"coordinate": "step_fraction", "phases": [{"until": 0.5, "layout": "dense"}]}},
        {"schedule": {"coordinate": "step_index", "phases": [{"until": None, "layout": "unknown"}]}},
    ],
)
def test_invalid_configuration(change):
    with pytest.raises(ValueError):
        AttentionConfig(**{**recipe(), **change})


@pytest.mark.parametrize(
    "rules,match",
    [
        ([{"attention_role": "typo", "use": "fa4_subblock"}], "Unknown attention role"),
        ([{"layers": [36], "attention_role": "cosmos3.gen", "use": "fa4_subblock"}], "Unknown layers"),
        (
            [
                {"layers": [0], "attention_role": "cosmos3.gen", "use": "fa4_subblock"},
                {"layers": [0], "attention_role": "cosmos3.gen", "use": "fa3_dense"},
            ],
            "Overlapping",
        ),
        ([{"attention_role": "cosmos3.und", "use": "fa4_subblock"}], "invariant"),
    ],
)
def test_inventory_rejects_ambiguous_or_incompatible_assignments(rules, match):
    raw = recipe()
    raw["layouts"]["mixed"] = {"default": "fa3_dense", "overrides": rules}
    with pytest.raises(ValueError, match=match):
        AttentionConfig(**raw).strategy.validate_inventory(inventory())


def test_request_progress_isolation_and_missing_metadata():
    strategy = AttentionConfig(**recipe()).strategy
    plan = ForwardStrategyPlan.from_strategy(strategy)
    first = ForwardContext(denoise_step_idx=10, total_denoise_steps=35)
    with override_forward_context(first):
        assert plan.current_layout() == 1
        # A nested request must not inherit the first request's progress or schedule.
        with pytest.raises(RuntimeError, match="aborted"):
            with override_forward_context(ForwardContext(denoise_step_idx=0, total_denoise_steps=4)):
                assert plan.current_layout() == 0
                raise RuntimeError("aborted")
        assert plan.current_layout() == 1
        first.total_denoise_steps = 11
        assert plan.current_layout() == 1
    with override_forward_context(ForwardContext()):
        with pytest.raises(ValueError, match="progress"):
            plan.current_layout()


@pytest.mark.parametrize(
    "options",
    [
        {"step_execution": True},
        {"enable_cpu_offload": True},
        {"enable_layerwise_offload": True},
        {"cache_backend": "cache_dit"},
        {"enable_distributed_layerwise_offload": True},
        {"num_gpus": 2},
    ],
)
def test_runtime_scope_rejection(options):
    kwargs = {
        "model": "unused",
        "model_class_name": "Cosmos3OmniDiffusersPipeline",
        "diffusion_attention_config": recipe(),
        "diffusion_compile_granularity": "full",
        **options,
    }
    with pytest.raises((ValueError, AssertionError)):
        OmniDiffusionConfig(**kwargs)


def test_eager_strategy_does_not_require_full_compilation():
    config = OmniDiffusionConfig(
        model="unused",
        diffusion_attention_config=recipe(),
        enforce_eager=True,
        diffusion_compile_granularity="regional",
    )
    validate_strategy_runtime(config)


@pytest.mark.parametrize("enforce_eager", [False, True])
@pytest.mark.parametrize("offload_option", ["enable_cpu_offload", "enable_layerwise_offload"])
def test_offload_allows_regional_or_eager_strategy(enforce_eager, offload_option):
    config = OmniDiffusionConfig(
        model="unused",
        diffusion_attention_config=recipe(),
        diffusion_compile_granularity="regional",
        enforce_eager=enforce_eager,
        **{offload_option: True},
    )
    validate_strategy_runtime(config)


def test_model_identity_can_be_resolved_after_configuration():
    config = OmniDiffusionConfig(
        model="unused", diffusion_attention_config=recipe(), diffusion_compile_granularity="full"
    )
    validate_strategy_runtime(config)
    with pytest.raises(ValueError, match="components contract"):
        validate_pipeline_attention_strategy(SimpleNamespace(), config)


def test_three_immutable_layouts_compile_complete_forwards_and_reuse_weights(monkeypatch):
    from vllm_omni.diffusion.attention import layer as layer_mod
    from vllm_omni.diffusion.attention.backends.sdpa import SDPABackend
    from vllm_omni.diffusion.attention.parallel.base import NoParallelAttention
    from vllm_omni.diffusion.config import set_current_diffusion_config

    # Distinct test-only specs stand in for three numerical methods. Production
    # recipes keep FLASH_ATTN; scheduling and compilation here use real code.
    raw = {
        "presets": {
            name: {"backend": provider}
            for name, provider in (("a", "FLASH_ATTN"), ("b", "TORCH_SDPA"), ("c", "CUDNN_ATTN"))
        },
        "layouts": {name: {"default": name} for name in ("a", "b", "c")},
        "schedule": {
            "coordinate": "step_index",
            "phases": [{"until": 10, "layout": "a"}, {"until": 20, "layout": "b"}, {"until": None, "layout": "c"}],
        },
    }
    config = SimpleNamespace(
        diffusion_attention_config=AttentionConfig(**raw),
        parallel_config=SimpleNamespace(ring_degree=1),
        diffusion_kv_cache_dtype=None,
    )
    monkeypatch.setattr(layer_mod, "build_parallel_attention_strategy", lambda **kw: NoParallelAttention())
    monkeypatch.setattr(
        layer_mod, "get_attn_backend_for_role", lambda **kw: (SDPABackend, kw["attention_config"].default)
    )
    monkeypatch.setattr(
        layer_mod.Attention,
        "forward",
        lambda self, q, k, v, metadata=None: (
            q + {"FLASH_ATTN": 1, "TORCH_SDPA": 2, "CUDNN_ATTN": 3}[self.attn_spec.backend]
        ),
    )

    class Block(torch.nn.Module):
        def __init__(self, index):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.ones(()))
            self.attn = layer_mod.build_attention(
                num_heads=2,
                head_size=4,
                causal=False,
                softmax_scale=0.5,
                prefix=f"blocks.{index}",
                role="example.self",
                operation=AttentionOperation(f"op.{index}", index, "example.self", None, component="transformer"),
            )

        def forward(self, q, layout):
            return self.attn(q, q, q, attention_layout=layout) * self.weight

    class Model(torch.nn.Module):
        attention_strategy_support = StrategyModelSupport()

        def __init__(self):
            super().__init__()
            self.blocks = torch.nn.ModuleList(Block(i) for i in range(12))
            finalize_attention_strategy(self)

        def forward(self, q, **kwargs):
            return self._attention_strategy_runner(q, **kwargs)

        def forward_with_attention_layout(self, q, *, attention_layout=None):
            for block in self.blocks:
                q = block(q, attention_layout)
            return q

    with set_current_diffusion_config(config):
        model = Model()
    identities = tuple(id(p) for p in model.parameters())
    runner = model._attention_strategy_runner
    changing = model.blocks[0].attn
    q = torch.zeros(1, 5, 2, 4)
    with pytest.raises(ValueError, match="explicit layout"):
        changing(q, q, q)
    for invalid in (-1, 3, True):
        with pytest.raises(ValueError, match="Invalid attention layout"):
            changing.for_layout(invalid)
    with pytest.raises(FrozenInstanceError):
        runner.plan.coordinate = "step_fraction"
    # Mutating the configuration after startup cannot edit the deployed schedule.
    config.diffusion_attention_config.strategy.phases = ((None, "a"),)
    graphs = []

    def backend(graph, inputs):
        graphs.append(graph)
        return graph.forward

    torch._dynamo.reset()
    runner.compile(backend=backend, dynamic=True)
    for repetition in range(2):
        for length in (5, 7):
            for step in (0, 9, 10, 19, 20, 34, 0):
                with override_forward_context(ForwardContext(denoise_step_idx=step, total_denoise_steps=35)):
                    q = torch.zeros(1, length, 2, 4)
                    for _ in range(2):  # CFG evaluations share the same layout.
                        torch.testing.assert_close(model(q), q + 12 * (1 + (step >= 10) + (step >= 20)))
        if repetition == 0:
            warmed = len(graphs)
        else:
            assert len(graphs) == warmed
    # Three full 12-block graphs, not three graphs per block.
    assert len(graphs) == 3
    assert tuple(id(p) for p in model.parameters()) == identities
    with pytest.raises(ValueError, match="deployment-owned"):
        model(q, attention_layout=0)
    with pytest.raises(ValueError, match="already compiled"):
        runner.compile()
    torch._dynamo.reset()


def test_role_only_lookup_cannot_substitute_strategy_assignment():
    config = AttentionConfig(**recipe())
    # A role-only caller cannot silently substitute the deployment default.
    with pytest.raises(ValueError, match="operation/layout"):
        config.resolve_with_source("cosmos3.gen", "self")


def test_unused_layout_does_not_prepare_or_constrain_operations():
    raw = recipe()
    raw["layouts"]["unused"] = {"default": "fa4_subblock"}
    raw["schedule"] = {
        "coordinate": "step_index",
        "phases": [{"until": None, "layout": "dense"}],
    }
    strategy = AttentionConfig(**raw).strategy
    strategy.validate_inventory(inventory())
    assert tuple(strategy.layouts) == ("dense",)
    assert ForwardStrategyPlan.from_strategy(strategy).layout_for_step(34, 35) == 0
    assert all(len(strategy.assignments(op)) == 1 for op in inventory())


def test_invariance_is_declared_by_the_operation():
    strategy = AttentionConfig(**recipe()).strategy
    with pytest.raises(ValueError, match="invariant"):
        strategy.assignments(
            AttentionOperation("cached.0", 0, "cosmos3.gen", "self", invariant=True, component="transformer")
        )
    # The resolver does not attach special semantics to model-specific role names.
    assert (
        len(strategy.assignments(AttentionOperation("uncached.0", 0, "cosmos3.gen", "self", component="transformer")))
        == 2
    )


@pytest.mark.parametrize("start_step", [0, 2, 5, 8])
def test_dense_to_sparse_schedule_boundaries_and_repeated_evaluations(start_step):
    raw = recipe()
    phases = [] if start_step == 0 else [{"until": start_step, "layout": "dense"}]
    phases.append({"until": None, "layout": "mixed"})
    raw["schedule"] = {"coordinate": "step_index", "phases": phases}
    plan = ForwardStrategyPlan.from_strategy(AttentionConfig(**raw).strategy)
    for step in (0, 1, 2, 3, 4, 0):
        with override_forward_context(ForwardContext(denoise_step_idx=step, total_denoise_steps=5)):
            for _ in range(2):  # CFG and solver evaluations cannot advance the schedule.
                assert plan.layout_names[plan.current_layout()] == ("mixed" if step >= start_step else "dense")


@pytest.mark.parametrize("step,total", [(None, None), (None, 5), (0, None), (-1, 5), (5, 5), (0, 0), (True, 5)])
def test_scheduled_layout_requires_valid_progress_even_without_a_transition(step, total):
    raw = recipe()
    raw["schedule"]["phases"] = [{"until": None, "layout": "mixed"}]
    plan = ForwardStrategyPlan.from_strategy(AttentionConfig(**raw).strategy)
    with override_forward_context(ForwardContext(denoise_step_idx=step, total_denoise_steps=total)):
        with pytest.raises(ValueError, match="progress"):
            plan.current_layout()


def test_static_layout_needs_no_denoising_progress():
    plan = ForwardStrategyPlan.from_strategy(
        AttentionConfig(presets={"base": "TORCH_SDPA"}, layout={"default": "base"}).strategy
    )
    with override_forward_context(ForwardContext()):
        assert plan.current_layout() == 0


@pytest.mark.parametrize("total,expected", [(1, [0]), (2, [0, 1]), (3, [0, 1, 2]), (10, [0] * 3 + [1] * 3 + [2] * 4)])
def test_fractional_schedule_rounding_and_collapsed_phases(total, expected):
    plan = ForwardStrategyPlan(("a", "b", "c"), "step_fraction", ((0.3, 0), (0.6, 1), (1.0, 2)))
    assert [plan.layout_for_step(step, total) for step in range(total)] == expected


def test_fractional_schedule_uses_current_total_without_request_state():
    plan = ForwardStrategyPlan(("a", "b"), "step_fraction", ((0.5, 0), (1.0, 1)))
    context = ForwardContext(denoise_step_idx=4, total_denoise_steps=10)
    with override_forward_context(context):
        assert plan.current_layout() == 0
        context.total_denoise_steps = 8
        assert plan.current_layout() == 1
        context.total_denoise_steps = 10
        assert plan.current_layout() == 0
