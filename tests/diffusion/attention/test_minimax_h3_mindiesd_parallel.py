# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""MiniMax-H3 sparse routes, caller-owned SDK lifecycle and RF label preservation."""

import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
import torch

from vllm_omni import platforms
from vllm_omni.diffusion import forward_context
from vllm_omni.diffusion.attention.backends import rainfusion_attn
from vllm_omni.diffusion.attention.backends.abstract import AttentionMetadata, VideoTokenLayout
from vllm_omni.diffusion.attention.parallel.ulysses_allgather import UlyssesAllGatherKVParallelAttention
from vllm_omni.diffusion.data import MindIEParallelConfig, OmniDiffusionConfig
from vllm_omni.diffusion.distributed.group_coordinator import SequenceParallelResources
from vllm_omni.diffusion.models.minimax_h3.denoise_loop import MiniMaxH3DenoiseBranch
from vllm_omni.diffusion.models.minimax_h3.packed_sequence import minimax_h3_packed_sequence

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.parallel, pytest.mark.cpu]

MODEL_DIR = Path(__file__).resolve().parents[3] / "vllm_omni/diffusion/models/minimax_h3"
HELPER_NAME = "vllm_omni.diffusion.models.minimax_h3.attention.rainfusion_sdk_dispatch"


def load_module(monkeypatch, name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, module)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def adapter(monkeypatch):
    calls, states = [], []

    def new_state(**kwargs):
        state = SimpleNamespace(**kwargs)
        states.append(state)
        return state

    def distributed_attention(q, k, v, spans, **kwargs):
        calls.append((q, k, v, spans, kwargs))
        return q + 1

    platform = SimpleNamespace(is_npu=lambda: True)
    monkeypatch.setattr(platforms, "current_omni_platform", platform)
    monkeypatch.setattr(forward_context, "get_ulysses_mode", lambda **_: "strict")
    stub = ModuleType("mindiesd.parallel")
    setattr(stub, "ParallelState", new_state)
    setattr(stub, "distributed_sparse_attention", distributed_attention)
    monkeypatch.setitem(sys.modules, stub.__name__, stub)
    module = load_module(monkeypatch, HELPER_NAME, MODEL_DIR / "attention/rainfusion_sdk_dispatch.py")
    monkeypatch.setattr(module.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(module.dist, "get_world_size", lambda group: 2)

    group = object()
    coordinator = SimpleNamespace(ring_world_size=1, allgather_world_size=1, resources=SequenceParallelResources())
    strategy = SimpleNamespace(
        name="ulysses",
        _ulysses_pg=group,
        _sp_group=coordinator,
        _scatter_idx=2,
        _gather_idx=1,
        _ulysses_a2a_permute=False,
    )
    plan = SimpleNamespace(used_len=14, video_spans=None, prefix_len=6, latent_shape=(2, 2, 2))
    impl = SimpleNamespace(
        rainfusion=SimpleNamespace(
            precision="bf16",
            sparse_type="rf_v3",
            bsa_backend="eagle",
            sparsity=0.8,
            parallel=MindIEParallelConfig(),
        ),
        softmax_scale=0.125,
        _resolve_plan=lambda metadata: plan,
    )

    attention = SimpleNamespace(
        causal=False,
        qkv_layout="BSND",
        _has_custom_attention=False,
        is_paged_kv_active=lambda: False,
        effective_attention=lambda: (impl, SimpleNamespace(get_name=lambda: "RAINFUSION_ATTN"), None),
        paged_kv_cache_role=None,
        _get_active_parallel_strategy=lambda: strategy,
        _with_kv_cache_dtype=lambda metadata: metadata,
    )
    query = torch.zeros((1, 8, 4, 8), dtype=torch.bfloat16)
    return SimpleNamespace(
        module=module,
        attention=attention,
        metadata=AttentionMetadata(),
        query=query,
        calls=calls,
        states=states,
        strategy=strategy,
        coordinator=coordinator,
        group=group,
        plan=plan,
        impl=impl,
        library=stub,
        platform=platform,
    )


def run(adapter, *, query=None):
    q = adapter.query if query is None else query
    return adapter.module.try_distributed_sparse_attention(adapter.attention, q, q, q, adapter.metadata)


@pytest.mark.parametrize("enabled", [False, True])
def test_pipeline_call_is_explicit_and_state_is_reused(adapter, enabled):
    adapter.impl.rainfusion.parallel = MindIEParallelConfig(
        overlap_comm=enabled,
        rainfusion_reorder_hoist=enabled,
        optimize_attention_pipeline=enabled,
        quantized_comm=not enabled,
    )
    assert torch.equal(run(adapter), adapter.query + 1)
    run(adapter)
    assert len(adapter.states) == 1
    resource = adapter.coordinator.resources.get_or_create(
        ("mindiesd", adapter.group, None, adapter.query.device, "rf_v3", "bf16", "eagle"), lambda: None
    )
    assert resource.completed_calls == 2
    assert not {"group", "cp_group"}.intersection(vars(resource))
    state = adapter.states[0]
    assert state.group is adapter.group
    assert state.quantized_comm is (not enabled)
    assert state.overlap_comm is enabled
    assert state.optimize_attention_pipeline is enabled
    assert state.bsa_backend == "eagle"
    q, k, v, spans, kwargs = adapter.calls[0]
    assert q is k is v is adapter.query
    assert spans == [{"start": 6, "latent_shape": [2, 2, 2]}]
    assert kwargs == dict(
        used_len=14, state=state, precision="bf16", sparse_type="rf_v3", pre_rearranged=False, scale=0.125, sparsity=0.8
    )


@pytest.mark.parametrize(
    "sparse_type,precision,backend,dtype",
    [
        ("rf_v2", "bf16", "cann", torch.bfloat16),
        ("rf_v2", "fp16", "cann", torch.float16),
        ("rf_v3", "bf16", "eagle", torch.bfloat16),
        ("rf_v3", "fp16", "eagle", torch.float16),
        ("rf_v3", "mix", "eagle", torch.bfloat16),
        ("rf_v3", "mxfp8", "cann", torch.bfloat16),
    ],
)
def test_sparse_routes_reach_the_public_sdk(adapter, sparse_type, precision, backend, dtype):
    adapter.impl.rainfusion.sparse_type = sparse_type
    adapter.impl.rainfusion.precision = precision
    adapter.impl.rainfusion.bsa_backend = backend
    q = torch.zeros((1, 8, 4, 128), dtype=dtype)
    assert torch.equal(run(adapter, query=q), q + 1)
    assert adapter.calls[0][-1]["precision"] == precision
    assert adapter.calls[0][-1]["sparse_type"] == sparse_type
    assert adapter.calls[0][-1]["pre_rearranged"] is False
    assert adapter.states[0].bsa_backend == backend


def use_cp(adapter, ulysses_size):
    adapter.coordinator.allgather_world_size = 2
    adapter.coordinator.allgather_group = object()
    adapter.coordinator.ulysses_world_size = ulysses_size
    adapter.coordinator.ulysses_group = adapter.group
    combined = object.__new__(UlyssesAllGatherKVParallelAttention)
    combined._sp_group = adapter.coordinator
    combined._ulysses = adapter.strategy
    adapter.strategy = combined
    adapter.attention._get_active_parallel_strategy = lambda: adapter.strategy
    adapter.module.dist.get_world_size = (
        lambda group=None: ulysses_size * 2
        if group is None
        else (2 if group is adapter.coordinator.allgather_group else ulysses_size)
    )


@pytest.mark.parametrize("ulysses_size,precision,backend", [(4, "mix", "eagle"), (8, "mxfp8", "cann")])
def test_cp_uses_the_existing_orthogonal_group_and_exact_sdk_keywords(adapter, ulysses_size, precision, backend):
    adapter.impl.rainfusion.precision = precision
    adapter.impl.rainfusion.bsa_backend = backend
    adapter.metadata.video_layout = VideoTokenLayout(pre_rearranged=True)
    use_cp(adapter, ulysses_size)
    adapter.plan.used_len = 251
    adapter.plan.video_spans = [{"start": 123, "latent_shape": [2, 8, 8]}]
    dtype = torch.float16 if precision == "fp16" else torch.bfloat16
    q = torch.zeros((1, 32, 56, 128), dtype=dtype)
    assert torch.equal(run(adapter, query=q), q + 1)
    kwargs = adapter.calls[0][-1]
    assert adapter.states[0].group is adapter.group
    assert adapter.states[0].cp_group is adapter.coordinator.allgather_group
    assert not {"group", "cp_group", "kv_gather_group", "comm_dtype", "comm_quant_scope"}.intersection(kwargs)
    assert kwargs["pre_rearranged"] is True


def test_explicit_fp16_attention_restores_the_bf16_producer_dtype(adapter):
    adapter.impl.rainfusion.precision = "fp16"
    adapter.impl.rainfusion.bsa_backend = "eagle"
    output = run(adapter)
    assert output.dtype == torch.bfloat16
    assert all(value.dtype == torch.float16 for value in adapter.calls[0][:3])


def test_rf_hoist_preserves_modalities_and_declines_an_unresolved_plan(adapter, monkeypatch):
    helper = load_module(
        monkeypatch,
        HELPER_NAME.replace("rainfusion_sdk_dispatch", "rf_reorder_hoist"),
        MODEL_DIR / "attention/rf_reorder_hoist.py",
    )
    plan = torch.tensor([0, 1, 4, 5, 2, 3, 10, 11, 12, 13, 6, 7, 8, 9])
    root = ModuleType("mindiesd")
    setattr(
        root, "build_rf_attention_plan", lambda **kwargs: SimpleNamespace(permutation=plan, inverse=torch.argsort(plan))
    )
    monkeypatch.setitem(sys.modules, "mindiesd", root)
    packed = minimax_h3_packed_sequence(
        text_len=4, latent_t=2, latent_h=4, latent_w=4, audio_t=1, include_keyframe_cond=False, seq_len=16
    )
    packed["update_mask"][:2] = False
    packed["audio_update_mask"] = torch.tensor([False, True])
    branch = MiniMaxH3DenoiseBranch(
        packed=packed,
        text_embeddings=torch.zeros(4, 2),
        token_tags=torch.arange(16),
        device=torch.device("cpu"),
    )
    branch.img_position_ids_dev = torch.arange(48).reshape(16, 3)
    position_names = ("video_target_pos_dev", "video_cond_pos_dev", "audio_target_pos_dev", "audio_cond_pos_dev")
    positions = [getattr(branch, name).clone() for name in position_names]
    groups = branch.timestep_group_dev.clone()
    model = SimpleNamespace(blocks=[SimpleNamespace(attn=SimpleNamespace(attention=adapter.attention))])
    use_cp(adapter, 4)
    config = adapter.impl.rainfusion
    config.enabled, config.skip_layers, config.start_step, config.end_step = True, frozenset(), 1, 0
    adapter.impl.layer_idx = 0
    adapter.impl._resolve_plan = lambda metadata: rainfusion_attn.RainFusionAttentionImpl._resolve_plan(
        adapter.impl, metadata
    )
    adapter.impl._resolve_layout_plan = lambda _: None
    assert not helper.maybe_hoist_branch(branch, model)
    assert not branch.static_kwargs["video_token_layout"].pre_rearranged
    adapter.impl._resolve_layout_plan = lambda _: adapter.plan
    assert helper.maybe_hoist_branch(branch, model)
    layout = branch.static_kwargs["video_token_layout"]
    packed = branch.static_kwargs["packed_seq_params"]
    assert isinstance(layout, VideoTokenLayout) and layout.pre_rearranged
    assert isinstance(packed, dict) and "vsa_prefix_segments" not in packed
    padded = torch.cat((plan, torch.tensor([14, 15])))
    for name, original in zip(position_names, positions, strict=True):
        assert torch.equal(getattr(branch, name), torch.argsort(padded)[original])
    assert torch.equal(branch.timestep_group_dev, groups[padded])
    assert padded.index_select(0, branch.img_pos_dev).tolist() == list(range(6, 14))
    assert padded.index_select(0, branch.audio_pos_dev).tolist() == [4, 5]
    assert padded.index_select(0, branch.text_pos_dev).tolist() == list(range(4))
    assert torch.equal(branch.token_tags_dev, padded)
    assert torch.equal(branch.static_kwargs["img_position_ids"][0], torch.arange(48).reshape(16, 3)[padded])
    adapter.metadata.video_layout = layout
    q = torch.zeros((1, 8, 4, 128), dtype=torch.bfloat16)
    assert run(adapter, query=q) is None  # No denoise context exists when hoisting runs.
    ctx = SimpleNamespace(denoise_step_idx=0, total_denoise_steps=3)
    monkeypatch.setattr(rainfusion_attn, "is_forward_context_available", lambda: True)
    monkeypatch.setattr(rainfusion_attn, "get_forward_context", lambda: ctx)
    assert run(adapter, query=q) is None
    ctx.denoise_step_idx = 1
    assert torch.equal(run(adapter, query=q), q + 1)
    assert adapter.calls[0][-1]["pre_rearranged"] is True


@pytest.mark.parametrize(
    "change",
    [
        lambda a: setattr(a.strategy, "name", "none"),
        lambda a: setattr(a.coordinator, "ring_world_size", 2),
        lambda a: setattr(a.coordinator, "allgather_world_size", 2),
        lambda a: setattr(a.strategy, "_scatter_idx", 1),
        lambda a: setattr(a.strategy, "_ulysses_pg", None),
        lambda a: setattr(a.module, "get_ulysses_mode", lambda **_: "advanced_uaa"),
        lambda a: setattr(a.attention, "causal", True),
        lambda a: setattr(a.attention, "_has_custom_attention", True),
        lambda a: setattr(a.attention, "is_paged_kv_active", lambda: True),
        lambda a: setattr(a.attention, "paged_kv_cache_role", "read"),
        lambda a: setattr(a.impl, "_resolve_plan", lambda metadata: None),
        lambda a: setattr(a.plan, "used_len", 17),
        lambda a: setattr(a.strategy, "_ulysses_a2a_permute", True),
        lambda a: setattr(a.metadata, "attn_mask", a.query),
        lambda a: setattr(a.metadata, "full_attn_spans", ((0, 1),)),
        lambda a: setattr(a.impl.rainfusion, "parallel", None),
        lambda a: setattr(a.metadata, "joint_key", a.query),
        lambda a: a.metadata.extra.update(minimax_h3_reference_kv_post_parallel=(object(), 1, False)),
        lambda a: a.metadata.extra.update(gate_compress=a.query),
        lambda a: a.metadata.extra.update(vdn_window=object()),
    ],
)
def test_unsupported_calls_decline_before_state_or_collective(adapter, change):
    change(adapter)
    assert run(adapter) is None
    assert not adapter.calls and not adapter.states


def test_collective_failure_poisons_resource_without_replay(adapter):
    def fail(*args, **kwargs):
        raise RuntimeError("submitted collective failed")

    adapter.library.distributed_sparse_attention = fail
    with pytest.raises(RuntimeError, match="collective failed"):
        run(adapter)
    adapter.library.distributed_sparse_attention = lambda *args, **kwargs: pytest.fail("poisoned resource replayed")
    with pytest.raises(RuntimeError, match="closed or failed"):
        run(adapter)


def test_teardown_drains_before_releasing_state_and_is_idempotent(adapter, monkeypatch, caplog):
    from contextlib import nullcontext

    run(adapter)
    resource = adapter.coordinator.resources.get_or_create(
        ("mindiesd", adapter.group, None, adapter.query.device, "rf_v3", "bf16", "eagle"), lambda: None
    )
    state, drained = resource.state, []
    caplog.set_level("DEBUG", logger=resource.__class__.__module__)
    monkeypatch.setattr(sys.modules[resource.__class__.__module__].logger, "handlers", [caplog.handler])
    monkeypatch.setattr(
        adapter.module.torch, "npu", SimpleNamespace(device=lambda device: nullcontext()), raising=False
    )
    monkeypatch.setattr(adapter.platform, "synchronize", lambda: drained.append(resource.state), raising=False)
    adapter.coordinator.resources.close()
    assert drained == [state] and resource.state is None
    assert caplog.records[-1].levelname == "DEBUG" and caplog.records[-1].completed_calls == 1
    adapter.coordinator.resources.close()
    assert drained == [state]
    with pytest.raises(RuntimeError, match="closed"):
        run(adapter)


def test_cp_rejects_disabled_rf_hoisting_at_config_build():
    sparse = {
        "backend": "RAINFUSION_ATTN",
        "block_sparse": {"sparsity": 0.8, "parallel": {"rainfusion_reorder_hoist": False}},
    }
    with pytest.raises(ValueError, match="requires rainfusion_reorder_hoist=true"):
        OmniDiffusionConfig(
            parallel_config={"ulysses_degree": 8, "allgather_degree": 2}, diffusion_attention_config={"default": sparse}
        )
