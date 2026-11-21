# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""CPU contracts for the narrow MiniMax-H3 MindIE-SD pipeline adapter."""

import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
import torch

from vllm_omni.diffusion.attention.backends.abstract import AttentionMetadata
from vllm_omni.diffusion.distributed.group_coordinator import SequenceParallelResources

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.parallel, pytest.mark.cpu]

ROOT = Path(__file__).resolve().parents[3]
MODEL_DIR = ROOT / "vllm_omni/diffusion/models/minimax_h3"
HELPER_NAME = "vllm_omni.diffusion.models.minimax_h3.attention.mindiesd_parallel"


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
    for name, exports in (
        ("vllm_omni.diffusion.forward_context", {"get_ulysses_mode": lambda **_: "strict"}),
        ("vllm_omni.platforms", {"current_omni_platform": platform}),
        ("mindiesd.parallel", {"ParallelState": new_state, "distributed_sparse_attention": distributed_attention}),
    ):
        stub = ModuleType(name)
        stub.__dict__.update(exports)
        monkeypatch.setitem(sys.modules, name, stub)
    module = load_module(monkeypatch, HELPER_NAME, MODEL_DIR / "attention/mindiesd_parallel.py")
    monkeypatch.setattr(module.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(module.dist, "get_world_size", lambda group: 2)

    group = object()
    coordinator = SimpleNamespace(
        ring_world_size=1,
        allgather_world_size=1,
        resources=SequenceParallelResources(),
    )
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
            sparsity=0.8,
            parallel=module.MindIEParallelConfig(),
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
        library=sys.modules["mindiesd.parallel"],
        platform=platform,
    )


def run(adapter, *, query=None):
    q = adapter.query if query is None else query
    return adapter.module.try_distributed_sparse_attention(adapter.attention, q, q, q, adapter.metadata)


def test_pipeline_call_is_explicit_and_state_is_reused(adapter):
    torch.testing.assert_close(run(adapter), adapter.query + 1)
    run(adapter)
    assert len(adapter.states) == 1
    resource = adapter.coordinator.resources.get_or_create(
        ("mindiesd", adapter.group, adapter.query.device), lambda: None
    )
    assert resource.completed_calls == 2
    state = adapter.states[0]
    assert state.overlap_comm is True
    assert state.optimize_attention_pipeline is True
    assert state.bsa_backend == "cann"
    q, k, v, spans, kwargs = adapter.calls[0]
    assert q is k is v is adapter.query
    assert spans == [{"start": 6, "latent_shape": [2, 2, 2]}]
    assert kwargs == dict(
        group=adapter.group,
        used_len=14,
        state=state,
        precision="bf16",
        sparse_type="rf_v3",
        pre_rearranged=False,
        scale=0.125,
        sparsity=0.8,
    )


def test_pre_rearranged_is_forwarded(adapter):
    adapter.metadata.video_layout = SimpleNamespace(pre_rearranged=True)
    run(adapter)
    assert adapter.calls[0][-1]["pre_rearranged"] is True


def test_live_resource_rejects_overlap_mode_change(adapter):
    run(adapter)
    adapter.impl.rainfusion.parallel = adapter.module.MindIEParallelConfig(overlap_comm=False)
    with pytest.raises(RuntimeError, match="configuration"):
        run(adapter)
    assert len(adapter.calls) == 1


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


@pytest.mark.parametrize(
    "shape,dtype",
    [
        ((2, 8, 4, 8), torch.bfloat16),
        ((1, 8, 3, 8), torch.bfloat16),
        ((1, 8, 4, 8), torch.float16),
    ],
)
def test_unsupported_tensor_contract_declines(adapter, shape, dtype):
    assert run(adapter, query=torch.zeros(shape, dtype=dtype)) is None
    assert not adapter.calls


@pytest.mark.parametrize("precision,sparse_type", [("mix", "rf_v3"), ("bf16", "rf_v2")])
def test_enabled_pipeline_rejects_non_target_attention_mode(adapter, precision, sparse_type):
    adapter.impl.rainfusion.precision = precision
    adapter.impl.rainfusion.sparse_type = sparse_type
    with pytest.raises(ValueError, match="precision='bf16'.*sparse_type='rf_v3'"):
        run(adapter)
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


def test_teardown_drains_before_releasing_state_and_is_idempotent(adapter, monkeypatch):
    from contextlib import nullcontext

    run(adapter)
    resource = adapter.coordinator.resources.get_or_create(
        ("mindiesd", adapter.group, adapter.query.device), lambda: None
    )
    state, drained = resource.state, []
    monkeypatch.setattr(
        adapter.module.torch, "npu", SimpleNamespace(device=lambda device: nullcontext()), raising=False
    )
    monkeypatch.setattr(adapter.platform, "synchronize", lambda: drained.append(resource.state), raising=False)
    adapter.coordinator.resources.close()
    assert drained == [state] and resource.state is None
    adapter.coordinator.resources.close()
    assert drained == [state]


@pytest.mark.parametrize(
    "target,name,value",
    [
        ("strategy", "_ulysses_pg", None),
        ("strategy", "_scatter_idx", 1),
        ("strategy", "_gather_idx", 2),
        ("rainfusion", "sparse_type", "rf_v2"),
    ],
)
def test_reorder_preflight_rejects_incompatible_executor(adapter, monkeypatch, target, name, value):
    setattr(adapter.strategy if target == "strategy" else adapter.impl.rainfusion, name, value)
    helper = load_module(
        monkeypatch,
        "vllm_omni.diffusion.models.minimax_h3.attention.rf_reorder_hoist",
        MODEL_DIR / "attention/rf_reorder_hoist.py",
    )
    layout = SimpleNamespace(pre_rearranged=False)
    branch = SimpleNamespace(static_kwargs={"video_token_layout": layout})
    model = SimpleNamespace(blocks=[SimpleNamespace(attn=SimpleNamespace(attention=adapter.attention))])
    with pytest.raises(ValueError, match="BF16 pure strict Ulysses"):
        helper.maybe_hoist_branch(branch, model)
    assert not layout.pre_rearranged and not adapter.calls and not adapter.states


def test_rf_relabel_preserves_modalities_and_padding(adapter, monkeypatch):
    name = "vllm_omni.diffusion.models.minimax_h3.attention.rf_reorder_hoist"
    helper = load_module(monkeypatch, name, MODEL_DIR / "attention/rf_reorder_hoist.py")
    branch = SimpleNamespace(
        seq_len=10,
        device=torch.device("cpu"),
        img_pos=torch.tensor([2, 3, 6, 7]),
        audio_pos=torch.tensor([4, 5]),
        text_pos_dev=torch.tensor([0, 1]),
        token_tags_dev=torch.arange(10),
        img_position_ids_dev=torch.arange(30).reshape(10, 3),
        static_kwargs={},
    )
    permutation = torch.tensor([0, 1, 4, 5, 6, 7, 2, 3])
    helper.relabel_branch(branch, permutation, torch.argsort(permutation))
    padded = torch.cat((permutation, torch.tensor([8, 9])))
    assert padded.index_select(0, branch.img_pos_dev).tolist() == [2, 3, 6, 7]
    assert padded.index_select(0, branch.audio_pos_dev).tolist() == [4, 5]
    assert branch.token_tags_dev.tolist() == padded.tolist()
    assert torch.equal(branch.static_kwargs["img_position_ids"][0], torch.arange(30).reshape(10, 3)[padded])
