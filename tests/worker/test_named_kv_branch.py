# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""CPU contracts for named KV allocation, binding and graph-executor lifecycle.

FA3 calls and capture failures are mocked here. Numerical and CUDA replay
coverage lives in test_named_kv_adapter_conformance_gpu.py.
"""

from __future__ import annotations

from contextlib import nullcontext
from math import ceil
from types import SimpleNamespace

import pytest
import torch
from pytest_mock import MockerFixture
from vllm.v1.kv_cache_interface import FullAttentionSpec

import vllm_omni.worker.named_kv.runtime as named_kv_module
from vllm_omni.model_executor.models.vibevoice.negative_qwen_adapter import Qwen2KVBranchAdapter
from vllm_omni.worker.gpu_model_runner import OmniGPUModelRunner
from vllm_omni.worker.named_kv.executor import ExecutionBuffers, NamedKVBranchExecutor
from vllm_omni.worker.named_kv.flash_attention import FlashAttentionKVBranchAdapter, UnsupportedNamedKVGraphError
from vllm_omni.worker.named_kv.ops import _named_kv_branch_attention_impl
from vllm_omni.worker.named_kv.runtime import _build_named_kv_manager
from vllm_omni.worker.named_kv.types import NamedKVAppendBatch
from vllm_omni.worker.named_kv_branch import (
    NamedCausalKVBranch,
    NamedKVBranchRequest,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _make_spec(block_size: int = 16) -> FullAttentionSpec:
    return FullAttentionSpec(
        block_size=block_size,
        num_kv_heads=2,
        head_size=16,
        dtype=torch.bfloat16,
    )


def _make_test_branch(
    *,
    num_blocks: int = 8,
    block_size: int = 16,
    max_sequence_tokens: int = 64,
    max_concurrent_requests: int = 2,
) -> NamedCausalKVBranch:
    """Build a NamedCausalKVBranch with a real KVCacheManager for CPU tests."""
    branch = object.__new__(NamedCausalKVBranch)
    branch.name = "negative"
    branch.device = torch.device("cpu")
    branch._entered = False
    branch._closed = False
    branch.block_size = block_size
    branch.max_sequence_tokens = max_sequence_tokens
    branch.max_blocks_per_request = ceil(max_sequence_tokens / block_size)
    branch.max_concurrent_requests = max_concurrent_requests
    branch._states = {}
    spec = _make_spec(block_size)
    branch._manager = _build_named_kv_manager(
        spec,
        ["layer0"],
        num_blocks,
        max_sequence_tokens,
    )
    branch._query_start_cpu = torch.tensor([0, 1], dtype=torch.int32)
    branch._query_start_gpu = branch._query_start_cpu.to(branch.device)
    return branch


def test_manager_describes_independent_layer_storage() -> None:
    spec = _make_spec(16)
    manager = _build_named_kv_manager(spec, ["a", "b"], 9, 64)
    tensors = manager.kv_cache_config.kv_cache_tensors
    # One layer-outermost tensor covers all layers in the group.
    assert len(tensors) == 1
    assert tensors[0].layers == ["a", "b"]
    assert tensors[0].layer_stride == spec.page_size_bytes * 9
    assert tensors[0].block_stride == spec.page_size_bytes
    assert tensors[0].size == 2 * spec.page_size_bytes * 9
    assert manager.block_pool.get_num_free_blocks() == 8


@pytest.mark.parametrize("failure_index", [0, 1])
@pytest.mark.parametrize("failure_kind", ["none", "exception", "mirror"])
def test_manager_failure_frees_batch_but_not_survivor(
    monkeypatch: pytest.MonkeyPatch, failure_index, failure_kind
) -> None:
    branch = _make_test_branch(num_blocks=9)
    for request_id in ("outside", "a", "b", "c"):
        branch.reset(request_id)
    branch._append_slots(["outside"])
    outside = tuple(branch._states["outside"].block_ids)
    if failure_kind == "mirror":

        class BrokenMirror:
            def __setitem__(self, key, value):
                raise ValueError("injected mirror failure")

        branch._states[("a", "b")[failure_index]].block_table = BrokenMirror()
    else:
        original = branch._manager.allocate_slots
        calls = 0

        def allocate(*args, **kwargs):
            nonlocal calls
            index = calls
            calls += 1
            if index == failure_index:
                if failure_kind == "none":
                    return None
                raise ValueError("injected allocation failure")
            return original(*args, **kwargs)

        monkeypatch.setattr(branch._manager, "allocate_slots", allocate)
    with pytest.raises((ValueError, RuntimeError)):
        branch._append_slots(["a", "b", "c"])
    assert set(branch._states) == {"outside"}
    assert tuple(branch._states["outside"].block_ids) == outside
    assert branch.get_sequence_length("outside") == 1
    assert branch.num_free_blocks == 7


def test_manager_and_immutable_mirror_refresh_only_at_boundaries(monkeypatch: pytest.MonkeyPatch) -> None:
    branch = _make_test_branch(num_blocks=9)
    branch.reset("a")
    allocate = branch._manager.allocate_slots
    get_ids = branch._manager.get_block_ids
    calls = []
    reads = []

    def record_allocate(request, **kwargs):
        calls.append(request.num_computed_tokens)
        return allocate(request, **kwargs)

    def record_ids(request_id):
        reads.append(request_id)
        return get_ids(request_id)

    monkeypatch.setattr(branch._manager, "allocate_slots", record_allocate)
    monkeypatch.setattr(branch._manager, "get_block_ids", record_ids)
    snapshots = []
    for _ in range(32):
        with branch.append_batch(["a"]) as step:
            snapshots.append(step)
    assert calls == [0, 16]
    assert reads == ["a", "a"]
    assert len(snapshots[0].block_ids[0]) == 1
    assert len(snapshots[-1].block_ids[0]) == 2
    assert branch.get_sequence_length("a") == 32
    old = snapshots[-1].block_ids
    branch.free("a")
    branch.reset("a")
    assert snapshots[-1].block_ids == old
    assert branch._states["a"].block_ids == ()
    assert branch.num_free_blocks == 8


def test_existing_request_boundary_failure_releases_whole_batch(monkeypatch: pytest.MonkeyPatch) -> None:
    branch = _make_test_branch(num_blocks=9)
    for request_id in ("a", "b"):
        branch.reset(request_id)
    for _ in range(16):
        branch._append_slots(["a", "b"])
    original = branch._manager.allocate_slots

    def fail_second(request, **kwargs):
        if request.request_id == "b":
            raise ValueError("boundary allocation failure")
        return original(request, **kwargs)

    monkeypatch.setattr(branch._manager, "allocate_slots", fail_second)
    with pytest.raises(ValueError, match="boundary allocation failure"):
        branch._append_slots(["a", "b"])
    assert not branch._states
    assert branch.num_free_blocks == 8


def test_block_mirror_written_only_at_boundaries() -> None:
    branch = _make_test_branch(num_blocks=9)
    branch.reset("a")
    writes = []

    class Mirror:
        def __setitem__(self, key, value):
            writes.append((key, value))

    branch._states["a"].block_table = Mirror()
    for _ in range(32):
        branch._append_slots(["a"])
    assert [key for key, _ in writes] == [(0, 0), (0, 1)]


def test_named_kv_branch_request_validates_deployment_contract() -> None:
    request = NamedKVBranchRequest(
        name="negative",
        memory_bytes=1024,
        layer_group=0,
        activation_margin_bytes=256,
    )
    assert request.name == "negative"
    assert request.memory_bytes == 1024

    with pytest.raises(ValueError, match="name must be non-empty"):
        NamedKVBranchRequest(name="", memory_bytes=1)
    with pytest.raises(ValueError, match="memory_bytes must be positive"):
        NamedKVBranchRequest(name="negative", memory_bytes=0)
    with pytest.raises(ValueError, match="layer_group must be non-negative"):
        NamedKVBranchRequest(name="negative", memory_bytes=1, layer_group=-1)
    with pytest.raises(ValueError, match="must be non-negative"):
        NamedKVBranchRequest(
            name="negative",
            memory_bytes=1,
            activation_margin_bytes=-1,
        )


def test_named_kv_public_reset_and_free_reject_active_context() -> None:
    branch = _make_test_branch(num_blocks=2, max_sequence_tokens=16)
    branch.reset("request")
    branch._append_slots(["request"])
    branch._entered = True

    with pytest.raises(RuntimeError, match="Cannot reset.*forward context"):
        branch.reset("request")
    with pytest.raises(RuntimeError, match="Cannot free.*forward context"):
        branch.free("request")

    assert "request" in branch._states
    assert branch.num_free_blocks == 0


def test_named_kv_internal_fault_cleanup_remains_legal_in_context() -> None:
    branch = _make_test_branch(num_blocks=2, max_sequence_tokens=16)
    branch.reset("request")
    branch._append_slots(["request"])
    branch._entered = True

    branch._free_unchecked("request")

    assert branch._states == {}
    assert branch.num_free_blocks == 1


def test_append_slots_validates_batch_before_mutating_any_request() -> None:
    branch = _make_test_branch(
        num_blocks=8,
        block_size=2,
        max_sequence_tokens=4,
    )
    branch.reset("request-a")
    branch.reset("request-b")

    # Advance request-a by two slots (crossing one block boundary).
    states, positions, slots = branch._append_slots(["request-a", "request-b"])
    assert positions == [0, 0]
    assert [state.num_tokens for state in states] == [1, 1]
    states, positions, slots = branch._append_slots(["request-a"])
    assert positions == [1]
    # Second block for request-a at the boundary, request-b untouched.
    state_a = branch._states["request-a"]
    assert len(state_a.block_ids) == 1
    states, positions, slots = branch._append_slots(["request-a"])
    assert positions == [2]
    assert len(state_a.block_ids) == 2

    # Batch validation failure must not advance the valid request.
    branch._states.pop("request-b")
    with pytest.raises(RuntimeError, match="must be reset before append"):
        branch._append_slots(["request-a", "request-b"])
    assert branch._states["request-a"].num_tokens == 3


def test_append_slots_fault_frees_whole_batch_on_bookkeeping_failure() -> None:
    # Two physical blocks: one null, one usable. request-a takes it,
    # request-b exhausts the pool and the whole logical batch is fault-freed.
    branch = _make_test_branch(
        num_blocks=2,
        block_size=1,
        max_sequence_tokens=8,
    )
    branch.reset("request-a")
    branch.reset("request-b")

    with pytest.raises(RuntimeError, match="exhausted its fixed GPU block pool"):
        branch._append_slots(["request-a", "request-b"])
    assert branch._states == {}
    assert branch.num_free_blocks == 1


def test_named_kv_fault_cleanup_does_not_mask_original_exception(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    branch = object.__new__(NamedCausalKVBranch)
    branch.name = "negative"
    errors: list[tuple[object, ...]] = []
    monkeypatch.setattr(
        branch,
        "_free_unchecked",
        lambda _request_id: (_ for _ in ()).throw(RuntimeError("secondary cleanup failure")),
    )
    monkeypatch.setattr(
        "vllm_omni.worker.named_kv.runtime.logger.exception",
        lambda *args, **_kwargs: errors.append(args),
    )

    try:
        raise ValueError("original forward failure")
    except ValueError:
        branch._cleanup_after_fault("request")
        with pytest.raises(ValueError, match="original forward failure"):
            raise

    assert len(errors) == 1
    assert errors[0][1] == "request"


def _fixed_concurrency_runner(*, positive_blocks: int):
    class FakeFullAttentionSpec:
        block_size = 16
        page_size_bytes = 16

    config = SimpleNamespace(
        scheduler_config=SimpleNamespace(max_num_seqs=2),
        cache_config=SimpleNamespace(
            enable_prefix_caching=False,
            cache_dtype="auto",
        ),
        parallel_config=SimpleNamespace(
            pipeline_parallel_size=1,
            prefill_context_parallel_size=1,
            decode_context_parallel_size=1,
            use_ubatching=False,
        ),
        model_config=SimpleNamespace(
            enable_sleep_mode=False,
            enforce_eager=True,
            max_model_len=64,
        ),
        speculative_config=None,
        kv_transfer_config=None,
        compilation_config=SimpleNamespace(
            static_forward_context={"layer": object()},
        ),
    )
    spec = FakeFullAttentionSpec()
    runner = SimpleNamespace(
        vllm_config=config,
        device="cpu",
        kv_cache_config=SimpleNamespace(
            kv_cache_groups=[SimpleNamespace(kv_cache_spec=spec)],
            num_blocks=positive_blocks,
        ),
        attn_groups=[
            [
                SimpleNamespace(
                    backend=object(),
                    layer_names=["layer"],
                )
            ]
        ],
        _kernel_block_sizes=[16],
    )
    return runner, FakeFullAttentionSpec


@pytest.mark.parametrize("physical_blocks,remainder,accepted", [(8, 0, False), (9, 0, True), (9, 15, True)])
def test_negative_capacity_null_block_and_rounding(
    monkeypatch: pytest.MonkeyPatch, physical_blocks, remainder, accepted
) -> None:
    runner, fake_spec = _fixed_concurrency_runner(positive_blocks=9)
    monkeypatch.setattr(named_kv_module, "FullAttentionSpec", fake_spec)

    class PassedCapacityError(Exception):
        pass

    def stop_before_gpu_allocation(self):
        assert self.num_blocks == physical_blocks
        assert self.allocated_memory_bytes == physical_blocks * 16
        raise PassedCapacityError

    monkeypatch.setattr(NamedCausalKVBranch, "_preflight_device_memory", stop_before_gpu_allocation)
    with pytest.raises(PassedCapacityError if accepted else ValueError):
        NamedCausalKVBranch(
            runner=runner,
            request=NamedKVBranchRequest(name="negative", memory_bytes=physical_blocks * 16 + remainder),
        )


@pytest.mark.parametrize("length,blocks,valid", [(64, 8, False), (64, 9, True), (17, 4, False), (17, 5, True)])
def test_positive_capacity_rounds_each_request(monkeypatch: pytest.MonkeyPatch, length, blocks, valid) -> None:
    runner, fake_spec = _fixed_concurrency_runner(positive_blocks=blocks)
    runner.vllm_config.model_config.max_model_len = length
    monkeypatch.setattr(named_kv_module, "FullAttentionSpec", fake_spec)
    branch = object.__new__(NamedCausalKVBranch)
    branch.request = NamedKVBranchRequest(name="negative", memory_bytes=1024)
    if valid:
        branch._validate_runner_contract(runner)
    else:
        with pytest.raises(ValueError, match="Positive KV pool"):
            branch._validate_runner_contract(runner)


@pytest.mark.parametrize("failure", [None, "bind", "forward"])
def test_binding_context_restores_cache_identity(failure) -> None:
    branch = object.__new__(NamedCausalKVBranch)
    originals = {name: object() for name in ("a", "b")}
    branch.layers = {name: SimpleNamespace(kv_cache=value) for name, value in originals.items()}
    branch.kv_caches = {name: object() for name in originals}
    if failure == "bind":
        del branch.kv_caches["b"]

    def run():
        with branch._bind_branch_kv_caches():
            assert branch.layers["a"].kv_cache is branch.kv_caches["a"]
            if failure == "forward":
                raise ValueError("forward failed")

    if failure:
        with pytest.raises(KeyError if failure == "bind" else ValueError):
            run()
    else:
        run()
    assert all(branch.layers[name].kv_cache is value for name, value in originals.items())


def test_fixed_concurrency_rejects_insufficient_positive_capacity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner, fake_spec = _fixed_concurrency_runner(positive_blocks=7)
    monkeypatch.setattr(named_kv_module, "FullAttentionSpec", fake_spec)

    with pytest.raises(
        ValueError,
        match=r"Positive KV pool.*max_concurrent_requests=2.*required_tokens=128",
    ):
        NamedCausalKVBranch(
            runner=runner,
            request=NamedKVBranchRequest(name="negative", memory_bytes=128),
        )


def test_fixed_concurrency_rejects_insufficient_negative_capacity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner, fake_spec = _fixed_concurrency_runner(positive_blocks=9)
    monkeypatch.setattr(named_kv_module, "FullAttentionSpec", fake_spec)

    with pytest.raises(
        ValueError,
        match=r"Named causal KV branch.*max_concurrent_requests=2.*required_tokens=128",
    ):
        NamedCausalKVBranch(
            runner=runner,
            # Seven 16-byte blocks cannot reserve 2 x ceil(64 / 16).
            request=NamedKVBranchRequest(name="negative", memory_bytes=112),
        )


def test_runner_acknowledges_named_kv_capability_after_model_load(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = object.__new__(OmniGPUModelRunner)
    model = SimpleNamespace(
        named_kv_branch_request=NamedKVBranchRequest(
            name="negative",
            memory_bytes=1,
        )
    )
    events: list[str] = []

    def load_model(_runner, *_args, **_kwargs) -> None:
        events.append("loaded")
        runner.model = model

    monkeypatch.setattr(
        "vllm_omni.worker.gpu_model_runner.GPUModelRunner.load_model",
        load_model,
    )
    for method_name in (
        "_maybe_enable_output_token_ids_for_model_sampler",
        "_init_talker_mtp",
        "_prewarm_attention_capture_workspaces",
    ):
        monkeypatch.setattr(
            runner,
            method_name,
            lambda method_name=method_name: events.append(
                f"{method_name}:{getattr(model, 'named_kv_branch_capability_acknowledged', False)}"
            ),
        )

    OmniGPUModelRunner.load_model(runner)

    assert model.named_kv_branch_capability_acknowledged is True
    assert events == [
        "loaded",
        "_maybe_enable_output_token_ids_for_model_sampler:True",
        "_init_talker_mtp:True",
        "_prewarm_attention_capture_workspaces:True",
    ]


def test_runner_does_not_modify_undeclared_model_during_load(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = object.__new__(OmniGPUModelRunner)
    model = SimpleNamespace()

    def load_model(_runner, *_args, **_kwargs) -> None:
        runner.model = model

    monkeypatch.setattr(
        "vllm_omni.worker.gpu_model_runner.GPUModelRunner.load_model",
        load_model,
    )
    for method_name in (
        "_maybe_enable_output_token_ids_for_model_sampler",
        "_init_talker_mtp",
        "_prewarm_attention_capture_workspaces",
    ):
        monkeypatch.setattr(runner, method_name, lambda: None)

    OmniGPUModelRunner.load_model(runner)

    assert not hasattr(model, "named_kv_branch_capability_acknowledged")


def test_undeclared_model_keeps_named_kv_runner_path_disabled() -> None:
    runner = object.__new__(OmniGPUModelRunner)
    runner.model = object()
    runner.named_kv_branches = {}
    OmniGPUModelRunner._maybe_bind_named_kv_branch(runner)
    assert runner.named_kv_branches == {}


def test_runner_closes_unpublished_named_branch_when_model_bind_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []

    class Branch:
        def __init__(self, *, runner, request) -> None:
            self.name = request.name
            events.append(f"construct:{request.name}")

        def close(self) -> None:
            events.append("close")

    def bind(_branch) -> None:
        events.append("bind")
        raise RuntimeError("injected bind failure")

    monkeypatch.setattr(named_kv_module, "NamedCausalKVBranch", Branch)
    runner = object.__new__(OmniGPUModelRunner)
    runner.model = SimpleNamespace(
        named_kv_branch_request=NamedKVBranchRequest(name="negative", memory_bytes=1),
        bind_named_kv_branch=bind,
    )
    runner.named_kv_branches = {}

    with pytest.raises(RuntimeError, match="injected bind failure"):
        OmniGPUModelRunner._maybe_bind_named_kv_branch(runner)

    assert events == ["construct:negative", "bind", "close"]
    assert runner.named_kv_branches == {}


def test_runner_preserves_bind_error_when_unpublished_branch_close_also_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Branch:
        def __init__(self, *, runner, request) -> None:
            self.name = request.name

        def close(self) -> None:
            raise RuntimeError("injected close failure")

    def bind(_branch) -> None:
        raise ValueError("original bind failure")

    monkeypatch.setattr(named_kv_module, "NamedCausalKVBranch", Branch)
    runner = object.__new__(OmniGPUModelRunner)
    runner.model = SimpleNamespace(
        named_kv_branch_request=NamedKVBranchRequest(name="negative", memory_bytes=1),
        bind_named_kv_branch=bind,
    )
    runner.named_kv_branches = {}

    with pytest.raises(ValueError, match="original bind failure"):
        OmniGPUModelRunner._maybe_bind_named_kv_branch(runner)

    assert runner.named_kv_branches == {}


def test_runner_rejects_invalid_named_branch_declaration() -> None:
    runner = object.__new__(OmniGPUModelRunner)
    runner.model = SimpleNamespace(named_kv_branch_request={"name": "negative"})
    runner.named_kv_branches = {}
    with pytest.raises(TypeError, match="must be a NamedKVBranchRequest"):
        OmniGPUModelRunner._maybe_bind_named_kv_branch(runner)
    assert runner.named_kv_branches == {}


# ---------------------------------------------------------------------------
# append_batch() contract tests
# ---------------------------------------------------------------------------


def test_append_batch_returns_immutable_snapshot() -> None:
    """append_batch yields a NamedKVAppendBatch with tuple snapshots."""
    branch = _make_test_branch(num_blocks=128)

    branch.reset("req-a")
    branch.reset("req-b")

    with branch.append_batch(["req-a", "req-b"]) as step:
        # Verify all fields are tuples (immutable snapshots).
        assert isinstance(step.request_ids, tuple)
        assert isinstance(step.positions, tuple)
        assert isinstance(step.slot_values, tuple)
        assert isinstance(step.seq_lens, tuple)
        assert isinstance(step.block_ids, tuple)
        assert all(isinstance(b, tuple) for b in step.block_ids)

        # Verify values.
        assert step.request_ids == ("req-a", "req-b")
        assert step.positions == (0, 0)
        assert step.seq_lens == (1, 1)
        assert len(step.slot_values) == 2
        assert len(step.block_ids) == 2

    # After context exit, num_tokens should be advanced.
    assert branch.get_sequence_length("req-a") == 1
    assert branch.get_sequence_length("req-b") == 1


def test_append_batch_fault_frees_touched_requests() -> None:
    """If the body raises, append_batch fault-frees every touched request."""
    branch = _make_test_branch(num_blocks=128)

    branch.reset("req-a")

    with pytest.raises(RuntimeError, match="injected failure"):
        with branch.append_batch(["req-a"]) as step:
            assert step.positions == (0,)
            raise RuntimeError("injected failure")

    # Request should be fault-freed.
    assert branch.get_sequence_length("req-a") == 0
    assert "req-a" not in branch._states


def test_append_batch_rejects_reentry() -> None:
    """append_batch cannot be re-entered while active."""
    branch = _make_test_branch(num_blocks=128)

    branch.reset("req-a")

    with pytest.raises(RuntimeError, match="cannot be re-entered"):
        with branch.append_batch(["req-a"]):
            with branch.append_batch(["req-a"]):
                pass


def _make_qwen_adapter() -> Qwen2KVBranchAdapter:
    layers = [
        SimpleNamespace(
            self_attn=SimpleNamespace(
                attn=SimpleNamespace(
                    layer_name=name,
                    num_kv_heads=2,
                    head_size=8,
                )
            )
        )
        for name in ("first", "second")
    ]
    adapter = Qwen2KVBranchAdapter(SimpleNamespace(layers=layers), 32)
    adapter._build_layer_config = lambda attn: {}
    return adapter


def _make_kv_caches() -> list[torch.Tensor]:
    return [torch.zeros(3, 16, 2, 8) for _ in range(2)]


def test_shuffled_registry_binds_in_decoder_order_once() -> None:
    adapter = _make_qwen_adapter()
    keys, values = _make_kv_caches(), _make_kv_caches()
    args = dict(branch_layer_names=("second", "first"), k_caches=keys, v_caches=values)
    adapter.bind_kv_caches(**args)
    assert [pair[0] for pair in adapter._layer_pairs] == adapter.language_model.layers
    assert adapter._layer_pairs[0][1] is keys[1]
    assert adapter._layer_pairs[1][1] is keys[0]
    with pytest.raises(RuntimeError, match="already bound"):
        adapter.bind_kv_caches(**args)
    adapter.close()
    with pytest.raises(RuntimeError, match="closed"):
        adapter.bind_kv_caches(**args)


@pytest.mark.parametrize("invalid", ["duplicate", "shape", "missing"])
def test_failed_bind_does_not_publish_partial_pairs(invalid: str) -> None:
    adapter = _make_qwen_adapter()
    keys, values = _make_kv_caches(), _make_kv_caches()
    names = ("first", "second")
    if invalid == "duplicate":
        names = ("first", "first")
    elif invalid == "shape":
        values[1] = values[1][..., :4]
    else:
        names = ("first", "unknown")
    with pytest.raises(ValueError):
        adapter.bind_kv_caches(branch_layer_names=names, k_caches=keys, v_caches=values)
    assert adapter._layer_pairs == []


@pytest.mark.parametrize("hnd", [False, True])
def test_four_dimensional_views_share_native_storage(hnd: bool) -> None:
    # Logical packed shape stays the same while physical stride order differs.
    cache = (
        torch.zeros(3, 2, 16, 16, dtype=torch.bfloat16)
        if hnd
        else torch.zeros(3, 16, 2, 16, dtype=torch.bfloat16).transpose(1, 2)
    )
    branch = SimpleNamespace(
        backend=type("FlashAttentionBackend", (), {}),
        kv_cache_spec=SimpleNamespace(
            page_size_padded=None, dtype=torch.bfloat16, block_size=16, num_kv_heads=2, head_size=8
        ),
        num_blocks=3,
        block_size=16,
        device=torch.device("cpu"),
        layer_names=("first",),
        kv_caches={"first": cache},
    )
    keys, values = FlashAttentionKVBranchAdapter(branch).get_kv_caches()
    for view in (keys[0], values[0]):
        assert view.shape == (3, 16, 2, 8)
        assert view.untyped_storage().data_ptr() == cache.untyped_storage().data_ptr()
        assert view.stride() == (cache.stride(0), cache.stride(2), cache.stride(1), 1)
    keys[0][1, 3, 0, 2] = 7
    values[0][1, 3, 0, 2] = 9
    assert cache[1, 0, 3, 2] == 7
    assert cache[1, 0, 3, 10] == 9


@pytest.mark.parametrize(
    "invalid", ["backend", "padding", "dtype", "block_size", "ndim", "shape", "strides", "cache_dtype", "device"]
)
def test_unsupported_pool_contract_rejected_before_execution(invalid: str) -> None:
    branch = SimpleNamespace(
        backend=type("FlashAttentionBackend", (), {}),
        kv_cache_spec=SimpleNamespace(
            page_size_padded=None, dtype=torch.bfloat16, block_size=16, num_kv_heads=2, head_size=8
        ),
        num_blocks=3,
        block_size=16,
        device=torch.device("cpu"),
        layer_names=("first",),
        kv_caches={"first": torch.zeros(3, 2, 16, 16, dtype=torch.bfloat16)},
    )
    if invalid == "backend":
        branch.backend = type("OtherBackend", (), {})
    elif invalid == "padding":
        branch.kv_cache_spec.page_size_padded = 4096
    elif invalid == "dtype":
        branch.kv_cache_spec.dtype = torch.float32
    elif invalid == "block_size":
        branch.kv_cache_spec.block_size = 7
    elif invalid == "ndim":
        branch.kv_caches["first"] = branch.kv_caches["first"].flatten()
    elif invalid == "shape":
        branch.kv_caches["first"] = branch.kv_caches["first"][:2]
    elif invalid == "strides":
        branch.kv_caches["first"] = torch.zeros(3, 2, 16, 32, dtype=torch.bfloat16)[..., ::2]
    elif invalid == "cache_dtype":
        branch.kv_caches["first"] = branch.kv_caches["first"].float()
    else:
        branch.device = torch.device("cuda:0")
    with pytest.raises(UnsupportedNamedKVGraphError):
        FlashAttentionKVBranchAdapter(branch)


def test_executor_reuses_bound_cache_objects_without_retrieving_views(mocker: MockerFixture) -> None:
    adapter = _make_qwen_adapter()
    keys, values = _make_kv_caches(), _make_kv_caches()
    adapter.bind_kv_caches(branch_layer_names=("second", "first"), k_caches=keys, v_caches=values)
    get_kv_caches = mocker.patch.object(
        FlashAttentionKVBranchAdapter, "get_kv_caches", side_effect=AssertionError("duplicate view retrieval")
    )
    executor = NamedKVBranchExecutor(
        mocker.Mock(),
        adapter,
        max_num_seqs=4,
        max_model_len=32,
        device=torch.device("cpu"),
        dtype=torch.bfloat16,
    )
    assert executor._k_caches[0] is keys[1]
    assert executor._v_caches[0] is values[1]
    get_kv_caches.assert_not_called()
    executor.close()


def _make_execution_buffers(
    batch_size: int = 1,
    hidden_size: int = 2,
    dtype: torch.dtype = torch.float32,
) -> ExecutionBuffers:
    block_table = torch.zeros(batch_size, 16, dtype=torch.int32)
    block_table[:, 0] = torch.arange(1, batch_size + 1, dtype=torch.int32)
    return ExecutionBuffers(
        embeddings=torch.zeros(batch_size, hidden_size, dtype=dtype),
        positions=torch.zeros(batch_size, dtype=torch.int64),
        slot_mapping=torch.arange(1, batch_size + 1, dtype=torch.int64) * 16,
        block_table=block_table,
        query_start_loc=torch.arange(batch_size + 1, dtype=torch.int32),
        seq_lens=torch.ones(batch_size, dtype=torch.int32),
        output=torch.zeros(batch_size, hidden_size, dtype=dtype),
    )


def test_run_without_graph_never_warms_up_or_appends(mocker: MockerFixture) -> None:
    executor = NamedKVBranchExecutor.__new__(NamedKVBranchExecutor)
    executor._closed = False
    executor.max_num_seqs = 4
    executor._compiled_fn = None
    executor._graphs = {}
    executor.branch = mocker.Mock()
    executor.warmup = mocker.Mock(side_effect=AssertionError("nested warmup"))
    buffers = _make_execution_buffers()
    output = buffers.output
    executor._ensure_buffers = mocker.Mock(return_value=buffers)
    executor._write_buffers = mocker.Mock()
    executor._buffer_args = mocker.Mock(return_value=())
    executor._eager_fn = mocker.Mock(side_effect=lambda: output.fill_(3))
    step = NamedKVAppendBatch(
        request_ids=("real",), positions=(0,), slot_values=(16,), seq_lens=(1,), block_ids=((1,),)
    )
    result = executor.run(step, [torch.ones(1, 2)])
    executor.warmup.assert_not_called()
    assert not executor.branch.mock_calls
    executor._eager_fn.assert_called_once_with()
    output.zero_()
    torch.testing.assert_close(result, torch.full((1, 2), 3.0))


def test_warmup_rejects_active_append_before_allocating(mocker: MockerFixture) -> None:
    executor = NamedKVBranchExecutor.__new__(NamedKVBranchExecutor)
    executor._closed = False
    executor.branch = mocker.Mock()
    executor.branch._ensure_not_entered.side_effect = RuntimeError("active append")
    executor.model_adapter = mocker.Mock()
    with pytest.raises(RuntimeError, match="active append"):
        executor.warmup([1])
    assert not executor.model_adapter.mock_calls


def test_repeat_warmup_keeps_workspace_and_compiled_callable(mocker: MockerFixture) -> None:
    executor = NamedKVBranchExecutor.__new__(NamedKVBranchExecutor)
    executor._closed = False
    executor.branch = mocker.Mock()
    executor.model_adapter = mocker.Mock()
    executor.max_num_seqs = 4
    compiled = mocker.Mock()
    executor._compiled_fn = compiled
    executor._graphs = {1: object()}
    executor._capture_owned_batch = mocker.Mock()
    executor.warmup([1, 2])
    executor.model_adapter.init_graph_workspace.assert_not_called()
    assert executor._compiled_fn is compiled
    executor._capture_owned_batch.assert_called_once_with(2)


def test_compile_failure_releases_scratch_and_preserves_error(mocker: MockerFixture) -> None:
    executor = NamedKVBranchExecutor.__new__(NamedKVBranchExecutor)
    executor.branch = mocker.Mock()
    executor.branch.append_batch.return_value = nullcontext(object())
    executor.branch.free.side_effect = RuntimeError("cleanup failure")
    executor._ensure_buffers = mocker.Mock(return_value=object())
    executor._write_buffers = mocker.Mock()
    executor._dummy_embeddings = mocker.Mock(return_value=[])
    executor._update_scheduler_metadata = mocker.Mock()
    executor._buffer_args = mocker.Mock(return_value=())
    executor._compiled_fn = mocker.Mock(side_effect=ValueError("compile failure"))
    executor._graphs = {}
    executor._graph_output_refs = {}
    with pytest.raises(ValueError, match="compile failure"):
        executor._capture_owned_batch(2)
    ids = executor.branch.append_batch.call_args.args[0]
    assert len(set(ids)) == 2
    assert [call.args[0] for call in executor.branch.free.call_args_list] == ids
    assert not executor._graphs
    assert not executor._graph_output_refs


@pytest.mark.parametrize("failure_site", ["compile", "capture"])
def test_failed_warmup_releases_executor_resources(
    mocker: MockerFixture, monkeypatch: pytest.MonkeyPatch, failure_site: str
) -> None:
    executor = NamedKVBranchExecutor.__new__(NamedKVBranchExecutor)
    executor._closed = False
    executor.branch = mocker.Mock()
    executor.max_num_seqs = 4
    executor.model_adapter = mocker.Mock()
    executor.model_adapter.close.side_effect = RuntimeError("cleanup failed")
    executor._compiled_fn = None
    executor._eager_fn = mocker.Mock()
    executor._graphs = {1: object()}
    executor._graph_output_refs = {1: torch.zeros(1)}
    executor._buffers = {1: object()}
    executor._k_caches = [torch.zeros(1)]
    executor._v_caches = [torch.zeros(1)]
    compiler = mocker.Mock(return_value=mocker.Mock())
    executor._capture_owned_batch = mocker.Mock()
    if failure_site == "compile":
        compiler.side_effect = ValueError("startup failed")
    else:
        executor._capture_owned_batch.side_effect = ValueError("startup failed")
    monkeypatch.setattr(torch, "compile", compiler)
    with pytest.raises(ValueError, match="startup failed"):
        executor.warmup([2])
    assert executor._closed
    assert not executor._graphs and not executor._graph_output_refs
    assert not executor._buffers and not executor._k_caches and not executor._v_caches
    assert executor._compiled_fn is None and executor._eager_fn is None
    executor.branch.close.assert_not_called()
    executor.close()
    executor.model_adapter.close.assert_called_once_with()
    with pytest.raises(RuntimeError, match="closed"):
        executor.warmup([1])


@pytest.mark.parametrize("fail", [False, True])
def test_negative_wrapper_appends_once_without_old_path_retry(mocker: MockerFixture, fail: bool) -> None:
    from vllm_omni.model_executor.models.vibevoice.negative_branch import VibeVoiceNegativeBranch

    store = mocker.Mock()
    store.name = "negative"
    step = object()
    store.append_batch.return_value = nullcontext(step)
    executor = mocker.Mock()
    executor.run.return_value = torch.ones(2, 4)
    wrapper = VibeVoiceNegativeBranch(store=store, language_model=mocker.Mock(), hidden_size=4, executor=executor)
    ids = ["first", "second"]
    embeddings = [torch.zeros(1, 4), torch.zeros(1, 4)]
    if fail:
        executor.run.side_effect = RuntimeError("execution failed")
        with pytest.raises(RuntimeError, match="execution failed"):
            wrapper.forward_step(ids, embeddings)
        assert [call.args[0] for call in store.free.call_args_list] == ids
    else:
        output = wrapper.forward_step(ids, embeddings)
        assert len(output) == 2
        store.free.assert_not_called()
    store.append_batch.assert_called_once_with(ids)
    executor.run.assert_called_once_with(step, embeddings)
    store.append_and_enter_batch.assert_not_called()
    store.append_and_enter.assert_not_called()


def test_dummy_embeddings_do_not_advance_rng() -> None:
    executor = NamedKVBranchExecutor.__new__(NamedKVBranchExecutor)
    executor.model_adapter = SimpleNamespace(hidden_size=2)
    executor.dtype = torch.float32
    executor.device = torch.device("cpu")
    before = torch.random.get_rng_state().clone()
    embeddings = executor._dummy_embeddings(4)
    assert len(embeddings) == 4
    assert all(torch.isfinite(x).all() for x in embeddings)
    torch.testing.assert_close(torch.random.get_rng_state(), before)


@pytest.mark.parametrize("preallocated", [False, True])
def test_explicit_fa3_and_fixed_splits(
    mocker: MockerFixture, monkeypatch: pytest.MonkeyPatch, preallocated: bool
) -> None:
    import vllm._custom_ops as custom_ops
    import vllm.v1.attention.backends.fa_utils as fa_utils

    scatter = mocker.Mock()
    attention = mocker.Mock()
    metadata = torch.zeros(17, dtype=torch.int32)
    scheduler = mocker.Mock(return_value=metadata)
    monkeypatch.setattr(custom_ops, "reshape_and_cache_flash", scatter)
    monkeypatch.setattr(fa_utils, "flash_attn_varlen_func", attention)
    monkeypatch.setattr(fa_utils, "get_scheduler_metadata", scheduler)
    q = torch.zeros(1, 4, 8)
    k = torch.zeros(1, 2, 8)
    cache = torch.zeros(2, 16, 2, 8)
    _named_kv_branch_attention_impl(
        q,
        k,
        k.clone(),
        torch.empty_like(q),
        cache,
        cache.clone(),
        torch.zeros(1, dtype=torch.int64),
        torch.zeros(1, 1, dtype=torch.int32),
        torch.arange(2, dtype=torch.int32),
        torch.ones(1, dtype=torch.int32),
        1,
        16,
        "auto",
        torch.ones(1),
        torch.ones(1),
        0.5,
        2,
        8,
        4,
        16,
        metadata if preallocated else None,
        3,
    )
    scatter.assert_called_once()
    assert attention.call_args.kwargs["fa_version"] == 3
    assert attention.call_args.kwargs["num_splits"] == 1
    assert attention.call_args.kwargs["scheduler_metadata"] is metadata
    if preallocated:
        scheduler.assert_not_called()
    else:
        assert scheduler.call_args.kwargs["num_splits"] == 1


def test_metadata_effective_size_is_stable_and_bounded(mocker: MockerFixture, monkeypatch: pytest.MonkeyPatch) -> None:
    import vllm.v1.attention.backends.fa_utils as fa_utils

    executor = NamedKVBranchExecutor.__new__(NamedKVBranchExecutor)
    executor._capacity = 256
    executor.dtype = torch.bfloat16
    workspaces = [torch.zeros(17, dtype=torch.int32) for _ in range(2)]
    pointers = [x.data_ptr() for x in workspaces]
    executor.model_adapter = SimpleNamespace(
        _layer_pairs=[(None, None, None, {"num_heads_q": 4, "num_kv_heads": 2, "head_size": 16})],
        _block_size=16,
        _scheduler_metadata=workspaces,
        _scheduler_metadata_sizes={},
    )
    bufs = _make_execution_buffers(hidden_size=64, dtype=torch.bfloat16)
    scheduler = mocker.Mock(return_value=torch.arange(5, dtype=torch.int32))
    monkeypatch.setattr(fa_utils, "get_scheduler_metadata", scheduler)
    executor._update_scheduler_metadata(bufs, 1)
    executor._update_scheduler_metadata(bufs, 1)
    assert executor.model_adapter._scheduler_metadata_sizes == {1: 5}
    assert [x.data_ptr() for x in workspaces] == pointers
    for workspace in workspaces:
        torch.testing.assert_close(workspace[:5], torch.arange(5, dtype=torch.int32))
        assert not workspace[5:].any()
    snapshots = [x.clone() for x in workspaces]
    scheduler.return_value = torch.zeros(6, dtype=torch.int32)
    with pytest.raises(RuntimeError, match="size changed"):
        executor._update_scheduler_metadata(bufs, 1)
    for workspace, snapshot in zip(workspaces, snapshots, strict=True):
        torch.testing.assert_close(workspace, snapshot)
    scheduler.return_value = torch.zeros(18, dtype=torch.int32)
    two_row_bufs = _make_execution_buffers(batch_size=2, hidden_size=64, dtype=torch.bfloat16)
    with pytest.raises(RuntimeError, match="exceeds"):
        executor._update_scheduler_metadata(two_row_bufs, 2)
    assert 2 not in executor.model_adapter._scheduler_metadata_sizes
