# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""CPU contracts for the fixed-pool named causal KV capability."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

import vllm_omni.worker.named_kv_branch as named_kv_module
from vllm_omni.worker.gpu_model_runner import OmniGPUModelRunner
from vllm_omni.worker.named_kv_branch import (
    NamedCausalKVBranch,
    NamedKVBranchRequest,
    _FixedBlockAllocator,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


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


def test_fixed_block_allocator_is_deterministic_and_rejects_corruption() -> None:
    allocator = _FixedBlockAllocator(3)
    assert [allocator.allocate(), allocator.allocate(), allocator.allocate()] == [
        0,
        1,
        2,
    ]
    assert allocator.num_free_blocks == 0
    with pytest.raises(RuntimeError, match="exhausted its fixed GPU block pool"):
        allocator.allocate()

    allocator.free([0, 1, 2])
    assert allocator.num_free_blocks == 3
    assert [allocator.allocate(), allocator.allocate(), allocator.allocate()] == [
        0,
        1,
        2,
    ]
    with pytest.raises(ValueError, match="Cannot free unallocated"):
        allocator.free([99])


def test_named_kv_public_reset_and_free_reject_active_context() -> None:
    branch = object.__new__(NamedCausalKVBranch)
    branch.name = "negative"
    branch.device = "cpu"
    branch.max_blocks_per_request = 1
    branch._entered = True
    branch._closed = False
    branch._allocator = _FixedBlockAllocator(1)
    block_id = branch._allocator.allocate()
    state = SimpleNamespace(block_ids=[block_id])
    branch._states = {"request": state}

    with pytest.raises(RuntimeError, match="Cannot reset.*forward context"):
        branch.reset("request")
    with pytest.raises(RuntimeError, match="Cannot free.*forward context"):
        branch.free("request")

    assert branch._states == {"request": state}
    assert branch.num_free_blocks == 0


def test_named_kv_internal_fault_cleanup_remains_legal_in_context() -> None:
    branch = object.__new__(NamedCausalKVBranch)
    branch.name = "negative"
    branch._entered = True
    branch._closed = False
    branch._allocator = _FixedBlockAllocator(1)
    block_id = branch._allocator.allocate()
    branch._states = {
        "request": SimpleNamespace(block_ids=[block_id]),
    }

    branch._free_unchecked("request")

    assert branch._states == {}
    assert branch.num_free_blocks == 1


def test_append_slots_validates_batch_before_mutating_any_request() -> None:
    branch = object.__new__(NamedCausalKVBranch)
    branch.name = "negative"
    branch.device = "cpu"
    branch._entered = False
    branch._closed = False
    branch.block_size = 2
    branch.max_sequence_tokens = 4
    branch.max_blocks_per_request = 4
    branch._allocator = _FixedBlockAllocator(8)
    branch._states = {}
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
    branch = object.__new__(NamedCausalKVBranch)
    branch.name = "negative"
    branch.device = "cpu"
    branch._entered = False
    branch._closed = False
    branch.block_size = 1
    branch.max_sequence_tokens = 8
    branch.max_blocks_per_request = 4
    branch._allocator = _FixedBlockAllocator(1)
    branch._states = {}
    branch.reset("request-a")
    branch.reset("request-b")

    # One block total: request-a takes it, request-b exhausts the pool and
    # the whole logical batch is fault-freed.
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
        "vllm_omni.worker.named_kv_branch.logger.exception",
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
    runner, fake_spec = _fixed_concurrency_runner(positive_blocks=8)
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
