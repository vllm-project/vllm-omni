# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""CPU contracts for named KV allocation, binding and executor lifecycle.

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
from vllm_omni.worker.gpu_model_runner import OmniGPUModelRunner
from vllm_omni.worker.named_kv.executor import NamedKVBranchExecutor
from vllm_omni.worker.named_kv.runtime import _build_named_kv_manager
from vllm_omni.worker.named_kv_branch import (
    NamedCausalKVBranch,
    NamedKVBranchRequest,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


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
    branch._manager = _build_named_kv_manager(
        _make_spec(block_size),
        ["layer0"],
        num_blocks,
        max_sequence_tokens,
    )
    branch._query_start_cpu = torch.tensor([0, 1], dtype=torch.int32)
    branch._query_start_gpu = branch._query_start_cpu.to(branch.device)
    return branch


class _FakeFullAttentionSpec:
    block_size = 16
    page_size_bytes = 16
    page_size_padded = None

    def __init__(self, *a, **kw) -> None:
        pass


def _fixed_concurrency_runner(*, positive_blocks: int):
    config = SimpleNamespace(
        scheduler_config=SimpleNamespace(max_num_seqs=2),
        cache_config=SimpleNamespace(
            enable_prefix_caching=False,
            cache_dtype="auto",
            block_size=16,
            kv_cache_memory_bytes=positive_blocks * 16,
        ),
        parallel_config=SimpleNamespace(
            pipeline_parallel_size=1,
            prefill_context_parallel_size=1,
            decode_context_parallel_size=1,
            use_ubatching=False,
            tensor_parallel_size=1,
        ),
        model_config=SimpleNamespace(
            enable_sleep_mode=False,
            enforce_eager=True,
            max_model_len=64,
            dtype=torch.bfloat16,
        ),
        speculative_config=None,
        kv_transfer_config=None,
        compilation_config=SimpleNamespace(static_forward_context={"layer": object()}),
    )
    spec = _FakeFullAttentionSpec()
    runner = SimpleNamespace(
        vllm_config=config,
        device="cpu",
        kv_cache_config=SimpleNamespace(
            kv_cache_groups=[SimpleNamespace(kv_cache_spec=spec)],
            num_blocks=positive_blocks,
        ),
        attn_groups=[[SimpleNamespace(backend=object(), layer_names=["layer"])]],
        _kernel_block_sizes=[16],
    )
    return runner, _FakeFullAttentionSpec


# ===========================================================================
# Allocation and capacity
# ===========================================================================


def test_manager_describes_independent_layer_storage() -> None:
    spec = _make_spec(16)
    manager = _build_named_kv_manager(spec, ["a", "b"], 9, 64)
    tensors = manager.kv_cache_config.kv_cache_tensors
    assert len(tensors) == 1
    assert tensors[0].layers == ["a", "b"]
    assert tensors[0].layer_stride == spec.page_size_bytes * 9
    assert tensors[0].block_stride == spec.page_size_bytes
    assert tensors[0].size == 2 * spec.page_size_bytes * 9
    assert manager.block_pool.get_num_free_blocks() == 8


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
    with pytest.raises(ValueError, match="must be non-negative"):
        NamedKVBranchRequest(name="negative", memory_bytes=1, activation_margin_bytes=-1)


@pytest.mark.parametrize("physical_blocks,remainder,accepted", [(8, 0, False), (9, 0, True)])
def test_negative_capacity_null_block_and_rounding(
    monkeypatch: pytest.MonkeyPatch, physical_blocks, remainder, accepted
) -> None:
    runner, fake_spec = _fixed_concurrency_runner(positive_blocks=9)
    monkeypatch.setattr(named_kv_module, "FullAttentionSpec", fake_spec)

    class PassedCapacityError(Exception):
        pass

    def stop_before_gpu_allocation(self):
        assert self.num_blocks == physical_blocks
        raise PassedCapacityError

    monkeypatch.setattr(NamedCausalKVBranch, "_preflight_device_memory", stop_before_gpu_allocation)
    with pytest.raises(PassedCapacityError if accepted else ValueError):
        NamedCausalKVBranch(
            runner=runner,
            request=NamedKVBranchRequest(name="negative", memory_bytes=physical_blocks * 16 + remainder),
        )


def test_fixed_concurrency_rejects_insufficient_positive_capacity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner, fake_spec = _fixed_concurrency_runner(positive_blocks=7)
    monkeypatch.setattr(named_kv_module, "FullAttentionSpec", fake_spec)
    with pytest.raises(ValueError, match=r"Positive KV pool.*max_concurrent_requests=2"):
        NamedCausalKVBranch(
            runner=runner,
            request=NamedKVBranchRequest(name="negative", memory_bytes=128),
        )


# ===========================================================================
# Runner capability
# ===========================================================================


def test_runner_acknowledges_named_kv_capability_after_model_load(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = object.__new__(OmniGPUModelRunner)
    model = SimpleNamespace(named_kv_branch_request=NamedKVBranchRequest(name="negative", memory_bytes=1))
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


# ===========================================================================
# Append batch and executor lifecycle
# ===========================================================================


def test_append_batch_returns_immutable_snapshot() -> None:
    branch = _make_test_branch(num_blocks=128)
    branch.reset("req-a")
    branch.reset("req-b")

    with branch.append_batch(["req-a", "req-b"]) as step:
        assert isinstance(step.request_ids, tuple)
        assert isinstance(step.positions, tuple)
        assert isinstance(step.block_ids, tuple)
        assert step.request_ids == ("req-a", "req-b")
        assert step.positions == (0, 0)
        assert step.seq_lens == (1, 1)

    assert branch.get_sequence_length("req-a") == 1
    assert branch.get_sequence_length("req-b") == 1


def test_append_batch_fault_frees_touched_requests() -> None:
    branch = _make_test_branch(num_blocks=128)
    branch.reset("req-a")

    with pytest.raises(RuntimeError, match="injected failure"):
        with branch.append_batch(["req-a"]):
            raise RuntimeError("injected failure")

    assert branch.get_sequence_length("req-a") == 0
    assert "req-a" not in branch._states


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


@pytest.mark.parametrize("fail", [False, True])
def test_negative_wrapper_appends_once_without_old_path_retry(mocker: MockerFixture, fail: bool) -> None:
    from vllm_omni.model_executor.models.vibevoice.negative_branch import VibeVoiceNegativeBranch

    store = mocker.Mock()
    store.name = "negative"
    step = object()
    store.append_batch.return_value = nullcontext(step)
    executor = mocker.Mock()
    executor.run.return_value = torch.ones(2, 4)
    wrapper = VibeVoiceNegativeBranch(
        store=store,
        language_model=mocker.Mock(),
        hidden_size=4,
        executor=executor,
    )
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
