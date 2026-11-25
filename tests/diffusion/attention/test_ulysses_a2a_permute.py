# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import ctypes
import math
import threading
from types import SimpleNamespace

import pytest
import torch

from vllm_omni.diffusion.attention.backends.abstract import AttentionMetadata
from vllm_omni.diffusion.attention.parallel import factory, ulysses
from vllm_omni.diffusion.distributed import a2a_permute
from vllm_omni.diffusion.forward_context import set_forward_context

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


@pytest.mark.parametrize("npu", [False, True])
@pytest.mark.parametrize("joint", [None, "front", "rear"])
@pytest.mark.parametrize("use_sync", [False, True])
def test_shared_a2a_routes_qkv_gate_and_reverse(monkeypatch, npu, joint, use_sync):
    """Both platforms share attention routing, including joint slicing."""
    monkeypatch.setenv("VLLM_OMNI_PACKED_QKV", "1")
    calls = []
    platform = SimpleNamespace(is_npu=lambda: npu, synchronize=lambda: calls.append("sync"))
    monkeypatch.setattr(ulysses, "current_omni_platform", platform)
    monkeypatch.setattr(a2a_permute, "current_omni_platform", platform)
    monkeypatch.setattr(a2a_permute, "ensure_a2a_permute_available", lambda: None)
    group = SimpleNamespace(
        ulysses_group=SimpleNamespace(group_name="ulysses"),
        ulysses_world_size=2,
        ulysses_rank=0,
        ring_world_size=1,
        allgather_world_size=1,
    )
    monkeypatch.setattr(ulysses.dist, "get_world_size", lambda pg: 2)
    strategy = ulysses.UlyssesParallelAttention(group, 2, 1, use_sync, ulysses_a2a_permute=True)
    assert strategy._packed_qkv is False
    query = torch.randn(1, 3, 4, 16)
    key = torch.randn(1, 3, 2, 16)
    value = torch.randn_like(key)
    gate = torch.randn_like(query)

    def qkv(q, k, v, group_name, world_size, zero_copy=False):
        assert q is query and k is key and v is value
        # Q/K/V stay inside the attention layer, so they are exchanged in place.
        assert (group_name, world_size, zero_copy) == ("ulysses", 2, True)
        calls.append("qkv")
        return tuple(t[:, :, : t.shape[2] // 2].repeat(1, 2, 1, 1) for t in (q, k, v))

    def forward(tensor, group_name, world_size):
        calls.append("gate")
        assert tensor is gate
        return tensor[:, :, :2].repeat(1, 2, 1, 1)

    def reverse(tensor, group_name, world_size):
        calls.append("reverse")
        assert tensor.shape == (1, 6, 2, 16)
        return tensor[:, :3].repeat(1, 1, 2, 1)

    monkeypatch.setattr(a2a_permute, "ulysses_qkv_fwd_group", qkv)
    monkeypatch.setattr(a2a_permute, "ulysses_qkv_fwd", forward)
    monkeypatch.setattr(a2a_permute, "ulysses_o_rev", reverse)
    monkeypatch.setattr(ulysses, "get_ulysses_mode", lambda **kwargs: "strict")
    metadata = AttentionMetadata(extra={"gate_compress": gate})
    if joint:
        metadata.joint_strategy = joint
        metadata.joint_query = torch.randn(1, 2, 4, 16)
        metadata.joint_key = torch.randn(1, 2, 2, 16)
        metadata.joint_value = torch.randn(1, 2, 2, 16)

        def gather(outputs, tensor, **kwargs):
            for output in outputs:
                output.copy_(tensor)

        monkeypatch.setattr(ulysses.dist, "all_gather", gather)
    q, k, v, metadata, context = strategy.pre_attention(query, key, value, metadata)
    assert q.shape == (1, 8 if joint else 6, 2, 16)
    assert k.shape == v.shape == (1, 8 if joint else 6, 1, 16)
    assert metadata.extra["gate_compress"].shape == (1, 6, 2, 16)
    output = strategy.post_attention(q, context)
    assert output.shape == (1, 5 if joint else 3, 4, 16)
    assert calls == (["gate", "qkv", "sync", "reverse", "sync"] if use_sync else ["gate", "qkv", "reverse"])


def test_qkv_views_survive_gate_exchange_and_output_is_copied_out(monkeypatch):
    """Two ranks exchange Q/K/V, the gate and the output through real shared buffers.

    The copy-engine transport runs on CPU threads, one per rank, writing peer
    buffers with memmove. Q/K/V are views of the rank's workspace, so they must
    still hold their exchanged values after the gate exchange; the output leaves
    the attention layer and must not alias the workspace.
    """
    world_size = 2
    barrier = threading.Barrier(world_size, timeout=30)

    class Stream:
        cuda_stream = 17

        def wait_event(self, event):
            pass

    class Event:
        def record(self, stream):
            pass

    def copy2d(dst, dpitch, src, spitch, width, height, kind, stream):
        for row in range(height):
            ctypes.memmove(dst + row * dpitch, src + row * spitch, width)
        return 0

    runtime = SimpleNamespace(lib=SimpleNamespace(cudaMemcpy2DAsync=copy2d), CUDART_CHECK=lambda result: None)
    platform = SimpleNamespace(is_npu=lambda: False, synchronize=lambda: None)
    monkeypatch.setattr(ulysses, "current_omni_platform", platform)
    monkeypatch.setattr(a2a_permute, "current_omni_platform", platform)
    monkeypatch.setattr(a2a_permute, "ensure_a2a_permute_available", lambda: None)
    monkeypatch.setattr(a2a_permute, "_ce_runtime", lambda: runtime)
    # Large requests skip the staged CUDA graph; that is where zero-copy applies.
    monkeypatch.setattr(a2a_permute, "_CE_GRAPH_MAX_INPUT_BYTES", 0)
    monkeypatch.setattr(a2a_permute, "_CE_QKV_GRAPH_MAX_INPUT_BYTES", 0)
    monkeypatch.setattr(torch.cuda, "current_stream", lambda device=None: Stream())
    monkeypatch.setattr(torch.cuda, "Stream", lambda device=None: Stream())
    monkeypatch.setattr(torch.cuda, "Event", Event)
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    monkeypatch.setattr(ulysses.dist, "get_world_size", lambda pg: world_size)
    monkeypatch.setattr(ulysses, "get_ulysses_mode", lambda **kwargs: "strict")

    # The registered ops are CUDA-only; call the Python bodies they wrap.
    def qkv_fwd(x, group_name, world_size):
        b, s, h, d = x.shape
        return a2a_permute._exchange_ce((x,), group_name, world_size).view(b, s * world_size, h // world_size, d)

    monkeypatch.setattr(a2a_permute, "ulysses_qkv_ce", a2a_permute.ulysses_qkv_ce._init_fn)
    monkeypatch.setattr(a2a_permute, "ulysses_qkv_fwd", qkv_fwd)
    monkeypatch.setattr(a2a_permute, "ulysses_o_rev", a2a_permute.ulysses_o_rev._init_fn)

    shape = (1, 3, 4, 8)
    inputs = [
        {
            name: torch.full(shape, 100.0 * rank + index) + torch.arange(math.prod(shape)).view(shape) / 1000
            for index, name in enumerate(("query", "key", "value", "gate"))
        }
        for rank in range(world_size)
    ]
    capacity = 3 * math.prod(shape)
    shared = [torch.zeros(capacity) for _ in range(world_size)]
    workspaces = {}
    for rank in range(world_size):
        handle = SimpleNamespace(
            rank=rank, buffer_ptrs=[buffer.data_ptr() for buffer in shared], barrier=lambda: barrier.wait()
        )
        workspaces[(torch.device("cpu"), f"ulysses{rank}")] = a2a_permute._SymmWorkspace(
            shared[rank], handle, shared[rank].nbytes, Stream.cuda_stream, ce_stream=Stream()
        )
    monkeypatch.setattr(a2a_permute, "_SYMM_WORKSPACES", workspaces)
    monkeypatch.setattr(
        a2a_permute,
        "_get_symm_buffer",
        lambda shape, dtype, device, group_name: workspaces[(device, group_name)].allocation[: shape[0]],
    )

    results: list[tuple[torch.Tensor, ...] | None] = [None] * world_size
    errors: list[BaseException] = []

    def run(rank):
        try:
            group = SimpleNamespace(
                ulysses_group=SimpleNamespace(group_name=f"ulysses{rank}"),
                ulysses_world_size=world_size,
                ulysses_rank=rank,
                ring_world_size=1,
                allgather_world_size=1,
            )
            strategy = ulysses.UlyssesParallelAttention(group, 2, 1, False, ulysses_a2a_permute=True)
            local = inputs[rank]
            metadata = AttentionMetadata(extra={"gate_compress": local["gate"]})
            q, k, v, metadata, ctx = strategy.pre_attention(local["query"], local["key"], local["value"], metadata)
            # Every rank finishes both exchanges before anyone checks its views.
            barrier.wait()
            workspace = shared[rank]
            start, end = workspace.data_ptr(), workspace.data_ptr() + workspace.nbytes
            assert all(start <= t.data_ptr() < end for t in (q, k, v))
            assert not start <= metadata.extra["gate_compress"].data_ptr() < end
            exchanged = (q.clone(), k.clone(), v.clone(), metadata.extra["gate_compress"].clone())
            # Reversing the exchanged query must return this rank's query, in
            # its own storage rather than in the reused workspace.
            output = strategy.post_attention(exchanged[0], ctx)
            assert not start <= output.data_ptr() < end
            results[rank] = (*exchanged, output)
        except BaseException as error:
            errors.append(error)
            barrier.abort()

    threads = [threading.Thread(target=run, args=(rank,)) for rank in range(world_size)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert not errors, errors

    heads = shape[2] // world_size
    for rank, outputs in enumerate(results):
        for output, name in zip(outputs, ("query", "key", "value", "gate")):
            expected = torch.cat([x[name][:, :, rank * heads : (rank + 1) * heads] for x in inputs], dim=1)
            torch.testing.assert_close(output, expected, rtol=0, atol=0)
        torch.testing.assert_close(outputs[4], inputs[rank]["query"], rtol=0, atol=0)


@pytest.mark.parametrize(
    "overrides, message",
    [
        ({"enforce_eager": False}, "enforce_eager"),
        ({"ulysses_mode": "advanced_uaa"}, "strict, pure Ulysses"),
        ({"ring_degree": 2}, "strict, pure Ulysses"),
        ({"allgather_degree": 2}, "strict, pure Ulysses"),
        ({"ulysses_degree": 1}, "strict, pure Ulysses"),
        ({"scatter_idx": 1}, "scatter_idx=2, gather_idx=1"),
        ({"gather_idx": 2}, "scatter_idx=2, gather_idx=1"),
    ],
)
def test_npu_backend_rejects_unsupported_config(monkeypatch, overrides, message):
    values = dict(ulysses_a2a_permute=True, ulysses_degree=2, ring_degree=1, ulysses_mode="strict")
    values.update(overrides)
    enforce_eager = values.pop("enforce_eager", True)
    scatter_idx = values.pop("scatter_idx", 2)
    gather_idx = values.pop("gather_idx", 1)
    config = SimpleNamespace(
        parallel_config=SimpleNamespace(**values),
        enforce_eager=enforce_eager,
    )
    monkeypatch.setattr(a2a_permute, "current_omni_platform", SimpleNamespace(is_npu=lambda: True))
    monkeypatch.setattr(factory, "is_forward_context_available", lambda: True)
    monkeypatch.setattr(factory, "get_forward_context", lambda: SimpleNamespace(omni_diffusion_config=config))
    with pytest.raises(ValueError, match=message):
        factory.build_parallel_attention_strategy(scatter_idx=scatter_idx, gather_idx=gather_idx, use_sync=False)


@pytest.mark.parametrize(
    "ulysses_size, ring_size, allgather_size, scatter_idx, gather_idx, message",
    [
        (1, 1, 1, 2, 1, "strict, pure Ulysses"),
        (2, 2, 1, 2, 1, "strict, pure Ulysses"),
        (2, 1, 2, 2, 1, "strict, pure Ulysses"),
        (2, 1, 1, 1, 1, "scatter_idx=2, gather_idx=1"),
        (2, 1, 1, 2, 2, "scatter_idx=2, gather_idx=1"),
    ],
)
def test_direct_npu_strategy_validates_initialized_groups(
    monkeypatch, ulysses_size, ring_size, allgather_size, scatter_idx, gather_idx, message
):
    monkeypatch.setattr(a2a_permute, "current_omni_platform", SimpleNamespace(is_npu=lambda: True))

    def unexpected():
        pytest.fail("Invalid topology/layout must be rejected before dependency initialization")

    monkeypatch.setattr(a2a_permute, "ensure_a2a_permute_available", unexpected)
    group = SimpleNamespace(
        ulysses_group=object(),
        ulysses_world_size=ulysses_size,
        ring_world_size=ring_size,
        allgather_world_size=allgather_size,
    )
    with pytest.raises(ValueError, match=message):
        ulysses.UlyssesParallelAttention(group, scatter_idx, gather_idx, False, ulysses_a2a_permute=True)


def test_npu_strategy_checks_each_forward_context_mode(monkeypatch):
    monkeypatch.setattr(a2a_permute, "current_omni_platform", SimpleNamespace(is_npu=lambda: True))
    monkeypatch.setattr(a2a_permute, "ensure_a2a_permute_available", lambda: None)
    group = SimpleNamespace(ulysses_group=object(), ulysses_world_size=2, ring_world_size=1, allgather_world_size=1)
    monkeypatch.setattr(factory, "get_sp_group", lambda: group)
    monkeypatch.setattr(factory, "get_sequence_parallel_world_size", lambda: 2)
    config = SimpleNamespace(
        enforce_eager=True,
        parallel_config=SimpleNamespace(ulysses_a2a_permute=True, ulysses_degree=2, ulysses_mode="strict"),
    )
    with set_forward_context(omni_diffusion_config=config):
        strategy = factory.build_parallel_attention_strategy(scatter_idx=2, gather_idx=1, use_sync=False)
    config.parallel_config.ulysses_mode = "advanced_uaa"
    tensor = torch.empty(1, 4, 2, 16)
    with set_forward_context(omni_diffusion_config=config):
        with pytest.raises(ValueError, match="ulysses_mode='strict'"):
            strategy.pre_attention(tensor, tensor, tensor, None)


def test_cuda_backend_does_not_require_eager(monkeypatch):
    config = SimpleNamespace(
        parallel_config=SimpleNamespace(ulysses_a2a_permute=True, ulysses_degree=2), enforce_eager=False
    )
    monkeypatch.setattr(a2a_permute, "current_omni_platform", SimpleNamespace(is_npu=lambda: False))
    monkeypatch.setattr(factory, "is_forward_context_available", lambda: True)
    monkeypatch.setattr(factory, "get_forward_context", lambda: SimpleNamespace(omni_diffusion_config=config))
    monkeypatch.setattr(factory, "get_sp_group", lambda: object())
    monkeypatch.setattr(factory, "get_sequence_parallel_world_size", lambda: 2)
    monkeypatch.setattr(factory, "UlyssesParallelAttention", lambda **kwargs: kwargs)
    result = factory.build_parallel_attention_strategy(scatter_idx=2, gather_idx=1, use_sync=False)
    assert result["ulysses_a2a_permute"] is True


def test_disabled_optimization_does_not_load_backend(monkeypatch):
    def unexpected():
        pytest.fail("Disabled optimization must not initialize a transport")

    monkeypatch.setattr(a2a_permute, "ensure_a2a_permute_available", unexpected)
    group = SimpleNamespace(ulysses_group=object(), ulysses_world_size=2, ring_world_size=1)
    strategy = ulysses.UlyssesParallelAttention(group, 2, 1, False)
    assert strategy.name == "ulysses"
