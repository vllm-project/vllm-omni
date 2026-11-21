# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Serving layout and collective-group contracts; real NPU gates are separate."""

import importlib.util
import sys
from datetime import timedelta
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def load_module():
    path = Path(__file__).resolve().parents[4] / "vllm_omni/diffusion/models/minimax_h3/vision_parallel.py"
    spec = importlib.util.spec_from_file_location("service_parallel_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _collective_worker(rank, rendezvous):
    """Exercise actual CPU collectives without loading the serving runtime."""
    dist.init_process_group("gloo", init_method=rendezvous, rank=rank, world_size=2, timeout=timedelta(seconds=30))
    try:
        module = load_module()
        for lengths in ((12, 4, 20), (4,), (0, 0)):
            plan = module.build_vision_plan(lengths, 2, num_heads=4)
            full = torch.arange(sum(lengths) * 8, dtype=torch.float32).reshape(sum(lengths), 4, 2)
            local = full[list(plan.rank_token_indices[rank])]
            transport = module.VisionAllToAll(plan, group=dist.group.WORLD)
            heads = transport.to_heads(local)
            torch.testing.assert_close(heads, full[:, rank * 2 : (rank + 1) * 2], rtol=0, atol=0)
            restored = transport.to_tokens(heads)
            torch.testing.assert_close(restored, local, rtol=0, atol=0)
            assert transport.calls == 2
            final = torch.arange(sum(lengths) // 4 * 6, dtype=torch.float32).reshape(sum(lengths) // 4, 6)
            ids = list(plan.rank_merger_indices[rank])
            gathered, deepstack = module.gather_vision_outputs(
                (final[ids], [(final + 100)[ids], (final + 200)[ids]]), plan, group=dist.group.WORLD
            )
            torch.testing.assert_close(gathered, final, rtol=0, atol=0)
            for actual, offset in zip(deepstack, (100, 200), strict=True):
                torch.testing.assert_close(actual, final + offset, rtol=0, atol=0)
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(not dist.is_available() or not dist.is_gloo_available(), reason="Gloo is required")
def test_real_collectives_preserve_uneven_empty_and_deepstack_layouts(tmp_path):
    mp.spawn(_collective_worker, args=((tmp_path / "vision-rendezvous").as_uri(),), nprocs=2, join=True)


def test_transport_uses_explicit_encoder_group(monkeypatch):
    m = load_module()
    group = object()
    monkeypatch.setattr(m.dist, "get_rank", lambda group=None: 1 if group is not None else 9)
    monkeypatch.setattr(m.dist, "get_world_size", lambda group=None: 2 if group is not None else 16)
    calls = []

    def exchange(out, value, *, group=None):
        calls.append(group)
        out.copy_(value)

    monkeypatch.setattr(m.dist, "all_to_all_single", exchange)
    plan = m.build_vision_plan((8, 12), 2)
    transport = m.VisionAllToAll(plan, group=group)
    assert (transport.rank, transport.size) == (1, 2)
    transport.exchange(torch.ones(2, 2))
    assert calls == [group]


def test_full_output_gather_restores_global_merger_order(monkeypatch):
    m = load_module()
    group = object()
    plan = m.build_vision_plan((12, 4, 20), 2)
    full = torch.arange(4 * 9 * 3).reshape(4, 9, 3).float()
    local_ids = plan.rank_merger_indices[0]
    local = full[:, list(local_ids)]
    seen = []

    def gather(received, value, *, group=None):
        seen.append(group)
        for rank, target in enumerate(received):
            ids = plan.rank_merger_indices[rank]
            target.zero_()
            target[:, : len(ids)] = full[:, list(ids)]

    monkeypatch.setattr(m.dist, "all_gather", gather)
    final, deep = m.gather_vision_outputs((local[0], list(local[1:])), plan, group=group)
    assert seen == [group]
    assert torch.equal(torch.stack([final, *deep]), full)


@pytest.mark.parametrize("value", [0, 2, 4, 16, -1, "8", "yes", True, None])
def test_service_rejects_unsupported_sp_size(mocker, value):
    m = load_module()
    with pytest.raises(ValueError):
        m.resolve_vision_sp_group(value, mocker.Mock(world_size=8, device_group=object()))


def test_service_is_opt_in_and_requires_exact_tp8(mocker):
    m = load_module()
    assert m.resolve_vision_sp_group(1, None) is None
    group = mocker.Mock(world_size=8, device_group=object())
    assert m.resolve_vision_sp_group(1, group) is None
    assert m.resolve_vision_sp_group(8, group) is group.device_group
    with pytest.raises(ValueError):
        m.resolve_vision_sp_group(8, mocker.Mock(world_size=4))
    with pytest.raises(ValueError):
        m.resolve_vision_sp_group(8, None)


def test_empty_rank_shard_keeps_explicit_hidden_dimension(mocker):
    # Exercise the real attention body without importing the serving dependency tree.
    import ast

    path = Path(__file__).resolve().parents[4] / "vllm_omni/diffusion/models/minimax_h3/vision_sp.py"
    tree = ast.parse(path.read_text())
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "attention_sp8")
    aligned = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "row_aligned_call")
    namespace = {"torch": torch, "segment_attention": lambda q, k, v: v}
    future = ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)
    module = ast.fix_missing_locations(ast.Module(body=[future, aligned, function], type_ignores=[]))
    exec(compile(module, str(path), "exec"), namespace)
    attention = mocker.Mock(num_heads=16, head_dim=2, qkv=lambda x: x.new_empty((x.shape[0], 96)), proj=lambda x: x)
    transport = mocker.Mock(
        plan=mocker.Mock(segment_lengths=(4,)),
        to_heads=lambda x: x.new_zeros((4, 2, 2)),
        to_tokens=lambda x: x.new_empty((0, 16, 2)),
        row_layout=lambda device, merged=False: (torch.empty(0, dtype=torch.long, device=device), 4),
    )
    hidden = torch.empty(0, 32)
    result = namespace["attention_sp8"](attention, hidden, (None, None), transport, lambda q, k, c, s: (q, k))
    assert result.shape == (0, 32)


def load_row_aligned_call():
    import ast

    path = Path(__file__).resolve().parents[4] / "vllm_omni/diffusion/models/minimax_h3/vision_sp.py"
    tree = ast.parse(path.read_text())
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "row_aligned_call")
    future = ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)
    namespace = {"torch": torch}
    exec(
        compile(ast.fix_missing_locations(ast.Module(body=[future, function], type_ignores=[])), str(path), "exec"),
        namespace,
    )
    return namespace["row_aligned_call"]


@pytest.mark.parametrize("rank", [0, 1, 7])
@pytest.mark.parametrize("lengths", [(4,), (12, 4, 20), (0, 0)])
@pytest.mark.parametrize("merged", [False, True])
def test_original_row_layout_is_cached_metadata_only(monkeypatch, rank, lengths, merged):
    module = load_module()
    monkeypatch.setattr(module.dist, "get_rank", lambda group=None: rank)
    monkeypatch.setattr(module.dist, "get_world_size", lambda group=None: 8)
    plan = module.build_vision_plan(lengths, 8)
    transport = module.VisionAllToAll(plan)
    positions, count = transport.row_layout(torch.device("cpu"), merged=merged)
    rows = plan.rank_merger_indices if merged else plan.rank_token_indices
    assert positions.tolist() == list(rows[rank])
    assert count == sum(map(len, rows))
    assert positions.dtype == torch.long
    assert transport.row_layout(torch.device("cpu"), merged=merged)[0] is positions
    assert transport.calls == 0 and transport.sent_bytes == 0


@pytest.mark.parametrize("rank", [0, 1, 7])
@pytest.mark.parametrize("merged", [False, True])
def test_row_aligned_kernel_retains_original_shape_and_positions(monkeypatch, rank, merged):
    module = load_module()
    monkeypatch.setattr(module.dist, "get_rank", lambda group=None: rank)
    monkeypatch.setattr(module.dist, "get_world_size", lambda group=None: 8)
    transport = module.VisionAllToAll(module.build_vision_plan((12, 4, 20), 8))
    indices, count = transport.row_layout(torch.device("cpu"), merged=merged)
    full = torch.arange(count * 3).reshape(count, 3).float()

    class LayoutSensitiveKernel(torch.nn.Module):
        # Emulate backend dependence on outer shape and row position, rather
        # than assuming a CPU Linear reproduces the NPU rounding failure.
        def forward(self, value):
            row = torch.arange(value.shape[0], dtype=value.dtype)[:, None]
            return value.sum(dim=-1, keepdim=True) + row + value.shape[0] * 0.25

    kernel = LayoutSensitiveKernel()
    expected = kernel(full).index_select(0, indices)
    local = full.index_select(0, indices)
    actual = load_row_aligned_call()(kernel, local, transport, merged=merged)
    assert torch.equal(actual.view(torch.uint8), expected.view(torch.uint8))
    if indices.numel():
        assert not torch.equal(kernel(local), expected)


def test_row_aligned_input_rejects_mismatched_layout(monkeypatch):
    module = load_module()
    monkeypatch.setattr(module.dist, "get_rank", lambda group=None: 0)
    monkeypatch.setattr(module.dist, "get_world_size", lambda group=None: 8)
    transport = module.VisionAllToAll(module.build_vision_plan((32,), 8))
    with pytest.raises(ValueError, match="shard layout"):
        load_row_aligned_call()(torch.nn.Identity(), torch.zeros(3, 4), transport)


def load_segment_attention():
    # Load the real short-segment body without importing the serving dependency tree.
    import ast

    path = Path(__file__).resolve().parents[4] / "vllm_omni/diffusion/models/minimax_h3/vision_sp.py"
    tree = ast.parse(path.read_text())
    nodes = [
        node
        for node in tree.body
        if (isinstance(node, ast.FunctionDef) and node.name == "segment_attention")
        or (isinstance(node, ast.Assign) and getattr(node.targets[0], "id", None) == "_MIN_SHORT_SEGMENT_HEADS")
    ]
    namespace = {"torch": torch, "F": torch.nn.functional}
    future = ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)
    exec(
        compile(ast.fix_missing_locations(ast.Module(body=[future, *nodes], type_ignores=[])), str(path), "exec"),
        namespace,
    )
    return namespace["segment_attention"]


@pytest.mark.parametrize("heads", [1, 2, 4])
@pytest.mark.parametrize("seq_len", [7, 1024])
def test_short_segment_head_tiling_preserves_values(heads, seq_len):
    generator = torch.Generator().manual_seed(20261118)
    q, k, v = (torch.randn(1, heads, seq_len, 16, generator=generator, dtype=torch.float32) for _ in range(3))
    expected = torch.nn.functional.scaled_dot_product_attention(q, k, v, dropout_p=0.0, is_causal=False)
    actual = load_segment_attention()(q, k, v)
    assert actual.shape == expected.shape
    assert torch.equal(actual.contiguous().view(torch.uint8), expected.contiguous().view(torch.uint8))
