# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""The indexed modulation custom ops stay opaque under torch.compile and survive CUDA-graph replay.

Each op must appear as a single node in the Dynamo graph (no graph break, no decomposition), the Inductor-compiled
``fullgraph=True`` result must be bit-identical to eager, and a captured CUDA graph must reproduce eager outputs
after its static inputs are refilled.
"""

import pytest
import torch

from tests.helpers.mark import hardware_marks
from vllm_omni.diffusion.layers import indexed_modulation as im

pytestmark = [
    pytest.mark.core_model,
    pytest.mark.diffusion,
    *hardware_marks(res={"cuda": "L4"}, num_cards=1),
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required"),
]

_EPS = 1e-6
_OPS = (
    torch.ops.vllm_omni.indexed_gate,
    torch.ops.vllm_omni.rms_norm_indexed_scale_shift,
    torch.ops.vllm_omni.indexed_gate_rms_norm_scale_shift,
)


def _inputs(rows: int, hidden: int, table_dtype: torch.dtype, seed: int = 0):
    g = torch.Generator(device="cuda").manual_seed(seed)
    bf = dict(device="cuda", dtype=torch.bfloat16)
    residual = torch.randn(rows, hidden, generator=g, **bf)
    branch = torch.randn(rows, hidden, generator=g, **bf)
    tables = (0.3 * torch.randn(12, 3 * hidden, generator=g, device="cuda")).to(table_dtype)
    weight = (1 + 0.1 * torch.randn(hidden, generator=g, device="cuda")).to(torch.bfloat16)
    indices = torch.randint(0, 12, (rows,), generator=g, device="cuda")
    return residual, branch, tables, weight, indices


def _block(residual, branch, tables, weight, indices):
    hidden = residual.shape[-1]
    gate, shift, scale = tables[:, :hidden], tables[:, hidden : 2 * hidden], tables[:, 2 * hidden :]
    gated = im.indexed_gate(residual, gate, branch, indices)
    normed = im.rms_norm_indexed_scale_shift(gated, weight, shift, scale, indices, _EPS)
    fused_residual, fused_normed = im.indexed_gate_rms_norm_scale_shift(
        residual, gate, branch, weight, shift, scale, indices, _EPS
    )
    return gated, normed, fused_residual, fused_normed


def _assert_equal(actual, expected) -> None:
    for a, e in zip(actual, expected, strict=True):
        assert torch.equal(a, e)


def test_ops_are_single_graph_nodes() -> None:
    graphs = []

    def backend(gm, example_inputs):
        graphs.append(gm)
        return gm.forward

    torch._dynamo.reset()
    args = _inputs(64, 3072, torch.float32)
    _assert_equal(torch.compile(_block, backend=backend, fullgraph=True)(*args), _block(*args))
    assert len(graphs) == 1
    # Dynamo records the overload packet or its default overload; compare packets.
    targets = [
        getattr(node.target, "overloadpacket", node.target)
        for node in graphs[0].graph.nodes
        if node.op == "call_function"
    ]
    for op in _OPS:
        assert targets.count(op) == 1, f"{op} not opaque in {targets}"


@pytest.mark.parametrize("table_dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize(("rows", "hidden"), [(1, 5376), (777, 5376), (33, 1000)])
def test_inductor_fullgraph_matches_eager(table_dtype: torch.dtype, rows: int, hidden: int) -> None:
    torch._dynamo.reset()
    compiled = torch.compile(_block, fullgraph=True, dynamic=False)
    args = _inputs(rows, hidden, table_dtype)
    _assert_equal(compiled(*args), _block(*args))


@pytest.mark.parametrize("compiled", [False, True], ids=["eager", "inductor"])
def test_cuda_graph_replay_matches_eager(compiled: bool) -> None:
    torch._dynamo.reset()
    fn = torch.compile(_block, fullgraph=True, dynamic=False) if compiled else _block
    static = _inputs(256, 5376, torch.float32, seed=0)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            fn(*static)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        outputs = fn(*static)
    for seed in (1, 2):
        fresh = _inputs(256, 5376, torch.float32, seed=seed)
        for dst, src in zip(static, fresh, strict=True):
            dst.copy_(src)
        graph.replay()
        torch.accelerator.synchronize()
        _assert_equal(outputs, _block(*fresh))
