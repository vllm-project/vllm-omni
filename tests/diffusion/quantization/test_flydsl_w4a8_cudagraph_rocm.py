# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""CUDA-graph capture/replay of the W4A8 fused ops on gfx950.

The FlyDSL A8W4 GEMM must be capturable so the Wan transformer can be replayed
per denoising step. With the default heuristic config (``QUARK_A8W4_AUTOTUNE``
unset) the op does no host sync on the hot path and the JIT compile is host-side,
so after a warm-up capture succeeds and replay is bit-identical to eager. The
opt-in timing autotune (``QUARK_A8W4_AUTOTUNE=1``, which synchronizes the device)
would abort capture -- these tests assert it is off and that replay is exact.

See ``/workspace/vllm_omni_svd/cuda_graph_w4a8_instructions.md`` for the full
recipe and how to wire a per-shape graph runner around the Wan core.
"""

import pytest
import torch

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.rocm, pytest.mark.MI355]


def _gcn_arch() -> str:
    if not (torch.cuda.is_available() and torch.version.hip):
        return ""
    return torch.cuda.get_device_properties(torch.accelerator.current_device_index()).gcnArchName


requires_gfx950 = pytest.mark.skipif(
    "gfx950" not in _gcn_arch(),
    reason=f"W4A8 requires CDNA4 scaled MFMA (gfx950); detected {_gcn_arch() or 'no ROCm device'}",
)
pytestmark.append(requires_gfx950)


@pytest.fixture(autouse=True)
def _cuda_default_device_no_autotune(monkeypatch):
    # The timing autotune syncs and would abort capture; keep the heuristic path.
    monkeypatch.delenv("QUARK_A8W4_AUTOTUNE", raising=False)
    torch.set_default_device("cuda")
    yield
    torch.set_default_device("cpu")


def _capture(call):
    """Warm up on a side stream (fills the compile caches), then capture ``call``.

    Returns the captured graph and its static output tensor. ``call`` reads from
    the caller's static input buffers, so replay = copy new input in + graph.replay().
    """
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        for _ in range(3):
            call()
    torch.cuda.current_stream().wait_stream(side)
    torch.accelerator.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        static_out = call()
    return graph, static_out


def test_plain_w4a8_cudagraph_replay_is_exact():
    from vllm_omni.quantization import flydsl_w4a8

    flydsl_w4a8.register_ops()
    n = k = 5120
    m = 4736  # tile-aligned
    torch.manual_seed(0)
    weight = (torch.randn(n, k, dtype=torch.bfloat16) * 0.1).contiguous()
    kernel_weight, kernel_scale = flydsl_w4a8.pack_weight(weight)
    bias = (torch.randn(n, dtype=torch.bfloat16) * 0.05).contiguous()
    static_x = (torch.randn(m, k, dtype=torch.bfloat16) * 0.5).contiguous()

    op = torch.ops.vllm_omni.flydsl_w4a8_gemm
    graph, static_out = _capture(lambda: op(static_x, kernel_weight, kernel_scale, bias, n))

    for seed in (1, 2):
        torch.manual_seed(seed)
        new_x = (torch.randn(m, k, dtype=torch.bfloat16) * 0.5).contiguous()
        static_x.copy_(new_x)
        graph.replay()
        torch.accelerator.synchronize()
        eager = op(new_x, kernel_weight, kernel_scale, bias, n)
        # Replay executes the exact recorded kernel, so it must match eager bit-for-bit.
        assert torch.equal(static_out, eager)


def test_svdquant_w4a8_cudagraph_replay_is_exact():
    """The SVD op additionally computes ``d = x @ proj_down.T`` in torch and hits
    the ragged-M padding path -- both must be capture-safe."""
    from vllm_omni.quantization import flydsl_w4a8

    flydsl_w4a8.register_ops()
    n = k = 5120
    rank = 32
    m = 4680  # ragged M -> exercises the padding path under capture
    torch.manual_seed(0)
    weight = (torch.randn(n, k, dtype=torch.bfloat16) * 0.1).contiguous()
    kernel_weight, kernel_scale = flydsl_w4a8.pack_weight(weight)
    proj_down = (torch.randn(rank, k, dtype=torch.bfloat16) * 0.05).contiguous()
    proj_up = (torch.randn(n, rank, dtype=torch.bfloat16) * 0.05).contiguous()
    static_x = (torch.randn(m, k, dtype=torch.bfloat16) * 0.5).contiguous()

    op = torch.ops.vllm_omni.flydsl_w4a8_svd_gemm
    graph, static_out = _capture(lambda: op(static_x, kernel_weight, kernel_scale, proj_down, proj_up, None, n))

    for seed in (1, 2):
        torch.manual_seed(seed)
        new_x = (torch.randn(m, k, dtype=torch.bfloat16) * 0.5).contiguous()
        static_x.copy_(new_x)
        graph.replay()
        torch.accelerator.synchronize()
        eager = op(new_x, kernel_weight, kernel_scale, proj_down, proj_up, None, n)
        assert torch.equal(static_out, eager)
