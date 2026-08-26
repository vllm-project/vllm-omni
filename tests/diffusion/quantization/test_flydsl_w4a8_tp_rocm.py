# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Tensor-parallel correctness for W4A8 SVDQuant on gfx950.

Two layers of testing:

* **Decomposition tests** (single GPU): simulate TP by manually splitting a
  layer's K (row-parallel) or N (column-parallel) and driving the fused
  ``flydsl_w4a8_svd_gemm`` per shard, then reduce/concatenate. This proves the
  linearity claim behind row-parallel SVD -- the per-rank partial
  ``d_p @ proj_up.T`` sums with the residual under the output all-reduce -- with
  no process group, so it runs on the dev box.

* **Multi-GPU test** (>= 2 GPUs): spawns a real TP group and runs a
  ``RowParallelLinear`` end to end. Skipped unless ``torch.accelerator.device_count()
  >= 2``; run it on a multi-GPU server. Set ``W4A8_TP_SIZE`` to override the
  world size (default 2).
"""

import os

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

M_RAGGED = 4680


def _cos(a: torch.Tensor, b: torch.Tensor) -> float:
    return torch.nn.functional.cosine_similarity(a.flatten().float(), b.flatten().float(), dim=0).item()


def _spectral_factors(out_features: int, in_features: int, rank: int, decay: float = 64.0):
    """A weight with a decaying spectrum plus its exact rank-R factors.

    Returns ``(W, proj_up (N, R), proj_down (R, K))`` with ``W = proj_up @ proj_down + residual``.
    """
    r = min(out_features, in_features)
    u, _ = torch.linalg.qr(torch.randn(out_features, r, device="cuda", dtype=torch.float32))
    v, _ = torch.linalg.qr(torch.randn(in_features, r, device="cuda", dtype=torch.float32))
    s = torch.exp(-torch.arange(r, device="cuda", dtype=torch.float32) / decay)
    w = (u * s) @ v.T
    scale = 0.1 / w.std()
    return w * scale, (u[:, :rank] * s[:rank] * scale).contiguous(), v[:, :rank].T.contiguous()


# ---------------------------------------------------------------------------
# Single-GPU decomposition tests (prove the sharding math)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("tp", [2, 4])
def test_svd_row_parallel_decomposition_matches_full(tp):
    """Row-parallel (K sharded): each rank runs the fused op on its (Wr_p, L1_p)
    shard with a *replicated* proj_up, and the sum of partials (the output
    all-reduce) equals the un-sharded fused result. K/tp is a multiple of 32, so
    per-shard MXFP8 activation quant matches the full one exactly."""
    from vllm_omni.quantization import flydsl_w4a8

    flydsl_w4a8.register_ops()
    in_features = out_features = 5120
    rank = 32
    torch.manual_seed(0)
    weight, proj_up, proj_down = _spectral_factors(out_features, in_features, rank)
    residual = (weight - proj_up @ proj_down).to(torch.bfloat16).contiguous()
    proj_up = proj_up.to(torch.bfloat16)  # replicated across ranks
    proj_down = proj_down.to(torch.bfloat16)
    x = torch.randn(M_RAGGED, in_features, device="cuda", dtype=torch.bfloat16) * 0.5

    kw, ks = flydsl_w4a8.pack_weight(residual)
    full = torch.ops.vllm_omni.flydsl_w4a8_svd_gemm(x, kw, ks, proj_down, proj_up, None, out_features)

    ksz = in_features // tp
    summed = torch.zeros(M_RAGGED, out_features, device="cuda", dtype=torch.float32)
    for p in range(tp):
        sl = slice(p * ksz, (p + 1) * ksz)
        kw_p, ks_p = flydsl_w4a8.pack_weight(residual[:, sl].contiguous())
        y_p = torch.ops.vllm_omni.flydsl_w4a8_svd_gemm(
            x[:, sl].contiguous(),
            kw_p,
            ks_p,
            proj_down[:, sl].contiguous(),  # L1 sharded on K
            proj_up,  # L2 replicated
            None,
            out_features,
        )
        summed += y_p.float()

    assert torch.isfinite(summed).all()
    assert _cos(summed, full) > 0.999


@pytest.mark.parametrize("tp", [2, 4])
def test_svd_column_parallel_decomposition_matches_full(tp):
    """Column-parallel (N sharded): each rank runs the fused op on its (Wr_n, L2_n)
    shard with a *replicated* proj_down, and concatenating the partials along N
    equals the un-sharded fused result."""
    from vllm_omni.quantization import flydsl_w4a8

    flydsl_w4a8.register_ops()
    in_features = out_features = 5120
    rank = 32
    torch.manual_seed(0)
    weight, proj_up, proj_down = _spectral_factors(out_features, in_features, rank)
    residual = (weight - proj_up @ proj_down).to(torch.bfloat16).contiguous()
    proj_up = proj_up.to(torch.bfloat16)
    proj_down = proj_down.to(torch.bfloat16)  # replicated across ranks
    x = torch.randn(M_RAGGED, in_features, device="cuda", dtype=torch.bfloat16) * 0.5

    kw, ks = flydsl_w4a8.pack_weight(residual)
    full = torch.ops.vllm_omni.flydsl_w4a8_svd_gemm(x, kw, ks, proj_down, proj_up, None, out_features)

    nsz = out_features // tp
    parts = []
    for p in range(tp):
        sl = slice(p * nsz, (p + 1) * nsz)
        kw_n, ks_n = flydsl_w4a8.pack_weight(residual[sl, :].contiguous())
        y_n = torch.ops.vllm_omni.flydsl_w4a8_svd_gemm(
            x,
            kw_n,
            ks_n,
            proj_down,  # L1 replicated
            proj_up[sl, :].contiguous(),  # L2 sharded on N
            None,
            nsz,
        )
        parts.append(y_n)

    cat = torch.cat(parts, dim=1)
    assert cat.shape == (M_RAGGED, out_features)
    assert _cos(cat, full) > 0.999


# ---------------------------------------------------------------------------
# Real multi-GPU test (run on a >= 2 GPU server)
# ---------------------------------------------------------------------------


def _row_parallel_tp_worker(rank: int, world_size: int, port: int, return_dict) -> None:
    """Build a RowParallelLinear (quark_svdquant + mxfp4_unshuffled) at the given
    TP rank, load a sharded checkpoint, run forward, and record the output.

    Runs in a spawned process. The full reference weight is reproduced identically
    on every rank from a fixed seed, decomposed + packed unshuffled, then loaded
    through the layer's own weight_loaders (which slice per rank on the K axis for
    row-parallel). ``process_weights_after_loading`` shuffles each rank's shard.
    """
    from vllm.config import VllmConfig, set_current_vllm_config
    from vllm.distributed import init_distributed_environment, initialize_model_parallel
    from vllm.model_executor.layers.linear import RowParallelLinear

    from vllm_omni.quantization import flydsl_w4a8
    from vllm_omni.quantization.quark_w4a8_config import DiffusionQuarkW4A8Config

    torch.accelerator.set_device_index(rank)
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    init_distributed_environment(world_size=world_size, rank=rank, local_rank=rank, backend="nccl")
    initialize_model_parallel(tensor_model_parallel_size=world_size)

    in_features = out_features = 5120
    svd_rank = 32
    torch.manual_seed(0)  # identical full weight on every rank
    weight, proj_up, proj_down = _spectral_factors(out_features, in_features, svd_rank)
    residual = (weight - proj_up @ proj_down).to(torch.bfloat16).contiguous()
    w_q, w_s = flydsl_w4a8.pack_weight_unshuffled(residual)  # full, natural-order

    cfg = DiffusionQuarkW4A8Config(
        svd_rank=svd_rank, is_checkpoint_w4a8_serialized=True, quark_export_format="mxfp4_unshuffled"
    )
    with set_current_vllm_config(VllmConfig()):
        layer = RowParallelLinear(
            input_size=in_features,
            output_size=out_features,
            bias=False,
            input_is_parallel=True,
            params_dtype=torch.bfloat16,
            quant_config=cfg,
            prefix="transformer.blocks.0.ffn.net_2",
        )

    # Load the full tensors through each param's weight_loader; RowParallelLinear
    # slices the input (K) axis for this rank. proj_up replicates, proj_down shards.
    layer.weight_packed.weight_loader(layer.weight_packed, w_q)
    layer.weight_scale.weight_loader(layer.weight_scale, w_s)
    layer.proj_up.weight_loader(layer.proj_up, proj_up.to(torch.bfloat16))
    layer.proj_down.weight_loader(layer.proj_down, proj_down.to(torch.bfloat16))
    layer.quant_method.process_weights_after_loading(layer)

    torch.manual_seed(1234)  # identical input on every rank
    x_full = torch.randn(M_RAGGED, in_features, device="cuda", dtype=torch.bfloat16) * 0.5
    ksz = in_features // world_size
    x_shard = x_full[:, rank * ksz : (rank + 1) * ksz].contiguous()

    out, _ = layer(x_shard)  # RowParallelLinear all-reduces internally
    if rank == 0:
        reference = torch.nn.functional.linear(x_full.float(), weight.float())
        return_dict["cos"] = _cos(out, reference)
        return_dict["finite"] = bool(torch.isfinite(out).all())
        return_dict["shape"] = tuple(out.shape)


@pytest.mark.skipif(
    torch.accelerator.device_count() < 2,
    reason="real TP test needs >= 2 GPUs; run on a multi-GPU server",
)
def test_svd_row_parallel_end_to_end_multigpu():
    """End-to-end row-parallel SVD through a real RowParallelLinear + TP group.

    Compares the reduced quantized output against the full-precision BF16 result;
    a broken all-reduce or wrongly sharded factor would produce garbage (cos far below
    the quant-error budget), so cos > 0.95 is a meaningful integration check.
    """
    import torch.multiprocessing as mp

    world_size = int(os.environ.get("W4A8_TP_SIZE", "2"))
    if torch.accelerator.device_count() < world_size:
        pytest.skip(f"need {world_size} GPUs, have {torch.accelerator.device_count()}")

    manager = mp.get_context("spawn").Manager()
    return_dict = manager.dict()
    mp.spawn(
        _row_parallel_tp_worker,
        args=(world_size, 29517, return_dict),
        nprocs=world_size,
        join=True,
    )

    assert return_dict.get("finite") is True
    assert return_dict.get("shape") == (M_RAGGED, 5120)
    assert return_dict.get("cos", 0.0) > 0.95, f"row-parallel TP cos={return_dict.get('cos')}"
