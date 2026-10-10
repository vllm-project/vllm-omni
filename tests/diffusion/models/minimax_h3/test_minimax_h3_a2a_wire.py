# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""The H3 all-to-all transports over a real Ulysses exchange, checked against the stock path.

Three arms, each making a different claim:
  * ``bf16``       - the installed default must be BIT-IDENTICAL to the stock exchange, and
                     nothing quantised may run.
  * ``int8``       - the quantised exchange must actually run (from the transport's own
                     counter, not inferred from the output) and stay inside tolerance per row.
  * ``int8-fused`` - as ``int8``, plus the fused kernels must be the ones that ran, and their
                     re-indexing must be bit-identical to the layout-preserving int8 path.

The counters matter: the reverse/o direction and the 5D directions always use the per-tensor
exchange, so a total call count above zero says nothing about whether the fused path ran at all.
"""

from __future__ import annotations

import os
from typing import Literal

import pytest
import torch

from tests.helpers.mark import hardware_marks
from vllm_omni.diffusion.distributed.parallel_state import (
    destroy_distributed_env,
    get_sp_group,
    init_distributed_environment,
    initialize_model_parallel,
)
from vllm_omni.platforms import current_omni_platform

# The collective exercises world_size 2 and 4 over real NCCL, so it needs four cards.
_FOUR_CARD = hardware_marks(res={"cuda": "H100"}, num_cards=4)

pytestmark = [*_FOUR_CARD, pytest.mark.core_model, pytest.mark.diffusion]

DeviceKind = Literal["cpu", "cuda"]

# Per-row relative error allowed for the int8 (UE5M3) packet. The codec's own single-process
# round trip measures ~4-6e-3; the exchange only moves bytes.
_MAX_REL_ERROR = 2e-2


def _require_gpus(world_size: int) -> None:
    available = current_omni_platform.get_device_count()
    if available < world_size:
        pytest.skip(f"Test requires {world_size} GPUs, found {available}")


def _per_row_rel_error(got: torch.Tensor, reference: torch.Tensor) -> float:
    """Worst row error relative to that row's own magnitude.

    A single global max would let a localised corruption hide underneath a large row
    elsewhere, so each row is normalised by itself.
    """
    diff = (got.float() - reference.float()).abs().flatten(1).amax(dim=1)
    scale = reference.float().abs().flatten(1).amax(dim=1).clamp_min(1e-6)
    return (diff / scale).max().item()


def _run_wire_exchange(local_rank: int, world_size: int, mode: str, master_port: int) -> None:
    os.environ.update(
        {
            "RANK": str(local_rank),
            "LOCAL_RANK": str(local_rank),
            "WORLD_SIZE": str(world_size),
            "MASTER_ADDR": "localhost",
            "MASTER_PORT": str(master_port),
            "H3_A2A_WIRE": mode,
        }
    )
    from vllm_omni.diffusion.distributed.comm import all_to_all_4D, register_seq_all_to_all_backend
    from vllm_omni.diffusion.models.minimax_h3 import a2a_wire

    device = torch.device(f"{current_omni_platform.device_type}:{local_rank}")
    current_omni_platform.set_device(device)
    init_distributed_environment()
    initialize_model_parallel(ulysses_degree=world_size)
    sp_group = get_sp_group().ulysses_group

    # The transport requires the DiT's head dim (VECTOR == 128) and equal head splits.
    batch, seq_per_rank, heads, head_size = 2, 16, 4 * world_size, 128
    torch.manual_seed(11 + local_rank)
    x = torch.randn(batch, seq_per_rank, heads, head_size, dtype=torch.bfloat16, device=device)

    try:
        # Reference: the stock exchange, with no transport registered.
        register_seq_all_to_all_backend()
        reference = all_to_all_4D(x, 2, 1, group=sp_group)

        # Install exactly the way the model does and run the same exchange.
        a2a_wire.install()
        got = all_to_all_4D(x, 2, 1, group=sp_group)
        stats = a2a_wire.stats()

        if mode == "bf16":
            assert torch.equal(got, reference), "the installed bf16 transport is not identical to stock"
            assert stats["calls"] == 0 and stats["fused_calls"] == 0, f"bf16 must not quantise: {stats}"
            return

        assert not stats["failed"], f"the wire fell back instead of running: {stats}"
        if mode == "int8":
            assert stats["calls"] > 0, f"the int8 exchange never ran: {stats}"
        else:
            # The fused whole-call override skips the layout transforms, so it is the only
            # thing that can have produced this result; `calls` cannot tell them apart.
            assert stats["fused_calls"] > 0, f"the fused path never ran: {stats}"

        assert got.shape == reference.shape
        rel = _per_row_rel_error(got, reference)
        assert rel < _MAX_REL_ERROR, f"{mode} per-row relative error {rel:.3e} exceeds {_MAX_REL_ERROR:.1e}"

        if mode == "int8-fused":
            # Compare against the layout-preserving int8 path. Both are quantised, so this is
            # independent of the codec's tolerance and fails if the fused row mapping is wrong.
            register_seq_all_to_all_backend(exchange=lambda t, g: a2a_wire.quant_all_to_all(t, g, world_size))
            plain = all_to_all_4D(x, 2, 1, group=sp_group)
            assert torch.equal(got, plain), "the fused re-indexing differs from the layout-preserving path"
    finally:
        register_seq_all_to_all_backend()
        destroy_distributed_env()


@pytest.mark.parallel
@pytest.mark.parametrize("mode", ["bf16", "int8", "int8-fused"])
@pytest.mark.parametrize("world_size", [2, 4])
def test_wire_exchange_matches_stock(world_size: int, mode: str) -> None:
    _require_gpus(world_size)
    torch.multiprocessing.spawn(
        _run_wire_exchange,
        args=(world_size, mode, 29641),
        nprocs=world_size,
    )
