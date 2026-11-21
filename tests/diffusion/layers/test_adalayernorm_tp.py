# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Tensor-parallel AdaLN modulation projections.

Unquantized modulation linears are column-parallel with their output gathered:
each rank keeps its rows of the checkpoint weight, and the gathered output is
the replicated layer's output.
"""

import os

import pytest
import torch
import torch.nn.functional as F
from vllm.model_executor import parameter
from vllm.model_executor.layers import linear
from vllm.model_executor.layers.utils import default_unquantized_gemm

from tests.helpers.distributed import configure_cpu_collective_worker, start_collective_workers
from tests.helpers.runtime import get_open_port
from vllm_omni.diffusion.layers import adalayernorm
from vllm_omni.diffusion.layers.adalayernorm import (
    AdaLayerNormContinuous,
    AdaLayerNormZero,
    AdaLayerNormZeroSingle,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

_DIM = 16
# (constructor, modulation width as a multiple of _DIM)
_NORMS = [
    (lambda: AdaLayerNormZero(_DIM, prefix="norm1"), 6),
    (lambda: AdaLayerNormZeroSingle(_DIM, prefix="norm"), 3),
    (lambda: AdaLayerNormContinuous(_DIM, _DIM, prefix="norm_out"), 2),
]


def _set_tp(monkeypatch, tp: int, rank: int) -> None:
    # Only the distributed metadata is substituted; constructors and loaders are vLLM's.
    for module in (linear, parameter):
        monkeypatch.setattr(module, "get_tensor_model_parallel_world_size", lambda: tp)
        monkeypatch.setattr(module, "get_tensor_model_parallel_rank", lambda: rank)
    monkeypatch.setattr(linear, "dispatch_unquantized_gemm", lambda *a, **kw: default_unquantized_gemm)


@pytest.mark.parametrize(("make_norm", "width"), _NORMS, ids=["zero", "zero_single", "continuous"])
@pytest.mark.parametrize("tp", [1, 2, 4])
def test_rank_rows_and_local_outputs_rebuild_the_replicated_projection(monkeypatch, make_norm, width, tp):
    weight = torch.randn(width * _DIM, _DIM)
    bias = torch.randn(width * _DIM)
    x = torch.randn(3, _DIM)
    local_outputs = []
    for rank in range(tp):
        _set_tp(monkeypatch, tp, rank)
        projection = make_norm().linear
        assert isinstance(projection, linear.ColumnParallelLinear)
        assert projection.gather_output
        projection.weight.weight_loader(projection.weight, weight)
        projection.bias.weight_loader(projection.bias, bias)
        torch.testing.assert_close(projection.weight.data, weight.chunk(tp, dim=0)[rank], rtol=0, atol=0)
        local_outputs.append(projection.quant_method.apply(projection, x, projection.bias))

    # The all-gather concatenates the ranks' outputs in rank order.
    torch.testing.assert_close(torch.cat(local_outputs, dim=-1), F.linear(x, weight, bias), rtol=0, atol=0)


@pytest.mark.parametrize("tp", [1, 2])
def test_lora_on_sharded_modulation_projection(monkeypatch, tp):
    """LoRA binds to the column-parallel projection and applies each rank's rows of the delta."""
    from vllm.config.lora import LoRAConfig

    from vllm_omni.lora.binding import from_layer

    rank_dim = 8
    weight, bias = torch.randn(6 * _DIM, _DIM), torch.randn(6 * _DIM)
    lora_a, lora_b = torch.randn(rank_dim, _DIM), torch.randn(6 * _DIM, rank_dim)
    x = torch.randn(3, _DIM)
    local_outputs = []
    for rank in range(tp):
        _set_tp(monkeypatch, tp, rank)
        projection = AdaLayerNormZero(_DIM, prefix="norm1").linear
        projection.weight.weight_loader(projection.weight, weight)
        projection.bias.weight_loader(projection.bias, bias)
        wrapped = from_layer(
            layer=projection,
            max_loras=1,
            lora_config=LoRAConfig(max_lora_rank=rank_dim, max_loras=1, lora_dtype=torch.float32),
            packed_modules_list=[],
        )
        assert isinstance(wrapped.base_layer, linear.ColumnParallelLinear)
        wrapped.set_lora(0, lora_a, lora_b)
        local_outputs.append(wrapped.apply(x, projection.bias))

    torch.testing.assert_close(torch.cat(local_outputs, dim=-1), F.linear(x, weight + lora_b @ lora_a, bias))


def test_quantized_modulation_projection_stays_replicated(monkeypatch):
    built: dict[str, object] = {}

    def replicated(input_size, output_size, **kwargs):
        built.update(input_size=input_size, output_size=output_size, **kwargs)
        return torch.nn.Identity()

    monkeypatch.setattr(adalayernorm, "ReplicatedLinear", replicated)
    quant_config = object()

    AdaLayerNormZero(_DIM, quant_config=quant_config, prefix="norm1")

    assert built["quant_config"] is quant_config
    assert (built["input_size"], built["output_size"], built["prefix"]) == (_DIM, 6 * _DIM, "norm1.linear")


def _gathered_forward_worker(local_rank: int, world_size: int, master_port: int) -> None:
    os.environ.update(
        {
            "RANK": str(local_rank),
            "LOCAL_RANK": str(local_rank),
            "WORLD_SIZE": str(world_size),
            "MASTER_ADDR": "localhost",
            "MASTER_PORT": str(master_port),
        }
    )
    configure_cpu_collective_worker()
    from vllm_omni.diffusion.distributed.parallel_state import (
        destroy_distributed_env,
        init_distributed_environment,
        initialize_model_parallel,
    )

    init_distributed_environment(backend="gloo")
    initialize_model_parallel(tensor_parallel_size=world_size, backend="gloo")
    linear.dispatch_unquantized_gemm = lambda *a, **kw: default_unquantized_gemm
    try:
        for make_norm, width in _NORMS:
            generator = torch.Generator().manual_seed(width)
            weight = torch.randn(width * _DIM, _DIM, generator=generator)
            bias = torch.randn(width * _DIM, generator=generator)
            x = torch.randn(2, 5, _DIM, generator=generator)
            emb = torch.randn(2, _DIM, generator=generator)

            norm = make_norm()
            norm.linear.weight.weight_loader(norm.linear.weight, weight)
            norm.linear.bias.weight_loader(norm.linear.bias, bias)
            # The same layer, replicated: what the norm computed before sharding.
            replicated = make_norm()
            replicated.linear = torch.nn.Linear(_DIM, width * _DIM)
            replicated.linear.weight.data.copy_(weight)
            replicated.linear.bias.data.copy_(bias)

            with torch.no_grad():
                sharded_out, replicated_out = norm(x, emb), replicated(x, emb)
            for got, want in zip(
                sharded_out if isinstance(sharded_out, tuple) else (sharded_out,),
                replicated_out if isinstance(replicated_out, tuple) else (replicated_out,),
            ):
                torch.testing.assert_close(got, want, rtol=0, atol=0)
    finally:
        destroy_distributed_env()


@pytest.mark.parametrize("world_size", [2])
def test_gathered_forward_matches_replicated_norm_across_gloo_ranks(world_size):
    start_collective_workers(
        _gathered_forward_worker, args=(world_size, get_open_port()), nprocs=world_size, device_kind="cpu"
    )
