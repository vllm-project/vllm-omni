# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import pytest
import torch
from vllm.model_executor.layers.linear import (
    ColumnParallelLinear,
    QKVParallelLinear,
    RowParallelLinear,
)

from vllm_omni.quantization import svdquant_config as svdquant

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]


def _linear_without_init(linear_type: type[torch.nn.Module], tp_rank: int):
    layer = object.__new__(linear_type)
    torch.nn.Module.__init__(layer)
    layer.tp_rank = tp_rank
    return layer


def _values(shape: tuple[int, ...], dtype: torch.dtype) -> torch.Tensor:
    values = torch.arange(torch.Size(shape).numel(), dtype=torch.int64).remainder(127)
    return values.reshape(shape).to(dtype)


def _method(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(svdquant, "_assert_supported", lambda: None)
    return svdquant.DiffusionSVDQuantLinearMethod(svdquant.DiffusionSVDQuantConfig(rank=8))


def test_column_parallel_loader_shards_svdquant_outputs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    layer = _linear_without_init(ColumnParallelLinear, tp_rank=1)
    method = _method(monkeypatch)
    method.create_weights(
        layer,
        input_size_per_partition=128,
        output_partition_sizes=[32],
        input_size=128,
        output_size=64,
        params_dtype=torch.bfloat16,
        weight_loader=layer.weight_loader,
    )

    loaded_tensors = {
        "qweight": _values((64, 64), torch.int8),
        "wscales": _values((8, 64), torch.float8_e4m3fn),
        "proj_up": _values((64, 8), torch.bfloat16),
        "wcscales": _values((64,), torch.bfloat16),
    }
    for name, loaded in loaded_tensors.items():
        parameter = getattr(layer, name)
        parameter.weight_loader(parameter, loaded)
        expected = loaded.narrow(parameter.output_dim, 32, 32)
        torch.testing.assert_close(parameter, expected)


def test_row_parallel_loader_shards_svdquant_inputs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    layer = _linear_without_init(RowParallelLinear, tp_rank=1)
    method = _method(monkeypatch)
    method.create_weights(
        layer,
        input_size_per_partition=64,
        output_partition_sizes=[64],
        input_size=128,
        output_size=64,
        params_dtype=torch.bfloat16,
        weight_loader=layer.weight_loader,
    )

    loaded_tensors = {
        "qweight": _values((64, 64), torch.int8),
        "wscales": _values((8, 64), torch.float8_e4m3fn),
        "proj_down": _values((128, 8), torch.bfloat16),
        "smooth_factor": _values((128,), torch.bfloat16),
    }
    for name, loaded in loaded_tensors.items():
        parameter = getattr(layer, name)
        parameter.weight_loader(parameter, loaded)
        shard_size = parameter.shape[parameter.input_dim]
        expected = loaded.narrow(parameter.input_dim, shard_size, shard_size)
        torch.testing.assert_close(parameter, expected)


def test_qkv_parallel_loader_shards_fused_svdquant_outputs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    layer = _linear_without_init(QKVParallelLinear, tp_rank=1)
    layer.total_num_heads = 4
    layer.total_num_kv_heads = 2
    layer.num_heads = 2
    layer.num_kv_heads = 1
    layer.num_kv_head_replicas = 1
    layer.head_size = 8
    layer.v_head_size = 8
    method = _method(monkeypatch)
    method.create_weights(
        layer,
        input_size_per_partition=128,
        output_partition_sizes=[16, 8, 8],
        input_size=128,
        output_size=64,
        params_dtype=torch.bfloat16,
        weight_loader=layer.weight_loader,
    )

    loaded_tensors = {
        "qweight": _values((64, 64), torch.int8),
        "wscales": _values((8, 64), torch.float8_e4m3fn),
        "proj_up": _values((64, 8), torch.bfloat16),
        "wcscales": _values((64,), torch.bfloat16),
    }
    expected_global_outputs = [slice(16, 32), slice(40, 48), slice(56, 64)]
    for name, loaded in loaded_tensors.items():
        parameter = getattr(layer, name)
        parameter.weight_loader(parameter, loaded)
        expected = torch.cat(
            [
                loaded.narrow(parameter.output_dim, part.start, part.stop - part.start)
                for part in expected_global_outputs
            ],
            dim=parameter.output_dim,
        )
        torch.testing.assert_close(parameter, expected)
