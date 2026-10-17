# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Real TP=2 CUDA regression for SANA-Video's distributed RMSNorm.

The TP=2 path must keep its global sum/count reduction.  It must not enter the
TP=1-only Triton helper, even when each local video-token shard is large enough
to satisfy the helper's size threshold.
"""

from __future__ import annotations

import os
from unittest.mock import patch

import pytest
import torch
from vllm.utils.network_utils import get_open_port

import vllm_omni.diffusion.layers.sana_rms_norm as sana_rms
import vllm_omni.diffusion.models.sana_video.transformer_sana_video as sana_transformer
from vllm_omni.diffusion.distributed.parallel_state import (
    destroy_distributed_env,
    init_distributed_environment,
    initialize_model_parallel,
)
from vllm_omni.platforms import current_omni_platform

_WORLD_SIZE = 2
_ROWS = 2048
_LOCAL_WIDTH = 1120
_EPS = 1e-5


def _raw_bf16_equal(left: torch.Tensor, right: torch.Tensor) -> bool:
    assert left.dtype == right.dtype == torch.bfloat16
    return torch.equal(left.view(torch.int16), right.view(torch.int16))


def _frozen_tp_reference(
    hidden_states: torch.Tensor,
    weight: torch.Tensor,
) -> torch.Tensor:
    """Frozen pre-fast-path expression, evaluated on the real TP group."""
    hidden_states_float = hidden_states.float()
    local_sum_sq = hidden_states_float.pow(2).sum(dim=-1, keepdim=True)
    global_sum_sq = sana_transformer.tensor_model_parallel_all_reduce(local_sum_sq)
    count = hidden_states.shape[-1] * _WORLD_SIZE
    normalized = hidden_states_float * torch.rsqrt(global_sum_sq / count + _EPS)
    normalized = normalized.to(torch.bfloat16)
    return normalized * weight


def _tp2_worker(local_rank: int, init_method: str) -> None:
    device = current_omni_platform.get_torch_device(local_rank)
    current_omni_platform.set_device(device)
    os.environ.update(
        {
            "RANK": str(local_rank),
            "LOCAL_RANK": str(local_rank),
            "WORLD_SIZE": str(_WORLD_SIZE),
            "MASTER_ADDR": "127.0.0.1",
            "MASTER_PORT": init_method.rsplit(":", 1)[-1],
        }
    )

    init_distributed_environment(
        world_size=_WORLD_SIZE,
        rank=local_rank,
        local_rank=local_rank,
        distributed_init_method=init_method,
        backend="nccl",
    )
    try:
        initialize_model_parallel(tensor_parallel_size=_WORLD_SIZE, backend="nccl")
        assert torch.distributed.get_backend() == "nccl"
        assert sana_transformer.get_tensor_model_parallel_world_size() == _WORLD_SIZE
        assert sana_transformer.get_tensor_model_parallel_rank() == local_rank

        generator = torch.Generator(device=device).manual_seed(6825 + local_rank)
        hidden_states = torch.randn(
            (1, _ROWS, _LOCAL_WIDTH),
            generator=generator,
            device=device,
            dtype=torch.bfloat16,
        )
        # Give both the inputs and affine shards rank-dependent distributions;
        # equal local values could hide a wrong or missing TP contribution.
        hidden_states = hidden_states + torch.tensor(
            (local_rank * 2 - 1) * 0.0625,
            device=device,
            dtype=torch.bfloat16,
        )
        weight = torch.linspace(
            0.5 + local_rank * 0.125,
            1.5 + local_rank * 0.125,
            _LOCAL_WIDTH,
            device=device,
            dtype=torch.bfloat16,
        )
        assert hidden_states.numel() >= sana_rms._MIN_ELEMENTS

        norm = sana_transformer.SanaDistributedRMSNorm(_LOCAL_WIDTH, eps=_EPS).to(
            device=device,
            dtype=torch.bfloat16,
        )
        with torch.no_grad():
            norm.weight.copy_(weight)
            # This uses its own collective so the expected expression and the
            # production call have identical NCCL reduction ordering.
            expected = _frozen_tp_reference(hidden_states, norm.weight)

        original_all_reduce = sana_transformer.tensor_model_parallel_all_reduce
        expected_local_sum = hidden_states.float().pow(2).sum(dim=-1, keepdim=True)
        collective_calls = 0

        def checked_all_reduce(value: torch.Tensor) -> torch.Tensor:
            nonlocal collective_calls
            collective_calls += 1
            assert torch.equal(value, expected_local_sum)
            reduced = original_all_reduce(value)
            # Return a separate buffer and poison the input after the collective.
            # This catches implementations that call the collective but ignore
            # its return value (custom all-reduce may return a new tensor).
            returned = reduced.clone()
            value.fill_(float("nan"))
            return returned

        def unexpected_fast_path(*_args, **_kwargs):
            raise AssertionError("TP=2 must not invoke a TP=1 exact RMSNorm helper or kernel")

        with (
            patch.object(sana_transformer, "tensor_model_parallel_all_reduce", checked_all_reduce),
            patch.object(sana_transformer, "exact_sana_rms_norm", unexpected_fast_path),
            patch.object(sana_transformer, "exact_sana_rms_norm_sum", unexpected_fast_path),
            patch.object(sana_rms, "exact_sana_rms_norm", unexpected_fast_path),
            patch.object(sana_rms, "exact_sana_rms_norm_sum", unexpected_fast_path),
            patch.object(sana_rms, "_launch_exact_sana_rms_norm", unexpected_fast_path),
            patch.object(sana_rms, "_launch_exact_sana_rms_norm_sum", unexpected_fast_path),
        ):
            with torch.no_grad():
                actual = norm(hidden_states)

        assert collective_calls == 1
        assert _raw_bf16_equal(actual, expected)
    finally:
        destroy_distributed_env()


def _require_two_bf16_cuda_devices() -> None:
    if not current_omni_platform.is_cuda() or not torch.cuda.is_available():
        pytest.skip("NVIDIA CUDA is required")
    if current_omni_platform.get_device_count() < _WORLD_SIZE:
        pytest.skip("TP=2 regression requires two CUDA devices")

    for device_index in range(_WORLD_SIZE):
        device = current_omni_platform.get_torch_device(device_index)
        current_omni_platform.set_device(device)
        if not torch.cuda.is_bf16_supported():
            pytest.skip(f"CUDA device {device_index} does not support BF16")
    current_omni_platform.set_device(current_omni_platform.get_torch_device(0))


@pytest.mark.core_model
@pytest.mark.diffusion
@pytest.mark.parallel
@pytest.mark.cuda
@pytest.mark.cards_2
def test_sana_video_distributed_rms_norm_tp2_keeps_global_reduction() -> None:
    _require_two_bf16_cuda_devices()
    init_method = f"tcp://127.0.0.1:{get_open_port()}"
    torch.multiprocessing.spawn(
        _tp2_worker,
        args=(init_method,),
        nprocs=_WORLD_SIZE,
        join=True,
    )
