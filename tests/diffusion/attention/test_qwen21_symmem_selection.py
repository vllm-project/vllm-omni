# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from unittest.mock import Mock

import pytest
import torch

from vllm_omni.diffusion.attention.parallel import factory
from vllm_omni.diffusion.data import DiffusionParallelConfig, OmniDiffusionConfig

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]


@pytest.mark.parametrize(
    "model,eager,dtype,ring,allgather,mode,auto,major,explicit,expected",
    [
        ("QwenImage21Pipeline", True, torch.bfloat16, 1, 1, "strict", True, 10, False, True),
        ("QwenImage21Pipeline", True, torch.bfloat16, 1, 2, "strict", True, 10, False, False),
        ("QwenImage21Pipeline", True, torch.bfloat16, 1, 2, "strict", True, 10, True, True),
        ("QwenImage21Pipeline", True, torch.bfloat16, 2, 1, "strict", True, 10, False, False),
        ("QwenImage21Pipeline", True, torch.bfloat16, 1, 1, "advanced_uaa", True, 10, False, False),
        ("QwenImage21Pipeline", True, torch.bfloat16, 1, 1, "strict", False, 10, False, False),
        ("QwenImage21Pipeline", True, torch.bfloat16, 1, 1, "strict", True, 9, False, False),
        ("QwenImage21Pipeline", False, torch.bfloat16, 1, 1, "strict", True, 10, False, False),
        ("QwenImage21Pipeline", True, torch.float32, 1, 1, "strict", True, 10, False, False),
        ("QwenImagePipeline", True, torch.bfloat16, 1, 1, "strict", True, 10, False, False),
    ],
)
def test_auto_symmem_stays_within_qualified_configuration(
    monkeypatch, model, eager, dtype, ring, allgather, mode, auto, major, explicit, expected
):
    parallel = Mock(
        spec=DiffusionParallelConfig,
        ulysses_degree=4,
        ring_degree=ring,
        allgather_degree=allgather,
        ulysses_mode=mode,
        ulysses_a2a_permute=explicit,
    )
    config = Mock(
        spec=OmniDiffusionConfig,
        parallel_config=parallel,
        model_class_name=model,
        enforce_eager=eager,
        dtype=dtype,
        extras={"qwen21_auto_symmem_ulysses": auto},
    )
    monkeypatch.setattr(factory, "is_forward_context_available", lambda: True)
    monkeypatch.setattr(factory, "get_forward_context", lambda: Mock(omni_diffusion_config=config))
    monkeypatch.setattr(factory, "get_sp_group", Mock())
    monkeypatch.setattr(factory, "get_sequence_parallel_world_size", lambda: 4 * ring * allgather)
    platform = Mock()
    platform.is_cuda.return_value = True
    platform.is_available.return_value = True
    platform.get_device_capability.return_value = Mock(major=major)
    monkeypatch.setattr(factory, "current_omni_platform", platform)
    strategy = Mock()
    name = "UlyssesAllGatherKVParallelAttention" if allgather > 1 else "UlyssesParallelAttention"
    monkeypatch.setattr(factory, name, strategy)

    factory.build_parallel_attention_strategy(scatter_idx=2, gather_idx=1, use_sync=False)

    assert strategy.call_args.kwargs["ulysses_a2a_permute"] is expected
