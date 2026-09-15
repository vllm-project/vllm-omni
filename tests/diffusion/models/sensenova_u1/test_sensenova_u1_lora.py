# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import os
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file
from vllm.lora.layers import BaseLayerWithLoRA
from vllm.lora.request import LoRARequest
from vllm.model_executor.layers.linear import ReplicatedLinear
from vllm.utils.network_utils import get_open_port

from tools.sensenova_u1.convert_lora_to_peft import convert_sensenova_lora
from vllm_omni.diffusion.lora.manager import DiffusionLoRAManager
from vllm_omni.diffusion.models.sensenova_u1.pipeline_sensenova_u1 import (
    SenseNovaU1Pipeline,
    _build_fm_head,
)

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]


def _prepare_cpu_linear_dispatch(module: torch.nn.Module) -> None:
    from vllm.model_executor.layers.utils import dispatch_cpu_unquantized_gemm
    from vllm.platforms import current_platform

    if not current_platform.is_cpu():
        return
    for child in module.modules():
        if isinstance(child, ReplicatedLinear):
            dispatch_cpu_unquantized_gemm(child, remove_weight=False)


@pytest.fixture(scope="module", autouse=True)
def _init_single_rank_tp_env():
    from vllm.config import VllmConfig, set_current_vllm_config
    from vllm.distributed.parallel_state import (
        cleanup_dist_env_and_memory,
        init_distributed_environment,
        initialize_model_parallel,
    )

    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ["MASTER_PORT"] = str(get_open_port())
    with set_current_vllm_config(VllmConfig()):
        init_distributed_environment(
            world_size=1,
            rank=0,
            local_rank=0,
            distributed_init_method="env://",
        )
        initialize_model_parallel()
        yield
        cleanup_dist_env_and_memory()


def test_sensenova_fm_head_uses_lora_compatible_linear_layers():
    fm_head = _build_fm_head(input_dim=8, intermediate_dim=16, output_dim=12)
    _prepare_cpu_linear_dispatch(fm_head)

    assert isinstance(fm_head[0], ReplicatedLinear)
    assert isinstance(fm_head[2], ReplicatedLinear)
    assert fm_head[0].return_bias is False
    assert fm_head[2].return_bias is False

    output = fm_head(torch.randn(2, 3, 8))
    assert output.shape == (2, 3, 12)


def test_sensenova_lora_manager_scans_fm_modules():
    assert SenseNovaU1Pipeline._lora_components == ["language_model", "fm_modules"]


def test_sensenova_fm_head_preserves_checkpoint_weight_names():
    pipeline = torch.nn.Module()
    pipeline.fm_modules = torch.nn.ModuleDict(
        {"fm_head": _build_fm_head(input_dim=8, intermediate_dim=16, output_dim=12)}
    )
    weights = {
        "fm_modules.fm_head.0.weight": torch.randn(16, 8),
        "fm_modules.fm_head.0.bias": torch.randn(16),
        "fm_modules.fm_head.2.weight": torch.randn(12, 16),
        "fm_modules.fm_head.2.bias": torch.randn(12),
    }

    loaded = SenseNovaU1Pipeline.load_weights(pipeline, weights.items())

    assert loaded == set(weights)
    for name, parameter in pipeline.named_parameters():
        assert torch.equal(parameter, weights[name])


def _write_official_fm_head_lora(source_path: Path, rank: int = 2) -> None:
    save_file(
        {
            "fm_modules.fm_head.0.lora_down.weight": torch.ones((rank, 8), dtype=torch.float32),
            "fm_modules.fm_head.0.lora_up.weight": torch.ones((16, rank), dtype=torch.float32),
            "fm_modules.fm_head.0.alpha": torch.tensor(rank, dtype=torch.int32),
            "fm_modules.fm_head.2.lora_down.weight": torch.ones((rank, 16), dtype=torch.float32),
            "fm_modules.fm_head.2.lora_up.weight": torch.ones((12, rank), dtype=torch.float32),
            "fm_modules.fm_head.2.alpha": torch.tensor(rank, dtype=torch.int32),
        },
        str(source_path),
    )


def test_converted_fm_head_lora_round_trip(tmp_path):
    source_path = tmp_path / "official.safetensors"
    adapter_dir = tmp_path / "adapter"
    _write_official_fm_head_lora(source_path)
    convert_sensenova_lora(source_path, adapter_dir)

    pipeline = torch.nn.Module()
    pipeline._lora_components = ["fm_modules"]
    pipeline.fm_modules = torch.nn.ModuleDict(
        {"fm_head": _build_fm_head(input_dim=8, intermediate_dim=16, output_dim=12)}
    )
    for parameter in pipeline.parameters():
        torch.nn.init.zeros_(parameter)
    _prepare_cpu_linear_dispatch(pipeline)

    manager = DiffusionLoRAManager(
        pipeline=pipeline,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    inputs = torch.ones((1, 8), dtype=torch.float32)
    baseline = pipeline.fm_modules["fm_head"](inputs)

    request = LoRARequest(lora_name="official", lora_int_id=1, lora_path=str(adapter_dir))
    manager.set_active_adapter(request)
    adapted = pipeline.fm_modules["fm_head"](inputs)

    assert isinstance(pipeline.fm_modules["fm_head"][0], BaseLayerWithLoRA)
    assert isinstance(pipeline.fm_modules["fm_head"][2], BaseLayerWithLoRA)
    assert not torch.equal(adapted, baseline)

    manager.set_active_adapter(None)
    restored = pipeline.fm_modules["fm_head"](inputs)
    assert torch.equal(restored, baseline)
