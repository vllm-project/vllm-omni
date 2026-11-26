# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from dataclasses import dataclass, field
from unittest.mock import Mock

import pytest
import torch

from tests.diffusion.lora.helpers import FakeLinearBase
from vllm_omni.diffusion.lora.manager import DiffusionLoRAManager
from vllm_omni.lora.loader import FilesystemLoRALoader
from vllm_omni.lora.request import LoRARequest

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _pipeline() -> torch.nn.Module:
    pipeline = torch.nn.Module()
    pipeline.transformer = torch.nn.Module()
    pipeline.transformer.foo = FakeLinearBase()
    return pipeline


def _request(name: str = "adapter") -> LoRARequest:
    return LoRARequest(lora_name=name, lora_int_id=1, lora_path=f"/unused/{name}")


@dataclass
class _Adapter:
    id: int
    loras: dict = field(default_factory=dict)


@dataclass
class _Config:
    r: int = 8
    lora_alpha: int = 16
    target_modules: list[str] = field(default_factory=lambda: ["foo"])


def _loaded_adapter(adapter_id: int = 1) -> tuple[_Adapter, _Config]:
    return _Adapter(adapter_id), _Config()


def test_default_loader_preserves_filesystem_arguments(monkeypatch: pytest.MonkeyPatch):
    lora_model, peft_helper = _loaded_adapter()
    from_dir = Mock(return_value=peft_helper)
    from_checkpoint = Mock(return_value=lora_model)
    resolve_path = Mock(return_value="/resolved/adapter")
    monkeypatch.setattr("vllm_omni.lora.loader.PEFTHelper.from_local_dir", from_dir)
    monkeypatch.setattr("vllm_omni.lora.loader.LoRAModel.from_local_checkpoint", from_checkpoint)
    monkeypatch.setattr("vllm_omni.lora.loader.get_adapter_absolute_path", resolve_path)
    manager = DiffusionLoRAManager(_pipeline(), torch.device("cpu"), torch.bfloat16)

    assert isinstance(manager._lora_loader, FilesystemLoRALoader)
    loaded_model, loaded_config = manager._load_adapter(_request())

    assert loaded_model is lora_model
    assert loaded_config is peft_helper
    resolve_path.assert_called_once_with("/unused/adapter")
    from_dir.assert_called_once_with("/resolved/adapter", max_position_embeddings=None)
    from_checkpoint.assert_called_once_with(
        "/resolved/adapter",
        expected_lora_modules=manager._expected_lora_modules,
        peft_helper=peft_helper,
        lora_model_id=1,
        device="cpu",
        dtype=torch.bfloat16,
        model_vocab_size=None,
        weights_mapper=None,
    )


def test_default_loader_preserves_pipeline_owned_loader(monkeypatch: pytest.MonkeyPatch):
    lora_model, peft_helper = _loaded_adapter()
    pipeline = _pipeline()
    pipeline_loader = Mock(return_value=(lora_model, peft_helper))
    pipeline._load_diffusion_lora_adapter = pipeline_loader
    from_dir = Mock()
    monkeypatch.setattr("vllm_omni.lora.loader.PEFTHelper.from_local_dir", from_dir)
    manager = DiffusionLoRAManager(pipeline, torch.device("cpu"), torch.bfloat16)

    loaded_model, loaded_config = manager._load_adapter(_request())

    assert loaded_model is lora_model
    assert loaded_config is peft_helper
    pipeline_loader.assert_called_once()
    from_dir.assert_not_called()


def test_pipeline_can_supply_loader_during_worker_initialization():
    pipeline = _pipeline()
    pipeline.diffusion_lora_loader = Mock()
    pipeline.diffusion_lora_loader.load.return_value = _loaded_adapter()
    manager = DiffusionLoRAManager(pipeline, torch.device("cpu"), torch.bfloat16)

    manager._load_adapter(_request())

    pipeline.diffusion_lora_loader.load.assert_called_once()


def test_custom_loaders_are_instance_scoped_and_can_supply_memory_adapters():
    first_loader = Mock()
    second_loader = Mock()
    first_adapter = _loaded_adapter(1)
    second_adapter = _loaded_adapter(2)
    first_loader.load.return_value = first_adapter
    second_loader.load.return_value = second_adapter
    first_manager = DiffusionLoRAManager(_pipeline(), torch.device("cpu"), torch.bfloat16, lora_loader=first_loader)
    second_manager = DiffusionLoRAManager(_pipeline(), torch.device("cpu"), torch.bfloat16, lora_loader=second_loader)
    first_request = _request("first-memory-adapter")
    second_request = _request("second-memory-adapter")

    assert first_manager._load_adapter(first_request) == first_adapter
    assert second_manager._load_adapter(second_request) == second_adapter
    first_loader.load.assert_called_once_with(first_request)
    second_loader.load.assert_called_once_with(second_request)
    assert first_manager._lora_loader is not second_manager._lora_loader
