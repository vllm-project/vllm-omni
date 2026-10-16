# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU unit tests for Flux3 custom raw LoRA routing."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import pytest
import torch
import torch.nn as nn
from vllm.lora.lora_model import LoRAModel
from vllm.lora.peft_helper import PEFTHelper
from vllm.lora.request import LoRARequest

import vllm_omni.diffusion.models.flux3.lora as flux3_lora_module
import vllm_omni.diffusion.models.flux3.pipeline_flux3 as pipeline_module
from vllm_omni.diffusion.lora.manager import DiffusionLoRAManager
from vllm_omni.diffusion.models.flux3.flux3_transformer import Flux3Transformer2DModel
from vllm_omni.diffusion.models.flux3.lora import load_flux3_raw_lora
from vllm_omni.diffusion.models.flux3.pipeline_flux3 import Flux3Pipeline

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _request(path: Path, adapter_id: int = 7) -> LoRARequest:
    """Create a LoRA request with a stable adapter identity."""
    return LoRARequest(lora_name="flux3-test", lora_int_id=adapter_id, lora_path=str(path))


def _component(mapper: object) -> nn.Module:
    """Create a component exposing a Flux3 LoRA mapper."""
    component = nn.Module()
    component.lora_weights_mapper = mapper  # type: ignore[attr-defined]
    return component


def _pipeline() -> Flux3Pipeline:
    """Build an uninitialised pipeline sufficient for LoRA hook unit tests."""
    pipeline = object.__new__(Flux3Pipeline)
    nn.Module.__init__(pipeline)
    pipeline._raw_lora_adapter_ids = set()
    return pipeline


def test_flux3_pipeline_uses_default_shared_lora_manager() -> None:
    """CPU pipeline setup, no input modal, no output modal, default manager selection, no dataset."""
    assert "lora_manager_cls" not in Flux3Pipeline.__dict__


def test_weights_mapper_prefers_transformer_then_transformer_2_then_dit() -> None:
    """CPU mapper lookup, no input modal, no output modal, component precedence, no dataset."""
    pipeline = _pipeline()
    transformer_mapper, transformer_2_mapper, dit_mapper = object(), object(), object()
    pipeline.transformer = _component(transformer_mapper)
    pipeline.transformer_2 = _component(transformer_2_mapper)
    pipeline.dit = _component(dit_mapper)

    assert pipeline._flux3_lora_weights_mapper() is transformer_mapper
    del pipeline.transformer
    assert pipeline._flux3_lora_weights_mapper() is transformer_2_mapper
    del pipeline.transformer_2
    assert pipeline._flux3_lora_weights_mapper() is dit_mapper


@pytest.mark.parametrize(
    ("checkpoint_key", "expected_key"),
    [
        ("diffusion_model.single_blocks.0.qkv.lora_A.weight", "single_blocks.0.qkv.lora_A.weight"),
        ("dit.emb_in.image.lora_B.weight", "emb_in.image.lora_B.weight"),
    ],
)
def test_flux3_weights_mapper_removes_raw_checkpoint_prefix(checkpoint_key: str, expected_key: str) -> None:
    """CPU mapper test, raw checkpoint input, mapped key output, Flux3 prefix setting, no dataset."""
    assert list(Flux3Transformer2DModel.lora_weights_mapper.apply([(checkpoint_key, None)])) == [(expected_key, None)]


def test_raw_loader_sorts_files_uses_cpu_and_later_file_wins(tmp_path: Path) -> None:
    """CPU raw loader, safetensors input, LoRA model output, sorted directory setting, no dataset."""
    adapter_dir = tmp_path / "raw"
    adapter_dir.mkdir()
    first_file, second_file = adapter_dir / "a.safetensors", adapter_dir / "b.safetensors"
    first_file.touch()
    second_file.touch()
    mapper, peft_helper, lora_model = object(), mock.Mock(), mock.sentinel.lora_model

    with (
        mock.patch.object(
            flux3_lora_module.safetensors.torch,
            "load_file",
            side_effect=[{"shared": mock.sentinel.first}, {"shared": mock.sentinel.second}],
        ) as load_file,
        mock.patch.object(flux3_lora_module, "infer_peft_helper_from_file", return_value=peft_helper),
        mock.patch.object(LoRAModel, "from_lora_tensors", return_value=lora_model) as from_tensors,
    ):
        assert load_flux3_raw_lora(
            lora_request=_request(adapter_dir), lora_path=adapter_dir, dtype=torch.bfloat16, weights_mapper=mapper
        ) == (lora_model, peft_helper)

    assert load_file.call_args_list == [
        mock.call(str(first_file), device="cpu"),
        mock.call(str(second_file), device="cpu"),
    ]
    from_tensors.assert_called_once_with(
        lora_model_id=7,
        tensors={"shared": mock.sentinel.second},
        peft_helper=peft_helper,
        device="cpu",
        dtype=torch.bfloat16,
        model_vocab_size=None,
        weights_mapper=mapper,
    )


def test_raw_loader_declines_peft_directory_and_shared_manager_falls_back(tmp_path: Path) -> None:
    """CPU shared manager, PEFT directory input, standard LoRA output, fallback setting, no dataset."""
    adapter_dir = tmp_path / "peft"
    adapter_dir.mkdir()
    (adapter_dir / "adapter_config.json").write_text("{}", encoding="utf-8")
    pipeline = _pipeline()
    manager = object.__new__(DiffusionLoRAManager)
    manager.pipeline, manager.dtype, manager._expected_lora_modules = pipeline, torch.float32, {"to_q"}
    peft_helper = SimpleNamespace(r=1, lora_alpha=1, target_modules=["to_q"])
    lora_model = SimpleNamespace(id=7, loras={})

    with (
        mock.patch.object(PEFTHelper, "from_local_dir", return_value=peft_helper) as from_local_dir,
        mock.patch.object(LoRAModel, "from_local_checkpoint", return_value=lora_model) as from_local_checkpoint,
    ):
        assert manager._load_adapter(_request(adapter_dir)) == (lora_model, peft_helper)

    from_local_dir.assert_called_once()
    from_local_checkpoint.assert_called_once()
    assert pipeline._raw_lora_adapter_ids == set()


def test_raw_loader_rejects_empty_directory_and_unsupported_suffix(tmp_path: Path) -> None:
    """CPU raw loader, invalid adapter input, error output, path validation setting, no dataset."""
    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()
    with pytest.raises(FileNotFoundError, match="No .safetensors files found"):
        load_flux3_raw_lora(
            lora_request=_request(empty_dir), lora_path=empty_dir, dtype=torch.float32, weights_mapper=None
        )
    invalid_file = tmp_path / "adapter.bin"
    invalid_file.touch()
    with pytest.raises(ValueError, match="must be a .safetensors file"):
        load_flux3_raw_lora(
            lora_request=_request(invalid_file), lora_path=invalid_file, dtype=torch.float32, weights_mapper=None
        )


def test_pipeline_hook_marks_only_successful_raw_loads_and_clears_reused_ids(tmp_path: Path) -> None:
    """CPU pipeline hook, raw and PEFT paths, adapter-ID state output, reuse setting, no dataset."""
    pipeline = _pipeline()
    raw_path, peft_path = tmp_path / "raw.safetensors", tmp_path / "peft"
    raw_path.touch()
    peft_path.mkdir()
    raw_loaded = (mock.sentinel.lora_model, mock.sentinel.peft_helper)
    with mock.patch.object(pipeline_module, "load_flux3_raw_lora", side_effect=[raw_loaded, None]) as load_raw:
        assert (
            pipeline._load_diffusion_lora_adapter(
                lora_request=_request(raw_path), lora_path=str(raw_path), dtype=torch.float32
            )
            == raw_loaded
        )
        assert pipeline._raw_lora_adapter_ids == {7}
        assert (
            pipeline._load_diffusion_lora_adapter(
                lora_request=_request(peft_path), lora_path=str(peft_path), dtype=torch.float32
            )
            is None
        )

    assert pipeline._raw_lora_adapter_ids == set()
    assert load_raw.call_count == 2


def test_raw_only_binding_validation() -> None:
    """CPU binding validation, raw LoRA weights input, validation result, raw-ID setting, no dataset."""
    pipeline = _pipeline()
    raw_model = SimpleNamespace(id=7, loras={"to_q": object(), "to_k": object()})
    pipeline._raw_lora_adapter_ids.add(7)
    with pytest.raises(ValueError, match="bound=1/2"):
        pipeline._validate_diffusion_lora_binding(lora_model=raw_model, bound_lora_names=frozenset({"to_q"}))

    pipeline._raw_lora_adapter_ids.clear()
    pipeline._validate_diffusion_lora_binding(lora_model=raw_model, bound_lora_names=frozenset())
