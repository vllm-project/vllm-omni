# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import json

import pytest
import torch
from safetensors.torch import load_file, save_file

from tools.sensenova_u1.convert_lora_to_peft import convert_sensenova_lora

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _source_tensors(rank: int = 2, alpha: int = 2) -> dict[str, torch.Tensor]:
    return {
        "language_model.model.layers.0.self_attn.q_proj_mot_gen.lora_down.weight": torch.arange(
            rank * 4, dtype=torch.float32
        ).reshape(rank, 4),
        "language_model.model.layers.0.self_attn.q_proj_mot_gen.lora_up.weight": torch.arange(
            4 * rank, dtype=torch.float32
        ).reshape(4, rank),
        "language_model.model.layers.0.self_attn.q_proj_mot_gen.alpha": torch.tensor(alpha, dtype=torch.int32),
        "fm_modules.fm_head.0.lora_down.weight": torch.ones((rank, 4), dtype=torch.float32),
        "fm_modules.fm_head.0.lora_up.weight": torch.full((4, rank), 2.0, dtype=torch.float32),
        "fm_modules.fm_head.0.alpha": torch.tensor(alpha, dtype=torch.int32),
    }


def _write_source(path, tensors: dict[str, torch.Tensor] | None = None) -> dict[str, torch.Tensor]:
    tensors = tensors or _source_tensors()
    save_file(tensors, str(path))
    return tensors


def test_convert_sensenova_lora_writes_peft_adapter(tmp_path):
    source_path = tmp_path / "official.safetensors"
    output_dir = tmp_path / "peft"
    source_tensors = _write_source(source_path)

    convert_sensenova_lora(source_path, output_dir)

    converted = load_file(str(output_dir / "adapter_model.safetensors"))
    assert set(converted) == {
        "base_model.model.language_model.model.layers.0.self_attn.q_proj_mot_gen.lora_A.weight",
        "base_model.model.language_model.model.layers.0.self_attn.q_proj_mot_gen.lora_B.weight",
        "base_model.model.fm_modules.fm_head.0.lora_A.weight",
        "base_model.model.fm_modules.fm_head.0.lora_B.weight",
    }
    assert torch.equal(
        converted["base_model.model.language_model.model.layers.0.self_attn.q_proj_mot_gen.lora_A.weight"],
        source_tensors["language_model.model.layers.0.self_attn.q_proj_mot_gen.lora_down.weight"],
    )
    assert torch.equal(
        converted["base_model.model.fm_modules.fm_head.0.lora_B.weight"],
        source_tensors["fm_modules.fm_head.0.lora_up.weight"],
    )

    config = json.loads((output_dir / "adapter_config.json").read_text(encoding="utf-8"))
    assert config == {
        "bias": "none",
        "lora_alpha": 2,
        "peft_type": "LORA",
        "r": 2,
        "target_modules": [
            "fm_modules.fm_head.0",
            "language_model.model.layers.0.self_attn.q_proj_mot_gen",
        ],
    }


def test_convert_sensenova_lora_sorts_target_modules(tmp_path):
    source_path = tmp_path / "official.safetensors"
    output_dir = tmp_path / "peft"
    tensors = _source_tensors()
    tensors = dict(reversed(list(tensors.items())))
    _write_source(source_path, tensors)

    convert_sensenova_lora(source_path, output_dir)

    config = json.loads((output_dir / "adapter_config.json").read_text(encoding="utf-8"))
    assert config["target_modules"] == sorted(config["target_modules"])


def test_convert_sensenova_lora_rejects_inconsistent_alpha(tmp_path):
    source_path = tmp_path / "official.safetensors"
    tensors = _source_tensors()
    tensors["fm_modules.fm_head.0.alpha"] = torch.tensor(4, dtype=torch.int32)
    _write_source(source_path, tensors)

    with pytest.raises(ValueError, match="single alpha"):
        convert_sensenova_lora(source_path, tmp_path / "peft")


def test_convert_sensenova_lora_rejects_inconsistent_rank(tmp_path):
    source_path = tmp_path / "official.safetensors"
    tensors = _source_tensors()
    tensors["fm_modules.fm_head.0.lora_down.weight"] = torch.ones((3, 4))
    tensors["fm_modules.fm_head.0.lora_up.weight"] = torch.ones((4, 3))
    _write_source(source_path, tensors)

    with pytest.raises(ValueError, match="single rank"):
        convert_sensenova_lora(source_path, tmp_path / "peft")


def test_convert_sensenova_lora_rejects_zero_rank(tmp_path):
    source_path = tmp_path / "official.safetensors"
    tensors = _source_tensors(rank=0)
    _write_source(source_path, tensors)

    with pytest.raises(ValueError, match="positive rank"):
        convert_sensenova_lora(source_path, tmp_path / "peft")


def test_convert_sensenova_lora_rejects_missing_down(tmp_path):
    source_path = tmp_path / "official.safetensors"
    tensors = _source_tensors()
    del tensors["fm_modules.fm_head.0.lora_down.weight"]
    _write_source(source_path, tensors)

    with pytest.raises(ValueError, match="missing lora_down"):
        convert_sensenova_lora(source_path, tmp_path / "peft")


def test_convert_sensenova_lora_rejects_missing_up(tmp_path):
    source_path = tmp_path / "official.safetensors"
    tensors = _source_tensors()
    del tensors["fm_modules.fm_head.0.lora_up.weight"]
    _write_source(source_path, tensors)

    with pytest.raises(ValueError, match="missing lora_up"):
        convert_sensenova_lora(source_path, tmp_path / "peft")


def test_convert_sensenova_lora_rejects_missing_alpha(tmp_path):
    source_path = tmp_path / "official.safetensors"
    tensors = _source_tensors()
    del tensors["fm_modules.fm_head.0.alpha"]
    _write_source(source_path, tensors)

    with pytest.raises(ValueError, match="missing alpha"):
        convert_sensenova_lora(source_path, tmp_path / "peft")


def test_convert_sensenova_lora_rejects_non_matrix_weights(tmp_path):
    source_path = tmp_path / "official.safetensors"
    tensors = _source_tensors()
    tensors["fm_modules.fm_head.0.lora_down.weight"] = torch.ones(8)
    _write_source(source_path, tensors)

    with pytest.raises(ValueError, match="must be matrices"):
        convert_sensenova_lora(source_path, tmp_path / "peft")


def test_convert_sensenova_lora_rejects_unknown_tensor(tmp_path):
    source_path = tmp_path / "official.safetensors"
    tensors = _source_tensors()
    tensors["fm_modules.fm_head.0.bias"] = torch.ones(4)
    _write_source(source_path, tensors)

    with pytest.raises(ValueError, match="Unexpected SenseNova LoRA tensor"):
        convert_sensenova_lora(source_path, tmp_path / "peft")


@pytest.mark.parametrize("existing_name", ["adapter_model.safetensors", "adapter_config.json"])
def test_convert_sensenova_lora_refuses_existing_output(tmp_path, existing_name):
    source_path = tmp_path / "official.safetensors"
    output_dir = tmp_path / "peft"
    _write_source(source_path)
    output_dir.mkdir()
    (output_dir / existing_name).write_bytes(b"existing")

    with pytest.raises(FileExistsError, match="PEFT adapter already exists"):
        convert_sensenova_lora(source_path, output_dir)


def test_convert_sensenova_lora_overwrite_replaces_both_files(tmp_path):
    source_path = tmp_path / "official.safetensors"
    output_dir = tmp_path / "peft"
    _write_source(source_path)
    output_dir.mkdir()
    (output_dir / "adapter_model.safetensors").write_bytes(b"old weights")
    (output_dir / "adapter_config.json").write_text('{"old": true}\n', encoding="utf-8")

    convert_sensenova_lora(source_path, output_dir, overwrite=True)

    converted = load_file(str(output_dir / "adapter_model.safetensors"))
    config = json.loads((output_dir / "adapter_config.json").read_text(encoding="utf-8"))
    assert len(converted) == 4
    assert config["target_modules"] == [
        "fm_modules.fm_head.0",
        "language_model.model.layers.0.self_attn.q_proj_mot_gen",
    ]
