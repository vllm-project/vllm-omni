# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Test the engine-independent reader; no native serving compatibility claim."""

import importlib.util
import json
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


@pytest.fixture
def reader_type():
    # The reader deliberately has no vLLM dependency. Import only that file so
    # these file-format tests do not initialize an engine or its package hooks.
    source = Path(__file__).resolve().parents[4] / "vllm_omni/model_executor/models/drvexo_unified/checkpoint.py"
    spec = importlib.util.spec_from_file_location("drvexo_checkpoint_unit", source)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.UnifiedBF16Checkpoint


@pytest.fixture
def checkpoint(tmp_path):
    components = {
        name: {"prefix": name + ".", "tensor_count": 1}
        for name in (
            "language_model",
            "vision_model",
            "audio_encoder",
            "connectors",
            "speech_model",
            "speech_codec",
            "speaker",
            "general_decision_head",
        )
    }
    values = {name + ".weight": torch.arange(4, dtype=torch.float32).reshape(2, 2) for name in components}
    values["language_model.weight"] = values["language_model.weight"].to(torch.bfloat16)
    save_file(values, tmp_path / "model.safetensors")
    (tmp_path / "config.json").write_text(
        json.dumps(
            {
                "model_type": "drvexo_speech_qwen35_4b",
                "drvexo_cognition": {"language_LoRA_merged": True, "embedded_v12_branch_tensors": 0},
            }
        )
    )
    (tmp_path / "components.json").write_text(
        json.dumps(
            {
                "format": "drvexo_unified_safetensors_v1",
                "components": components,
            }
        )
    )
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "metadata": {"format": "drvexo_unified_safetensors_v1"},
                "weight_map": {name: "model.safetensors" for name in values},
            }
        )
    )
    return tmp_path, values


def test_all_components_and_original_values(reader_type, checkpoint):
    root, values = checkpoint
    reader = reader_type(root)
    assert dict(reader.dtypes) == {"BF16": 1, "F32": 7}
    for name in reader.components:
        tensors = dict(reader.iter_component(name))
        assert list(tensors) == ["weight"]
        assert torch.equal(tensors["weight"], values[name + ".weight"])
        assert tensors["weight"].dtype == values[name + ".weight"].dtype
    assert len(list(reader.iter_thinker_weights())) == 2


@pytest.mark.parametrize(
    "variant",
    [
        "unmerged",
        "quantized",
        "old_8b",
        "missing_component",
        "missing_components_map",
        "missing_count",
        "missing_prefix",
        "missing_weight_map",
        "invalid_component_row",
        "wrong_count",
        "escaping_shard",
        "missing_tensor",
        "extra_tensor",
        "duplicate_owner",
    ],
)
def test_incomplete_or_wrong_packages_rejected(reader_type, checkpoint, variant):
    root, values = checkpoint
    path = root / "config.json"
    config = json.loads(path.read_text())
    if variant == "unmerged":
        config["drvexo_cognition"]["language_LoRA_merged"] = False
    elif variant == "quantized":
        config["quantization_config"] = {"quant_algo": "FP8"}
    elif variant == "old_8b":
        config["model_type"] = "drvexo_speech_8b"
    path.write_text(json.dumps(config))
    path = root / "components.json"
    components = json.loads(path.read_text())
    if variant == "missing_component":
        del components["components"]["vision_model"]
    elif variant == "missing_components_map":
        del components["components"]
    elif variant == "missing_count":
        del components["components"]["vision_model"]["tensor_count"]
    elif variant == "missing_prefix":
        del components["components"]["vision_model"]["prefix"]
    elif variant == "invalid_component_row":
        components["components"]["vision_model"] = None
    elif variant == "wrong_count":
        components["components"]["language_model"]["tensor_count"] = 2
    elif variant == "duplicate_owner":
        components["components"]["vision_model"]["prefix"] = "language_model."
    path.write_text(json.dumps(components))
    path = root / "model.safetensors.index.json"
    index = json.loads(path.read_text())
    if variant == "escaping_shard":
        index["weight_map"]["language_model.weight"] = "../model.safetensors"
    elif variant == "missing_tensor":
        index["weight_map"]["language_model.missing"] = index["weight_map"].pop("language_model.weight")
    elif variant == "extra_tensor":
        save_file({**values, "language_model.extra": torch.zeros(1)}, root / "model.safetensors")
    elif variant == "missing_weight_map":
        del index["weight_map"]
    path.write_text(json.dumps(index))
    with pytest.raises(ValueError):
        reader_type(root)
