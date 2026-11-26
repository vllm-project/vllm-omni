# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import json

import pytest
import torch
from safetensors import safe_open
from safetensors.torch import save_file

from tools.quantize_minicpmo_ascend import export_checkpoint

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def test_export_indexes_every_tensor_in_its_shard(tmp_path):
    source = tmp_path / "source"
    output = tmp_path / "output"
    source.mkdir()
    output.mkdir()
    index_name = "model.safetensors.index.json"
    (source / "config.json").write_text(
        json.dumps({"model_type": "minicpmo", "version": "4.5", "num_hidden_layers": 1}), encoding="utf-8"
    )
    original = {f"llm.model.layers.0.proj{i}.weight": f"model-{i % 2}.safetensors" for i in range(7)}
    original["llm.lm_head.weight"] = "model-0.safetensors"
    for shard in set(original.values()):
        save_file({name: torch.ones(2, 3) for name, path in original.items() if path == shard}, source / shard)
    (source / index_name).write_text(json.dumps({"metadata": {}, "weight_map": original}), encoding="utf-8")

    export_checkpoint(source, output)

    expected = original | {
        f"{name}_{suffix}": shard
        for name, shard in original.items()
        if name.startswith("llm.model.layers.")
        for suffix in ("scale", "offset")
    }
    assert json.loads((output / index_name).read_text(encoding="utf-8"))["weight_map"] == expected
    for shard in set(expected.values()):
        with safe_open(output / shard, framework="pt", device="cpu") as weights:
            assert set(weights.keys()) == {name for name, path in expected.items() if path == shard}
