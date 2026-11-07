# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Exercise small real safetensors views, relocation and path containment."""

import json
import shutil

import pytest
import torch
from safetensors.torch import save_file

from vllm_omni.model_executor.models.drvexo_speech_8b.checkpoint import (
    FORMAT,
    VIEW_FORMAT,
    UnifiedCheckpoint,
    iter_view,
    load_checkpoint,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def checkpoint(root):
    root.mkdir()
    tensors = {"thinker.weight": torch.arange(6, dtype=torch.bfloat16).reshape(2, 3), "speaker.value": torch.ones(4)}
    save_file(tensors, root / "model.safetensors")
    (root / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {"format": FORMAT}, "weight_map": {k: "model.safetensors" for k in tensors}})
    )
    (root / "components.json").write_text(
        json.dumps(
            {
                "format": FORMAT,
                "components": {
                    "thinker": {"prefix": "thinker.", "tensor_count": 1, "payload_kind": "state_dict"},
                    "speaker": {"prefix": "speaker.", "tensor_count": 1, "payload_kind": "tensor"},
                },
            }
        )
    )
    for component in ("thinker", "speaker"):
        sub = root / component
        sub.mkdir()
        (sub / "weights.json").write_text(
            json.dumps({"format": VIEW_FORMAT, "component": component, "index": "../model.safetensors.index.json"})
        )
    return tensors


def test_stage_views_share_one_checkpoint_after_relocation(tmp_path):
    expected = checkpoint(tmp_path / "before")
    root = tmp_path / "after"
    shutil.move(tmp_path / "before", root)
    loaded = dict(iter_view(root / "thinker/weights.json"))
    assert set(loaded) == {"weight"}
    assert torch.equal(loaded["weight"], expected["thinker.weight"])
    assert loaded["weight"].dtype == torch.bfloat16
    assert torch.equal(load_checkpoint(root / "speaker/weights.json"), expected["speaker.value"])
    assert set(load_checkpoint(root / "thinker/weights.json")["state_dict"]) == {"weight"}


def test_rejects_shard_path_traversal_and_symlinks(tmp_path):
    root = tmp_path / "model"
    checkpoint(root)
    index_file = root / "model.safetensors.index.json"
    index = json.loads(index_file.read_text())
    index["weight_map"]["thinker.weight"] = "../outside.safetensors"
    index_file.write_text(json.dumps(index))
    with pytest.raises(ValueError, match="Invalid tensor index"):
        UnifiedCheckpoint(root)
    index["weight_map"]["thinker.weight"] = "link.safetensors"
    shutil.copy(root / "model.safetensors", tmp_path / "outside.safetensors")
    (root / "link.safetensors").symlink_to(tmp_path / "outside.safetensors")
    index_file.write_text(json.dumps(index))
    with pytest.raises(ValueError, match="escapes"):
        UnifiedCheckpoint(root)


def test_rejects_incomplete_component(tmp_path):
    root = tmp_path / "model"
    checkpoint(root)
    manifest = root / "components.json"
    data = json.loads(manifest.read_text())
    data["components"]["thinker"]["tensor_count"] = 2
    manifest.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="Incomplete component"):
        dict(UnifiedCheckpoint(root).iter_tensors("thinker"))
