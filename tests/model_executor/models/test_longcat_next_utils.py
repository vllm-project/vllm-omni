# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""LongCat-Next checkpoint resolution utilities.

vllm_config.model_config.model can be a Hub ID rather than a local
directory, and every raw file access in the decoder stages (generation_config.json,
model.safetensors.index.json shards, WEIGHT_PATH placeholder resolution) needs
the actual local snapshot path. These tests run without any network access:
a local directory must pass through untouched, a Hub ID must go through
huggingface_hub's snapshot_download with the requested revision, the
WEIGHT_PATH placeholder / absolute / bare-relative branches of
resolve_checkpoint_relative_path must resolve against the snapshot dir, and
load_weight_subtree must group checkpoint shards, strip the prefix, and fail
closed on missing module keys.
"""

import json

import pytest
import torch

from vllm_omni.model_executor.models.longcat_next.longcat_next_utils import (
    load_weight_subtree,
    resolve_checkpoint_relative_path,
    resolve_checkpoint_snapshot,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

_HUB_ID = "meituan-longcat/LongCat-Next"


class _StubApi:
    def __init__(self):
        self.calls = []

    def snapshot_download(self, repo_id, **kwargs):
        self.calls.append((repo_id, kwargs))
        return f"/hf-cache/hub/models--{repo_id.replace('/', '--')}/snapshots/abc123"


def _install_stub(monkeypatch):
    stub = _StubApi()
    monkeypatch.setattr("vllm_omni.transformers_utils.repo_utils.hf_api", lambda: stub)
    return stub


def test_local_directory_passes_through_unchanged(tmp_path, monkeypatch):
    stub = _install_stub(monkeypatch)

    resolved = resolve_checkpoint_snapshot(str(tmp_path))

    assert resolved == str(tmp_path)
    assert stub.calls == []


def test_local_directory_passes_through_even_with_revision(tmp_path, monkeypatch):
    stub = _install_stub(monkeypatch)

    resolved = resolve_checkpoint_snapshot(str(tmp_path), revision="v1.0")

    assert resolved == str(tmp_path)
    assert stub.calls == []


def test_hub_id_resolves_through_snapshot_download(monkeypatch):
    stub = _install_stub(monkeypatch)

    resolved = resolve_checkpoint_snapshot(_HUB_ID, revision="v1.0")

    assert resolved == f"/hf-cache/hub/models--{_HUB_ID.replace('/', '--')}/snapshots/abc123"
    assert stub.calls == [(_HUB_ID, {"revision": "v1.0"})]


def test_hub_id_without_revision_passes_none(monkeypatch):
    stub = _install_stub(monkeypatch)

    resolved = resolve_checkpoint_snapshot(_HUB_ID)

    assert resolved == f"/hf-cache/hub/models--{_HUB_ID.replace('/', '--')}/snapshots/abc123"
    assert stub.calls == [(_HUB_ID, {"revision": None})]


# --- resolve_checkpoint_relative_path: all three resolution branches --------


def test_relative_path_placeholder_resolves_against_model_dir(tmp_path):
    resolved = resolve_checkpoint_relative_path(
        "WEIGHT_PATH_TO_LONGCAT_NEXT/vocoder/cosy24k.safetensors",
        str(tmp_path),
    )

    assert resolved == str(tmp_path / "vocoder" / "cosy24k.safetensors")


def test_absolute_path_passes_through_unchanged(tmp_path):
    absolute = "/opt/weights/cosy24k.safetensors"

    resolved = resolve_checkpoint_relative_path(absolute, str(tmp_path))

    assert resolved == absolute


def test_bare_relative_path_joins_model_dir(tmp_path):
    resolved = resolve_checkpoint_relative_path("vocoder/cosy24k.safetensors", str(tmp_path))

    assert resolved == str(tmp_path / "vocoder" / "cosy24k.safetensors")


# --- load_weight_subtree: shard grouping, prefix stripping, fail-closed -----


class _TinyDecoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = torch.nn.Linear(4, 4)
        self.head = torch.nn.Linear(4, 2)


def _write_sharded_checkpoint(root, *, extra_module_key=False):
    """Two shards: the prefix subtree spans both, a sibling key shares one."""
    from safetensors.torch import save_file

    prefix = "model.audio_tokenizer"
    module = _TinyDecoder()
    proj = module.proj.state_dict()
    head = module.head.state_dict()
    tensors = {
        f"{prefix}.proj.weight": proj["weight"],
        f"{prefix}.proj.bias": proj["bias"],
        # head.* deliberately lives in the SECOND shard.
        f"{prefix}.head.weight": head["weight"],
        f"{prefix}.head.bias": head["bias"],
        # A sibling-subtree key sharing shard 1 must not leak into the load.
        "model.backbone.layer.0.weight": torch.zeros(2, 2),
    }
    if extra_module_key:
        tensors[f"{prefix}.extra.weight"] = torch.zeros(2, 2)
    shard_map = {
        f"{prefix}.proj.weight": "shard-1.safetensors",
        f"{prefix}.proj.bias": "shard-1.safetensors",
        "model.backbone.layer.0.weight": "shard-1.safetensors",
        f"{prefix}.head.weight": "shard-2.safetensors",
        f"{prefix}.head.bias": "shard-2.safetensors",
    }
    if extra_module_key:
        shard_map[f"{prefix}.extra.weight"] = "shard-2.safetensors"
    by_shard: dict[str, dict[str, torch.Tensor]] = {}
    for key, shard in shard_map.items():
        by_shard.setdefault(shard, {})[key] = tensors[key]
    for shard, shard_tensors in by_shard.items():
        save_file(shard_tensors, str(root / shard))
    (root / "model.safetensors.index.json").write_text(json.dumps({"weight_map": shard_map}))
    return module, proj, head


def test_load_weight_subtree_loads_prefix_keys_prefix_stripped(tmp_path):
    module, proj, head = _write_sharded_checkpoint(tmp_path)

    missing, unexpected = load_weight_subtree(module, str(tmp_path), "model.audio_tokenizer")

    assert missing == []
    assert unexpected == []
    # Prefix-stripped names, exact values, from BOTH shards.
    assert torch.equal(module.proj.weight, proj["weight"])
    assert torch.equal(module.proj.bias, proj["bias"])
    assert torch.equal(module.head.weight, head["weight"])
    assert torch.equal(module.head.bias, head["bias"])


def test_load_weight_subtree_raises_on_unknown_prefix(tmp_path):
    module, _, _ = _write_sharded_checkpoint(tmp_path)

    with pytest.raises(ValueError, match="No weights under prefix 'model.visual_tokenizer'"):
        load_weight_subtree(module, str(tmp_path), "model.visual_tokenizer")


def test_load_weight_subtree_raises_on_missing_module_keys(tmp_path):
    # The checkpoint covers every module key here, so drop one from the
    # index+shards to emulate a stale/partial snapshot.
    module, _, _ = _write_sharded_checkpoint(tmp_path)
    index_path = tmp_path / "model.safetensors.index.json"
    weight_map = json.loads(index_path.read_text())["weight_map"]
    del weight_map["model.audio_tokenizer.head.bias"]
    index_path.write_text(json.dumps({"weight_map": weight_map}))

    with pytest.raises(RuntimeError, match="randomly initialized"):
        load_weight_subtree(module, str(tmp_path), "model.audio_tokenizer")


def test_load_weight_subtree_warns_only_on_unexpected_checkpoint_keys(tmp_path):
    module, _, _ = _write_sharded_checkpoint(tmp_path, extra_module_key=True)

    missing, unexpected = load_weight_subtree(module, str(tmp_path), "model.audio_tokenizer")

    assert missing == []
    assert unexpected == ["extra.weight"]
