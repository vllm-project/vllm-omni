# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Offline CI setup rejects corrupt assets before a trusted conversion."""

import hashlib
import json
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

from tools import prepare_zonos2_ci as assets

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def converted_fixture(directory: Path) -> dict[str, torch.Tensor]:
    directory.mkdir()
    tensors = {f"weight.{i}": torch.tensor([i], dtype=torch.float32) for i in range(507)}
    entries = [
        {
            "key": key,
            "shape": [1],
            "dtype": "float32",
            "sha256": hashlib.sha256(value.view(torch.uint8).numpy()).hexdigest(),
        }
        for key, value in tensors.items()
    ]
    (directory / "manifest.json").write_text(
        json.dumps({"source_sha256": assets.MODEL_HASHES, "tensor_count": 507, "tensors": entries})
    )
    (directory / "config.json").write_text(
        json.dumps({"model_type": "zonos2", "architectures": ["Zonos2ForConditionalGeneration"]})
    )
    save_file(tensors, directory / "model.safetensors")
    return tensors


def test_corrupt_cached_original_never_reaches_converter(tmp_path, monkeypatch):
    monkeypatch.delenv("ZONOS2_TEST_MODEL_PATH", raising=False)
    original = tmp_path / "original"
    original.mkdir()
    for name in assets.MODEL_HASHES:
        (original / name).write_bytes(b"corrupted")

    def forbidden(*args, **kwargs):
        pytest.fail("Corrupt original checkpoint must not invoke a converter or downloader")

    monkeypatch.setattr(assets.subprocess, "run", forbidden)
    with pytest.raises(ValueError, match="incorrect pinned asset"):
        assets.prepare(tmp_path, offline=True)


def test_offline_missing_asset_does_not_contact_network(tmp_path, monkeypatch):
    monkeypatch.delenv("ZONOS2_TEST_MODEL_PATH", raising=False)

    def forbidden(*args, **kwargs):
        pytest.fail("Offline CI preparation contacted the network")

    monkeypatch.setattr(assets.subprocess, "run", forbidden)
    monkeypatch.setattr(assets.urllib.request, "urlopen", forbidden)
    with pytest.raises(FileNotFoundError, match="Offline preparation"):
        assets.prepare(tmp_path, offline=True)


def test_tensor_corruption_is_detected_even_when_source_manifest_is_unchanged(tmp_path):
    directory = tmp_path / "converted"
    tensors = converted_fixture(directory)
    assets.verify_model(directory)
    tensors["weight.7"] += 1
    save_file(tensors, directory / "model.safetensors")
    with pytest.raises(ValueError, match="tensor SHA256 mismatch"):
        assets.verify_model(directory)


def test_unpinned_speaker_code_is_rejected(tmp_path):
    for name in assets.SPEAKER_HASHES:
        (tmp_path / name).write_bytes(b"unexpected code or weights")
    with pytest.raises(ValueError, match="incorrect pinned asset"):
        assets.verify_files(tmp_path, assets.SPEAKER_HASHES)
