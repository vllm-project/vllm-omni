# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""VAE class selection of the Wan-family VAE benchmark, for local checkpoints and Hub repo ids (no downloads)."""

import json

import pytest

from benchmarks.diffusion import bench_wan_family_vae as benchmark

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]


def _write_config(root, config):
    (root / "vae").mkdir(parents=True)
    (root / "vae" / "config.json").write_text(json.dumps(config))


def test_local_checkpoint_reads_class_from_vae_config(tmp_path, monkeypatch):
    _write_config(tmp_path, {"_class_name": "AutoencoderKLQwenImage"})
    monkeypatch.setattr(
        "huggingface_hub.hf_hub_download", lambda *a, **k: pytest.fail("a local checkpoint must not hit the Hub")
    )
    assert benchmark._vae_class_name(str(tmp_path)) == "AutoencoderKLQwenImage"


def test_repo_id_reads_class_from_hub_vae_config(tmp_path, monkeypatch):
    _write_config(tmp_path, {"_class_name": "AutoencoderKLQwenImage"})
    calls = []

    def fake_download(repo_id, filename, subfolder=None, **kwargs):
        calls.append((repo_id, filename, subfolder))
        return str(tmp_path / subfolder / filename)

    monkeypatch.setattr("huggingface_hub.hf_hub_download", fake_download)
    assert benchmark._vae_class_name("Qwen/Qwen-Image") == "AutoencoderKLQwenImage"
    assert calls == [("Qwen/Qwen-Image", "config.json", "vae")]


def test_config_without_class_name_is_an_error(tmp_path):
    _write_config(tmp_path, {"in_channels": 3})
    with pytest.raises(ValueError, match="_class_name"):
        benchmark._vae_class_name(str(tmp_path))


def test_explicit_module_class_overrides_config_class() -> None:
    from vllm_omni.diffusion.models.qwen_image.autoencoder_kl_qwenimage import AutoencoderKLQwenImage

    name = "vllm_omni.diffusion.models.qwen_image.autoencoder_kl_qwenimage:AutoencoderKLQwenImage"
    assert benchmark._vae_class(name) is AutoencoderKLQwenImage
