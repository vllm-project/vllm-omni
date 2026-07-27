# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]

DIFFUSERS_SOUND_TOKENIZER_CHECKPOINT_NAME = "diffusion_pytorch_model.safetensors"


class _FakeAVAEAudioTokenizer:
    def __init__(self, **kwargs) -> None:
        self.kwargs = kwargs
        self.sample_rate = int(kwargs["sample_rate"])
        self.audio_channels = int(kwargs["audio_channels"])
        self.latent_ch = int(kwargs["io_channels"])
        self.temporal_compression_factor = int(kwargs["hop_size"])

    def get_latent_num_samples(self, num_audio_samples: int) -> int:
        return int(num_audio_samples) // self.temporal_compression_factor

    def get_audio_num_samples(self, num_latent_samples: int) -> int:
        return int(num_latent_samples) * self.temporal_compression_factor

    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        return torch.zeros(latents.shape[0], self.audio_channels, 8)


def _write_component(root: Path, config: dict | None = None, checkpoint_name: str | None = None) -> Path:
    tokenizer_dir = root / "sound_tokenizer"
    tokenizer_dir.mkdir(parents=True)
    if checkpoint_name:
        (tokenizer_dir / checkpoint_name).write_bytes(b"stub")
    (tokenizer_dir / "config.json").write_text(json.dumps(config or {}), encoding="utf-8")
    return tokenizer_dir


def _patch_fake_avae(monkeypatch: pytest.MonkeyPatch, created: dict) -> None:
    from vllm_omni.diffusion.models.cosmos3 import sound_tokenizer

    class FakeAVAE(_FakeAVAEAudioTokenizer):
        def __init__(self, **kwargs) -> None:
            created.update(kwargs)
            super().__init__(**kwargs)

    monkeypatch.setattr(sound_tokenizer, "Cosmos3AVAEAudioTokenizer", FakeAVAE)
    monkeypatch.setattr(sound_tokenizer, "get_local_device", lambda: torch.device("cpu"))


def test_from_config_loads_local_diffusers_component(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    from vllm_omni.diffusion.models.cosmos3 import sound_tokenizer

    model_dir = tmp_path / "model"
    tokenizer_dir = _write_component(model_dir, checkpoint_name=DIFFUSERS_SOUND_TOKENIZER_CHECKPOINT_NAME)
    created = {}
    _patch_fake_avae(monkeypatch, created)

    tokenizer = sound_tokenizer.Cosmos3SoundTokenizer.from_config(
        SimpleNamespace(
            model=str(model_dir),
            custom_pipeline_args={"sound_sample_rate": 32000, "sound_hop_size": 800, "sound_dim": 3},
            dtype=torch.float32,
        )
    )

    assert created["checkpoint_path"] == str(tokenizer_dir / DIFFUSERS_SOUND_TOKENIZER_CHECKPOINT_NAME)
    assert created["config_path"] == str(tokenizer_dir / "config.json")
    assert (tokenizer.sample_rate, tokenizer.latent_ch, tokenizer.hop_size, tokenizer.latent_fps) == (
        32000,
        3,
        800,
        40.0,
    )


def test_from_config_downloads_component_from_hf_repo(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    import huggingface_hub

    from vllm_omni.diffusion.models.cosmos3 import sound_tokenizer

    cache_dir = tmp_path / "hf"
    _write_component(cache_dir, checkpoint_name=DIFFUSERS_SOUND_TOKENIZER_CHECKPOINT_NAME)
    calls = []
    created = {}
    _patch_fake_avae(monkeypatch, created)

    def fake_snapshot_download(repo_id: str, *, revision: str | None, allow_patterns: list[str]) -> str:
        calls.append((repo_id, revision, allow_patterns))
        return str(cache_dir)

    monkeypatch.setattr(huggingface_hub, "snapshot_download", fake_snapshot_download)

    sound_tokenizer.Cosmos3SoundTokenizer.from_config(
        SimpleNamespace(
            model="nvidia/cosmos3",
            revision="test-rev",
            custom_pipeline_args={"sound_sample_rate": 32000, "sound_hop_size": 800, "sound_dim": 3},
            dtype=torch.float32,
        )
    )

    assert created["checkpoint_path"].endswith(DIFFUSERS_SOUND_TOKENIZER_CHECKPOINT_NAME)
    assert calls == [
        (
            "nvidia/cosmos3",
            "test-rev",
            ["sound_tokenizer/config.json", f"sound_tokenizer/{DIFFUSERS_SOUND_TOKENIZER_CHECKPOINT_NAME}"],
        )
    ]


@pytest.mark.parametrize(
    ("checkpoint_name", "message"),
    [
        (None, "no AVAE sound tokenizer checkpoint"),
        ("model.safetensors", DIFFUSERS_SOUND_TOKENIZER_CHECKPOINT_NAME),
    ],
)
def test_default_component_requires_diffusers_checkpoint_name(tmp_path, checkpoint_name, message) -> None:
    from vllm_omni.diffusion.models.cosmos3 import sound_tokenizer

    model_dir = tmp_path / "model"
    _write_component(model_dir, checkpoint_name=checkpoint_name)

    with pytest.raises(ValueError, match=message):
        sound_tokenizer.Cosmos3SoundTokenizer.from_config(
            SimpleNamespace(model=str(model_dir), custom_pipeline_args={}, dtype=torch.float32)
        )


def test_component_config_precedence_and_conflict_detection(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    from vllm_omni.diffusion.models.cosmos3 import sound_tokenizer

    component_config = {
        "sampling_rate": 48000,
        "dec_out_channels": 2,
        "vocoder_input_dim": 64,
        "hop_size": 1920,
    }
    model_dir = tmp_path / "model"
    _write_component(model_dir, component_config, DIFFUSERS_SOUND_TOKENIZER_CHECKPOINT_NAME)
    created = {}
    _patch_fake_avae(monkeypatch, created)

    tokenizer = sound_tokenizer.Cosmos3SoundTokenizer.from_config(
        SimpleNamespace(
            model=str(model_dir),
            custom_pipeline_args={
                "sound_normalize_latents": True,
                "sound_normalization_type": "tanh",
                "sound_tanh_input_scale": 2.0,
            },
            model_config={
                "sound_tokenizer": {
                    "sample_rate": 32000,
                    "audio_channels": 1,
                    "io_channels": 3,
                    "hop_size": 800,
                    "normalize_latents": False,
                    "normalization_type": "none",
                }
            },
            dtype=torch.float32,
        )
    )

    assert (created["sample_rate"], created["audio_channels"], created["io_channels"], created["hop_size"]) == (
        48000,
        2,
        64,
        1920,
    )
    assert (created["normalize_latents"], created["normalization_type"], created["tanh_input_scale"]) == (
        True,
        "tanh",
        2.0,
    )
    assert (tokenizer.sample_rate, tokenizer.latent_ch, tokenizer.hop_size, tokenizer.latent_fps) == (
        48000,
        64,
        1920,
        25.0,
    )

    with pytest.raises(ValueError, match=r"sample_rate.*48000.*32000"):
        sound_tokenizer.Cosmos3SoundTokenizer.from_config(
            SimpleNamespace(
                model=str(model_dir),
                custom_pipeline_args={"sound_sample_rate": 32000},
                dtype=torch.float32,
            )
        )


def test_avae_uses_diffusers_decoder_state_dict_layout(tmp_path) -> None:
    from safetensors.torch import save_file

    from vllm_omni.diffusion.models.cosmos3.audio_tokenizer import avae

    config = {
        "sampling_rate": 8000,
        "hop_size": 2,
        "dec_dim": 4,
        "dec_c_mults": [1],
        "dec_strides": [2],
        "dec_out_channels": 1,
        "vocoder_input_dim": 2,
        "normalization_type": "none",
    }
    checkpoint_path = tmp_path / DIFFUSERS_SOUND_TOKENIZER_CHECKPOINT_NAME
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")

    decoder = avae.OobleckDecoder(4, 2, 1, [2], [1])
    save_file({f"decoder.{key}": value for key, value in decoder.state_dict().items()}, str(checkpoint_path))

    tokenizer = avae.Cosmos3AVAEAudioTokenizer(
        checkpoint_path=checkpoint_path,
        config_path=config_path,
        dtype=torch.float32,
        device="cpu",
    )

    keys = set(tokenizer.state_dict())
    assert {"decoder.conv1.weight_g", "decoder.block.0.conv_t1.weight_g", "decoder.conv2.weight_g"} <= keys
    assert not any(key.startswith(("decoder.layers.", "model.decoder.")) for key in keys)
    assert tokenizer.decode(torch.zeros(1, 2, 3)).shape == (1, 1, 6)
    with pytest.raises(NotImplementedError, match="decoder-only"):
        tokenizer.encode(torch.zeros(1, 1, 6))
