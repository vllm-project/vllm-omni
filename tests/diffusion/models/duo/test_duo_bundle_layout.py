# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Duo must locate bundle components across the released layouts.

The released Duo variants disagree on where their components live: the
causal-base bundle nests DiT shards under `realization/backbone/transformer/`,
the quantized release keeps them directly at `realization/`, the internal
Realization checkpoint puts VAEs under `FL2VA/`, and the public bundles use
`codec/`. Hardcoding one layout works for exactly one release, so these tests
pin the probe order and the loud failure.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from vllm_omni.diffusion.models.duo.pipeline import _resolve_weights_subfolder
from vllm_omni.diffusion.models.duo.vae_bridge import resolve_codec_dir

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _write_sharded(directory: Path, index_name: str = "model.safetensors.index.json") -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "config.json").write_text("{}", encoding="utf-8")
    (directory / index_name).write_text("{}", encoding="utf-8")
    return directory


def test_dit_root_layout_is_used_directly(tmp_path):
    # The quantized release ships its shards at the root, with no subdirectory.
    _write_sharded(tmp_path)
    assert _resolve_weights_subfolder(tmp_path) is None


def test_dit_transformer_subfolder_is_probed(tmp_path):
    # causal-base nests a `transformer/` directory under `realization/backbone/`.
    _write_sharded(tmp_path / "transformer")
    assert _resolve_weights_subfolder(tmp_path) == "transformer"


def test_dit_backbone_subfolder_is_probed(tmp_path):
    _write_sharded(tmp_path / "backbone", "diffusion_pytorch_model.safetensors.index.json")
    assert _resolve_weights_subfolder(tmp_path) == "backbone"


def test_dit_root_wins_over_a_nested_subfolder(tmp_path):
    # Ambiguity is resolved by precedence, and the root is the shallowest.
    _write_sharded(tmp_path)
    _write_sharded(tmp_path / "transformer")
    assert _resolve_weights_subfolder(tmp_path) is None


def test_dit_layout_without_shards_fails_loudly(tmp_path):
    # An empty or wrong directory must not silently resolve to something the
    # loader would then read as an empty checkpoint.
    (tmp_path / "transformer").mkdir()
    with pytest.raises(ValueError, match="no loadable DiT shards"):
        _resolve_weights_subfolder(tmp_path)


def test_codec_dir_supports_the_released_codec_layout(tmp_path):
    # `inclusionAI/duo-v1` ships codec/{video,audio}_vae with weights only.
    (tmp_path / "codec" / "video_vae").mkdir(parents=True)
    assert resolve_codec_dir("video", tmp_path) == tmp_path / "codec" / "video_vae"


def test_codec_dir_prefers_the_released_layout_when_both_exist(tmp_path):
    # A public bundle ships `codec/`; the internal `FL2VA/` copy would be a
    # different checkpoint, so the released layout stays authoritative.
    (tmp_path / "codec" / "video_vae").mkdir(parents=True)
    (tmp_path / "FL2VA" / "video_vae").mkdir(parents=True)
    assert resolve_codec_dir("video", tmp_path) == tmp_path / "codec" / "video_vae"


def test_codec_dir_supports_the_internal_fl2va_layout(tmp_path):
    # The internal Realization checkpoint uses FL2VA/<kind>_vae.
    (tmp_path / "FL2VA" / "audio_vae").mkdir(parents=True)
    assert resolve_codec_dir("audio", tmp_path) == tmp_path / "FL2VA" / "audio_vae"


def test_codec_dir_missing_component_names_every_candidate(tmp_path):
    with pytest.raises(FileNotFoundError) as excinfo:
        resolve_codec_dir("video", tmp_path)
    message = str(excinfo.value)
    # The error has to say where it looked, or a wrong `--model` is undebuggable.
    assert "FL2VA" in message
    assert "codec" in message
