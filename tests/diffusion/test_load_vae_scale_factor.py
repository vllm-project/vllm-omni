# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import json

from vllm_omni.diffusion.models.utils import load_vae_scale_factor


def _make_vae_config(tmp_path, data):
    vae_dir = tmp_path / "vae"
    vae_dir.mkdir()
    (vae_dir / "config.json").write_text(json.dumps(data))
    return str(tmp_path)


def test_standard_block_out_channels(tmp_path):
    path = _make_vae_config(tmp_path, {"block_out_channels": [128, 256, 512, 512]})
    assert load_vae_scale_factor(path) == 8


def test_temporal_downsample(tmp_path):
    path = _make_vae_config(tmp_path, {"temporal_downsample": [True, True, True]})
    assert load_vae_scale_factor(path, config_key="temporal_downsample", exponent_offset=0) == 8


def test_exponent_offset_zero(tmp_path):
    path = _make_vae_config(tmp_path, {"block_out_channels": [128, 256, 512, 512]})
    assert load_vae_scale_factor(path, exponent_offset=0) == 16


def test_missing_config_returns_default(tmp_path):
    assert load_vae_scale_factor(str(tmp_path)) == 8
    assert load_vae_scale_factor(str(tmp_path), default=16) == 16


def test_missing_key_returns_default(tmp_path):
    path = _make_vae_config(tmp_path, {"other_key": "value"})
    assert load_vae_scale_factor(path) == 8
