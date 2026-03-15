# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
from unittest.mock import patch

from vllm_omni.diffusion.models.omnigen2.pipeline_omnigen2 import (
    _load_transformer_config,
    _parse_transformer_config,
)

FULL_CONFIG = {
    "patch_size": 2,
    "in_channels": 16,
    "out_channels": 16,
    "hidden_size": 2520,
    "num_layers": 32,
    "num_refiner_layers": 2,
    "num_attention_heads": 21,
    "num_kv_heads": 7,
    "multiple_of": 256,
    "ffn_dim_multiplier": None,
    "norm_eps": 1e-5,
    "axes_dim_rope": [40, 40, 40],
    "axes_lens": [1024, 1664, 1664],
    "text_feat_dim": 2048,
    "timestep_scale": 1000.0,
}


class TestParseTransformerConfig:
    def test_full_config(self):
        result = _parse_transformer_config(FULL_CONFIG)

        assert result["patch_size"] == 2
        assert result["hidden_size"] == 2520
        assert result["num_layers"] == 32
        assert result["axes_dim_rope"] == (40, 40, 40)
        assert result["axes_lens"] == (1024, 1664, 1664)
        assert isinstance(result["axes_dim_rope"], tuple)
        assert isinstance(result["axes_lens"], tuple)
        # All keys should be present
        assert len(result) == len(FULL_CONFIG)

    def test_empty_config(self):
        """Empty/falsy config returns empty dict (caller uses defaults)."""
        assert _parse_transformer_config({}) == {}
        assert _parse_transformer_config(None) == {}

    def test_partial_config(self):
        """Only provided keys are extracted."""

        partial = {"hidden_size": 4096, "num_layers": 48}
        result = _parse_transformer_config(partial)
        assert result == {"hidden_size": 4096, "num_layers": 48}

    def test_extra_keys_ignored(self):
        """Keys not in the whitelist are dropped."""

        config = {"hidden_size": 2520, "_class_name": "SomeClass", "extra": 123}
        result = _parse_transformer_config(config)
        assert "hidden_size" in result
        assert "_class_name" not in result
        assert "extra" not in result


class TestLoadTransformerConfig:
    def test_local_path_with_config(self, tmp_path):
        """Reads config.json from local model directory."""
        config_dir = tmp_path / "transformer"
        config_dir.mkdir()
        config_file = config_dir / "config.json"
        config_file.write_text(json.dumps(FULL_CONFIG))

        result = _load_transformer_config(str(tmp_path), local_files_only=True)
        assert result["hidden_size"] == 2520
        assert result["axes_dim_rope"] == [40, 40, 40]

    def test_local_path_without_config(self, tmp_path):
        """Returns empty dict when config.json doesn't exist."""
        result = _load_transformer_config(str(tmp_path), local_files_only=True)
        assert result == {}

    def test_online_hf_download(self):
        """Download transformer/config.json from HuggingFace for real."""

        result = _load_transformer_config("OmniGen2/OmniGen2", local_files_only=False)

        assert result["hidden_size"] == 2520
        assert result["num_layers"] == 32
        assert result["num_attention_heads"] == 21
        assert result["num_kv_heads"] == 7
        assert result["patch_size"] == 2
        assert result["axes_dim_rope"] == [40, 40, 40]
        assert result["axes_lens"] == [1024, 1664, 1664]

    def test_online_hf_download_failure(self):
        """Returns empty dict when HF download fails."""

        with patch(
            "huggingface_hub.hf_hub_download",
            side_effect=Exception("network error"),
        ):
            result = _load_transformer_config("org/model", local_files_only=False)
        assert result == {}
