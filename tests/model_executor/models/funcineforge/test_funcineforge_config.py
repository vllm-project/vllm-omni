# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for FunCineForge configuration."""

import pytest

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


class TestFunCineForgeConfig:
    """Tests for FunCineForgeConfig dataclass."""

    def test_default_instantiation(self):
        """Config can be instantiated with all defaults."""
        from vllm_omni.model_executor.models.funcineforge.config import (
            FunCineForgeConfig,
        )

        cfg = FunCineForgeConfig()
        assert cfg.codec_unit == 6761
        assert cfg.timespk_unit == 1550
        assert cfg.face_size == 512
        assert cfg.sample_rate == 24000

    def test_special_token_ids(self):
        """Special token IDs are consistent (SOS < EOS < turn < fill)."""
        from vllm_omni.model_executor.models.funcineforge.config import (
            FunCineForgeConfig,
        )

        cfg = FunCineForgeConfig()
        assert cfg.sos == 6561
        assert cfg.eos == 6562
        assert cfg.turn_of_speech == 6563
        assert cfg.fill_token == 6564
        # SOS is the first flow codebook sentinel; codec_unit includes
        # additional LM-side reserved IDs in the official checkpoint.
        assert cfg.sos == cfg.flow["codebook_size"]

    def test_timespeaker_tag_ids(self):
        """Timespeaker tag IDs are in expected ranges."""
        from vllm_omni.model_executor.models.funcineforge.config import (
            FunCineForgeConfig,
        )

        cfg = FunCineForgeConfig()
        # Type IDs: 1500-1503
        assert cfg.pangbai == 1500
        assert cfg.duoren == 1503
        # Gender: 1504-1505
        assert cfg.male == 1504
        assert cfg.female == 1505
        # Age: 1506-1510
        assert cfg.child == 1506
        assert cfg.elderly == 1510
        # Speaker IDs start after age
        assert cfg.speaker_id_start == 1511

    def test_flow_config_defaults(self):
        """Flow config contains expected keys and values."""
        from vllm_omni.model_executor.models.funcineforge.config import (
            FunCineForgeConfig,
        )

        cfg = FunCineForgeConfig()
        assert cfg.flow["codebook_size"] == 6561
        assert cfg.flow["dit_conf"]["dim"] == 1024
        assert cfg.flow["dit_conf"]["depth"] == 22
        assert cfg.flow["mel_feat_conf"]["n_mel_channels"] == 80
        assert cfg.flow["feat_token_ratio"] == 2

    def test_vocoder_config_defaults(self):
        """Vocoder config contains expected keys."""
        from vllm_omni.model_executor.models.funcineforge.config import (
            FunCineForgeConfig,
        )

        cfg = FunCineForgeConfig()
        hifigan = cfg.vocoder["CausalHiFTGenerator_conf"]
        assert hifigan["in_channels"] == 80
        assert hifigan["sampling_rate"] == 24000
        assert hifigan["upsample_rates"] == [8, 5, 3]

    def test_generation_limits(self):
        """Generation limits are consistent with token rate."""
        from vllm_omni.model_executor.models.funcineforge.config import (
            FunCineForgeConfig,
        )

        cfg = FunCineForgeConfig()
        # max_length = 60s * 25 fps = 1500
        assert cfg.max_length == cfg.token_rate * 60
        # min_length = 2s * 25 fps = 50
        assert cfg.min_length == cfg.token_rate * 2

    def test_custom_instantiation(self):
        """Config can be instantiated with custom values."""
        from vllm_omni.model_executor.models.funcineforge.config import (
            FunCineForgeConfig,
        )

        cfg = FunCineForgeConfig(codec_unit=8192, sample_rate=16000)
        assert cfg.codec_unit == 8192
        assert cfg.sample_rate == 16000
        # Other defaults unchanged
        assert cfg.face_size == 512

    def test_checkpoint_paths(self):
        """Checkpoint paths follow DeepSpeed layout."""
        from vllm_omni.model_executor.models.funcineforge.config import (
            FunCineForgeConfig,
        )

        cfg = FunCineForgeConfig()
        assert "mp_rank_00_model_states.pt" in cfg.llm_ckpt
        assert "mp_rank_00_model_states.pt" in cfg.flow_ckpt
        assert "avg_5_removewn.pt" in cfg.vocoder_ckpt
