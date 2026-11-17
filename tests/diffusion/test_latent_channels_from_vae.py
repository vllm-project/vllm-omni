# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import pytest

from vllm_omni.diffusion.models.utils import latent_channels_from_vae

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


class _FakeConfig:
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


class _FakeVAE:
    def __init__(self, config=None):
        if config is not None:
            self.config = config


class _FakeDictConfig(dict):
    pass


class TestLatentChannelsFromVae:
    def test_standard_latent_channels(self):
        vae = _FakeVAE(_FakeConfig(latent_channels=16))
        assert latent_channels_from_vae(vae) == 16

    def test_custom_latent_channels(self):
        vae = _FakeVAE(_FakeConfig(latent_channels=32))
        assert latent_channels_from_vae(vae) == 32

    def test_z_dim_config_key(self):
        vae = _FakeVAE(_FakeConfig(z_dim=16))
        assert latent_channels_from_vae(vae, config_key="z_dim") == 16

    def test_none_vae_returns_default(self):
        assert latent_channels_from_vae(None) == 16

    def test_none_vae_returns_custom_default(self):
        assert latent_channels_from_vae(None, default=32) == 32

    def test_missing_config_returns_default(self):
        vae = _FakeVAE()
        assert latent_channels_from_vae(vae) == 16

    def test_missing_key_returns_default(self):
        vae = _FakeVAE(_FakeConfig())
        assert latent_channels_from_vae(vae) == 16

    def test_dict_config_access(self):
        config = _FakeDictConfig(latent_channels=24)
        vae = _FakeVAE(config)
        assert latent_channels_from_vae(vae) == 24

    def test_hunyuan_default_32(self):
        vae = _FakeVAE(_FakeConfig(latent_channels=32))
        assert latent_channels_from_vae(vae, default=32) == 32

    def test_qwen_z_dim_with_none_vae(self):
        assert latent_channels_from_vae(None, config_key="z_dim") == 16
