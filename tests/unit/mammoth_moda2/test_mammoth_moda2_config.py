# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Unit tests for MammothModa2 config classes.

These tests verify the correct behavior of the Mammothmoda2Config class,
particularly the get_text_config() method which is called during parent
class initialization (PretrainedConfig.__init__ -> validation -> get_text_config).

Regression test for: AttributeError: 'Mammothmoda2Config' object has no attribute 'llm_config'
"""

import pytest

from vllm_omni.transformers_utils.configs.mammoth_moda2 import (
    Mammothmoda2Config,
    Mammothmoda2Qwen2_5_VLConfig,
    Mammothmoda2Qwen2_5_VLTextConfig,
)


@pytest.mark.cpu
class TestMammothmoda2Config:
    """Tests for Mammothmoda2Config class."""

    def test_get_text_config_returns_nested_text_config(self):
        """
        Verify that get_text_config() returns the nested text_config from llm_config.

        This is a regression test for the AttributeError that occurred when
        get_text_config() was called during parent class initialization before
        llm_config was set.
        """
        config = Mammothmoda2Config(
            llm_config={
                "model_type": "mammothmoda2_qwen2_5_vl",
            }
        )

        assert config.llm_config is not None
        assert isinstance(config.llm_config, Mammothmoda2Qwen2_5_VLConfig)

        text_config = config.get_text_config()
        assert text_config is not None
        assert isinstance(text_config, Mammothmoda2Qwen2_5_VLTextConfig)

        # The text_config should be the same object as llm_config.text_config
        assert text_config is config.llm_config.text_config

    def test_get_text_config_returns_none_when_llm_config_is_none(self):
        """Verify that get_text_config() returns None when llm_config is None."""
        config = Mammothmoda2Config(llm_config=None)

        assert config.llm_config is None
        assert config.get_text_config() is None

    def test_init_sets_llm_config_before_parent_init(self):
        """
        Verify that llm_config is set before parent __init__ is called.

        This is critical because the parent PretrainedConfig.__init__ calls
        validation which in turn calls get_text_config(). If llm_config is not
        set before super().__init__(), an AttributeError will be raised.
        """
        # This should not raise an AttributeError
        config = Mammothmoda2Config(
            llm_config={
                "model_type": "mammothmoda2_qwen2_5_vl",
            }
        )
        assert config.llm_config is not None


@pytest.mark.cpu
class TestMammothmoda2Qwen2_5_VLConfig:
    """Tests for Mammothmoda2Qwen2_5_VLConfig class."""

    def test_has_text_config_attribute(self):
        """Verify that Mammothmoda2Qwen2_5_VLConfig has text_config attribute."""
        config = Mammothmoda2Qwen2_5_VLConfig()
        assert hasattr(config, "text_config")
        assert isinstance(config.text_config, Mammothmoda2Qwen2_5_VLTextConfig)

    def test_text_config_has_expected_attributes(self):
        """Verify that the nested text_config has expected attributes for token validation."""
        config = Mammothmoda2Qwen2_5_VLConfig()
        text_config = config.text_config

        assert hasattr(text_config, "vocab_size")
        assert hasattr(text_config, "image_token_id")
        assert hasattr(text_config, "video_token_id")
