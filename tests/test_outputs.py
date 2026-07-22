# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for OmniRequestOutput class."""

import pytest
from PIL import Image

from vllm_omni.outputs import OmniRequestOutput

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


class TestOmniRequestOutput:
    """Tests for OmniRequestOutput class."""

    def test_from_diffusion(self):
        """Test creating output from diffusion model."""
        images = [Image.new("RGB", (64, 64), color="red")]
        output = OmniRequestOutput.from_diffusion(
            request_id="test-123",
            images=images,
            prompt="a cat",
            metrics={"steps": 50},
        )
        assert output.request_id == "test-123"
        assert output.images == images
        assert output.prompt == "a cat"
        assert output.metrics == {"steps": 50}
        assert output.is_diffusion_output
        assert output.num_images == 1

    def test_prompt_token_ids_none_when_no_request_output(self):
        """Test prompt_token_ids returns None when no request_output."""
        output = OmniRequestOutput.from_diffusion(
            request_id="test-123",
            images=[],
            prompt="a cat",
        )
        assert output.prompt_token_ids is None

    def test_outputs_empty_when_no_request_output(self):
        """Test outputs returns empty list when no request_output."""
        output = OmniRequestOutput.from_diffusion(
            request_id="test-123",
            images=[],
            prompt="a cat",
        )
        assert output.outputs == []

    def test_to_dict_diffusion(self):
        """Test to_dict for diffusion output."""
        output = OmniRequestOutput.from_diffusion(
            request_id="test-123",
            images=[Image.new("RGB", (64, 64), color="red")],
            prompt="a cat",
            metrics={"steps": 50},
        )
        result = output.to_dict()

        assert result["request_id"] == "test-123"
        assert result["finished"] is True
        assert result["final_output_type"] == "image"
        assert result["num_images"] == 1
        assert result["prompt"] == "a cat"
