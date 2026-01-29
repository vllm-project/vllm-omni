# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for OmniRequestOutput class."""

from unittest.mock import Mock

from PIL import Image

from vllm_omni.outputs import OmniRequestOutput


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

    def test_from_pipeline(self):
        """Test creating output from pipeline stage."""
        mock_request_output = Mock()
        mock_request_output.request_id = "pipeline-123"
        mock_request_output.prompt_token_ids = [1, 2, 3]
        mock_request_output.outputs = [Mock()]
        mock_request_output.encoder_prompt_token_ids = None
        mock_request_output.prompt_logprobs = None
        mock_request_output.num_cached_tokens = 10
        mock_request_output.kv_transfer_params = None
        mock_request_output.multimodal_output = {"image": Mock()}

        output = OmniRequestOutput.from_pipeline(
            stage_id=0,
            final_output_type="text",
            request_output=mock_request_output,
        )

        assert output.request_id == "pipeline-123"
        assert output.stage_id == 0
        assert output.final_output_type == "text"
        assert output.is_pipeline_output

    def test_prompt_token_ids_property(self):
        """Test prompt_token_ids property for streaming compatibility."""
        mock_request_output = Mock()
        mock_request_output.prompt_token_ids = [1, 2, 3, 4, 5]

        output = OmniRequestOutput.from_pipeline(
            stage_id=0,
            final_output_type="text",
            request_output=mock_request_output,
        )

        assert output.prompt_token_ids == [1, 2, 3, 4, 5]

    def test_prompt_token_ids_none_when_no_request_output(self):
        """Test prompt_token_ids returns None when no request_output."""
        output = OmniRequestOutput.from_diffusion(
            request_id="test-123",
            images=[],
            prompt="a cat",
        )
        assert output.prompt_token_ids is None

    def test_outputs_property(self):
        """Test outputs property for chat completion compatibility."""
        mock_output = Mock()
        mock_request_output = Mock()
        mock_request_output.outputs = [mock_output]

        output = OmniRequestOutput.from_pipeline(
            stage_id=0,
            final_output_type="text",
            request_output=mock_request_output,
        )

        assert output.outputs == [mock_output]

    def test_outputs_empty_when_no_request_output(self):
        """Test outputs returns empty list when no request_output."""
        output = OmniRequestOutput.from_diffusion(
            request_id="test-123",
            images=[],
            prompt="a cat",
        )
        assert output.outputs == []

    def test_encoder_prompt_token_ids_property(self):
        """Test encoder_prompt_token_ids property."""
        mock_request_output = Mock()
        mock_request_output.encoder_prompt_token_ids = [10, 20, 30]

        output = OmniRequestOutput.from_pipeline(
            stage_id=0,
            final_output_type="text",
            request_output=mock_request_output,
        )

        assert output.encoder_prompt_token_ids == [10, 20, 30]

    def test_num_cached_tokens_property(self):
        """Test num_cached_tokens property."""
        mock_request_output = Mock()
        mock_request_output.num_cached_tokens = 42

        output = OmniRequestOutput.from_pipeline(
            stage_id=0,
            final_output_type="text",
            request_output=mock_request_output,
        )

        assert output.num_cached_tokens == 42

    def test_multimodal_output_property(self):
        """Test multimodal_output property."""
        mock_request_output = Mock()
        mock_audio = Mock()
        expected_output = {"audio": mock_audio}
        mock_request_output.multimodal_output = expected_output

        output = OmniRequestOutput.from_pipeline(
            stage_id=0,
            final_output_type="audio",
            request_output=mock_request_output,
        )

        assert output.multimodal_output is expected_output

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

    def test_to_dict_pipeline(self):
        """Test to_dict for pipeline output."""
        mock_request_output = Mock()
        mock_request_output.request_id = "pipeline-123"

        output = OmniRequestOutput.from_pipeline(
            stage_id=0,
            final_output_type="text",
            request_output=mock_request_output,
        )
        result = output.to_dict()

        assert result["request_id"] == "pipeline-123"
        assert result["finished"] is True
        assert result["final_output_type"] == "text"
        assert result["stage_id"] == 0
