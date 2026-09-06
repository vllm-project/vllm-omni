# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""L1 unit tests for vllm_omni.inputs.data module."""

import pytest
import torch

from vllm_omni.inputs.data import (
    OmniDiffusionSamplingParams,
    OmniEmbedsPrompt,
    OmniTextPrompt,
    OmniTokenInputs,
    OmniTokensPrompt,
    token_inputs_omni,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


# ---------------------------------------------------------------------------
# token_inputs_omni
# ---------------------------------------------------------------------------
class TestTokenInputsOmni:
    """Tests for the token_inputs_omni helper function."""

    def test_minimal(self) -> None:
        """Only required arg (prompt_token_ids) is provided."""
        result = token_inputs_omni(prompt_token_ids=[1, 2, 3])
        assert result["type"] == "token"
        assert result["prompt_token_ids"] == [1, 2, 3]
        assert "prompt" not in result
        assert "cache_salt" not in result
        assert "prompt_embeds" not in result
        assert "additional_information" not in result

    def test_with_prompt(self) -> None:
        """Prompt string is attached when provided."""
        result = token_inputs_omni(prompt_token_ids=[1], prompt="hello")
        assert result["prompt"] == "hello"

    def test_with_cache_salt(self) -> None:
        """Cache salt is attached when provided."""
        result = token_inputs_omni(prompt_token_ids=[1], cache_salt="salt-1")
        assert result["cache_salt"] == "salt-1"

    def test_with_prompt_embeds(self) -> None:
        """Prompt embeddings tensor is attached when provided."""
        embeds = torch.randn(3, 768)
        result = token_inputs_omni(prompt_token_ids=[1, 2, 3], prompt_embeds=embeds)
        assert torch.equal(result["prompt_embeds"], embeds)

    def test_with_additional_information(self) -> None:
        """Additional information dict is attached when provided."""
        info = {"speaker_id": 42, "codes": torch.zeros(10)}
        result = token_inputs_omni(prompt_token_ids=[1], additional_information=info)
        assert result["additional_information"]["speaker_id"] == 42

    def test_all_optional_fields(self) -> None:
        """All optional fields are set when every argument is provided."""
        embeds = torch.randn(2, 128)
        info = {"key": "value"}
        result = token_inputs_omni(
            prompt_token_ids=[10, 20],
            prompt="test",
            cache_salt="s",
            prompt_embeds=embeds,
            additional_information=info,
        )
        assert result["prompt"] == "test"
        assert result["cache_salt"] == "s"
        assert torch.equal(result["prompt_embeds"], embeds)
        assert result["additional_information"] == info

    def test_empty_token_ids(self) -> None:
        """Empty token ID list is accepted."""
        result = token_inputs_omni(prompt_token_ids=[])
        assert result["prompt_token_ids"] == []


# ---------------------------------------------------------------------------
# OmniDiffusionSamplingParams
# ---------------------------------------------------------------------------
class TestOmniDiffusionSamplingParams:
    """Tests for the OmniDiffusionSamplingParams dataclass."""

    def test_defaults(self) -> None:
        """Default values match the documented interface."""
        params = OmniDiffusionSamplingParams()
        assert params.num_inference_steps is None
        assert params.guidance_scale == 0.0
        assert params.num_outputs_per_prompt == 1
        assert params.num_frames == 1
        assert params.batch_size == 1
        assert params.save_output is True
        assert params.debug is False

    def test_batch_size_reflects_num_outputs(self) -> None:
        """batch_size property should equal num_outputs_per_prompt."""
        params = OmniDiffusionSamplingParams(num_outputs_per_prompt=4)
        assert params.batch_size == 4

    def test_clone_produces_independent_copy(self) -> None:
        """clone() returns a deep copy that is independent."""
        params = OmniDiffusionSamplingParams(
            num_inference_steps=30,
            extra_args={"key": [1, 2, 3]},
        )
        cloned = params.clone()
        assert cloned.num_inference_steps == 30
        assert cloned.extra_args == {"key": [1, 2, 3]}

        cloned.num_inference_steps = 99
        cloned.extra_args["key"].append(4)
        assert params.num_inference_steps == 30
        assert params.extra_args["key"] == [1, 2, 3]

    def test_str_representation(self) -> None:
        """__str__ returns a non-empty string representation."""
        params = OmniDiffusionSamplingParams(seed=42)
        text = str(params)
        assert "seed" in text
        assert "42" in text

    def test_seed_assignment(self) -> None:
        """Seed can be set via constructor."""
        params = OmniDiffusionSamplingParams(seed=123)
        assert params.seed == 123

    def test_lora_fields(self) -> None:
        """LoRA-related fields have correct defaults."""
        params = OmniDiffusionSamplingParams()
        assert params.lora_request is None
        assert params.lora_scale == 1.0

    def test_latent_fields_default_none(self) -> None:
        """All latent tensor fields default to None."""
        params = OmniDiffusionSamplingParams()
        assert params.latents is None
        assert params.noise_pred is None
        assert params.image_latent is None
        assert params.timesteps is None
        assert params.output is None


# ---------------------------------------------------------------------------
# OmniTextPrompt / OmniTokensPrompt / OmniEmbedsPrompt (TypedDict checks)
# ---------------------------------------------------------------------------
class TestOmniPromptTypes:
    """Tests for OmniTextPrompt, OmniTokensPrompt, OmniEmbedsPrompt."""

    def test_omni_text_prompt_basic(self) -> None:
        """OmniTextPrompt accepts standard TextPrompt fields."""
        prompt: OmniTextPrompt = {"prompt": "Describe the scene."}
        assert prompt["prompt"] == "Describe the scene."

    def test_omni_text_prompt_with_extra_fields(self) -> None:
        """OmniTextPrompt accepts negative_prompt and additional_information."""
        embeds = torch.randn(1, 128)
        prompt: OmniTextPrompt = {
            "prompt": "A cat",
            "negative_prompt": "blurry",
            "prompt_embeds": embeds,
            "additional_information": {"style": "photo"},
        }
        assert prompt["negative_prompt"] == "blurry"
        assert prompt["additional_information"]["style"] == "photo"

    def test_omni_tokens_prompt_basic(self) -> None:
        """OmniTokensPrompt accepts prompt_token_ids."""
        prompt: OmniTokensPrompt = {"prompt_token_ids": [1, 2, 3]}
        assert prompt["prompt_token_ids"] == [1, 2, 3]

    def test_omni_token_inputs_type_field(self) -> None:
        """OmniTokenInputs must have type='token'."""
        inputs: OmniTokenInputs = {"type": "token", "prompt_token_ids": [5, 6]}
        assert inputs["type"] == "token"

    def test_omni_embeds_prompt_with_additional_info(self) -> None:
        """OmniEmbedsPrompt accepts additional_information."""
        embeds = torch.randn(1, 4, 768)
        prompt: OmniEmbedsPrompt = {
            "prompt_embeds": embeds,
            "additional_information": {"duration": 5.0},
        }
        assert prompt["additional_information"]["duration"] == 5.0
