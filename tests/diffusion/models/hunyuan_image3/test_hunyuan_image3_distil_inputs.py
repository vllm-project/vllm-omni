# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for HunyuanImage3 prepare_inputs_for_generation to ensure
distilled model parameters (guidance, timesteps_r) are properly passed."""

import pytest
import torch

from vllm_omni.diffusion.models.hunyuan_image3.hunyuan_image3_transformer import (
    HunyuanImage3Text2ImagePipeline,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


class MockHunyuanImage3Pipeline:
    """Minimal mock that replicates prepare_inputs_for_generation method."""

    def __init__(self):
        # Mock config for distilled models
        self.config = type("Config", (), {})()
        self.hf_config = type("HFConfig", (), {"cfg_distilled": True, "use_meanflow": True})()

    # Bind the real method from the pipeline class
    from vllm_omni.diffusion.models.hunyuan_image3.pipeline_hunyuan_image3 import (
        HunyuanImage3Pipeline as _Real,
    )

    prepare_inputs_for_generation = _Real.prepare_inputs_for_generation
    _update_model_kwargs_for_generation = _Real._update_model_kwargs_for_generation


class TestPrepareInputsForGenerationDistilled:
    """Tests that distilled model parameters are passed through
    prepare_inputs_for_generation."""

    def setup_method(self):
        self.pipeline = MockHunyuanImage3Pipeline()

    def _create_mock_kwargs(self, include_distilled_params: bool = True) -> dict:
        """Create mock kwargs for prepare_inputs_for_generation."""
        kwargs = {
            "position_ids": torch.tensor([[0, 1, 2]]),
            "custom_pos_emb": (torch.tensor([[0.0]]), torch.tensor([[1.0]])),
            "mode": "gen_image",
            "images": torch.randn(1, 4, 64, 64),
            "image_mask": torch.ones(1, 64, 64),
            "timestep": torch.tensor([0.5]),
            "gen_timestep_scatter_index": torch.tensor([[5]]),
            "cond_vae_images": None,
            "cond_timestep": None,
            "cond_vae_image_mask": None,
            "cond_vit_images": None,
            "cond_vit_image_mask": None,
            "vit_kwargs": None,
            "cond_timestep_scatter_index": None,
            "query_lens": [10],
            "seq_lens": [100],
            "num_image_tokens": 4096,
            "num_special_tokens": 3,
            "ar_kv_reuse_len": 0,
            "full_attn_spans": None,
            "use_cache": True,
        }
        if include_distilled_params:
            # CFG distilled model parameters
            kwargs["guidance"] = torch.tensor([2500.0])
            kwargs["guidance_scatter_index"] = torch.tensor([[4]])
            # MeanFlow distilled model parameters
            kwargs["timesteps_r"] = torch.tensor([0.3])
            kwargs["timesteps_r_scatter_index"] = torch.tensor([[3]])
        else:
            kwargs["guidance"] = None
            kwargs["guidance_scatter_index"] = None
            kwargs["timesteps_r"] = None
            kwargs["timesteps_r_scatter_index"] = None
        return kwargs

    def test_guidance_passed_to_model_inputs(self):
        """Test that guidance is passed through prepare_inputs_for_generation."""
        kwargs = self._create_mock_kwargs(include_distilled_params=True)
        input_ids = torch.tensor([[1, 2, 3]])
        attention_mask = torch.ones(1, 3)

        model_inputs = self.pipeline.prepare_inputs_for_generation(
            input_ids=input_ids,
            past_key_values=None,
            attention_mask=attention_mask,
            inputs_embeds=None,
            tokenizer_output=None,
            batch_gen_image_info=None,
            generator=None,
            **kwargs,
        )

        assert "guidance" in model_inputs, "guidance should be in model_inputs"
        assert model_inputs["guidance"] is not None, "guidance should not be None"
        assert torch.equal(model_inputs["guidance"], kwargs["guidance"])

    def test_guidance_scatter_index_passed_to_model_inputs(self):
        """Test that guidance_scatter_index is passed through."""
        kwargs = self._create_mock_kwargs(include_distilled_params=True)
        input_ids = torch.tensor([[1, 2, 3]])
        attention_mask = torch.ones(1, 3)

        model_inputs = self.pipeline.prepare_inputs_for_generation(
            input_ids=input_ids,
            past_key_values=None,
            attention_mask=attention_mask,
            inputs_embeds=None,
            tokenizer_output=None,
            batch_gen_image_info=None,
            generator=None,
            **kwargs,
        )

        assert "guidance_scatter_index" in model_inputs
        assert model_inputs["guidance_scatter_index"] is not None
        assert torch.equal(model_inputs["guidance_scatter_index"], kwargs["guidance_scatter_index"])

    def test_timesteps_r_passed_to_model_inputs(self):
        """Test that timesteps_r is passed through for MeanFlow models."""
        kwargs = self._create_mock_kwargs(include_distilled_params=True)
        input_ids = torch.tensor([[1, 2, 3]])
        attention_mask = torch.ones(1, 3)

        model_inputs = self.pipeline.prepare_inputs_for_generation(
            input_ids=input_ids,
            past_key_values=None,
            attention_mask=attention_mask,
            inputs_embeds=None,
            tokenizer_output=None,
            batch_gen_image_info=None,
            generator=None,
            **kwargs,
        )

        assert "timesteps_r" in model_inputs, "timesteps_r should be in model_inputs"
        assert model_inputs["timesteps_r"] is not None
        assert torch.equal(model_inputs["timesteps_r"], kwargs["timesteps_r"])

    def test_timesteps_r_scatter_index_passed_to_model_inputs(self):
        """Test that timesteps_r_scatter_index is passed through."""
        kwargs = self._create_mock_kwargs(include_distilled_params=True)
        input_ids = torch.tensor([[1, 2, 3]])
        attention_mask = torch.ones(1, 3)

        model_inputs = self.pipeline.prepare_inputs_for_generation(
            input_ids=input_ids,
            past_key_values=None,
            attention_mask=attention_mask,
            inputs_embeds=None,
            tokenizer_output=None,
            batch_gen_image_info=None,
            generator=None,
            **kwargs,
        )

        assert "timesteps_r_scatter_index" in model_inputs
        assert model_inputs["timesteps_r_scatter_index"] is not None
        assert torch.equal(model_inputs["timesteps_r_scatter_index"], kwargs["timesteps_r_scatter_index"])

    def test_num_special_tokens_passed_to_model_inputs(self):
        """Test that num_special_tokens is passed through."""
        kwargs = self._create_mock_kwargs(include_distilled_params=True)
        input_ids = torch.tensor([[1, 2, 3]])
        attention_mask = torch.ones(1, 3)

        model_inputs = self.pipeline.prepare_inputs_for_generation(
            input_ids=input_ids,
            past_key_values=None,
            attention_mask=attention_mask,
            inputs_embeds=None,
            tokenizer_output=None,
            batch_gen_image_info=None,
            generator=None,
            **kwargs,
        )

        assert "num_special_tokens" in model_inputs
        assert model_inputs["num_special_tokens"] == kwargs["num_special_tokens"]

    def test_decode_update_keeps_all_special_token_positions(self):
        """Test decode step uses timestep, guidance, and timestep_r positions."""
        model_kwargs = self._create_mock_kwargs(include_distilled_params=True)
        model_kwargs.update(
            {
                "tokenizer_output": object(),
                "attention_mask": torch.ones(1, 1, 7, 7, dtype=torch.bool),
                "image_mask": torch.tensor([[False, False, False, False, True, True, True]]),
                "gen_timestep_scatter_index": torch.tensor([[1]]),
                "guidance_scatter_index": torch.tensor([[2]]),
                "timesteps_r_scatter_index": torch.tensor([[3]]),
            }
        )

        updated = self.pipeline._update_model_kwargs_for_generation({}, model_kwargs)

        assert torch.equal(updated["position_ids"], torch.tensor([[1, 2, 3, 4, 5, 6]]))
        assert updated["num_special_tokens"] == 3
        assert updated["guidance_scatter_index"] is model_kwargs["guidance_scatter_index"]
        assert updated["timesteps_r_scatter_index"] is model_kwargs["timesteps_r_scatter_index"]
        assert updated["attention_mask"].shape == (1, 1, 6, 7)

    def test_distilled_params_none_when_not_provided(self):
        """Test that distilled params are None when not provided."""
        kwargs = self._create_mock_kwargs(include_distilled_params=False)
        input_ids = torch.tensor([[1, 2, 3]])
        attention_mask = torch.ones(1, 3)

        model_inputs = self.pipeline.prepare_inputs_for_generation(
            input_ids=input_ids,
            past_key_values=None,
            attention_mask=attention_mask,
            inputs_embeds=None,
            tokenizer_output=None,
            batch_gen_image_info=None,
            generator=None,
            **kwargs,
        )

        # All distilled params should be present but None
        assert "guidance" in model_inputs
        assert model_inputs["guidance"] is None
        assert "guidance_scatter_index" in model_inputs
        assert model_inputs["guidance_scatter_index"] is None
        assert "timesteps_r" in model_inputs
        assert model_inputs["timesteps_r"] is None
        assert "timesteps_r_scatter_index" in model_inputs
        assert model_inputs["timesteps_r_scatter_index"] is None

    def test_all_distilled_keys_present_in_model_inputs(self):
        """Test that all 4 distilled keys are always present (even if None)."""
        kwargs = self._create_mock_kwargs(include_distilled_params=False)
        input_ids = torch.tensor([[1, 2, 3]])
        attention_mask = torch.ones(1, 3)

        model_inputs = self.pipeline.prepare_inputs_for_generation(
            input_ids=input_ids,
            past_key_values=None,
            attention_mask=attention_mask,
            inputs_embeds=None,
            tokenizer_output=None,
            batch_gen_image_info=None,
            generator=None,
            **kwargs,
        )

        required_distilled_keys = [
            "guidance",
            "guidance_scatter_index",
            "timesteps_r",
            "timesteps_r_scatter_index",
        ]
        for key in required_distilled_keys:
            assert key in model_inputs, f"{key} must be in model_inputs dict"


def test_ar_kv_reuse_rebases_distilled_scatter_indices():
    pipeline = object.__new__(HunyuanImage3Text2ImagePipeline)
    model_kwargs = {
        "query_lens": [8],
        "attention_mask": torch.ones(1, 1, 8, 12, dtype=torch.bool),
        "position_ids": torch.arange(8).unsqueeze(0),
        "image_mask": torch.ones(1, 8, dtype=torch.bool),
        "gen_timestep_scatter_index": torch.tensor([[5]]),
        "guidance_scatter_index": torch.tensor([[6]]),
        "timesteps_r_scatter_index": torch.tensor([[7]]),
    }

    truncated = pipeline._truncate_reused_prefix(
        input_ids=torch.arange(8).unsqueeze(0),
        model_kwargs=model_kwargs,
        positive_reuse_len=3,
    )

    assert truncated.tolist() == [[3, 4, 5, 6, 7]]
    assert model_kwargs["gen_timestep_scatter_index"].tolist() == [[2]]
    assert model_kwargs["guidance_scatter_index"].tolist() == [[3]]
    assert model_kwargs["timesteps_r_scatter_index"].tolist() == [[4]]


def test_cfg_parallel_splits_meanflow_scatter_indices():
    model_kwargs = {
        "timesteps_r_scatter_index": torch.tensor([[1], [2], [3], [4]]),
    }

    HunyuanImage3Text2ImagePipeline._split_model_kwargs_for_cfg_parallel(
        model_kwargs,
        batch_size=2,
        cfg_rank=1,
    )

    assert model_kwargs["timesteps_r_scatter_index"].tolist() == [[3], [4]]
