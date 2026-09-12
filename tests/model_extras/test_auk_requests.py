# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Unit tests for the AuK request helpers."""

from __future__ import annotations

import numpy as np
import pytest
from vllm import SamplingParams

from vllm_omni.inputs.data import OmniDiffusionSamplingParams
from vllm_omni.model_extras.auk import (
    NO_PROMPT_AUDIO,
    auk_prompt,
    auk_sampling_params,
    resolve_gen_frames,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

INSTRUCTION = "Say the following with the same voice: 'Ladies and gentlemen'"

EXPECTED_WITH_AUDIO = (
    "<|im_start|>system\n"
    "You are a helpful assistant.<|im_end|>\n"
    "<|im_start|>user\n"
    f"{INSTRUCTION}<|audio_bos|><|AUDIO|><|audio_eos|><|im_end|>\n"
    "<|im_start|>assistant\n"
)

EXPECTED_TEXT_ONLY = (
    "<|im_start|>system\n"
    "You are a helpful assistant.<|im_end|>\n"
    "<|im_start|>user\n"
    f"{INSTRUCTION}{NO_PROMPT_AUDIO}<|im_end|>\n"
    "<|im_start|>assistant\n"
)


def _audio() -> tuple[np.ndarray, int]:
    return np.zeros(24000, dtype=np.float32), 24000


class TestPromptRendering:
    def test_with_audio(self):
        audio = _audio()
        prompt = auk_prompt(INSTRUCTION, audio, gen_seconds=6.0)
        assert prompt["prompt"] == EXPECTED_WITH_AUDIO
        assert prompt["multi_modal_data"] == {"audio": audio}
        assert prompt["additional_information"]["auk"]["has_audio"] is True

    def test_text_only_appends_the_marker(self):
        prompt = auk_prompt(INSTRUCTION, gen_seconds=6.0)
        assert prompt["prompt"] == EXPECTED_TEXT_ONLY
        assert "multi_modal_data" not in prompt
        assert prompt["additional_information"]["auk"]["has_audio"] is False

    def test_marker_is_not_appended_twice(self):
        once = auk_prompt(INSTRUCTION, gen_seconds=6.0)
        twice = auk_prompt(INSTRUCTION + NO_PROMPT_AUDIO, gen_seconds=6.0)
        assert once["prompt"] == twice["prompt"]
        assert twice["prompt"].count(NO_PROMPT_AUDIO) == 1

    def test_marker_is_not_added_when_audio_is_present(self):
        prompt = auk_prompt(INSTRUCTION, _audio())
        assert NO_PROMPT_AUDIO not in prompt["prompt"]

    def test_knob_defaults(self):
        knobs = auk_prompt(INSTRUCTION, _audio())["additional_information"]["auk"]
        assert knobs == {
            "gen_seconds": None,
            "sway": -1.0,
            "t_grid": None,
            "vae_sample": False,
            "has_audio": True,
        }

    def test_knobs_are_carried(self):
        grid = [0.0, 0.25, 1.0]
        knobs = auk_prompt(
            INSTRUCTION,
            _audio(),
            gen_seconds=3.5,
            sway=0.0,
            t_grid=grid,
            vae_sample=True,
        )["additional_information"]["auk"]
        assert knobs["gen_seconds"] == 3.5
        assert knobs["sway"] == 0.0
        assert knobs["vae_sample"] is True
        assert knobs["t_grid"] == grid
        # A copy, so a caller mutating its list cannot reach the request.
        assert knobs["t_grid"] is not grid

    def test_sampling_knobs_are_absent_from_the_prompt(self):
        knobs = auk_prompt(INSTRUCTION, _audio())["additional_information"]["auk"]
        assert not {"nfe", "cfg", "seed"} & set(knobs)


class TestSamplingParams:
    def test_two_stages_in_order(self):
        params = auk_sampling_params()
        assert len(params) == 2
        assert isinstance(params[0], SamplingParams)
        assert isinstance(params[1], OmniDiffusionSamplingParams)

    def test_encoder_stage_is_prefill_only(self):
        encoder = auk_sampling_params()[0]
        assert encoder.max_tokens == 1
        assert encoder.temperature == 0.0
        assert encoder.detokenize is False

    def test_defaults_are_the_base_recipe(self):
        diffusion = auk_sampling_params()[1]
        assert diffusion.num_inference_steps == 32
        assert diffusion.guidance_scale == 2.0
        assert diffusion.seed == 0

    def test_overrides(self):
        encoder, diffusion = auk_sampling_params(nfe=4, cfg=0.0, seed=None)
        assert diffusion.num_inference_steps == 4
        assert diffusion.guidance_scale == 0.0
        assert diffusion.seed is None
        assert encoder.seed is None


class TestDurationPolicy:
    @pytest.mark.parametrize(
        ("gen_seconds", "expected"),
        [
            (6.0, 300),
            (0.02, 1),
            (0.001, 1),
            (5.59, 280),
            (1.0, 50),
        ],
    )
    def test_explicit_duration_rounds_up(self, gen_seconds: float, expected: int):
        assert resolve_gen_frames(gen_seconds, 0) == expected

    def test_explicit_duration_wins_over_the_source(self):
        assert resolve_gen_frames(2.0, 500) == 100

    def test_source_length_is_the_default(self):
        assert resolve_gen_frames(None, 137) == 137

    def test_text_only_without_a_duration_raises(self):
        with pytest.raises(ValueError, match="gen_seconds is required"):
            resolve_gen_frames(None, 0)

    @pytest.mark.parametrize("gen_seconds", [0.0, -1.0])
    def test_non_positive_duration_raises(self, gen_seconds: float):
        with pytest.raises(ValueError, match="must be positive"):
            resolve_gen_frames(gen_seconds, 100)

    def test_custom_rate(self):
        assert resolve_gen_frames(1.0, 0, sample_rate=16000, hop=320) == 50
