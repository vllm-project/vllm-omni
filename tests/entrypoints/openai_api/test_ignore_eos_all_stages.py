"""Test that ignore_eos from request is applied to all stages."""

from dataclasses import dataclass

import pytest
from vllm.sampling_params import SamplingParams

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


@dataclass
class FakeRequest:
    """Fake request with fields read by _build_sampling_params_list_from_request."""

    temperature: float | None = None
    ignore_eos: bool | None = None
    top_p: float | None = None
    top_k: int | None = None
    max_tokens: int | None = None
    stop: list | None = None


class TestIgnoreEosAllStages:
    """Test that ignore_eos applies to all stages, not just comprehension."""

    def _make_handler(self, default_params_list, comprehension_idx=0, mocker=None):
        from vllm_omni.entrypoints.openai.serving_chat import (
            OmniOpenAIServingChat,
        )

        handler = OmniOpenAIServingChat.__new__(OmniOpenAIServingChat)
        handler.engine_client = mocker.MagicMock()
        handler.engine_client.default_sampling_params_list = default_params_list
        handler._get_comprehension_stage_index = mocker.MagicMock(
            return_value=comprehension_idx,
        )
        return handler

    def test_ignore_eos_true_applies_to_all_stages(self, mocker):
        """Every stage should have ignore_eos=True when request sets it."""
        default_params_list = [
            SamplingParams(temperature=0.4, max_tokens=2048),
            SamplingParams(temperature=0.9, max_tokens=4096),
            SamplingParams(temperature=0.0, max_tokens=65536),
        ]

        request = FakeRequest(ignore_eos=True)
        handler = self._make_handler(default_params_list, mocker=mocker)
        result = handler._build_sampling_params_list_from_request(request)

        for idx, params in enumerate(result):
            assert params.ignore_eos is True, f"Stage {idx}: ignore_eos should be True"

    def test_ignore_eos_none_preserves_default(self, mocker):
        """When ignore_eos is None, stages should keep default (False)."""
        default_params_list = [
            SamplingParams(max_tokens=2048),
            SamplingParams(max_tokens=4096),
        ]

        request = FakeRequest(ignore_eos=None)
        handler = self._make_handler(default_params_list, mocker=mocker)
        result = handler._build_sampling_params_list_from_request(request)

        for idx, params in enumerate(result):
            assert params.ignore_eos is False, f"Stage {idx}: ignore_eos should remain False"
