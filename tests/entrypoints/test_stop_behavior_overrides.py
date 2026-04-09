from unittest.mock import MagicMock

from vllm.sampling_params import SamplingParams


class TestStopBehaviorOverrides:
    """Test that stop-behavior fields propagate to all stages."""

    def _make_handler(self, default_params_list, comprehension_idx=0):
        """Create a minimal OmniOpenAIServingChat for testing."""
        from vllm_omni.entrypoints.openai.serving_chat import (
            OmniOpenAIServingChat,
        )

        handler = OmniOpenAIServingChat.__new__(OmniOpenAIServingChat)
        handler.engine_client = MagicMock()
        handler.engine_client.default_sampling_params_list = default_params_list
        handler._get_comprehension_stage_index = MagicMock(
            return_value=comprehension_idx,
        )
        return handler

    def test_ignore_eos_propagated_to_all_stages(self):
        """ignore_eos=True should apply to every stage."""
        default_params_list = [
            SamplingParams(temperature=0.4, max_tokens=2048),
            SamplingParams(temperature=0.9, max_tokens=4096),
            SamplingParams(temperature=0.0, max_tokens=65536),
        ]

        request = MagicMock()
        request.ignore_eos = True

        handler = self._make_handler(default_params_list)
        result = handler._build_sampling_params_list_from_request(request)

        for idx, params in enumerate(result):
            assert params.ignore_eos is True, f"Stage {idx}: ignore_eos should be True"

    def test_other_stages_keep_own_temperature(self):
        """Temperature should NOT be overridden on non-comprehension stages."""
        default_params_list = [
            SamplingParams(temperature=0.4, max_tokens=2048),
            SamplingParams(temperature=0.9, max_tokens=4096),
        ]

        request = MagicMock()
        request.temperature = 0.1
        request.ignore_eos = True

        handler = self._make_handler(default_params_list)
        result = handler._build_sampling_params_list_from_request(request)

        # Stage 1 should keep its own temperature
        assert result[1].temperature == 0.9

    def test_ignore_eos_none_leaves_default(self):
        """When ignore_eos is not set, default should be preserved."""
        default_params_list = [
            SamplingParams(max_tokens=2048),
            SamplingParams(max_tokens=4096),
        ]

        request = MagicMock()
        request.ignore_eos = None

        handler = self._make_handler(default_params_list)
        result = handler._build_sampling_params_list_from_request(request)

        assert result[1].ignore_eos is False
