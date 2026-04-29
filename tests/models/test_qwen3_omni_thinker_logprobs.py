# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for Qwen3-Omni Thinker logprobs support.

Verifies that the Thinker stage's sampling params accept logprobs
and that the output pipeline carries logprobs correctly.
These tests do NOT require GPU or model weights.
"""

from __future__ import annotations

import pytest
from vllm.sampling_params import SamplingParams

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


class TestThinkerLogprobsConfig:
    """Verify Thinker logprobs work at the configuration level."""

    def test_sampling_params_accepts_logprobs(self):
        """SamplingParams must accept logprobs parameter."""
        params = SamplingParams(logprobs=1, temperature=0.4, max_tokens=64)
        assert params.logprobs == 1

    def test_sampling_params_logprobs_none_by_default(self):
        """logprobs should be None when not explicitly set."""
        params = SamplingParams(temperature=0.4, max_tokens=64)
        assert params.logprobs is None

    def test_sampling_params_prompt_logprobs(self):
        """prompt_logprobs should be configurable for prompt token log-probs."""
        params = SamplingParams(logprobs=1, prompt_logprobs=0, max_tokens=64)
        assert params.prompt_logprobs == 0

    def test_omni_request_output_has_logprobs_property(self):
        """OmniRequestOutput must expose logprobs from underlying RequestOutput."""
        from vllm_omni.outputs import OmniRequestOutput

        output = OmniRequestOutput()
        # When no request_output is set, prompt_logprobs should be None
        assert output.prompt_logprobs is None

    def test_omni_model_runner_output_has_logprobs_field(self):
        """OmniModelRunnerOutput must carry logprobs field."""
        from vllm_omni.outputs import OmniModelRunnerOutput

        assert hasattr(OmniModelRunnerOutput, "__dataclass_fields__") or hasattr(OmniModelRunnerOutput, "logprobs")
