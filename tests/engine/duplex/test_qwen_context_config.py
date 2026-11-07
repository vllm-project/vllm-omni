# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import pytest

from vllm_omni.config.stage_config import DuplexSessionRuntimeConfig, load_deploy_config

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def test_context_defaults_keep_legacy_behavior():
    config = DuplexSessionRuntimeConfig()
    assert config.qwen_context_policy == "legacy"
    assert config.qwen_context_max_tokens is None
    assert config.qwen_summary_max_tokens == 512
    assert config.qwen_summary_input_max_tokens == 8192
    assert config.qwen_summary_timeout_s == 30


@pytest.mark.parametrize("policy", ["legacy", "budget", "budget_summary"])
def test_context_policy_loads_from_deployment(tmp_path, policy):
    path = tmp_path / "deploy.yaml"
    path.write_text(
        "duplex_session:\n"
        f"  qwen_context_policy: {policy}\n"
        "  qwen_context_max_tokens: 4096\n"
        "  qwen_summary_max_tokens: 128\n"
        "  qwen_summary_input_max_tokens: 2048\n"
        "  qwen_summary_timeout_s: 5.5\n"
        "stages: []\n"
    )
    config = load_deploy_config(path).duplex_session
    assert config.qwen_context_policy == policy
    assert config.qwen_context_max_tokens == 4096
    assert config.qwen_summary_max_tokens == 128
    assert config.qwen_summary_input_max_tokens == 2048
    assert config.qwen_summary_timeout_s == 5.5


@pytest.mark.parametrize("value", ["auto", "", None, True, 1])
def test_context_policy_rejects_unknown_values(value):
    with pytest.raises(ValueError, match="qwen_context_policy"):
        DuplexSessionRuntimeConfig(qwen_context_policy=value)


@pytest.mark.parametrize(
    "name", ["qwen_context_max_tokens", "qwen_summary_max_tokens", "qwen_summary_input_max_tokens"]
)
@pytest.mark.parametrize("value", [0, -1, True, False, "1024", 1.5, float("inf"), float("nan")])
def test_token_limits_require_positive_integers(name, value):
    with pytest.raises(ValueError, match=name):
        DuplexSessionRuntimeConfig(**{name: value})


@pytest.mark.parametrize("value", [0, -1, True, "30", None, float("inf"), float("nan")])
def test_summary_timeout_requires_finite_positive_number(value):
    with pytest.raises(ValueError, match="qwen_summary_timeout_s"):
        DuplexSessionRuntimeConfig(qwen_summary_timeout_s=value)
