# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from pathlib import Path

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


def test_documented_context_overlay_loads_mrv2_deployment(tmp_path):
    repo = Path(__file__).resolve().parents[3]
    guide = (repo / "docs/serving/realtime_duplex_api.md").read_text()
    section = guide.split("### Qwen long-session context policies (experimental)", 1)[1]
    example = section.split("```yaml\n", 1)[1].split("```", 1)[0]
    # Copy the deployment profiles beside the extracted example, matching the
    # documented relative-base layout without writing into the source tree.
    for profile in (repo / "vllm_omni/deploy").glob("*.yaml"):
        (tmp_path / profile.name).write_text(profile.read_text())
    path = tmp_path / "context-memory.yaml"
    path.write_text(example)
    deploy = load_deploy_config(path)
    assert deploy.session_mode == "duplex"
    assert deploy.model_runner == "v2"
    assert deploy.duplex_session.qwen_context_policy == "budget_summary"
    assert deploy.duplex_session.qwen_context_max_tokens == 4096
    assert deploy.duplex_session.qwen_summary_max_tokens == 512
    assert len(deploy.stages) == 3
