# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Live session deploy/pipeline config and how it reaches the engine args."""

import dataclasses
from types import SimpleNamespace

import pytest

from tests.helpers.stage_config import get_deploy_config_path
from vllm_omni.config.live_session import LiveSessionConfig, LiveSessionDeployConfig
from vllm_omni.config.stage_config import (
    add_live_session_logits_processor,
    live_session_capacity,
    load_deploy_config,
    streaming_session_capacity,
)
from vllm_omni.entrypoints.openai.live.processor import LiveSessionProcessor, load_processor_class
from vllm_omni.model_executor.models.minicpmo_4_5.pipeline import MINICPMO_4_5_PIPELINE
from vllm_omni.model_executor.models.personaplex.pipeline import PERSONAPLEX_PIPELINE

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

_LOGITS = "pkg.mod:Live"


def test_deploy_config_defaults():
    config = LiveSessionDeployConfig.from_dict(None)

    assert config.session_lifetime_s == 1800
    assert config.max_sessions is None


def test_deploy_config_from_dict():
    config = LiveSessionDeployConfig.from_dict({"session_lifetime_s": 60, "max_sessions": 2})

    assert (config.session_lifetime_s, config.max_sessions) == (60, 2)


@pytest.mark.parametrize(
    ("raw", "match"),
    [
        pytest.param({"session_cap": 2}, "Unknown live_session_config keys", id="unknown-key"),
        pytest.param({"session_lifetime_s": 0}, "session_lifetime_s must be positive", id="lifetime-zero"),
        pytest.param({"max_sessions": 0}, "max_sessions must be positive or null", id="max-sessions-zero"),
        pytest.param({"max_sessions": -1}, "max_sessions must be positive or null", id="max-sessions-negative"),
    ],
)
def test_deploy_config_rejects(raw, match):
    with pytest.raises(ValueError, match=match):
        LiveSessionDeployConfig.from_dict(raw)


@pytest.mark.parametrize(
    ("config", "errors"),
    [
        pytest.param(LiveSessionConfig("pkg.mod:Proc", {0: _LOGITS}), [], id="valid"),
        pytest.param(LiveSessionConfig(""), ["live_session_processor is required"], id="no-processor"),
        pytest.param(LiveSessionConfig("pkg.mod:Proc", {-1: _LOGITS}), ["key -1 must be a stage id"], id="negative"),
        pytest.param(LiveSessionConfig("pkg.mod:Proc", {"0": _LOGITS}), ["key '0' must be a stage id"], id="str-key"),
        pytest.param(LiveSessionConfig("pkg.mod:Proc", {0: "pkg.mod.Live"}), ["must be 'module:Class'"], id="dotted"),
    ],
)
def test_live_session_config_validation(config, errors):
    found = config.get_validation_errors()

    assert len(found) == len(errors)
    for error, fragment in zip(found, errors):
        assert fragment in error


def test_pipeline_rejects_logits_processor_on_missing_stage():
    live = LiveSessionConfig("pkg.mod:Proc", {9: _LOGITS})

    with pytest.raises(ValueError, match="logits_processor references non-existent stage 9"):
        dataclasses.replace(MINICPMO_4_5_PIPELINE, live_session_config=live)


@pytest.mark.parametrize("pipeline", [MINICPMO_4_5_PIPELINE, PERSONAPLEX_PIPELINE], ids=lambda p: p.model_type)
def test_bundled_pipelines_declare_a_loadable_live_processor(pipeline):
    assert pipeline.get_validation_errors() == []
    assert issubclass(load_processor_class(pipeline.live_session_config.live_session_processor), LiveSessionProcessor)


@pytest.mark.parametrize(
    ("deploy", "max_sessions"),
    [("minicpmo_4_5_live.yaml", None), ("personaplex_live.yaml", 2)],
)
def test_bundled_live_deploy_configs(deploy, max_sessions):
    config = load_deploy_config(get_deploy_config_path(deploy))

    assert config.session_mode == "turn"
    assert config.live_session_config == LiveSessionDeployConfig(session_lifetime_s=1800, max_sessions=max_sessions)


def _deploy(live=None, session_mode="turn", worker_max_sessions=4):
    return SimpleNamespace(
        live_session_config=live,
        session_mode=session_mode,
        duplex_session=SimpleNamespace(worker_max_sessions=worker_max_sessions),
    )


@pytest.mark.parametrize(
    ("deploy", "live", "streaming"),
    [
        pytest.param(_deploy(), None, 1, id="chat"),
        pytest.param(_deploy(LiveSessionDeployConfig()), None, 1, id="live-unbounded"),
        pytest.param(_deploy(LiveSessionDeployConfig(max_sessions=3)), 3, 3, id="live-bounded"),
        pytest.param(_deploy(session_mode="duplex", worker_max_sessions=5), None, 5, id="duplex"),
    ],
)
def test_session_capacity(deploy, live, streaming):
    assert live_session_capacity(deploy) == live
    assert streaming_session_capacity(deploy) == streaming


def _pipeline(logits_processor):
    return SimpleNamespace(live_session_config=LiveSessionConfig("pkg.mod:Proc", logits_processor))


def test_logits_processor_added_on_live_deployments_only():
    pipeline = _pipeline({0: _LOGITS})
    chat_args: dict = {}
    live_args: dict = {}

    add_live_session_logits_processor(chat_args, 0, pipeline, _deploy())
    add_live_session_logits_processor(live_args, 0, pipeline, _deploy(LiveSessionDeployConfig()))

    assert chat_args == {}
    assert live_args == {"logits_processors": [_LOGITS]}


def test_logits_processor_only_on_its_stage():
    args: dict = {}

    add_live_session_logits_processor(args, 1, _pipeline({0: _LOGITS}), _deploy(LiveSessionDeployConfig()))

    assert args == {}


def test_logits_processor_appended_once_to_existing_ones():
    args = {"logits_processors": ["other:Proc"]}
    deploy = _deploy(LiveSessionDeployConfig())

    add_live_session_logits_processor(args, 0, _pipeline({0: _LOGITS}), deploy)
    add_live_session_logits_processor(args, 0, _pipeline({0: _LOGITS}), deploy)

    assert args == {"logits_processors": ["other:Proc", _LOGITS]}


def test_logits_processor_skipped_without_pipeline_live_config():
    args: dict = {}

    add_live_session_logits_processor(
        args, 0, SimpleNamespace(live_session_config=None), _deploy(LiveSessionDeployConfig())
    )

    assert args == {}
