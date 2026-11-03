# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import argparse
import json

import pytest

from vllm_omni.utils.profiler import add_profiler_config_arg, parse_json_object, parse_profiler_config
from vllm_omni.utils.tracking_parser import TrackingArgumentParser

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


@pytest.fixture(params=[argparse.ArgumentParser, TrackingArgumentParser])
def parser(request):
    parser = request.param()
    add_profiler_config_arg(parser)
    return parser


def test_profiler_config_defaults_to_none(parser, monkeypatch):
    monkeypatch.setenv("VLLM_TORCH_PROFILER_DIR", "/tmp/legacy-profile")
    assert parser.parse_args([]).profiler_config is None


@pytest.mark.parametrize(
    "config",
    [
        {"profiler": "torch", "torch_profiler_dir": "./perf", "torch_profiler_record_shapes": True},
        {"profiler": "cuda"},
        {},
    ],
)
def test_profiler_config_accepts_json_objects(parser, config):
    args = parser.parse_args(["--profiler-config", json.dumps(config)])
    assert args.profiler_config == config


@pytest.mark.parametrize("value", ["{", '{"profiler":}', "", "[]", '"torch"', "42", "true", "null"])
def test_profiler_config_rejects_invalid_cli_values(parser, value, capsys):
    with pytest.raises(SystemExit) as exc_info:
        parser.parse_args(["--profiler-config", value])
    assert exc_info.value.code == 2
    error = capsys.readouterr().err
    assert "--profiler-config" in error
    assert "must be valid JSON" in error or "must be a JSON object" in error


def test_parse_profiler_config_reports_json_error():
    with pytest.raises(argparse.ArgumentTypeError, match="--profiler-config must be valid JSON"):
        parse_profiler_config("{")


def test_parse_profiler_config_requires_object():
    with pytest.raises(argparse.ArgumentTypeError, match="--profiler-config must be a JSON object"):
        parse_profiler_config("[]")


def test_parse_json_object_preserves_values():
    assert parse_json_object('{"nested": {"values": [1, true, null]}}') == {"nested": {"values": [1, True, None]}}


@pytest.mark.parametrize("value", ["{", "[]"])
def test_parse_json_object_names_the_invalid_flag(value):
    with pytest.raises(argparse.ArgumentTypeError, match="--config must be"):
        parse_json_object(value, flag_name="--config")
