"""Tests for TrackingArgumentParser and related utilities."""

import argparse
import json
from unittest.mock import patch

import pytest
import yaml
from vllm.utils.argparse_utils import FlexibleArgumentParser

from vllm_omni.config.stage_config import (
    _PIPELINE_REGISTRY,
    DeployConfig,
    PipelineConfig,
    StageConfigFactory,
    StageDeployConfig,
    StagePipelineConfig,
)
from vllm_omni.utils.tracking_parser import TrackingArgumentParser, TrackingNamespace

### Fake pipeline/deploy config for integration tests

_TEST_MODEL = "_test_tracking"

_TEST_PIPELINE = PipelineConfig(
    model_type=_TEST_MODEL,
    stages=(
        StagePipelineConfig(stage_id=0, model_stage="stage0"),
        StagePipelineConfig(stage_id=1, model_stage="stage1"),
        StagePipelineConfig(stage_id=2, model_stage="stage2"),
    ),
)

_TEST_DEPLOY = DeployConfig(
    async_chunk=False,
    stages=[
        StageDeployConfig(stage_id=0, gpu_memory_utilization=0.5, max_num_seqs=32),
        StageDeployConfig(stage_id=1, gpu_memory_utilization=0.6, max_num_seqs=64),
        StageDeployConfig(stage_id=2, gpu_memory_utilization=0.1, max_num_seqs=16),
    ],
)


@pytest.fixture()
def mock_stages(monkeypatch):
    """Register a fake pipeline and mock deploy YAML loading."""
    monkeypatch.setitem(_PIPELINE_REGISTRY._loaded, _TEST_MODEL, _TEST_PIPELINE)
    monkeypatch.setattr(
        "vllm_omni.config.stage_config.load_deploy_config",
        lambda _path: _TEST_DEPLOY,
    )
    return __file__


### Tests for TrackingNamespace
def test_tracking_namespaces_cant_be_nested():
    """Ensure tracking namespaces explode if we try to nest them."""
    track_ns = TrackingNamespace(
        unfiltered_ns=argparse.Namespace(foo="bar"),
        explicit_keys=frozenset(),
    )

    with pytest.raises(ValueError):
        TrackingNamespace(
            unfiltered_ns=track_ns,
            explicit_keys=frozenset(),
        )


def test_tracking_namespaces_init():
    """Check simple initialization for tracking namespaces."""
    unfiltered_ns = argparse.Namespace(foo="bar")
    tracked_ns = TrackingNamespace(
        unfiltered_ns=unfiltered_ns,
        explicit_keys=frozenset({"foo"}),
    )
    assert tracked_ns.foo == "bar"
    assert tracked_ns.explicit_keys == frozenset({"foo"})


def test_tracking_filtering():
    """Ensure tracking namespaces are filterable."""
    unfiltered_ns = argparse.Namespace(foo="bar", baz="foobar")
    tracked_ns = TrackingNamespace(
        unfiltered_ns=unfiltered_ns,
        explicit_keys=frozenset({"foo"}),
    )
    assert tracked_ns.foo == "bar"
    assert tracked_ns.baz == "foobar"
    assert tracked_ns.explicit_keys == frozenset({"foo"})
    # baz gets dropped because it's not marked in explicit_keys
    assert tracked_ns.get_explicit_kwargs_dict() == {"foo": "bar"}


def test_setattr_writes_through_to_unfiltered_ns():
    """Ensure mutating an attribute on TrackingNamespace forwards to
    get_explicit_kwargs_dict() and vars()."""
    unfiltered_ns = argparse.Namespace(model="original")
    tracked_ns = TrackingNamespace(
        unfiltered_ns=unfiltered_ns,
        explicit_keys=frozenset({"model"}),
    )
    assert tracked_ns.model == "original"
    assert tracked_ns.get_explicit_kwargs_dict() == {"model": "original"}
    assert vars(tracked_ns) == {"model": "original"}

    # Ensure if we update the namespace, it's forwarded correctly
    tracked_ns.model = "updated"
    assert tracked_ns.model == "updated"
    assert tracked_ns.get_explicit_kwargs_dict() == {"model": "updated"}
    assert vars(tracked_ns) == {"model": "updated"}


### Tests for simple cases (no nested parsers or groups)
def test_vars_does_not_expose_internals():
    """Ensure vars on a TrackingArgumentParser is identical to running on the real one."""
    tracking = TrackingArgumentParser()
    flexible = FlexibleArgumentParser()
    namespaces = []

    # For both parser, register two args, but only pass one
    for parser in [tracking, flexible]:
        parser.add_argument("--foo", type=int, default=42)
        parser.add_argument("--bar", type=str, default="x")
        namespaces.append(parser.parse_args(["--foo", "100"]))

    # Ensure that vars are the same result, since it isn't filtered
    tracked_ns, flexible_ns = namespaces
    tracked_kwargs = vars(tracked_ns)
    real_kwargs = vars(flexible_ns)
    assert tracked_kwargs == real_kwargs == {"foo": 100, "bar": "x"}
    assert tracked_ns.explicit_keys == {"foo"}


def test_default_not_detected():
    """Ensure omitted defaults aren't in explicit keys and take defaults."""
    p = TrackingArgumentParser()
    p.add_argument("--foo", type=int, default=42)
    ns = p.parse_args([])
    assert ns.explicit_keys == set()
    assert isinstance(ns, TrackingNamespace)
    assert ns.foo == 42


@pytest.mark.parametrize("val", ["42", "100"])
def test_explicit_value_equal_to_default(val):
    """Ensure explicit keys correctly handles passed values."""
    p = TrackingArgumentParser()
    p.add_argument("--foo", type=int, default=42)
    ns = p.parse_args(["--foo", val])
    assert ns.explicit_keys == {"foo"}
    assert isinstance(ns, TrackingNamespace)
    assert ns.foo == int(val)


def test_equals_syntax():
    """Ensure equals syntax is handled correctly wrt explicit keys."""
    p = TrackingArgumentParser()
    p.add_argument("--foo", type=int, default=42)
    ns = p.parse_args(["--foo=100"])
    assert ns.explicit_keys == {"foo"}
    assert isinstance(ns, TrackingNamespace)
    assert ns.foo == 100


def test_explicit_none_via_store_const():
    """Ensure passing `None` to the namespace, it isn't filtered from explicit_keys."""
    parser = TrackingArgumentParser()
    parser.add_argument("--foo", action="store_const", const=None, default="Something else")
    ns = parser.parse_args(["--foo"])
    # User explicitly passed --foo, so it should be in the explicit keys even though it's None
    assert ns.explicit_keys == {"foo"}
    assert isinstance(ns, TrackingNamespace)
    assert ns.foo is None


def test_multiple_args_mixed():
    """Ensure that explicit keys are correct when some are passed and others aren't."""
    p = TrackingArgumentParser()
    p.add_argument("--foo", type=int, default=1)
    p.add_argument("--bar", type=str, default="x")
    p.add_argument("--baz", type=float, default=0.5)
    ns = p.parse_args(["--foo", "10", "--baz", "0.9"])
    assert ns.explicit_keys == {"foo", "baz"}
    assert isinstance(ns, TrackingNamespace)
    assert ns.bar == "x"


def test_store_true_default():
    """Ensure that store true is handled correctly when omitted."""
    p = TrackingArgumentParser()
    p.add_argument("--verbose", action="store_true")
    ns = p.parse_args([])
    assert ns.explicit_keys == set()
    assert isinstance(ns, TrackingNamespace)
    assert ns.verbose is False


def test_store_true_explicit():
    """Ensure that store true is handled correctly when passed."""
    p = TrackingArgumentParser()
    p.add_argument("--verbose", action="store_true")
    ns = p.parse_args(["--verbose"])
    assert ns.explicit_keys == {"verbose"}
    assert isinstance(ns, TrackingNamespace)
    assert ns.verbose is True


def test_store_false_default():
    """Ensure that store false is handled correctly when omitted."""
    p = TrackingArgumentParser()
    p.add_argument("--disable-x", action="store_false", dest="enable_x", default=True)
    ns = p.parse_args([])
    assert ns.explicit_keys == set()
    assert isinstance(ns, TrackingNamespace)
    assert ns.enable_x is True


def test_store_false_explicit():
    """Ensure that store false is handled correctly when passed."""
    p = TrackingArgumentParser()
    p.add_argument("--disable-x", action="store_false")
    ns = p.parse_args(["--disable-x"])
    assert ns.explicit_keys == {"disable_x"}
    assert isinstance(ns, TrackingNamespace)
    assert ns.disable_x is False


def test_dest_is_reflected_in_explicit_keys():
    """Ensure that explicit keys use dest correctly."""
    p = TrackingArgumentParser()
    p.add_argument("--foo", type=int, dest="bar")
    ns = p.parse_args(["--foo", "100"])
    assert ns.explicit_keys == {"bar"}
    assert isinstance(ns, TrackingNamespace)
    assert ns.bar == 100


def test_boolean_optional_action_default():
    """Check that boolean optional actions are handled correctly when omitted."""
    p = TrackingArgumentParser()
    p.add_argument("--flag", action=argparse.BooleanOptionalAction)
    ns = p.parse_args([])
    assert ns.explicit_keys == set()
    assert isinstance(ns, TrackingNamespace)
    assert ns.flag is None


def test_boolean_optional_action_positive():
    """Check that boolean optional actions are handled correctly."""
    p = TrackingArgumentParser()
    p.add_argument("--flag", action=argparse.BooleanOptionalAction, default=None)
    ns = p.parse_args(["--flag"])
    assert ns.explicit_keys == {"flag"}
    assert isinstance(ns, TrackingNamespace)
    assert ns.flag is True


def test_no_option_strings_are_handled():
    """Ensure --no-<feature> sets <feature> in the explicit keys correctly."""
    p = TrackingArgumentParser()
    p.add_argument("--flag", action=argparse.BooleanOptionalAction, default=None)
    ns = p.parse_args(["--no-flag"])
    assert ns.explicit_keys == {"flag"}
    assert isinstance(ns, TrackingNamespace)
    assert ns.flag is False


def test_json_type():
    """Ensure json type is handled correctly."""
    p = TrackingArgumentParser()
    p.add_argument("--cfg", type=json.loads, default="{}")
    ns = p.parse_args(["--cfg", '{"a": 1}'])
    assert "cfg" in ns.explicit_keys
    assert isinstance(ns, TrackingNamespace)
    assert ns.cfg == {"a": 1}


def test_choices():
    """Ensure choices are handled correctly."""
    p = TrackingArgumentParser()
    p.add_argument("--mode", choices=["fast", "slow"], default="fast")
    ns = p.parse_args(["--mode", "slow"])
    assert "mode" in ns.explicit_keys
    assert ns is not None
    assert ns.mode == "slow"


def test_explicit_positional_arg():
    """Ensure positional args are handled correctly when provided."""
    p = TrackingArgumentParser()
    p.add_argument("name", nargs="?", default=None)
    ns = p.parse_args(["hello"])
    assert "name" in ns.explicit_keys
    assert isinstance(ns, TrackingNamespace)
    assert ns.name == "hello"


def test_omitted_positional_arg():
    """Ensure positional args are handled correctly when omitted."""
    p = TrackingArgumentParser()
    p.add_argument("name", nargs="?", default=None)
    ns = p.parse_args([])
    assert "name" not in ns.explicit_keys
    assert isinstance(ns, TrackingNamespace)
    assert ns.name is None


def test_explicit_nargs():
    """Ensure that variable num args are handled correctly when provided."""
    p = TrackingArgumentParser()
    p.add_argument("--items", nargs="*", default=None)
    ns = p.parse_args(["--items", "a", "b"])
    assert "items" in ns.explicit_keys
    assert isinstance(ns, TrackingNamespace)
    assert ns.items == ["a", "b"]


def test_omitted_nargs():
    """Ensure that variable num args are handled correctly when omitted."""
    p = TrackingArgumentParser()
    p.add_argument("--items", nargs="*", default=None)
    ns = p.parse_args([])
    assert "items" not in ns.explicit_keys
    assert isinstance(ns, TrackingNamespace)
    assert ns.items is None


def test_parse_known_args_tracking():
    """Ensure parse_known_args is also trackable"""
    p = TrackingArgumentParser()
    p.add_argument("--foo", type=int, default=42)
    ns, remaining = p.parse_known_args(["--foo", "10", "--unknown", "val"])
    assert ns.explicit_keys == {"foo"}
    assert isinstance(ns, TrackingNamespace)
    assert ns.foo == 10
    assert remaining == ["--unknown", "val"]


### Tests for group handling
def test_group_arg_default():
    """Ensure that that groups with defaults are handled correctly."""
    p = TrackingArgumentParser()
    g = p.add_argument_group("TestGroup")
    g.add_argument("--bar", type=str, default="baz")
    ns = p.parse_args([])
    assert ns.explicit_keys == set()
    assert isinstance(ns, TrackingNamespace)
    assert ns.bar == "baz"


def test_group_arg_explicit():
    """Ensure that that groups are handled correctly."""
    p = TrackingArgumentParser()
    g = p.add_argument_group("TestGroup")
    g.add_argument("--bar", type=str, default="baz")
    ns = p.parse_args(["--bar", "qux"])
    assert ns.explicit_keys == {"bar"}
    assert isinstance(ns, TrackingNamespace)
    assert ns.bar == "qux"


def test_multiple_groups():
    """Test multiple groups behave correctly."""
    p = TrackingArgumentParser()
    g1 = p.add_argument_group("Group1")
    g2 = p.add_argument_group("Group2")
    g1.add_argument("--a", type=int, default=1)
    g2.add_argument("--b", type=int, default=2)
    ns = p.parse_args(["--b", "20"])
    assert ns.explicit_keys == {"b"}
    assert isinstance(ns, TrackingNamespace)
    assert ns.a == 1
    assert ns.b == 20


def test_omitted_mutually_exclusive_group():
    """Ensure that that mutually exclusive groups with defaults are handled correctly."""
    p = TrackingArgumentParser()
    grp = p.add_mutually_exclusive_group()
    grp.add_argument("--json", action="store_true")
    grp.add_argument("--text", action="store_true")
    ns = p.parse_args([])
    assert ns.explicit_keys == set()
    assert isinstance(ns, TrackingNamespace)


def test_mutually_exclusive_group():
    """Ensure that that mutually exclusive groups are handled correctly."""
    p = TrackingArgumentParser()
    grp = p.add_mutually_exclusive_group()
    grp.add_argument("--json", action="store_true")
    grp.add_argument("--text", action="store_true")
    ns = p.parse_args(["--json"])
    assert ns.explicit_keys == {"json"}
    assert isinstance(ns, TrackingNamespace)
    assert ns.json is True
    assert ns.text is False


### Tests for subparser handling
def test_subparser_explicit_detection():
    """Ensure that subparsers are handled correctly."""
    p = TrackingArgumentParser()
    sub = p.add_subparsers()
    child = sub.add_parser("foo")
    child.add_argument("--bar", type=str)
    ns = p.parse_args(["foo", "--bar", "baz"])
    assert isinstance(child, TrackingArgumentParser)
    assert isinstance(ns, TrackingNamespace)
    assert ns.explicit_keys == {"bar"}
    assert ns.bar == "baz"


def test_subparser_group_args():
    """Ensure that subparsers with groups are handled correctly."""
    p = TrackingArgumentParser()
    sub = p.add_subparsers()
    child = sub.add_parser("foo")
    g = child.add_argument_group("Config")
    g.add_argument("--port", type=int, default=8000)
    g.add_argument("--host", type=str, default="localhost")
    ns = p.parse_args(["foo", "--port", "9000"])
    assert ns.explicit_keys == {"port"}
    assert isinstance(ns, TrackingNamespace)
    assert ns.host == "localhost"


### Tests for specific behaviors against FlexibleArgumentParser
def test_config_file_args_detected(tmp_path):
    """Ensure config is handled correctly for vLLM's FlexibleArgumentParser."""
    p = TrackingArgumentParser()
    p.add_argument("--foo", type=int)
    p.add_argument("--bar", type=int)
    cfg = tmp_path / "test.yaml"
    cfg.write_text(yaml.dump({"foo": 100}))
    ns = p.parse_args(["--config", str(cfg)])
    assert isinstance(ns, TrackingNamespace)
    assert ns.explicit_keys == {"foo"}
    assert ns.foo == 100


def test_cli_overrides_config(tmp_path):
    """Ensure tracking parser handles config vs cli overrides correctly."""
    p = TrackingArgumentParser()
    p.add_argument("--foo", type=int)
    cfg = tmp_path / "test.yaml"
    cfg.write_text(yaml.dump({"foo": 100}))
    ns = p.parse_args(["--config", str(cfg), "--foo", "200"])
    assert isinstance(ns, TrackingNamespace)
    assert "foo" in ns.explicit_keys
    assert ns.foo == 200


### Integration tests for arg resolution through StageConfigFactory
def test_explicit_cli_arg_reaches_runtime_overrides(mock_stages):
    """Explicitly passed CLI values reach runtime_overrides on all stages."""
    p = TrackingArgumentParser()
    p.add_argument("--max-num-seqs", type=int, default=64)
    ns = p.parse_args(["--max-num-seqs", "999"])

    explicit_kwargs = ns.get_explicit_kwargs_dict()
    stages = StageConfigFactory._create_from_registry(
        _TEST_MODEL,
        explicit_kwargs,
        deploy_config_path=mock_stages,
    )
    for stage in stages:
        assert stage.runtime_overrides.get("max_num_seqs") == 999


def test_omitted_default_not_in_runtime_overrides(mock_stages):
    """Omitted defaults are overridden by deploy config values"""
    p = TrackingArgumentParser()
    p.add_argument("--max-num-seqs", type=int, default=64)
    ns = p.parse_args([])

    explicit_kwargs = ns.get_explicit_kwargs_dict()
    stages = StageConfigFactory._create_from_registry(
        _TEST_MODEL,
        explicit_kwargs,
        deploy_config_path=mock_stages,
    )
    for stage in stages:
        assert stage.runtime_overrides == {}


def test_config_file_args_reach_runtime_overrides(mock_stages):
    """Args from --config YAML must be treated as explicitly passed and
    flow through to runtime_overrides."""
    p = TrackingArgumentParser()
    p.add_argument("--max-num-seqs", type=int, default=64)
    p.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    with patch(
        "vllm.utils.argparse_utils.FlexibleArgumentParser.load_config_file", return_value=["--max-num-seqs", "999"]
    ):
        ns = p.parse_args(["--config", "fake.yaml"])

    explicit_kwargs = ns.get_explicit_kwargs_dict()
    stages = StageConfigFactory._create_from_registry(
        _TEST_MODEL,
        explicit_kwargs,
        deploy_config_path=mock_stages,
    )
    for stage in stages:
        assert stage.runtime_overrides.get("max_num_seqs") == 999
        assert "gpu_memory_utilization" not in stage.runtime_overrides


def test_per_stage_override_routes_correctly(mock_stages):
    """Ensure stage_<N>_<key> only affects the targeted stage."""
    p = TrackingArgumentParser()
    p.add_argument("--stage-0-gpu-memory-utilization", type=float)
    ns = p.parse_args(["--stage-0-gpu-memory-utilization", "0.42"])

    explicit_kwargs = ns.get_explicit_kwargs_dict()
    stages = StageConfigFactory._create_from_registry(
        _TEST_MODEL,
        explicit_kwargs,
        deploy_config_path=mock_stages,
    )
    assert stages[0].runtime_overrides == {"gpu_memory_utilization": 0.42}
    assert stages[1].runtime_overrides == {}
    assert stages[2].runtime_overrides == {}


def test_explicit_args_omitted_from_yaml(mock_stages):
    """Ensure only passed args end up in runtime overrides (regardless of whether
    they are defined in the yaml config)."""
    p = TrackingArgumentParser()
    p.add_argument("--enforce-eager", action="store_true")
    p.add_argument("--max-num-seqs", type=int, default=64)
    p.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    # NOTE: enforce eager is not set in the mock config yaml.
    ns = p.parse_args(["--enforce-eager"])

    explicit_kwargs = ns.get_explicit_kwargs_dict()
    stages = StageConfigFactory._create_from_registry(
        _TEST_MODEL,
        explicit_kwargs,
        deploy_config_path=mock_stages,
    )

    for stage in stages:
        assert stage.runtime_overrides == {"enforce_eager": True}
