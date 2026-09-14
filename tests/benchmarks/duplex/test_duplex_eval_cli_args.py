# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Offline unit tests for the Omni-DuplexEval ``vllm bench serve`` CLI wiring.

M3 / T22-T33: dataset choice extension, the ``--duplex-eval-*`` argument group
(exactly 15 options) and every ``preprocess_serve_args`` validation rule, each
with a positive and a negative case.

All tests are fully offline. A missing/stale vLLM that makes the CLI package
un-importable is reported as a *skip*, never an error: ``cli_args`` is a leaf
module (stdlib only) and is loaded by file path in that case.

See ``plans/omni-duplex-eval-backend-design.md`` §5.4 and §9.
"""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path

import pytest

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

_REPO_ROOT = Path(__file__).resolve().parents[3]
_DUPLEX_EVAL_PREFIX = "--duplex-eval-"
_EXPECTED_OPTIONS = frozenset(
    {
        "--duplex-eval-split",
        "--duplex-eval-family",
        "--duplex-eval-media-root",
        "--duplex-eval-limit",
        "--duplex-eval-ids",
        "--duplex-eval-exclude-ids",
        "--duplex-eval-output-dir",
        "--duplex-eval-score-dir",
        "--duplex-eval-ref-audio",
        "--duplex-eval-fps",
        "--duplex-eval-pace",
        "--duplex-eval-clock",
        "--duplex-eval-overwrite",
        "--duplex-eval-allow-invalid-clock",
        "--duplex-eval-no-artifacts",
    }
)
_EXPECTED_DEFAULTS = {
    "duplex_eval_split": "all",
    "duplex_eval_family": "all",
    "duplex_eval_media_root": None,
    "duplex_eval_limit": None,
    "duplex_eval_ids": None,
    "duplex_eval_exclude_ids": None,
    "duplex_eval_output_dir": Path("duplex-eval-responses"),
    "duplex_eval_score_dir": Path("duplex-eval-scores"),
    "duplex_eval_ref_audio": None,
    "duplex_eval_fps": 1.0,
    "duplex_eval_pace": "realtime",
    "duplex_eval_clock": "media",
    "duplex_eval_overwrite": False,
    "duplex_eval_allow_invalid_clock": False,
    "duplex_eval_no_artifacts": False,
}


def _load_cli_args():
    """Return the ``cli_args`` module, falling back to a file-path load.

    ``vllm_omni.entrypoints.cli.__init__`` imports ``vllm.entrypoints.launchers``
    on some vLLM builds. Loading the leaf module directly keeps these tests
    runnable even when that optional package import fails.
    """
    try:
        from vllm_omni.entrypoints.cli.benchmark import cli_args

        return cli_args
    except ImportError:
        path = _REPO_ROOT / "vllm_omni" / "entrypoints" / "cli" / "benchmark" / "cli_args.py"
        if not path.exists():  # pragma: no cover - repository layout guard
            pytest.skip("cli_args.py is not available", allow_module_level=True)
        spec = importlib.util.spec_from_file_location("_duplex_eval_cli_args_under_test", path)
        if spec is None or spec.loader is None:  # pragma: no cover - defensive
            pytest.skip("cli_args.py cannot be loaded", allow_module_level=True)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module


cli_args = _load_cli_args()


def _require_dataset_module() -> None:
    """Skip when the dataset adapter (and therefore vLLM) is unavailable."""
    pytest.importorskip("vllm_omni.benchmarks.data_modules.duplex_eval_dataset")


def _serve_args(**overrides) -> argparse.Namespace:
    """A fully valid ``omni-duplex-eval`` namespace that passes every rule."""
    base: dict = {
        "dataset_name": "omni-duplex-eval",
        "backend": "openai-realtime-duplex",
        "endpoint": "/v1/realtime",
        "duplex_eval_ref_audio": "tests/assets/ref.wav",
        "ignore_eos": False,
        "profile": False,
        "skip_tokenizer_init": False,
        "probe_request_rate": 0.0,
        "duplex_eval_pace": "realtime",
        "duplex_eval_allow_invalid_clock": False,
        "duplex_eval_ids": None,
        "duplex_eval_exclude_ids": None,
        "explicit_keys": (),
        "num_prompts": 5,
        "max_concurrency": 2,
    }
    base.update(overrides)
    return argparse.Namespace(**base)


def _duplex_actions(parser: argparse.ArgumentParser) -> dict[str, argparse.Action]:
    return {
        action.dest: action
        for action in parser._actions
        if action.option_strings and any(opt.startswith(_DUPLEX_EVAL_PREFIX) for opt in action.option_strings)
    }


# ================================================================== #
# T22: dataset choices contain ``omni-duplex-eval``
# ================================================================== #


class TestDatasetChoices:
    """T22: the new dataset token is registered on the serve parser."""

    def test_constant_contains_token(self) -> None:
        assert "omni-duplex-eval" in cli_args._OMNI_BENCH_DATASET_CHOICES

    def test_extend_omni_choices_adds_token(self) -> None:
        parser = argparse.ArgumentParser()
        parser.add_argument("--dataset-name", dest="dataset_name", choices=["sharegpt", "hf"])
        cli_args.extend_omni_choices(parser)
        action = next(a for a in parser._actions if a.dest == "dataset_name")
        assert "omni-duplex-eval" in action.choices
        for choice in cli_args._OMNI_BENCH_DATASET_CHOICES:
            assert choice in action.choices

    def test_extend_omni_choices_handles_shadow_parser(self) -> None:
        parser = argparse.ArgumentParser()
        parser.add_argument("--dataset-name", dest="dataset_name", choices=["hf"])
        shadow = argparse.ArgumentParser()
        shadow.add_argument("--dataset-name", dest="dataset_name", choices=["hf"])
        setattr(parser, "_shadow", shadow)
        cli_args.extend_omni_choices(parser)
        for target in (parser, shadow):
            action = next(a for a in target._actions if a.dest == "dataset_name")
            assert "omni-duplex-eval" in action.choices


# ================================================================== #
# T23: ``add_duplex_eval_cli_args`` registers exactly 15 options
# ================================================================== #


class TestDuplexEvalArgRegistration:
    """T23: option set, defaults, types/choices and ``add_omni_args`` hookup."""

    def test_exactly_fifteen_options(self) -> None:
        parser = argparse.ArgumentParser()
        cli_args.add_duplex_eval_cli_args(parser)
        option_strings = {
            opt for action in parser._actions for opt in action.option_strings if opt.startswith(_DUPLEX_EVAL_PREFIX)
        }
        assert option_strings == set(_EXPECTED_OPTIONS)
        assert len(option_strings) == 15

    def test_no_judge_options_registered(self) -> None:
        parser = argparse.ArgumentParser()
        cli_args.add_duplex_eval_cli_args(parser)
        for action in parser._actions:
            assert "judge" not in action.dest
            for opt in action.option_strings:
                assert "judge" not in opt

    def test_defaults(self) -> None:
        _require_dataset_module()
        parser = argparse.ArgumentParser()
        cli_args.add_duplex_eval_cli_args(parser)
        args = parser.parse_args([])
        for dest, expected in _EXPECTED_DEFAULTS.items():
            assert getattr(args, dest) == expected, dest

    def test_types_and_choices(self) -> None:
        _require_dataset_module()
        parser = argparse.ArgumentParser()
        cli_args.add_duplex_eval_cli_args(parser)
        actions = _duplex_actions(parser)
        assert actions["duplex_eval_limit"].type is cli_args._non_negative_int
        assert actions["duplex_eval_fps"].type is cli_args._positive_finite_float
        assert actions["duplex_eval_ref_audio"].type is cli_args._existing_file
        assert actions["duplex_eval_output_dir"].type is Path
        assert actions["duplex_eval_score_dir"].type is Path
        assert actions["duplex_eval_ids"].nargs == "+"
        assert actions["duplex_eval_exclude_ids"].nargs == "+"
        assert actions["duplex_eval_family"].choices == ["all", "rtd", "pr"]
        assert actions["duplex_eval_pace"].choices == ["realtime", "as-fast-as-possible"]
        assert actions["duplex_eval_clock"].choices == ["media"]
        for dest in ("duplex_eval_overwrite", "duplex_eval_allow_invalid_clock", "duplex_eval_no_artifacts"):
            assert actions[dest].default is False
            assert actions[dest].nargs == 0

    def test_registered_by_add_omni_args(self, monkeypatch) -> None:
        parser = argparse.ArgumentParser()
        for helper in (
            "add_daily_omni_cli_args",
            "add_omniinteract_cli_args",
            "add_seed_tts_cli_args",
            "add_multi_stage_cli_args",
            "add_diffusion_cli_args",
        ):
            monkeypatch.setattr(cli_args, helper, lambda _parser: None)
        recorded: list[bool] = []
        original = cli_args.add_duplex_eval_cli_args

        def _spy(target: argparse.ArgumentParser) -> None:
            recorded.append(True)
            original(target)

        monkeypatch.setattr(cli_args, "add_duplex_eval_cli_args", _spy)
        cli_args.add_omni_args(parser)
        assert recorded == [True]
        assert _EXPECTED_OPTIONS <= {
            opt for action in parser._actions for opt in action.option_strings if opt.startswith(_DUPLEX_EVAL_PREFIX)
        }


# ================================================================== #
# T24: ``_non_negative_int``
# ================================================================== #


class TestNonNegativeInt:
    """T24: accepts 0/positive, rejects negatives and non-integers."""

    def test_accepts_zero_and_positive(self) -> None:
        assert cli_args._non_negative_int("0") == 0
        assert cli_args._non_negative_int("7") == 7

    def test_rejects_negative(self) -> None:
        with pytest.raises(argparse.ArgumentTypeError, match="non-negative"):
            cli_args._non_negative_int("-1")

    def test_rejects_non_integer(self) -> None:
        with pytest.raises(ValueError, match="invalid literal"):
            cli_args._non_negative_int("abc")


# ================================================================== #
# T25-T33: ``preprocess_serve_args`` validation rules
# ================================================================== #


class TestBackendRule:
    """T25: backend must be ``openai-realtime-duplex``."""

    def test_valid_backend_passes(self) -> None:
        args = _serve_args()
        cli_args.preprocess_serve_args(args)
        assert args.num_prompts == 3  # not explicit → default

    def test_wrong_backend_raises(self) -> None:
        with pytest.raises(ValueError, match="backend openai-realtime-duplex"):
            cli_args.preprocess_serve_args(_serve_args(backend="openai-chat-omni"))


class TestEndpointRule:
    """T26: endpoint must be ``/v1/realtime``."""

    def test_valid_endpoint_passes(self) -> None:
        cli_args.preprocess_serve_args(_serve_args(endpoint="/v1/realtime"))

    def test_wrong_endpoint_raises(self) -> None:
        with pytest.raises(ValueError, match="endpoint /v1/realtime"):
            cli_args.preprocess_serve_args(_serve_args(endpoint="/v1/chat/completions"))


class TestRefAudioRule:
    """T27: ``--duplex-eval-ref-audio`` is mandatory."""

    def test_valid_ref_audio_passes(self) -> None:
        cli_args.preprocess_serve_args(_serve_args(duplex_eval_ref_audio="/tmp/ref.wav"))

    def test_missing_ref_audio_raises(self) -> None:
        with pytest.raises(ValueError, match="ref-audio"):
            cli_args.preprocess_serve_args(_serve_args(duplex_eval_ref_audio=None))


class TestBannedFlags:
    """T28: ``--ignore-eos`` / ``--profile`` / ``--skip-tokenizer-init`` banned."""

    @pytest.mark.parametrize("flag", ["ignore_eos", "profile", "skip_tokenizer_init"])
    def test_banned_flag_raises(self, flag: str) -> None:
        with pytest.raises(ValueError, match="does not support"):
            cli_args.preprocess_serve_args(_serve_args(**{flag: True}))

    def test_all_disabled_passes(self) -> None:
        cli_args.preprocess_serve_args(_serve_args(ignore_eos=False, profile=False, skip_tokenizer_init=False))


class TestProbeRequestRateRule:
    """T29: ``--probe-request-rate`` is rejected."""

    def test_zero_passes(self) -> None:
        cli_args.preprocess_serve_args(_serve_args(probe_request_rate=0.0))

    def test_positive_raises(self) -> None:
        with pytest.raises(ValueError, match="probe-request-rate"):
            cli_args.preprocess_serve_args(_serve_args(probe_request_rate=1.0))


class TestPaceClockRule:
    """T30: non-realtime pace needs ``--duplex-eval-allow-invalid-clock``."""

    def test_realtime_pace_passes(self) -> None:
        cli_args.preprocess_serve_args(_serve_args(duplex_eval_pace="realtime"))

    def test_fast_pace_without_optin_raises(self) -> None:
        with pytest.raises(ValueError, match="clock=invalid"):
            cli_args.preprocess_serve_args(_serve_args(duplex_eval_pace="as-fast-as-possible"))

    def test_fast_pace_with_optin_passes(self) -> None:
        cli_args.preprocess_serve_args(
            _serve_args(duplex_eval_pace="as-fast-as-possible", duplex_eval_allow_invalid_clock=True)
        )


class TestIdsExcludeMutex:
    """T31: ``--duplex-eval-ids`` and ``--duplex-eval-exclude-ids`` are exclusive."""

    def test_single_selector_passes(self) -> None:
        cli_args.preprocess_serve_args(_serve_args(duplex_eval_ids=["565"]))
        cli_args.preprocess_serve_args(_serve_args(duplex_eval_exclude_ids=["565"]))

    def test_both_selectors_raise(self) -> None:
        with pytest.raises(ValueError, match="mutually exclusive"):
            cli_args.preprocess_serve_args(_serve_args(duplex_eval_ids=["565"], duplex_eval_exclude_ids=["566"]))


class TestNumPromptsDefault:
    """T32: omitted ``--num-prompts`` folds to 3; explicit value is preserved."""

    def test_default_is_three(self) -> None:
        args = _serve_args(num_prompts=99, explicit_keys=())
        cli_args.preprocess_serve_args(args)
        assert args.num_prompts == 3

    def test_explicit_value_preserved(self) -> None:
        args = _serve_args(num_prompts=7, explicit_keys=("num_prompts",))
        cli_args.preprocess_serve_args(args)
        assert args.num_prompts == 7


class TestMaxConcurrencyRule:
    """T33: omitted ``--max-concurrency`` folds to 1; non-positive is rejected."""

    def test_default_is_one(self) -> None:
        args = _serve_args(max_concurrency=None)
        cli_args.preprocess_serve_args(args)
        assert args.max_concurrency == 1

    @pytest.mark.parametrize("value", [0, -3])
    def test_non_positive_raises(self, value: int) -> None:
        with pytest.raises(ValueError, match="max-concurrency"):
            cli_args.preprocess_serve_args(_serve_args(max_concurrency=value))

    def test_positive_value_preserved(self) -> None:
        args = _serve_args(max_concurrency=4)
        cli_args.preprocess_serve_args(args)
        assert args.max_concurrency == 4
