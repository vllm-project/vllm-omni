# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Offline unit tests for the Omni-DuplexEval ``patch.py`` wiring (M3).

Covers the data dispatch in ``get_samples``, ``is_duplex_eval`` exact
equality, per-request metadata attachment, the realtime dispatcher, the
generation-ledger finalizer and the result assembly / placeholder session
metrics contract.

Everything is exercised with stubs, monkeypatching and in-memory objects: no
GPU, weights, network or real server process is required.

See ``plans/omni-duplex-eval-backend-design.md`` §5.3, §4.4 and §9.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import inspect
from pathlib import Path
from types import SimpleNamespace

import pytest

try:
    import vllm_omni.benchmarks.patch.patch as patch_mod
    from vllm_omni.benchmarks.duplex import omni_duplex_eval_eval as eval_mod
    from vllm_omni.benchmarks.duplex import omni_duplex_eval_judge as judge_mod
except ImportError:  # pragma: no cover - offline stub environment
    pytest.skip("vLLM / patch module is not importable", allow_module_level=True)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

_DUPLEX_NAME = "omni-duplex-eval"


def _make_options(**overrides) -> patch_mod.DuplexEvalSessionOptions:
    base: dict = {
        "response_root": Path("/tmp/duplex-responses"),
        "score_dir": Path("/tmp/duplex-scores"),
        "ref_audio": "tests/assets/ref.wav",
        "write_artifacts": False,
    }
    base.update(overrides)
    return patch_mod.DuplexEvalSessionOptions(**base)


def _make_sample(
    sample_id: str = "s1",
    split: str = "RTD_OCR",
    family: str = "rtd",
) -> SimpleNamespace:
    return SimpleNamespace(id=sample_id, split=split, family=family, task_type=None, video="", question_audio=None)


def _make_duplex_request(sample=None, options=None) -> patch_mod.DuplexEvalSampleRequest:
    return patch_mod.DuplexEvalSampleRequest(
        prompt="",
        prompt_len=0,
        expected_output_len=0,
        request_id="0",
        duplex_eval_sample=sample if sample is not None else _make_sample(),
        duplex_eval_options=options if options is not None else _make_options(),
    )


def _make_rfi() -> patch_mod.RequestFuncInput:
    return patch_mod.RequestFuncInput(
        prompt="",
        api_url="http://127.0.0.1:8000",
        prompt_len=0,
        output_len=0,
        model="test-model",
    )


def _serve_dataset_args(**overrides) -> argparse.Namespace:
    base: dict = {
        "dataset_name": _DUPLEX_NAME,
        "backend": "openai-realtime-duplex",
        "dataset_path": None,
        "seed": 1,
        "request_id_prefix": "",
        "disable_shuffle": True,
        "num_prompts": 0,
        "duplex_eval_split": "RTD_OCR",
        "duplex_eval_family": "all",
        "duplex_eval_media_root": None,
        "duplex_eval_limit": 4,
        "duplex_eval_ids": None,
        "duplex_eval_exclude_ids": ["565"],
    }
    base.update(overrides)
    return argparse.Namespace(**base)


class _StubDataset:
    """Captures ``__init__``/``sample`` calls and returns a fixed row count."""

    def __init__(self, returns: int = 3) -> None:
        self.returns = returns
        self.init_kwargs: dict | None = None
        self.sample_kwargs: dict | None = None

    def __call__(self, **kwargs):  # used as the monkeypatched class
        self.init_kwargs = kwargs
        return self

    def sample(self, tokenizer, num_requests, *, request_id_prefix="", options=None):
        self.sample_kwargs = {
            "tokenizer": tokenizer,
            "num_requests": num_requests,
            "request_id_prefix": request_id_prefix,
            "options": options,
        }
        return [object() for _ in range(self.returns)]


# ================================================================== #
# ``get_samples`` dispatch
# ================================================================== #


class TestGetSamplesDispatch:
    """``dataset_name == "omni-duplex-eval"`` routes to ``DuplexEvalDataset``."""

    def test_dispatches_and_folds_num_prompts(self, monkeypatch) -> None:
        stub = _StubDataset(returns=3)
        options_sentinel = object()
        monkeypatch.setattr(patch_mod, "DuplexEvalDataset", stub)
        monkeypatch.setattr(patch_mod, "options_from_args", lambda args: options_sentinel)

        args = _serve_dataset_args()
        requests = patch_mod.get_samples(args, tokenizer=None)

        assert len(requests) == 3
        assert args.num_prompts == 3  # 0/all folded to measured length
        assert stub.init_kwargs is not None
        assert stub.init_kwargs["dataset"] == patch_mod.DEFAULT_DUPLEX_EVAL_DATASET
        assert stub.init_kwargs["split"] == "RTD_OCR"
        assert stub.init_kwargs["limit"] == 4
        assert stub.init_kwargs["exclude_ids"] == ["565"]
        assert stub.init_kwargs["disable_shuffle"] is True
        assert stub.sample_kwargs["num_requests"] == 0
        assert stub.sample_kwargs["options"] is options_sentinel

    def test_dataset_path_overrides_default(self, monkeypatch) -> None:
        stub = _StubDataset(returns=1)
        monkeypatch.setattr(patch_mod, "DuplexEvalDataset", stub)
        monkeypatch.setattr(patch_mod, "options_from_args", lambda args: object())

        patch_mod.get_samples(_serve_dataset_args(dataset_path="/mirror"), tokenizer=None)

        assert stub.init_kwargs["dataset"] == "/mirror"

    def test_empty_selection_raises(self, monkeypatch) -> None:
        monkeypatch.setattr(patch_mod, "DuplexEvalDataset", _StubDataset(returns=0))
        monkeypatch.setattr(patch_mod, "options_from_args", lambda args: object())

        with pytest.raises(ValueError, match="No Omni-DuplexEval samples"):
            patch_mod.get_samples(_serve_dataset_args(), tokenizer=None)


# ================================================================== #
# ``is_duplex_eval`` exact equality
# ================================================================== #


class TestIsDuplexEvalExactEquality:
    """T: the dispatch predicate is an exact ``==`` match, not a prefix."""

    def test_source_uses_exact_equality(self) -> None:
        source = inspect.getsource(patch_mod.get_samples)
        assert 'is_duplex_eval = args.dataset_name == "omni-duplex-eval"' in source
        assert "startswith" not in source.split("is_duplex_eval")[1].split("\n")[0]

    def test_exact_name_dispatches(self, monkeypatch) -> None:
        stub = _StubDataset(returns=1)
        monkeypatch.setattr(patch_mod, "DuplexEvalDataset", stub)
        monkeypatch.setattr(patch_mod, "options_from_args", lambda args: object())
        patch_mod.get_samples(_serve_dataset_args(dataset_name=_DUPLEX_NAME), tokenizer=None)
        assert stub.init_kwargs is not None

    @pytest.mark.parametrize("name", ["omni-duplex-eval-x", "duplex-eval", "omni-duplex"])
    def test_near_miss_falls_through(self, monkeypatch, name: str) -> None:
        created: list[dict] = []

        class _Guarded:
            def __init__(self, **kwargs):
                created.append(kwargs)

            def sample(self, *args, **kwargs):  # pragma: no cover - must not run
                raise AssertionError("duplex dispatch must not trigger")

        fallback: list[bool] = []
        monkeypatch.setattr(patch_mod, "DuplexEvalDataset", _Guarded)
        monkeypatch.setattr(patch_mod, "options_from_args", lambda args: object())
        monkeypatch.setattr(patch_mod, "get_samples_old", lambda args, tokenizer: fallback.append(True) or [])

        args = _serve_dataset_args(dataset_name=name)
        with contextlib.suppress(Exception):
            patch_mod.get_samples(args, tokenizer=None)

        assert created == []
        assert fallback == [True]


# ================================================================== #
# ``_attach_duplex_eval_to_request_func_input``
# ================================================================== #


class TestAttachToRequestFuncInput:
    """T: per-request sample/options are mounted and the call is idempotent."""

    def test_mounts_and_is_idempotent(self) -> None:
        sample = _make_sample()
        options = _make_options()
        request = _make_duplex_request(sample=sample, options=options)
        rfi = _make_rfi()

        patch_mod._attach_duplex_eval_to_request_func_input(request, rfi)
        assert rfi.duplex_eval_sample is sample
        assert rfi.duplex_eval_options is options

        patch_mod._attach_duplex_eval_to_request_func_input(request, rfi)
        assert rfi.duplex_eval_sample is sample
        assert rfi.duplex_eval_options is options

    def test_ignores_non_duplex_request(self) -> None:
        plain = patch_mod.SampleRequest(prompt="p", prompt_len=1, expected_output_len=1, request_id="0")
        rfi = _make_rfi()
        patch_mod._attach_duplex_eval_to_request_func_input(plain, rfi)
        assert not hasattr(rfi, "duplex_eval_sample")
        assert not hasattr(rfi, "duplex_eval_options")


# ================================================================== #
# Realtime dispatch
# ================================================================== #


class TestRealtimeDispatch:
    """A request carrying ``duplex_eval_sample`` uses ``_async_request_duplex_eval``."""

    def test_routes_to_duplex_eval(self, monkeypatch) -> None:
        seen: list[object] = []

        async def _fake(rfi, *, pbar=None):
            seen.append(pbar)
            return "SENTINEL"

        monkeypatch.setattr(patch_mod, "_async_request_duplex_eval", _fake)
        rfi = _make_rfi()
        rfi.duplex_eval_sample = _make_sample()

        result = asyncio.run(patch_mod.async_request_openai_realtime_duplex(rfi, session=None, pbar=None))

        assert result == "SENTINEL"
        assert seen == [None]


# ================================================================== #
# ``_async_request_duplex_eval`` placeholder session metrics
# ================================================================== #


class TestDuplexEvalRequestPlaceholder:
    """Per-output ``duplex_session_metrics`` is a dict with three ``None`` keys."""

    def test_placeholder_keys_are_none(self, monkeypatch) -> None:
        async def _fake_case(sample, config):
            return patch_mod.DuplexEvalCaseResult(id="s1", split="RTD_OCR", family="rtd", success=True, latency_s=0.5)

        monkeypatch.setattr(patch_mod, "run_duplex_eval_case", _fake_case)
        rfi = _make_rfi()
        rfi.duplex_eval_sample = _make_sample()
        rfi.duplex_eval_options = _make_options()

        output = asyncio.run(patch_mod._async_request_duplex_eval(rfi, pbar=None))

        assert output.success is True
        assert set(output.duplex_session_metrics) == {
            "mean_ttft_ms",
            "mean_ttfp_ms",
            "mean_rtf",
            "source",
        }
        for key in ("mean_ttft_ms", "mean_ttfp_ms", "mean_rtf"):
            assert output.duplex_session_metrics[key] is None
        assert output.duplex_session_metrics["source"] == "omni-duplex-eval"
        assert output.duplex_eval_case_result.success is True

    def test_missing_options_marks_failure(self, monkeypatch) -> None:
        rfi = _make_rfi()
        rfi.duplex_eval_sample = _make_sample()
        # no duplex_eval_options mounted → the request must fail, not raise
        output = asyncio.run(patch_mod._async_request_duplex_eval(rfi, pbar=None))
        assert output.success is False
        assert output.error
        assert output.duplex_eval_case_result is not None


# ================================================================== #
# Finalize: generation ledger only, no judge in the serve path
# ================================================================== #


class TestFinalizeNoJudge:
    """``finalize_duplex_eval_batch`` never invokes a judge entry point."""

    def test_finalize_never_calls_judge(self, monkeypatch) -> None:
        judge_calls: list[object] = []

        def _boom(*args, **kwargs):
            judge_calls.append(args)
            raise AssertionError("judge entry point must not run in the serve path")

        monkeypatch.setattr(judge_mod, "DuplexJudge", _boom)
        monkeypatch.setattr(judge_mod, "select_judge_text", _boom)
        monkeypatch.setattr(eval_mod, "evaluate_sample", _boom)

        request = _make_duplex_request()
        case_result = patch_mod.DuplexEvalCaseResult(id="s1", split="RTD_OCR", family="rtd", success=True)
        output = patch_mod.MixRequestFuncOutput()
        setattr(output, "duplex_eval_case_result", case_result)

        summary = patch_mod.finalize_duplex_eval_batch([request], [output])

        assert judge_calls == []
        assert summary is not None
        assert summary["phase"] == "generate"
        assert summary["judge_enabled"] is False
        assert summary["scored"] == 0
        assert "score_summary" not in summary

    def test_patch_module_has_no_judge_symbols(self) -> None:
        assert not hasattr(patch_mod, "DuplexJudge")
        assert not hasattr(patch_mod, "select_judge_text")
        source = Path(patch_mod.__file__).read_text(encoding="utf-8")
        assert "omni_duplex_eval_judge" not in source


# ================================================================== #
# Result assembly + session-metrics aggregation semantics
# ================================================================== #


class TestResultAssembly:
    """``phase == "generate"`` is what the benchmark writes into the result."""

    def test_result_wiring_is_present(self) -> None:
        source = inspect.getsource(patch_mod.benchmark)
        assert "duplex_eval_summary = finalize_duplex_eval_batch(input_requests, outputs)" in source
        assert "if duplex_eval_summary is not None:" in source
        assert 'result["duplex_eval"] = duplex_eval_summary' in source

    def test_summary_phase_is_generate(self) -> None:
        request = _make_duplex_request()
        case_result = patch_mod.DuplexEvalCaseResult(id="s1", split="RTD_OCR", family="rtd", success=True)
        output = patch_mod.MixRequestFuncOutput()
        setattr(output, "duplex_eval_case_result", case_result)
        summary = patch_mod.finalize_duplex_eval_batch([request], [output])
        assert summary is not None
        assert summary["phase"] == "generate"


class TestPlaceholderSuppressesPseudoZero:
    """A ``None`` placeholder must not become a valid ``0.0`` metric sample."""

    def _run(self, with_placeholder: bool):
        from vllm.benchmarks.serve import TaskType

        output = patch_mod.MixRequestFuncOutput()
        output.ttft = 0.0
        output.latency = 1.0
        output.success = True
        output.output_len = 0
        if with_placeholder:
            output.duplex_session_metrics = {
                "mean_ttft_ms": None,
                "mean_ttfp_ms": None,
                "mean_rtf": None,
                "source": "omni-duplex-eval",
            }
        request = patch_mod.SampleRequest(prompt="", prompt_len=0, expected_output_len=0, request_id="0")
        metrics, _ = patch_mod.calculate_metrics(
            input_requests=[request],
            outputs=[output],
            dur_s=1.0,
            tokenizer=None,
            selected_percentiles=[50.0],
            goodput_config_dict={},
            task_type=TaskType.GENERATION,
            selected_percentile_metrics=[],
            max_concurrency=1,
            request_rate=float("inf"),
            benchmark_duration=1.0,
        )
        return metrics

    def test_placeholder_yields_zero_samples(self, capsys) -> None:
        metrics = self._run(with_placeholder=True)
        capsys.readouterr()
        assert metrics.num_ttft_samples == 0
        assert getattr(metrics, "num_audio_ttfp_samples", 0) == 0
        assert getattr(metrics, "num_audio_rtf_samples", 0) == 0

    def test_without_placeholder_zero_ttft_counts(self, capsys) -> None:
        metrics = self._run(with_placeholder=False)
        capsys.readouterr()
        assert metrics.num_ttft_samples == 1
