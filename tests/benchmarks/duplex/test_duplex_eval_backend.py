# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Offline unit tests for the Omni-DuplexEval backend module (D1 + D2).

Covers contract, routing, exclude/take semantics, generation-ledger math,
artifact publication, merge protocol and session-options translation.
See ``plans/omni-duplex-eval-backend-design.md`` §9.1 for the test matrix.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

try:
    from vllm_omni.entrypoints.cli.benchmark import cli_args as _cli_args
except ImportError:  # vllm may be missing in the offline stub environment
    _cli_args = None  # type: ignore[assignment]

# ------------------------------------------------------------------ #
# Module-level markers (required by tools/pre_commit/check_test_marks.py)
# ------------------------------------------------------------------ #
pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

# ------------------------------------------------------------------ #
# Internal helpers used by multiple tests
# ------------------------------------------------------------------ #


@dataclass
class _FakeOutput:
    """Minimal output stub with the attributes ``finalize_duplex_eval_batch`` reads."""

    duplex_eval_case_result: object = None
    success: bool = True
    latency: float = 1.0
    error: str = ""
    prompt_len: int = 0
    duplex_session_metrics: dict[str, object] | None = None


def _make_options(**overrides: Any) -> dict[str, Any]:
    """Return a full ``DuplexEvalSessionOptions`` field dict with testable defaults."""
    base = {
        "response_root": Path("/tmp/responses"),
        "score_dir": Path("/tmp/scores"),
        "ref_audio": "tests/assets/ref.wav",
        "fps": 1.0,
        "mix": "question",
        "pace": "realtime",
        "clock": "media",
        "unit_ms": 1000,
        "overwrite": False,
        "exclude_ids": (),
        "write_artifacts": False,  # off by default to avoid filesystem I/O
    }
    base.update(overrides)
    return base


def _make_sample(
    sample_id: str = "s1",
    split: str = "RTD_OCR",
    family: str = "rtd",
    task_type: str | None = None,
) -> SimpleNamespace:
    """Return a lightweight sample-like object for tests that don't need the real dataclass."""
    return SimpleNamespace(
        id=sample_id,
        split=split,
        family=family,
        task_type=task_type,
        video="",
        question_audio=None,
    )


# ================================================================== #
# T1: ``DuplexEvalDataset.sample`` contract
# ================================================================== #


class TestDuplexEvalDatasetContract:
    """T1: mandatory ``options``, ``num_requests=0``, prefix, shuffle guard."""

    def test_options_is_mandatory_keyword_only(self) -> None:
        """Calling ``sample()`` without ``options`` raises TypeError."""
        from vllm_omni.benchmarks.data_modules.duplex_eval_dataset import (
            DuplexEvalDataset,
        )

        dataset = DuplexEvalDataset(dataset="dummy", disable_shuffle=True)
        with pytest.raises(TypeError, match="options"):
            dataset.sample(None, 1)  # type: ignore[call-arg]

    def test_num_requests_zero_returns_all(self, monkeypatch) -> None:
        """``num_requests=0`` selects every available sample (no capping)."""
        from vllm_omni.benchmarks.data_modules.duplex_eval_dataset import (
            DuplexEvalDataset,
            DuplexEvalSessionOptions,
        )

        samples_pool = [_make_sample(f"s{i}") for i in range(5)]
        monkeypatch.setattr(
            "vllm_omni.benchmarks.data_modules.duplex_eval_dataset.load_samples",
            lambda *a, **kw: samples_pool,
        )
        options = DuplexEvalSessionOptions(**_make_options())
        # Pin a single split so the split-agnostic stub is not reused per split
        # when ``split="all"`` expands to every known split.
        dataset = DuplexEvalDataset(dataset="dummy", split="RTD_OCR", disable_shuffle=True)
        requests = dataset.sample(None, 0, options=options)
        assert len(requests) == 5

    def test_request_id_prefix(self, monkeypatch) -> None:
        from vllm_omni.benchmarks.data_modules.duplex_eval_dataset import (
            DuplexEvalDataset,
            DuplexEvalSessionOptions,
        )

        monkeypatch.setattr(
            "vllm_omni.benchmarks.data_modules.duplex_eval_dataset.load_samples",
            lambda *a, **kw: [_make_sample("x1")],
        )
        options = DuplexEvalSessionOptions(**_make_options())
        dataset = DuplexEvalDataset(dataset="dummy", disable_shuffle=True)
        requests = dataset.sample(None, 1, request_id_prefix="pre-", options=options)
        assert requests[0].request_id == "pre-0"

    def test_disable_shuffle_preserves_order(self, monkeypatch) -> None:
        from vllm_omni.benchmarks.data_modules.duplex_eval_dataset import (
            DuplexEvalDataset,
            DuplexEvalSessionOptions,
        )

        ids = [f"s{i}" for i in range(10)]
        samples_pool = [_make_sample(sample_id=id_) for id_ in ids]
        monkeypatch.setattr(
            "vllm_omni.benchmarks.data_modules.duplex_eval_dataset.load_samples",
            lambda *a, **kw: samples_pool,
        )
        options = DuplexEvalSessionOptions(**_make_options())
        dataset = DuplexEvalDataset(dataset="dummy", disable_shuffle=True)
        requests = dataset.sample(None, 10, options=options)
        assert [r.duplex_eval_sample.id for r in requests] == ids


# ================================================================== #
# T2: Dataset routing
# ================================================================== #


class TestDatasetRouting:
    """T2: directory / .parquet / JSON manifest all reach different loaders."""

    def test_json_manifest_routes_to_read_manifest(self, tmp_path, monkeypatch) -> None:
        from vllm_omni.benchmarks.data_modules.duplex_eval_dataset import (
            DuplexEvalDataset,
            DuplexEvalSessionOptions,
        )

        manifest = tmp_path / "samples.json"
        manifest.write_text(
            json.dumps([{"id": "m1", "split": "RTD_OCR", "video": "clip.mp4"}]),
            encoding="utf-8",
        )
        options = DuplexEvalSessionOptions(**_make_options())
        dataset = DuplexEvalDataset(dataset=str(manifest), disable_shuffle=True)
        requests = dataset.sample(None, 1, options=options)
        assert len(requests) == 1
        assert requests[0].duplex_eval_sample.id == "m1"


# ================================================================== #
# T3: ``DuplexEvalSampleRequest`` field integrity
# ================================================================== #


class TestSampleRequestFields:
    """T3: fields are correctly mounted; others are zero-values."""

    def test_fields_are_mounted(self, monkeypatch) -> None:
        from vllm_omni.benchmarks.data_modules.duplex_eval_dataset import (
            DuplexEvalDataset,
            DuplexEvalSampleRequest,
            DuplexEvalSessionOptions,
        )

        ds_sample = _make_sample("s42", split="PR_correction", family="pr")
        monkeypatch.setattr(
            "vllm_omni.benchmarks.data_modules.duplex_eval_dataset.load_samples",
            lambda *a, **kw: [ds_sample],
        )
        options = DuplexEvalSessionOptions(**_make_options())
        dataset = DuplexEvalDataset(dataset="dummy", disable_shuffle=True)
        requests = dataset.sample(None, 1, options=options)
        req = requests[0]
        assert isinstance(req, DuplexEvalSampleRequest)
        assert req.duplex_eval_sample is ds_sample
        assert req.duplex_eval_options is options
        assert req.prompt == ""
        assert req.prompt_len == 0
        assert req.multi_modal_data is None


# ================================================================== #
# T4: ``normalize_exclude_ids()``
# ================================================================== #


class TestNormalizeExcludeIds:
    """T4: bare id and split/id forms; invalid format raises."""

    def test_bare_id_accepted(self) -> None:
        from vllm_omni.benchmarks.data_modules.duplex_eval_dataset import (
            normalize_exclude_ids,
        )

        result = normalize_exclude_ids(["565"])
        assert "565" in result

    def test_split_id_accepted(self) -> None:
        from vllm_omni.benchmarks.data_modules.duplex_eval_dataset import (
            normalize_exclude_ids,
        )

        result = normalize_exclude_ids(["RTD_OCR/565"])
        assert "RTD_OCR/565" in result

    def test_empty_input_returns_empty(self) -> None:
        from vllm_omni.benchmarks.data_modules.duplex_eval_dataset import (
            normalize_exclude_ids,
        )

        assert normalize_exclude_ids([]) == frozenset()

    def test_invalid_format_raises(self) -> None:
        from vllm_omni.benchmarks.data_modules.duplex_eval_dataset import (
            normalize_exclude_ids,
        )

        with pytest.raises(ValueError, match="invalid exclude-id"):
            normalize_exclude_ids(["a/b/c"])  # too many slashes

    def test_whitespace_skipped(self) -> None:
        from vllm_omni.benchmarks.data_modules.duplex_eval_dataset import (
            normalize_exclude_ids,
        )

        result = normalize_exclude_ids(["  ", "565"])
        assert len(result) == 1


# ================================================================== #
# T5: exclude BEFORE limit per split
# ================================================================== #


class TestExcludeBeforeLimit:
    """T5: exclude is applied per-split before limit (§8.5 D-2)."""

    def test_exclude_before_limit(self, monkeypatch) -> None:
        from vllm_omni.benchmarks.data_modules.duplex_eval_dataset import (
            DuplexEvalDataset,
            DuplexEvalSessionOptions,
        )

        # 6 samples across 2 splits (3 each)
        samples = [
            _make_sample("s0", split="RTD_OCR", family="rtd"),
            _make_sample("s1", split="RTD_OCR", family="rtd"),
            _make_sample("s2", split="RTD_OCR", family="rtd"),
            _make_sample("s3", split="PR_correction", family="pr"),
            _make_sample("s4", split="PR_correction", family="pr"),
            _make_sample("s5", split="PR_correction", family="pr"),
        ]

        def fake_load(dataset, *, split, family, media_root, ids):
            return [s for s in samples if s.split == split and (family == "all" or s.family == family)]

        monkeypatch.setattr(
            "vllm_omni.benchmarks.data_modules.duplex_eval_dataset.load_samples",
            fake_load,
        )

        exclude = ("s0",)
        options = DuplexEvalSessionOptions(**_make_options(exclude_ids=exclude, write_artifacts=False))
        # limit=4 per split → after exclude there are 2 per split,
        # but we cap at 2 each = 4 total
        dataset = DuplexEvalDataset(
            dataset="dummy",
            split="all",
            family="all",
            limit=2,
            exclude_ids=exclude,
            disable_shuffle=True,
        )
        requests = dataset.sample(None, 10, options=options)
        selected_ids = [r.duplex_eval_sample.id for r in requests]
        assert "s0" not in selected_ids, "excluded sample still present"
        assert len(selected_ids) <= 4, "limit after exclude not respected"
        # With 2 per split after exclude(3-1=2), limit=2 takes all, so total=4
        assert len(selected_ids) == 4


# ================================================================== #
# T6: ``--ids`` and ``--exclude-ids`` mutex (dataset-level).
# ================================================================== #


class TestIdsExcludeMutex:
    """T6: the preprocess-serve check rejects both ids and exclude-ids."""

    def test_mutex_raises_value_error(self) -> None:
        """The serve-side CLI check rejects both flags.

        Uses a guard so the test is skipped when vllm (and therefore
        ``cli_args``) is not importable (offline stub environment).
        """
        if _cli_args is None:
            pytest.skip("cli_args not importable (vllm may be missing)")

        import argparse

        args = argparse.Namespace(
            dataset_name="omni-duplex-eval",
            backend="openai-realtime-duplex",
            endpoint="/v1/realtime",
            duplex_eval_ref_audio="tests/assets/ref.wav",
            duplex_eval_pace="realtime",
            duplex_eval_allow_invalid_clock=False,
            duplex_eval_ids=["s1"],
            duplex_eval_exclude_ids=["s2"],
            explicit_keys=(),
        )
        with pytest.raises(ValueError, match="mutually exclusive"):
            _cli_args.preprocess_serve_args(args)  # type: ignore[union-attr]


# ================================================================== #
# T10: ``generation_summary`` counting
# ================================================================== #


class TestGenerationSummary:
    """T10: total/generated/generation_failed arithmetic."""

    def test_all_success(self) -> None:
        from vllm_omni.benchmarks.duplex_eval import (
            DuplexEvalCaseResult,
            generation_summary,
        )

        results = [
            DuplexEvalCaseResult(id="s1", split="RTD_OCR", family="rtd", success=True),
            DuplexEvalCaseResult(id="s2", split="RTD_OCR", family="rtd", success=True),
        ]
        s = generation_summary(results)
        assert s["total"] == 2
        assert s["generated"] == 2
        assert s["generation_failed"] == 0
        assert s["failure_ids"] == []

    def test_partial_failure(self) -> None:
        from vllm_omni.benchmarks.duplex_eval import (
            DuplexEvalCaseResult,
            generation_summary,
        )

        results = [
            DuplexEvalCaseResult(id="ok", split="RTD_OCR", family="rtd", success=True),
            DuplexEvalCaseResult(
                id="fail",
                split="PR_correction",
                family="pr",
                success=False,
                error="timeout",
            ),
        ]
        s = generation_summary(results)
        assert s["total"] == 2
        assert s["generated"] == 1
        assert s["generation_failed"] == 1
        assert "PR_correction/fail" in s["failure_ids"]


# ================================================================== #
# T11: ``finalize_duplex_eval_batch`` no duplex samples → ``None``
# ================================================================== #


class TestFinalizeNone:
    """T11: returns None when no duplex-eval sample ran."""

    def test_returns_none_when_no_duplex_results(self) -> None:
        from vllm_omni.benchmarks.data_modules.omniinteract_dataset import (
            OmniInteractSampleRequest,
        )
        from vllm_omni.benchmarks.duplex_eval import finalize_duplex_eval_batch

        # Non-duplex requests — should be filtered out
        requests = [SimpleNamespace(__class__=OmniInteractSampleRequest)]
        outputs = [SimpleNamespace()]
        result = finalize_duplex_eval_batch(requests, outputs)
        assert result is None


# ================================================================== #
# T12: ``finalize_duplex_eval_batch`` never calls a judge (no network)
# ================================================================== #


class TestFinalizeNoNetwork:
    """T12: no judge/network calls inside finalize (v2 §5.2)."""

    def test_finalize_succeeds_without_judge(self, monkeypatch) -> None:
        from vllm_omni.benchmarks.data_modules.duplex_eval_dataset import (
            DuplexEvalSampleRequest,
            DuplexEvalSessionOptions,
        )
        from vllm_omni.benchmarks.duplex_eval import (
            DuplexEvalCaseResult,
            finalize_duplex_eval_batch,
        )

        # If finalize ever touches a judge, this would raise.
        monkeypatch.setattr(
            "vllm_omni.benchmarks.data_modules.duplex_eval_dataset.load_samples",
            lambda *a, **kw: [],
        )

        options = DuplexEvalSessionOptions(**_make_options())
        sample_ds = _make_sample("s1")
        req = DuplexEvalSampleRequest(
            prompt="",
            prompt_len=0,
            expected_output_len=0,
            request_id="0",
            duplex_eval_sample=sample_ds,
            duplex_eval_options=options,
        )
        result = DuplexEvalCaseResult(id="s1", split="RTD_OCR", family="rtd", success=True)
        output = _FakeOutput(duplex_eval_case_result=result)

        summary = finalize_duplex_eval_batch([req], [output])
        assert summary is not None
        assert summary["generated"] == 1


# ================================================================== #
# T13: ``finalize_*`` output has no judge fields
# ================================================================== #


class TestFinalizeNoJudgeFields:
    """T13: returned dict must NOT contain score_summary or judge_*."""

    def test_no_judge_fields(self, monkeypatch) -> None:
        from vllm_omni.benchmarks.data_modules.duplex_eval_dataset import (
            DuplexEvalSampleRequest,
            DuplexEvalSessionOptions,
        )
        from vllm_omni.benchmarks.duplex_eval import (
            DuplexEvalCaseResult,
            finalize_duplex_eval_batch,
        )

        options = DuplexEvalSessionOptions(**_make_options())
        sample_ds = _make_sample("s1")
        req = DuplexEvalSampleRequest(
            prompt="",
            prompt_len=0,
            expected_output_len=0,
            request_id="0",
            duplex_eval_sample=sample_ds,
            duplex_eval_options=options,
        )
        result = DuplexEvalCaseResult(id="s1", split="RTD_OCR", family="rtd", success=True)
        output = _FakeOutput(duplex_eval_case_result=result)

        summary = finalize_duplex_eval_batch([req], [output])
        assert summary is not None
        assert "score_summary" not in summary
        assert "judge_enabled" in summary
        assert summary["judge_enabled"] is False
        assert summary["scored"] == 0
        assert summary["score_failed"] == 0


# ================================================================== #
# T14: artifact atomicity
# ================================================================== #


class TestFinalizeArtifacts:
    """T14: batch_summary.json and eval_manifest.jsonl written atomically."""

    def test_artifacts_written(self, tmp_path, monkeypatch) -> None:
        from vllm_omni.benchmarks.data_modules.duplex_eval_dataset import (
            DuplexEvalSampleRequest,
            DuplexEvalSessionOptions,
        )
        from vllm_omni.benchmarks.duplex_eval import (
            BATCH_ARTIFACTS,
            DuplexEvalCaseResult,
            finalize_duplex_eval_batch,
        )

        score_dir = tmp_path / "scores"
        options = DuplexEvalSessionOptions(
            **_make_options(
                score_dir=score_dir,
                write_artifacts=True,
            )
        )
        sample_ds = _make_sample("s1")
        req = DuplexEvalSampleRequest(
            prompt="",
            prompt_len=0,
            expected_output_len=0,
            request_id="0",
            duplex_eval_sample=sample_ds,
            duplex_eval_options=options,
        )
        result = DuplexEvalCaseResult(id="s1", split="RTD_OCR", family="rtd", success=True)
        output = _FakeOutput(duplex_eval_case_result=result)

        summary = finalize_duplex_eval_batch([req], [output])
        assert summary is not None
        assert summary["artifacts_complete"] is True
        for name in BATCH_ARTIFACTS:
            assert (score_dir / name).exists(), f"{name} not written"

    def test_artifact_failure_reported(self, tmp_path, monkeypatch) -> None:
        from vllm_omni.benchmarks.data_modules.duplex_eval_dataset import (
            DuplexEvalSampleRequest,
            DuplexEvalSessionOptions,
        )
        from vllm_omni.benchmarks.duplex_eval import (
            DuplexEvalCaseResult,
            finalize_duplex_eval_batch,
        )

        score_dir = tmp_path / "scores"
        # Make the score_dir a file so writing fails.
        score_dir.touch()

        options = DuplexEvalSessionOptions(
            **_make_options(
                score_dir=score_dir,
                write_artifacts=True,
            )
        )
        sample_ds = _make_sample("s1")
        req = DuplexEvalSampleRequest(
            prompt="",
            prompt_len=0,
            expected_output_len=0,
            request_id="0",
            duplex_eval_sample=sample_ds,
            duplex_eval_options=options,
        )
        result = DuplexEvalCaseResult(id="s1", split="RTD_OCR", family="rtd", success=True)
        output = _FakeOutput(duplex_eval_case_result=result)

        summary = finalize_duplex_eval_batch([req], [output])
        assert summary is not None
        assert summary["artifacts_complete"] is False
        assert len(summary.get("artifact_errors", [])) > 0


# ================================================================== #
# T15: duplex_eval should not pollute ``result_percentile_metrics``.
#       Structural assertion: the module does not export a judge.
# ================================================================== #


class TestNoJudgePollution:
    """T15: structural assertion — duplex_eval module has no judge imports."""

    def test_no_judge_import(self) -> None:
        import vllm_omni.benchmarks.duplex_eval as de

        source = Path(de.__file__).read_text(encoding="utf-8")
        assert "judge" not in source.lower() or "merge_duplex_eval_into_result" in source
        # Verify that the judge-evaluation function names are not present.
        assert "_evaluate_rows" not in source
        assert "evaluate_sample" not in source


# ================================================================== #
# T16: placeholder ``duplex_session_metrics`` suppress pseudo-zero TTFT
# ================================================================== #


class TestPlaceholderSessionMetrics:
    """T16: ``duplex_session_metrics`` is a dict with ``mean_ttft_ms: None``."""

    def test_placeholder_structure(self) -> None:
        metrics = {
            "mean_ttft_ms": None,
            "mean_ttfp_ms": None,
            "mean_rtf": None,
            "source": "omni-duplex-eval",
        }
        assert isinstance(metrics, dict)
        assert metrics["mean_ttft_ms"] is None


# ================================================================== #
# T17: ``options_from_args`` default values
# ================================================================== #


class TestOptionsFromArgs:
    """T17: CLI args → DuplexEvalSessionOptions with correct defaults."""

    def test_defaults(self) -> None:
        from vllm_omni.benchmarks.duplex_eval import options_from_args

        args = SimpleNamespace()
        opts = options_from_args(args)
        assert opts.fps == 1.0
        assert opts.mix == "question"
        assert opts.pace == "realtime"
        assert opts.clock == "media"
        assert opts.unit_ms == 1000
        assert opts.overwrite is False
        assert opts.exclude_ids == ()

    def test_fields_are_read(self) -> None:
        from vllm_omni.benchmarks.duplex_eval import options_from_args

        args = SimpleNamespace(
            duplex_eval_output_dir="custom-responses",
            duplex_eval_score_dir="custom-scores",
            duplex_eval_ref_audio="audio.wav",
            duplex_eval_fps=2.0,
            duplex_eval_mix="question",
            duplex_eval_pace="as-fast-as-possible",
            duplex_eval_clock="media",
            duplex_eval_unit_ms=500,
            duplex_eval_overwrite=True,
            duplex_eval_exclude_ids=["565"],
            duplex_eval_no_artifacts=True,
        )
        opts = options_from_args(args)
        assert str(opts.response_root).endswith("custom-responses")
        assert str(opts.score_dir).endswith("custom-scores")
        assert opts.ref_audio == "audio.wav"
        assert opts.fps == 2.0
        assert opts.pace == "as-fast-as-possible"
        assert opts.unit_ms == 500
        assert opts.overwrite is True
        assert opts.exclude_ids == ("565",)
        assert opts.write_artifacts is False


# ================================================================== #
# T19: ``_locate_result_file`` uniqueness
# ================================================================== #


class TestLocateResultFile:
    """T19: 0 or ≥2 matches → AssertionError; 1 match → returned."""

    def test_zero_matches_raises(self, tmp_path) -> None:
        from vllm_omni.benchmarks.duplex_eval import _locate_result_file

        with pytest.raises(AssertionError, match="expected exactly one"):
            _locate_result_file(
                bench_dir=tmp_path,
                test_name="nonexistent",
                dataset_name="omni-duplex-eval",
                flow=1,
                num_prompt=3,
                since=time.time(),
            )

    def test_exactly_one_match(self, tmp_path) -> None:
        from vllm_omni.benchmarks.duplex_eval import _locate_result_file

        result = tmp_path / "result_my_test_abc_omni-duplex-eval_1_3_in42_out84_20250301-120000.json"
        result.write_text("{}", encoding="utf-8")
        since = time.time() - 10
        found = _locate_result_file(
            bench_dir=tmp_path,
            test_name="my_test",
            dataset_name="omni-duplex-eval",
            flow=1,
            num_prompt=3,
            since=since,
        )
        assert found.name == result.name


# ================================================================== #
# T20: ``merge_duplex_eval_into_result`` phase guard
# ================================================================== #


class TestMergePhaseGuard:
    """T20: ``phase != "generate"`` → AssertionError; normal merge advances phase."""

    def test_refuses_second_merge(self, tmp_path) -> None:
        from vllm_omni.benchmarks.duplex_eval import (
            merge_duplex_eval_into_result,
        )

        duplex = {
            "phase": "generate+evaluate+summarize",
            "total": 3,
            "generated": 3,
        }
        payload = {
            "completed": True,
            "e2el": {"mean": 1.5},
            "Hardware": {"gpu": "H100"},
            "duplex_eval": duplex,
        }
        result_path = tmp_path / "result.json"
        result_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        with pytest.raises(AssertionError, match="refusing to merge"):
            merge_duplex_eval_into_result(
                result_path,
                score_summary={"samples": 3},
                judge_meta={"model": "judge-model"},
                phases=["generate", "evaluate", "summarize"],
            )

    def test_normal_merge_succeeds(self, tmp_path) -> None:
        from vllm_omni.benchmarks.duplex_eval import (
            merge_duplex_eval_into_result,
        )

        duplex = {
            "phase": "generate",
            "total": 3,
            "generated": 3,
            "score_dir": "duplex-eval-scores",
        }
        payload = {
            "completed": True,
            "e2el": {"mean": 1.5},
            "Hardware": {"gpu": "H100"},
            "baseline": {"mean_tpot_ms": 200.0},
            "duplex_eval": duplex,
        }
        result_path = tmp_path / "result.json"
        result_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        merged = merge_duplex_eval_into_result(
            result_path,
            score_summary={"samples": 3},
            judge_meta={"model": "judge-model"},
            phases=["generate", "evaluate", "summarize"],
        )
        d = merged["duplex_eval"]
        assert d["phase"] == "generate+evaluate+summarize"
        assert d["phases"] == ["generate", "evaluate", "summarize"]
        assert d["judge_enabled"] is True
        assert d["scored"] == 3
        # Extra fields from the original payload were NOT touched.
        assert merged["completed"] is True
        assert merged["e2el"]["mean"] == 1.5


# ================================================================== #
# T21: idempotent merge / atomicity
# ================================================================== #


class TestMergeIdempotent:
    """T21: second merge is blocked; .tmp residue does not exist on success."""

    def test_second_merge_blocked(self, tmp_path) -> None:
        from vllm_omni.benchmarks.duplex_eval import (
            merge_duplex_eval_into_result,
        )

        duplex = {
            "phase": "generate",
            "total": 1,
            "generated": 1,
            "score_dir": "duplex-eval-scores",
        }
        payload = {"completed": True, "duplex_eval": duplex}
        result_path = tmp_path / "result.json"
        result_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        merge_duplex_eval_into_result(
            result_path,
            score_summary={"samples": 1},
            judge_meta={},
            phases=["generate", "evaluate", "summarize"],
        )
        # Second merge → phase guard blocks.
        with pytest.raises(AssertionError, match="refusing to merge"):
            merge_duplex_eval_into_result(
                result_path,
                score_summary={"samples": 1},
                judge_meta={},
                phases=["generate", "evaluate", "summarize"],
            )

    def test_no_tmp_residue_on_success(self, tmp_path) -> None:
        from vllm_omni.benchmarks.duplex_eval import (
            merge_duplex_eval_into_result,
        )

        duplex = {
            "phase": "generate",
            "total": 1,
            "generated": 1,
            "score_dir": "duplex-eval-scores",
        }
        payload = {"completed": True, "duplex_eval": duplex}
        result_path = tmp_path / "result.json"
        result_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        merge_duplex_eval_into_result(
            result_path,
            score_summary={"samples": 1},
            judge_meta={},
            phases=["generate", "evaluate", "summarize"],
        )
        tmp_files = list(tmp_path.glob("*.tmp"))
        assert len(tmp_files) == 0, f".tmp residue: {tmp_files}"
