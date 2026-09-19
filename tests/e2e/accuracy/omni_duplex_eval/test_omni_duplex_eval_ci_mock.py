# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""CPU-based mock tests for the Omni-DuplexEval CI guard.

Runs the *real* ``test_omni_duplex_eval_ci()`` guard function without GPU /
real model by stubbing the heavy I/O:

  - ``_run_cli`` is replaced by a spy that records argv (verifies the exact
    ``vllm bench omni-duplex-eval`` argument assembly) and never executes.
  - ``resolve_ref_audio`` is stubbed to a local path.
  - Fake score files are pre-written under ``tmp_path/scores/<split>/`` using
    the sample IDs from ``omni_duplex_eval_ci_config.json`` (minus excludes),
    so ``summarize_scores()`` and the Phase-4 assertions run for real.

This validates parameter construction, the score-file-count assertion, the
``protocol_pin`` check, the RTD/PR ``by_task`` key checks, and the threshold
comparisons — all against the genuine guard logic.

Also includes unit tests for the ``judge_server`` fixture endpoint resolution,
``_run_cli`` (success / non-zero / timeout), and ``summarize_scores`` shape.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pytest

from vllm_omni.benchmarks.duplex.omni_duplex_eval_eval import summarize_scores
from vllm_omni.benchmarks.duplex.omni_duplex_eval_metrics import PROTOCOL_PIN

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

_CONFIG_PATH = Path(__file__).resolve().parent / "omni_duplex_eval_ci_config.json"


def _load_config() -> dict:
    return json.loads(_CONFIG_PATH.read_text(encoding="utf-8"))


def _sample_ids(config: dict) -> list[str]:
    """Mirror the ID assembly in ``test_omni_duplex_eval_ci()``."""
    all_ids = sorted(set(sid for sids in config["sample_ids_per_split"].values() for sid in sids))
    exclude = set(config.get("exclude_ids", []))
    return [sid for sid in all_ids if sid not in exclude]


def _rtd_task_type(split: str) -> str:
    return split.replace("RTD_", "").lower()


def _pr_task_type(split: str) -> str:
    if "correction" in split:
        return "correction"
    if "post_event" in split:
        return "post_event_reminder"
    return "proactive_reminder"


# ---------------------------------------------------------------------------
# Fake score file writers (shaped like evaluate_sample() output)
# ---------------------------------------------------------------------------


def _fake_rtd_score(
    sample_id: str, split: str, *, content_score: float = 2.8, temporal_score: int = 3, protocol_pin: str = PROTOCOL_PIN
) -> dict:
    return {
        "id": sample_id,
        "family": "rtd",
        "task_type": _rtd_task_type(split),
        "protocol_pin": protocol_pin,
        "temporal": {
            "sentences": [
                {
                    "sentence": "The object is moving left.",
                    "start": 0.0,
                    "end": 2.0,
                    "sentence_start": 0.0,
                    "sentence_end": 2.0,
                    "sentence_duration": 2.0,
                    "window_start": 0.0,
                    "window_end": 1.0,
                    "error": None,
                    "temporal_score": temporal_score,
                    "is_relevant": 1,
                }
            ],
            "summary": {
                "avg_temporal_score": float(temporal_score),
                "total_sentences": 1,
                "evaluated_sentences": 1,
                "error_count": 0,
                "relevant_sentences_count": 1,
                "irrelevant_sentences_count": 0,
                "score_distribution": {
                    "3_points": 1 if temporal_score == 3 else 0,
                    "2_points": 1 if temporal_score == 2 else 0,
                    "1_point": 0,
                    "0_points": 0,
                },
                "relevance_stats": {
                    "first_relevant_time": 0.0,
                    "total_relevant_duration": 2.0,
                    "total_irrelevant_duration": 0.0,
                    "irrelevant_duration_ratio": 0.0,
                },
            },
        },
        "content": {"content_score": content_score, "content_reasoning": "Looks correct."},
    }


def _fake_pr_score(sample_id: str, split: str, *, all_success: bool = True, protocol_pin: str = PROTOCOL_PIN) -> dict:
    return {
        "id": sample_id,
        "family": "pr",
        "task_type": _pr_task_type(split),
        "protocol_pin": protocol_pin,
        "pr": {
            "events": [
                {
                    "response_segment": "I see the object moving.",
                    "success_score": 1 if all_success else 0,
                    "reasoning": "Correct.",
                }
            ],
            "all_success": all_success,
            "total_score": 1 if all_success else 0,
        },
    }


def _write_scores_from_config(
    score_root: Path,
    config: dict,
    *,
    content_score: float = 2.8,
    temporal_score: int = 3,
    pr_success: bool = True,
    protocol_pin: str = PROTOCOL_PIN,
) -> int:
    """Write one fake score file per selected sample ID into the split dir.

    Returns the number of files written (== expected ``summary["samples"]``).
    """
    exclude = set(config.get("exclude_ids", []))
    count = 0
    for split, ids in config["sample_ids_per_split"].items():
        subdir = score_root / split
        subdir.mkdir(parents=True, exist_ok=True)
        for sid in ids:
            if sid in exclude:
                continue
            if split.startswith("RTD"):
                row = _fake_rtd_score(
                    sid, split, content_score=content_score, temporal_score=temporal_score, protocol_pin=protocol_pin
                )
            else:
                row = _fake_pr_score(sid, split, all_success=pr_success, protocol_pin=protocol_pin)
            (subdir / f"{sid}.json").write_text(json.dumps(row), encoding="utf-8")
            count += 1
    return count


# ---------------------------------------------------------------------------
# Driving the REAL guard test function
# ---------------------------------------------------------------------------


@dataclass
class _FakeServer:
    """Minimal typed stand-in for the ``omni_server`` fixture.

    Only the read-only attributes the guard touches (host / port / model) are
    modeled; keeps the CPU-only mock free of the real OmniServer dependency.
    """

    host: str
    port: int
    model: str


def _call_real_guard(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, judge_url: str = "http://127.0.0.1:8001"
) -> list[list[str]]:
    """Call ``test_omni_duplex_eval_ci()`` with stubbed heavy I/O.

    Returns the argv lists captured by the ``_run_cli`` spy.
    """
    from tests.e2e.accuracy.omni_duplex_eval import test_omni_duplex_eval_ci as guard

    fake_server = _FakeServer(host="127.0.0.1", port=9999, model="mock-model")
    calls: list[list[str]] = []

    def spy_run_cli(argv: list[str]) -> None:
        calls.append(argv)

    monkeypatch.setattr(guard, "_run_cli", spy_run_cli)
    monkeypatch.setattr(guard, "resolve_ref_audio", lambda: tmp_path / "ref.wav")
    monkeypatch.setattr(guard, "_RESULT_DIR", tmp_path)  # write summary.json to tmp_path, not real results/
    guard.test_omni_duplex_eval_ci(fake_server, tmp_path, judge_url)
    return calls


def test_real_guard_full_flow_passes(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Real guard passes end-to-end with good scores and stubbed CLI."""
    config = _load_config()
    expected = len(_sample_ids(config))
    written = _write_scores_from_config(tmp_path / "scores", config)
    assert written == expected, f"test setup: expected {expected} score files, wrote {written}"
    _call_real_guard(monkeypatch, tmp_path)  # must not raise


def test_real_guard_assembles_generate_and_evaluate_argv(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Verify the exact CLI argument assembly for both phases."""
    config = _load_config()
    _write_scores_from_config(tmp_path / "scores", config)
    calls = _call_real_guard(monkeypatch, tmp_path)

    assert len(calls) == 2, f"expected generate + evaluate, got {len(calls)} calls"
    generate, evaluate = calls
    assert generate[:4] == ["vllm", "bench", "omni-duplex-eval", "generate"]
    assert evaluate[:4] == ["vllm", "bench", "omni-duplex-eval", "evaluate"]

    # generate: dataset ref, split, ids, url, model, ref-audio, response-root
    assert generate[generate.index("--dataset") + 1] == config["dataset"]
    assert generate[generate.index("--split") + 1] == "all"
    gen_ids = generate[generate.index("--ids") + 1 : generate.index("--url")]
    assert gen_ids == _sample_ids(config)
    assert generate[generate.index("--url") + 1] == "ws://127.0.0.1:9999/v1/realtime?duplex=1"
    assert generate[generate.index("--model") + 1] == "mock-model"
    assert Path(generate[generate.index("--ref-audio") + 1]).name == "ref.wav"
    assert generate[generate.index("--response-root") + 1] == str(tmp_path / "responses")

    # evaluate: dataset ref, ids, roots, judge config
    assert evaluate[evaluate.index("--dataset") + 1] == config["dataset"]
    eval_ids = evaluate[evaluate.index("--ids") + 1 : evaluate.index("--response-root")]
    assert eval_ids == _sample_ids(config)
    assert evaluate[evaluate.index("--response-root") + 1] == str(tmp_path / "responses")
    assert evaluate[evaluate.index("--score-root") + 1] == str(tmp_path / "scores")
    assert evaluate[evaluate.index("--judge-model") + 1] == config["judge"]["model"]
    assert evaluate[evaluate.index("--judge-base-url") + 1] == "http://127.0.0.1:8001"
    assert evaluate[evaluate.index("--judge-video-mode") + 1] == config["judge"].get("video_mode", "video_url")
    assert int(evaluate[evaluate.index("--judge-fps") + 1]) == config["judge"].get("fps", 2)
    assert int(evaluate[evaluate.index("--eval-workers") + 1]) == config["judge"].get("eval_workers", 4)


def test_real_guard_uses_dataset_revision_when_set(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """When ``dataset_revision`` is set, the dataset ref becomes ``ds@rev``."""
    from tests.e2e.accuracy.omni_duplex_eval import test_omni_duplex_eval_ci as guard

    config = _load_config()
    _write_scores_from_config(tmp_path / "scores", config)

    patched = dict(config)
    patched["dataset_revision"] = "abcd1234"
    monkeypatch.setattr(guard, "_load_ci_config", lambda: patched)

    calls = _call_real_guard(monkeypatch, tmp_path)
    generate, evaluate = calls
    assert generate[generate.index("--dataset") + 1] == f"{config['dataset']}@abcd1234"
    assert evaluate[evaluate.index("--dataset") + 1] == f"{config['dataset']}@abcd1234"


def test_real_guard_fails_on_low_rtd_content(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Real guard raises when RTD content score is below threshold.

    The production thresholds are 0.0 (record mode). We temporarily inject
    a non-zero threshold to verify the assertion logic still catches low scores.
    """
    from tests.e2e.accuracy.omni_duplex_eval import test_omni_duplex_eval_ci as guard

    monkeypatch.setattr(guard, "_MIN_RTD_MEAN_CONTENT_SCORE", 2.0)
    monkeypatch.setattr(guard, "_MIN_RTD_MEAN_TEMPORAL_SCORE", 0.0)
    config = _load_config()
    _write_scores_from_config(tmp_path / "scores", config, content_score=1.0, temporal_score=3)
    with pytest.raises(AssertionError, match="RTD content score"):
        _call_real_guard(monkeypatch, tmp_path)


def test_real_guard_fails_on_low_pr_success(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Real guard raises when PR success rate is below threshold.

    The production thresholds are 0.0 (record mode). We temporarily inject
    a non-zero threshold to verify the assertion logic still catches low scores.
    """
    from tests.e2e.accuracy.omni_duplex_eval import test_omni_duplex_eval_ci as guard

    monkeypatch.setattr(guard, "_MIN_PR_MEAN_ALL_SUCCESS", 0.5)
    config = _load_config()
    _write_scores_from_config(tmp_path / "scores", config, pr_success=False)
    with pytest.raises(AssertionError, match="PR success rate"):
        _call_real_guard(monkeypatch, tmp_path)


def test_real_guard_fails_on_missing_score_files(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Real guard raises when score file count does not match sample count."""
    config = _load_config()
    _write_scores_from_config(tmp_path / "scores", config)
    # Add an extra file to break the count assertion
    extra_dir = tmp_path / "scores" / "RTD_world_knowledge"
    extra_dir.mkdir(parents=True, exist_ok=True)
    (extra_dir / "extra.json").write_text("{}", encoding="utf-8")
    with pytest.raises(AssertionError, match="expected .* score files"):
        _call_real_guard(monkeypatch, tmp_path)


def test_real_guard_fails_on_protocol_pin_mismatch(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Real guard raises when its PROTOCOL_PIN constant drifts from the one
    embedded in ``summarize_scores()`` output.

    ``summarize_scores()`` hardcodes the pin, so the mismatch is simulated by
    patching the guard module's copy of the constant.
    """
    from tests.e2e.accuracy.omni_duplex_eval import test_omni_duplex_eval_ci as guard

    config = _load_config()
    _write_scores_from_config(tmp_path / "scores", config)
    monkeypatch.setattr(guard, "PROTOCOL_PIN", "deadbeef")
    with pytest.raises(AssertionError, match="protocol_pin changed"):
        _call_real_guard(monkeypatch, tmp_path)


# ---------------------------------------------------------------------------
# judge_server fixture endpoint resolution logic (mirrors conftest.py)
# ---------------------------------------------------------------------------


def _resolve_judge_url() -> str:
    """Mirror of the ``judge_server`` fixture body in ``conftest.py``."""
    import os

    config = _load_config()
    judge_cfg = config["judge"]
    env_name = judge_cfg.get("base_url_env")
    base_url = os.environ.get(env_name) if env_name else None
    if base_url is not None:
        return base_url
    return str(judge_cfg.get("base_url", "http://127.0.0.1:8001"))


def test_judge_server_uses_env_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("VLLM_DUPLEX_EVAL_JUDGE_URL", "http://judge-env:8000")
    assert _resolve_judge_url() == "http://judge-env:8000"


def test_judge_server_falls_back_to_config(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("VLLM_DUPLEX_EVAL_JUDGE_URL", raising=False)
    config = _load_config()
    assert _resolve_judge_url() == config["judge"]["base_url"]


# ---------------------------------------------------------------------------
# _run_cli unit tests
# ---------------------------------------------------------------------------


def test_run_cli_success() -> None:
    from tests.e2e.accuracy.omni_duplex_eval.test_omni_duplex_eval_ci import _run_cli

    _run_cli(["echo", "ok"])


def test_run_cli_nonzero_exit() -> None:
    from tests.e2e.accuracy.omni_duplex_eval.test_omni_duplex_eval_ci import _run_cli

    with pytest.raises(AssertionError, match="failed"):
        _run_cli(["false"])


def test_run_cli_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    from tests.e2e.accuracy.omni_duplex_eval import test_omni_duplex_eval_ci as guard

    monkeypatch.setattr(guard, "_CLI_TIMEOUT", 1)
    with pytest.raises(AssertionError, match="timed out"):
        guard._run_cli(["sleep", "5"])


# ---------------------------------------------------------------------------
# summarize_scores shape / boundary tests
# ---------------------------------------------------------------------------


@pytest.fixture
def score_root(tmp_path: Path) -> Path:
    root = tmp_path / "scores"
    root.mkdir(parents=True, exist_ok=True)
    return root


def test_summarize_scores_from_config(score_root: Path) -> None:
    """Full config-derived score set produces the expected summary keys."""
    config = _load_config()
    _write_scores_from_config(score_root, config)
    summary = summarize_scores(score_root)
    assert summary["protocol_pin"] == PROTOCOL_PIN
    assert summary["samples"] == len(_sample_ids(config))
    assert summary["rtd"]["by_task"].keys() == {
        "RTD_world_knowledge",
        "RTD_counting",
        "RTD_fine_grained_movement",
        "RTD_interaction_relation",
        "RTD_OCR",
        "RTD_Omni",
    }
    assert summary["pr"]["by_task"].keys() == {
        "correction",
        "proactive_reminder",
        "post_event_reminder",
    }


def test_summarize_scores_empty(score_root: Path) -> None:
    """Empty score directory returns minimal summary."""
    summary = summarize_scores(score_root)
    assert summary["protocol_pin"] == PROTOCOL_PIN
    assert summary["samples"] == 0
    assert "rtd" not in summary
    assert "pr" not in summary
