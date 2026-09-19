# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""CI guard for Omni-DuplexEval scoring regression.

Pipeline: generate -> evaluate -> summarize -> assert.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from tests.e2e.online_serving.helpers.minicpmo_4_5_duplex import (
    SERVER_PARAMS as DUPLEX_TEST_PARAMS,
)
from tests.e2e.online_serving.helpers.minicpmo_4_5_duplex import (
    realtime_url,
    resolve_ref_audio,
)
from tests.helpers.mark import hardware_test
from vllm_omni.benchmarks.duplex.omni_duplex_eval_eval import summarize_scores
from vllm_omni.benchmarks.duplex.omni_duplex_eval_metrics import PROTOCOL_PIN

# Thresholds (hardcoded, community accuracy test pattern).
# baseline commit: 873e9bff7c545c5cda79fdb93a867f09e443b61d
#
# RECORD MODE: all thresholds are 0.0 until the first nightly run produces
# real scores. After that, tighten to ~50% of the measured baseline (paper
# MiniCPM-o 4.5 on the 100-point scale maps to ~1.15 content / ~2.40
# temporal / ~0.20 PR on the code scales; see README "Reference paper").
_MIN_RTD_MEAN_CONTENT_SCORE = 0.0  # 3-point scale (0.00-3.00)
_MIN_RTD_MEAN_TEMPORAL_SCORE = 0.0  # 0-3 scale
_MIN_PR_MEAN_ALL_SUCCESS = 0.0  # 0-1 scale

_CONFIG_PATH = Path(__file__).resolve().parent / "omni_duplex_eval_ci_config.json"


def _load_ci_config() -> dict:
    return json.loads(_CONFIG_PATH.read_text(encoding="utf-8"))


pytestmark = [pytest.mark.full_model, pytest.mark.omni]

_DUPLEX_SERVER_PARAMS = list(DUPLEX_TEST_PARAMS)


_RESULT_DIR = Path(__file__).resolve().parent / "results"


@hardware_test(res={"cuda": ["H100", "B200"], "npu": "A3"}, num_cards=1)
@pytest.mark.parametrize("omni_server", _DUPLEX_SERVER_PARAMS, indirect=True)
def test_omni_duplex_eval_ci(omni_server, tmp_path: Path, judge_server: str) -> None:
    """CI guard: generate -> evaluate -> summarize -> assert scores."""
    config = _load_ci_config()
    response_root = tmp_path / "responses"
    score_root = tmp_path / "scores"
    response_root.mkdir(parents=True, exist_ok=True)
    score_root.mkdir(parents=True, exist_ok=True)

    all_ids = sorted(set(sid for sids in config["sample_ids_per_split"].values() for sid in sids))
    exclude = set(config.get("exclude_ids", []))
    sample_ids = [sid for sid in all_ids if sid not in exclude]

    # Phase 1: Generate
    dataset_ref = config["dataset"]
    if config.get("dataset_revision"):
        dataset_ref = f"{dataset_ref}@{config['dataset_revision']}"
    _run_cli(
        [
            "vllm",
            "bench",
            "omni-duplex-eval",
            "generate",
            "--dataset",
            dataset_ref,
            "--split",
            "all",
            "--ids",
            *sample_ids,
            "--url",
            realtime_url(omni_server),
            "--model",
            omni_server.model,
            "--ref-audio",
            str(resolve_ref_audio()),
            "--response-root",
            str(response_root),
        ]
    )

    # Phase 2: Evaluate
    judge_cfg = config["judge"]
    _run_cli(
        [
            "vllm",
            "bench",
            "omni-duplex-eval",
            "evaluate",
            "--dataset",
            dataset_ref,
            "--split",
            "all",
            "--ids",
            *sample_ids,
            "--response-root",
            str(response_root),
            "--score-root",
            str(score_root),
            "--judge-model",
            judge_cfg["model"],
            "--judge-base-url",
            judge_server,
            "--judge-video-mode",
            judge_cfg.get("video_mode", "video_url"),
            "--judge-fps",
            str(judge_cfg.get("fps", 2)),
            "--eval-workers",
            str(judge_cfg.get("eval_workers", 4)),
        ]
    )

    score_files = list(Path(score_root).rglob("*.json"))
    assert len(score_files) == len(sample_ids), f"expected {len(sample_ids)} score files, found {len(score_files)}"

    # Phase 3: Summarize + persist result JSON (Buildkite artifact upload pattern)
    summary = summarize_scores(score_root)
    _RESULT_DIR.mkdir(parents=True, exist_ok=True)
    result_path = _RESULT_DIR / "summary.json"
    result_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[Omni-DuplexEval] summary saved to {result_path}")
    print(json.dumps(summary, ensure_ascii=False, indent=2))

    # Phase 4: Assertions
    assert summary["protocol_pin"] == PROTOCOL_PIN, f"protocol_pin changed: {summary['protocol_pin']} != {PROTOCOL_PIN}"
    assert summary["samples"] == len(sample_ids), f"expected {len(sample_ids)} samples, got {summary['samples']}"

    rtd = summary.get("rtd", {})
    pr = summary.get("pr", {})
    assert rtd.get("by_task", {}).keys() >= {
        "RTD_world_knowledge",
        "RTD_counting",
        "RTD_fine_grained_movement",
        "RTD_interaction_relation",
        "RTD_OCR",
        "RTD_Omni",
    }, f"RTD by_task keys mismatch: {rtd.get('by_task', {}).keys()}"
    assert pr.get("by_task", {}).keys() >= {
        "correction",
        "proactive_reminder",
        "post_event_reminder",
    }, f"PR by_task keys mismatch: {pr.get('by_task', {}).keys()}"

    assert rtd.get("mean_content_score", 0.0) >= _MIN_RTD_MEAN_CONTENT_SCORE, (
        f"RTD content score {rtd.get('mean_content_score'):.4f} < {_MIN_RTD_MEAN_CONTENT_SCORE}"
    )
    assert rtd.get("mean_avg_temporal_score", 0.0) >= _MIN_RTD_MEAN_TEMPORAL_SCORE, (
        f"RTD temporal score {rtd.get('mean_avg_temporal_score'):.4f} < {_MIN_RTD_MEAN_TEMPORAL_SCORE}"
    )
    assert pr.get("mean_all_success", 0.0) >= _MIN_PR_MEAN_ALL_SUCCESS, (
        f"PR success rate {pr.get('mean_all_success'):.4f} < {_MIN_PR_MEAN_ALL_SUCCESS}"
    )


_CLI_TIMEOUT = 1800


def _run_cli(argv: list[str]) -> None:
    """Run a CLI command and assert success within timeout.

    stdout/stderr are printed in real-time (not captured) so CI logs
    show full generate / evaluate progress — same pattern as PR#6817
    perf tests (``-s -v`` + ``run_benchmark()``).
    """
    label = " ".join(argv[:3])
    print(f"\n[Omni-DuplexEval] Running: {' '.join(argv)}\n", flush=True)
    try:
        result = subprocess.run(argv, capture_output=False, timeout=_CLI_TIMEOUT)
    except subprocess.TimeoutExpired:
        raise AssertionError(f"[{label}] timed out after {_CLI_TIMEOUT}s: {' '.join(argv)}") from None
    assert result.returncode == 0, f"[{label}] failed (exit code {result.returncode}): {' '.join(argv)}"
    print(f"\n[Omni-DuplexEval] Done: {label}\n", flush=True)
