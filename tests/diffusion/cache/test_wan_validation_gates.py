# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import json
import runpy
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]
SCRIPTS = Path(__file__).resolve().parents[3] / "tools/wan_teacache_validation"


@pytest.mark.parametrize(
    "selection", [{"threshold": 0.1}, {"calibration_all_gates_passed": True, "diagnostic_only": True}]
)
def test_unqualified_selection_cannot_start_independent_acceptance(tmp_path, monkeypatch, selection):
    (tmp_path / "selection-pp1.json").write_text(json.dumps(selection))
    monkeypatch.setenv("WAN_VALIDATION_ROOT", str(tmp_path))
    monkeypatch.setattr(
        sys, "argv", ["campaign.py", "--pp", "1", "--cfg", "1", "--phase", "validate", "--root", str(tmp_path / "run")]
    )
    monkeypatch.setattr("subprocess.run", lambda *a, **k: pytest.fail("Unqualified candidate started a model run"))
    with pytest.raises(RuntimeError, match="all-gates-passing"):
        runpy.run_path(str(SCRIPTS / "campaign.py"), run_name="__main__")


@pytest.mark.parametrize("gate", ["cache_use_passed", "timing_quality_passed"])
def test_junit_retains_nonmetric_acceptance_failure(tmp_path, monkeypatch, gate):
    comparison = tmp_path / "comparison.json"
    comparison.write_text(
        json.dumps(
            {
                "mean_ssim": 0.99,
                "min_video_ssim": 0.98,
                "mean_temporal_error_ratio": 0.01,
                "latency_ratio": 0.8,
                "videos": [],
                gate: False,
            }
        )
    )
    (tmp_path / "results.json").write_text(json.dumps({"args": {"split": "validation", "pp": 2, "cfg": 2}}))
    report = tmp_path / "report.xml"
    monkeypatch.setattr(sys, "argv", ["quality_junit.py", str(comparison), str(report)])
    runpy.run_path(str(SCRIPTS / "quality_junit.py"), run_name="__main__")
    suite = ET.parse(report).getroot()
    assert suite.attrib["tests"] == "5"
    assert suite.attrib["failures"] == "1"
    assert suite.find(f"testcase[@name='{gate}']/failure") is not None


def _write_run(run, *, pp, cfg, requests, split="smoke", drop_groups=(), drop_latent=None, trace_detail="detailed"):
    """Write a synthetic paired campaign run in the layout generate.py produces."""
    import torch

    sys.path.insert(0, str(SCRIPTS))
    from audit import expected_groups

    steps = 6 if split == "smoke" else 50
    args = {"split": split, "pp": pp, "cfg": cfg, "threshold": 0.1, "cache_warmup_steps": 0, "limit": None}
    for mode in ("none", "cache"):
        (run / mode).mkdir(parents=True)
        records = [{"name": name, "prompt": prompt, "seed": seed, "seconds": 1.0} for name, prompt, seed in requests]
        (run / mode / "results.json").write_text(json.dumps({"args": args, "records": records}))
        (run / "trace" / mode).mkdir(parents=True)
        for name, _, _ in requests:
            if (mode, name) != drop_latent:
                torch.save(torch.ones(2, 3), run / "trace" / mode / f"{name}.pt")
    rows = [
        {
            "kind": "decision",
            "request_name": name,
            "step": step,
            "compute": step % 2 == 0,
            "pp_rank": p,
            "cfg_rank": c,
            "context": branch,
        }
        for p, c, branch in sorted(expected_groups(pp, cfg))
        if (p, c, branch) not in drop_groups
        for name, _, _ in requests
        for step in range(steps)
    ]
    (run / "trace/cache/rank-0.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    (run / "provenance.json").write_text(
        json.dumps({"sha": "abc", "source_hashes": {"a": "1"}, "trace_detail": trace_detail})
    )
    (run / "production-profile.json").write_text(json.dumps({"pp_size": pp, "cfg_size": cfg}))


_SMOKE_REQUESTS = [("00-42", "a ball", 42), ("01-42", "a cat", 42)]


def _audit(run, pp, cfg):
    sys.path.insert(0, str(SCRIPTS))
    from audit import audit_run

    return audit_run(run, pp, cfg)


def test_audit_accepts_full_pp_cfg_request_coverage(tmp_path):
    _write_run(tmp_path, pp=2, cfg=2, requests=_SMOKE_REQUESTS)
    summary = _audit(tmp_path, 2, 2)
    assert summary["all_branches_skipped_blocks"]
    assert set(summary["counts"]) == {
        "pp0/cfg0/teacache_positive",
        "pp0/cfg1/teacache_negative",
        "pp1/cfg0/teacache_positive",
        "pp1/cfg1/teacache_negative",
    }
    assert len(summary["latents"]) == 2


def test_audit_rejects_missing_off_diagonal_groups(tmp_path):
    drop = {(0, 1, "teacache_negative"), (1, 0, "teacache_positive")}
    _write_run(tmp_path, pp=2, cfg=2, requests=_SMOKE_REQUESTS, drop_groups=drop)
    with pytest.raises(AssertionError, match="Missing decisions"):
        _audit(tmp_path, 2, 2)


@pytest.mark.parametrize("mode", ["none", "cache"])
def test_audit_rejects_missing_latent_pair(tmp_path, mode):
    _write_run(tmp_path, pp=1, cfg=1, requests=_SMOKE_REQUESTS, drop_latent=(mode, "01-42"))
    with pytest.raises(AssertionError, match="latents do not match"):
        _audit(tmp_path, 1, 1)


def test_audit_rejects_incomplete_step_sequence(tmp_path):
    _write_run(tmp_path, pp=1, cfg=1, requests=_SMOKE_REQUESTS)
    log = tmp_path / "trace/cache/rank-0.jsonl"
    log.write_text("".join(line + "\n" for line in log.read_text().splitlines() if '"step": 3' not in line))
    with pytest.raises(AssertionError, match="step sequence"):
        _audit(tmp_path, 1, 1)


def _calibration_inputs(tmp_path, prompts):
    inputs = tmp_path / "inputs"
    (inputs / "model").mkdir(parents=True)
    (inputs / "prompts.json").write_text(json.dumps({"calibration": prompts, "validation": []}))
    for name in ("model.json", "coefficients-pp1.json", "coefficients-pp1.fit.json"):
        (inputs / name).write_text("{}")
    return inputs


def _calibration_candidate(campaign, prompts, *, limit=None, timing="matched", cfg=1):
    run = campaign / "tune-0.1"
    used = prompts[:limit] if limit else prompts
    requests = [(f"{i:02d}-{seed}", prompt, seed) for i, prompt in enumerate(used) for seed in (17, 29)]
    _write_run(run, pp=1, cfg=cfg, requests=requests, split="calibration")
    if limit:
        for mode in ("none", "cache"):
            results = json.loads((run / mode / "results.json").read_text())
            results["args"]["limit"] = limit
            (run / mode / "results.json").write_text(json.dumps(results))
    metrics = {"mean_ssim": 0.99, "min_video_ssim": 0.98, "mean_temporal_error_ratio": 0.01, "latency_ratio": 0.8}
    metrics["passed"] = True
    if timing is not None:
        minimal = campaign / "tune-0.1-timing"
        _write_run(minimal, pp=1, cfg=cfg, requests=requests, split="calibration", trace_detail="minimal")
        if timing == "other-sha":
            (minimal / "provenance.json").write_text(
                json.dumps({"sha": "def", "source_hashes": {"a": "1"}, "trace_detail": "minimal"})
            )
        (minimal / "cache/comparison.json").write_text(json.dumps(metrics))
        metrics["timing_source"] = str(minimal / "cache/comparison.json")
    (run / "cache/comparison.json").write_text(json.dumps(metrics))


def _freeze(tmp_path, monkeypatch, inputs, campaign, cfg=1):
    output = tmp_path / "frozen"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "freeze.py",
            "--pp",
            "1",
            "--cfg",
            str(cfg),
            "--inputs",
            str(inputs),
            "--output",
            str(output),
            "--source-checkout",
            str(tmp_path),
            str(campaign),
        ],
    )
    monkeypatch.setattr("subprocess.check_output", lambda *a, **k: "abc\n")
    runpy.run_path(str(SCRIPTS / "freeze.py"), run_name="__main__")
    return json.loads((output / "selection-pp1.json").read_text())


_PROMPTS = [f"prompt {i}" for i in range(24)]


def test_freeze_promotes_full_manifest_with_matched_timing(tmp_path, monkeypatch):
    inputs = _calibration_inputs(tmp_path, _PROMPTS)
    _calibration_candidate(tmp_path / "campaign", _PROMPTS)
    selection = _freeze(tmp_path, monkeypatch, inputs, tmp_path / "campaign")
    assert selection["calibration_all_gates_passed"] and not selection["diagnostic_only"]
    assert selection["receipt"]["calibration_requests"] == 48


@pytest.mark.parametrize(
    "candidate",
    [{"limit": 6}, {"timing": None}, {"timing": "other-sha"}],
    ids=["refinement-subset", "missing-timing-run", "timing-from-other-source"],
)
def test_freeze_keeps_unreceipted_candidate_diagnostic(tmp_path, monkeypatch, candidate):
    inputs = _calibration_inputs(tmp_path, _PROMPTS)
    _calibration_candidate(tmp_path / "campaign", _PROMPTS, **candidate)
    selection = _freeze(tmp_path, monkeypatch, inputs, tmp_path / "campaign")
    assert not selection["calibration_all_gates_passed"] and selection["diagnostic_only"]
