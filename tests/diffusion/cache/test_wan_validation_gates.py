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
