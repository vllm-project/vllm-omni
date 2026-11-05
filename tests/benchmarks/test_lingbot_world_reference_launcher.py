# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Exercise the actual CI launcher without provisioning accelerator workloads."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]
ROOT = Path(__file__).resolve().parents[2]
LAUNCHER = ROOT / "tests/dfx/lingbot_world/run_ci.sh"


@pytest.fixture
def launcher(tmp_path, lingbot_subprocess_env):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    python = bin_dir / "python"
    python.write_text(
        f"#!{sys.executable}\n"
        "import json,os,sys\nfrom pathlib import Path\n"
        "if sys.argv[1:3] == ['-m','pytest']:\n"
        "    command=str(Path(__file__).with_name('pytest'))\n"
        "    os.execv(command,[command,*sys.argv[3:]])\n"
        "elif sys.argv[1:3] == ['-m','tests.dfx.lingbot_world.prepare']:\n"
        "    stage=sys.argv[3]\n"
        "    if stage=='devices':\n"
        "        root=Path(sys.argv[sys.argv.index('--root')+1])\n"
        "        (root/'allocation.json').write_text(json.dumps({'visible_devices':'0,1,2,3'}))\n"
        "        print('0,1,2,3')\n"
        "    elif stage=='reference':\n"
        "        Path(sys.argv[sys.argv.index('--env-output')+1]).write_text(json.dumps({'model_path':'unused'}))\n"
        "    elif stage!='evaluators': raise AssertionError(stage)\n"
        "elif sys.argv[1:2]==['-c'] and 'evaluator_cache_key' in sys.argv[2]:\n"
        "    print('contract-cache')\n"
        "else: os.execv(sys.executable,[sys.executable,*sys.argv[1:]])\n"
    )
    pytest_command = bin_dir / "pytest"
    pytest_command.write_text(
        f"#!{sys.executable}\n"
        "import json,os,sys\nfrom pathlib import Path\n"
        "Path(os.environ['CONTRACT_SELECTION']).write_text(json.dumps({'arguments':sys.argv[1:], 'phase':os.environ.get('LINGBOT_CI_PHASE')}))\n"
        "driver=os.environ.get('CONTRACT_PYTEST_DRIVER')\n"
        "if driver:\n"
        "    assert sys.argv[2]=='tests/dfx/lingbot_world/test_quality.py'\n"
        "    os.execv(sys.executable,[sys.executable,'-m','pytest','-q','--noconftest',driver,'--rootdir',str(Path(driver).parent),'--junitxml='+os.environ['CONTRACT_OUTPUT']+'/pytest.xml'])\n"
        "sys.exit(17)\n"
    )
    for executable in (python, pytest_command):
        executable.chmod(0o755)
    profile = tmp_path / "profile.json"
    profile.write_text(json.dumps({"platform": "cuda", "dies": 4}))
    generation = tmp_path / "generation"
    generation.mkdir()
    (generation / "generation.json").write_text(
        json.dumps({"passed": False, "reference_only": True, "measurement_only": True, "complete": True})
    )
    output = tmp_path / "output"
    selection = tmp_path / "selection.json"
    environment = {
        **lingbot_subprocess_env,
        "PATH": str(bin_dir) + os.pathsep + os.environ["PATH"],
        "PYTHONPATH": str(ROOT),
    }
    # Keep the unset-switch cases distinct from explicit acceptance mode (0).
    environment.pop("LINGBOT_REFERENCE_ONLY")
    environment.update(
        LINGBOT_EVALUATOR_ROOT=str(tmp_path / "evaluators"),
        CONTRACT_SELECTION=str(selection),
        CONTRACT_OUTPUT=str(output),
        CONTRACT_GENERATION=str(generation),
        PYTEST_DISABLE_PLUGIN_AUTOLOAD="1",
    )

    def run(stage, opt_in=None, *, driver=None):
        if stage in ("collect", "aggregate", "reference-collect", "reference-aggregate"):
            profile.write_text(json.dumps({"platform": "npu", "dies": 4}))
        process_environment = dict(environment)
        if stage in ("collect", "reference-collect"):
            process_environment["LINGBOT_MEASUREMENT_REPEAT"] = "1"
        if opt_in is not None:
            process_environment["LINGBOT_REFERENCE_ONLY"] = opt_in
        if driver is not None:
            process_environment["CONTRACT_PYTEST_DRIVER"] = str(driver)
        return subprocess.run(
            ["bash", str(LAUNCHER), stage, str(profile), str(output), str(generation)],
            cwd=tmp_path,
            env=process_environment,
            capture_output=True,
            text=True,
            timeout=60,
        )

    return run, output, selection, tmp_path


@pytest.mark.parametrize(
    "stage,opt_in,effective,test_file,phase",
    [
        ("generate", None, "generate", "test_generate.py", "acceptance"),
        ("generate", "0", "generate", "test_generate.py", "acceptance"),
        ("evaluate", None, "evaluate", "test_quality.py", "acceptance"),
        ("evaluate", "0", "evaluate", "test_quality.py", "acceptance"),
        ("collect", None, "collect", "test_collect.py", "acceptance"),
        ("aggregate", None, "aggregate", "test_aggregate.py", "acceptance"),
        ("smoke", "1", "smoke", "test_smoke.py", "acceptance"),
        (
            "collect",
            "1",
            "reference-collect",
            "test_reference.py::test_collect_npu_reference_measurement",
            "reference-collection",
        ),
        (
            "aggregate",
            "1",
            "reference-aggregate",
            "test_reference.py::test_aggregate_reference_measurements",
            "reference-collection",
        ),
        ("generate", "1", "reference", "test_reference.py::test_collect_cuda_reference_cohort", "reference-collection"),
        (
            "evaluate",
            "1",
            "reference-quality",
            "test_reference.py::test_collect_reference_quality",
            "reference-collection",
        ),
    ],
)
def test_existing_ci_commands_select_the_explicit_collection_or_acceptance_phase(
    launcher, stage, opt_in, effective, test_file, phase
):
    run, output, selection, _ = launcher
    result = run(stage, opt_in)
    assert result.returncode == 17, result.stdout + result.stderr
    selected = json.loads(selection.read_text())
    assert selected["arguments"][1] == "tests/dfx/lingbot_world/" + test_file
    assert selected["phase"] == phase
    if stage == "collect":
        assert "--lingbot-measurement-repeat=1" in selected["arguments"]
    configuration = json.loads((output / "ci-configuration.json").read_text())
    assert configuration["requested_stage"] == stage
    assert configuration["effective_stage"] == effective
    assert configuration["phase"] == phase
    assert configuration["reference_only"] is (phase == "reference-collection")
    assert configuration["reference_only_opt_in"] is (opt_in == "1")
    if phase == "reference-collection":
        assert "do not establish candidate acceptance" in result.stdout
    assert ET.parse(output / "pytest.xml").find("testcase/failure") is not None


@pytest.mark.parametrize("opt_in", ["", "true", "yes", "01", "2"])
def test_invalid_reference_switch_cannot_start_provisioning(launcher, opt_in):
    run, output, selection, _ = launcher
    result = run("generate", opt_in)
    assert result.returncode == 2
    assert "LINGBOT_REFERENCE_ONLY must be unset, 0, or 1" in result.stderr
    assert not selection.exists()
    assert ET.parse(output / "pytest.xml").find("testcase/failure") is not None


@pytest.mark.parametrize("stage", ["unknown", "reference-ish"])
def test_reference_phase_rejects_unknown_stages(launcher, stage):
    run, output, selection, _ = launcher
    result = run(stage, "1")
    assert result.returncode == 2
    assert "Unknown LingBot CI stage in reference collection mode" in result.stderr
    assert not selection.exists()
    assert ET.parse(output / "pytest.xml").find("testcase/failure") is not None


def test_default_acceptance_launcher_rejects_reference_only_generation_artifacts(launcher):
    run, output, selection, temporary = launcher
    driver = temporary / "test_acceptance_contract.py"
    driver.write_text(
        "import os\nfrom pathlib import Path\n"
        "from tests.dfx.lingbot_world.test_quality import evaluate_quality\n\n"
        "def test_candidate_acceptance(monkeypatch):\n"
        "    evaluate_quality({'output':Path(os.environ['CONTRACT_OUTPUT']), 'generation':Path(os.environ['CONTRACT_GENERATION'])},monkeypatch)\n"
    )
    result = run("evaluate", driver=driver)
    assert result.returncode == 1, result.stdout + result.stderr
    assert "Reference-only collection cannot establish candidate acceptance" in result.stdout
    selected = json.loads(selection.read_text())
    assert selected["arguments"][1] == "tests/dfx/lingbot_world/test_quality.py"
    assert selected["phase"] == "acceptance"
    assert not json.loads((output / "quality.json").read_text())["passed"]
    assert list(ET.parse(output / "pytest.xml").iter("failure"))
