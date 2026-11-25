# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""CPU contracts for READY routing, not ROCm model qualification."""

import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

from tests.buildkite.test_amd_pipeline import (
    _TINY_DIFFUSION_BASE_NODES,
    AMD_READY_PIPELINE,
    _find_step,
    _rendered_ready_step,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

LABEL = "ROCm · Tiny Diffusion Offline Base Expansion"
ARTIFACT = "rocm-tiny-diffusion-offline-ready"
MARKER = "core_model and diffusion and rocm and MI325 and cards_1"
MODELS = (
    "Wan",
    "LongCatImage",
    "LongCatImageEdit",
    "Flux",
    "FluxKontext",
    "Flux2",
    "StableDiffusion3",
    "Krea2",
    "StableDiffusionXL",
)
NODES = tuple(
    f"tests/model_tests/diffusion/test_common_offline.py::test_pipeline_on_supported_tasks[{model}Pipeline]"
    for model in MODELS
)


def _command() -> str:
    return "\n".join(_find_step(LABEL, AMD_READY_PIPELINE)["commands"])


def test_tiny_offline_ready_scope() -> None:
    step = _find_step(LABEL, AMD_READY_PIPELINE)
    command = _command()
    nodes = tuple(re.findall(r'"(tests/[^"\n]+)"', command))
    assert nodes == NODES
    assert len(nodes) == 9
    assert set(nodes).isdisjoint(_TINY_DIFFUSION_BASE_NODES)
    assert "LTX2" not in command
    assert step["grade"] == "Blocking"
    assert step["agent_pool"] == "mi300_1"
    assert step["depends_on"] == "amd-build"
    assert step["env"]["DIFFUSION_ATTENTION_BACKEND"] == "TORCH_SDPA"
    assert step["artifact_paths"] == [f"artifacts/{ARTIFACT}/**/*"]
    assert step["timeout_in_minutes"] == 90
    assert command.count(f'-m "{MARKER}"') == 2
    assert "--run-level core_model --shard-id 0 --num-shards 1" in command
    assert "ROCM_DIFFUSION_RUN_LEVEL=core_model" in command
    assert f"1 9 60m {ARTIFACT}" in command
    assert "pytest -n" not in command
    assert "run-rocm-diffusion-nightly.sh" in command


def test_tiny_offline_ready_renders_blocking() -> None:
    rendered = _rendered_ready_step(LABEL)
    assert rendered["agents"]["queue"] == "amd_mi300_1"
    assert rendered["depends_on"] == "amd-build"
    assert rendered["soft_fail"] is False


@pytest.mark.parametrize(
    ("scenario", "status", "executed"),
    [
        ("pass", 0, True),
        ("subtests", 0, True),
        ("collection-missing", 1, False),
        ("collection-duplicate", 1, False),
        ("collection-wrong-node", 1, False),
        ("collection-error", 2, False),
        ("junit-missing", 5, True),
        ("junit-duplicate", 5, True),
        ("junit-wrong-node", 5, True),
        ("junit-skipped", 5, True),
        ("junit-failure", 5, True),
        ("junit-error", 5, True),
        ("subtest-skip", 5, True),
        ("runner-failure", 7, True),
    ],
)
def test_tiny_offline_ready_fails_closed(tmp_path: Path, scenario: str, status: int, executed: bool) -> None:
    """Run the real job shell with fake collection/runner; never import GPU code."""
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    expected = tmp_path / "nodes.txt"
    expected.write_text("\n".join(NODES), encoding="utf-8")

    def executable(name: str, body: str) -> None:
        path = fake_bin / name
        path.write_text("#!/bin/bash\nset -euo pipefail\n" + body, encoding="utf-8")
        path.chmod(0o755)

    executable("python", 'exec "$REAL_PYTHON" "$@"\n')
    # Telemetry failure is diagnostic only and must not overwrite test status.
    executable("rocm-smi", "exit 1\n")
    executable("pytest", 'exec "$REAL_PYTHON" "$STATE/collect.py" "$@"\n')
    (tmp_path / "collect.py").write_text(
        "import os, sys\n"
        "from pathlib import Path\n"
        "nodes = Path(os.environ['STATE'], 'nodes.txt').read_text().splitlines()\n"
        "assert sys.argv[1:] == ['-q', *nodes, '-m', os.environ['MARKER'], "
        "'--run-level', 'core_model', '--shard-id', '0', '--num-shards', '1', '--collect-only']\n"
        "scenario = os.environ['SCENARIO']\n"
        "if scenario == 'collection-missing': nodes.pop()\n"
        "if scenario == 'collection-duplicate': nodes[-1] = nodes[0]\n"
        "if scenario == 'collection-wrong-node': nodes[-1] = 'tests/unexpected.py::test_wrong'\n"
        "print('\\n'.join(nodes))\n"
        "raise SystemExit(2 if scenario == 'collection-error' else 0)\n",
        encoding="utf-8",
    )
    runner = tmp_path / ".buildkite/amd/scripts/run-rocm-diffusion-nightly.sh"
    runner.parent.mkdir(parents=True)
    runner.write_text('#!/bin/bash\nexec "$REAL_PYTHON" "$STATE/runner.py" "$@"\n', encoding="utf-8")
    (tmp_path / "runner.py").write_text(
        "import os, sys, xml.etree.ElementTree as ET\n"
        "from pathlib import Path\n"
        "state = Path(os.environ['STATE'])\n"
        "nodes = (state / 'nodes.txt').read_text().splitlines()\n"
        "assert sys.argv[1:] == ['1', str(len(nodes)), '60m', os.environ['ARTIFACT'], "
        "*nodes, '-m', os.environ['MARKER']]\n"
        "assert os.environ['ROCM_DIFFUSION_RUN_LEVEL'] == 'core_model'\n"
        "(state / 'executed').touch()\n"
        "scenario = os.environ['SCENARIO']\n"
        "suite = ET.Element('testsuite', tests=str(len(nodes) * (3 if scenario == 'subtests' else 1)), "
        "skipped='1' if scenario == 'subtest-skip' else '0', failures='0', errors='0')\n"
        "if scenario == 'junit-missing': nodes.pop()\n"
        "if scenario == 'junit-duplicate': nodes[-1] = nodes[0]\n"
        "if scenario == 'junit-wrong-node': nodes[-1] = 'tests/unexpected.py::test_wrong'\n"
        "for node in nodes:\n"
        "    module, name = node.split('::', 1)\n"
        "    case = ET.SubElement(suite, 'testcase', classname=module.removesuffix('.py').replace('/', '.'), name=name)\n"
        "    if scenario in {'junit-skipped', 'junit-failure', 'junit-error'}:\n"
        "        ET.SubElement(case, scenario.removeprefix('junit-'))\n"
        "ET.ElementTree(suite).write(state / 'artifacts' / os.environ['ARTIFACT'] / 'pytest.xml')\n"
        "raise SystemExit(7 if scenario == 'runner-failure' else 0)\n",
        encoding="utf-8",
    )
    result = subprocess.run(
        ["bash", "-c", _command().replace("$$", "$")],
        cwd=tmp_path,
        env={
            **os.environ,
            "PATH": f"{fake_bin}{os.pathsep}{os.environ['PATH']}",
            "REAL_PYTHON": sys.executable,
            "BUILDKITE_BUILD_CHECKOUT_PATH": str(tmp_path),
            "STATE": str(tmp_path),
            "SCENARIO": scenario,
            "ARTIFACT": ARTIFACT,
            "MARKER": MARKER,
        },
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == status, result.stdout + result.stderr
    assert (tmp_path / "executed").exists() is executed
    artifacts = tmp_path / "artifacts" / ARTIFACT
    assert (artifacts / "expected-nodes.txt").read_text().splitlines() == sorted(NODES)
    if executed:
        assert "real_seconds" in (artifacts / "runtime.txt").read_text()
        for name in ("host-memory-before", "host-memory-after", "gpu-memory-before", "gpu-memory-after"):
            assert (artifacts / f"{name}.txt").exists()
