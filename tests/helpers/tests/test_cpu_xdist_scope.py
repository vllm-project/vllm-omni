# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from __future__ import annotations

import json
from pathlib import Path

import pytest

pytest_plugins = ("pytester",)
pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _project(pytester: pytest.Pytester, monkeypatch: pytest.MonkeyPatch) -> None:
    source = Path(__file__).parents[1] / "fixtures" / "pytest_collection.py"
    pytester.makepyfile(scope_policy=source.read_text())
    pytester.makeconftest('pytest_plugins = ("scope_policy", "scope_observer")\n')
    pytester.makeini("[pytest]\nmarkers =\n    core_model: core\n    cpu: cpu\n    xdist_group: serial\n")
    pytester.makepyfile(
        scope_observer="""
        import json
        import os
        from pathlib import Path
        import pytest

        @pytest.hookimpl(trylast=True)
        def pytest_configure(config):
            worker = getattr(config, 'workerinput', {})
            if not worker:
                return
            Path(worker['workerid'] + '.json').write_text(json.dumps({
                'scope': worker.get('omni_cpu_xdist_scope'),
                'port': os.environ.get('MASTER_PORT'),
                'dist': config.option.dist,
                'args': config.args,
                'ignore': config.option.ignore,
            }))
        """
    )
    for relative in (
        "entrypoints/test_entrypoint.py",
        "engine/test_engine.py",
        "diffusion/cache/test_cache.py",
        "diffusion/models/test_model.py",
        "model_executor/test_executor.py",
    ):
        target = pytester.path / "tests" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            "import json, os\nfrom pathlib import Path\nimport pytest\n"
            "pytestmark = [pytest.mark.core_model, pytest.mark.cpu]\n"
            "def test_selected(request):\n"
            "    scope = os.environ.get('VLLM_OMNI_CPU_XDIST_SCOPE')\n"
            "    assert request.config.workerinput.get('omni_cpu_xdist_scope') == scope\n"
            "    assert bool(request.node.get_closest_marker('xdist_group')) == (scope is None)\n"
            "    Path('ran').mkdir(exist_ok=True)\n"
            "    Path('ran', Path(__file__).name + '.json').write_text(\n"
            "        json.dumps(request.config.workerinput['workerid']))\n"
        )
    monkeypatch.setenv("PYTEST_DISABLE_PLUGIN_AUTOLOAD", "1")


def _selection() -> list[str]:
    return ["tests/entrypoints", "tests/engine", "tests/diffusion", "--ignore=tests/diffusion/models"]


def test_simple_cpu_scope_passes_authorization_to_real_workers(pytester, monkeypatch):
    scope, expected = "simple-engine", 3
    _project(pytester, monkeypatch)
    monkeypatch.setenv("VLLM_OMNI_CPU_XDIST_SCOPE", scope)
    result = pytester.runpytest_subprocess(
        "-p",
        "xdist",
        "-q",
        "-n",
        "4",
        "--dist=loadfile",
        *_selection(),
        "-m",
        "core_model and cpu",
        timeout=30,
    )
    result.assert_outcomes(passed=expected)
    workers = [json.loads(path.read_text()) for path in pytester.path.glob("gw*.json")]
    assert len(workers) == 4
    assert all(worker["scope"] == scope and worker["dist"] == "no" for worker in workers)
    assert len({worker["port"] for worker in workers}) == 4
    assert len(list((pytester.path / "ran").iterdir())) == expected


def test_simple_cpu_scope_rejects_a_changed_selection_before_starting_workers(pytester, monkeypatch):
    scope = "simple-engine"
    _project(pytester, monkeypatch)
    monkeypatch.setenv("VLLM_OMNI_CPU_XDIST_SCOPE", scope)
    result = pytester.runpytest_subprocess(
        "-p",
        "xdist",
        "-q",
        "-n",
        "4",
        "--dist=loadfile",
        *_selection(),
        "-m",
        "core_model and cpu",
        "--deselect=tests/engine/test_engine.py::test_selected",
        timeout=30,
    )
    assert result.ret == pytest.ExitCode.USAGE_ERROR
    result.stderr.fnmatch_lines(["*CPU xdist selection does not match*"])
    assert not list(pytester.path.glob("gw*.json"))
    assert not (pytester.path / "ran").exists()


def test_simple_cpu_scope_rejects_the_nightly_scope(pytester, monkeypatch):
    _project(pytester, monkeypatch)
    monkeypatch.setenv("VLLM_OMNI_CPU_XDIST_SCOPE", "nightly-coverage")
    result = pytester.runpytest_subprocess(
        "-p",
        "xdist",
        "-q",
        "-n",
        "4",
        "--dist=loadfile",
        *_selection(),
        "-m",
        "core_model and cpu",
        timeout=30,
    )
    assert result.ret == pytest.ExitCode.USAGE_ERROR
    result.stderr.fnmatch_lines(["*Unknown VLLM_OMNI_CPU_XDIST_SCOPE*"])
    assert not list(pytester.path.glob("gw*.json"))
    assert not (pytester.path / "ran").exists()
