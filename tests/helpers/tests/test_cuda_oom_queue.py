# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import os
import subprocess
import sys
from pathlib import Path
from textwrap import dedent

import pytest

pytestmark = [pytest.mark.cpu, pytest.mark.core_model]


def run_queue(tmp_path, source):
    (tmp_path / "test_cases.py").write_text(dedent(source))
    (tmp_path / "pytest.ini").write_text("[pytest]\n")
    env = os.environ.copy()
    env.update(
        PYTHONPATH=os.pathsep.join([str(Path(__file__).resolve().parents[3]), env.get("PYTHONPATH", "")]),
        PYTEST_DISABLE_PLUGIN_AUTOLOAD="1",
        PYTEST_ADDOPTS="",
    )
    return subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-q",
            "-p",
            "xdist.plugin",
            "-p",
            "tests.helpers.fixtures.cuda_oom_queue",
            "-c",
            str(tmp_path / "pytest.ini"),
            "--dist=loadgroup",
            "-n",
            "2",
            str(tmp_path / "test_cases.py"),
        ],
        cwd=tmp_path,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        timeout=45,
    )


def test_retry_runs_after_tail_and_peer_teardown(tmp_path):
    result = run_queue(
        tmp_path,
        """
        import time
        from pathlib import Path

        def test_oom():
            if not Path("first").exists():
                Path("first").touch()
                raise RuntimeError("Dummy run failed: CUDA out of memory")
            assert Path("tail").exists() and Path("peer_done").exists()

        def test_peer(request):
            request.addfinalizer(lambda: Path("peer_done").touch())
            deadline = time.monotonic() + 10
            while not Path("tail").exists():
                assert time.monotonic() < deadline, "the other worker must keep taking work"
                time.sleep(0.01)
            time.sleep(0.2)

        def test_tail():
            assert Path("first").exists()
            Path("tail").touch()
    """,
    )
    assert result.returncode == 0, result.stdout
    assert "3 passed" in result.stdout and "1 rerun" in result.stdout


def test_permanent_oom_and_unrelated_failure(tmp_path):
    result = run_queue(
        tmp_path,
        """
        from pathlib import Path

        def test_permanent_oom(subtests):
            with Path("attempts").open("a") as output:
                output.write("x")
            with subtests.test(msg="inference"):
                raise RuntimeError("CUDA out of memory")

        def test_other_failure():
            with Path("ordinary").open("a") as output:
                output.write("x")
            raise AssertionError("CUDA out of memory is only mentioned in the assertion")
    """,
    )
    assert result.returncode == 1, result.stdout
    assert "1 rerun" in result.stdout and "INTERNALERROR" not in result.stdout
    assert (tmp_path / "attempts").read_text() == "xx"
    assert (tmp_path / "ordinary").read_text() == "x"


def test_worker_crash_reports_failure_and_runs_remaining_group_members(tmp_path):
    result = run_queue(
        tmp_path,
        """
        import os
        from pathlib import Path
        import pytest

        pytestmark = pytest.mark.xdist_group("same-worker")

        def test_crashes():
            with Path("attempts").open("a") as stream:
                stream.write("x")
            os._exit(1)

        def test_after_crash():
            Path("finished").touch()
    """,
    )
    assert result.returncode == 1, result.stdout
    assert "1 failed, 1 passed" in result.stdout
    assert (tmp_path / "attempts").read_text() == "x"
    assert (tmp_path / "finished").exists()
