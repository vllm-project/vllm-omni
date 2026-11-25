# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Server teardown must let resource owners finish before killing their tracker."""

import os
import subprocess
import sys
import textwrap
import time

import psutil
import pytest

from tests.helpers.runtime import OmniServer

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


@pytest.mark.skipif(os.name != "posix", reason="Requires POSIX signal handlers and shared memory")
def test_shutdown_preserves_tracker_during_slow_cooperative_cleanup(tmp_path):
    # The server needs longer than the old harness's 15s deadline, but less
    # than diffusion worker cleanup's existing 25s escalation budget.
    script = tmp_path / "server.py"
    script.write_text(
        textwrap.dedent(
            """\
            import signal
            import subprocess
            import sys
            import time
            from multiprocessing import shared_memory

            segment = shared_memory.SharedMemory(create=True, size=1)
            child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'])

            def shutdown(signum, frame):
                time.sleep(16)
                segment.close()
                segment.unlink()
                child.terminate()
                child.wait()
                print('CLEAN', flush=True)
                sys.exit(0)

            signal.signal(signal.SIGTERM, shutdown)
            print('READY', flush=True)
            time.sleep(120)
            """
        )
    )
    log_path = tmp_path / "server.log"
    with log_path.open("w") as log:
        proc = subprocess.Popen([sys.executable, str(script)], stdout=log, stderr=subprocess.STDOUT)
        try:
            deadline = time.monotonic() + 10
            while "READY" not in log_path.read_text():
                assert proc.poll() is None, log_path.read_text()
                assert time.monotonic() < deadline, "Subprocess did not initialize shared memory"
                time.sleep(0.02)

            # Avoid constructing a model server; exercise the real process
            # teardown method against a real server/child/tracker process tree.
            server = object.__new__(OmniServer)
            server._kill_process_tree(proc.pid)
            assert proc.poll() == 0, log_path.read_text()
        finally:
            if proc.poll() is None:
                parent = psutil.Process(proc.pid)
                children = parent.children(recursive=True)
                parent.kill()
                for child in children:
                    try:
                        child.kill()
                    except psutil.NoSuchProcess:
                        pass
                proc.wait(timeout=5)

    output = log_path.read_text()
    assert "CLEAN" in output
    assert "resource_tracker" not in output
    assert "KeyError" not in output
