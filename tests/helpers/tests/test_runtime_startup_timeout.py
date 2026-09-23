# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Startup deadlines must be independent of system clock corrections."""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from tests.helpers import runtime

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


@pytest.fixture(params=["server", "stage_cli"])
def startup_server(request, tmp_path, monkeypatch):
    monkeypatch.setattr(runtime, "cleanup_test_environment", lambda: None)
    proc = MagicMock(spec=runtime.subprocess.Popen)
    proc.poll.return_value = None
    monkeypatch.setattr(runtime.subprocess, "Popen", lambda *args, **kwargs: proc)
    if request.param == "server":
        # Exercise the wait-loop clock without reserving or inspecting real ports.
        monkeypatch.setattr(runtime.OmniServer, "_reserve_port", lambda self: None)
        monkeypatch.setattr(runtime.OmniServer, "_owns_listening_port", lambda self: True)
        return runtime.OmniServer("fake-model", [])

    config = tmp_path / "stages.yaml"
    config.write_text("stages:\n  - stage_id: 0\n  - stage_id: 1\n")
    server = runtime.OmniServerStageCli("fake-model", str(config), [])

    def launch(stage_id, *, headless, replica_id=0):
        server.stage_procs[(stage_id, replica_id)] = proc

    monkeypatch.setattr(server, "_launch_stage", launch)
    return server


@pytest.mark.parametrize("wall_clock_jump", [3600, -3600])
@pytest.mark.parametrize("becomes_ready", [True, False], ids=["ready", "timeout"])
def test_startup_deadline_ignores_wall_clock_changes(startup_server, monkeypatch, wall_clock_jump, becomes_ready):
    elapsed = 0.0
    wall_offset = 0.0
    first_probe_at = None
    probes = 0

    def sleep(seconds):
        nonlocal elapsed
        elapsed += seconds

    # Replace only this module's clock, leaving pytest's own timing untouched.
    monkeypatch.setattr(
        runtime,
        "time",
        SimpleNamespace(
            time=lambda: 1_700_000_000 + elapsed + wall_offset,
            monotonic=lambda: elapsed,
            perf_counter=lambda: elapsed,
            sleep=sleep,
        ),
    )

    def connect(address):
        nonlocal probes, wall_offset, first_probe_at
        assert address == (startup_server.host, startup_server.port)
        if first_probe_at is None:
            first_probe_at = elapsed
        # Even a backwards clock correction must not extend the deadline.
        assert elapsed - first_probe_at < runtime.SERVER_STARTUP_TIMEOUT_S, "startup exceeded its elapsed-time budget"
        probes += 1
        if probes == 1:
            wall_offset += wall_clock_jump
            return 111
        return 0 if becomes_ready else 111

    sock = MagicMock()
    sock.__enter__.return_value = sock
    sock.connect_ex.side_effect = connect
    monkeypatch.setattr(runtime.socket, "socket", lambda *args, **kwargs: sock)

    if becomes_ready:
        startup_server._start_server()
        assert probes == 2
        assert first_probe_at is not None
        assert elapsed - first_probe_at == 2
    else:
        with pytest.raises(RuntimeError, match=f"failed to start within {runtime.SERVER_STARTUP_TIMEOUT_S} seconds"):
            startup_server._start_server()
        assert first_probe_at is not None
        assert elapsed - first_probe_at == runtime.SERVER_STARTUP_TIMEOUT_S
