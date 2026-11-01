# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""The server/runner fixtures free the Whisper judge around each instance.

iter_omni_server and iter_omni_runner both wrap their body in the same
_whisper_device_free_around() context manager, so one representative path
exercises its enter/exit and exception behavior.
"""

import threading
from types import SimpleNamespace

import pytest

from tests.helpers import runtime
from tests.helpers.runtime import OmniServerParams

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


@pytest.fixture
def events(monkeypatch) -> list[str]:
    """Record construction and release in order."""
    calls: list[str] = []

    class FakeServer:
        def __init__(self, *args, **kwargs):
            calls.append("construct")

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    monkeypatch.setattr(runtime, "OmniServer", FakeServer)
    monkeypatch.setattr(runtime, "release_audio_transcriber", lambda: calls.append("release"))
    monkeypatch.setattr(
        "tests.helpers.stage_config.stage_config_path_for_run_level",
        lambda path, run_level: None,
    )
    return calls


def _server_generator():
    request = SimpleNamespace(
        param=OmniServerParams(model="fake-model"),
        node=SimpleNamespace(get_closest_marker=lambda name: None),
    )
    return runtime.iter_omni_server(request, "core_model", threading.Lock())


def test_releases_before_construction_and_after_teardown(events):
    generator = _server_generator()
    generator.send(None)

    assert events == ["release", "construct"], "the device must be clear before the model loads"

    with pytest.raises(StopIteration):
        generator.send(None)

    assert events == ["release", "construct", "release"]


def test_failing_test_still_releases(events):
    """A test that raises must not strand Whisper on the device."""
    generator = _server_generator()
    generator.send(None)

    with pytest.raises(RuntimeError, match="assertion blew up"):
        generator.throw(RuntimeError("assertion blew up"))

    assert events[-1] == "release"
