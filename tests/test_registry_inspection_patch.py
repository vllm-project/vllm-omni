# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The model-info child's output file outranks its exit status.

vLLM's registry inspects each architecture in a subprocess that writes its
result to a file and only then exits, so a child that finished the inspection
and crashed on the way out still left a usable answer behind. Upstream calls
``check_returncode()`` first and throws it away, which takes the server down.
Pure CPU: no model, no accelerator, no real subprocess.
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any

import pytest

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

OUTPUT_NAME = "registry_output.tmp"
RESULT = {"architecture": "SomeOmniModel"}


def _fake_child(tmp_path: Path, *, returncode: int, writes: bytes | None, stderr: bytes = b""):
    """Stand in for the inspection child: optionally write a result, then exit."""

    def run(_command: object, input: object = None, capture_output: bool = False, **_kwargs: object):
        del input, capture_output
        if writes is not None:
            (tmp_path / OUTPUT_NAME).write_bytes(writes)
        return subprocess.CompletedProcess(["child"], returncode, b"", stderr)

    return run


@pytest.fixture
def install(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    """Point the registry's tempdir and subprocess at the test, then patch it.

    Returns ``install(child) -> registry``. ``tmp_path`` stands in for the
    ``TemporaryDirectory`` that ``_run_in_subprocess`` creates and destroys, so
    the result file outlives the call and the child can be told where to write.
    """
    from vllm.model_executor.models import registry

    from vllm_omni.patch import _patch_registry_inspection_trusts_its_output

    class _TemporaryDirectory:
        def __enter__(self) -> str:
            return str(tmp_path)

        def __exit__(self, *_exc: object) -> None:
            return None

    class _Tempfile:
        TemporaryDirectory = _TemporaryDirectory

    def _install(child):
        class _Subprocess:
            CompletedProcess = subprocess.CompletedProcess
            run = staticmethod(child)

        monkeypatch.setattr(registry, "_omni_registry_output_trusted", False, raising=False)
        monkeypatch.setattr(registry, "tempfile", _Tempfile, raising=False)
        monkeypatch.setattr(registry, "subprocess", _Subprocess, raising=False)
        _patch_registry_inspection_trusts_its_output()
        return registry

    return _install


def _pickled(registry: Any, value: object) -> bytes:
    """Serialize with the registry's own pickle, the one it will read back."""
    return registry.pickle.dumps(value)


def test_complete_result_survives_a_child_that_aborts_on_exit(install, monkeypatch, tmp_path):
    """The answer comes back, and the crash is reported rather than hidden."""
    from vllm.model_executor.models import registry as unpatched

    import vllm_omni.patch as patch_mod

    warnings: list[str] = []
    monkeypatch.setattr(
        patch_mod._PATCH_LOGGER,
        "warning",
        lambda msg, *args: warnings.append(msg % args if args else msg),
    )

    registry = install(
        _fake_child(
            tmp_path,
            returncode=-6,
            writes=_pickled(unpatched, RESULT),
            stderr=b"corrupted size vs. prev_size\n",
        )
    )

    assert registry._run_in_subprocess(lambda: None) == RESULT

    assert len(warnings) == 1
    assert "exited -6" in warnings[0]
    assert "corrupted size vs. prev_size" in warnings[0]


def test_clean_child_is_unchanged(install, tmp_path):
    from vllm.model_executor.models import registry as unpatched

    registry = install(_fake_child(tmp_path, returncode=0, writes=_pickled(unpatched, RESULT)))

    assert registry._run_in_subprocess(lambda: None) == RESULT


def test_child_that_produced_nothing_still_raises(install, tmp_path):
    registry = install(_fake_child(tmp_path, returncode=1, writes=None, stderr=b"boom: a real failure"))

    with pytest.raises(RuntimeError, match="boom: a real failure"):
        registry._run_in_subprocess(lambda: None)


def test_child_killed_mid_write_still_raises(install, tmp_path):
    """A truncated result is not a result, so the exit status stands."""
    registry = install(
        _fake_child(tmp_path, returncode=-9, writes=b"\x80\x05\x95truncated", stderr=b"boom: killed mid-write")
    )

    with pytest.raises(RuntimeError, match="boom: killed mid-write"):
        registry._run_in_subprocess(lambda: None)


def test_patch_is_idempotent(install, tmp_path):
    from vllm.model_executor.models import registry as unpatched

    from vllm_omni.patch import _patch_registry_inspection_trusts_its_output

    registry = install(_fake_child(tmp_path, returncode=0, writes=_pickled(unpatched, RESULT)))
    once = registry.subprocess

    _patch_registry_inspection_trusts_its_output()

    assert registry.subprocess is once
