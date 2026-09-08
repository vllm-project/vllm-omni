# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from __future__ import annotations

import sys
from collections.abc import Callable
from pathlib import Path
from threading import Event, Thread
from types import ModuleType
from typing import Any

import pytest
from diffusers import utils as diffusers_utils

from vllm_omni.model_executor.models.dynin_omni import dynin_omni_common

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

_FLAX_WEIGHTS_NAME = "FLAX_WEIGHTS_NAME"
_FLAX_WEIGHTS_FILENAME = "diffusion_flax_model.msgpack"
_GATE_MODULE = "_dynin_magvit_test_gate"
_MAGVIT_SOURCE = (
    "from diffusers.utils import FLAX_WEIGHTS_NAME\nclass MAGVITv2:\n    flax_weights_name = FLAX_WEIGHTS_NAME\n"
)
# Blocks inside module execution until the test releases it, so tests can observe
# what other callers do while a MAGVIT import is in flight.
_GATED_MAGVIT_SOURCE = (
    f"import {_GATE_MODULE} as gate\n"
    "from diffusers.utils import FLAX_WEIGHTS_NAME\n"
    "gate.entered.set()\n"
    "if not gate.release.wait(timeout=10):\n"
    "    raise RuntimeError('test gate was never released')\n"
    "class MAGVITv2:\n"
    "    flax_weights_name = FLAX_WEIGHTS_NAME\n"
)


class _FakeSnapshots:
    """Fake MAGVIT remote-code snapshots served for directory sources and registered repo ids."""

    def __init__(self, root: Path) -> None:
        self._root = root
        self.by_repo_id: dict[str, str] = {}

    def create(self, name: str, source: str, *, repo_id: str | None = None) -> Path:
        snapshot = self._root / name
        snapshot.mkdir()
        (snapshot / "modeling_magvitv2.py").write_text(source)
        if repo_id is not None:
            self.by_repo_id[repo_id] = str(snapshot.resolve())
        return snapshot

    def resolve(self, *, source: str, **kwargs) -> str:
        del kwargs
        if Path(source).is_dir():
            return str(Path(source).resolve())
        if source in self.by_repo_id:
            return self.by_repo_id[source]
        raise FileNotFoundError(f"{source} is not cached (local_files_only)")


class _Task:
    """Run ``fn`` on a daemon thread; a deadlocked call fails on ``result`` instead of hanging pytest."""

    def __init__(self, fn: Callable[..., Any], *args: Any) -> None:
        self._done = Event()
        self._value: Any = None
        self._error: BaseException | None = None

        def run() -> None:
            try:
                self._value = fn(*args)
            except BaseException as e:  # noqa: BLE001 - re-raised from result()
                self._error = e
            finally:
                self._done.set()

        Thread(target=run, daemon=True).start()

    def finished(self, timeout: float) -> bool:
        return self._done.wait(timeout)

    def result(self, timeout: float) -> Any:
        assert self._done.wait(timeout), "call did not finish in time"
        if self._error is not None:
            raise self._error
        return self._value


@pytest.fixture
def remote_snapshots(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Isolate remote-code registries per test and drop the modules a test registered."""
    settings = dynin_omni_common.MAGVIT_REMOTE_SETTINGS
    for env_name in (settings.repo_env, settings.revision_env, settings.local_only_env):
        monkeypatch.delenv(env_name, raising=False)
    packages: dict[str, str] = {}
    monkeypatch.setattr(dynin_omni_common, "_DYNIN_REMOTE_PACKAGE_BY_SNAPSHOT", packages)
    monkeypatch.setattr(dynin_omni_common, "_DYNIN_REMOTE_ATTR_CACHE", {})
    yield _FakeSnapshots(tmp_path)
    for package in packages.values():
        for module_name in list(sys.modules):
            if module_name == package or module_name.startswith(f"{package}."):
                sys.modules.pop(module_name, None)


@pytest.fixture
def remote_module(remote_snapshots: _FakeSnapshots, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(dynin_omni_common, "_resolve_remote_snapshot_dir", remote_snapshots.resolve)
    return remote_snapshots.create


@pytest.fixture
def import_gate(monkeypatch: pytest.MonkeyPatch):
    gate = ModuleType(_GATE_MODULE)
    gate.entered = Event()  # type: ignore[attr-defined]
    gate.release = Event()  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, _GATE_MODULE, gate)
    yield gate
    gate.release.set()


def _get_magvit(source: str, **kwargs):
    return dynin_omni_common.get_dynin_magvit_attr("MAGVITv2", source=source, local_files_only=True, **kwargs)


def test_magvit_remote_import_supplies_removed_diffusers_export(
    remote_module,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delattr(diffusers_utils, _FLAX_WEIGHTS_NAME, raising=False)
    source = remote_module("success", _MAGVIT_SOURCE)

    model_class = _get_magvit(str(source))

    assert model_class.flax_weights_name == _FLAX_WEIGHTS_FILENAME
    assert _FLAX_WEIGHTS_NAME not in vars(diffusers_utils)
    assert _get_magvit(str(source)) is model_class
    assert _FLAX_WEIGHTS_NAME not in vars(diffusers_utils)


def test_magvit_remote_import_preserves_existing_diffusers_export(
    remote_module,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    existing = object()
    monkeypatch.setattr(diffusers_utils, _FLAX_WEIGHTS_NAME, existing, raising=False)
    source = remote_module("existing", _MAGVIT_SOURCE)

    model_class = _get_magvit(str(source))

    assert model_class.flax_weights_name is existing
    assert getattr(diffusers_utils, _FLAX_WEIGHTS_NAME) is existing


@pytest.mark.parametrize("export_exists", [False, True])
def test_magvit_remote_import_restores_diffusers_after_failure(
    remote_module,
    monkeypatch: pytest.MonkeyPatch,
    export_exists: bool,
) -> None:
    existing = object()
    if export_exists:
        monkeypatch.setattr(diffusers_utils, _FLAX_WEIGHTS_NAME, existing, raising=False)
    else:
        monkeypatch.delattr(diffusers_utils, _FLAX_WEIGHTS_NAME, raising=False)
    source = remote_module(
        f"failure-{export_exists}",
        "from diffusers.utils import FLAX_WEIGHTS_NAME\nraise RuntimeError('remote import failed')\n",
    )

    with pytest.raises(ImportError, match="Failed to resolve 'MAGVITv2'"):
        _get_magvit(str(source))

    if export_exists:
        assert getattr(diffusers_utils, _FLAX_WEIGHTS_NAME) is existing
    else:
        assert _FLAX_WEIGHTS_NAME not in vars(diffusers_utils)
    package = dynin_omni_common._DYNIN_REMOTE_PACKAGE_BY_SNAPSHOT[str(source.resolve())]
    assert f"{package}.modeling_magvitv2" not in sys.modules


def test_magvit_remote_import_falls_back_to_default_repo_for_weights_only_source(
    remote_module,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Production resolves the VQ weights repo (e.g. showlab/magvitv2), which ships
    # no remote code, so the accessor must fetch the code from the default repo.
    monkeypatch.delattr(diffusers_utils, _FLAX_WEIGHTS_NAME, raising=False)
    weights_only = tmp_path / "weights-only"
    weights_only.mkdir()
    (weights_only / "config.json").write_text("{}")
    remote_module("default-repo", _MAGVIT_SOURCE, repo_id=dynin_omni_common.DEFAULT_MAGVIT_REMOTE_CODE_REPO)

    model_class = _get_magvit(str(weights_only))

    assert model_class.flax_weights_name == _FLAX_WEIGHTS_FILENAME
    assert _FLAX_WEIGHTS_NAME not in vars(diffusers_utils)


def test_magvit_remote_import_downloads_repo_id_snapshot_at_revision(
    remote_snapshots: _FakeSnapshots,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delattr(diffusers_utils, _FLAX_WEIGHTS_NAME, raising=False)
    repo_id = dynin_omni_common.DEFAULT_MAGVIT_REMOTE_CODE_REPO
    revision = "5a869517cbbda6ac4de8b438fc59b7f053cb4238"
    snapshot = remote_snapshots.create("hub", _MAGVIT_SOURCE)
    download_calls: list[dict] = []

    def snapshot_download(**kwargs):
        download_calls.append(kwargs)
        return str(snapshot.resolve())

    monkeypatch.setattr(dynin_omni_common, "snapshot_download", snapshot_download)

    model_class = _get_magvit(repo_id, revision=revision)

    assert model_class.flax_weights_name == _FLAX_WEIGHTS_FILENAME
    assert len(download_calls) == 1
    assert download_calls[0]["repo_id"] == repo_id
    assert download_calls[0]["revision"] == revision
    assert download_calls[0]["local_files_only"] is True
    assert "*.py" in download_calls[0]["allow_patterns"]
    assert _FLAX_WEIGHTS_NAME not in vars(diffusers_utils)


def test_concurrent_magvit_imports_serialize_module_execution_only(
    remote_module,
    import_gate,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delattr(diffusers_utils, _FLAX_WEIGHTS_NAME, raising=False)
    gated = str(remote_module("gated", _GATED_MAGVIT_SOURCE))
    plain = str(remote_module("plain", _MAGVIT_SOURCE))
    plain_resolved = Event()
    resolve_snapshot_dir = dynin_omni_common._resolve_remote_snapshot_dir

    def resolve_and_flag(*, source: str, **kwargs) -> str:
        if source == plain:
            plain_resolved.set()
        return resolve_snapshot_dir(source=source, **kwargs)

    monkeypatch.setattr(dynin_omni_common, "_resolve_remote_snapshot_dir", resolve_and_flag)

    first = _Task(_get_magvit, gated)
    assert import_gate.entered.wait(timeout=5)
    # The shim is active exactly while the remote module executes.
    assert _FLAX_WEIGHTS_NAME in vars(diffusers_utils)

    second = _Task(_get_magvit, plain)
    # Snapshot resolution is not serialized behind the in-flight import...
    assert plain_resolved.wait(timeout=5)
    # ...but module execution is.
    assert not second.finished(timeout=0.5)

    import_gate.release.set()
    assert first.result(timeout=5).flax_weights_name == _FLAX_WEIGHTS_FILENAME
    assert second.result(timeout=5).flax_weights_name == _FLAX_WEIGHTS_FILENAME
    assert _FLAX_WEIGHTS_NAME not in vars(diffusers_utils)


def test_concurrent_lookups_of_same_module_share_one_execution(
    remote_module,
    import_gate,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delattr(diffusers_utils, _FLAX_WEIGHTS_NAME, raising=False)
    gated = str(remote_module("gated", _GATED_MAGVIT_SOURCE))

    first = _Task(_get_magvit, gated)
    assert import_gate.entered.wait(timeout=5)
    # A second caller for the same module must wait for the execution in flight
    # rather than pick up the half-executed entry from ``sys.modules``.
    second = _Task(_get_magvit, gated)
    assert not second.finished(timeout=0.5)

    import_gate.release.set()
    first_class = first.result(timeout=5)
    assert second.result(timeout=5) is first_class
    assert first_class.flax_weights_name == _FLAX_WEIGHTS_FILENAME
    assert _FLAX_WEIGHTS_NAME not in vars(diffusers_utils)


def test_cached_magvit_lookup_is_not_blocked_by_in_flight_import(
    remote_module,
    import_gate,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delattr(diffusers_utils, _FLAX_WEIGHTS_NAME, raising=False)
    cached = str(remote_module("cached", _MAGVIT_SOURCE))
    gated = str(remote_module("gated", _GATED_MAGVIT_SOURCE))
    model_class = _get_magvit(cached)

    pending = _Task(_get_magvit, gated)
    assert import_gate.entered.wait(timeout=5)

    cache_hit = _Task(_get_magvit, cached)
    assert cache_hit.result(timeout=2) is model_class
    assert not pending.finished(timeout=0)

    import_gate.release.set()
    assert pending.result(timeout=5).flax_weights_name == _FLAX_WEIGHTS_FILENAME
    assert _FLAX_WEIGHTS_NAME not in vars(diffusers_utils)
