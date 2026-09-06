# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from __future__ import annotations

import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Event
from types import ModuleType

import pytest
from diffusers import utils as diffusers_utils

from vllm_omni.model_executor.models.dynin_omni import dynin_omni_common

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

_FLAX_WEIGHTS_NAME = "FLAX_WEIGHTS_NAME"


@pytest.fixture
def remote_module(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    snapshots: list[str] = []

    def create(name: str, source: str) -> Path:
        snapshot = tmp_path / name
        snapshot.mkdir()
        (snapshot / "modeling_magvitv2.py").write_text(source)
        snapshots.append(str(snapshot.resolve()))
        return snapshot

    def resolve_snapshot_dir(*, source: str, **kwargs) -> str:
        del kwargs
        return str(Path(source).resolve())

    monkeypatch.setattr(dynin_omni_common, "_resolve_remote_snapshot_dir", resolve_snapshot_dir)
    yield create

    for snapshot in snapshots:
        package = dynin_omni_common._DYNIN_REMOTE_PACKAGE_BY_SNAPSHOT.pop(snapshot, None)
        if package is not None:
            for module_name in list(sys.modules):
                if module_name == package or module_name.startswith(f"{package}."):
                    sys.modules.pop(module_name, None)
        for cache_key in list(dynin_omni_common._DYNIN_REMOTE_ATTR_CACHE):
            if cache_key[2] == snapshot:
                dynin_omni_common._DYNIN_REMOTE_ATTR_CACHE.pop(cache_key, None)


def test_magvit_remote_import_supplies_removed_diffusers_export(
    remote_module,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delattr(diffusers_utils, _FLAX_WEIGHTS_NAME, raising=False)
    source = remote_module(
        "success",
        "from diffusers.utils import FLAX_WEIGHTS_NAME\nclass MAGVITv2:\n    flax_weights_name = FLAX_WEIGHTS_NAME\n",
    )

    model_class = dynin_omni_common.get_dynin_magvit_attr(
        "MAGVITv2",
        source=str(source),
        local_files_only=True,
    )

    assert model_class.flax_weights_name == "diffusion_flax_model.msgpack"
    assert _FLAX_WEIGHTS_NAME not in vars(diffusers_utils)
    assert (
        dynin_omni_common.get_dynin_magvit_attr(
            "MAGVITv2",
            source=str(source),
            local_files_only=True,
        )
        is model_class
    )
    assert _FLAX_WEIGHTS_NAME not in vars(diffusers_utils)


def test_magvit_remote_import_preserves_existing_diffusers_export(
    remote_module,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    existing = object()
    monkeypatch.setattr(diffusers_utils, _FLAX_WEIGHTS_NAME, existing, raising=False)
    source = remote_module(
        "existing",
        "from diffusers.utils import FLAX_WEIGHTS_NAME\nclass MAGVITv2:\n    flax_weights_name = FLAX_WEIGHTS_NAME\n",
    )

    model_class = dynin_omni_common.get_dynin_magvit_attr(
        "MAGVITv2",
        source=str(source),
        local_files_only=True,
    )

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
        dynin_omni_common.get_dynin_magvit_attr(
            "MAGVITv2",
            source=str(source),
            local_files_only=True,
        )

    if export_exists:
        assert getattr(diffusers_utils, _FLAX_WEIGHTS_NAME) is existing
    else:
        assert _FLAX_WEIGHTS_NAME not in vars(diffusers_utils)
    package = dynin_omni_common._DYNIN_REMOTE_PACKAGE_BY_SNAPSHOT[str(source.resolve())]
    assert f"{package}.modeling_magvitv2" not in sys.modules


def test_concurrent_magvit_imports_keep_compatibility_window_serialized(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delattr(diffusers_utils, _FLAX_WEIGHTS_NAME, raising=False)
    first_path = tmp_path / "first"
    second_path = tmp_path / "second"
    first_path.mkdir()
    second_path.mkdir()
    first_source = str(first_path.resolve())
    second_source = str(second_path.resolve())
    first_entered = Event()
    release_first = Event()
    second_started = Event()
    second_entered = Event()
    model_classes = {first_source: type("FirstMAGVITv2", (), {}), second_source: type("SecondMAGVITv2", (), {})}

    def load_remote_module(*, source: str, **kwargs) -> ModuleType:
        del kwargs
        assert _FLAX_WEIGHTS_NAME in vars(diffusers_utils)
        if source == first_source:
            first_entered.set()
            assert release_first.wait(timeout=5)
        else:
            second_entered.set()
        module = ModuleType(f"remote_{Path(source).name}")
        setattr(module, "MAGVITv2", model_classes[source])
        return module

    def load_second():
        second_started.set()
        return dynin_omni_common.get_dynin_magvit_attr(
            "MAGVITv2",
            source=second_source,
            local_files_only=True,
        )

    monkeypatch.setattr(dynin_omni_common, "_load_remote_module", load_remote_module)
    try:
        with ThreadPoolExecutor(max_workers=2) as executor:
            first = executor.submit(
                dynin_omni_common.get_dynin_magvit_attr,
                "MAGVITv2",
                source=first_source,
                local_files_only=True,
            )
            assert first_entered.wait(timeout=5)
            second = executor.submit(load_second)
            assert second_started.wait(timeout=5)
            assert not second_entered.wait(timeout=0.5)
            release_first.set()
            assert first.result(timeout=5) is model_classes[first_source]
            assert second.result(timeout=5) is model_classes[second_source]
    finally:
        release_first.set()
        for cache_key in list(dynin_omni_common._DYNIN_REMOTE_ATTR_CACHE):
            if cache_key[2] in {first_source, second_source}:
                dynin_omni_common._DYNIN_REMOTE_ATTR_CACHE.pop(cache_key, None)

    assert second_entered.is_set()
    assert _FLAX_WEIGHTS_NAME not in vars(diffusers_utils)
