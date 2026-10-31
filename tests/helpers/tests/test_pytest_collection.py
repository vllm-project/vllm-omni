# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from tests.helpers.fixtures.pytest_collection import _pin_xdist_serial

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


class _Item:
    def __init__(self, path: Path, marks: set[str]):
        self.path = path
        self._marks: dict[str, object] = {name: True for name in marks}

    def get_closest_marker(self, name: str):
        return self._marks.get(name)

    def add_marker(self, marker):
        mark = marker.mark
        self._marks[mark.name] = mark

    def group_name(self) -> str | None:
        mark = self._marks.get("xdist_group")
        if mark is True or mark is None:
            return None
        kwargs = getattr(mark, "kwargs", None)
        if not isinstance(kwargs, dict):
            return None
        name = kwargs.get("name")
        return name if isinstance(name, str) else None


def _config(*, xdist: bool):
    return SimpleNamespace(pluginmanager=SimpleNamespace(hasplugin=lambda name: xdist and name == "xdist"))


def test_cpu_tests_share_a_file_group_and_unmarked_tests_stay_serial(tmp_path):
    cpu_item = _Item(tmp_path / "cpu_case.py", {"cpu"})
    opted_in = _Item(tmp_path / "opted_in.py", {"xdist"})
    plain = _Item(tmp_path / "plain.py", set())

    _pin_xdist_serial(_config(xdist=True), [cpu_item, opted_in, plain])

    assert cpu_item.get_closest_marker("xdist").name == "xdist"
    assert cpu_item.group_name() == f"file:{(tmp_path / 'cpu_case.py').as_posix()}"
    assert opted_in.group_name() is None
    assert plain.group_name() == "serial"


def test_pin_xdist_serial_is_noop_without_xdist(tmp_path):
    plain = _Item(tmp_path / "plain.py", set())

    _pin_xdist_serial(_config(xdist=False), [plain])

    assert plain.group_name() is None
