# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import pytest

from tools.pre_commit import check_test_marks

pytestmark = [pytest.mark.cpu, pytest.mark.core_model]


@pytest.mark.parametrize("kind", ["offline", "online"])
def test_common_diffusion_marks_are_checked_in_helper(kind, monkeypatch):
    path = f"tests/model_tests/diffusion/test_common_{kind}.py"
    assert check_test_marks.get_files_missing_markers([path]) == {}

    read_file = check_test_marks.read_test_file
    monkeypatch.setattr(
        check_test_marks,
        "read_test_file",
        lambda filename: "" if filename.endswith("case_filtering.py") else read_file(filename),
    )
    assert check_test_marks.get_files_missing_markers([path]) == {path: ["Level", "Hardware"]}


def test_unrelated_tests_cannot_inherit_diffusion_marks(monkeypatch):
    path = "tests/test_missing_marks.py"
    monkeypatch.setattr(check_test_marks, "read_test_file", lambda filename: "get_parametrized_options(settings)")
    assert check_test_marks.get_files_missing_markers([path]) == {path: ["Level", "Hardware"]}


def test_bare_platform_mark_is_rejected(monkeypatch):
    path = "tests/test_bare_platform.py"
    # Split so this file itself does not contain a bare platform mark literal.
    for name in ("cuda", "npu", "gpu", "musa", "rocm", "xpu"):
        bare = "pytest.mark." + name
        monkeypatch.setattr(
            check_test_marks,
            "read_test_file",
            lambda filename, bare=bare: f"pytestmark = [pytest.mark.core_model, {bare}]\n",
        )
        assert check_test_marks.get_files_missing_markers([path]) == {path: ["Bare platform"]}


def test_cpu_mark_is_allowed(monkeypatch):
    path = "tests/test_cpu_mark.py"
    monkeypatch.setattr(
        check_test_marks,
        "read_test_file",
        lambda filename: "pytestmark = [pytest.mark.core_model, pytest.mark.cpu]\n",
    )
    assert check_test_marks.get_files_missing_markers([path]) == {}


def test_platform_and_sku_mentions_in_comments_and_strings_are_ignored(monkeypatch):
    path = "tests/test_cpu_mark.py"
    platform = "pytest.mark." + "cuda"
    sku = "pytest.mark." + "H100"
    sources = [
        f"pytestmark = [pytest.mark.core_model, pytest.mark.cpu]\n# GPU counterpart uses {platform}\n",
        f'pytestmark = [pytest.mark.core_model, pytest.mark.cpu]\nNOTE = "{platform}"\n',
        f'"""Document {platform} and {sku} here."""\npytestmark = [pytest.mark.core_model, pytest.mark.cpu]\n',
        "pytestmark = [pytest.mark.core_model]\n"
        f"# Do not write {sku} by hand.\n"
        "@hardware_test(res={'cuda': 'L4'}, num_cards=1)\n",
    ]
    for source in sources:
        monkeypatch.setattr(check_test_marks, "read_test_file", lambda filename, source=source: source)
        assert check_test_marks.get_files_missing_markers([path]) == {}


def test_cuda_sku_helper_is_accepted(monkeypatch):
    path = "tests/test_cuda_sku.py"
    monkeypatch.setattr(
        check_test_marks,
        "read_test_file",
        lambda filename: "pytestmark = [pytest.mark.core_model]\n\n@hardware_test(res={'cuda': 'L4'}, num_cards=1)\n",
    )
    assert check_test_marks.get_files_missing_markers([path]) == {}
