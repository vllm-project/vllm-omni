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
