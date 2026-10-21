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


@pytest.mark.parametrize(
    "path",
    [
        "tests/dfx/reliability/test_qwen3_omni.py",
        "tests/dfx/reliability/test_wan22.py",
        "tests/dfx/reliability/test_hunyuan_image.py",
        "tests/dfx/reliability/test_minicpmo_4_5.py",
    ],
)
def test_reliability_model_files_carry_marks(path):
    assert check_test_marks.get_files_missing_markers([path]) == {}


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


@pytest.mark.parametrize(
    "source",
    [
        '@hardware_test(res={"rocm": "mi300_1"})\ndef test_transfer(): pass',
        '@hardware_marks(res={"rocm": "MI325"})\ndef test_transfer(): pass',
        '@hardware_test({"cuda": "H100"})\ndef test_transfer(): pass',
        '@hardware_test(res={"cuda": "H100"}, unknown=True)\ndef test_transfer(): pass',
        '@hardware_test(res={"cuda": "H100"}, num_cards=True)\ndef test_transfer(): pass',
        '@hardware_test(res={"cuda": "H100"}, num_cards=None)\ndef test_transfer(): pass',
        "@hardware_test(res=None)\ndef test_transfer(): pass",
        "@hardware_test(res={})\ndef test_transfer(): pass",
        "@hardware_test()\ndef test_transfer(): pass",
        '@hardware_test(res={"cpu": "MI325"})\ndef test_transfer(): pass',
        '@hardware_test(res={"rocm": ["MI325"]})\ndef test_transfer(): pass',
        '@hardware_test(res={"cuda": "H100"}, num_cards={"rocm": 1})\ndef test_transfer(): pass',
        'pytestmark = hardware_marks(res={"cuda": "H100", "rocm": "MI325"}, '
        'num_cards={"cuda": 2, "rocm": 1})\ndef test_transfer(): pass',
        "from tests.helpers.mark import hardware_marks as marks\n"
        '@marks(res={"rocm": "MI325"})\nclass TestTransfer: pass',
        'def collection_marks():\n    return hardware_marks(res={"rocm": "mi300_1"})',
    ],
)
def test_hardware_contract_rejects_collection_failures(source):
    assert check_test_marks.hardware_contract_errors(source) == ["Invalid hardware helper"]


def test_invalid_hardware_contract_is_wired_into_hook(monkeypatch):
    source = '@pytest.mark.core_model\n@hardware_marks(res={"rocm": "MI325"})\ndef test_transfer(): pass\n'
    path = "tests/test_invalid_hardware_helper.py"
    monkeypatch.setattr(check_test_marks, "read_test_file", lambda filename: source)
    assert check_test_marks.get_files_missing_markers([path]) == {path: ["Invalid hardware helper"]}


def test_valid_helper_alias_satisfies_marker_presence_check(monkeypatch):
    source = (
        "from tests.helpers.mark import hardware_test as hw\n"
        '@pytest.mark.core_model\n@hw(res={"rocm": "MI325"})\ndef test_x(): pass'
    )
    path = "tests/test_alias.py"
    monkeypatch.setattr(check_test_marks, "read_test_file", lambda filename: source)
    assert check_test_marks.get_files_missing_markers([path]) == {}


@pytest.mark.parametrize(
    "source",
    [
        "pytestmark = [pytest.mark.cpu]\n@pytest.mark.cuda\ndef test_graph(): pass",
        'pytestmark = pytest.mark.cpu\n@hardware_test(res={"rocm": "MI325"})\ndef test_transfer(): pass',
        'pytestmark = pytest.mark.cpu\n@hardware_test(res={"npu": "A3"})\ndef test_transfer(): pass',
        "pytestmark: list = [pytest.mark.gpu]\n@pytest.mark.cpu\ndef test_reference(): pass",
        "class TestTransfer:\n    pytestmark = pytest.mark.cpu\n"
        '    @hardware_test(res={"rocm": "MI325"})\n    def test_transfer(self): pass',
        "pytestmark = pytest.mark.cpu\n"
        '@pytest.mark.parametrize("case", [pytest.param(1, marks=hardware_marks(res={"cuda": "L4"}))])\n'
        "def test_transfer(case): pass",
    ],
)
def test_hardware_contract_rejects_inherited_cpu_accelerator_marks(source):
    assert check_test_marks.hardware_contract_errors(source) == ["Conflicting CPU/accelerator marks"]


@pytest.mark.parametrize(
    "source",
    [
        '@hardware_test(res={"cuda": ["H100", "B200"], "rocm": "MI325"})\ndef test_transfer(): pass',
        'pytestmark = hardware_marks(res={"rocm": "MI325"}, num_cards=1)\ndef test_transfer(): pass',
        'cases = [pytest.param(1, marks=hardware_marks(res={"cuda": "L4"}, num_cards=2))]',
        '@hardware_test(res={"cuda": "H100", "rocm": "MI325"}, '
        'num_cards={"cuda": 2, "rocm": 1})\ndef test_distributed(): pass',
        '@pytest.mark.cpu\ndef test_plan(): pass\n@hardware_test(res={"rocm": "MI325"})\ndef test_transfer(): pass',
        "@hardware_test(res=resource_map, num_cards=card_count)\ndef test_dynamic(): pass",
        "def test_invalid_resource():\n    with pytest.raises(ValueError):\n"
        '        hardware_marks(res={"rocm": "mi300_1"})',
        "pytestmark = pytest.mark.cpu\ndef test_helper_decorates_a_probe():\n"
        '    @hardware_test(res={"rocm": "MI325"})\n    def test_probe(): pass',
        '@hardware_test(res={"rocm": "MI325"})\nclass TestTransfer:\n    def test_transfer(self): pass',
        "@pytest.mark.parametrize(\n"
        '    "device",\n'
        '    [pytest.param("cpu", marks=pytest.mark.cpu),\n'
        '     pytest.param("cuda", marks=hardware_marks(res={"cuda": "L4"}))],\n'
        ")\ndef test_transfer(device): pass",
    ],
)
def test_hardware_contract_preserves_valid_cpu_and_accelerator_selections(source):
    assert check_test_marks.hardware_contract_errors(source) == []


def test_existing_mixed_cpu_cuda_params_remain_per_item():
    source = check_test_marks.read_test_file("tests/model_executor/models/ming_tts/test_fused_istft.py")
    assert source is not None
    assert "Conflicting CPU/accelerator marks" not in check_test_marks.hardware_contract_errors(source)
