# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import pytest

from tests.helpers.mark import (
    _gpu_res_platforms,
    hardware_marks,
    hardware_test,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _mark_names(marks) -> set[str]:
    return {mark.name for mark in marks}


def test_gpu_platforms_come_from_pyproject_gpu_tag():
    assert "npu" not in _gpu_res_platforms()
    assert "gpu" not in _gpu_res_platforms()


def test_default_num_cards_adds_cards_1():
    names = _mark_names(hardware_marks(res={"cuda": "H100"}))
    assert "cards_1" in names
    assert "H100" in names
    assert "cards_2" not in names


def test_h100_two_cards_is_selectable_with_cards_2():
    names = _mark_names(hardware_marks(res={"cuda": "H100"}, num_cards=2))
    assert names >= {"H100", "cuda", "gpu", "cards_2"}
    assert "cards_1" not in names
    assert "cards_4" not in names


def test_h100_four_cards_is_selectable_with_cards_4():
    names = _mark_names(hardware_marks(res={"cuda": "H100"}, num_cards=4))
    assert "cards_4" in names
    assert "cards_2" not in names


def test_npu_card_count_mark_is_independent_of_sku():
    names = _mark_names(hardware_marks(res={"npu": "A2"}, num_cards=4))
    assert "cards_4" in names
    assert "npu" in names
    assert "gpu" not in names


def test_hardware_marks_rejects_mixed_counts_on_one_item():
    with pytest.raises(ValueError, match="different per-platform num_cards"):
        hardware_marks(
            res={"cuda": "H100", "rocm": "MI325"},
            num_cards={"cuda": 4, "rocm": 2},
        )


def test_same_count_multi_platform_stays_one_item():
    names = _mark_names(hardware_marks(res={"cuda": "H100", "rocm": "MI325"}, num_cards=2))
    assert names >= {"H100", "cuda", "MI325", "rocm", "cards_2"}

    @hardware_test(res={"cuda": "H100", "rocm": "MI325"}, num_cards=2)
    def _probe():
        return None

    names = {mark.name for mark in _probe.pytestmark}
    assert "parametrize" not in names
    assert "cards_2" in names
    assert "H100" in names
    assert "MI325" in names


def test_hardware_test_decorator_applies_cards_mark():
    @hardware_test(res={"cuda": "H100"}, num_cards=2)
    def _probe():
        return None

    names = {mark.name for mark in _probe.pytestmark}
    assert "cards_2" in names
    assert "H100" in names


def _hardware_platform_params(func):
    marks = func.pytestmark if isinstance(func.pytestmark, list) else [func.pytestmark]
    for mark in marks:
        if mark.name != "parametrize":
            continue
        argnames = mark.args[0]
        if argnames == "_normalized_hardware_marks":
            return mark.args[1]
    raise AssertionError("expected @hardware_test to parametrize _normalized_hardware_marks")


def test_mixed_card_counts_split_so_cuda_filter_does_not_see_rocm_count():
    @hardware_test(res={"cuda": "H100", "rocm": "MI325"}, num_cards={"cuda": 4, "rocm": 2})
    def _probe():
        return None

    variants = {param.id: {mark.name for mark in param.marks} for param in _hardware_platform_params(_probe)}
    cuda_names = variants["cuda"]
    rocm_names = variants["rocm"]

    assert cuda_names >= {"H100", "cuda", "cards_4"}
    assert "cards_2" not in cuda_names
    assert "rocm" not in cuda_names
    assert "MI325" not in cuda_names

    assert rocm_names >= {"MI325", "rocm", "cards_2"}
    assert "cards_4" not in rocm_names
    assert "cuda" not in rocm_names
    assert "H100" not in rocm_names


def test_unsupported_card_count_raises():
    with pytest.raises(ValueError, match="num_cards=16"):
        hardware_marks(res={"cuda": "H100"}, num_cards=16)


@pytest.mark.parametrize("platform", ["cpu", "gpu"])
def test_res_rejects_cpu_and_gpu(platform):
    with pytest.raises(ValueError, match="not a res dict key"):
        hardware_marks(res={platform: "H100"})


def test_npu_rejects_unsupported_sku():
    with pytest.raises(ValueError, match="Invalid npu resource type"):
        hardware_marks(res={"npu": "A100"})


@pytest.mark.parametrize("sku", ["H200", "B200", "H800"])
def test_cuda_registered_skus_attach_sku_and_cards(sku):
    names = _mark_names(hardware_marks(res={"cuda": sku}, num_cards=2))
    assert names >= {sku, "cuda", "gpu", "cards_2"}


def test_cuda_rejects_sku_from_another_platform():
    with pytest.raises(ValueError, match="Invalid cuda resource type"):
        hardware_marks(res={"cuda": "MI325"})
