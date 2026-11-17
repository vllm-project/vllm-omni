# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import pytest

from tests.model_tests.diffusion import case_filtering
from tests.model_tests.diffusion.config_types import DiffusionAccs, ModelTypeMarker

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


@pytest.mark.parametrize(
    ("test_group", "cards_mark", "other_cards_mark"),
    [
        ([DiffusionAccs.HSDP], "cards_2", "cards_4"),
        ([DiffusionAccs.CFG_PARALLEL, DiffusionAccs.TENSOR_PARALLEL], "cards_4", "cards_2"),
    ],
)
def test_multigpu_groups_keep_cuda_marks_and_add_rocm(monkeypatch, test_group, cards_mark, other_cards_mark):
    class FourDeviceForeignPlatform:
        @staticmethod
        def device_count() -> int:
            return 4

        @staticmethod
        def is_cuda() -> bool:
            return False

    monkeypatch.setattr(case_filtering, "current_omni_platform", FourDeviceForeignPlatform())

    marks = case_filtering.get_test_group_marks(
        "ProbePipeline",
        test_group,
        model_marks=None,
        model_type_marker=ModelTypeMarker.DIFFUSION,
    )
    names = {mark.name for mark in marks}

    assert names >= {
        "full_model",
        "diffusion",
        "L4",
        "B200",
        "cuda",
        "MI325",
        "rocm",
        cards_mark,
    }
    assert other_cards_mark not in names


def test_unqualified_multigpu_model_keeps_cuda_marks_without_rocm(monkeypatch):
    class FourDeviceForeignPlatform:
        @staticmethod
        def device_count() -> int:
            return 4

        @staticmethod
        def is_cuda() -> bool:
            return False

    monkeypatch.setattr(case_filtering, "current_omni_platform", FourDeviceForeignPlatform())

    marks = case_filtering.get_test_group_marks(
        "LTX2Pipeline",
        [DiffusionAccs.CFG_PARALLEL, DiffusionAccs.TENSOR_PARALLEL],
        model_marks=None,
        model_type_marker=ModelTypeMarker.DIFFUSION,
        rocm_multigpu=False,
    )
    names = {mark.name for mark in marks}

    assert names >= {"full_model", "diffusion", "L4", "B200", "cuda", "cards_4"}
    assert names.isdisjoint({"MI325", "rocm"})
