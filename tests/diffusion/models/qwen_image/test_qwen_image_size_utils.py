import pytest

from vllm_omni.diffusion.utils.size_utils import (
    normalize_min_aligned_size,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


@pytest.mark.parametrize(
    ("height", "width", "expected"),
    [
        (1, 1, (16, 16)),
        (15, 15, (16, 16)),
        (17, 17, (16, 16)),
        (31, 33, (16, 32)),
        (64, 80, (64, 80)),
    ],
)
def test_normalize_min_aligned_size_clamps_to_minimum_aligned_shape(height, width, expected):
    assert normalize_min_aligned_size(height, width, alignment=16) == expected


def test_normalize_min_aligned_size_rejects_invalid_alignment():
    with pytest.raises(ValueError, match="positive alignment"):
        normalize_min_aligned_size(16, 16, alignment=0)
