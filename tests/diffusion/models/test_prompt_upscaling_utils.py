from types import SimpleNamespace
from typing import Any, cast

import pytest

from vllm_omni.diffusion.request import OmniDiffusionRequest
from vllm_omni.diffusion.utils.prompt_utils import do_prompt_upscaling

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def mock_request(extra_args: dict[str, Any]) -> OmniDiffusionRequest:
    return cast(
        OmniDiffusionRequest,
        SimpleNamespace(sampling_params=SimpleNamespace(extra_args=extra_args)),
    )


@pytest.mark.parametrize(
    "extra_args,expected",
    [
        ({"use_prompt_upscaling": True}, True),
        ({"use_prompt_upscaling": False}, False),
        ({}, False),
    ],
)
def test_do_prompt_upscaling(extra_args, expected):
    """Ensure we only enable upscale if it's explicitly requested."""
    assert do_prompt_upscaling(mock_request(extra_args)) is expected


def test_do_prompt_upscaling_rejects_non_bool():
    """Ensure do_prompt_upscaling requires a boolean value."""
    with pytest.raises(TypeError, match="must be a bool"):
        do_prompt_upscaling(mock_request({"use_prompt_upscaling": object()}))
