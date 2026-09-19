# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Reference-image admission is a per-checkpoint capability, not per-pipeline.

T2I and Edit checkpoints share MageFlowPipeline, so the pre-process function
must resolve the variant identity and refuse reference images for T2I weights;
the engine's dummy warmup request is the one exception, where the image is
dropped so warmup stays on the T2I path.
"""

from types import SimpleNamespace

import pytest
from PIL import Image

from vllm_omni.diffusion.models.mage_flow.pipeline_mage_flow import (
    get_mage_flow_pre_process_func,
)
from vllm_omni.diffusion.models.mage_flow.prompt_utils import (
    get_mage_flow_variant_defaults,
)
from vllm_omni.diffusion.request import (
    DUMMY_DIFFUSION_REQUEST_ID,
    OmniDiffusionRequest,
)
from vllm_omni.inputs.data import OmniDiffusionSamplingParams

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]


def _od_config(model: str):
    return SimpleNamespace(model=model, revision=None)


def _request(request_id: str, with_image: bool = True) -> OmniDiffusionRequest:
    prompt = {"prompt": "a red apple"}
    if with_image:
        prompt["multi_modal_data"] = {"image": Image.new("RGB", (8, 8))}
    return OmniDiffusionRequest(
        prompt=prompt,
        request_id=request_id,
        sampling_params=OmniDiffusionSamplingParams(),
    )


@pytest.mark.parametrize(
    ("identity", "expected"),
    [
        ("microsoft/Mage-Flow", False),
        ("microsoft/Mage-Flow-Base", False),
        ("microsoft/Mage-Flow-Turbo", False),
        ("microsoft/Mage-Flow-Edit", True),
        ("microsoft/Mage-Flow-Edit-Turbo", True),
        ("/turbo-disk/Mage-Flow", False),
    ],
)
def test_variant_defaults_resolve_reference_support(identity: str, expected: bool) -> None:
    assert get_mage_flow_variant_defaults(identity).supports_reference_images is expected


def test_pre_process_rejects_reference_images_for_t2i_checkpoint(tmp_path) -> None:
    model_dir = tmp_path / "Mage-Flow"
    model_dir.mkdir()
    pre_process = get_mage_flow_pre_process_func(_od_config(str(model_dir)))

    with pytest.raises(ValueError, match="text-to-image"):
        pre_process(_request("real-request"))


def test_pre_process_accepts_reference_images_for_edit_checkpoint(tmp_path) -> None:
    model_dir = tmp_path / "Mage-Flow-Edit"
    model_dir.mkdir()
    pre_process = get_mage_flow_pre_process_func(_od_config(str(model_dir)))

    request = pre_process(_request("real-request"))

    assert request.prompt["multi_modal_data"]["image"] is not None


def test_pre_process_strips_reference_image_from_dummy_warmup(tmp_path) -> None:
    model_dir = tmp_path / "Mage-Flow"
    model_dir.mkdir()
    pre_process = get_mage_flow_pre_process_func(_od_config(str(model_dir)))

    request = pre_process(_request(DUMMY_DIFFUSION_REQUEST_ID))

    assert "image" not in request.prompt["multi_modal_data"]


def test_pre_process_keeps_reference_image_for_edit_dummy_warmup(tmp_path) -> None:
    model_dir = tmp_path / "Mage-Flow-Edit"
    model_dir.mkdir()
    pre_process = get_mage_flow_pre_process_func(_od_config(str(model_dir)))

    request = pre_process(_request(DUMMY_DIFFUSION_REQUEST_ID))

    assert request.prompt["multi_modal_data"]["image"] is not None
