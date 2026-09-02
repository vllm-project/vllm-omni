# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import pytest

from vllm_omni.diffusion.models.cosmos_dreams.geometry import CosmosDreamsResolutionPolicy
from vllm_omni.diffusion.models.cosmos_dreams.pipeline_cosmos_dreams_transfer import (
    resolve_cosmos_dreams_transfer_geometry,
)


def _sampling_params(
    *,
    resolution: str = "480",
    height: int | None = None,
    width: int | None = None,
    **extra,
):
    return SimpleNamespace(
        height=height,
        width=width,
        extra_args={"resolution": resolution, **extra},
    )


def test_transfer_input_video_drives_bucket_before_hint_control() -> None:
    params = _sampling_params(edge={"control": {"height": 480, "width": 480}})
    prompt = {
        "multi_modal_data": {"video": {"height": 480, "width": 832}},
    }

    geometry = resolve_cosmos_dreams_transfer_geometry(
        params,
        prompt,
        CosmosDreamsResolutionPolicy(),
    )

    assert geometry.session_key == (480, 832)


def test_transfer_hint_control_drives_bucket_without_input_video() -> None:
    params = _sampling_params(depth={"control": {"height": 832, "width": 480}})

    geometry = resolve_cosmos_dreams_transfer_geometry(
        params,
        {"prompt": "test"},
        CosmosDreamsResolutionPolicy(),
    )

    assert geometry.session_key == (832, 480)


def test_transfer_serialized_dimensions_must_match_generated_bucket() -> None:
    params = _sampling_params(
        height=720,
        width=1280,
        edge={"control": {"height": 480, "width": 832}},
    )

    with pytest.raises(ValueError, match="serialized dimensions do not match"):
        resolve_cosmos_dreams_transfer_geometry(
            params,
            {"prompt": "test"},
            CosmosDreamsResolutionPolicy(),
        )


def test_transfer_generated_bucket_must_pass_dreams_policy() -> None:
    params = _sampling_params(resolution="720", edge={"control": {"height": 480, "width": 832}})
    policy = CosmosDreamsResolutionPolicy(
        default_resolution=(480, 832),
        max_pixels=480 * 832,
    )

    with pytest.raises(ValueError, match="max_pixels"):
        resolve_cosmos_dreams_transfer_geometry(params, {"prompt": "test"}, policy)
