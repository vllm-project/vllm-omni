# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import pytest

from vllm_omni.diffusion.models.cosmos3_nano_sim_bimanual.geometry import Cosmos3NanoSimBimanualResolutionPolicy
from vllm_omni.diffusion.models.cosmos3_nano_sim_bimanual.pipeline_cosmos3_nano_sim_transfer import (
    resolve_cosmos3_nano_sim_transfer_geometry,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


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

    geometry = resolve_cosmos3_nano_sim_transfer_geometry(
        params,
        prompt,
        Cosmos3NanoSimBimanualResolutionPolicy(),
    )

    assert geometry.session_key == (480, 832)


def test_transfer_hint_control_drives_bucket_without_input_video() -> None:
    params = _sampling_params(depth={"control": {"height": 832, "width": 480}})

    geometry = resolve_cosmos3_nano_sim_transfer_geometry(
        params,
        {"prompt": "test"},
        Cosmos3NanoSimBimanualResolutionPolicy(),
    )

    assert geometry.session_key == (832, 480)


def test_transfer_serialized_dimensions_must_match_generated_bucket() -> None:
    params = _sampling_params(
        height=720,
        width=1280,
        edge={"control": {"height": 480, "width": 832}},
    )

    with pytest.raises(ValueError, match="serialized dimensions do not match"):
        resolve_cosmos3_nano_sim_transfer_geometry(
            params,
            {"prompt": "test"},
            Cosmos3NanoSimBimanualResolutionPolicy(),
        )


def test_transfer_generated_bucket_must_pass_bimanual_policy() -> None:
    params = _sampling_params(resolution="720", edge={"control": {"height": 480, "width": 832}})
    policy = Cosmos3NanoSimBimanualResolutionPolicy(
        default_resolution=(480, 832),
        max_pixels=480 * 832,
    )

    with pytest.raises(ValueError, match="max_pixels"):
        resolve_cosmos3_nano_sim_transfer_geometry(params, {"prompt": "test"}, policy)
