# SPDX-License-Identifier: Apache-2.0

import pytest

from vllm_omni.experimental.ar_diffusion.tick_protocol import (
    ARDiffusionChunkMetadata,
    ARDiffusionControlInput,
    ARDiffusionTickRequest,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def test_tick_round_trip_preserves_identity_and_control_snapshot() -> None:
    tick = ARDiffusionTickRequest(
        session_id="world-1",
        request_id="chunk-request-7",
        chunk_index=7,
        applied_event_ids=(10, 11),
        prompt="turn left",
        controls=(
            ARDiffusionControlInput(
                track="camera",
                schema="lingbot.camera_trajectory.v1",
                data={"poses": [[1.0, 0.0]], "intrinsics": [[2.0]]},
            ),
        ),
    )

    restored = ARDiffusionTickRequest.from_extra_args(tick.to_extra_args())

    assert restored == tick
    assert ARDiffusionChunkMetadata.from_tick(tick).to_dict() == {
        "session_id": "world-1",
        "request_id": "chunk-request-7",
        "chunk_index": 7,
        "applied_event_ids": [10, 11],
    }


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("session_id", "", "session_id"),
        ("request_id", "", "request_id"),
        ("chunk_index", -1, "chunk_index"),
        ("applied_event_ids", (2, 1), "strictly increasing"),
        ("applied_event_ids", (1, 1), "strictly increasing"),
    ],
)
def test_tick_rejects_invalid_identity(
    field: str,
    value: object,
    message: str,
) -> None:
    values = {
        "session_id": "world-1",
        "request_id": "request-1",
        "chunk_index": 0,
        "applied_event_ids": (),
    }
    values[field] = value

    with pytest.raises(ValueError, match=message):
        ARDiffusionTickRequest(**values)


def test_tick_snapshot_is_detached_and_deeply_immutable() -> None:
    payload = {"samples": [{"keys": ["w"]}]}
    event_ids = [1, 2]
    controls = [
        ARDiffusionControlInput(
            track="camera",
            schema="example.camera.v1",
            data=payload,
        )
    ]
    tick = ARDiffusionTickRequest(
        session_id="world-1",
        request_id="request-a",
        chunk_index=0,
        applied_event_ids=event_ids,  # type: ignore[arg-type]
        controls=controls,  # type: ignore[arg-type]
    )
    serialized = tick.to_extra_args()

    payload["samples"][0]["keys"].append("j")
    event_ids.append(3)
    controls.clear()
    serialized["ar_diffusion_tick"]["controls"][0]["data"]["samples"][0]["keys"].append("k")

    assert tick.applied_event_ids == (1, 2)
    assert isinstance(tick.controls, tuple)
    assert tick.to_extra_args()["ar_diffusion_tick"]["controls"] == [
        {
            "track": "camera",
            "schema": "example.camera.v1",
            "data": {"samples": [{"keys": ["w"]}]},
        }
    ]
    with pytest.raises(TypeError):
        tick.controls[0].data["samples"] = ()  # type: ignore[index]
    with pytest.raises(AttributeError):
        tick.controls[0].data["samples"][0]["keys"].append("j")


def test_legacy_request_has_no_typed_tick() -> None:
    assert ARDiffusionTickRequest.from_extra_args({"session_id": "legacy"}) is None
