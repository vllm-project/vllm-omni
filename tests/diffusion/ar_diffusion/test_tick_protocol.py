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

    restored = ARDiffusionTickRequest.from_extra_args(
        tick.to_extra_args(),
        request_id=tick.request_id,
    )

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


def test_tick_rejects_request_id_mismatch() -> None:
    extra_args = ARDiffusionTickRequest(
        session_id="world-1",
        request_id="request-a",
        chunk_index=0,
    ).to_extra_args()

    with pytest.raises(ValueError, match="must match"):
        ARDiffusionTickRequest.from_extra_args(
            extra_args,
            request_id="request-b",
        )


def test_legacy_request_has_no_typed_tick() -> None:
    assert (
        ARDiffusionTickRequest.from_extra_args(
            {"session_id": "legacy"},
            request_id="request-1",
        )
        is None
    )
