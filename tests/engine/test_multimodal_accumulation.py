# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm_omni.outputs.mm_outputs import MultimodalPayload
from vllm_omni.outputs.multimodal_accumulation import (
    drain_delta_payload,
    is_non_final_delta_audio_chunk,
    replace_snapshot_keys,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def test_chunk_accumulation_policy_replaces_snapshots_and_drains_delta_state():
    accumulated = MultimodalPayload.from_dict(
        {
            "audio": torch.tensor([1.0]),
            "meta.tts_is_last_chunk": torch.tensor([0]),
            "meta.turn_end": torch.tensor([0]),
            "meta.stable_request_value": "keep",
        }
    )
    incoming = MultimodalPayload.from_dict(
        {
            "audio": torch.tensor([2.0]),
            "meta.tts_is_last_chunk": torch.tensor([1]),
            "meta.turn_end": torch.tensor([1]),
        }
    )
    assert accumulated is not None
    assert incoming is not None

    replace_snapshot_keys(accumulated, incoming)
    merged = accumulated.merged_with(incoming)

    assert not is_non_final_delta_audio_chunk(merged, "audio")

    drain_delta_payload(merged)

    assert "audio" not in merged
    assert "meta.tts_is_last_chunk" not in merged
    assert "meta.turn_end" not in merged
    assert merged.metadata["meta.stable_request_value"] == "keep"
