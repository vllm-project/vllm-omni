# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import pytest
import torch

from vllm_omni.outputs.mm_outputs import MultimodalPayload
from vllm_omni.outputs.multimodal_accumulation import (
    drain_delta_payload,
    is_non_final_delta_audio_chunk,
    replace_snapshot_keys,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


@pytest.mark.parametrize("tensor_marker", [False, True])
def test_replay_marker_is_replaced_by_live_output_not_accumulated(tensor_marker):
    """A replay marker is a step snapshot, never generated content/history."""
    from types import SimpleNamespace

    from vllm_omni.engine.orchestrator import Orchestrator
    from vllm_omni.outputs.output_modality import TensorAccumulationStrategy
    from vllm_omni.outputs.output_processor import OmniRequestState

    state = OmniRequestState.__new__(OmniRequestState)
    state.mm_accumulated = MultimodalPayload()
    state.mm_type = None
    for replay in (True, False, True, False):
        marker = torch.tensor([replay]) if tensor_marker else replay
        state.add_multimodal_tensor({"duplex_recovery_replay": marker}, "latent")
        state.mm_accumulated.consolidate_tensors(TensorAccumulationStrategy.CONCAT_DIM0)
        output = SimpleNamespace(outputs=[SimpleNamespace(multimodal_output=dict(state.mm_accumulated))])
        assert Orchestrator._output_is_duplex_recovery_replay(output) is replay


def test_chunk_accumulation_policy_replaces_snapshots_and_drains_delta_state():
    accumulated = MultimodalPayload.from_dict(
        {
            "audio": torch.tensor([1.0]),
            "meta.segment_end": torch.tensor([0]),
            "meta.tts_is_last_chunk": torch.tensor([0]),
            "meta.turn_end": torch.tensor([0]),
            "meta.stable_request_value": "keep",
        }
    )
    incoming = MultimodalPayload.from_dict(
        {
            "audio": torch.tensor([2.0]),
            "meta.segment_end": torch.tensor([1]),
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
    assert "meta.segment_end" not in merged
    assert "meta.tts_is_last_chunk" not in merged
    assert "meta.turn_end" not in merged
    assert merged.metadata["meta.stable_request_value"] == "keep"


def test_context_version_is_a_snapshot_not_a_cumulative_tensor():
    accumulated = MultimodalPayload.from_dict({"meta.duplex_context_version": torch.tensor([1])})
    incoming = MultimodalPayload.from_dict({"meta.duplex_context_version": torch.tensor([2])})
    replace_snapshot_keys(accumulated, incoming)
    merged = accumulated.merged_with(incoming)
    assert merged["meta.duplex_context_version"].tolist() == [2]
    drain_delta_payload(merged)
    assert "meta.duplex_context_version" not in merged
