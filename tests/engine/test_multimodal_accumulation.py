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
from vllm_omni.outputs.output_modality import OutputModality

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


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


def test_meta_finished_is_replaced_per_chunk_not_appended():
    """The per-step end-of-stream flag is a snapshot, like segment_end.

    Accumulating the 0-d per-step tensors into a list makes the final
    torch.cat raise ("zero-dimensional tensor cannot be concatenated"),
    which aborted whole TTS requests once K-step decode let streams
    survive to the consolidation point.
    """
    accumulated = MultimodalPayload.from_dict(
        {
            "audio": torch.zeros(4),
            "meta.finished": torch.tensor(False),
        }
    )
    incoming = MultimodalPayload.from_dict(
        {
            "audio": torch.ones(2),
            "meta.finished": torch.tensor(True),
        }
    )
    assert accumulated is not None
    assert incoming is not None

    replace_snapshot_keys(accumulated, incoming)
    merged = accumulated.merged_with(incoming)

    flag = merged.tensors["meta.finished"]
    assert not isinstance(flag, list)
    assert bool(flag.reshape(())) is True


def test_meta_finished_consolidates_to_latest_flag_without_cat():
    """Even a list of 0-d finished flags must not go through torch.cat."""
    payload = MultimodalPayload.from_dict({"audio": torch.zeros(3)})
    payload.tensors["meta.finished"] = [torch.tensor(False), torch.tensor(True)]

    payload.consolidate_tensors(OutputModality.TEXT)

    assert bool(payload.tensors["meta.finished"].reshape(())) is True
