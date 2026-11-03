# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import pytest
import torch

from vllm_omni.outputs.mm_outputs import MultimodalPayload
from vllm_omni.outputs.multimodal_accumulation import (
    drain_delta_payload,
    is_non_final_delta_audio_chunk,
    replace_nemotron_snapshot_keys,
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


def test_replace_nemotron_snapshot_keys_is_scoped_to_nemotron_marker():
    marker = "meta.nvc_logical_prompt_len"
    accumulated = MultimodalPayload.from_dict(
        {
            "codes.audio": torch.ones(2, 4, dtype=torch.long),
            "meta.nvc_tts_step": torch.tensor([1]),
            marker: 3,
        }
    )
    assert accumulated is not None

    # Marked wake re-ships the whole prefix -> prior tensors dropped before merge.
    marked = MultimodalPayload.from_dict(
        {
            "codes.audio": torch.ones(5, 4, dtype=torch.long),
            "meta.nvc_tts_step": torch.tensor([2]),
            marker: 3,
        }
    )
    assert marked is not None
    replace_nemotron_snapshot_keys(accumulated, marked)
    merged = accumulated.merged_with(marked)
    assert isinstance(merged["codes.audio"], torch.Tensor)
    assert merged["codes.audio"].shape[0] == 5
    assert isinstance(merged["meta.nvc_tts_step"], torch.Tensor)
    assert merged["meta.nvc_tts_step"].tolist() == [2]

    # Unmarked codec deltas keep accumulating under the same key.
    unmarked = MultimodalPayload.from_dict({"codes.audio": torch.ones(3, 4, dtype=torch.long)})
    assert unmarked is not None
    replace_nemotron_snapshot_keys(merged, unmarked)
    grown = merged.merged_with(unmarked)
    assert isinstance(grown["codes.audio"], list)
    assert sum(t.shape[0] for t in grown["codes.audio"]) == 8


def test_replace_nemotron_snapshot_keys_detects_nested_meta_marker():
    # Marker nested under a "meta" dict (unflattened producer form) must still
    # scope the REPLACE, with no global key registration required.
    incoming = MultimodalPayload.from_dict(
        {"codes.audio": torch.ones(1, 4, dtype=torch.long), "meta": {"nvc_logical_prompt_len": 3}}
    )
    assert incoming is not None
    accumulated = MultimodalPayload.from_dict({"codes.audio": torch.ones(4, 4, dtype=torch.long)})
    assert accumulated is not None

    replace_nemotron_snapshot_keys(accumulated, incoming)
    assert "codes.audio" not in accumulated


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

    flag = merged["meta.finished"]
    assert not isinstance(flag, list)
    assert bool(flag.reshape(())) is True


def test_meta_finished_consolidates_to_latest_flag_without_cat():
    """Even a list of 0-d finished flags must not go through torch.cat."""
    payload = MultimodalPayload.from_dict({"audio": torch.zeros(3)})
    payload.tensors["meta.finished"] = [torch.tensor(False), torch.tensor(True)]

    payload.consolidate_tensors(OutputModality.TEXT)

    assert bool(payload["meta.finished"].reshape(())) is True
