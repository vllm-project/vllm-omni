# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Per-request code rows keep the runner payload layout with one host copy."""

import pytest
import torch

from vllm_omni.model_executor.models.moss_tts.local_model_state import _CodeRowsSnapshot
from vllm_omni.model_executor.output_snapshot import PackedOutputSnapshot

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def test_rows_map_to_batch_positions_and_copy_once():
    codes = torch.arange(6, dtype=torch.long).reshape(2, 3)
    snapshot = _CodeRowsSnapshot(codes, [2, 0], 4)
    assert isinstance(snapshot, PackedOutputSnapshot) and snapshot.producer_event is None
    rows = snapshot["codes"]["audio"]
    assert [tuple(row.shape) for row in rows] == [(1, 3), (0, 3), (1, 3), (0, 3)]
    assert torch.equal(rows[2], codes[:1]) and torch.equal(rows[0], codes[1:])

    copies = []

    def copy(tensor):
        copies.append(tensor)
        return tensor.clone()

    host = snapshot.copy_to_cpu(copy)["codes"]["audio"]
    assert len(copies) == 1 and copies[0] is codes
    codes.zero_()
    assert host[2].tolist() == [[0, 1, 2]] and host[0].tolist() == [[3, 4, 5]]
    assert host[1].numel() == 0 and host[1].dtype == torch.long
