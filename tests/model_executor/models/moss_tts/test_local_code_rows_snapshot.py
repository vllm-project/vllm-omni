# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Per-request code rows keep the runner payload layout with one host copy."""

import numpy as np
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


@pytest.mark.parametrize("with_first_audio", [False, True])
def test_partitioned_copy_matches_runner_partition_of_generic_rows(with_first_audio):
    from vllm_omni.model_executor.output_snapshot import RequestOutputSnapshot
    from vllm_omni.worker_v2.omni_ar_model_runner import OmniARModelRunner

    codes = torch.arange(6, dtype=torch.long).reshape(2, 3)
    flags = torch.tensor([False, False, True, False]) if with_first_audio else None
    generic = _CodeRowsSnapshot(codes, [2, 0], 4, flags).copy_to_cpu(lambda tensor: tensor.clone())
    copies = []

    def copy(tensor):
        copies.append(tensor)
        return tensor.clone()

    host = _CodeRowsSnapshot(codes, [2, 0], 4, flags, partitioned=True).copy_to_cpu(copy)
    assert isinstance(host, RequestOutputSnapshot) and host.client is None and len(copies) == 1 + with_first_audio
    qsl = np.arange(5, dtype=np.int32)
    expected, client = OmniARModelRunner._build_async_chunk_outputs_from_mm(
        generic, qsl, np.ones(4, dtype=np.int32), 4, 4, 4
    )
    assert client is None
    assert [list(p) for p in host.inter_stage] == [list(p) for p in expected]
    for got, want in zip(host.inter_stage, expected):
        assert got["codes.audio"].dtype == want["codes.audio"].dtype
        assert torch.equal(got["codes.audio"], want["codes.audio"])
        if with_first_audio:
            assert torch.equal(got["meta.first_audio"], want["meta.first_audio"])
    if with_first_audio:
        flags.zero_()
        assert bool(host.inter_stage[2]["meta.first_audio"])
