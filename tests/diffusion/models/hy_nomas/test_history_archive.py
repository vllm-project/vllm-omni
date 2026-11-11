# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import pytest
import torch

from vllm_omni.diffusion.models.hy_nomas.configuration_hy_nomas import HyNomasARConfig
from vllm_omni.diffusion.models.hy_nomas.history_archive import HyNomasHistoryArchive
from vllm_omni.errors import OmniClientError

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]


def stage(archive, *, frame_start=0):
    key = torch.arange(24, dtype=torch.float32).reshape(6, 2, 2)
    mask = torch.tensor([[True, True, True, True, True, False]])
    archive.stage(
        ((key, -key), (key + 10, -key - 10)),
        mask,
        frame_start=frame_start,
        frame_count=2,
        tokens_per_frame=2,
        segment=frame_start // 2 + 1,
        text_tokens=2,
        action_start=1,
    )
    return key, mask


def test_archive_copies_only_pending_rows_and_publishes_atomically():
    archive = HyNomasHistoryArchive(max_bytes=780, max_frames=4)
    key, mask = stage(archive)
    assert archive.records == () and archive.frames == archive.payload_bytes == 0
    key.fill_(99)
    mask.fill_(False)
    archive.publish()
    record = archive.records[0]
    assert archive.frames == 2 and archive.payload_bytes == 390
    torch.testing.assert_close(record.layers[0][0], torch.arange(24, dtype=torch.float32).reshape(6, 2, 2))
    assert record.mask.tolist() == [[True, True, True, True, True, False]]
    assert (record.segment, record.text_tokens, record.action_start) == (1, 2, 1)
    assert record.frame_rows(1) == slice(2, 4)
    with pytest.raises(IndexError):
        record.frame_rows(2)
    stage(archive, frame_start=2)
    archive.rollback()
    assert archive.records == (record,) and archive.payload_bytes == 390
    stage(archive, frame_start=2)
    archive.publish()
    assert archive.frames == 4 and archive.payload_bytes == 780
    archive.clear()
    assert archive.records == () and archive.frames == archive.payload_bytes == 0
    with pytest.raises(RuntimeError, match="no pending"):
        archive.publish()


@pytest.mark.parametrize("limits", [{"max_bytes": 779, "max_frames": 4}, {"max_bytes": 780, "max_frames": 3}])
def test_archive_quota_rejects_whole_chunk_without_changing_committed_records(limits):
    archive = HyNomasHistoryArchive(**limits)
    stage(archive)
    archive.publish()
    committed = archive.records
    with pytest.raises(OmniClientError, match="quota exceeded"):
        stage(archive, frame_start=2)
    assert archive.records is committed and archive.payload_bytes == 390 and archive.frames == 2
    with pytest.raises(RuntimeError, match="no pending"):
        archive.publish()


def test_archive_rejects_gaps_and_overlapping_transactions():
    archive = HyNomasHistoryArchive(max_bytes=1000, max_frames=4)
    with pytest.raises(ValueError, match="contiguous"):
        stage(archive, frame_start=1)
    stage(archive)
    with pytest.raises(RuntimeError, match="pending"):
        stage(archive)
    archive.rollback()
    stage(archive)
    archive.publish()
    with pytest.raises(ValueError, match="contiguous"):
        stage(archive)


@pytest.mark.parametrize(
    "args",
    [
        {"ar_history_archive_max_bytes": -1},
        {"ar_history_archive_max_bytes": True},
        {"ar_history_archive_max_bytes": 1.5},
        {"ar_history_archive_max_bytes": 1024},
        {"ar_history_archive_max_frames": 0},
        {"ar_history_archive_max_frames": True},
    ],
)
def test_archive_configuration_rejects_invalid_limits_and_full_mode(args):
    with pytest.raises(ValueError, match="archive|archival"):
        HyNomasARConfig.from_args(args)


def test_archive_is_disabled_by_default_and_explicitly_enabled_for_recent():
    assert HyNomasARConfig.from_args({}).history_archive_max_bytes == 0
    config = HyNomasARConfig.from_args({"ar_history_mode": "recent", "ar_history_archive_max_bytes": 1024})
    assert config.history_archive_max_bytes == 1024
