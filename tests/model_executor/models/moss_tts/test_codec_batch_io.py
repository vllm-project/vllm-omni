# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
import pytest
import torch

from vllm_omni.model_executor.models.moss_tts.modeling_moss_tts_codec import _MossCodecStreamSession


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_slot_id_staging_survives_many_inflight_copies():
    session = object.__new__(_MossCodecStreamSession)
    session._device = torch.device("cuda")
    session._state_slot_ids = torch.arange(64, device="cuda")
    contiguous = session._device_slot_ids([3, 4, 5])
    assert contiguous.data_ptr() == session._state_slot_ids[3:].data_ptr()
    stream = torch.cuda.Stream()
    copies = []
    with torch.cuda.stream(stream):
        # Keep copies in flight while the CPU allocates subsequent staging
        # buffers; mutating one reusable pinned buffer would corrupt old rows.
        torch.cuda._sleep(5_000_000)
        for step in range(128):
            expected = [(step + offset) % 64 for offset in (7, 1, 30, 2)]
            copies.append((expected, session._device_slot_ids(expected)))
    stream.synchronize()
    for expected, actual in copies:
        assert actual.cpu().tolist() == expected


pytestmark = [pytest.mark.core_model, pytest.mark.cuda]
