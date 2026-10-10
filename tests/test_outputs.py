# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import pytest
import torch

from vllm_omni.outputs.mm_outputs import MultimodalPayload
from vllm_omni.outputs.output_modality import OutputModality

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def test_streamed_funaudiochat_waveform_deltas_consolidate_and_keep_sample_rate() -> None:
    first = MultimodalPayload.from_raw(
        {
            "audio": torch.tensor([0.1, 0.2], dtype=torch.float32),
            "sr": torch.tensor(24000, dtype=torch.int32),
        },
        "audio",
    )
    second = MultimodalPayload.from_raw(
        {
            "audio": torch.tensor([0.3, 0.4, 0.5], dtype=torch.float32),
            "sr": torch.tensor(24000, dtype=torch.int32),
        },
        "audio",
    )

    assert first is not None and second is not None
    accumulated = first.merged_with(second)
    assert isinstance(accumulated.tensors["audio"], list)

    accumulated.consolidate_tensors(OutputModality.AUDIO)
    accumulated.consolidate_metadata()

    torch.testing.assert_close(accumulated["audio"], torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5]))
    assert accumulated["audio"].dtype == torch.float32
    assert accumulated["sr"].item() == 24000
