# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch

from vllm_omni.core.sched.omni_generation_scheduler import OmniGenerationScheduler

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def test_generation_scheduler_records_audio_output_for_edf(mocker):
    scheduler = OmniGenerationScheduler.__new__(OmniGenerationScheduler)
    scheduler.chunk_transfer_adapter = SimpleNamespace(
        playback_deadline_enabled=True,
        record_audio_output=mocker.Mock(),
    )
    multimodal_output = {
        "model_outputs": torch.zeros(24000),
        "sr": torch.tensor(24000),
    }

    scheduler._record_playback_deadline_output("req", multimodal_output)

    scheduler.chunk_transfer_adapter.record_audio_output.assert_called_once_with("req", 24000, 24000)


def test_generation_scheduler_skips_audio_accounting_when_edf_is_disabled(mocker):
    scheduler = OmniGenerationScheduler.__new__(OmniGenerationScheduler)
    scheduler.chunk_transfer_adapter = SimpleNamespace(
        playback_deadline_enabled=False,
        record_audio_output=mocker.Mock(),
    )

    scheduler._record_playback_deadline_output("req", {"model_outputs": torch.zeros(24000)})

    scheduler.chunk_transfer_adapter.record_audio_output.assert_not_called()
