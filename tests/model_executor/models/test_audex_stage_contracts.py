# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Audex generation-stage contracts that do not require checkpoints."""

import pytest

from vllm_omni.model_executor.models.audex.audex_code2wav import AudexCode2Wav
from vllm_omni.model_executor.models.audex.audex_xcodec import AudexXCodec1

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def test_audex_generation_stages_publish_nonzero_logits_vocab() -> None:
    assert AudexCode2Wav.logits_vocab_size == 1
    assert AudexXCodec1.logits_vocab_size == 1
